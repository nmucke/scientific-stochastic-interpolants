"""Residual Ito map: a deterministic time stepper fine-tuned into an Ito map.

Method 1 of docs/plans/deterministic_to_ito_map_finetuning.md. An MSE-trained
deterministic model F is the conditional mean of the next state, and the mean
exactly pins down the Brownian-averaged full-span drift of the Ito map (the
anchor identity). F is therefore frozen, and a new Ito map is trained on the
normalized residual process only:

    R~ = (X_1 - F(x_n, h)) / rho,    J_t = beta(t) R~ + (noise part),

with base = 0 (Follmer point mass) or a Gaussian base for the flow-matching
variant. The composite

    X_hat_{s,t}(x, W) = phi(t) + rho * J_hat_{s,t}((x - phi(s)) / rho, W),
    phi(t) = alpha(t) * anchor + beta(t) * F(x_n, h),

is exactly a full-state Ito map (affine time-dependent change of variables)
with drift [phi(t) - phi(s)]/(t - s) + rho * g_hat and diffusion rho * sigma_t.
At (s, t) = (0, 1) it reduces to deterministic prediction + generated residual,

    X_hat_{0,1}(x_n, W) = F(x_n) + rho * J_hat_{0,1}(0, W),

so the Brownian-averaged endpoint is F by construction. Training uses the
unchanged ItoMapTrainer machinery: the trainer moves each batch to residual
coordinates through ``to_residual_batch`` and drives the inner map directly.
"""

from typing import Any, Optional

import torch

from scisi.deterministic_models.deterministic_model import DeterministicModel
from scisi.models.base_model import BaseModel
from scisi.models.interpolations import _expand_t
from scisi.models.ito_maps.brownian import BrownianSample
from scisi.models.ito_maps.calibration import ResidualStats
from scisi.models.ito_maps.ito_map_model import ItoMapModel

DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_STEPS = 1
DEFAULT_NUM_PHYSICAL_STEPS = 10


class ResidualItoMapModel(BaseModel):
    """Mean-anchored residual Ito map (frozen F + residual Ito map).

    Args:
        mean_model: Trained deterministic mean model F. Frozen (its parameters
            never receive gradients and it stays in eval mode), but included
            in the checkpoint so the composite is self-contained.
        ito_map: Ito map trained on the normalized residual process. Its
            interpolation defines the residual path: a stochastic (Follmer)
            interpolation uses base = 0; a deterministic interpolation gives
            the flow-matching variant (Gaussian residual base, sigma = 0).
        residual_stats: Stage-0 calibration of the residual (rho = ``std`` is
            the normalization scale; ``mean`` is kept as a bias diagnostic and
            never subtracted).
        mask_path: Optional inpainting mask path (see ``BaseModel``).
    """

    def __init__(
        self,
        mean_model: DeterministicModel,
        ito_map: ItoMapModel,
        residual_stats: ResidualStats,
        mask_path: Optional[str] = None,
    ) -> None:
        """Initialize the composite model."""
        super(ResidualItoMapModel, self).__init__(mask_path=mask_path)

        self.mean_model = mean_model
        self.ito_map = ito_map
        self._stochastic_path = ito_map._stochastic_path

        for param in self.mean_model.parameters():
            param.requires_grad_(False)
        self.mean_model.eval()

        if torch.any(residual_stats.std <= 0):
            raise ValueError(
                f"Residual std must be positive, got {residual_stats.std.tolist()}."
            )
        self.register_buffer("residual_mean", residual_stats.mean.clone().float())
        self.register_buffer("residual_std", residual_stats.std.clone().float())

    @classmethod
    def from_deterministic(
        cls,
        det_model: DeterministicModel,
        ito_map: ItoMapModel,
        residual_stats: ResidualStats,
        **kwargs: Any,
    ) -> "ResidualItoMapModel":
        """Build the composite from a trained deterministic model (Method 1)."""
        return cls(
            mean_model=det_model,
            ito_map=ito_map,
            residual_stats=residual_stats,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Module plumbing
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "ResidualItoMapModel":
        """Train mode for the residual map only; the frozen mean model stays
        in eval mode (dropout in F would make the residual targets noisy)."""
        super(ResidualItoMapModel, self).train(mode)
        self.mean_model.eval()
        return self

    def to(self, *args: Any, **kwargs: Any) -> "ResidualItoMapModel":
        """Move/cast the composite, including the children's non-parameter
        state (masks, stashed teacher) that plain module recursion misses."""
        super(ResidualItoMapModel, self).to(*args, **kwargs)
        self.ito_map.to(*args, **kwargs)
        self.mean_model.to(*args, **kwargs)
        return self

    # ------------------------------------------------------------------
    # Residual coordinates
    # ------------------------------------------------------------------

    def _rho(self, x: torch.Tensor) -> torch.Tensor:
        """Residual scale broadcast to the state layout [1, C, 1, 1]."""
        return self.residual_std.view(1, -1, *([1] * (x.ndim - 2)))

    def _mean_step(
        self,
        x: torch.Tensor,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Frozen mean prediction F(x_n, h), gradient-free."""
        with torch.no_grad():
            return self.mean_model._step(
                x,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
            )

    def _anchor(self, x_n: torch.Tensor) -> torch.Tensor:
        """Deterministic anchor of the full-state path: x_n for the Follmer
        (point-mass) construction, 0 for the Gaussian-base variant."""
        return x_n if self._stochastic_path else torch.zeros_like(x_n)

    def _phi(
        self, t: torch.Tensor, anchor: torch.Tensor, mean_pred: torch.Tensor
    ) -> torch.Tensor:
        """Anchor path phi(t) = alpha(t) * anchor + beta(t) * F(x_n, h)."""
        t_expanded = _expand_t(t, anchor)
        interpolation = self.ito_map.interpolation
        return (
            interpolation.alpha(t_expanded) * anchor
            + interpolation.beta(t_expanded) * mean_pred
        )

    def _phi_diff(
        self, t: torch.Tensor, anchor: torch.Tensor, mean_pred: torch.Tensor
    ) -> torch.Tensor:
        """Time derivative phi'(t) of the anchor path."""
        t_expanded = _expand_t(t, anchor)
        interpolation = self.ito_map.interpolation
        return (
            interpolation.alpha_diff(t_expanded) * anchor
            + interpolation.beta_diff(t_expanded) * mean_pred
        )

    def to_residual_batch(self, batch: dict) -> dict:
        """Move a full-state training batch to normalized residual coordinates.

        Replaces ``(base, target)`` by ``(0, R~ = (target - F(base, h)) / rho)``;
        the conditioning keys stay untouched (the residual law depends on the
        state through them). ``ItoMapTrainer._prepare_batch`` calls this, after
        which the trainer's loss machinery applies verbatim to the residual
        process through the inner map.
        """
        mean_pred = self._mean_step(
            batch["base"],
            field_history=batch.get("field_history"),
            field_cond=batch.get("field_cond"),
            pars_cond=batch.get("pars_cond"),
        )
        batch["target"] = (batch["target"] - mean_pred) / self._rho(batch["target"])
        batch["base"] = torch.zeros_like(batch["base"])
        return batch

    # ------------------------------------------------------------------
    # Full-state map evaluation
    # ------------------------------------------------------------------

    def map(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        brownian_sample: Optional[BrownianSample] = None,
        brownian_features: Optional[torch.Tensor] = None,
        martingale_increment: Optional[torch.Tensor] = None,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full-state composite map phi(t) + rho * J_hat_{s,t}((x - phi(s))/rho).

        The inner map's martingale increment is in residual coordinates, so
        the composite diffusion is rho * sigma_t, as the change of variables
        requires. ``field_history`` is required: its last slice is x_n, which
        both the anchor path and the mean prediction depend on. An explicit
        ``martingale_increment`` must likewise be given in residual
        coordinates.
        """
        if field_history is None:
            raise ValueError(
                "ResidualItoMapModel.map needs field_history: its last slice "
                "is the current state x_n that anchors the composite."
            )
        x_n = field_history[:, :, :, :, -1]
        mean_pred = self._mean_step(x_n, field_history, field_cond, pars_cond)
        anchor = self._anchor(x_n)
        rho = self._rho(x)

        j = (x - self._phi(s, anchor, mean_pred)) / rho
        j = self.ito_map.map(
            x=j,
            s=s,
            t=t,
            brownian_sample=brownian_sample,
            brownian_features=brownian_features,
            martingale_increment=martingale_increment,
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
        )
        return self._phi(t, anchor, mean_pred) + rho * j

    def drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full-state diagonal drift phi'(t) + rho * g_hat_{t,t}((x - phi(t))/rho)."""
        if field_history is None:
            raise ValueError(
                "ResidualItoMapModel.drift needs field_history: its last "
                "slice is the current state x_n that anchors the composite."
            )
        x_n = field_history[:, :, :, :, -1]
        mean_pred = self._mean_step(x_n, field_history, field_cond, pars_cond)
        anchor = self._anchor(x_n)
        rho = self._rho(x)

        j = (x - self._phi(t, anchor, mean_pred)) / rho
        return self._phi_diff(t, anchor, mean_pred) + rho * self.ito_map.drift(
            j, t, field_history, field_cond, pars_cond
        )

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Not supported: residual Ito maps train with the ItoMapTrainer."""
        raise RuntimeError(
            "ResidualItoMapModel has a multi-evaluation loss (diagonal + "
            "consistency, in residual coordinates) and cannot train through "
            "the generic (pred, target) contract; use "
            "scisi.training.ito_map_trainer.ItoMapTrainer."
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        field_history: torch.Tensor,
        base: Optional[torch.Tensor] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_steps: int = DEFAULT_NUM_STEPS,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        return_field_history: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Sample the endpoint F(x_n) + rho * J_hat_{0,1}: deterministic
        prediction plus generated residual.

        ``base`` is the current state x_n (defaults to the last history
        slice). The residual map starts from its own base - the zero point
        mass, or the Gaussian base for the flow-matching variant. Running the
        whole ``num_steps`` partition in residual coordinates and composing
        once at the end is identical to iterating the composite map (the
        phi shifts telescope). Unused solver kwargs are absorbed for
        ``BaseModel._sample_trajectory`` compatibility.
        """
        if (batch_size > 1) and (field_history.shape[0] == 1):
            base, field_history, field_cond, pars_cond = self._prepare_batch(
                batch_size=batch_size,
                base=base,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
            )

        field_history = field_history.to(self.device)
        field_cond = field_cond.to(self.device) if field_cond is not None else None
        pars_cond = pars_cond.to(self.device) if pars_cond is not None else None

        x_n = base if base is not None else field_history[:, :, :, :, -1]
        x_n = x_n.to(self.device)

        mean_pred = self._mean_step(x_n, field_history, field_cond, pars_cond)

        residual_base = torch.zeros_like(x_n) if self._stochastic_path else None
        j = self.ito_map.sample(
            field_history=field_history,
            base=residual_base,
            batch_size=batch_size,
            num_steps=num_steps,
            field_cond=field_cond,
            pars_cond=pars_cond,
        )

        x = mean_pred.cpu() + self._rho(j).cpu() * j

        if return_field_history:
            field_history = torch.cat(
                [field_history[:, :, :, :, 1:].cpu(), x.unsqueeze(-1)], dim=-1
            )
            return x, field_history

        return x

    def sample_trajectory(
        self,
        field_history: torch.Tensor,
        base: Optional[torch.Tensor] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_steps: int = DEFAULT_NUM_STEPS,
        num_physical_steps: int = DEFAULT_NUM_PHYSICAL_STEPS,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Autoregressive rollout of composite one-step endpoint samples."""
        return self._sample_trajectory(
            field_history=field_history,
            base=base,
            batch_size=batch_size,
            num_steps=num_steps,
            num_physical_steps=num_physical_steps,
            field_cond=field_cond,
            pars_cond=pars_cond,
            gaussian_base=False,
        )
