"""Trainer for Ito map models.

Adds to the BaseTrainer the Ito-map-specific batch preparation (two-time
sampling, Brownian path simulation) and the two-term loss

    L = L_SI + lambda * L_consistency,

where L_SI regresses the diagonal drift G_hat_{t,t} onto its closed-form
target (or a frozen teacher's converted drift), and L_consistency enforces
two-time consistency either via the Lagrangian time-derivative match (LSD,
using torch.func.jvp with a finite-difference fallback) or via progressive
semigroup composition (PSD).
"""

import logging
from typing import Any, Optional

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from scisi.models.interpolations import _clamp_time, _expand_t
from scisi.models.ito_maps.brownian import BrownianSample
from scisi.training.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

CONSISTENCY_MODES = ("lsd", "psd")
OFF_DIAGONAL_DISTRIBUTIONS = ("uniform", "logit_normal")
DERIVATIVE_MODES = ("jvp", "finite_difference")

LOGIT_NORMAL_SHIFT = 0.6


class ItoMapTrainer(BaseTrainer):
    """Trainer for Ito map models (from scratch or teacher-distilled).

    Args:
        consistency_mode: ``"lsd"`` (Lagrangian self-distillation via the
            time derivative) or ``"psd"`` (progressive semigroup composition).
        consistency_weight: Weight lambda of the consistency term. 0 trains
            the diagonal loss only.
        off_diagonal_distribution: Distribution of the off-diagonal pair
            (s, t): ``"uniform"`` (two U(0,1) draws, reordered) or
            ``"logit_normal"`` (the paper's t = sigmoid(0.6 + Z_t),
            s = t * sigmoid(Z_s)).
        derivative_mode: ``"jvp"`` computes the LSD time derivative with
            torch.func.jvp; ``"finite_difference"`` is a fallback for wrapped
            architectures forward-mode AD cannot trace.
        finite_difference_eps: Step size of the finite-difference fallback.
        couple_noise_to_path: If True (default), the interpolant noise is
            taken from the simulated Brownian path (z_t = W_t / sqrt(t)), so
            the drift net sees mutually consistent (state, path-feature)
            inputs. If False, or when sigma = 0, the noise is independent.

    Teacher mode is controlled by the model: when ``model.teacher`` is set
    (via the ItoMapModel distillation constructors), diagonal and consistency
    targets use the converted teacher drift; otherwise the closed-form
    interpolant target is used. The teacher is duck-typed - any object with
    ``drift(x, t, field_history, field_cond, pars_cond)``.

    When the base trainer's weight EMA is enabled (``ema_decay``), the EMA
    model provides the stop-gradient targets of the consistency term.
    """

    def __init__(
        self,
        *args: Any,
        consistency_mode: str = "lsd",
        consistency_weight: float = 1.0,
        off_diagonal_distribution: str = "logit_normal",
        derivative_mode: str = "jvp",
        finite_difference_eps: float = 1e-3,
        couple_noise_to_path: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the trainer."""
        super(ItoMapTrainer, self).__init__(*args, **kwargs)

        if consistency_mode not in CONSISTENCY_MODES:
            raise ValueError(f"Unknown consistency mode: {consistency_mode}")
        if off_diagonal_distribution not in OFF_DIAGONAL_DISTRIBUTIONS:
            raise ValueError(
                f"Unknown off-diagonal distribution: {off_diagonal_distribution}"
            )
        if derivative_mode not in DERIVATIVE_MODES:
            raise ValueError(f"Unknown derivative mode: {derivative_mode}")

        self.consistency_mode = consistency_mode
        self.consistency_weight = consistency_weight
        self.off_diagonal_distribution = off_diagonal_distribution
        self.derivative_mode = derivative_mode
        self.finite_difference_eps = finite_difference_eps
        self.couple_noise_to_path = couple_noise_to_path

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def _sample_off_diagonal_times(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the off-diagonal time pair (s, t) with s <= t."""
        if self.off_diagonal_distribution == "uniform":
            times = torch.rand(batch_size, 2, device=self.device)
            s, _ = times.min(dim=1, keepdim=True)
            t, _ = times.max(dim=1, keepdim=True)
            return s, t

        z_t = torch.randn(batch_size, 1, device=self.device)
        z_s = torch.randn(batch_size, 1, device=self.device)
        t = torch.sigmoid(LOGIT_NORMAL_SHIFT + z_t)
        s = t * torch.sigmoid(z_s)
        return s, t

    def _prepare_batch(self, batch: dict) -> dict:
        """Device transfer plus sampled times and the Brownian path."""
        batch = super()._prepare_batch(batch)
        batch_size = batch["base"].shape[0]

        batch["t_diag"] = torch.rand(batch_size, 1, device=self.device)
        batch["s_off"], batch["t_off"] = self._sample_off_diagonal_times(batch_size)

        sampler = self.model.path_sampler
        batch["brownian_sample"] = (
            sampler.sample(batch["base"].shape, self.device)
            if sampler is not None
            else None
        )

        return batch

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _noise_at(
        self,
        brownian_sample: Optional[BrownianSample],
        t: torch.Tensor,
        shape: torch.Size,
    ) -> torch.Tensor:
        """Interpolant noise at time t, coupled to the path when configured."""
        if (brownian_sample is not None) and self.couple_noise_to_path:
            t_clamped = _clamp_time(t)
            w_t = brownian_sample.w_at(t_clamped)
            return w_t / _expand_t(torch.sqrt(t_clamped), w_t)
        return torch.randn(shape, device=self.device)

    @property
    def _target_model(self) -> torch.nn.Module:
        """Model providing stop-gradient consistency targets (EMA if enabled)."""
        return self.ema_model if self.ema_model is not None else self.model

    def _diagonal_target(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        batch: dict,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Regression target for the diagonal drift at (x_t, t)."""
        teacher = self.model.teacher
        if teacher is not None:
            with torch.no_grad():
                return teacher.drift(
                    x_t,
                    t,
                    batch.get("field_history"),
                    batch.get("field_cond"),
                    batch.get("pars_cond"),
                )
        return self.model.G_diag_target(
            base=batch["base"], target=batch["target"], noise=noise, t=t
        )

    def _lsd_loss(
        self,
        x_s: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        brownian_sample: Optional[BrownianSample],
        brownian_features: Optional[torch.Tensor],
        batch: dict,
    ) -> torch.Tensor:
        """Lagrangian self-distillation loss.

        Matches the Lagrangian derivative of the map,
        G_hat_{s,t} + (t - s) * dG_hat_{s,t}/dt, against the (stop-gradient)
        diagonal drift evaluated at the mapped state X_hat_{s,t}.
        """
        field_history = batch.get("field_history")
        field_cond = batch.get("field_cond")
        pars_cond = batch.get("pars_cond")

        def g_of_t(t_in: torch.Tensor) -> torch.Tensor:
            return self.model.G(
                x=x_s,
                s=s,
                t=t_in,
                brownian_features=brownian_features,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
            )

        drift = None
        if self.derivative_mode == "jvp":
            try:
                # Flash/efficient attention kernels have no forward-mode AD;
                # the math backend does, so force it inside the jvp.
                with sdpa_kernel([SDPBackend.MATH]):
                    drift, drift_dt = torch.func.jvp(
                        g_of_t, (t,), (torch.ones_like(t),)
                    )
            except NotImplementedError:
                logger.warning(
                    "torch.func.jvp is not supported by this architecture; "
                    "falling back to finite differences for the LSD time "
                    "derivative."
                )
                self.derivative_mode = "finite_difference"

        if drift is None:
            drift = g_of_t(t)
            drift_dt = (
                g_of_t(t + self.finite_difference_eps) - drift
            ) / self.finite_difference_eps

        dt = _expand_t(t - s, drift)
        pred = drift + dt * drift_dt

        with torch.no_grad():
            x_hat_t = x_s + dt * drift
            if brownian_sample is not None:
                x_hat_t = x_hat_t + brownian_sample.martingale_increment(s, t)

            teacher = self.model.teacher
            if teacher is not None:
                consistency_target = teacher.drift(
                    x_hat_t, t, field_history, field_cond, pars_cond
                )
            else:
                consistency_target = self._target_model.G(
                    x=x_hat_t,
                    s=t,
                    t=t,
                    brownian_features=brownian_features,
                    field_history=field_history,
                    field_cond=field_cond,
                    pars_cond=pars_cond,
                )

        return self.loss_fn(pred, consistency_target)

    def _psd_loss(
        self,
        x_s: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        brownian_sample: Optional[BrownianSample],
        brownian_features: Optional[torch.Tensor],
        batch: dict,
    ) -> torch.Tensor:
        """Progressive self-distillation loss.

        Matches the direct map X_hat_{s,t} against the (stop-gradient)
        semigroup composition X_hat_{u,t} o X_hat_{s,u} for s < u < t. The
        martingale increments cancel between the branches since both use the
        same Brownian path.
        """
        field_history = batch.get("field_history")
        field_cond = batch.get("field_cond")
        pars_cond = batch.get("pars_cond")

        u = s + (t - s) * torch.rand_like(s)

        map_kwargs = {
            "brownian_sample": brownian_sample,
            "brownian_features": brownian_features,
            "field_history": field_history,
            "field_cond": field_cond,
            "pars_cond": pars_cond,
        }

        direct = self.model.map(x=x_s, s=s, t=t, **map_kwargs)

        with torch.no_grad():
            target_model = self._target_model
            x_u = target_model.map(x=x_s, s=s, t=u, **map_kwargs)
            composed = target_model.map(x=x_u, s=u, t=t, **map_kwargs)

        return self.loss_fn(direct, composed)

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        """Diagonal loss plus weighted two-time consistency loss."""
        base, target = batch["base"], batch["target"]
        brownian_sample = batch["brownian_sample"]
        brownian_features = self.model.encode_brownian(brownian_sample)

        # Diagonal loss L_SI at s = t.
        t_diag = batch["t_diag"]
        noise_diag = self._noise_at(brownian_sample, t_diag, base.shape)
        x_t = self.model.interpolant(base=base, target=target, noise=noise_diag, t=t_diag)

        pred_diag = self.model.G(
            x=x_t,
            s=t_diag,
            t=t_diag,
            brownian_features=brownian_features,
            field_history=batch.get("field_history"),
            field_cond=batch.get("field_cond"),
            pars_cond=batch.get("pars_cond"),
        )
        target_diag = self._diagonal_target(x_t, t_diag, batch, noise_diag)
        loss = self.loss_fn(pred_diag, target_diag)

        if self.consistency_weight == 0:
            return loss

        # Consistency loss on the off-diagonal pair (s, t).
        s_off, t_off = batch["s_off"], batch["t_off"]
        noise_off = self._noise_at(brownian_sample, s_off, base.shape)
        x_s = self.model.interpolant(base=base, target=target, noise=noise_off, t=s_off)

        consistency_fn = (
            self._lsd_loss if self.consistency_mode == "lsd" else self._psd_loss
        )
        consistency_loss = consistency_fn(
            x_s=x_s,
            s=s_off,
            t=t_off,
            brownian_sample=brownian_sample,
            brownian_features=brownian_features,
            batch=batch,
        )

        return loss + self.consistency_weight * consistency_loss
