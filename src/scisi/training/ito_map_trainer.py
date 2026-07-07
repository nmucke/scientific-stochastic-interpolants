"""Trainer for Ito map models.

Adds to the BaseTrainer the Ito-map-specific batch preparation (two-time
sampling, Brownian path simulation) and the two-term loss

    L = L_SI + lambda * L_consistency,

where L_SI regresses the diagonal drift G_hat_{t,t} onto its closed-form
target (or a frozen teacher's converted drift), and L_consistency enforces
two-time consistency either via the Lagrangian time-derivative match (LSD,
using torch.func.jvp with a finite-difference fallback) or via progressive
semigroup composition (PSD).

Also trains residual Ito maps (deterministic-to-Ito-map fine-tuning, Method 1
of docs/plans/deterministic_to_ito_map_finetuning.md): a model exposing
``to_residual_batch`` has each batch moved to normalized residual coordinates
during batch preparation, and the losses drive its inner map (``ito_map``)
directly - the loss machinery itself is unchanged.
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
        freeze: Optional list of dotted submodule names (resolved on the
            model, e.g. ``["drift_model.net"]``) whose parameters are frozen
            at construction. Used for staged fine-tuning (Method 2 of the
            deterministic fine-tuning plan).
        unfreeze_at_epoch: Epoch at which the frozen modules are unfrozen
            again (``None`` keeps them frozen throughout). Requires
            ``freeze``.
        teacher_warmup_epochs: If > 0 and the model has a teacher attached,
            the teacher is detached at the start of this epoch, switching the
            diagonal/consistency targets from the teacher to the exact data
            targets (e.g. a GaussianShellTeacher warm-up, Method 3). 0 keeps
            an attached teacher for the whole run.

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
        freeze: Optional[list[str]] = None,
        unfreeze_at_epoch: Optional[int] = None,
        teacher_warmup_epochs: int = 0,
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
        if (unfreeze_at_epoch is not None) and not freeze:
            raise ValueError("unfreeze_at_epoch requires a non-empty freeze list.")

        self.consistency_mode = consistency_mode
        self.consistency_weight = consistency_weight
        self.off_diagonal_distribution = off_diagonal_distribution
        self.derivative_mode = derivative_mode
        self.finite_difference_eps = finite_difference_eps
        self.couple_noise_to_path = couple_noise_to_path

        self.freeze = list(freeze) if freeze else []
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.teacher_warmup_epochs = teacher_warmup_epochs

        self._frozen_modules = [
            self.model.get_submodule(name) for name in self.freeze
        ]
        for module in self._frozen_modules:
            module.requires_grad_(False)
        if self.freeze:
            logger.info(f"Froze modules: {self.freeze}")

    # ------------------------------------------------------------------
    # Model access
    # ------------------------------------------------------------------

    @property
    def _map_model(self) -> torch.nn.Module:
        """The Ito map the losses drive.

        A residual wrapper (ResidualItoMapModel) exposes its inner map as
        ``ito_map``; the trainer drives that inner map directly, in the
        residual coordinates ``to_residual_batch`` moved the batch to. For a
        plain ItoMapModel this is the model itself.
        """
        return getattr(self.model, "ito_map", self.model)

    # ------------------------------------------------------------------
    # Epoch schedule (staged unfreezing, teacher warm-up)
    # ------------------------------------------------------------------

    def _on_epoch_start(self, epoch: int) -> None:
        """Staged unfreezing and the teacher warm-up switchover."""
        if (self.unfreeze_at_epoch is not None) and (epoch == self.unfreeze_at_epoch):
            for module in self._frozen_modules:
                module.requires_grad_(True)
            logger.info(f"Epoch {epoch}: unfroze modules {self.freeze}")

        if (
            (self.teacher_warmup_epochs > 0)
            and (epoch == self.teacher_warmup_epochs)
            and (self._map_model.teacher is not None)
        ):
            # The teacher is stashed outside the module registry
            # (ItoMapModel.__init__), hence the object.__setattr__.
            object.__setattr__(self._map_model, "_teacher", None)
            logger.info(
                f"Epoch {epoch}: teacher warm-up finished, switching to the "
                "exact data targets."
            )

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
        """Device transfer plus sampled times and the Brownian path.

        Residual wrappers additionally move (base, target) to normalized
        residual coordinates here, so all downstream losses see the residual
        process.
        """
        batch = super()._prepare_batch(batch)
        if hasattr(self.model, "to_residual_batch"):
            batch = self.model.to_residual_batch(batch)
        batch_size = batch["base"].shape[0]

        batch["t_diag"] = torch.rand(batch_size, 1, device=self.device)
        batch["s_off"], batch["t_off"] = self._sample_off_diagonal_times(batch_size)

        sampler = self._map_model.path_sampler
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
        """Interpolant noise at time t, coupled to the path when configured.

        ``standard_normal_at`` compensates the path representation's variance
        deficit at small t (KL truncation / grid interpolation), so the noise
        marginal is exactly N(0, I) at every t.
        """
        if (brownian_sample is not None) and self.couple_noise_to_path:
            return brownian_sample.standard_normal_at(_clamp_time(t))
        return torch.randn(shape, device=self.device)

    @property
    def _target_model(self) -> torch.nn.Module:
        """Map providing stop-gradient consistency targets (EMA if enabled).

        Like ``_map_model``, residual wrappers are unwrapped to their inner
        map (the EMA model is a copy of the wrapper).
        """
        model = self.ema_model if self.ema_model is not None else self.model
        return getattr(model, "ito_map", model)

    def _diagonal_target(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        batch: dict,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Regression target for the diagonal drift at (x_t, t)."""
        teacher = self._map_model.teacher
        if teacher is not None:
            with torch.no_grad():
                return teacher.drift(
                    x_t,
                    t,
                    batch.get("field_history"),
                    batch.get("field_cond"),
                    batch.get("pars_cond"),
                )
        return self._map_model.G_diag_target(
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
            return self._map_model.G(
                x=x_s,
                s=s,
                t=t_in,
                brownian_features=brownian_features,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
            )

        # The time-derivative branch always runs in full precision:
        # torch.func.jvp does not compose with autocast (dual tensors bypass
        # the cast layer, mixing bf16 and fp32 mid-net), and finite
        # differences under bf16 are garbage (eps is below bf16 resolution
        # inside the Fourier time embedding). The stop-gradient target below
        # still runs under autocast.
        drift = None
        with torch.autocast(device_type=t.device.type, enabled=False):
            if self.derivative_mode == "jvp":
                try:
                    # Flash/efficient attention kernels have no forward-mode
                    # AD; the math backend does, so force it inside the jvp.
                    with sdpa_kernel([SDPBackend.MATH]):
                        drift, drift_dt = torch.func.jvp(
                            g_of_t, (t,), (torch.ones_like(t),)
                        )
                except (NotImplementedError, RuntimeError) as err:
                    # Backends signal missing forward-mode support
                    # inconsistently (NotImplementedError or RuntimeError);
                    # anything else is a real bug and must propagate.
                    message = str(err).lower()
                    if isinstance(err, RuntimeError) and not any(
                        marker in message
                        for marker in ("forward ad", "forward-mode", "jvp")
                    ):
                        raise
                    logger.warning(
                        "torch.func.jvp is not supported by this architecture "
                        f"({err}); falling back to finite differences for the "
                        "LSD time derivative."
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

            teacher = self._map_model.teacher
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

        direct = self._map_model.map(x=x_s, s=s, t=t, **map_kwargs)

        with torch.no_grad():
            target_model = self._target_model
            x_u = target_model.map(x=x_s, s=s, t=u, **map_kwargs)
            composed = target_model.map(x=x_u, s=u, t=t, **map_kwargs)

        return self.loss_fn(direct, composed)

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        """Diagonal loss plus weighted two-time consistency loss."""
        base, target = batch["base"], batch["target"]
        brownian_sample = batch["brownian_sample"]
        brownian_features = self._map_model.encode_brownian(brownian_sample)

        # Diagonal loss L_SI at s = t.
        t_diag = batch["t_diag"]
        noise_diag = self._noise_at(brownian_sample, t_diag, base.shape)
        x_t = self._map_model.interpolant(
            base=base, target=target, noise=noise_diag, t=t_diag
        )

        pred_diag = self._map_model.G(
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
        x_s = self._map_model.interpolant(
            base=base, target=target, noise=noise_off, t=s_off
        )

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
