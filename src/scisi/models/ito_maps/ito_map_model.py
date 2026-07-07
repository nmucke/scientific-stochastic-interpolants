"""Ito map model.

Implements the two-time stochastic flow map of arXiv:2606.11156:

    X_hat_{s,t}(x, W) = x + (t - s) * G_hat_{s,t}(x, W) + (M_t - M_s),

where ``G_hat`` is a learned average drift conditioned on a compressed
encoding of the Brownian path W, and ``M_t = int_0^t sigma_u dW_u`` is the
martingale part (computed in closed form from the path, not learned). Once
trained, the map jumps from any time s to any time t in one network
evaluation while still sampling from the correct endpoint law. With
``sigma = 0`` it degenerates to a deterministic flow map (one-step flow
matching).
"""

import logging
from copy import deepcopy
from typing import Any, Optional

import torch
import torch.nn as nn

from scisi.architectures.embeddings import TwoTimeCondEncoder
from scisi.deterministic_models.deterministic_model import DeterministicModel
from scisi.models.base_model import BaseModel
from scisi.models.flow_matching_model import FlowMatchingModel
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
from scisi.models.interpolations import MIN_TIME, _clamp_time, _expand_t
from scisi.models.ito_maps.brownian import (
    DEFAULT_DENSE_GRID_SIZE,
    DEFAULT_NUM_GRID_POINTS,
    DEFAULT_NUM_KL_TERMS,
    BrownianEncoder,
    BrownianPathSampler,
    BrownianSample,
    DyadicEncoder,
    GammaMatchedSigmaSchedule,
    PaperSigmaSchedule,
    SigmaSchedule,
    ZeroSigmaSchedule,
)

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_STEPS = 1
DEFAULT_NUM_PHYSICAL_STEPS = 10


class FlowMatchingTeacher(nn.Module):
    """Frozen flow-matching teacher for Ito-map distillation.

    Converts the teacher's velocity into the diagonal SDE drift for the chosen
    sigma schedule via the velocity->score identity (anchor a0 = 0):

        G_{t,t}(x) = v(x, t) + (sigma_t^2 / 2) * score_from_velocity(x, v, t).

    In the paper's canonical setting (linear path, sigma = sqrt(2(1-t))) this
    reduces to G = (1 + t) v - x; the general mixin-based conversion is used
    so non-canonical schedules also work.
    """

    def __init__(self, model: FlowMatchingModel, sigma_schedule: SigmaSchedule) -> None:
        """Initialize and freeze the teacher."""
        super(FlowMatchingTeacher, self).__init__()
        self.model = model
        self.sigma_schedule = sigma_schedule
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

    def drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Diagonal drift target G_{t,t}(x) from the frozen teacher."""
        velocity = self.model.drift_model(x, t, field_history, field_cond, pars_cond)
        t_expanded = _expand_t(_clamp_time(t), x)
        score = self.model.interpolation.score_from_velocity(
            x=x,
            v=velocity,
            t=t_expanded,
            a0=torch.zeros_like(x),
        )
        return velocity + 0.5 * self.sigma_schedule(t_expanded) ** 2 * score


class FollmerTeacher(nn.Module):
    """Frozen Follmer stochastic-interpolant teacher for Ito-map distillation.

    Converts the teacher's Follmer drift ``b_theta`` (trained with diffusion
    gamma_t) into the diagonal drift for the chosen sigma schedule via the
    prior-score correction:

        G_{t,t}(x) = b + ((sigma_t^2 - gamma_t^2) / 2) * prior_score(x, t).

    Requires ``field_history`` (the previous state is the point-mass base of
    the Follmer construction).
    """

    def __init__(
        self, model: FollmerStochasticInterpolant, sigma_schedule: SigmaSchedule
    ) -> None:
        """Initialize and freeze the teacher."""
        super(FollmerTeacher, self).__init__()
        self.model = model
        self.sigma_schedule = sigma_schedule
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

    def drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Diagonal drift target G_{t,t}(x) from the frozen teacher."""
        if field_history is None:
            raise ValueError(
                "FollmerTeacher needs field_history: the previous state is "
                "the point-mass base of the Follmer prior score."
            )
        drift = self.model.drift_model(x, t, field_history, field_cond, pars_cond)

        t_expanded = _expand_t(_clamp_time(t), x)
        sigma = self.sigma_schedule(t_expanded)
        gamma = self.model.interpolation.gamma(t_expanded)

        base = field_history[:, :, :, :, -1]
        prior_score = self.model._prior_score(x, base, drift, t_expanded)

        # Small-t guard (batch-safe analogue of _drift_with_prior_score's
        # `if t < MIN_TIME: return drift`): the prior score diverges like
        # 1/t^2 near 0, so the correction is dropped there per-sample.
        correction = 0.5 * (sigma**2 - gamma**2) * prior_score
        correction = torch.where(
            _expand_t(t, x) < MIN_TIME, torch.zeros_like(correction), correction
        )

        return drift + correction


def warm_start_from_teacher(student: nn.Module, teacher: nn.Module) -> dict:
    """Warm-start a student drift net from a teacher drift net.

    Copies every parameter/buffer whose name and shape match. Parameters that
    only differ in input channels (dim 1, e.g. the skip-connection conv of a
    first block that gained Brownian-feature channels) get the teacher
    weights in the leading slice and zeros in the new channels, so the
    student initially ignores the new inputs there. Teacher keys under
    ``cond_encoder.`` are mapped into the student's
    ``cond_encoder.t_encoder.`` branch (two-time embedding), and the final
    linear layer of the ``s_encoder`` branch is zeroed, so the two-time
    embedding exactly reproduces the teacher's single-time embedding at init.

    Layers whose internal width is tied to the input channel count (the
    ConvNext blocks' depthwise conv, GroupNorm and FiLM projection) cannot be
    mapped when the channel count changes; they are left at their fresh init
    and reported as skipped.

    Returns:
        dict: Report with ``copied``, ``expanded`` and ``skipped`` key lists.
    """
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()

    copied, expanded, skipped = [], [], []
    two_time_prefix = "cond_encoder.t_encoder."

    for name, student_param in student_state.items():
        teacher_name = name
        if teacher_name not in teacher_state and name.startswith(two_time_prefix):
            teacher_name = "cond_encoder." + name[len(two_time_prefix) :]

        if teacher_name not in teacher_state:
            skipped.append(name)
            continue

        teacher_param = teacher_state[teacher_name]
        if teacher_param.shape == student_param.shape:
            student_state[name] = teacher_param.clone()
            copied.append(name)
        elif (
            teacher_param.ndim == student_param.ndim
            and teacher_param.ndim >= 2
            and teacher_param.shape[0] == student_param.shape[0]
            and teacher_param.shape[2:] == student_param.shape[2:]
            and teacher_param.shape[1] < student_param.shape[1]
        ):
            new_param = torch.zeros_like(student_param)
            new_param[:, : teacher_param.shape[1]] = teacher_param
            student_state[name] = new_param
            expanded.append(name)
        else:
            skipped.append(name)

    student.load_state_dict(student_state)

    # Zero the final linear layer of the s-branch so the student initially
    # ignores the second time input.
    cond_encoder = getattr(student, "cond_encoder", None)
    if isinstance(cond_encoder, TwoTimeCondEncoder):
        for module in reversed(list(cond_encoder.s_encoder)):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
                break

    logger.info(
        f"Warm start from teacher: {len(copied)} copied, "
        f"{len(expanded)} zero-expanded, {len(skipped)} left at init."
    )
    if skipped:
        logger.info(f"Warm start skipped keys: {skipped}")

    return {"copied": copied, "expanded": expanded, "skipped": skipped}


class NextStepDriftAdapter(nn.Module):
    """Adapts a next-state network to the drift-net contract: G = net(...) - x.

    Used by the deterministic warm start (Method 2 of the fine-tuning plan,
    docs/plans/deterministic_to_ito_map_finetuning.md) when the teacher's
    network predicts the next state itself (``residual=False``); a residual
    teacher already outputs mu(x) - x and needs no adapter. The fixed skip
    makes the warm-started map satisfy X_hat_{0,1}(x) = x + G(x) = F(x) at
    initialization (the anchor identity).
    """

    def __init__(self, net: nn.Module) -> None:
        """Wrap the next-state network."""
        super(NextStepDriftAdapter, self).__init__()
        self.net = net

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Drift from the next-state prediction: net(x, ...) - x."""
        out = self.net(
            x=x,
            cond=cond,
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
            **kwargs,
        )
        return out - x


def _pin_scalar_encoder_at_zero(encoder: nn.Sequential) -> None:
    """Make a scalar time encoder output its embedding of 0 for ALL inputs.

    Zeroes the first linear layer's weight and folds that layer's output at
    input 0 into its bias, so every downstream (copied) layer processes the
    embedding-of-zero regardless of the actual time input. Gradients still
    flow into the zeroed weight, so time dependence can regrow during
    fine-tuning.

    A deterministic teacher only ever saw the constant placeholder cond = 0
    (PR 2's ``_step``), so exact functional equivalence at init requires the
    student's time pathway to be pinned to the teacher's embedding of 0 -
    merely copying the embedding (the flow-matching warm start) would make
    the init output vary with (s, t).
    """
    device = next(encoder.parameters()).device
    features = torch.zeros(1, 1, device=device)
    for module in encoder:
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                bias_at_zero = module(features)[0]
                module.weight.zero_()
                module.bias.copy_(bias_at_zero)
            return
        features = module(features)
    raise ValueError(
        "Expected a linear layer in the scalar time encoder to pin; none found."
    )


class ItoMapModel(BaseModel):
    """Ito map: a learned two-time stochastic flow map.

    Args:
        interpolation: Interpolation defining the marginal path. Stochastic
            interpolations (exposing ``gamma``) use the repo's Follmer
            convention (point-mass base = previous state); deterministic ones
            use the flow-matching convention (Gaussian base = the noise).
        drift_model: Network parametrizing G_hat_{s,t}(x, W-features). Must
            accept a two-time cond [B, 2] (e.g. UNet with
            ``two_time_cond: true``); Brownian features enter through the
            ``field_cond`` channel-concat pathway.
        sigma_schedule: Diffusion schedule of the SDE, an instance or one of
            the strings ``"paper"``, ``"zero"``, ``"gamma_matched"``.
        brownian_encoder: Encoder compressing the Brownian path into
            conditioning channels. ``None`` leaves G unconditioned on W
            (required for sigma = 0, where W never enters).
        num_grid_points: Grid resolution of the Brownian sampler (path mode).
        brownian_mode: ``"kl"`` (closed-form, memory-light) or ``"path"``.
        num_kl_terms: Number of KL terms held by the sampler in kl mode.
        dense_grid_size: Dense grid for the kl-mode cumulative integrals.
        mask_path: Optional inpainting mask path.
    """

    def __init__(
        self,
        interpolation: nn.Module,
        drift_model: nn.Module,
        sigma_schedule: SigmaSchedule | str = "paper",
        brownian_encoder: Optional[BrownianEncoder] = None,
        num_grid_points: int = DEFAULT_NUM_GRID_POINTS,
        brownian_mode: str = "kl",
        num_kl_terms: int = DEFAULT_NUM_KL_TERMS,
        dense_grid_size: int = DEFAULT_DENSE_GRID_SIZE,
        mask_path: Optional[str] = None,
    ) -> None:
        """Initialize the Ito map model."""
        super(ItoMapModel, self).__init__(mask_path=mask_path)

        self.interpolation = interpolation
        self.drift_model = drift_model
        self._stochastic_path = hasattr(interpolation, "gamma")

        if isinstance(sigma_schedule, str):
            sigma_schedule = self._resolve_sigma_schedule(sigma_schedule)
        self.sigma_schedule = sigma_schedule

        self.brownian_encoder = brownian_encoder
        if self.sigma_schedule.is_zero:
            if brownian_encoder is not None:
                raise ValueError(
                    "sigma = 0 has no Brownian path; set brownian_encoder to "
                    "None for the deterministic flow-map case."
                )
            self.path_sampler = None
        else:
            if (
                isinstance(brownian_encoder, DyadicEncoder)
                and brownian_mode == "kl"
                and num_kl_terms < 4 * brownian_encoder.num_features_per_channel
            ):
                raise ValueError(
                    f"DyadicEncoder(depth={brownian_encoder.depth}) needs "
                    f"dyadic detail at scale 1/{brownian_encoder.num_features_per_channel}, "
                    f"which {num_kl_terms} KL terms cannot represent - the "
                    "fine-level Haar coefficients would be near-degenerate. "
                    "Use brownian_mode='path' or set num_kl_terms >= "
                    f"{4 * brownian_encoder.num_features_per_channel}."
                )
            self.path_sampler = BrownianPathSampler(
                sigma_schedule=self.sigma_schedule,
                num_grid_points=num_grid_points,
                mode=brownian_mode,
                num_kl_terms=num_kl_terms,
                dense_grid_size=dense_grid_size,
            )
            if brownian_encoder is None:
                logger.warning(
                    "ItoMapModel with sigma != 0 but no Brownian encoder: "
                    "G is not conditioned on the path, which limits the map "
                    "beyond-Gaussian accuracy."
                )

        # The frozen teacher is stashed outside the module registry so its
        # parameters are excluded from state_dict() and the optimizer.
        object.__setattr__(self, "_teacher", None)

    def _resolve_sigma_schedule(self, name: str) -> SigmaSchedule:
        """Resolve a sigma schedule from its config string."""
        if name == "paper":
            return PaperSigmaSchedule()
        if name == "zero":
            return ZeroSigmaSchedule()
        if name == "gamma_matched":
            if not self._stochastic_path:
                raise ValueError(
                    "gamma_matched sigma schedule needs a stochastic "
                    "interpolation exposing gamma."
                )
            return GammaMatchedSigmaSchedule(self.interpolation)
        raise ValueError(f"Unknown sigma schedule: {name}")

    @property
    def model(self) -> nn.Module:
        """
        Get the drift model.

        This is to ensure compatibility with the rest of the code base.
        """
        return self.drift_model

    @property
    def teacher(self) -> Optional[nn.Module]:
        """The frozen distillation teacher, or None for from-scratch training."""
        return self._teacher

    def to(self, *args: Any, **kwargs: Any) -> "ItoMapModel":
        """Move/cast the model (and the stashed teacher, if any)."""
        super(ItoMapModel, self).to(*args, **kwargs)
        if (self._teacher is not None) and hasattr(self._teacher, "to"):
            self._teacher.to(*args, **kwargs)
        return self

    def __deepcopy__(self, memo: dict) -> "ItoMapModel":
        """Deepcopy that shares (not duplicates) the stashed frozen teacher.

        The trainer's weight EMA deep-copies the model; without this the EMA
        copy would silently carry a second, never-used copy of the teacher in
        memory. The teacher is frozen, so sharing the reference is safe.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if key == "_teacher":
                object.__setattr__(result, key, value)
            else:
                object.__setattr__(result, key, deepcopy(value, memo))
        return result

    # ------------------------------------------------------------------
    # Distillation constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_flow_matching(
        cls,
        flow_matching_model: FlowMatchingModel,
        drift_model: nn.Module,
        sigma_schedule: SigmaSchedule | str = "paper",
        **kwargs: Any,
    ) -> "ItoMapModel":
        """Build an Ito map that distills a trained flow-matching model.

        Reuses the teacher's interpolation, warm-starts the student net from
        the teacher net, and stashes the frozen teacher for the trainer's
        distillation targets.
        """
        model = cls(
            interpolation=flow_matching_model.interpolation,
            drift_model=drift_model,
            sigma_schedule=sigma_schedule,
            **kwargs,
        )
        model.distill_from(flow_matching_model)
        return model

    @classmethod
    def from_stochastic_interpolant(
        cls,
        si_model: FollmerStochasticInterpolant,
        drift_model: nn.Module,
        sigma_schedule: SigmaSchedule | str = "gamma_matched",
        **kwargs: Any,
    ) -> "ItoMapModel":
        """Build an Ito map that distills a trained Follmer interpolant."""
        model = cls(
            interpolation=si_model.interpolation,
            drift_model=drift_model,
            sigma_schedule=sigma_schedule,
            **kwargs,
        )
        model.distill_from(si_model)
        return model

    @classmethod
    def from_deterministic(
        cls,
        det_model: DeterministicModel,
        drift_model: nn.Module,
        interpolation: nn.Module,
        sigma_schedule: SigmaSchedule | str = "gamma_matched",
        **kwargs: Any,
    ) -> "ItoMapModel":
        """Build an Ito map warm-started from a deterministic next-step model.

        Method 2 (weight-surgery warm start) of the fine-tuning plan
        (docs/plans/deterministic_to_ito_map_finetuning.md). Unlike the
        distillation constructors, the deterministic model is NOT attached as
        a teacher - it has no drift; training uses the from-scratch objective
        (optionally with a GaussianShellTeacher warm-up). The deterministic
        model also defines no interpolation, so one must be supplied.
        """
        model = cls(
            interpolation=interpolation,
            drift_model=drift_model,
            sigma_schedule=sigma_schedule,
            **kwargs,
        )
        model.warm_start_from_deterministic(det_model)
        return model

    def warm_start_from_deterministic(self, det_model: DeterministicModel) -> dict:
        """Warm-start the drift net from a deterministic model (Method 2).

        Three surgery steps, together making the map compute exactly the
        deterministic model at initialization:

        1. Copy the teacher network into the drift net
           (``warm_start_from_teacher``: backbone copied, Brownian input
           pathway left at its zero init, s-branch of the two-time embedding
           zeroed).
        2. Pin the t-branch of the two-time embedding to the teacher's
           embedding of the constant placeholder cond = 0 it was trained
           with, so the net is blind to (s, t) at init.
        3. Fix the output head to the drift contract: a ``residual=False``
           teacher predicts the next state, so G := net(x, ...) - x
           (``NextStepDriftAdapter``); a ``residual=True`` teacher already
           predicts the increment mu(x) - x and needs no adapter.

        The initialized map is X_hat_{s,t}(x) = x + (t - s)(F(x) - x) plus
        the martingale part - linear transport toward the deterministic
        endpoint, whose Brownian-averaged endpoint at (0, 1) is exactly
        F(x_n) (the anchor identity holds at init).

        Returns:
            dict: Weight-surgery report (``copied``/``expanded``/``skipped``).
        """
        report = warm_start_from_teacher(self.drift_model, det_model.network)

        cond_encoder = getattr(self.drift_model, "cond_encoder", None)
        if isinstance(cond_encoder, TwoTimeCondEncoder):
            _pin_scalar_encoder_at_zero(cond_encoder.t_encoder)

        if not det_model.residual:
            self.drift_model = NextStepDriftAdapter(self.drift_model)

        return report

    def distill_from(self, teacher_model: nn.Module) -> dict:
        """Attach a frozen teacher and warm-start the student net from it.

        The teacher is duck-typed: any object exposing
        ``drift(x, t, field_history, field_cond, pars_cond)`` works (e.g. a
        non-neural analytic teacher). Known model classes are wrapped so their
        native parametrization is converted to the diagonal drift for this
        model's sigma schedule.

        Returns:
            dict: Weight-surgery report (empty if the teacher has no net).
        """
        if isinstance(teacher_model, DeterministicModel):
            # Has a (raising) drift method, so it would slip through the
            # duck-typed branch and only fail at training time.
            raise TypeError(
                "A DeterministicModel has no drift to distill from; use "
                "warm_start_from_deterministic / from_deterministic (Method "
                "2) or ResidualItoMapModel.from_deterministic (Method 1) "
                "instead."
            )
        if isinstance(teacher_model, FlowMatchingModel):
            wrapper: nn.Module = FlowMatchingTeacher(teacher_model, self.sigma_schedule)
        elif isinstance(teacher_model, FollmerStochasticInterpolant):
            wrapper = FollmerTeacher(teacher_model, self.sigma_schedule)
        elif hasattr(teacher_model, "drift"):
            wrapper = teacher_model
            for param in getattr(wrapper, "parameters", lambda: [])():
                param.requires_grad_(False)
        else:
            raise TypeError(
                f"Cannot distill from {type(teacher_model).__name__}: expected "
                "a FlowMatchingModel, a FollmerStochasticInterpolant, or any "
                "object exposing drift(x, t, field_history, field_cond, "
                "pars_cond)."
            )

        report: dict = {}
        if hasattr(teacher_model, "drift_model"):
            report = warm_start_from_teacher(self.drift_model, teacher_model.drift_model)

        object.__setattr__(self, "_teacher", wrapper)
        if hasattr(wrapper, "to") and (next(self.parameters(), None) is not None):
            wrapper.to(self.device)
        return report

    # ------------------------------------------------------------------
    # Core map evaluation
    # ------------------------------------------------------------------

    def encode_brownian(
        self, brownian_sample: Optional[BrownianSample]
    ) -> Optional[torch.Tensor]:
        """Encode a Brownian sample into conditioning channels (or None)."""
        if (brownian_sample is None) or (self.brownian_encoder is None):
            return None
        return self.brownian_encoder(brownian_sample)

    def G(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        brownian_features: Optional[torch.Tensor] = None,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate the learned average drift G_hat_{s,t}.

        Args:
            x (torch.Tensor): State at time s [B, C, H, W].
            s (torch.Tensor): Start time [B, 1].
            t (torch.Tensor): End time [B, 1].
            brownian_features (torch.Tensor): Encoded Brownian path
                [B, K*C, H, W]. Can be None. Requires a drift net exposing a
                ``brownian_features`` kwarg (e.g. UNet with
                ``brownian_feature_channels``).
            field_history (torch.Tensor): Field history tensor [B, C, H, W, L]. Can be None.
            field_cond (torch.Tensor): Field conditional tensor [B, C_field_cond, H, W]. Can be None.
            pars_cond (torch.Tensor): pars conditional tensor [B, D_pars_cond]. Can be None.
        """
        cond = torch.cat([s, t], dim=1)
        # Brownian features enter through the net's dedicated zero-initialized
        # projection (UNet ``brownian_feature_channels``), NOT the field_cond
        # concat: this keeps the init conv structurally identical to a
        # teacher's, so weight surgery transfers the whole input stage.
        extra_kwargs = {}
        if brownian_features is not None:
            extra_kwargs["brownian_features"] = brownian_features
        return self.drift_model(
            x=x,
            cond=cond,
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
            **extra_kwargs,
        )

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
        """Apply the Ito map: x + (t - s) * G_hat + (M_t - M_s).

        ``brownian_features`` / ``martingale_increment`` are computed from
        ``brownian_sample`` when not given explicitly.
        """
        if brownian_features is None:
            brownian_features = self.encode_brownian(brownian_sample)

        drift = self.G(
            x=x,
            s=s,
            t=t,
            brownian_features=brownian_features,
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
        )

        out = x + _expand_t(t - s, x) * drift

        if (martingale_increment is None) and (brownian_sample is not None):
            martingale_increment = brownian_sample.martingale_increment(s, t)
        if martingale_increment is not None:
            out = out + martingale_increment

        return out

    # ------------------------------------------------------------------
    # Training targets
    # ------------------------------------------------------------------

    def interpolant(
        self,
        base: torch.Tensor,
        target: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Marginal-path state I_t.

        Stochastic interpolations use the Follmer convention (base = previous
        state, explicit noise); deterministic ones use the flow-matching
        convention (the noise is the Gaussian base of the path).
        """
        if self._stochastic_path:
            return self.interpolation.forward(base=base, target=target, t=t, noise=noise)
        return self.interpolation.forward(base=noise, target=target, t=t)

    def G_diag_target(
        self,
        base: torch.Tensor,
        target: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Closed-form per-sample regression target for the diagonal drift.

        G_{t,t} is the drift of the sigma_t-diffusion SDE with the
        interpolant's marginals: forward_diff plus the score correction

            target = dI/dt + ((sigma_t^2 - gamma_ref^2) / 2) * (-noise / sigma_path),

        where ``-noise / sigma_path`` is the per-sample score representation
        of the affine Gaussian path (``sigma_path`` from the
        AffineGaussianPathMixin) and ``gamma_ref`` is the diffusion already
        implied by forward_diff's conditional expectation (gamma_t for the
        Follmer construction, 0 for deterministic paths).

        In the paper's canonical setting (linear path, Gaussian base,
        sigma = sqrt(2(1-t))) this reduces exactly to X_1 - 2 X_0.

        Warning: combining the ``paper`` sigma schedule with a *stochastic*
        interpolation makes the correction grow like 1/sqrt(t) near t = 0
        (target std ~100x normal at the 1e-4 clamp) - per-sample targets stay
        finite but heavy-tailed. Prefer ``gamma_matched`` for stochastic
        interpolations, where the correction is identically zero.
        """
        t_clamped = _clamp_time(t)
        t_expanded = _expand_t(t_clamped, base)

        if self._stochastic_path:
            diff = self.interpolation.forward_diff(
                base=base, target=target, t=t, noise=noise
            )
            gamma_ref_sq = self.interpolation.gamma(t_expanded) ** 2
        else:
            diff = self.interpolation.forward_diff(base=noise, target=target, t=t)
            gamma_ref_sq = torch.zeros_like(t_expanded)

        sigma_path = self.interpolation.sigma(t_expanded)
        score_per_sample = -noise / sigma_path

        sigma = self.sigma_schedule(t_expanded)
        return diff + 0.5 * (sigma**2 - gamma_ref_sq) * score_per_sample

    def drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mean-path diagonal drift G_hat_{t,t} with zero Brownian features.

        Approximation caveat: zero features are the feature *mean*, but
        training always conditions the diagonal on real sampled features and
        the net is nonlinear, so this evaluation is slightly off the training
        distribution: E[G(., xi)] != G(., E[xi]). It is exact for a net that
        ignores the features (e.g. right after teacher warm start, where the
        feature projection is zero-initialized). If a bias-sensitive teacher
        signal is needed (PR 3), average ``G`` over a few sampled Brownian
        feature draws instead.
        """
        brownian_features = None
        if self.brownian_encoder is not None:
            num_features = self.brownian_encoder.num_features_per_channel
            brownian_features = torch.zeros(
                x.shape[0],
                num_features * x.shape[1],
                *x.shape[2:],
                device=x.device,
                dtype=x.dtype,
            )
        return self.G(
            x=x,
            s=t,
            t=t,
            brownian_features=brownian_features,
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
        )

    def forward(
        self,
        base: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Not supported: Ito maps train with the ItoMapTrainer."""
        raise RuntimeError(
            "ItoMapModel has a multi-evaluation loss (diagonal + consistency) "
            "and cannot train through the generic (pred, target) contract; "
            "use scisi.training.ito_map_trainer.ItoMapTrainer."
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
        """Sample the endpoint X_hat_{0,1} with ``num_steps`` map evaluations.

        ``num_steps = 1`` is the headline any-step case; larger values
        partition [0, 1] and reuse the same Brownian path across the partition
        (paper Algorithm 2). Unused solver kwargs (stepper, diffusion_term)
        are accepted for BaseModel._sample_trajectory compatibility.

        Note: in ``kl`` Brownian mode the truncated series cannot represent
        detail below scale ~1/num_kl_terms, so per-substep martingale
        increments are under-dispersed for fine partitions (a warning is
        emitted). ``num_steps = 1`` is unaffected; for many-step sampling use
        ``brownian_mode='path'`` or raise ``num_kl_terms``.
        """
        if (
            (num_steps > 1)
            and (self.path_sampler is not None)
            and (self.path_sampler.mode == "kl")
            and (self.path_sampler.num_kl_terms < 4 * num_steps)
        ):
            logger.warning(
                f"kl Brownian mode with {self.path_sampler.num_kl_terms} terms "
                f"under-disperses martingale increments over 1/{num_steps} "
                "subintervals; use brownian_mode='path' or num_kl_terms >= "
                f"{4 * num_steps} for many-step sampling."
            )
        if (batch_size > 1) and (field_history.shape[0] == 1):
            base, field_history, field_cond, pars_cond = self._prepare_batch(
                batch_size=batch_size,
                base=base,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
            )

        field_history = (
            field_history.to(self.device) if field_history is not None else None
        )
        field_cond = field_cond.to(self.device) if field_cond is not None else None
        pars_cond = pars_cond.to(self.device) if pars_cond is not None else None

        if base is None:
            if self._stochastic_path:
                base = field_history[:, :, :, :, -1]
            else:
                base = torch.randn_like(field_history[:, :, :, :, 0])
        base = base.to(self.device)

        brownian_sample = None
        brownian_features = None
        if self.path_sampler is not None:
            brownian_sample = self.path_sampler.sample(base.shape, self.device)
            brownian_features = self.encode_brownian(brownian_sample)

        t_grid = torch.linspace(0, 1, num_steps + 1, device=self.device)

        x = base
        for i in range(num_steps):
            s = t_grid[i].expand(x.shape[0], 1)
            t = t_grid[i + 1].expand(x.shape[0], 1)
            x = self.map(
                x=x,
                s=s,
                t=t,
                brownian_sample=brownian_sample,
                brownian_features=brownian_features,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
            ).detach()

        x = x.cpu()

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
        """Autoregressive rollout of one-step (or few-step) endpoint samples."""
        return self._sample_trajectory(
            field_history=field_history,
            base=base,
            batch_size=batch_size,
            num_steps=num_steps,
            num_physical_steps=num_physical_steps,
            field_cond=field_cond,
            pars_cond=pars_cond,
            gaussian_base=not self._stochastic_path,
        )
