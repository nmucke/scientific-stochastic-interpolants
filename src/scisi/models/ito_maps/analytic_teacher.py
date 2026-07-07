"""Analytic Gaussian-shell drift teacher for Ito-map fine-tuning.

Method 3 of docs/plans/deterministic_to_ito_map_finetuning.md. Under the
approximation p(X_1 | x_n, h) ~= N(F(x_n, h), rho^2 I), the interpolant
marginal is Gaussian and the diagonal drift target G_{t,t} is a closed-form
function of the frozen deterministic model F - a noise-free regression target
for the first few fine-tuning epochs (fast, low gradient variance), after
which the trainer switches to the exact single-sample data targets
(``ItoMapTrainer(teacher_warmup_epochs=...)``), removing the Gaussian bias.
"""

from typing import Optional

import torch
import torch.nn as nn

from scisi.deterministic_models.deterministic_model import DeterministicModel
from scisi.models.interpolations import _clamp_time, _expand_t
from scisi.models.ito_maps.brownian import SigmaSchedule


class GaussianShellTeacher(nn.Module):
    """Closed-form diagonal drift teacher under a Gaussian-shell step law.

    With mean_t = alpha(t) * anchor + beta(t) * F and u = x - mean_t:

        Var_t   = beta^2 rho^2 + sigma_path^2
        score   = -u / Var_t
        b_t(x)  = alpha' anchor + beta' F
                  + (beta' beta rho^2 + sigma_path' sigma_path) u / Var_t
        G_{t,t} = b_t + (sigma_t^2 / 2) score,

    where ``sigma_path`` is the interpolation's source-noise scale (gamma(t)
    sqrt(t) for the repo's Follmer paths, alpha(t) for Gaussian-base paths)
    and ``anchor`` is the point-mass base x_n (zero for Gaussian-base paths).

    Implements the duck-typed teacher interface
    ``drift(x, t, field_history, field_cond, pars_cond)``, so
    ``ItoMapModel.distill_from`` accepts it directly (no weight surgery - the
    teacher has no drift network).

    Args:
        interpolation: Interpolation of the Ito map being trained.
        sigma_schedule: Sigma schedule of the Ito map being trained.
        mean_model: Frozen deterministic mean model F. ``None`` means residual
            coordinates (F = 0, anchor = 0, rho defaults to 1) - the warm-up
            teacher for Method 1's residual map.
        residual_std: Calibrated per-channel rho. ``None`` means 1 (correct in
            normalized residual coordinates).
    """

    def __init__(
        self,
        interpolation: nn.Module,
        sigma_schedule: SigmaSchedule,
        mean_model: Optional[DeterministicModel] = None,
        residual_std: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize the teacher."""
        super(GaussianShellTeacher, self).__init__()

        self.interpolation = interpolation
        self.sigma_schedule = sigma_schedule
        self._stochastic_path = hasattr(interpolation, "gamma")

        self.mean_model = mean_model
        if self.mean_model is not None:
            for param in self.mean_model.parameters():
                param.requires_grad_(False)
            self.mean_model.eval()

        if residual_std is None:
            residual_std = torch.ones(1)
        if torch.any(residual_std <= 0):
            raise ValueError(
                f"Residual std must be positive, got {residual_std.tolist()}."
            )
        self.register_buffer("residual_std", residual_std.clone().float())

    def _rho(self, x: torch.Tensor) -> torch.Tensor:
        """Residual scale broadcast to the state layout [1, C, 1, 1]."""
        return self.residual_std.view(1, -1, *([1] * (x.ndim - 2)))

    def drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Analytic diagonal drift target G^gauss_{t,t}(x)."""
        t_expanded = _expand_t(_clamp_time(t), x)

        if self.mean_model is None:
            # Residual coordinates: F = 0, anchor = 0.
            anchor = torch.zeros_like(x)
            mean_pred = torch.zeros_like(x)
        else:
            if field_history is None:
                raise ValueError(
                    "GaussianShellTeacher with a mean model needs "
                    "field_history: its last slice is the current state x_n."
                )
            x_n = field_history[:, :, :, :, -1]
            with torch.no_grad():
                mean_pred = self.mean_model._step(
                    x_n,
                    field_history=field_history,
                    field_cond=field_cond,
                    pars_cond=pars_cond,
                )
            # Point-mass base for Follmer paths; Gaussian-base paths have a
            # zero-mean base and hence no deterministic anchor.
            anchor = x_n if self._stochastic_path else torch.zeros_like(x_n)

        interpolation = self.interpolation
        alpha = interpolation.alpha(t_expanded)
        alpha_diff = interpolation.alpha_diff(t_expanded)
        beta = interpolation.beta(t_expanded)
        beta_diff = interpolation.beta_diff(t_expanded)
        sigma_path = interpolation.sigma(t_expanded)
        sigma_path_diff = interpolation.sigma_diff(t_expanded)

        rho_sq = self._rho(x) ** 2
        variance = beta**2 * rho_sq + sigma_path**2
        u = x - alpha * anchor - beta * mean_pred

        coefficient = (
            beta_diff * beta * rho_sq
            + sigma_path_diff * sigma_path
            - 0.5 * self.sigma_schedule(t_expanded) ** 2
        )
        return alpha_diff * anchor + beta_diff * mean_pred + coefficient * u / variance
