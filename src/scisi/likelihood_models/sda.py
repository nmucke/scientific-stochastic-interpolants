"""SDA (score-based data assimilation) likelihood -- BASELINE.

Single-window adaptation of Rozet & Louppe, "Score-based Data Assimilation"
(rozet_score-based_2023).

IMPORTANT -- SCOPE OF THIS IMPLEMENTATION
-----------------------------------------
The full SDA method assimilates an entire trajectory **all at once** through a
trajectory score network (a score over the joint state sequence). Our harness is
**autoregressive**: it assimilates one observation window (one physical step) at
a time, feeding each posterior sample back as the next step's history. This
module therefore implements SDA's *single-window* guidance only -- the
DPS-style Gaussian observation guidance with SDA's covariance approximation,
applied to one window's state. It is NOT the full all-at-once trajectory score.

SDA's single-window guidance (their Eq. for the observation log-likelihood
gradient) approximates the measurement covariance seen at pseudo-time ``tau`` by

    Gamma_tau = sigma^2 I + gamma_tau^2 H H^T,

i.e. the raw noise ``R = sigma^2 I`` *inflated* by the denoiser variance
projected into observation space, with ``gamma_tau^2`` the source variance
``rho_tau`` read off the interpolation path (``gamma^2 t`` for SI, ``alpha^2``
for FM). The guidance score is then

    g_tau = grad_{x_tau} [ -1/2 (y - H xhat_1)^T Gamma_tau^{-1} (y - H xhat_1) ],

with ``xhat_1 = E[x_1 | x_tau]`` the Tweedie denoiser (autograd through the
network, as in DPS). This is the "diagonal-plus-low-rank" heuristic SDA uses to
avoid forming the dense denoiser covariance.

Returns ``(score, log_likelihood)`` to match the
``InterpolantGaussianLikelihood`` / ``DPSGaussianLikelihood`` interface so it
plugs into the existing SI / FM posteriors unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from scisi.likelihood_models.observation_operators import LinearObservationOperator

MIN_TIME = 1e-4


class SDALikelihood(nn.Module):
    """Single-window SDA observation-guidance likelihood (see module docstring)."""

    def __init__(
        self,
        model: nn.Module,
        obs_operator: nn.Module = LinearObservationOperator,
        variance: float = 0.05,
        ensemble_size: int = 1,
        model_class: str = "si",
    ) -> None:
        super(SDALikelihood, self).__init__()
        self.obs_operator = obs_operator
        self.model = model
        self.original_variance = variance
        self.ensemble_size = ensemble_size
        if model_class not in ("si", "fm"):
            raise ValueError(f"model_class must be 'si' or 'fm', got {model_class!r}.")
        self.model_class = model_class
        self.anchor = "x0" if model_class == "si" else "zeros"
        self.interpolant = self.model.interpolation
        self._HHt: Optional[torch.Tensor] = None
        # FIX 3: ask DiffusionPosterior to apply the raw likelihood score with the
        # FM score->state coefficient (a_tau + 0.5 g^2) instead of its legacy
        # 0.5 g^2 sqrt(t) weight (which under-powers SDA ~10-16x).
        self.guidance_weight = "fm_coeff"

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Unused (score interface is the entry point)."""
        pass

    def _get_HHt(self, ref: torch.Tensor) -> torch.Tensor:
        """Return ``H H^T`` (cached), shape [N_y, N_y]."""
        if (
            self._HHt is None
            or self._HHt.device != ref.device
            or self._HHt.dtype != ref.dtype
        ):
            H = self.obs_operator.obs_matrix.to(device=ref.device, dtype=ref.dtype)
            self._HHt = H @ H.transpose(0, 1)
        return self._HHt

    def _denoise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor],
        pars_cond: Optional[torch.Tensor],
        score: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tweedie denoiser xhat_1 = E[x_1 | x_tau] plus the source variance rho."""
        if t.dim() < x.dim():
            t_grid = t.reshape(t.shape[0], *([1] * (x.dim() - 1)))
        else:
            t_grid = t
        alpha = self.interpolant.alpha(t_grid)
        beta = self.interpolant.beta(t_grid)
        sigma_tau = self.interpolant.sigma(t_grid)

        if self.model_class == "fm":
            s = self.model.score(x, t, field_history, field_cond, pars_cond)
        else:
            v = self.model.drift(x, t, field_history, field_cond, pars_cond)
            a0_si = field_history[..., -1]
            s = self.interpolant.score_from_velocity(x=x, v=v, t=t_grid, a0=a0_si)

        a0 = field_history[..., -1] if self.anchor == "x0" else torch.zeros_like(x)
        xhat1 = (x + sigma_tau**2 * s - alpha * a0) / beta
        # rho_tau = sigma_tau^2 (the source / denoiser variance scale).
        rho = (sigma_tau**2).reshape(-1)[0]
        return xhat1, rho

    def score(
        self,
        observations: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        drift: Optional[torch.Tensor] = None,
        score: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SDA single-window guidance score (Gamma_tau covariance, autograd)."""
        t_clamped = torch.clamp(t, min=MIN_TIME, max=1.0 - MIN_TIME)

        # Compute the guidance on a DETACHED grad-enabled copy so the caller's
        # working state (the posterior's autoregressive ``base``) is never mutated
        # in place by ``requires_grad_`` -- doing so corrupts the SI feedback loop
        # (the next step's base would no longer equal field_history[..., -1]).
        x_g = x.detach().requires_grad_(True)

        with torch.enable_grad():
            xhat1, rho = self._denoise(
                x_g, t_clamped, field_history, field_cond, pars_cond, score
            )
            residual = observations - self.obs_operator(xhat1)  # [B, N_y]

            # Gamma_tau = sigma^2 I + gamma_tau^2 H H^T  (SDA diagonal-plus-low-rank).
            HHt = self._get_HHt(x_g)  # [N_y, N_y]
            N_y = HHt.shape[0]
            eye = torch.eye(N_y, device=HHt.device, dtype=HHt.dtype)
            gamma = self.original_variance * eye + float(rho) * HHt
            # Gamma is held fixed w.r.t. x (detached); the gradient flows only
            # through the residual (xhat_1). Writing the quadratic form with the
            # solve INSIDE the graph makes autograd return the correct
            # grad = Gamma^{-1} (y - H xhat_1) (Gamma symmetric).
            sol = torch.linalg.solve(
                gamma.detach(), residual.transpose(0, 1)
            ).transpose(0, 1)  # [B, N_y]
            weighted = 0.5 * (residual * sol).sum(dim=1)  # [B] = 1/2 ||r||^2_{Gamma^-1}

            grad = torch.autograd.grad(outputs=weighted.sum(), inputs=x_g)[0]

        # FIX 3: drop the 1/||Gamma^-1 r|| step-normalisation -- it is exactly the
        # DPS rescaling SDA warns against, and with DiffusionPosterior's legacy
        # 0.5 g^2 sqrt(t) application weight it under-powered the guidance ~10-16x.
        # Return the RAW likelihood score s_lik = -grad = J^T H^T Gamma^-1 r; the
        # correct FM score->state application weight (a_tau + 0.5 g^2) is applied
        # by DiffusionPosterior, which keys on ``self.guidance_weight == 'fm_coeff'``
        # (set below). Folding the weight into the returned value instead would make
        # it diverge as t -> 0 (a_tau/g^2 ~ 1/(g0^2 t^2)); applying it in the
        # posterior keeps the returned guidance bounded and matches the analytical
        # ``sda_posterior`` exactly.
        guidance = -grad

        log_likelihood = -weighted.detach()
        return guidance.detach(), log_likelihood


__all__ = ["SDALikelihood"]
