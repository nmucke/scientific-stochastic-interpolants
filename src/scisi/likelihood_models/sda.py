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

SDA's single-window guidance (Rozet & Louppe Eq. 15) approximates the
measurement covariance seen at pseudo-time ``tau`` by

    Gamma_tau = sigma^2 I + gamma_sda * (sigma_tau / mu_tau)^2 H H^T,

i.e. the raw noise ``R = sigma^2 I`` *inflated* by the Gaussian denoiser
variance ``(sigma/mu)^2`` projected into observation space and scaled by the
tunable spectral factor ``gamma_sda`` (SDA default ``1e-2``; their ``Gamma =
gamma I`` approximation of ``Q Lambda (Lambda + I)^-1 Q^-1``). In our affine
path convention the denoiser mean weight is ``mu_tau = beta_tau`` and the source
scale is ``sigma_tau`` (``= alpha_tau`` for FM, ``gamma_tau sqrt(t)`` for SI),
so ``(sigma_tau / mu_tau)^2 = (sigma_tau / beta_tau)^2`` -- SDA's ``(sigma/mu)^2``
verbatim. The guidance score is then

    g_tau = grad_{x_tau} [ -1/2 (y - H xhat_1)^T Gamma_tau^{-1} (y - H xhat_1) ],

with ``xhat_1 = E[x_1 | x_tau]`` the Tweedie denoiser (autograd through the
network, as in DPS). ``g_tau`` is injected into the reverse SDE with the SDA
weight ``g_tau^2`` (the h-transform / classifier-guidance footing: the
likelihood score enters ``-g^2 (s_prior + s_lik)`` on equal footing with the
prior score), applied by ``DiffusionPosterior`` via ``guidance_weight``.

Note: SDA's released code uses the purely diagonal ``sigma^2 + gamma_sda
(sigma/mu)^2`` in observation space (i.e. ``H H^T`` dropped); for a selection /
subsampling ``H`` (``H H^T = I``) that coincides with the ``H H^T`` form used
here, and for a general ``H`` this is the more faithful rendering of Eq. 15's
``A Gamma A^T``.

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
        gamma_sda: float = 1e-2,
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
        # SDA's tunable spectral factor gamma in Gamma_tau = R + gamma (sigma/mu)^2 HH^T
        # (Rozet & Louppe Eq. 15; released-code default 1e-2).
        self.gamma_sda = float(gamma_sda)
        # Inject the raw likelihood score into the reverse SDE with the SDA weight
        # g_tau^2 (h-transform / classifier-guidance footing: s_lik enters
        # -g^2 (s_prior + s_lik) on equal footing with the prior score). Keyed by
        # DiffusionPosterior on ``guidance_weight == 'g_squared'``.
        self.guidance_weight = "g_squared"

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
        # SDA denoiser variance in obs space: gamma_sda * (sigma/mu)^2 with the
        # denoiser mean weight mu_tau = beta_tau (Rozet & Louppe Eq. 15). NOT the
        # bare source variance sigma_tau^2 -- the 1/mu^2 = 1/beta^2 factor is what
        # makes the covariance grow toward the noise end as SDA prescribes.
        rho = (self.gamma_sda * sigma_tau**2 / beta**2).reshape(-1)[0]
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

            # Gamma_tau = sigma^2 I + gamma_sda (sigma/mu)^2 H H^T (SDA Eq. 15;
            # rho already carries gamma_sda * (sigma_tau/beta)^2).
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

        # Return the RAW likelihood score s_lik = -grad = J^T H^T Gamma^-1 r, with
        # NO 1/||Gamma^-1 r|| step-normalisation (that DPS rescaling is exactly what
        # SDA warns against). The SDA injection weight g_tau^2 -- the h-transform /
        # classifier-guidance footing that puts s_lik on the same -g^2 footing as
        # the prior score -- is applied by DiffusionPosterior, which keys on
        # ``self.guidance_weight == 'g_squared'`` (set above).
        guidance = -grad

        log_likelihood = -weighted.detach()
        return guidance.detach(), log_likelihood


__all__ = ["SDALikelihood"]
