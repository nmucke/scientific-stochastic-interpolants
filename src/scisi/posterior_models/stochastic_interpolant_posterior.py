import pdb
from functools import partial
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import tqdm

from scisi.posterior_models.base_posterior import BasePosterior
from scisi.sampling.sde_solvers import euler_maruyama_step, heun_step

# Pseudo-time below which the guidance correction is switched off. The
# observation interpolant is singular at tau = 0 (beta_0 = 0), so the
# correction is applied only from tau = dtau onwards (paper, Section
# "implementation": "the correction is applied from tau = dtau onwards").
MIN_TIME = 1e-4


class StochasticInterpolantPosterior(BasePosterior):
    """Stochastic-interpolant SDE posterior sampler (paper Sampler 1).

    Realises the SI-SDE member of the unified family

        b_post = b_prior + w_tau * (G_tau @ Sbar),
        w_tau  = a_tau + 1/2 gamma_tau**2,

    with native diffusion ``g_tau = gamma_tau`` and source the point mass
    ``delta_{x0}`` (so the SI loop is initialised at ``x = x0`` exactly). The
    prior drift is the trained SI drift ``b_theta`` (the score is built in), the
    weight ``w_tau`` is formed from the path's velocity--score coefficient
    ``a_tau`` plus ``1/2 gamma_tau**2``, and the corrected likelihood score
    ``G_tau @ Sbar`` is supplied by the interpolant likelihood. The whole
    update is assembled as a single Euler--Maruyama step

        x <- x + b_post * dtau + gamma_tau * sqrt(dtau) * z.

    The optional SMC particle-filter resampling (``resample``) is **off by
    default**; it is an add-on not part of the paper's methodology.
    """

    def __init__(
        self,
        model: nn.Module,
        likelihood_model: nn.Module,
        diffusion_term: Optional[Callable] = None,
        resample: bool = False,
    ) -> None:
        """
        Initialize stochastic interpolant posterior.

        Args:
            model: Trained SI model (``FollmerStochasticInterpolant``).
            likelihood_model: Interpolant Gaussian likelihood returning the
                corrected score ``G_tau @ Sbar``.
            diffusion_term: Diffusion schedule ``g_tau``; defaults to the
                native ``gamma_tau``.
            resample: Optional SMC resampling, off by default (not part of the
                paper method).
        """
        super(StochasticInterpolantPosterior, self).__init__(
            model=model,
            likelihood_model=likelihood_model,
            diffusion_term=diffusion_term,
            gaussian_base=False,
        )

        self.resample = resample
        self.counter = 0
        self.weights = None
        self.log_likelihood = None
        self.integral_variance = lambda t: 2 / 3 - t.sqrt() + (1 / 3) * (t.sqrt()) ** 3

    def _guidance_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Guidance weight w_tau = a_tau + 1/2 g_tau**2 (paper Eq. guidance_weight).

        ``a_tau`` is the path's velocity--score coefficient; ``g_tau`` is the
        diffusion schedule (native ``gamma_tau`` by default).
        """
        a_tau = self.model.interpolation.velocity_score_coeff(t)
        return a_tau + 0.5 * self.diffusion_term(t) ** 2

    def _one_step(
        self,
        base: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: torch.Tensor,
        pars_cond: torch.Tensor,
        observations: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """One Euler--Maruyama step of the SI-SDE posterior."""

        # Fresh grad-enabled leaf (not an in-place mutation of the caller's tensor,
        # which can corrupt the autoregressive feedback / fail on a non-leaf base).
        base = base.detach().requires_grad_(True)

        if self.log_likelihood is None:
            self.log_likelihood = []

        # Prior drift b_prior = b_theta (score already built into the SI drift).
        drift = self.model.drift(base, t, field_history, field_cond, pars_cond)

        # Brownian increment z * sqrt(dtau).
        noise = self.diffusion_term(t) * torch.randn_like(base) * dt.sqrt()

        # Guidance correction, applied only from tau = dtau onwards.
        if float(t.reshape(-1)[0]) >= MIN_TIME:
            corrected_score, log_likelihood = self.likelihood_model.score(
                observations=observations,
                x=base,
                t=t,
                drift=drift,
                diffusion_term=self.diffusion_term,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
                dt=dt,
            )
            corrected_score = corrected_score.detach()
            w_tau = self._guidance_weight(t)

            self.log_likelihood.append(
                torch.nan_to_num(log_likelihood.detach(), nan=float("-inf"))
            )
        else:
            corrected_score = torch.zeros_like(base)
            w_tau = torch.zeros_like(t)

        # Combined posterior drift b_post = b_prior + w_tau * G_tau * Sbar.
        posterior_drift = drift + w_tau * corrected_score

        # Single Euler--Maruyama update: x += b_post * dtau + g * sqrt(dtau) * z.
        base = base + posterior_drift * dt + noise

        return base.detach()

    def _post_sample(
        self,
        base: torch.Tensor,
        observations: torch.Tensor,
        field_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optional SMC resampling (off by default)."""

        if not self.resample:
            return base, field_history

        diff = observations - self.likelihood_model.obs_operator(base.to(self.device))

        log_likelihood = torch.linalg.norm(diff, dim=1) ** 2
        log_likelihood = - 0.5 * log_likelihood / self.likelihood_model.original_variance

        if self.weights is None:
            self.weights = torch.ones(base.shape[0], device=self.device) / base.shape[0]

        self.weights = torch.exp(log_likelihood) * self.weights
        self.weights = self.weights / self.weights.sum()  # type: ignore[attr-defined]

        self.log_likelihood = None

        N_eff = 1 / (self.weights ** 2).sum()
        N_eff = N_eff.to("cpu").item()

        N_threshold = base.shape[0] / 2

        if N_eff < N_threshold:
            resample_indices = torch.multinomial(
                self.weights, num_samples=base.shape[0], replacement=True
            )
            resample_indices = resample_indices.to("cpu")

            self.weights = 1 / base.shape[0] * torch.ones_like(self.weights)

            print(f"Resampling {resample_indices.shape[0]} particles")

            return base[resample_indices], field_history[resample_indices]

        return base, field_history
