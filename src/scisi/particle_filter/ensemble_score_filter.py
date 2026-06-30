"""Ensemble Score Filter (EnSF) baseline -- Bao et al. (bao_ensemble_2024).

GENERATIVE / SCORE baseline (NOT a true-solver method)
------------------------------------------------------
Unlike the classical EnKF / particle-filter baselines (which propagate members
with the real jax-cfd solver), EnSF is a *score-based generative* filter and so
reuses the **learned prior**: at each assimilation step the forecast ensemble is
drawn from ``p_theta(. | x^{n-1})`` (one learned-prior physical-step sample per
member), exactly the prior the SI / FM samplers use. EnSF then performs a
score-based ANALYSIS update that injects the Gaussian observation likelihood
``N(y; H x, R)`` via a short reverse diffusion over the ensemble's empirical
score.

This is a reasonable, finite, documented adaptation of EnSF to our harness:

* Forecast: ``x^n_f ~ p_theta(. | x^{n-1})`` per member (learned prior).
* Analysis: starting from a Gaussian latent, run ``analysis_steps`` of a reverse
  diffusion whose drift combines
    - the PRIOR score, approximated by the forecast ensemble's empirical
      Gaussian score ``s_prior(z) = -(z - mu_f) / diag(Cov_f)`` (the diagonal
      Gaussian fit of the forecast ensemble -- the "ensemble score" of EnSF), and
    - the OBSERVATION-likelihood score ``s_obs(z) = H^T R^{-1} (y - H z)`` (a
      tempered Gaussian guidance term),
  annealed by a cosine-like temperature ``g(t)`` so the observation term is
  switched on smoothly (Bao et al.'s damping function). The reverse diffusion is
  integrated with Euler--Maruyama.

The empirical diagonal-Gaussian prior score is the documented APPROXIMATION (the
exact EnSF trains a per-step score network on the forecast ensemble; here the
forecast ensemble is summarised by its first two diagonal moments, which is
finite, cheap and adequate for a baseline). The result is an
:class:`AssimResult`-compatible posterior trajectory, scored by the same
``compute_metrics`` path as every other method.
"""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn as nn

from scisi.likelihood_models.observation_operators import LinearObservationOperator


class EnsembleScoreFilter(nn.Module):
    """Ensemble Score Filter (single-window, learned-prior forecast)."""

    def __init__(
        self,
        analysis_steps: int = 20,
        damping: float = 1.0,
        eps: float = 1e-3,
    ) -> None:
        super(EnsembleScoreFilter, self).__init__()
        self.analysis_steps = analysis_steps
        self.damping = damping
        self.eps = eps

    @staticmethod
    def _empirical_gaussian(ensemble: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Diagonal Gaussian fit (mean, variance) of an ensemble [E, ...]."""
        mu = ensemble.mean(dim=0, keepdim=True)
        var = ensemble.var(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        return mu, var

    def _analysis_update(
        self,
        forecast: torch.Tensor,
        observations: torch.Tensor,
        obs_operator: LinearObservationOperator,
        variance: float,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Score-based analysis update injecting N(y; H z, R).

        Runs a short reverse diffusion whose drift is the sum of the forecast
        ensemble's empirical Gaussian (prior) score and the tempered Gaussian
        observation-likelihood score. Returns the analysis ensemble.
        """
        E = forecast.shape[0]
        device = forecast.device

        mu_f, var_f = self._empirical_gaussian(forecast)  # [1, ...]

        # Initialise the analysis ensemble at the forecast (the prior is the
        # forecast distribution; we relax it toward the observation-consistent
        # posterior). A small jitter gives the reverse diffusion non-zero spread.
        z = forecast.clone()
        std_f = var_f.sqrt()
        z = z + 0.05 * std_f * torch.randn(
            z.shape, generator=generator, device=device
        )

        n = self.analysis_steps
        dt = 1.0 / n
        for i in range(n):
            # Pseudo-time t in (0, 1]; temperature ramps the observation term on.
            t = 1.0 - i * dt
            g = self.damping * (1.0 - t)  # 0 -> 1 across the sweep

            # Empirical Gaussian prior score (pulls back toward the forecast).
            prior_score = -(z - mu_f) / var_f

            # Observation-likelihood score, tempered by g. BOUNDED ensemble-Kalman /
            # PiGDM-style gain: normalise the residual by the forecast obs-space
            # variance + R, i.e. H^T (H Cov_f H^T + R)^{-1} (y - H z) with the
            # diagonal forecast variance, instead of the raw H^T R^{-1} (y - H z).
            # The raw 1/variance (=400 at R=0.0025) made the explicit Langevin step
            # scale as ~var_f/R >> 1 (past the stability limit) and diverged to 1e12;
            # dividing by (var_f_obs + R) caps each step to a Kalman-like increment.
            resid = observations - obs_operator(z)  # [E, N_y]
            var_f_obs = obs_operator(var_f)  # [1, N_y] diagonal forecast obs variance
            obs_score = obs_operator.transpose(resid / (var_f_obs + variance))  # [E,C,H,W]

            drift = prior_score + g * obs_score
            noise = torch.randn(z.shape, generator=generator, device=device)
            z = z + drift * dt * std_f**2 + std_f * (2.0 * dt) ** 0.5 * noise * (
                t**0.5
            )

        return z.detach()

    @torch.no_grad()
    def run(
        self,
        prior,  # LoadedPrior
        truth_obs,  # TruthAndObs
        obs_operator: LinearObservationOperator,
        ensemble_size: int,
        variance: float,
        num_steps: int,
        num_physical_steps: int,
        len_field_history: int,
        seed: int,
        device: str,
        gaussian_base: bool,
    ):
        """Run the autoregressive EnSF and return an AssimResult.

        Imported lazily by the driver (which owns the AssimResult dataclass).
        """
        from cases.navier_stokes._ns_pipeline import AssimResult  # local import

        model = prior.si_model
        gen = torch.Generator(device="cpu").manual_seed(int(seed) + 7919)

        # Replicate the single-trajectory history across the ensemble.
        base = truth_obs.init_base
        field_history = truth_obs.field_history
        field_cond = truth_obs.field_cond
        pars_cond = truth_obs.pars_cond
        if ensemble_size > 1 and field_history.shape[0] == 1:
            base, field_history, field_cond, pars_cond = model._prepare_batch(
                batch_size=ensemble_size,
                base=base,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
            )

        C, Hh, Ww = obs_operator.C, obs_operator.H, obs_operator.W
        T = num_physical_steps
        posterior = torch.zeros(ensemble_size, C, Hh, Ww, T)
        for t in range(len_field_history):
            posterior[..., t] = field_history[..., t]

        observations = truth_obs.observations  # [1, N_y, T]
        n_assim = T - len_field_history
        start = time.time()

        for t in range(len_field_history, T):
            # --- Forecast: one learned-prior physical step per member -------- #
            base_in = None if gaussian_base else base
            forecast, _ = model.sample(
                base=base_in,
                field_history=field_history.to(device),
                field_cond=(field_cond.to(device) if field_cond is not None else None),
                pars_cond=(pars_cond.to(device) if pars_cond is not None else None),
                num_steps=num_steps,
                batch_size=ensemble_size,
                return_field_history=True,
                gaussian_base=gaussian_base,
            )
            forecast = forecast.to("cpu")

            # --- Analysis: score-based observation update -------------------- #
            obs_t = observations[:, :, t].expand(ensemble_size, -1)  # [E, N_y]
            analysis = self._analysis_update(
                forecast=forecast,
                observations=obs_t,
                obs_operator=obs_operator,
                variance=variance,
                generator=gen,
            )

            posterior[..., t] = analysis
            # Feed the analysis ensemble back as the next step's history.
            base = analysis
            field_history = torch.cat(
                [field_history[..., 1:], analysis.unsqueeze(-1)], dim=-1
            )

        elapsed = time.time() - start
        return AssimResult(
            posterior_trajectory=posterior,
            true_trajectory=truth_obs.true_trajectory,
            nfe_per_step=float(num_steps),
            seconds_per_step=elapsed / max(n_assim, 1),
        )


__all__ = ["EnsembleScoreFilter"]
