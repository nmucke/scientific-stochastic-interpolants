import pdb
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import tqdm

from scisi.posterior_models.base_posterior import BasePosterior
from scisi.sampling.sde_solvers import euler_maruyama_step, heun_step

# Pseudo-time below which the guidance correction is switched off (the
# observation interpolant is singular at tau = 0, beta_0 = 0).
MIN_TIME = 1e-4


def endpoint_vanishing_diffusion(
    interpolation: nn.Module, scale: float = 1.0
) -> Callable:
    """Endpoint-vanishing FM-SDE diffusion ``g_tau = scale * sqrt(alpha*beta)``.

    This schedule vanishes at both endpoints and, crucially, keeps the lifted
    drift finite as ``tau -> 1``: the recovered FM score (Eq. fm_score) diverges
    there (its denominator carries ``alpha_tau -> 0``), and ``g_tau**2 propto
    alpha*beta`` vanishes fast enough that ``1/2 g_tau**2 s_tau`` stays bounded
    (paper, Sampler 2).

    Args:
        interpolation: The model interpolation exposing ``alpha`` and ``beta``.
        scale: Overall multiplier on the schedule.

    Returns:
        Callable ``t -> g_tau``.
    """

    def g(t: torch.Tensor) -> torch.Tensor:
        alpha = interpolation.alpha(t)
        beta = interpolation.beta(t)
        return scale * torch.sqrt(torch.clamp(alpha * beta, min=0.0))

    return g


class FlowMatchingPosterior(BasePosterior):
    """Flow-matching posterior sampler (paper Samplers 2 & 3).

    Realises the two FM members of the unified family

        b_post = b_prior + w_tau * (G_tau @ Sbar),
        w_tau  = a_tau + 1/2 g_tau**2,

    on the FM path (anchor ``a0 = 0``, source ``N(0, I)``). A single
    ``_one_step`` is parameterised by the diffusion term:

    - ``diffusion_term = None``  -> **FM-ODE** (Sampler 3): ``g_tau = 0``,
      ``w_tau = a_tau``, prior drift ``v_theta`` only, deterministic Euler step
      (no Brownian increment). Randomness enters solely through the
      ``N(0, I)`` latent.
    - ``diffusion_term`` callable -> **FM-SDE** (Sampler 2): ``g_tau > 0``
      (e.g. the endpoint-vanishing ``sqrt(alpha*beta)``), prior drift
      ``v_theta + 1/2 g_tau**2 s_tau`` with the score recovered via
      ``FlowMatchingModel.score`` (Eq. fm_score), Brownian increment retained.

    The FM source ``N(0, I)`` is used directly with no ``Phi_0^obs``
    reweighting (asserted), since at ``tau = 0`` the latent is independent of
    ``x_1`` and the tilt is constant.
    """

    def __init__(
        self,
        model: nn.Module,
        likelihood_model: nn.Module,
        diffusion_term: Optional[Callable] = None,
        drift_time_shift: float = 0.0,
    ) -> None:
        """
        Initialize the flow-matching posterior.

        Args:
            model: Trained FM model (``FlowMatchingModel``).
            likelihood_model: Interpolant Gaussian likelihood returning the
                corrected score ``G_tau @ Sbar`` (constructed with the FM source
                moments and ``anchor='zeros'``).
            diffusion_term: ``None`` -> FM-ODE (deterministic); a callable
                ``t -> g_tau`` -> FM-SDE (e.g. ``endpoint_vanishing_diffusion``).
            drift_time_shift: Drift-evaluation pseudo-time offset in units of
                ``dt`` (see :class:`BasePosterior`). ``0.0`` (default) is the
                current left-endpoint Euler; ``1.0`` recovers the bespoke FM/DM
                right-endpoint convention, ``0.5`` the midpoint.
        """
        # The base class would try ``model.interpolation.gamma`` /
        # ``model.diffusion_term`` for a None diffusion, neither of which exists
        # on the FM model. We instead resolve the diffusion here: a zero
        # schedule for the ODE, the supplied callable for the SDE.
        self.is_ode = diffusion_term is None
        if self.is_ode:
            resolved_diffusion = lambda t: torch.zeros_like(t)
        else:
            resolved_diffusion = diffusion_term

        super(FlowMatchingPosterior, self).__init__(
            model=model,
            likelihood_model=likelihood_model,
            diffusion_term=resolved_diffusion,
            gaussian_base=True,
            drift_time_shift=drift_time_shift,
        )

        # SPEC assert (a): FM init needs no Phi_0^obs reweighting; the source
        # N(0, I) is used directly. The likelihood must therefore carry the FM
        # anchor a0 = 0 (no x0 term in the observation interpolant at tau = 0).
        assert getattr(self.likelihood_model, "anchor", "zeros") == "zeros", (
            "FM posterior requires anchor='zeros' (a0=0) so that Phi_0^obs is "
            "constant and the N(0, I) init needs no reweighting."
        )

    def _guidance_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Guidance weight w_tau = a_tau + 1/2 g_tau**2 (paper Eq. guidance_weight)."""
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
        """One step of the FM posterior (ODE or SDE, per ``diffusion_term``)."""

        # Fresh grad-enabled leaf (not an in-place mutation of the caller's tensor,
        # which can corrupt the autoregressive feedback / fail on a non-leaf base).
        base = base.detach().requires_grad_(True)

        # FM velocity v_theta.
        velocity = self.model.drift(base, t, field_history, field_cond, pars_cond)

        apply_guidance = float(t.reshape(-1)[0]) >= MIN_TIME

        # FM score s_tau (Eq. fm_score) is needed for the SDE lift and for the
        # FM source moments mu_s = -alpha^2 s in the guidance correction. It is
        # recovered from the velocity ALREADY computed above via the shared
        # velocity->score identity (exactly what ``model.score`` does internally,
        # minus its redundant second UNet forward at the same ``(x, t)``).
        if (not self.is_ode) or apply_guidance:
            t_expanded = (
                t.reshape(t.shape[0], *([1] * (base.dim() - 1)))
                if t.dim() < base.dim()
                else t
            )
            score = self.model.interpolation.score_from_velocity(
                x=base, v=velocity, t=t_expanded, a0=torch.zeros_like(base)
            )
        else:
            score = None

        # Prior drift b_prior. ODE: g = 0 so b_prior = v. SDE: lift with the
        # recovered score, b_prior = v + 1/2 g^2 s (s via Eq. fm_score).
        if self.is_ode:
            prior_drift = velocity
            noise = torch.zeros_like(base)
        else:
            g = self.diffusion_term(t)
            prior_drift = velocity + 0.5 * g**2 * score
            noise = g * torch.randn_like(base) * dt.sqrt()

            # SPEC assert (b): with g propto sqrt(alpha*beta) the lifted drift
            # stays bounded as tau -> 1 (the recovered FM score diverges there).
            assert torch.isfinite(prior_drift).all(), (
                "FM-SDE prior drift is non-finite; use an endpoint-vanishing "
                "g_tau (e.g. sqrt(alpha*beta)) to keep 1/2 g^2 s bounded as "
                "tau -> 1."
            )

        # Guidance correction, applied only from tau = dtau onwards.
        if apply_guidance:
            corrected_score, _ = self.likelihood_model.score(
                observations=observations,
                x=base,
                t=t,
                drift=velocity,
                score=score,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
                dt=dt,
            )
            corrected_score = corrected_score.detach()
            w_tau = self._guidance_weight(t)
        else:
            corrected_score = torch.zeros_like(base)
            w_tau = torch.zeros_like(t)

        # Combined posterior drift b_post = b_prior + w_tau * G_tau * Sbar.
        posterior_drift = prior_drift + w_tau * corrected_score

        # Single (Euler / Euler--Maruyama) update.
        base = base + posterior_drift * dt + noise

        return base.detach()
