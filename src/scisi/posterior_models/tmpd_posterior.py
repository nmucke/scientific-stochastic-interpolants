"""TMPD guided reverse-SDE posterior -- NON-PAPER baseline.

Runs the reverse SDE of Boys et al., "Tweedie Moment Projected Diffusions for
Inverse Problems" (arXiv 2310.06721) on the FM prior read as a diffusion model,
with the TMPD guidance of :class:`scisi.likelihood_models.tmpd.TMPDLikelihood`.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn

from scisi.posterior_models.base_posterior import BasePosterior

logger = logging.getLogger(__name__)

GUIDANCE_WEIGHTS = ("score_substitution", "g_squared")


class TMPDPosterior(BasePosterior):
    """TMPD posterior (guided reverse SDE, Euler--Maruyama) -- NON-PAPER baseline.

    Like :class:`DiffusionPosterior` this is a guided reverse-diffusion baseline
    and is **not** a member of the paper's unified observation-interpolant
    family; it is excluded from the unified-family claims.

    TMPD replaces the prior score ``s`` by ``s + f^y`` in the reverse SDE (no
    guidance scale, no thresholding). On our affine path the reverse SDE with
    diffusion ``g_tau`` (pseudo-time ``tau: 0 -> 1``, noise -> data) is

        dx = [v(x, tau) + 1/2 g_tau^2 s(x, tau)] dtau + g_tau dW,

    and the FM velocity itself carries the score,
    ``v = (beta_dot/beta) x + a_tau s`` (anchor 0; ``a_tau`` =
    ``interpolation.velocity_score_coeff``). Substituting ``s -> s + f^y`` in
    BOTH places gives the guided drift

        b_post = v + 1/2 g^2 s + (a_tau + 1/2 g_tau^2) f^y.

    This is the faithful TMPD rule. In TMPD's own VP setting the reverse SDE
    uses the NATIVE noising diffusion, for which ``a_tau = 1/2 g_tau^2`` and the
    weight reduces to the familiar ``g_tau^2`` (``-g^2 (s + f^y)`` in the
    DM-time reverse drift). Our drivers, however, run the DM baselines with the
    endpoint-vanishing ``g_tau^2 = g0^2 alpha beta`` (``build_dm_model``), for
    which ``a_tau != 1/2 g^2`` (rectified flow: ``a = (1-tau)/tau`` vs
    ``1/2 g^2 = 1/2 tau (1-tau)``); there SDA's ``g^2`` footing
    (``DiffusionPosterior`` with ``guidance_weight='g_squared'``) is NOT score
    substitution and does not target the guided marginals. We therefore use
    ``w_tau = a_tau + 1/2 g_tau^2`` (default ``guidance_weight=
    "score_substitution"``), which keeps TMPD's defining property -- exact
    posterior sampling for Gaussian data (TMPD Prop. 3) -- for ANY diffusion
    schedule: with exact ``f^y = grad log p(y | x_tau)`` the guided SDE is the
    member of the ``v + 1/2 g^2 s`` family for the posterior path. The SDA-style
    ``g_tau^2`` weight is available as ``guidance_weight="g_squared"`` for
    ablations only.

    Diffusion term: same mechanism as the DM baselines (``DiffusionPosterior``):
    ``diffusion_term=None`` resolves to ``model.diffusion_term`` (the
    endpoint-vanishing ``g0 sqrt(alpha beta)`` of ``build_dm_model``). Unlike
    ``DiffusionPosterior``, the prior drift ``v + 1/2 g^2 s`` is assembled here
    with ``self.diffusion_term`` (so an explicit schedule is used consistently in
    the drift, the guidance weight and the noise), and ``v``, ``s`` and the
    guidance come from ONE network forward per step (the likelihood's
    grad-enabled forward), plus the VJPs TMPD requires.

    Initialisation: Gaussian source ``N(0, I)`` (``gaussian_base=True``); the
    model's own ``_compute_first_step`` (if any) runs the ``tau = 0`` step
    unguided, as for SDA (TMPD's guidance is degenerate at ``beta = 0``).
    """

    def __init__(
        self,
        model: nn.Module,
        likelihood_model: nn.Module,
        diffusion_term: Optional[Callable] = None,
        guidance_weight: str = "score_substitution",
        drift_time_shift: float = 0.0,
    ) -> None:
        """
        Args:
            model: Diffusion prior (``DenoiseDiffusionModel``, e.g.
                ``from_flow_matching``) or ``FlowMatchingModel`` (then pass an
                explicit ``diffusion_term``).
            likelihood_model: :class:`TMPDLikelihood` (needs ``guided_terms``).
            diffusion_term: Reverse-SDE diffusion ``t -> g_tau``; ``None`` uses
                ``model.diffusion_term`` (as ``DiffusionPosterior``).
            guidance_weight: ``"score_substitution"`` (default, faithful TMPD:
                ``a_tau + 1/2 g^2``) or ``"g_squared"`` (SDA footing, ablation).
            drift_time_shift: See :class:`BasePosterior`.
        """
        super(TMPDPosterior, self).__init__(
            model=model,
            likelihood_model=likelihood_model,
            diffusion_term=diffusion_term,
            gaussian_base=True,
            drift_time_shift=drift_time_shift,
        )
        if guidance_weight not in GUIDANCE_WEIGHTS:
            raise ValueError(
                f"guidance_weight must be one of {GUIDANCE_WEIGHTS}, got "
                f"{guidance_weight!r}."
            )
        if not hasattr(likelihood_model, "guided_terms"):
            raise TypeError("TMPDPosterior needs a TMPDLikelihood (guided_terms).")
        self.guidance_weight = guidance_weight

    def _guidance_weight(self, t: torch.Tensor) -> torch.Tensor:
        """``a_tau + 1/2 g^2`` (score substitution) or ``g^2`` (SDA footing)."""
        g2 = self.diffusion_term(t) ** 2
        if self.guidance_weight == "g_squared":
            return g2
        return self.model.interpolation.velocity_score_coeff(t) + 0.5 * g2

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
        """One Euler--Maruyama step of the TMPD-guided reverse SDE."""
        base = base.detach()
        terms = self.likelihood_model.guided_terms(
            observations=observations,
            x=base,
            t=t,
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
        )
        t_grid = t.reshape(t.shape[0], *([1] * (base.dim() - 1)))
        g = self.diffusion_term(t_grid)
        prior_drift = terms["velocity"] + 0.5 * g**2 * terms["score"]
        # Zero below MIN_TIME (handled in the likelihood).
        posterior_drift = prior_drift + self._guidance_weight(t_grid) * terms["guidance"]
        noise = g * torch.randn_like(base) * dt.sqrt()
        return (base + posterior_drift * dt + noise).detach()


__all__ = ["TMPDPosterior"]
