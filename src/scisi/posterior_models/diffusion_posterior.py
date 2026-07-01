import logging
import pdb
from functools import partial
from re import L
from typing import Callable, Optional

import torch
import torch.nn as nn
import tqdm

from scisi.posterior_models.base_posterior import BasePosterior

logger = logging.getLogger(__name__)


class DiffusionPosterior(BasePosterior):
    """Diffusion (VP/VE) posterior — NON-PAPER baseline.

    This is a guided reverse-diffusion sampler kept as a DPS-style baseline. It
    is **not** a member of the paper's unified observation-interpolant family:
    its likelihood term is scaled either by an ad-hoc ``1/2 g_tau**2 sqrt(t)``
    factor (default DPS branch) or, for SDA (``guidance_weight = "g_squared"``),
    by the classifier-guidance / h-transform weight ``g_tau**2`` that puts the
    likelihood score on the same ``-g^2`` footing as the prior score. Neither is
    the paper's own guidance weight ``w_tau = a_tau + 1/2 g_tau**2`` times the
    multiplicative gain ``G_tau``; this class is intentionally excluded from the
    unified-family claims. Use ``StochasticInterpolantPosterior`` /
    ``FlowMatchingPosterior`` for the paper samplers.
    """

    def __init__(
        self,
        model: nn.Module,
        likelihood_model: nn.Module,
        diffusion_term: Optional[Callable] = None,
        resample: bool = True,
    ) -> None:
        """
        Initialize diffusion posterior.

        Args:
            model: Model.
            likelihood_model: Likelihood model.
        """
        super(DiffusionPosterior, self).__init__(
            model=model,
            likelihood_model=likelihood_model,
            diffusion_term=diffusion_term,
            gaussian_base=True,
        )

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
        """One step of the diffusion posterior."""

        # Fresh grad-enabled leaf (not an in-place mutation of the caller's tensor,
        # which can corrupt the autoregressive feedback / fail on a non-leaf base).
        base = base.detach().requires_grad_(True)
        # Compute the drift
        drift = self.model.drift(base, t, field_history, field_cond, pars_cond).detach()

        score = self.model.score(base, t, field_history, field_cond, pars_cond)
        velocity = self.model._get_velocity_from_score(base, t, score)

        # Compute the likelihood score
        # The DPS likelihood returns (guidance_score, log_likelihood); unpack it.
        likelihood_out = self.likelihood_model.score(
            observations=observations,
            x=base,
            t=t,
            drift=velocity,
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
            dt=dt,
        )
        likelihood_score = (
            likelihood_out[0] if isinstance(likelihood_out, tuple) else likelihood_out
        ).detach()

        # Euler-Maruyama drift step
        base = base + drift * dt

        # Euler-Maruyama diffusion step
        base = base + self.diffusion_term(t) * torch.randn_like(base) * dt.sqrt()

        # Likelihood score step. Default (legacy DPS-style) weight is
        # ``0.5 g^2 sqrt(t)``. A likelihood may instead request the SDA / classifier-
        # guidance injection weight ``g^2`` by setting ``guidance_weight =
        # "g_squared"`` (used by SDALikelihood): this is the exact h-transform
        # footing, putting the likelihood score on the same ``-g^2`` footing as the
        # prior score (``s = s_prior + s_lik``), matching Rozet & Louppe.
        g_t = self.diffusion_term(t)
        if getattr(self.likelihood_model, "guidance_weight", None) == "g_squared":
            base = base + g_t**2 * likelihood_score * dt
        else:
            base = base + 0.5 * g_t**2 * likelihood_score * dt * t.sqrt()

        return base.detach()
