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
    """Diffusion posterior."""

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

        base.requires_grad = True
        # Compute the drift
        drift = self.model.drift(base, t, field_history, field_cond, pars_cond).detach()

        score = self.model.score(base, t, field_history, field_cond, pars_cond)
        velocity = self.model._get_velocity_from_score(base, t, score)

        # Compute the likelihood score
        likelihood_score = self.likelihood_model.score(
            observations=observations,
            x=base,
            t=t,
            drift=velocity,
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
            dt=dt,
        ).detach()

        # Euler-Maruyama drift step
        base = base + drift * dt

        # Euler-Maruyama diffusion step
        base = base + self.diffusion_term(t) * torch.randn_like(base) * dt.sqrt()

        # Likelihood score step
        base = (
            base + 0.5 * self.diffusion_term(t) ** 2 * likelihood_score * dt * t.sqrt()
        )

        return base.detach()
