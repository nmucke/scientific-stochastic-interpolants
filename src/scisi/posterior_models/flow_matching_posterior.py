import pdb
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import tqdm

from scisi.posterior_models.base_posterior import BasePosterior
from scisi.sampling.sde_solvers import euler_maruyama_step, heun_step

MIN_TIME = 1e-4


class FlowMatchingPosterior(BasePosterior):
    """Stochastic interpolant posterior."""

    def __init__(
        self,
        model: nn.Module,
        likelihood_model: nn.Module,
        diffusion_term: Optional[Callable] = None,
    ) -> None:
        """
        Initialize stochastic interpolant posterior.

        Args:
            model: Model.
            likelihood_model: Likelihood model.
        """
        super(FlowMatchingPosterior, self).__init__(
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
        """One step of the posterior."""

        base.requires_grad = True

        # Compute the drift
        drift = self.model.drift_model(base, t, field_history, field_cond, pars_cond)

        # Compute the guidance score
        likelihood_score = self.likelihood_model.score(
            observations=observations,
            x=base,
            t=t,
            drift=drift,
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
            dt=dt,
        ).detach()

        # Euler drift step
        base = base + drift * dt

        # Guidance score step
        base = base + likelihood_score * dt

        return base.detach()
