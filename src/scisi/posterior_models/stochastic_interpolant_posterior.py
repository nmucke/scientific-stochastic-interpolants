import pdb
from functools import partial
from re import L
from typing import Callable, Optional

import torch
import torch.nn as nn
import tqdm

from scisi.posterior_models.base_posterior import BasePosterior
from scisi.sampling.sde_solvers import euler_maruyama_step, heun_step

MIN_TIME = 1e-4


class StochasticInterpolantPosterior(BasePosterior):
    """Stochastic interpolant posterior."""

    def __init__(
        self,
        model: nn.Module,
        likelihood_model: nn.Module,
        diffusion_term: Optional[Callable] = None,
        resample: bool = True,
    ) -> None:
        """
        Initialize stochastic interpolant posterior.

        Args:
            model: Model.
            likelihood_model: Likelihood model.
        """
        super(StochasticInterpolantPosterior, self).__init__(
            model=model,
            likelihood_model=likelihood_model,
            diffusion_term=diffusion_term,
            gaussian_base=False,
        )

        self.resample = resample

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
        if self.default_diffusion_term:
            drift = self.model.drift(base, t, field_history, field_cond, pars_cond)
        else:
            drift = self.model._drift_with_prior_score(
                base, t, field_history, field_cond, pars_cond, self.diffusion_term
            )
        # Compute the likelihood score
        likelihood_score = self.likelihood_model.score(
            observations=observations,
            x=base,
            t=t,
            drift=drift,
            diffusion_term=self.diffusion_term,
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
        base = base + likelihood_score

        # if self.resample:
        #     base, field_history = self._resample(
        #         base, observations, t, field_history, field_cond, pars_cond
        #     )
        return base.detach()

    def _resample(
        self,
        base: torch.Tensor,
        observations: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: torch.Tensor,
        pars_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Resample particles."""
        weights = self.likelihood_model.likelihood_weights(
            observations=observations,
            x=base,
            t=t,
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
        )

        if not torch.any(torch.isnan(weights)):
            # Resample indices using multinomial distribution based on weights
            indices = torch.multinomial(
                weights, num_samples=base.shape[0], replacement=True
            )

            # Reorder base according to sampled indices
            base = base[indices]
            field_history = field_history[indices]

        return base, field_history
