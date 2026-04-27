import pdb
from functools import partial
from re import L
from typing import Callable, List, Optional

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
        self.counter = 0
        self.weights = None
        self.log_likelihood = None
        self.integral_variance = lambda t: 2 / 3 - t.sqrt() + (1 / 3) * (t.sqrt()) ** 3

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

        if self.log_likelihood is None:
            self.log_likelihood = []

        drift = self.model.drift(base, t, field_history, field_cond, pars_cond)

        # Compute the likelihood score
        likelihood_score, log_likelihood = self.likelihood_model.score(
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

        likelihood_score = likelihood_score.detach()

        self.log_likelihood.append(log_likelihood.detach())

        # Euler-Maruyama drift step
        base = base + drift * dt

        # Euler-Maruyama diffusion step
        base = base + self.diffusion_term(t) * torch.randn_like(base) * dt.sqrt()

        # Likelihood score step
        base = base + likelihood_score * dt

        return base.detach()

    def _post_step(
        self,
        base: torch.Tensor,
        observations: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Post-step of the posterior."""

        if not self.resample:
            return base, field_history

        log_likelihood = torch.cat(self.log_likelihood, dim=-1)

        if self.weights is None:
            self.weights = torch.ones(base.shape[0], device=self.device) / base.shape[0]

        self.weights = torch.exp(log_likelihood[0]) * self.weights
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
