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

        # Compute the drift
        # if self.default_diffusion_term:
        #     drift = self.model.drift(base, t, field_history, field_cond, pars_cond)
        # else:
        #     drift = self.model._drift_with_prior_score(
        #         base, t, field_history, field_cond, pars_cond, self.diffusion_term
        #     )
        drift = self.model.drift(base, t, field_history, field_cond, pars_cond)

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
        base = base + likelihood_score * self.diffusion_term(t) ** 2 * dt

        # # Compute the drift
        # if self.default_diffusion_term:
        #     drift = self.model.drift(base, t, field_history, field_cond, pars_cond)
        # else:
        #     drift = self.model._drift_with_prior_score(
        #         base, t, field_history, field_cond, pars_cond, self.diffusion_term
        #     )
        # self.drift_term.append(drift.detach().to("cpu"))

        return base.detach()

    def _pre_step(
        self,
        base: torch.Tensor,
        observations: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Pre-step of the posterior."""

        self.drift_term: List[torch.Tensor] = []

        return base

    def _post_step(
        self,
        base: torch.Tensor,
        observations: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Post-step of the posterior."""

        return base, field_history

        # t = t.cpu()

        # self.counter += 1
        # if self.weights is None:
        #     self.weights = torch.ones(base.shape[0], device=self.device) / base.shape[0]

        # drift_term = torch.cat(self.drift_term, dim=0)

        # pred = base + drift_term * (1.0 - t)

        # # Add noise = integral of the diffusion term from t to 1
        # pred = pred + torch.randn_like(base) * self.integral_variance(t)

        # pred_obs = self.likelihood_model.obs_operator(pred.to(self.device))

        # obs_diff = observations - pred_obs

        # prob = torch.distributions.Normal(
        #     0, self.likelihood_model.original_variance + 1e-3
        # ).log_prob(obs_diff)
        # self.weights = prob.mean(dim=1) * self.weights
        # self.weights = self.weights / self.weights.sum()  # type: ignore[attr-defined]

        # log_likelihood = self.likelihood_model._compute_log_likelihood(
        #     x_obs=self.likelihood_model.obs_operator(pred.to(self.device)),
        #     observations=observations,
        #     variance=self.likelihood_model.original_variance + 1e-3,
        # )
        # log_likelihood = log_likelihood.to('cpu')

        # temp_weights = torch.softmax(log_likelihood, dim=0).to(self.device)
        # self.weights = self.weights * temp_weights
        # self.weights = self.weights / self.weights.sum()

        # if self.counter % 25 == 0:
        #     resample_indices = torch.multinomial(
        #         self.weights, num_samples=base.shape[0], replacement=True
        #     )
        #     resample_indices = resample_indices.to("cpu")

        #     self.weights = 1 / base.shape[0] * torch.ones_like(self.weights)

        #     return base[resample_indices], field_history[resample_indices]
        # else:
        #     return base, field_history
