import pdb
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import tqdm

from scisi.sampling.sde_solvers import euler_maruyama_step, heun_step

MIN_TIME = 1e-4


class StochasticInterpolantPosterior(nn.Module):
    """Stochastic interpolant posterior."""

    def __init__(
        self,
        model: nn.Module,
        likelihood_model: nn.Module,
        diffusion_term: Callable,
    ) -> None:
        """
        Initialize stochastic interpolant posterior.

        Args:
            model: Model.
            likelihood_model: Likelihood model.
        """
        super(StochasticInterpolantPosterior, self).__init__()
        self.model = model
        self.likelihood_model = likelihood_model
        self.diffusion_term = diffusion_term

    def _get_device(self) -> str:
        """Get the device of the model."""
        return next(self.parameters()).device  # type: ignore[no-any-return]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pass

    def sample(
        self,
        base: torch.Tensor,
        batch_size: int,
        num_steps: int,
        field_history: torch.Tensor,
        observations: torch.Tensor,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
        sde_stepper: Callable = euler_maruyama_step,
        return_field_history: bool = False,
    ) -> torch.Tensor:
        """Sample from the posterior."""

        base, field_history, field_cond, pars_cond = self.model._prepare_batch(
            base, field_history, field_cond, pars_cond, batch_size
        )

        dt = torch.tensor(1 / num_steps, device=self._get_device())
        t_vec = torch.linspace(0, 1, num_steps, device=self._get_device()).unsqueeze(0)

        fixed_input = {
            "field_history": field_history,
            "field_cond": field_cond,
            "pars_cond": pars_cond,
            "dt": dt,
        }

        base = self.model._compute_first_step(
            base=base,
            t=t_vec[:, 0],
            sde_stepper=sde_stepper,
            **fixed_input,
        ).detach()

        fixed_input["drift_model"] = partial(
            self._posterior_drift,
            observations=observations,
        )
        fixed_input["diffusion_term"] = self.diffusion_term

        for i in range(1, num_steps):
            t = t_vec[:, i : i + 1]
            base.requires_grad = True
            base = sde_stepper(x=base, t=t, **fixed_input).detach()

        # Add the new base to the field history
        if return_field_history:
            field_history = torch.cat(
                [field_history[:, :, :, :, 1:], base.unsqueeze(-1)], dim=-1
            )
            return base, field_history

        return base

    def _likelihood_score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
        observations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Likelihood score."""

        # Compute the interpolant of the observation
        base_obs = self.likelihood_model.obs_operator(field_history[:, :, :, :, -1])
        interpolant_obs = self.model.interpolation.forward(
            base_obs, observations, t, torch.zeros_like(base_obs)
        )

        # Compute the scale of the interpolant of the observation
        interpolant_variance = (
            self.model.interpolation.beta(t) ** 2
            * self.likelihood_model.original_variance
        )
        interpolant_variance = (
            interpolant_variance + self.model.interpolation.gamma(t) ** 2 * t
        )

        return self.likelihood_model.score(
            x=x,
            observations=interpolant_obs,
            variance=interpolant_variance,
        )

    def _posterior_drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
        observations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Posterior drift."""

        prior_drift = self.model._drift_with_prior_score(
            x, t, field_history, field_cond, pars_cond, self.diffusion_term
        )

        likelihood_score = self._likelihood_score(
            x, t, field_history, field_cond, pars_cond, observations
        )

        return prior_drift + 0.5 * self.diffusion_term(t) ** 2 * likelihood_score

    def sample_trajectory(
        self,
        base: torch.Tensor,
        field_history: torch.Tensor,
        observations: torch.Tensor,
        batch_size: int = 1,
        num_steps: int = 100,
        num_physical_steps: int = 10,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
        sde_stepper: Callable = euler_maruyama_step,
    ) -> torch.Tensor:
        """Sample a trajectory from the Follmer stochastic interpolant with posterior drift."""

        trajectory = [
            field_history[:, :, :, :, i].cpu() for i in range(field_history.shape[-1])
        ]

        fixed_input = {
            "num_steps": num_steps,
            "batch_size": batch_size,
            "return_field_history": True,
            "sde_stepper": sde_stepper,
        }

        cond_input = lambda i: {
            "field_cond": field_cond[:, :, :, :, i] if field_cond is not None else None,
            "pars_cond": pars_cond[:, i : i + 1] if pars_cond is not None else None,
            "observations": observations[:, :, i],
        }
        pbar = tqdm.tqdm(range(0, num_physical_steps - field_history.shape[-1]))

        for i in pbar:
            base, field_history = self.sample(
                base=base,
                field_history=field_history,
                **cond_input(i),
                **fixed_input,  # type: ignore[arg-type]
            )

            trajectory.append(base.cpu())

        return torch.stack(trajectory, dim=-1)
