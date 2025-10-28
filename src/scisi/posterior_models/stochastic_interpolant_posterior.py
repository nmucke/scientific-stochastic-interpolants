import pdb
from functools import partial
from re import L
from typing import Callable, Optional

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
        diffusion_term: Optional[Callable] = None,
        resample: bool = True,
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
        self.default_diffusion_term = False
        if diffusion_term is None:
            self.default_diffusion_term = True
            self.diffusion_term = self.model.interpolation.gamma
        self.resample = resample

    @property
    def device(self) -> str:
        """Get the device of the model."""
        return next(self.parameters()).device  # type: ignore[no-any-return]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pass

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

    def sample(
        self,
        base: torch.Tensor,
        batch_size: int,
        num_steps: int,
        field_history: torch.Tensor,
        observations: torch.Tensor,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
        stepper: Callable = euler_maruyama_step,
        return_field_history: bool = False,
    ) -> torch.Tensor:
        """Sample from the posterior."""

        if (batch_size > 1) and (base.shape[0] == 1):
            base, field_history, field_cond, pars_cond = self.model._prepare_batch(
                batch_size=batch_size,
                base=base,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
            )

        dt = torch.tensor(1 / num_steps, device=self.device)
        t_vec = torch.linspace(0, 1, num_steps, device=self.device).unsqueeze(0)

        fixed_input = {
            "field_history": field_history,
            "field_cond": field_cond,
            "pars_cond": pars_cond,
            "dt": dt,
        }

        base = self.model._compute_first_step(
            base=base,
            t=t_vec[:, 0],
            stepper=stepper,
            **fixed_input,
        ).detach()

        # with torch.no_grad():
        for i in range(1, num_steps - 1):
            t = t_vec[:, i : i + 1]
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
                **fixed_input,
            ).detach()
            # Euler-Maruyama drift step
            base = base + drift * dt
            # Euler-Maruyama diffusion step
            base = base + self.diffusion_term(t) * torch.randn_like(base) * torch.sqrt(  # type: ignore[misc]
                dt
            )
            # Likelihood score step
            base = base + likelihood_score

            base = base.detach()

            # if self.resample:
            #     base, field_history = self._resample(
            #         base, observations, t, field_history, field_cond, pars_cond
            #     )

        # Add the new base to the field history
        if return_field_history:
            field_history = torch.cat(
                [field_history[:, :, :, :, 1:], base.unsqueeze(-1)], dim=-1
            )
            return base, field_history

        return base

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
        stepper: Callable = euler_maruyama_step,
    ) -> torch.Tensor:
        """Sample a trajectory from the Follmer stochastic interpolant with posterior drift."""

        if (batch_size > 1) and (base.shape[0] == 1):
            base, field_history, field_cond, pars_cond = self.model._prepare_batch(
                batch_size=batch_size,
                base=base,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
            )

        trajectory = [
            field_history[:, :, :, :, i].cpu() for i in range(field_history.shape[-1])
        ]

        fixed_input = {
            "num_steps": num_steps,
            "batch_size": batch_size,
            "return_field_history": True,
            "stepper": stepper,
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


class StochasticInterpolantParticleFlowPosterior(nn.Module):
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
        super(StochasticInterpolantParticleFlowPosterior, self).__init__()
        self.model = model
        self.likelihood_model = likelihood_model
        self.diffusion_term = diffusion_term
        if self.diffusion_term is None:
            self.diffusion_term = self.model.interpolation.gamma

    @property
    def device(self) -> str:
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
        stepper: Callable = euler_maruyama_step,
        return_field_history: bool = False,
    ) -> torch.Tensor:
        """Sample from the posterior."""

        if (batch_size > 1) and (base.shape[0] == 1):
            base, field_history, field_cond, pars_cond = self.model._prepare_batch(
                batch_size=batch_size,
                base=base,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
            )

        dt = torch.tensor(1 / num_steps, device=self.device)
        t_vec = torch.linspace(0, 1, num_steps, device=self.device).unsqueeze(0)

        prior_prediction = self.model.sample(
            base=base,
            field_history=field_history,
            batch_size=batch_size,
            num_steps=25,
            field_cond=field_cond,
            pars_cond=pars_cond,
            stepper=stepper,
            diffusion_term=self.diffusion_term,
        )

        fixed_input = {
            "field_history": field_history,
            "field_cond": field_cond,
            "pars_cond": pars_cond,
            "dt": dt,
        }

        base = prior_prediction.clone().detach()

        base = self.model._compute_first_step(
            base=base,
            t=t_vec[:, 0],
            stepper=stepper,
            **fixed_input,
        ).detach()

        # with torch.no_grad():
        for i in range(1, num_steps - 1):
            t = t_vec[:, i : i + 1]
            base.requires_grad = True

            sigma = self.diffusion_term(t)  # type: ignore[misc]

            # Compute the drift
            # prior_drift = self.model.drift(
            #     base, t_vec[:, -1:], field_history, field_cond, pars_cond
            # )
            # prior_score = self.model._prior_score(
            #     base, field_history[:, :, :, :, -1], prior_drift, t=t_vec[:, -1:]
            # )

            intermediate_prior_drift = self.model.drift(
                base, t, field_history, field_cond, pars_cond
            )
            intermediate_prior_score = self.model._prior_score(
                base, field_history[:, :, :, :, -1], intermediate_prior_drift, t=t
            )

            # Compute the likelihood score
            likelihood_score = self.likelihood_model._compute_likelihood_score(
                x=base,
                observations=observations,
                variance=self.likelihood_model.original_variance,
            ).detach()

            base = base + 0.5 * sigma**2 * intermediate_prior_score * dt
            base = base + likelihood_score * dt
            base = base + sigma * torch.randn_like(base) * torch.sqrt(dt)

            base = base.detach()

        # Add the new base to the field history
        if return_field_history:
            field_history = torch.cat(
                [field_history[:, :, :, :, 1:], base.unsqueeze(-1)], dim=-1
            )
            return base, field_history

        return base

    def _posterior_drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
        observations: torch.Tensor | None = None,
        dt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Posterior drift."""

        prior_drift = self.model.drift_model(x, t, field_history, field_cond, pars_cond)

        return prior_drift

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
        stepper: Callable = euler_maruyama_step,
    ) -> torch.Tensor:
        """Sample a trajectory from the Follmer stochastic interpolant with posterior drift."""

        if (batch_size > 1) and (base.shape[0] == 1):
            base, field_history, field_cond, pars_cond = self.model._prepare_batch(
                batch_size=batch_size,
                base=base,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
            )

        trajectory = [
            field_history[:, :, :, :, i].cpu() for i in range(field_history.shape[-1])
        ]

        fixed_input = {
            "num_steps": num_steps,
            "batch_size": batch_size,
            "return_field_history": True,
            "stepper": stepper,
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
