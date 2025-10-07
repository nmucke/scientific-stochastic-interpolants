import pdb
from functools import partial
from re import L
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
        sde_stepper: Callable = euler_maruyama_step,
        return_field_history: bool = False,
    ) -> torch.Tensor:
        """Sample from the posterior."""

        if (batch_size > 1) and (base.shape[0] == 1):
            base, field_history, field_cond, pars_cond = self.model._prepare_batch(
                base, field_history, field_cond, pars_cond, batch_size
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
            sde_stepper=sde_stepper,
            **fixed_input,
        ).detach()

        # with torch.no_grad():
        for i in range(1, num_steps):
            t = t_vec[:, i : i + 1]
            base.requires_grad = True

            # Compute the drift
            drift = self.model.drift_model(
                base, t, field_history, field_cond, pars_cond
            )
            # Compute the likelihood score
            likelihood_score = self.likelihood_model.score(
                observations=observations,
                x=base,
                t=t,
                drift=drift,
                **fixed_input,
            ).detach()
            # Euler-Maruyama drift step
            base = base + drift * dt
            # Euler-Maruyama diffusion step
            base = base + self.diffusion_term(t) * torch.randn_like(base) * torch.sqrt(
                dt
            )
            # Likelihood score step
            base = base + likelihood_score

            base = base.detach()
            # base = sde_stepper(x=base, t=t, **fixed_input).detach() + likelihood_score

        # t = t_vec[:, -1 : -1]
        # base = base + self.model.drift_model(base, t, field_history, field_cond, pars_cond) * dt
        # base = base + self.diffusion_term(t) * torch.randn_like(base) * torch.sqrt(dt)

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

        # prior_drift = self.model._drift_with_prior_score(
        #     x, t, field_history, field_cond, pars_cond, self.diffusion_term
        # )
        prior_drift = self.model.drift_model(x, t, field_history, field_cond, pars_cond)

        # likelihood_score = self.likelihood_model.score(
        #     observations=observations,
        #     x=x,
        #     t=t,
        #     field_history=field_history,
        #     field_cond=field_cond,
        #     pars_cond=pars_cond,
        #     dt=dt,
        # )
        return prior_drift  # + likelihood_score

        # eps = 1e-4
        # lam_max = 1.0
        # lam = lambda t: torch.sqrt(self.model.interpolation.beta(t)) / (self.model.interpolation.gamma(t)**2 + eps) # lam_max * (1 - self.model.interpolation.beta(t)**2 + eps) / (1 + eps) + self.model.interpolation.beta(t)

        # import matplotlib.pyplot as plt
        # tt = torch.linspace(0, 1, 100, device=self.device)
        # plt.plot(tt.cpu(), lam(tt).cpu()*self.model.interpolation.gamma(tt).cpu()**2)
        # plt.show()
        # return prior_drift + 1.0 * self.diffusion_term(t)**2 * likelihood_score

        # return prior_drift + self.model.interpolation.gamma(t)**2 * likelihood_score

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

        if (batch_size > 1) and (base.shape[0] == 1):
            base, field_history, field_cond, pars_cond = self.model._prepare_batch(
                base, field_history, field_cond, pars_cond, batch_size
            )

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
