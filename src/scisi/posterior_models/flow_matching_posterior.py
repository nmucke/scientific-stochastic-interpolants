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


# class FlowMatchingPosterior(nn.Module):
#     """Stochastic interpolant posterior."""

#     def __init__(
#         self,
#         model: nn.Module,
#         likelihood_model: nn.Module,
#         **kwargs: Any,
#     ) -> None:
#         """
#         Initialize stochastic interpolant posterior.

#         Args:
#             model: Model.
#             likelihood_model: Likelihood model.
#         """
#         super(FlowMatchingPosterior, self).__init__()
#         self.model = model
#         self.likelihood_model = likelihood_model

#     @property
#     def device(self) -> str:
#         """Get the device of the model."""
#         return next(self.parameters()).device  # type: ignore[no-any-return]

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass."""
#         pass

#     def sample(
#         self,
#         base: torch.Tensor,
#         batch_size: int,
#         num_steps: int,
#         field_history: torch.Tensor,
#         observations: torch.Tensor,
#         field_cond: torch.Tensor | None = None,
#         pars_cond: torch.Tensor | None = None,
#         stepper: Callable = euler_maruyama_step,
#         return_field_history: bool = False,
#     ) -> torch.Tensor:
#         """Sample from the posterior."""

#         if (batch_size > 1) and (field_history.shape[0] == 1):
#             base, field_history, field_cond, pars_cond = self.model._prepare_batch(
#                 batch_size=batch_size,
#                 base=base,
#                 field_history=field_history,
#                 field_cond=field_cond,
#                 pars_cond=pars_cond,
#             )

#         dt = torch.tensor(1 / num_steps, device=self.device)
#         t_vec = torch.linspace(0, 1, num_steps, device=self.device).unsqueeze(0)

#         fixed_input = {
#             "field_history": field_history,
#             "field_cond": field_cond,
#             "pars_cond": pars_cond,
#             "dt": dt,
#         }

#         base = torch.randn_like(field_history[..., 0]) if base is None else base

#         for i in range(0, num_steps - 1):
#             t = t_vec[:, i : i + 1]
#             base.requires_grad = True

#             # Compute the drift
#             drift = self.model.drift_model(
#                 base, t, field_history, field_cond, pars_cond
#             )

#             # Compute the guidance score
#             likelihood_score = self.likelihood_model.score(
#                 observations=observations,
#                 x=base,
#                 t=t,
#                 drift=drift,
#                 **fixed_input,
#             ).detach()

#             # Euler drift step
#             base = base + drift * dt

#             # Guidance score step
#             base = base + likelihood_score * dt

#             base = base.detach()

#         if return_field_history:
#             field_history = torch.cat(
#                 [field_history[:, :, :, :, 1:], base.unsqueeze(-1)], dim=-1
#             )
#             return base, field_history

#         return base

#     def sample_trajectory(
#         self,
#         base: torch.Tensor,
#         field_history: torch.Tensor,
#         observations: torch.Tensor,
#         batch_size: int = 1,
#         num_steps: int = 100,
#         num_physical_steps: int = 10,
#         field_cond: torch.Tensor | None = None,
#         pars_cond: torch.Tensor | None = None,
#         stepper: Callable = euler_maruyama_step,
#     ) -> torch.Tensor:
#         """Sample a trajectory from the Follmer stochastic interpolant with posterior drift."""

#         if (batch_size > 1) and (field_history.shape[0] == 1):
#             base, field_history, field_cond, pars_cond = self.model._prepare_batch(
#                 batch_size=batch_size,
#                 base=base,
#                 field_history=field_history,
#                 field_cond=field_cond,
#                 pars_cond=pars_cond,
#             )

#         trajectory = [
#             field_history[..., i].cpu() for i in range(field_history.shape[-1])
#         ]

#         fixed_input = {
#             "num_steps": num_steps,
#             "batch_size": batch_size,
#             "return_field_history": True,
#             "stepper": stepper,
#         }

#         cond_input = lambda i: {
#             "field_cond": field_cond[:, :, :, :, i] if field_cond is not None else None,
#             "pars_cond": pars_cond[:, i : i + 1] if pars_cond is not None else None,
#             "observations": observations[:, :, i],
#         }
#         pbar = tqdm.tqdm(range(0, num_physical_steps - field_history.shape[-1]))

#         for i in pbar:
#             base, field_history = self.sample(
#                 base=None,
#                 field_history=field_history,
#                 **cond_input(i),
#                 **fixed_input,  # type: ignore[arg-type]
#             )

#             trajectory.append(base.cpu())

#         return torch.stack(trajectory, dim=-1)
