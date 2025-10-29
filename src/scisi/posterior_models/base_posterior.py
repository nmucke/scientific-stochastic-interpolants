import logging
import pdb
from abc import abstractmethod
from functools import partial
from re import L
from typing import Callable, Optional

import torch
import torch.nn as nn
import tqdm

from scisi.sampling.sde_solvers import euler_maruyama_step, heun_step

logger = logging.getLogger(__name__)

MIN_TIME = 1e-4
MAX_BATCH_SIZE = 4


class BasePosterior(nn.Module):
    """Base posterior."""

    def __init__(
        self,
        model: nn.Module,
        likelihood_model: nn.Module,
        diffusion_term: Optional[Callable] = None,
        gaussian_base: bool = False,
    ) -> None:
        """
        Initialize base posterior.

        Args:
            model: Model.
            likelihood_model: Likelihood model.
            diffusion_term: Diffusion term.
            resample: Whether to resample the base.
            gaussian_base: Whether to sample the base from a Gaussian distribution.
        """
        super(BasePosterior, self).__init__()

        self.model = model
        self.likelihood_model = likelihood_model

        if diffusion_term is None:
            self.default_diffusion_term = True
            try:
                self.diffusion_term = self.model.interpolation.gamma
            except:
                self.diffusion_term = self.model.diffusion_term
        else:
            self.diffusion_term = diffusion_term
            self.default_diffusion_term = False

        self.gaussian_base = gaussian_base

    @property
    def device(self) -> str:
        """Get the device of the model."""
        return next(self.parameters()).device  # type: ignore[no-any-return]

    @abstractmethod
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
        pass

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

        ensemble_size = field_history.shape[0]
        observations = observations.to(self.device)

        # Prepare the batch
        if (batch_size > 1) and (field_history.shape[0] == 1):
            base, field_history, field_cond, pars_cond = self.model._prepare_batch(
                batch_size=batch_size,
                base=base,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
            )

        # Prepare the time
        dt = torch.tensor(1 / num_steps, device=self.device)
        t_vec = torch.linspace(0, 1, num_steps, device=self.device).unsqueeze(0)

        # Prepare the fixed input and make sure the tensors are on the correct device
        fixed_input = lambda batch_ids: {
            "field_history": (
                field_history[batch_ids].to(self.device)
                if field_history is not None
                else None
            ),
            "field_cond": (
                field_cond[batch_ids].to(self.device)
                if field_cond is not None
                else None
            ),
            "pars_cond": (
                pars_cond[batch_ids].to(self.device) if pars_cond is not None else None
            ),
            "dt": dt,
        }

        # Sample init state from Gaussian distribution
        if self.gaussian_base:
            base = torch.randn_like(field_history[..., 0]) if base is None else base

        # Sample first step
        start_time = 0
        if hasattr(self.model, "_compute_first_step"):
            with torch.no_grad():
                for batch_idx in range(0, ensemble_size, batch_size):
                    batch_ids = torch.arange(
                        batch_idx, min(batch_idx + batch_size, ensemble_size)
                    )

                    base[batch_ids] = self.model._compute_first_step(
                        base=base[batch_ids].to(self.device),
                        t=t_vec[:, 0],
                        stepper=stepper,
                        **fixed_input(batch_ids),
                    ).cpu()

                start_time = 1

        # Sample remaining steps
        for i in range(start_time, num_steps - 1):
            t = t_vec[:, i : i + 1]
            for batch_idx in range(0, ensemble_size, batch_size):

                # Prepare the batch
                batch_ids = torch.arange(
                    batch_idx, min(batch_idx + batch_size, ensemble_size)
                )

                # Sample one step
                base[batch_ids] = self._one_step(
                    base=base[batch_ids].to(self.device),
                    observations=observations,
                    t=t,
                    **fixed_input(batch_ids),
                ).cpu()

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
        ensemble_size: int = 1,
        batch_size: int = MAX_BATCH_SIZE,
        num_steps: int = 100,
        num_physical_steps: int = 10,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
        stepper: Callable = euler_maruyama_step,
    ) -> torch.Tensor:
        """Sample a trajectory from the diffusion model with posterior drift."""

        len_field_history = field_history.shape[-1]

        if (ensemble_size > 1) and (field_history.shape[0] == 1):
            base, field_history, field_cond, pars_cond = self.model._prepare_batch(
                batch_size=ensemble_size,
                base=base,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
            )

        trajectory = [
            field_history[..., i].cpu() for i in range(field_history.shape[-1])
        ]

        fixed_input = {
            "num_steps": num_steps,
            "batch_size": batch_size,
            "return_field_history": True,
            "stepper": stepper,
        }

        cond_input = lambda t_idx: {
            "field_cond": (
                field_cond[:, :, :, :, t_idx] if field_cond is not None else None
            ),
            "pars_cond": (
                pars_cond[:, t_idx : t_idx + 1] if pars_cond is not None else None
            ),
            "observations": observations[:, :, t_idx],
        }
        pbar = tqdm.tqdm(range(0, num_physical_steps - len_field_history))

        for t_idx in pbar:
            base, field_history = self.sample(
                base=None if self.gaussian_base else base,
                field_history=field_history,
                **cond_input(t_idx),
                **fixed_input,  # type: ignore[arg-type]
            )

            trajectory.append(base.cpu())

        return torch.stack(trajectory, dim=-1)
