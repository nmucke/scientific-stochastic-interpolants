import pdb
from abc import abstractmethod
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import tqdm

from scisi.sampling.ode_solvers import euler_step
from scisi.sampling.sde_solvers import euler_maruyama_step

DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_STEPS = 100
DEFAULT_NUM_PHYSICAL_STEPS = 10


class BaseModel(nn.Module):
    """Base model."""

    def __init__(self, mask_path: Optional[str] = None) -> None:
        """Initialize the base model."""
        super(BaseModel, self).__init__()
        self.diffusion_term = None
        if mask_path is not None:
            self.mask = np.load(mask_path)["mask"]
            self.mask = torch.tensor(self.mask).unsqueeze(0).unsqueeze(0)
            self.mask = self.mask.to(dtype=torch.float32)
        else:
            self.mask = torch.tensor(1.0)

    @property
    def device(self) -> str:
        """Get the device of the model."""
        return next(self.parameters()).device  # type: ignore[no-any-return]

    def to(self, device: str) -> "BaseModel":
        """Move the model to the device."""
        super(BaseModel, self).to(device)
        self.mask = self.mask.to(device)
        return self

    @abstractmethod
    def drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Drift of the model."""
        pass

    def _prepare_batch(
        self,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> tuple[Optional[torch.Tensor], ...]:
        """Prepare the batch for the sample method."""

        for key, value in kwargs.items():
            if value is not None:
                # Create repeat pattern based on tensor dimensions
                # For n-dim tensor: (batch_size, 1, 1, ..., 1) with (n-1) ones
                repeat_dims = [batch_size] + [1] * (value.ndim - 1)
                kwargs[key] = value.repeat(*repeat_dims)

        return tuple(kwargs.values())

    def _prepare_time(self, num_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare the time vector for the sample method."""

        t_vec = torch.linspace(0, 1, num_steps+1, device=self.device).unsqueeze(0)
        dt = (t_vec[0,  1] - t_vec[0,  0] ).to(self.device)

        return dt, t_vec

    def _integrate(
        self,
        base: torch.Tensor,
        t_vec: torch.Tensor,
        fixed_input: dict,
        stepper: Callable,
    ) -> torch.Tensor:
        """Integrate the model."""

        for i in range(0, t_vec.shape[1] - 1):
            base = stepper(x=base, t=t_vec[:, i : i + 1], **fixed_input).detach()

        return base  # * fixed_input["mask"]

    def _sample(
        self,
        field_history: torch.Tensor,
        stepper: Callable,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_steps: int = DEFAULT_NUM_STEPS,
        base: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        return_field_history: bool = False,
        diffusion_term: Optional[Callable] = None,
        drift: Optional[Callable] = None,
        with_first_step: bool = False,
    ) -> torch.Tensor:
        """Sample from the model."""
        if (batch_size > 1) and (field_history.shape[0] == 1):
            base, field_history, field_cond, pars_cond = self._prepare_batch(
                batch_size=batch_size,
                base=base,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
            )

        # If base is not provided, sample noise
        base = (
            torch.randn_like(field_history[:, :, :, :, 0], device=self.device)
            if base is None
            else base
        )

        dt, t_vec = self._prepare_time(num_steps=num_steps)

        fixed_input = {
            "field_history": (
                field_history.to(self.device) if field_history is not None else None
            ),
            "field_cond": (
                field_cond.to(self.device) if field_cond is not None else None
            ),
            "pars_cond": pars_cond.to(self.device) if pars_cond is not None else None,
            "dt": dt,
            "mask": self.mask,
        }

        if with_first_step:
            base = self._compute_first_step(
                base=base.to(self.device),
                t=t_vec[:, 0:1],
                stepper=stepper,
                **fixed_input,
            ).detach()
            t_vec = t_vec[:, 1:]

        fixed_input["diffusion_term"] = (
            self.diffusion_term if diffusion_term is None else diffusion_term
        )
        fixed_input["drift_model"] = self.drift if drift is None else drift

        base = self._integrate(
            base=base,
            t_vec=t_vec,
            fixed_input=fixed_input,
            stepper=stepper,
        ).cpu()

        # Add the new base to the field history
        if return_field_history:
            field_history = torch.cat(
                [field_history[:, :, :, :, 1:].cpu(), base.unsqueeze(-1)], dim=-1
            )
            return base, field_history

        return base

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Sample from the model."""
        pass

    # Concrete shared rollout used by subclasses' sample_trajectory; note that
    # @abstractmethod would be inert here anyway (nn.Module, not ABCMeta).
    def _sample_trajectory(
        self,
        base: torch.Tensor,
        field_history: torch.Tensor,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_steps: int = DEFAULT_NUM_STEPS,
        num_physical_steps: int = DEFAULT_NUM_PHYSICAL_STEPS,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        stepper: Callable = euler_maruyama_step,
        diffusion_term: Optional[Callable] = None,
        gaussian_base: Optional[bool] = False,
    ) -> torch.Tensor:
        """Sample a trajectory from the model."""
        if (batch_size > 1) and (field_history.shape[0] == 1):
            base, field_history, field_cond, pars_cond = self._prepare_batch(
                batch_size=batch_size,
                base=base,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
            )

        trajectory = [
            field_history[:, :, :, :, i].cpu() for i in range(field_history.shape[-1])
        ]

        with torch.no_grad():
            pbar = tqdm.tqdm(range(0, num_physical_steps - field_history.shape[-1]))
            for i in pbar:
                base, field_history = self.sample(
                    base=None if gaussian_base else base,
                    field_history=field_history,
                    num_steps=num_steps,
                    batch_size=batch_size,
                    return_field_history=True,
                    stepper=stepper,
                    diffusion_term=diffusion_term,
                    field_cond=(
                        field_cond[:, :, :, :, i] if field_cond is not None else None
                    ),
                    pars_cond=(
                        pars_cond[:, i : i + 1] if pars_cond is not None else None
                    ),
                )
                trajectory.append(base.cpu())

        return torch.stack(trajectory, dim=-1)

    @abstractmethod
    def sample_trajectory(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Sample from the model."""
        pass

    def forward(
        self,
        base: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        pass
