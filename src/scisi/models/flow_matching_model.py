import pdb
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import tqdm

from scisi.models.interpolations import _expand_t
from scisi.sampling.ode_solvers import euler_step

MIN_TIME = 1e-4
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_STEPS = 100
DEFAULT_NUM_PHYSICAL_STEPS = 10


class FlowMatchingModel(nn.Module):
    """Flow Matching model."""

    def __init__(
        self,
        interpolation: nn.Module,
        drift_model: nn.Module,
    ) -> None:
        """Initialize Flow Matching model."""
        super(FlowMatchingModel, self).__init__()

        self.interpolation = interpolation
        self.drift_model = drift_model

    def _get_device(self) -> str:
        """Get the device of the model."""
        return next(self.parameters()).device  # type: ignore[no-any-return]

    def drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the drift of the Flow Matching model.

        Args:
            x (torch.Tensor): Input tensor [B, C, H, W].
            t (torch.Tensor): Time tensor [B, 1].
            field_history (torch.Tensor): Field history tensor [B, C, H, W, L]. Can be None.
            field_cond (torch.Tensor): Field conditional tensor [B, C_field_cond, H, W]. Can be None.
            pars_cond (torch.Tensor): pars conditional tensor [B, D_pars_cond]. Can be None.
        """

        return self.drift_model(x, t, field_history, field_cond, pars_cond)

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
        """
        Forward pass for the Flow Matching model when training the drift model.

        Args:
            base (torch.Tensor): Base tensor [B, C, H, W].
            target (torch.Tensor): Target tensor [B, C, H, W].
            t (torch.Tensor): Time tensor [B, 1].
            noise (torch.Tensor): Noise tensor [B, C, H, W].
            field_cond (torch.Tensor): Field conditional tensor [B, C_field_cond, H, W]. Can be None.
            field_history (torch.Tensor): Field history tensor [B, C, H, W, L]. Can be None.
            pars_cond (torch.Tensor): pars conditional tensor [B, D_pars_cond]. Can be None.

        Returns:
            torch.Tensor: Drift tensor [B, C, H, W].
            torch.Tensor: Interpolation derivative tensor [B, C, H, W].
        """

        interpolant = self.interpolation.forward(
            base=noise,
            target=target,
            t=t,
        )

        pred_drift = self.drift_model(
            x=interpolant,
            cond=t,
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
        )

        true_drift = self.interpolation.forward_diff(base=noise, target=target, t=t)

        return pred_drift, true_drift

    def _prepare_batch(
        self,
        base: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ) -> tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """Prepare the batch for the sample method."""

        base = base.repeat(batch_size, 1, 1, 1) if base is not None else None
        field_history = field_history.repeat(batch_size, 1, 1, 1, 1)
        field_cond = (
            field_cond.repeat(batch_size, 1, 1, 1, 1)
            if field_cond is not None
            else None
        )
        pars_cond = pars_cond.repeat(batch_size, 1) if pars_cond is not None else None

        return base, field_history, field_cond, pars_cond

    def sample(
        self,
        field_history: torch.Tensor,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_steps: int = DEFAULT_NUM_STEPS,
        base: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        return_field_history: bool = False,
        ode_stepper: Callable = euler_step,
    ) -> torch.Tensor:
        """Sample from the Flow Matching model."""

        if (batch_size > 1) and (field_history.shape[0] == 1):
            base, field_history, field_cond, pars_cond = self._prepare_batch(
                base, field_history, field_cond, pars_cond, batch_size
            )

        if base is None:
            # The flow matching model always solves the ODE from noise to data
            base = torch.randn_like(field_history[:, :, :, :, 0])

        dt = torch.tensor(1 / num_steps, device=self._get_device())
        t_vec = torch.linspace(0, 1, num_steps, device=self._get_device()).unsqueeze(0)

        fixed_input = {
            "field_history": field_history,
            "field_cond": field_cond,
            "pars_cond": pars_cond,
            "drift_model": self.drift,
            "dt": dt,
        }

        # Sample from the Flow Matching model
        for i in range(0, num_steps):
            t = t_vec[:, i : i + 1]
            base = ode_stepper(x=base, t=t, **fixed_input).detach()

        # Add the new base to the field history
        if return_field_history:
            field_history = torch.cat(
                [field_history[:, :, :, :, 1:], base.unsqueeze(-1)], dim=-1
            )
            return base, field_history

        return base

    def sample_trajectory(
        self,
        field_history: torch.Tensor,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_steps: int = DEFAULT_NUM_STEPS,
        num_physical_steps: int = DEFAULT_NUM_PHYSICAL_STEPS,
        base: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        ode_stepper: Callable = euler_step,
    ) -> torch.Tensor:
        """Sample a trajectory from the Diffusion model."""

        if (batch_size > 1) and (field_history.shape[0] == 1):
            base, field_history, field_cond, pars_cond = self._prepare_batch(
                base, field_history, field_cond, pars_cond, batch_size
            )

        trajectory = [
            field_history[:, :, :, :, i].cpu() for i in range(field_history.shape[-1])
        ]

        fixed_input = {
            "num_steps": num_steps,
            "batch_size": batch_size,
            "return_field_history": True,
            "ode_stepper": ode_stepper,
        }
        cond_input = lambda i: {
            "field_cond": field_cond[:, :, :, :, i] if field_cond is not None else None,
            "pars_cond": pars_cond[:, i : i + 1] if pars_cond is not None else None,
        }

        with torch.no_grad():
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
