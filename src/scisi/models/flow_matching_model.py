import pdb
from functools import partial
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import tqdm

from scisi.models.base_model import BaseModel
from scisi.models.interpolations import _expand_t
from scisi.sampling.ode_solvers import euler_step

MIN_TIME = 1e-4
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_STEPS = 100
DEFAULT_NUM_PHYSICAL_STEPS = 10


class FlowMatchingModel(BaseModel):
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

    @property
    def model(self) -> nn.Module:
        """
        Get the drift model.

        This is to ensure compatibility with the rest of the code base.
        """
        return self.drift_model

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

    def sample(
        self,
        field_history: torch.Tensor,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_steps: int = DEFAULT_NUM_STEPS,
        base: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        return_field_history: bool = False,
        stepper: Callable = euler_step,
        gaussian_base: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Sample from the Flow Matching model."""

        return self._sample(
            field_history=field_history,
            stepper=stepper,
            batch_size=batch_size,
            num_steps=num_steps,
            base=base,
            field_cond=field_cond,
            pars_cond=pars_cond,
            return_field_history=return_field_history,
        )

    def sample_trajectory(
        self,
        field_history: torch.Tensor,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_steps: int = DEFAULT_NUM_STEPS,
        num_physical_steps: int = DEFAULT_NUM_PHYSICAL_STEPS,
        base: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        stepper: Callable = euler_step,
    ) -> torch.Tensor:
        """Sample a trajectory from the Flow Matching model."""

        return self._sample_trajectory(
            field_history=field_history,
            batch_size=batch_size,
            num_steps=num_steps,
            num_physical_steps=num_physical_steps,
            base=base,
            field_cond=field_cond,
            pars_cond=pars_cond,
            stepper=stepper,
            gaussian_base=True,
        )
