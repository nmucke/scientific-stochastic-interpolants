import pdb
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import tqdm

from scisi.models.base_model import BaseModel
from scisi.models.interpolations import _expand_t
from scisi.sampling.sde_solvers import euler_maruyama_step

MIN_TIME = 1e-4
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_STEPS = 100
DEFAULT_NUM_PHYSICAL_STEPS = 10


class DiffusionModel(BaseModel):
    """Diffusion model."""

    def __init__(
        self,
        interpolation: nn.Module,
        score_model: nn.Module,
        diffusion_term: Optional[nn.Module] = None,
    ) -> None:
        """Initialize Diffusion model."""
        super(DiffusionModel, self).__init__()

        self.interpolation = interpolation
        self.score_model = score_model

        self.diffusion_term = diffusion_term
        if diffusion_term is None:
            self.diffusion_term = self.interpolation.alpha

    @property
    def model(self) -> nn.Module:
        """
        Get the drift model.

        This is to ensure compatibility with the rest of the code base.

        Returns:
            nn.Module: The drift model.
        """
        return self.score_model

    def _get_velocity_from_score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
    ) -> torch.Tensor:
        """Get the velocity of the Diffusion model."""

        alpha = self.interpolation.alpha(t)
        beta = self.interpolation.beta(t)
        alpha_diff = self.interpolation.alpha_diff(t)
        beta_diff = self.interpolation.beta_diff(t)

        score_coeff = alpha**2 * beta_diff / beta - alpha_diff * alpha
        x_coeff = beta_diff / beta
        return score_coeff * score + x_coeff * x

    def drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the drift of the Diffusion model.

        Args:
            x (torch.Tensor): Input tensor [B, C, H, W].
            t (torch.Tensor): Time tensor [B, 1].
            field_history (torch.Tensor): Field history tensor [B, C, H, W, L]. Can be None.
            field_cond (torch.Tensor): Field conditional tensor [B, C_field_cond, H, W]. Can be None.
            pars_cond (torch.Tensor): pars conditional tensor [B, D_pars_cond]. Can be None.
        """

        score = self.score_model(x, t, field_history, field_cond, pars_cond)

        velocity = self._get_velocity_from_score(x, t, score)

        return velocity + 0.5 * self.diffusion_term(t) ** 2 * score  # type: ignore[misc]

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
        Forward pass for the diffusion model when training the drift model.

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

        pred_score = self.score_model(
            x=interpolant,
            cond=t,
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
        )

        # We return the negative score because the trainer is set to minimize pred_score - true_score
        # But the actual loss is pred_score + noise / alpha(t)
        true_score = -noise / self.interpolation.alpha(_expand_t(t, noise))

        return pred_score, true_score

    def _compute_first_step(
        self,
        base: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        stepper: Optional[Callable] = euler_maruyama_step,
    ) -> torch.Tensor:
        """Compute the first step of the Follmer stochastic interpolant."""

        drift = lambda x, t, field_history, field_cond, pars_cond: (
            0.5
            * self.diffusion_term(t) ** 2  # type: ignore[misc]
            * self.score_model(x, t, field_history, field_cond, pars_cond)
        )

        return stepper(  # type: ignore[misc]
            drift_model=drift,
            diffusion_term=self.diffusion_term,
            x=base,
            t=t,
            dt=dt,
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
        )

    def sample(
        self,
        field_history: torch.Tensor,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_steps: int = DEFAULT_NUM_STEPS,
        base: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        return_field_history: bool = False,
        stepper: Callable = euler_maruyama_step,
        diffusion_term: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Sample from the Diffusion model."""

        return self._sample(
            base=base,
            field_history=field_history,
            batch_size=batch_size,
            num_steps=num_steps,
            field_cond=field_cond,
            pars_cond=pars_cond,
            return_field_history=return_field_history,
            stepper=stepper,
            diffusion_term=diffusion_term,
            with_first_step=True,
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
        stepper: Callable = euler_maruyama_step,
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
