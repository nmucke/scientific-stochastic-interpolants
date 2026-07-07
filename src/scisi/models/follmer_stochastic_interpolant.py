import pdb
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import tqdm

from scisi.models.base_model import BaseModel
from scisi.sampling.ode_solvers import euler_step
from scisi.sampling.sde_solvers import euler_maruyama_step

MIN_TIME = 1e-4
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_STEPS = 100
DEFAULT_NUM_PHYSICAL_STEPS = 10


class FollmerStochasticInterpolant(BaseModel):
    """Follmer stochastic interpolant."""

    def __init__(
        self,
        interpolation: nn.Module,
        drift_model: nn.Module,
        diffusion_term: Optional[nn.Module] = None,
        mask_path: Optional[str] = None,
    ) -> None:
        """Initialize Follmer stochastic interpolant."""
        super(FollmerStochasticInterpolant, self).__init__(mask_path=mask_path)

        self.interpolation = interpolation
        self.drift_model = drift_model

        self.diffusion_term = diffusion_term
        if diffusion_term is None:
            self.diffusion_term = self.interpolation.gamma

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
        Compute the drift of the Follmer stochastic interpolant.

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
        Forward pass for the Follmer stochastic interpolant when training the drift model.

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
            base=base,
            target=target,
            t=t,
            noise=noise,
        )

        pred_drift = self.drift_model(
            x=interpolant,
            cond=t,
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
        )

        true_diff = self.interpolation.forward_diff(
            base=base, target=target, t=t, noise=noise
        )

        return pred_drift, true_diff

    def _compute_first_step(
        self,
        base: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        stepper: Callable = euler_maruyama_step,
        mask: torch.Tensor = torch.tensor(1.0),
    ) -> torch.Tensor:
        """Compute the first step of the Follmer stochastic interpolant."""
        base = stepper(
            drift_model=self.drift,
            diffusion_term=self.interpolation.gamma,
            x=base,
            t=t,
            dt=dt,
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
            mask=mask,
        )
        return base  # * self.mask

    def sample(
        self,
        base: torch.Tensor,
        field_history: torch.Tensor,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_steps: int = DEFAULT_NUM_STEPS,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        return_field_history: bool = False,
        stepper: Callable = euler_maruyama_step,
        diffusion_term: Optional[Callable] = None,
        ode_drift: bool = False,
    ) -> torch.Tensor:
        """Sample from the Follmer stochastic interpolant."""

        if diffusion_term is None:
            # If no diffusion term is provided, use the interpolant's gamma and trained drift model
            diffusion_term = self.interpolation.gamma
            drift_model = self.drift
        else:
            drift_model = partial(
                self._drift_with_prior_score, diffusion_term=diffusion_term
            )

        if ode_drift:
            drift_model = self._ode_drift
            stepper = euler_step

        return self._sample(
            field_history=field_history,
            stepper=stepper,
            base=base,
            batch_size=batch_size,
            num_steps=num_steps,
            field_cond=field_cond,
            pars_cond=pars_cond,
            return_field_history=return_field_history,
            diffusion_term=diffusion_term,
            drift=drift_model,
            with_first_step=True,
        )

    def sample_trajectory(
        self,
        field_history: torch.Tensor,
        base: Optional[torch.Tensor] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_steps: int = DEFAULT_NUM_STEPS,
        num_physical_steps: int = DEFAULT_NUM_PHYSICAL_STEPS,
        stepper: Callable = euler_maruyama_step,
        diffusion_term: Optional[Callable] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample a trajectory from the Follmer stochastic interpolant."""

        return self._sample_trajectory(
            field_history=field_history,
            base=base,
            field_cond=field_cond,
            pars_cond=pars_cond,
            batch_size=batch_size,
            num_steps=num_steps,
            num_physical_steps=num_physical_steps,
            stepper=stepper,
            diffusion_term=diffusion_term,
        )

    def _prior_score(
        self,
        x: torch.Tensor,
        base: torch.Tensor,
        drift: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the prior score of the Follmer stochastic interpolant."""

        gamma = self.interpolation.gamma(t)
        gamma_diff = self.interpolation.gamma_diff(t)
        beta = self.interpolation.beta(t)
        beta_diff = self.interpolation.beta_diff(t)
        alpha = self.interpolation.alpha(t)
        alpha_diff = self.interpolation.alpha_diff(t)

        A = t * gamma * (beta_diff * gamma - beta * gamma_diff)
        A = 1 / (A + 1e-6)

        c = beta_diff * x + (beta * alpha_diff - beta_diff * alpha) * base

        return A * (beta * drift - c)

    def _drift_with_prior_score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        diffusion_term: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Compute the posterior drift of the Follmer stochastic interpolant."""

        if diffusion_term is None:
            diffusion_term = self.interpolation.gamma

        drift = self.drift_model(x, t, field_history, field_cond, pars_cond)

        if t < MIN_TIME:
            return drift

        # Compute the prior score
        prior_score = self._prior_score(x, field_history[:, :, :, :, -1], drift, t)
        drift = (
            drift
            + 0.5
            * (diffusion_term(t) ** 2 - self.interpolation.gamma(t) ** 2)
            * prior_score
        )

        return drift
