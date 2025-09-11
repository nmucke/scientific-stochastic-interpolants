from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import tqdm

from scisi.sampling.sde_solvers import euler_maruyama_step

MIN_TIME = 1e-4


class FollmerStochasticInterpolant(nn.Module):
    """Follmer stochastic interpolant."""

    def __init__(
        self,
        interpolation: nn.Module,
        drift_model: nn.Module,
    ) -> None:
        """Initialize Follmer stochastic interpolant."""
        super(FollmerStochasticInterpolant, self).__init__()

        self.interpolation = interpolation
        self.drift_model = drift_model

    def _get_device(self) -> str:
        """Get the device of the model."""
        return next(self.parameters()).device  # type: ignore[no-any-return]

    def drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor | None = None,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
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

    def score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor | None = None,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the score of the Follmer stochastic interpolant."""
        return self.drift_model(x, t, field_history, field_cond, pars_cond)

    def forward(
        self,
        base: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
        field_history: torch.Tensor | None = None,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
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

    def sample(
        self,
        base: torch.Tensor,
        field_history: torch.Tensor,
        batch_size: int = 1,
        num_steps: int = 100,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
        return_field_history: bool = False,
        sde_stepper: Callable = euler_maruyama_step,
        diffusion_term: Callable | None = None,
        drift_model: nn.Module = None,
    ) -> torch.Tensor:
        """Sample from the Follmer stochastic interpolant."""

        if diffusion_term is None:
            diffusion_term = self.interpolation.gamma

        if drift_model is None:
            drift_model = self.drift_model

        dt = torch.tensor(1 / num_steps, device=self._get_device())
        t_vec = torch.linspace(0, 1, num_steps, device=self._get_device()).unsqueeze(0)

        # Repeat the data if batch_size > 1
        if batch_size > 1:
            base = base.repeat(batch_size, 1, 1, 1)
            field_history = field_history.repeat(batch_size, 1, 1, 1, 1)
            if field_cond is not None:
                field_cond = field_cond.repeat(batch_size, 1, 1, 1)
            if pars_cond is not None:
                pars_cond = pars_cond.repeat(batch_size, 1)

        # Sample from the Follmer stochastic interpolant
        for i in range(0, num_steps):
            t = t_vec[:, i : i + 1]

            base = sde_stepper(
                drift_model=drift_model,
                diffusion_term=(
                    diffusion_term if t > MIN_TIME else self.interpolation.gamma
                ),
                x=base,
                t=t,
                dt=dt,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
            )

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
        batch_size: int = 1,
        num_steps: int = 100,
        num_physical_steps: int = 10,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
        sde_stepper: Callable = euler_maruyama_step,
        diffusion_term: Callable | None = None,
    ) -> torch.Tensor:
        """Sample a trajectory from the Follmer stochastic interpolant."""

        if diffusion_term is None:
            diffusion_term = self.interpolation.gamma

        trajectory = [
            field_history[:, :, :, :, i].cpu() for i in range(field_history.shape[-1])
        ]
        with torch.no_grad():
            pbar = tqdm.tqdm(range(0, num_physical_steps - field_history.shape[-1]))
            for _ in pbar:
                base, field_history = self.sample(
                    base=base,
                    batch_size=batch_size,
                    num_steps=num_steps,
                    field_history=field_history,
                    field_cond=field_cond,
                    pars_cond=pars_cond,
                    return_field_history=True,
                    sde_stepper=sde_stepper,
                    diffusion_term=diffusion_term,
                )
                trajectory.append(base.cpu())

        trajectory = torch.stack(trajectory, dim=-1)

        return trajectory

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
        A = 1 / A

        c = beta_diff * x + (beta * alpha_diff - beta_diff * alpha) * base

        return A * (beta * drift - c)

    def posterior_sample_trajectory(
        self,
        base: torch.Tensor,
        field_history: torch.Tensor,
        observations: torch.Tensor,
        likelihood_model: nn.Module,
        batch_size: int = 1,
        num_steps: int = 100,
        num_physical_steps: int = 10,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
        sde_stepper: Callable = euler_maruyama_step,
        diffusion_term: Callable | None = None,
    ) -> torch.Tensor:

        trajectory = [
            field_history[:, :, :, :, i].cpu() for i in range(field_history.shape[-1])
        ]

        pbar = tqdm.tqdm(enumerate(range(field_history.shape[-1], num_physical_steps)))
        for i, _ in pbar:
            drift_model = partial(
                self.posterior_drift,
                observations=observations[:, :, i],
                likelihood_model=likelihood_model,
                diffusion_term=diffusion_term,
            )
            base, field_history = self.sample(
                base=base,
                batch_size=batch_size,
                num_steps=num_steps,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
                return_field_history=True,
                sde_stepper=sde_stepper,
                diffusion_term=diffusion_term,
                drift_model=drift_model,
            )
            trajectory.append(base.cpu())

        trajectory = torch.stack(trajectory, dim=-1)

        return trajectory

    def posterior_drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        observations: torch.Tensor,
        likelihood_model: nn.Module,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
        diffusion_term: Callable | None = None,
    ) -> torch.Tensor:
        """Compute the posterior drift of the Follmer stochastic interpolant."""

        prior_drift = self.drift_model(x, t, field_history, field_cond, pars_cond)

        if t < MIN_TIME:
            return prior_drift

        if diffusion_term is None:
            diffusion_term = self.interpolation.gamma

        # Compute the interpolant of the observation
        base_obs = likelihood_model.obs_operator(field_history[:, :, :, :, -1])
        interpolant_obs = self.interpolation.forward(
            base_obs, observations, t, torch.zeros_like(base_obs)
        )

        # Compute the scale of the interpolant of the observation
        interpolant_scale = (
            self.interpolation.beta(t) ** 2 * likelihood_model.original_scale
        )
        interpolant_scale = interpolant_scale + self.interpolation.gamma(t) ** 2 * t

        # Update the observation and scale of the likelihood model
        likelihood_model.update_obs(interpolant_obs)
        likelihood_model.update_scale(interpolant_scale)
        likelihood_score = likelihood_model.score(x)

        # Compute the posterior drift
        prior_score = self._prior_score(
            x, field_history[:, :, :, :, -1], prior_drift, t
        )
        posterior_drift = (
            prior_drift
            + 0.5
            * (diffusion_term(t) ** 2 - self.interpolation.gamma(t) ** 2)
            * prior_score
        )
        posterior_drift = (
            posterior_drift + 0.5 * diffusion_term(t) ** 2 * likelihood_score
        )

        return posterior_drift
