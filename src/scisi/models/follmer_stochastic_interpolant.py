import pdb
from typing import Callable
from functools import partial

import torch
import torch.nn as nn
import tqdm

from scisi.sampling.sde_solvers import euler_maruyama_step

MIN_TIME = 1e-4
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_STEPS = 100
DEFAULT_NUM_PHYSICAL_STEPS = 10


class FollmerStochasticInterpolant(nn.Module):
    """Follmer stochastic interpolant."""

    def __init__(
        self,
        interpolation: nn.Module,
        drift_model: nn.Module,
        likelihood_model: nn.Module | None = None,
        diffusion_term: nn.Module | None = None,
        observations: torch.Tensor | None = None,
    ) -> None:
        """Initialize Follmer stochastic interpolant."""
        super(FollmerStochasticInterpolant, self).__init__()

        self.interpolation = interpolation
        self.drift_model = drift_model

        self.likelihood_model = likelihood_model
        self.observations = observations

        self.diffusion_term = diffusion_term
        if diffusion_term is None:
            self.diffusion_term = self.interpolation.gamma

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

    def _prepare_batch(
        self, 
        base: torch.Tensor, 
        field_history: torch.Tensor, 
        field_cond: torch.Tensor | None = None, 
        pars_cond: torch.Tensor | None = None,
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Prepare the batch for the sample method."""
        
        base = base.repeat(batch_size, 1, 1, 1)
        field_history = field_history.repeat(batch_size, 1, 1, 1, 1)
        field_cond = field_cond.repeat(batch_size, 1, 1, 1) if field_cond is not None else None
        pars_cond = pars_cond.repeat(batch_size, 1) if pars_cond is not None else None

        return base, field_history, field_cond, pars_cond
    
    def _compute_first_step(
        self,
        base: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
        sde_stepper: Callable = euler_maruyama_step,        
    ) -> torch.Tensor:
        """Compute the first step of the Follmer stochastic interpolant."""
        return sde_stepper(
                drift_model=self.drift_model,
                diffusion_term=self.interpolation.gamma,
                x=base,
                t=t,
                dt=dt,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
            )

    def sample(
        self,
        base: torch.Tensor,
        field_history: torch.Tensor,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_steps: int = DEFAULT_NUM_STEPS,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
        return_field_history: bool = False,
        sde_stepper: Callable = euler_maruyama_step,
        diffusion_term: Callable | None = None,
    ) -> torch.Tensor:
        """Sample from the Follmer stochastic interpolant."""

        if diffusion_term is None:
            # If no diffusion term is provided, use the interpolant's gamma and trained drift model
            diffusion_term = self.interpolation.gamma
            drift_model = self.drift_model
        else:
            drift_model = partial(self._drift_with_prior_score, diffusion_term=diffusion_term)

        base, field_history, field_cond, pars_cond = self._prepare_batch(
            base, field_history, field_cond, pars_cond, batch_size
        )

        dt = torch.tensor(1 / num_steps, device=self._get_device())
        t_vec = torch.linspace(0, 1, num_steps, device=self._get_device()).unsqueeze(0)

        fixed_input = {
            "field_history": field_history,
            "field_cond": field_cond,
            "pars_cond": pars_cond,
            "dt": dt
        }

        base = self._compute_first_step(
            base=base, 
            t=t_vec[:, 0], 
            sde_stepper=sde_stepper,
            **fixed_input,
        ).detach()

        fixed_input["drift_model"] = drift_model
        fixed_input["diffusion_term"] = diffusion_term

        # Sample from the Follmer stochastic interpolant
        for i in range(1, num_steps):
            t = t_vec[:, i : i + 1]
            base = sde_stepper(x=base, t=t, **fixed_input).detach()

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
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_steps: int = DEFAULT_NUM_STEPS,
        num_physical_steps: int = DEFAULT_NUM_PHYSICAL_STEPS,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
        sde_stepper: Callable = euler_maruyama_step,
        diffusion_term: Callable | None = None,
    ) -> torch.Tensor:
        """Sample a trajectory from the Follmer stochastic interpolant."""

        trajectory = [
            field_history[:, :, :, :, i].cpu() for i in range(field_history.shape[-1])
        ]

        fixed_input = {
            "num_steps": num_steps,
            "batch_size": batch_size,
            "return_field_history": True,
            "sde_stepper": sde_stepper,
            "diffusion_term": diffusion_term,
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
                    **fixed_input,
                )
                trajectory.append(base.cpu())

        return torch.stack(trajectory, dim=-1)

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
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
        diffusion_term: Callable | None = None,
    ) -> torch.Tensor:
        """Compute the posterior drift of the Follmer stochastic interpolant."""

        drift = self.drift_model(x, t, field_history, field_cond, pars_cond)

        if t < MIN_TIME:
            return drift

        # Compute the posterior drift
        prior_score = self._prior_score(
            x, field_history[:, :, :, :, -1], drift, t
        )
        drift = (
            drift
            + 0.5
            * (diffusion_term(t) ** 2 - self.interpolation.gamma(t) ** 2)
            * prior_score
        )

        return drift

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
        """Sample a trajectory from the Follmer stochastic interpolant with posterior drift."""

        trajectory = [
            field_history[:, :, :, :, i].cpu() for i in range(field_history.shape[-1])
        ]

        if diffusion_term is not None:
            self.diffusion_term = diffusion_term

        self.likelihood_model = likelihood_model

        pbar = tqdm.tqdm(
            enumerate(range(0, num_physical_steps - field_history.shape[-1]))
        )
        for i, _ in pbar:
            self.observations = observations[:, :, i].to(self._get_device())
            self.likelihood_model.update_obs(self.observations)

            base, field_history = self.sample(
                base=base,
                batch_size=batch_size,
                num_steps=num_steps,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
                return_field_history=True,
                sde_stepper=sde_stepper,
                drift_model=self.posterior_drift,
            )
            trajectory.append(base.detach().cpu())

        trajectory = torch.stack(trajectory, dim=-1)

        return trajectory

    def posterior_drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the posterior drift of the Follmer stochastic interpolant."""

        prior_drift = self.drift_model(x, t, field_history, field_cond, pars_cond)

        if t < MIN_TIME:
            return prior_drift

        # Compute the interpolant of the observation
        base_obs = self.likelihood_model.obs_operator(field_history[:, :, :, :, -1])  # type: ignore[union-attr]
        interpolant_obs = self.interpolation.forward(
            base_obs, self.observations, t, torch.zeros_like(base_obs)
        )

        # Compute the scale of the interpolant of the observation
        interpolant_scale = (
            self.interpolation.beta(t) ** 2 * self.likelihood_model.original_scale  # type: ignore[union-attr]
        )
        interpolant_scale = interpolant_scale + self.interpolation.gamma(t) ** 2 * t

        # Update the observation and scale of the likelihood model
        self.likelihood_model.update_obs(interpolant_obs)  # type: ignore[union-attr]
        self.likelihood_model.update_scale(interpolant_scale)  # type: ignore[union-attr]
        likelihood_score = self.likelihood_model.score(x)  # type: ignore[union-attr]

        # Compute the posterior drift
        prior_score = self._prior_score(
            x, field_history[:, :, :, :, -1], prior_drift, t
        )
        posterior_drift = (
            prior_drift
            + 0.5
            * (self.diffusion_term(t) ** 2 - self.interpolation.gamma(t) ** 2)  # type: ignore[misc]
            * prior_score
        )

        posterior_drift = (
            posterior_drift + 0.5 * self.diffusion_term(t) ** 2 * likelihood_score  # type: ignore[misc]
        )

        return posterior_drift
