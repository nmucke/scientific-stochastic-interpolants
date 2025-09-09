from typing import Callable

import torch
import torch.nn as nn
import tqdm

from scisinterpolant.sampling.sde_solvers import euler_maruyama_step


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
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the drift of the Follmer stochastic interpolant.
        """
        return self.drift_model(x, t, field_cond, pars_cond)

    def score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the score of the Follmer stochastic interpolant."""
        return self.drift_model(x, t, field_cond, pars_cond)

    def forward(
        self,
        base: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
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
        batch_size: int = 1,
        num_steps: int = 100,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
        return_next_field_cond: bool = False,
        sde_stepper: Callable = euler_maruyama_step,
    ) -> torch.Tensor:
        """Sample from the Follmer stochastic interpolant."""

        dt = torch.tensor(1 / num_steps, device=self._get_device())
        t_vec = torch.linspace(0, 1, num_steps, device=self._get_device()).unsqueeze(0)

        # Repeat the data if batch_size > 1
        if batch_size > 1:
            base = base.repeat(batch_size, 1, 1, 1)
            if field_cond is not None:
                field_cond = field_cond.repeat(batch_size, 1, 1, 1)
            if pars_cond is not None:
                pars_cond = pars_cond.repeat(batch_size, 1)

        # Sample from the Follmer stochastic interpolant
        with torch.no_grad():
            for i in range(0, num_steps):
                t = t_vec[:, i : i + 1]

                base = sde_stepper(
                    drift_model=self.drift_model,
                    diffusion_model=self.interpolation.gamma,
                    x=base,
                    t=t,
                    dt=dt,
                    field_cond=field_cond,
                    pars_cond=pars_cond,
                )

        if return_next_field_cond and field_cond is not None:
            field_cond = torch.cat([field_cond[:, 1:], base], dim=1)
            return base, field_cond

        return base

    def sample_trajectory(
        self,
        base: torch.Tensor,
        batch_size: int = 1,
        num_steps: int = 100,
        num_physical_steps: int = 10,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
        sde_stepper: Callable = euler_maruyama_step,
    ) -> torch.Tensor:
        """Sample a trajectory from the Follmer stochastic interpolant."""

        trajectory = []
        pbar = tqdm.tqdm(range(0, num_physical_steps))
        for _ in pbar:
            base, field_cond = self.sample(
                base=base,
                batch_size=batch_size,
                num_steps=num_steps,
                field_cond=field_cond,
                pars_cond=pars_cond,
                return_next_field_cond=True,
                sde_stepper=sde_stepper,
            )
            trajectory.append(base.cpu())

        trajectory = torch.stack(trajectory, dim=-1)

        return trajectory
