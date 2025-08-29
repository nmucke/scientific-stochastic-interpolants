import torch
import torch.nn as nn


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

        x = self.interpolation.forward(
            base=base,
            target=target,
            t=t,
            noise=noise,
        )

        drift = self.drift_model(
            x=x,
            cond=t,
            field_cond=field_cond,
            pars_cond=pars_cond,
        )

        x_diff = self.interpolation.forward_diff(
            base=base, target=target, t=t, noise=noise
        )

        return drift, x_diff
