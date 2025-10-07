from typing import Optional

import torch
import torch.nn as nn

from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant


class AnalyticalStochasticInterpolant(nn.Module):
    """Analytical stochastic interpolant."""

    def __init__(
        self,
        interpolation: nn.Module,
        drift_model: nn.Module,
        diffusion_term: Optional[nn.Module] = None,
    ) -> None:
        """Initialize analytical stochastic interpolant."""
        super(AnalyticalStochasticInterpolant, self).__init__()
        self.interpolation = interpolation
        self.drift_model = drift_model
        self.diffusion_term = diffusion_term

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pass


def main() -> None:
    """Main function."""
    pass


if __name__ == "__main__":
    main()
