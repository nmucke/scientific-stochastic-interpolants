from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseSDESolver(ABC):
    """SDE solver for the stochastic interpolant."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = device

    def _sample_wiener_process(
        self,
        shape: tuple[int, ...],
        dt: float,
    ) -> torch.Tensor:
        """Sample noise for the SDE solver."""
        return torch.randn(shape, device=self.device) * torch.sqrt(dt)

    @abstractmethod
    def _compute_one_step(
        self,
        x: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute one step of the SDE."""
        pass

    def solve(
        self,
        x0: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Solve the SDE."""
        x = x0
        dt = 1 / num_steps

        for _ in range(num_steps):
            x = self._compute_one_step(x, dt)
        return x
