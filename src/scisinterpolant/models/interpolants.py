"""Interpolants.

This module contains the interpolant models for the scisinterpolant package.
"""

from abc import ABC
import torch
import torch.nn as nn

class BaseDeterministicInterpolation(nn.Module, ABC):
    """Base deterministic interpolant."""

    def __init__(self) -> None:
        """Initialize base deterministic interpolant."""
        super(BaseDeterministicInterpolation, self).__init__()

    @abstractmethod
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Alpha function."""
        pass

    @abstractmethod
    def alpha_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Alpha derivative coefficient."""
        pass

    @abstractmethod
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Beta function."""
        pass

    @abstractmethod
    def beta_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Beta derivative."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pass

    @abstractmethod
    def forward_diff(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward derivative."""
        pass


class LinearDeterministicInterpolation(BaseDeterministicInterpolation):
    """Linear deterministic interpolant."""

    def __init__(self) -> None:
        """Initialize linear interpolant."""
        super(LinearDeterministicInterpolation, self).__init__()   

    def alpha_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Alpha derivative coefficient."""
        return -1

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Alpha function."""
        return 1 - t

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Beta function."""
        return t

    def beta_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Beta derivative."""
        return 1

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.alpha(t) * x + self.beta(t) * x

    def forward_diff(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward derivative."""
        return self.alpha_diff(t) * x + self.beta_diff(t) * x

class QuadraticDeterministicInterpolation(BaseDeterministicInterpolation):
    """Quadratic deterministic interpolant."""

    def __init__(self) -> None:
        """Initialize quadratic interpolant."""
        super(QuadraticDeterministicInterpolation, self).__init__()   

    def alpha_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Alpha derivative coefficient."""
        return -1

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Alpha function."""
        return 1 - t

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Beta function."""
        return t ** 2

    def beta_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Beta derivative."""
        return 2 * t

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.alpha(t) * x + self.beta(t) * x

    def forward_diff(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward derivative."""
        return self.alpha_diff(t) * x + self.beta_diff(t) * x


class LinearStochasticInterpolation(LinearDeterministicInterpolation):
    """Linear stochastic interpolant."""

    def __init__(self) -> None:
        """Initialize linear stochastic interpolant."""
        super(LinearStochasticInterpolation, self).__init__()   

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """Gamma function."""
        return 1 - t

    def gamma_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Gamma derivative."""
        return -1

    def forward(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.alpha(t) * x + self.beta(t) * x + self.gamma(t) * z

    def forward_diff(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Forward derivative."""
        return self.alpha_diff(t) * x + self.beta_diff(t) * x + self.gamma_diff(t) * z


class QuadraticStochasticInterpolation(QuadraticDeterministicInterpolation):
    """Quadratic stochastic interpolant."""

    def __init__(self) -> None:
        """Initialize quadratic stochastic interpolant."""
        super(QuadraticStochasticInterpolation, self).__init__()   

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """Gamma function."""
        return 1 - t

    def gamma_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Gamma derivative."""
        return -1

    def forward(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.alpha(t) * x + self.beta(t) * x + self.gamma(t) * z

    def forward_diff(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Forward derivative."""
        return self.alpha_diff(t) * x + self.beta_diff(t) * x + self.gamma_diff(t) * z