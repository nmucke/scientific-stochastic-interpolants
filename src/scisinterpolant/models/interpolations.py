"""Interpolants.

This module contains the interpolant models for the scisinterpolant package.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class LinearDeterministicInterpolation(nn.Module):
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

    def forward(
        self, base: torch.Tensor, target: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        t = self._reshape_t(t)
        return self.alpha(t) * base + self.beta(t) * target

    def forward_diff(
        self, base: torch.Tensor, target: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Forward derivative."""
        t = self._reshape_t(t)
        return self.alpha_diff(t) * base + self.beta_diff(t) * target


class QuadraticDeterministicInterpolation(nn.Module):
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
        return t**2

    def beta_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Beta derivative."""
        return 2 * t

    def forward(
        self, base: torch.Tensor, target: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        t = self._reshape_t(t)
        return self.alpha(t) * base + self.beta(t) * target

    def forward_diff(
        self, base: torch.Tensor, target: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Forward derivative."""
        t = self._reshape_t(t)
        return self.alpha_diff(t) * base + self.beta_diff(t) * target


class LinearStochasticInterpolation(nn.Module):
    """Linear stochastic interpolant."""

    def __init__(self) -> None:
        """Initialize linear stochastic interpolant."""
        super(LinearStochasticInterpolation, self).__init__()

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Alpha function."""
        return 1 - t

    def alpha_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Alpha derivative."""
        return -1

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Beta function."""
        return t

    def beta_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Beta derivative."""
        return 1

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """Gamma function."""
        return 1 - t

    def gamma_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Gamma derivative."""
        return -1

    def forward(
        self,
        base: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        t = self._reshape_t(t)
        return self.alpha(t) * base + self.beta(t) * target + self.gamma(t) * noise

    def forward_diff(
        self,
        base: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Forward derivative."""
        t = self._reshape_t(t)
        return (
            self.alpha_diff(t) * base
            + self.beta_diff(t) * target
            + self.gamma_diff(t) * noise
        )


class QuadraticStochasticInterpolation(nn.Module):
    """Quadratic stochastic interpolant."""

    def __init__(self) -> None:
        """Initialize quadratic stochastic interpolant."""
        super(QuadraticStochasticInterpolation, self).__init__()

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Alpha function."""
        return 1 - t

    def alpha_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Alpha derivative coefficient."""
        return -1

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Beta function."""
        return t**2

    def beta_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Beta derivative."""
        return 2 * t

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """Gamma function."""
        return 1 - t

    def gamma_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Gamma derivative."""
        return -1

    def forward(
        self,
        base: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        t = self._reshape_t(t)
        return self.alpha(t) * base + self.beta(t) * target + self.gamma(t) * noise

    def forward_diff(
        self,
        base: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Forward derivative."""
        t = self._reshape_t(t)
        return (
            self.alpha_diff(t) * base
            + self.beta_diff(t) * target
            + self.gamma_diff(t) * noise
        )
