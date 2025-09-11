"""Interpolants.

This module contains the interpolant models for the scisi package.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from einops import rearrange


def _expand_t(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Expand time tensor from [B, 1] to have as many dims as x.

    First dim is always batch and second dim is always time.
    """
    return rearrange(t, "b 1 -> b 1" + " ".join(["1" for _ in range(len(x.shape) - 2)]))


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
        t = _expand_t(t, base)
        return self.alpha(t) * base + self.beta(t) * target

    def forward_diff(
        self, base: torch.Tensor, target: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Forward derivative."""
        t = _expand_t(t, base)
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
        t = _expand_t(t, base)
        return self.alpha(t) * base + self.beta(t) * target

    def forward_diff(
        self, base: torch.Tensor, target: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Forward derivative."""
        t = _expand_t(t, base)
        return self.alpha_diff(t) * base + self.beta_diff(t) * target


class LinearStochasticInterpolation(nn.Module):
    """Linear stochastic interpolant."""

    def __init__(
        self,
        gamma_multiplier: float = 0.1,
        wiener_process: bool = True,
    ) -> None:
        """Initialize linear stochastic interpolant."""
        super(LinearStochasticInterpolation, self).__init__()
        self.gamma_multiplier = gamma_multiplier

        if wiener_process:
            self.transform_noise = lambda x, t: x * torch.sqrt(t)
        else:
            self.transform_noise = lambda x, t: x

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
        return self.gamma_multiplier * (1 - t)

    def gamma_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Gamma derivative."""
        return -self.gamma_multiplier

    def forward(
        self,
        base: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        t = _expand_t(t, base)
        return (
            self.alpha(t) * base
            + self.beta(t) * target
            + self.gamma(t) * self.transform_noise(noise, t)
        )

    def forward_diff(
        self,
        base: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Forward derivative."""
        t = _expand_t(t, base)
        return (
            self.alpha_diff(t) * base
            + self.beta_diff(t) * target
            + self.gamma_diff(t) * self.transform_noise(noise, t)
        )


class QuadraticStochasticInterpolation(nn.Module):
    """Quadratic stochastic interpolant."""

    def __init__(
        self, gamma_multiplier: float = 0.1, wiener_process: bool = True
    ) -> None:
        """Initialize quadratic stochastic interpolant."""
        super(QuadraticStochasticInterpolation, self).__init__()
        self.gamma_multiplier = gamma_multiplier

        if wiener_process:
            self.transform_noise = lambda x, t: x * torch.sqrt(t)
        else:
            self.transform_noise = lambda x, t: x

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
        return self.gamma_multiplier * (1 - t)

    def gamma_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Gamma derivative."""
        return -self.gamma_multiplier

    def forward(
        self,
        base: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        t = _expand_t(t, base)
        return (
            self.alpha(t) * base
            + self.beta(t) * target
            + self.gamma(t) * self.transform_noise(noise, t)
        )

    def forward_diff(
        self,
        base: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Forward derivative."""
        t = _expand_t(t, base)
        return (
            self.alpha_diff(t) * base
            + self.beta_diff(t) * target
            + self.gamma_diff(t) * self.transform_noise(noise, t)
        )
