"""Interpolants.

This module contains the interpolant models for the scisi package.
"""

import pdb
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from einops import rearrange

# Clamp pseudo-time away from the endpoints {0, 1} when evaluating
# schedule-derived quantities whose denominators vanish there.
MIN_TIME = 1e-4


def _clamp_time(t: torch.Tensor) -> torch.Tensor:
    """Clamp pseudo-time away from the singular endpoints 0 and 1."""
    return torch.clamp(t, min=MIN_TIME, max=1.0 - MIN_TIME)


def _expand_t(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Expand time tensor from [B, 1] to have as many dims as x.

    First dim is always batch and second dim is always time.
    """
    return rearrange(
        t, "b 1 -> b 1 " + " ".join(["1" for _ in range(len(x.shape) - 2)])
    )


class AffineGaussianPathMixin:
    """Shared velocity<->score identity for affine Gaussian probability paths.

    Implements the single source of truth for the velocity--score duality of
    Eqs. (general_velocity_of_score)/(general_score_velocity) and the
    velocity--score coefficient ``a_tau`` of Eq. (vscoef_def), general in the
    schedules ``alpha_tau``, ``beta_tau`` and the source scale ``sigma_tau``.

    Concrete interpolations expose ``alpha``/``alpha_diff``, ``beta``/
    ``beta_diff`` and the source scale via ``sigma``/``sigma_diff``. For the
    deterministic/flow-matching paths the source scale is ``sigma = alpha``;
    for the stochastic-interpolant paths it is ``sigma = gamma * sqrt(t)``.
    The deterministic anchor ``a0`` is ``x0`` for SI and ``0`` for FM.
    """

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Source scale sigma_tau (defaults to alpha; SI overrides)."""
        return self.alpha(t)

    def sigma_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Derivative of the source scale (defaults to alpha_diff)."""
        return self.alpha_diff(t)

    def velocity_score_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Velocity--score coefficient a_tau (paper Eq. vscoef_def).

        a_tau = sigma * (beta_diff * sigma - sigma_diff * beta) / beta,
        with sigma = alpha for FM/diffusion and sigma = gamma * sqrt(t) for SI.

        Args:
            t (torch.Tensor): Time tensor.

        Returns:
            torch.Tensor: The velocity--score coefficient a_tau.
        """
        t = _clamp_time(t)
        sigma = self.sigma(t)
        sigma_diff = self.sigma_diff(t)
        beta = self.beta(t)
        beta_diff = self.beta_diff(t)
        return sigma * (beta_diff * sigma - sigma_diff * beta) / beta

    def velocity_from_score(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        a0: torch.Tensor,
    ) -> torch.Tensor:
        """Velocity from score (paper Eq. general_velocity_of_score).

        v = (beta_diff / beta) * x
            + (alpha_diff - beta_diff * alpha / beta) * a0
            + a_tau * s.

        Args:
            x (torch.Tensor): State tensor at pseudo-time t.
            s (torch.Tensor): Score tensor at (x, t).
            t (torch.Tensor): Time tensor.
            a0 (torch.Tensor): Deterministic anchor (x0 for SI, 0 for FM).

        Returns:
            torch.Tensor: The velocity field.
        """
        t = _clamp_time(t)
        alpha = self.alpha(t)
        alpha_diff = self.alpha_diff(t)
        beta = self.beta(t)
        beta_diff = self.beta_diff(t)
        a_tau = self.velocity_score_coeff(t)
        return (
            (beta_diff / beta) * x
            + (alpha_diff - beta_diff * alpha / beta) * a0
            + a_tau * s
        )

    def score_from_velocity(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
        a0: torch.Tensor,
    ) -> torch.Tensor:
        """Score from velocity (paper Eq. general_score_velocity).

        s = (beta * v - beta_diff * x - (alpha_diff * beta - beta_diff * alpha) * a0)
            / (beta * a_tau).

        Args:
            x (torch.Tensor): State tensor at pseudo-time t.
            v (torch.Tensor): Velocity tensor at (x, t).
            t (torch.Tensor): Time tensor.
            a0 (torch.Tensor): Deterministic anchor (x0 for SI, 0 for FM).

        Returns:
            torch.Tensor: The score field.
        """
        t = _clamp_time(t)
        alpha = self.alpha(t)
        alpha_diff = self.alpha_diff(t)
        beta = self.beta(t)
        beta_diff = self.beta_diff(t)
        a_tau = self.velocity_score_coeff(t)
        numerator = (
            beta * v
            - beta_diff * x
            - (alpha_diff * beta - beta_diff * alpha) * a0
        )
        return numerator / (beta * a_tau)


class LinearDeterministicInterpolation(AffineGaussianPathMixin, nn.Module):
    """Linear deterministic interpolant."""

    def __init__(self) -> None:
        """Initialize linear interpolant."""
        super(LinearDeterministicInterpolation, self).__init__()

    def alpha_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Alpha derivative coefficient."""
        return -torch.ones_like(t)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Alpha function."""
        return 1 - t

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Beta function."""
        return t

    def beta_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Beta derivative."""
        return torch.ones_like(t)

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


class QuadraticDeterministicInterpolation(AffineGaussianPathMixin, nn.Module):
    """Quadratic deterministic interpolant."""

    def __init__(self) -> None:
        """Initialize quadratic interpolant."""
        super(QuadraticDeterministicInterpolation, self).__init__()

    def alpha_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Alpha derivative coefficient."""
        return -torch.ones_like(t)

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


class LinearStochasticInterpolation(AffineGaussianPathMixin, nn.Module):
    """Linear stochastic interpolant."""

    def __init__(
        self,
        gamma_multiplier: float = 1.0,
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
        return -1 * torch.ones_like(t)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Beta function."""
        return t

    def beta_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Beta derivative."""
        return torch.ones_like(t)

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """Gamma function."""
        return self.gamma_multiplier * (1 - t)

    def gamma_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Gamma derivative."""
        return -self.gamma_multiplier * torch.ones_like(t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Source scale sigma_tau = gamma_tau * sqrt(t) for SI."""
        return self.gamma(t) * torch.sqrt(t)

    def sigma_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Derivative of the SI source scale sigma_tau = gamma_tau * sqrt(t)."""
        sqrt_t = torch.sqrt(_clamp_time(t))
        return self.gamma_diff(t) * sqrt_t + self.gamma(t) / (2 * sqrt_t)

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


class QuadraticStochasticInterpolation(AffineGaussianPathMixin, nn.Module):
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
        return -1 * torch.ones_like(t)

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

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Source scale sigma_tau = gamma_tau * sqrt(t) for SI."""
        return self.gamma(t) * torch.sqrt(t)

    def sigma_diff(self, t: torch.Tensor) -> torch.Tensor:
        """Derivative of the SI source scale sigma_tau = gamma_tau * sqrt(t)."""
        sqrt_t = torch.sqrt(_clamp_time(t))
        return self.gamma_diff(t) * sqrt_t + self.gamma(t) / (2 * sqrt_t)

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
