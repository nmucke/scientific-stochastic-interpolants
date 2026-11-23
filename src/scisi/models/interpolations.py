"""Interpolants.

This module contains the interpolant models for the scisi package.
"""

import pdb
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from scisi.sampling.noise_covariance import IdentityCovariance, NoiseCovariance

# Clamp pseudo-time away from the endpoints {0, 1} when evaluating
# schedule-derived quantities whose denominators vanish there.
MIN_TIME = 1e-4


def _clamp_time(t: torch.Tensor) -> torch.Tensor:
    """Clamp pseudo-time away from the singular endpoints 0 and 1."""
    return torch.clamp(t, min=MIN_TIME, max=1.0 - MIN_TIME)


# Gauss-Legendre nodes of the variance-clock quadrature; exact for polynomial
# gamma**2 up to degree 2 * QUADRATURE_NODES - 1.
QUADRATURE_NODES = 64


def affine_gamma_variance_clock(gamma_multiplier: float, s, t):
    """q(t) - q(s) = int_s^t gamma_r^2 dr for gamma_r = a (1 - r).

    Closed form a^2 [(1 - s)^3 - (1 - t)^3] / 3, factored as a difference of
    cubes so that it stays accurate on short intervals. Works for floats and
    tensors alike.
    """
    left, right = 1.0 - s, 1.0 - t
    return gamma_multiplier**2 * (t - s) * (left * left + left * right + right * right) / 3.0


def quadrature_variance_clock(gamma: Callable, s, t, num_nodes: int = QUADRATURE_NODES):
    """q(t) - q(s) = int_s^t gamma_r^2 dr by Gauss-Legendre quadrature.

    Fallback for any deterministic gamma (additive, state-independent
    diffusion). ``s`` and ``t`` are floats or broadcastable tensors; floats in
    give a float out, tensors give a float64 tensor.
    """
    as_float = not torch.is_tensor(s) and not torch.is_tensor(t)
    s = torch.as_tensor(s, dtype=torch.float64)
    t = torch.as_tensor(t, dtype=torch.float64)
    nodes, weights = (torch.from_numpy(a) for a in np.polynomial.legendre.leggauss(num_nodes))
    half = 0.5 * (t - s)
    r = (s + t)[..., None] * 0.5 + half[..., None] * nodes  # [..., nodes] in [s, t]
    value = half * (weights * gamma(r) ** 2).sum(-1)
    return float(value) if as_float else value


def matched_beta_quadrature(gamma: Callable, gamma_diff: Callable, t):
    """Matched schedule beta_t and beta'_t for any decreasing gamma (WP1).

    beta_t^2 = (q(t) - t gamma_t^2) / q(1) and beta'_t = -t gamma_t gamma'_t /
    (beta_t q(1)); q(t) - t gamma_t^2 = -2 int_0^t r gamma gamma' dr >= 0 when
    gamma decreases. With C = Sigma / q(1) this makes Var(I_t | h) = q(t) C,
    so a Gaussian target has a path-independent drift. ``t`` is a tensor.
    """
    total = quadrature_variance_clock(gamma, 0.0, 1.0)

    def beta_of(time):
        excess = quadrature_variance_clock(gamma, torch.zeros_like(time), time)
        return ((excess - time * gamma(time) ** 2).clamp_min(0.0) / total).sqrt()

    t64 = t.to(torch.float64)
    beta = beta_of(t64)
    # beta' is finite at t = 0 (0/0 in this form): evaluate it at clamped time.
    clamped = _clamp_time(t64)
    beta_diff = -clamped * gamma(clamped) * gamma_diff(clamped) / (
        beta_of(clamped) * total
    )
    return beta.to(t.dtype), beta_diff.to(t.dtype)


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
        covariance = getattr(self, "noise_covariance", None)
        if covariance is not None and not covariance.is_identity:
            raise NotImplementedError(
                "The velocity-score identities assume white interpolant noise."
            )
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


class AdditiveNoiseMixin:
    """Noise covariance and variance clock of an additive-noise interpolant.

    I_t = alpha_t c + beta_t y + gamma_t sqrt(t) C^{1/2} z with white z: the
    interpolant colours the white ``noise`` it is given with the
    ``noise_covariance`` C it holds, so the model, the trainer and the
    samplers all read C from one place. Assumption: additive,
    state-independent diffusion with deterministic gamma.
    """

    def _set_noise_covariance(self, noise_covariance: Optional[NoiseCovariance]):
        self.noise_covariance = (
            IdentityCovariance() if noise_covariance is None else noise_covariance
        )

    def variance_clock(self, s, t):
        """q(t) - q(s) = int_s^t gamma_r^2 dr, so Var(M_t - M_s) = (q(t) - q(s)) C.

        Default: quadrature over ``self.gamma``; classes with gamma = a(1 - t)
        override it with the closed form.
        """
        return quadrature_variance_clock(self.gamma, s, t)


class LinearStochasticInterpolation(
    AdditiveNoiseMixin, AffineGaussianPathMixin, nn.Module
):
    """Linear stochastic interpolant."""

    def __init__(
        self,
        gamma_multiplier: float = 1.0,
        wiener_process: bool = True,
        noise_covariance: Optional[NoiseCovariance] = None,
    ) -> None:
        """Initialize linear stochastic interpolant (C = Id unless given)."""
        super(LinearStochasticInterpolation, self).__init__()
        self.gamma_multiplier = gamma_multiplier
        self._set_noise_covariance(noise_covariance)

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

    def variance_clock(self, s, t):
        """q(t) - q(s) for gamma = a(1 - t): a^2 [(1 - s)^3 - (1 - t)^3] / 3."""
        return affine_gamma_variance_clock(self.gamma_multiplier, s, t)

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
            + self.gamma(t) * self.transform_noise(self.noise_covariance.sqrt(noise), t)
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
            + self.gamma_diff(t)
            * self.transform_noise(self.noise_covariance.sqrt(noise), t)
        )


class QuadraticStochasticInterpolation(
    AdditiveNoiseMixin, AffineGaussianPathMixin, nn.Module
):
    """Quadratic stochastic interpolant."""

    def __init__(
        self,
        gamma_multiplier: float = 0.1,
        wiener_process: bool = True,
        noise_covariance: Optional[NoiseCovariance] = None,
    ) -> None:
        """Initialize quadratic stochastic interpolant (C = Id unless given)."""
        super(QuadraticStochasticInterpolation, self).__init__()
        self.gamma_multiplier = gamma_multiplier
        self._set_noise_covariance(noise_covariance)

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

    def variance_clock(self, s, t):
        """q(t) - q(s) for gamma = a(1 - t): a^2 [(1 - s)^3 - (1 - t)^3] / 3."""
        return affine_gamma_variance_clock(self.gamma_multiplier, s, t)

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
            + self.gamma(t) * self.transform_noise(self.noise_covariance.sqrt(noise), t)
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
            + self.gamma_diff(t)
            * self.transform_noise(self.noise_covariance.sqrt(noise), t)
        )


class MatchedStochasticInterpolation(LinearStochasticInterpolation):
    """Stochastic interpolant with the matched schedule beta_t = t sqrt(3 - 2t).

    alpha_t = 1 - t, gamma_t = a(1 - t) and beta_t^2 = (q(t) - t gamma_t^2) /
    q(1), which for this gamma is t^2 (3 - 2t) for every a > 0 (WP1). If the
    target is Gaussian, y | h ~ N(m, Sigma), and C = Sigma / q(1), then
    Cov(R_t, I_t | h) = 0: the drift is alpha'_t c + beta'_t m for every x and
    the Itô map is exact at any step count without path features.
    ``matched_beta_quadrature`` gives the same schedule for other gammas.
    """

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """beta_t = t sqrt(3 - 2t)."""
        return t * torch.sqrt(3 - 2 * t)

    def beta_diff(self, t: torch.Tensor) -> torch.Tensor:
        """beta'_t = 3 (1 - t) / sqrt(3 - 2t)."""
        return 3 * (1 - t) / torch.sqrt(3 - 2 * t)
