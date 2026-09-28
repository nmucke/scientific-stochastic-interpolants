"""Closed-form analytical prior models for the linear-Gaussian case.

Implements the src/scisi model interface (drift / score) using exact Gaussian
math, so no network is trained.  All three model types (SI, FM, DM) and a
factory for the identity observation operator are provided.

State layout throughout: ``[B, 1, 1, d]``.
Time arrives as ``[B, 1]`` and is expanded to ``[B, 1, 1, 1]`` for broadcasting.
``field_history`` is ``[B, 1, 1, d, L]``; ``field_history[..., -1]`` = x0.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from scisi.likelihood_models.observation_operators import LinearObservationOperator
from scisi.models.diffusion_model import DenoiseDiffusionModel
from scisi.models.flow_matching_model import FlowMatchingModel
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
from scisi.models.interpolations import (
    LinearDeterministicInterpolation,
    LinearStochasticInterpolation,
)
from scisi.posterior_models.flow_matching_posterior import endpoint_vanishing_diffusion

# Clamp t away from the singular endpoints so denominators stay finite.
_T_MIN = 1e-4
_T_MAX = 1.0 - 1e-4


def _expand_t(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Expand ``t`` from ``[B, 1]`` to ``[B, 1, 1, 1]`` (x's rank)."""
    extra = x.dim() - 2
    return t.reshape(t.shape[0], 1, *([1] * extra))


# ---------------------------------------------------------------------------
# SI analytical drift model
# ---------------------------------------------------------------------------


class AnalyticalSIDriftModel(nn.Module):
    """Closed-form SI (Foellmer) prior drift for the Gaussian prior.

    The prior transition is ``x1 | x0 ~ N(x0, prior_var * I)`` and the
    interpolant is

        x_tau = alpha x0 + beta x1 + gamma W_tau,
        alpha = 1 - tau,  beta = tau,  gamma = g0 (1 - tau),

    giving marginal ``x_tau | x0 ~ N(x0, cbar I)`` with
    ``cbar = beta^2 prior_var + tau gamma^2``. The Foellmer SDE drift is

        b(x, t, x0) = alpha_d x0 + beta_d x0 + coef (x - x0),
        coef = (beta beta_d prior_var + t gamma gamma_d) / cbar,

    where ``alpha_d = -1``, ``beta_d = 1``, ``gamma_d = -g0``.
    Since ``alpha_d x0 + beta_d x0 = 0``, the drift simplifies to

        b = coef (x - x0).
    """

    def __init__(self, g0: float = 1.0, prior_var: float = 1.0) -> None:
        super().__init__()
        self.g0 = float(g0)
        self.prior_var = float(prior_var)
        # Dummy parameter so that nn.Module.parameters() is non-empty and
        # BasePosterior.device (which calls next(self.parameters())) works.
        self._device_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the SI Foellmer drift.

        Args:
            x: State ``[B, 1, 1, d]``.
            t: Pseudo-time ``[B, 1]``.
            field_history: ``[B, 1, 1, d, L]``; ``[..., -1]`` is x0.
            field_cond, pars_cond: Ignored (no conditioning).

        Returns:
            Drift tensor ``[B, 1, 1, d]``.
        """
        t_scalar = torch.clamp(t, _T_MIN, _T_MAX)
        t4 = _expand_t(t_scalar, x)  # [B, 1, 1, 1]

        x0 = field_history[..., -1]  # [B, 1, 1, d]

        beta = t4
        beta_d = torch.ones_like(t4)
        gamma = self.g0 * (1.0 - t4)
        gamma_d = -self.g0 * torch.ones_like(t4)

        cbar = beta ** 2 * self.prior_var + t4 * gamma ** 2
        coef = (beta * beta_d * self.prior_var + t4 * gamma * gamma_d) / cbar

        return coef * (x - x0)


# ---------------------------------------------------------------------------
# FM analytical velocity model
# ---------------------------------------------------------------------------


class AnalyticalFMVelocityModel(nn.Module):
    """Closed-form FM (rectified-flow) velocity for the Gaussian prior.

    The FM path is

        x_tau = alpha z + beta x1,
        alpha = 1 - tau,  beta = tau,  z ~ N(0, I),

    with target ``x1 ~ N(x0, prior_var I)``.  The marginal of ``x_tau`` is
    ``N(beta x0, var_path I)`` with ``var_path = alpha^2 + beta^2 prior_var``.

    The conditional moments given x_tau (and x0) are:

        mean_x1 = x0 + slope_x1 (x - beta x0),   slope_x1 = beta prior_var / var_path
        mean_z  = slope_z (x - beta x0),           slope_z  = alpha / var_path

    The FM velocity (alpha_d = -1, beta_d = 1):

        v = alpha_d mean_z + beta_d mean_x1 = -mean_z + mean_x1.
    """

    def __init__(self, prior_var: float = 1.0) -> None:
        super().__init__()
        self.prior_var = float(prior_var)
        # Dummy parameter so that nn.Module.parameters() is non-empty and
        # BasePosterior.device (which calls next(self.parameters())) works.
        self._device_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the FM velocity.

        Args:
            x: State ``[B, 1, 1, d]``.
            t: Pseudo-time ``[B, 1]``.
            field_history: ``[B, 1, 1, d, L]``; ``[..., -1]`` is x0.
            field_cond, pars_cond: Ignored.

        Returns:
            Velocity tensor ``[B, 1, 1, d]``.
        """
        t_scalar = torch.clamp(t, _T_MIN, _T_MAX)
        t4 = _expand_t(t_scalar, x)  # [B, 1, 1, 1]

        x0 = field_history[..., -1]  # [B, 1, 1, d]

        alpha = 1.0 - t4
        beta = t4
        var_path = alpha ** 2 + beta ** 2 * self.prior_var

        mean_xtau = beta * x0
        residual = x - mean_xtau

        slope_x1 = (beta * self.prior_var) / var_path
        slope_z = alpha / var_path

        mean_x1 = x0 + slope_x1 * residual
        mean_z = slope_z * residual

        # alpha_d = -1, beta_d = +1
        return -mean_z + mean_x1


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def build_si_model(
    d: int,
    g0: float = 1.0,
    prior_var: float = 1.0,
) -> FollmerStochasticInterpolant:
    """Build a FollmerStochasticInterpolant with the closed-form SI drift.

    Args:
        d: State dimension (spatial DOFs).
        g0: Diffusion-strength base ``gamma_tau = g0 (1 - tau)``.
        prior_var: Prior variance (``Cov(x1 | x0) = prior_var I``).

    Returns:
        A ``FollmerStochasticInterpolant`` wrapping the analytical SI drift.
    """
    interpolation = LinearStochasticInterpolation(gamma_multiplier=g0)
    drift_model = AnalyticalSIDriftModel(g0=g0, prior_var=prior_var)
    return FollmerStochasticInterpolant(
        interpolation=interpolation,
        drift_model=drift_model,
    )


def build_fm_model(
    d: int,
    prior_var: float = 1.0,
) -> FlowMatchingModel:
    """Build a FlowMatchingModel with the closed-form FM velocity.

    Args:
        d: State dimension (spatial DOFs).
        prior_var: Prior variance (``Cov(x1 | x0) = prior_var I``).

    Returns:
        A ``FlowMatchingModel`` wrapping the analytical FM velocity.
    """
    interpolation = LinearDeterministicInterpolation()
    drift_model = AnalyticalFMVelocityModel(prior_var=prior_var)
    return FlowMatchingModel(
        interpolation=interpolation,
        drift_model=drift_model,
    )


def build_dm_model(
    d: int,
    prior_var: float = 1.0,
    g0: float = 1.0,
) -> DenoiseDiffusionModel:
    """Build a DenoiseDiffusionModel from the closed-form FM velocity model.

    Constructs the FM model first, then wraps it in a velocity-mode
    ``DenoiseDiffusionModel`` with the endpoint-vanishing diffusion
    ``g_tau = g0 sqrt(alpha beta)``.

    Args:
        d: State dimension.
        prior_var: Prior variance.
        g0: Scale for the endpoint-vanishing diffusion coefficient.

    Returns:
        A velocity-mode ``DenoiseDiffusionModel``.
    """
    fm_model = build_fm_model(d, prior_var=prior_var)
    diffusion_term = endpoint_vanishing_diffusion(fm_model.interpolation, scale=g0)
    return DenoiseDiffusionModel.from_flow_matching(
        fm_model,
        diffusion_term=diffusion_term,
    )


def build_identity_obs_operator(d: int) -> LinearObservationOperator:
    """Build a ``LinearObservationOperator`` that is the identity (H = I).

    Uses the ``"random"`` type with ``percent_obs=1.0`` and ``seed=0``:
    this selects ALL ``d`` DOFs in sorted order (deterministic), giving
    ``H = I_d``.

    Args:
        d: State dimension (number of observed DOFs).

    Returns:
        A ``LinearObservationOperator`` with ``obs_matrix = I_d``.
    """
    return LinearObservationOperator(
        type="random",
        data_size=(1, 1, d),
        percent_obs=1.0,
        seed=0,
    )


__all__ = [
    "AnalyticalSIDriftModel",
    "AnalyticalFMVelocityModel",
    "build_si_model",
    "build_fm_model",
    "build_dm_model",
    "build_identity_obs_operator",
]
