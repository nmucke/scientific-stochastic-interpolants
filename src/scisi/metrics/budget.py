"""Label-free quadratic budgets for additively forced dissipative systems.

Setting. ``du = (N(u) - D u) dtau + eps dxi`` with ``D`` linear, self-adjoint
and non-negative, ``xi`` a Wiener process with covariance ``Q``, and a
self-adjoint weight ``W`` that commutes with ``D`` and is conserved by the
nonlinearity: ``<u, W N(u)> = 0``. For ``q(u) = 1/2 <u, W u>`` Ito's formula
gives

    d q = ( -<u, W D u> + 1/2 eps^2 tr(W Q) ) dtau + eps <u, W dxi>.

Integrating over one physical step ``Delta`` from ``c`` to ``y`` and using the
trapezoid rule for the dissipation integral (local error ``O(Delta^3)``):

    q(y) - q(c) = -(Delta/2) [<c, W D c> + <y, W D y>] + Delta I + martingale,

with injection rate ``I = 1/2 eps^2 tr(W Q)``. Collecting terms with
``q_pm(x) = 1/2 <x, W (Id +- Delta D) x>`` gives the identity

    E[ q_+(y) | history ] = q_-(c) + Delta I,

so the residual ``r = q_+(y) - q_-(c) - Delta I`` is (to trapezoid accuracy)
a martingale difference: mean zero given the history, whatever the model of
the nonlinearity. Its conditional variance is about
``Delta eps^2 <u, W Q W u>``.

Nothing here assumes a grid, periodicity, a channel count or physical
constants: the caller supplies ``W``, ``D``, the inner product and ``I``.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

Operator = Callable[[torch.Tensor], torch.Tensor]


def channel_sum_spatial_mean(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Default inner product for ``[B, C, *spatial]``: mean over space, sum over C.

    Returns ``[B]``. This is the area-mean (domain-size free) inner product;
    the injection rate must be expressed in the same convention.
    """
    if u.ndim < 2:
        raise ValueError("Expected [B, C, *spatial] tensors")
    product = (u * v).flatten(2)
    return product.mean(-1).sum(-1)


class QuadraticBudget:
    """One-step budget of ``q_W(u) = 1/2 <u, W u>`` (trapezoid form).

    Assumption: additive, state-independent forcing; ``D`` linear,
    self-adjoint, non-negative; ``W`` self-adjoint, commuting with ``D`` and
    conserved by the nonlinearity. ``residual`` is then a martingale
    difference up to the ``O(Delta^3)`` trapezoid error.

    Args:
        weight: linear operator ``W`` on ``[B, C, *spatial]``; ``None`` = identity.
        dissipation: linear operator ``D`` on ``[B, C, *spatial]``.
        injection_rate: ``1/2 eps^2 tr(W Q)`` per unit physical time, in the
            convention of ``inner``.
        step: physical time ``Delta`` between ``y_prev`` and ``y_next``.
        inner: ``(u, v) -> [B]``; default :func:`channel_sum_spatial_mean`.
        forcing_quadratic_variation: optional ``u -> eps^2 <u, W Q W u>``
            (``[B]``), used by :meth:`martingale_variance`.

    All inputs must be in physical (denormalised) units.
    """

    def __init__(
        self,
        weight: Optional[Operator],
        dissipation: Operator,
        injection_rate: float,
        step: float,
        inner: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        forcing_quadratic_variation: Optional[Operator] = None,
    ) -> None:
        if not step > 0:
            raise ValueError("step must be positive")
        self.weight = weight if weight is not None else (lambda u: u)
        self.dissipation = dissipation
        self.injection_rate = float(injection_rate)
        self.step = float(step)
        self.inner = inner if inner is not None else channel_sum_spatial_mean
        self.forcing_quadratic_variation = forcing_quadratic_variation

    @property
    def injection(self) -> float:
        """Expected injection over one step, ``Delta * I``."""
        return self.injection_rate * self.step

    def quadratic(self, x: torch.Tensor) -> torch.Tensor:
        """``q(x) = 1/2 <x, W x>`` per sample, ``[B]``."""
        return 0.5 * self.inner(x, self.weight(x))

    def dissipation_rate(self, x: torch.Tensor) -> torch.Tensor:
        """``<x, W D x>`` per sample, ``[B]``."""
        return self.inner(x, self.weight(self.dissipation(x)))

    def q_plus(self, x: torch.Tensor) -> torch.Tensor:
        """``1/2 <x, W (Id + Delta D) x>``."""
        return self.quadratic(x) + 0.5 * self.step * self.dissipation_rate(x)

    def q_minus(self, x: torch.Tensor) -> torch.Tensor:
        """``1/2 <x, W (Id - Delta D) x>``."""
        return self.quadratic(x) - 0.5 * self.step * self.dissipation_rate(x)

    def residual(self, y_next: torch.Tensor, y_prev: torch.Tensor) -> torch.Tensor:
        """``q_+(y_next) - q_-(y_prev) - Delta I`` per sample, ``[B]``.

        Mean zero given the history for draws from the true one-step law.
        A persistently negative mean means too little energy in ``y_next``
        (for example an underdispersed forecast).
        """
        if y_next.shape != y_prev.shape:
            raise ValueError("y_next and y_prev must have the same shape")
        return self.q_plus(y_next) - self.q_minus(y_prev) - self.injection

    def implied_spread(self, mean_next: torch.Tensor, y_prev: torch.Tensor) -> torch.Tensor:
        """``tr(W (Id + Delta D) Cov(y | h))`` implied by the budget, ``[B]``.

        From ``E q_+(y) = q_+(E y) + 1/2 tr(W (Id + Delta D) Cov)``: twice
        ``q_-(c) + Delta I - q_+(E[y | h])``. Needs the conditional mean.
        """
        return 2.0 * (self.q_minus(y_prev) + self.injection - self.q_plus(mean_next))

    def martingale_variance(self, y_prev: torch.Tensor) -> torch.Tensor:
        """Approximate ``Var(residual | h) = Delta eps^2 <c, W Q W c>``, ``[B]``."""
        if self.forcing_quadratic_variation is None:
            raise ValueError("forcing_quadratic_variation was not supplied")
        return self.step * self.forcing_quadratic_variation(y_prev)


def u_statistic(residuals: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Unbiased estimate of ``(E[r | h])^2`` from ``E >= 2`` i.i.d. members.

    ``(1 / (E (E - 1))) sum_{e != e'} r_e r_e'`` along ``dim``. Unlike
    ``mean(r)^2`` it carries no ``Var(r) / E`` bias, so it does not penalise
    spread. It can be negative.
    """
    count = residuals.shape[dim]
    if count < 2:
        raise ValueError("Need at least two members")
    total = residuals.sum(dim)
    squares = residuals.square().sum(dim)
    return (total.square() - squares) / (count * (count - 1))
