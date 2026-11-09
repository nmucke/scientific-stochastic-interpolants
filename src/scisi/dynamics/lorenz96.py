"""Lorenz 96 solver, deterministic and with additive model noise.

The deterministic system on the periodic lattice ``i = 0..d-1`` is

    dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F.

:class:`StochasticLorenz96` turns this into a discrete-time Markov chain with one
step per assimilation interval ``interval``. The interval is cut into
``noise_stages`` equal segments; each segment is integrated with RK4 and is
followed by an additive Gaussian kick, so that

    x^{k+1} = z(x^k, xi_1 .. xi_{n-1}) + sqrt(q_stage) xi_n,   q_stage = noise_std^2 / n.

``noise_std^2`` is the SUM of the kick variances, whatever ``noise_stages`` is. It is
the one-step conditional variance only for ``noise_stages = 1``, which gives the
exactly Gaussian kernel ``N(Phi_interval(x), noise_std^2 I)``. For
``noise_stages > 1`` earlier kicks are amplified by the flow, so the one-step
variance exceeds the budget and stage counts are not noise-matched. Those kicks
also make the kernel non-Gaussian, but only weakly at short intervals. As
``noise_stages`` grows towards the number of RK4 steps the chain approaches the
additive-noise SDE. In every case the LAST kick is Gaussian and enters additively,
which is what lets the particle filter integrate it out exactly (see
:mod:`scisi.particle_filter.bootstrap`).

Every function works on any tensor layout: ``dim`` selects the lattice axis and
all other axes are vectorised ensemble axes. The particle filter uses ``dim=0``
(state-major ``[d, N]``), which keeps the rolls contiguous for large ``N``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


def lorenz96_rhs(x: torch.Tensor, forcing: float, dim: int = -1) -> torch.Tensor:
    """Evaluate the Lorenz 96 vector field along lattice axis ``dim``."""
    x_p1 = torch.roll(x, -1, dims=dim)
    x_m1 = torch.roll(x, 1, dims=dim)
    x_m2 = torch.roll(x, 2, dims=dim)
    return (x_p1 - x_m2) * x_m1 - x + forcing


def rk4_step(x: torch.Tensor, dt: float, forcing: float, dim: int = -1) -> torch.Tensor:
    """Advance ``x`` by one classical fourth-order Runge--Kutta step."""
    k1 = lorenz96_rhs(x, forcing, dim)
    k2 = lorenz96_rhs(x + 0.5 * dt * k1, forcing, dim)
    k3 = lorenz96_rhs(x + 0.5 * dt * k2, forcing, dim)
    k4 = lorenz96_rhs(x + dt * k3, forcing, dim)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(
    x: torch.Tensor, num_steps: int, dt: float, forcing: float, dim: int = -1
) -> torch.Tensor:
    """Apply ``num_steps`` RK4 steps."""
    for _ in range(num_steps):
        x = rk4_step(x, dt, forcing, dim)
    return x


def integrate_state_major(
    x: torch.Tensor, num_steps: int, dt: float, forcing: float
) -> torch.Tensor:
    """Allocation-free RK4 for a state-major ensemble ``[d, n]``.

    Same arithmetic as :func:`integrate` with ``dim=0``. Temporaries dominate the
    cost of the plain version and do not parallelise (allocation contention), so
    the particle filter uses this path. Each right-hand side reads a periodically
    padded buffer instead of rolling.
    """
    d = x.shape[0]
    pad = x.new_empty(d + 3, x.shape[1])  # rows: x_{-2}, x_{-1}, x_0..x_{d-1}, x_d
    state = pad[2 : d + 2]
    k = torch.empty_like(x)
    total = torch.empty_like(x)
    start = x.clone()

    def rhs(out: torch.Tensor) -> None:
        pad[0:2] = pad[d : d + 2]
        pad[d + 2] = pad[2]
        torch.sub(pad[3 : d + 3], pad[0:d], out=out)
        out.mul_(pad[1 : d + 1]).sub_(state).add_(forcing)

    for _ in range(num_steps):
        state.copy_(start)
        rhs(k)
        total.copy_(k)
        torch.add(start, k, alpha=0.5 * dt, out=state)
        rhs(k)
        total.add_(k, alpha=2.0)
        torch.add(start, k, alpha=0.5 * dt, out=state)
        rhs(k)
        total.add_(k, alpha=2.0)
        torch.add(start, k, alpha=dt, out=state)
        rhs(k)
        total.add_(k)
        start.add_(total, alpha=dt / 6.0)
    return start


@dataclass(frozen=True)
class StochasticLorenz96:
    """One assimilation interval of Lorenz 96 with additive Gaussian model noise."""

    dim_state: int = 40
    forcing: float = 8.0
    dt: float = 0.01
    interval: float = 0.05
    noise_std: float = 0.0
    noise_stages: int = 1

    def __post_init__(self) -> None:
        steps = self.interval / self.dt
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError(f"interval {self.interval} is not a multiple of dt {self.dt}")
        if int(round(steps)) % self.noise_stages != 0:
            raise ValueError(
                f"{int(round(steps))} RK4 steps cannot be split into "
                f"{self.noise_stages} equal noise stages"
            )

    @property
    def steps_per_stage(self) -> int:
        return int(round(self.interval / self.dt)) // self.noise_stages

    @property
    def stage_variance(self) -> float:
        """Variance of one additive kick, including the final one."""
        return self.noise_std**2 / self.noise_stages

    def forecast_to_last_kick(
        self,
        x: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        dim: int = -1,
    ) -> torch.Tensor:
        """Propagate through the interval, stopping just before the last kick."""
        std = self.stage_variance**0.5
        fast = dim == 0 and x.dim() == 2
        for stage in range(self.noise_stages):
            if fast:
                x = integrate_state_major(x, self.steps_per_stage, self.dt, self.forcing)
            else:
                x = integrate(x, self.steps_per_stage, self.dt, self.forcing, dim)
            if stage < self.noise_stages - 1 and std > 0.0:
                x = x + std * _randn_like(x, generator)
        return x

    def step(
        self,
        x: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        dim: int = -1,
    ) -> torch.Tensor:
        """Draw ``x^{k+1} ~ p(. | x^k)``."""
        z = self.forecast_to_last_kick(x, generator, dim)
        if self.stage_variance > 0.0:
            z = z + self.stage_variance**0.5 * _randn_like(z, generator)
        return z

    def spin_up(
        self,
        num_members: int,
        num_intervals: int,
        generator: Optional[torch.Generator] = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Draw ``[num_members, d]`` states from near the invariant measure."""
        x = self.forcing + torch.randn(
            num_members, self.dim_state, generator=generator, dtype=dtype
        )
        for _ in range(num_intervals):
            x = self.step(x, generator)
        return x

    def trajectory(
        self,
        x0: torch.Tensor,
        num_intervals: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Return ``[..., num_intervals + 1, d]`` starting from ``x0``."""
        states = [x0]
        for _ in range(num_intervals):
            states.append(self.step(states[-1], generator))
        return torch.stack(states, dim=-2)


def _randn_like(x: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    return torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)


__all__ = [
    "StochasticLorenz96",
    "integrate",
    "integrate_state_major",
    "lorenz96_rhs",
    "rk4_step",
]
