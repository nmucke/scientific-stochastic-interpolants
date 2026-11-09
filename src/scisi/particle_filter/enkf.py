"""Stochastic (perturbed-observation) ensemble Kalman filter.

Model-agnostic: the forecast is any object exposing ``step(x, generator)`` on an
ensemble ``[E, d]`` and the observation model is linear and Gaussian,
``y = H x + eps``, ``eps ~ N(0, R)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


def gaspari_cohn(distance: torch.Tensor, radius: float) -> torch.Tensor:
    """Gaspari--Cohn fifth-order taper with compact support ``2 * radius``."""
    r = distance / radius
    near = (
        -0.25 * r**5 + 0.5 * r**4 + 0.625 * r**3 - (5.0 / 3.0) * r**2 + 1.0
    )
    far = (
        r**5 / 12.0
        - 0.5 * r**4
        + 0.625 * r**3
        + (5.0 / 3.0) * r**2
        - 5.0 * r
        + 4.0
        - (2.0 / 3.0) / r.clamp_min(1e-12)
    )
    return torch.where(r <= 1.0, near, torch.where(r < 2.0, far, torch.zeros_like(r)))


def periodic_taper(dim_state: int, radius: float, dtype=torch.float64) -> torch.Tensor:
    """``[d, d]`` Gaspari--Cohn taper on a periodic 1D lattice.

    Uses the chordal distance ``(d / pi) sin(pi |i - j| / d)`` of a ring of
    circumference ``d``, not the arc length ``min(|i - j|, d - |i - j|)``. The
    taper is positive definite for chordal distance on a circle, whereas arc length
    loses that once ``radius`` exceeds about ``d / 4``.

    The compact support ``2 * radius`` is therefore in chordal distance. Chords are
    shorter than arcs (equal only at zero; the antipodal site is at chord ``d / pi``
    against arc ``d / 2``), so the taper reaches further in sites than ``radius``
    suggests: it vanishes from arc length ``(d / pi) asin(2 pi radius / d)`` on, and
    only if ``radius <= d / (2 pi)``. For ``radius > d / (2 pi)`` every chord is
    below ``2 * radius``, so no entry is zero (at ``d = 16`` that is any radius above
    2.55).
    """
    index = torch.arange(dim_state, dtype=dtype)
    gap = (index[:, None] - index[None, :]).abs()
    chord = (dim_state / math.pi) * torch.sin(math.pi * gap / dim_state)
    return gaspari_cohn(chord, radius)


@dataclass
class EnKFResult:
    """Analysis ensembles ``[T, E, d]``."""

    analysis: torch.Tensor


class EnsembleKalmanFilter:
    """Stochastic EnKF with multiplicative inflation and optional localisation.

    Args:
        obs_matrix: ``H`` of shape ``[d_y, d]``.
        obs_variance: Scalar ``r`` with ``R = r I``.
        inflation: Multiplicative factor on forecast anomalies; ``1.0`` is none.
        localisation: Optional ``[d, d]`` taper applied by Schur product to the
            forecast covariance (see :func:`periodic_taper`).
    """

    def __init__(
        self,
        obs_matrix: torch.Tensor,
        obs_variance: float,
        inflation: float = 1.0,
        localisation: Optional[torch.Tensor] = None,
    ) -> None:
        self.H = obs_matrix
        self.obs_variance = float(obs_variance)
        self.inflation = float(inflation)
        self.localisation = localisation

    def analyse(
        self,
        forecast: torch.Tensor,
        observation: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Update a forecast ensemble ``[E, d]`` with one observation ``[d_y]``."""
        members = forecast.shape[0]
        H = self.H.to(forecast)
        mean = forecast.mean(0, keepdim=True)
        anomalies = self.inflation * (forecast - mean)
        forecast = mean + anomalies

        P = anomalies.T @ anomalies / (members - 1)
        if self.localisation is not None:
            P = P * self.localisation.to(P)
        PHt = P @ H.T
        S = H @ PHt + self.obs_variance * torch.eye(
            H.shape[0], dtype=forecast.dtype, device=forecast.device
        )
        eps = self.obs_variance**0.5 * torch.randn(
            members, H.shape[0], generator=generator, dtype=forecast.dtype
        )
        # Centre the perturbations so the mean update is the Kalman update.
        eps = eps - eps.mean(0, keepdim=True)
        innovation = observation.to(forecast) - forecast @ H.T - eps
        return forecast + torch.linalg.solve(S, innovation.T).T @ PHt.T

    def run(
        self,
        model,
        initial_ensemble: torch.Tensor,
        observations: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> EnKFResult:
        """Cycle forecast and analysis over ``observations`` ``[T, d_y]``."""
        x = initial_ensemble
        analysis = []
        for y in observations:
            x = self.analyse(model.step(x, generator), y, generator)
            analysis.append(x)
        return EnKFResult(analysis=torch.stack(analysis))


__all__ = [
    "EnKFResult",
    "EnsembleKalmanFilter",
    "gaspari_cohn",
    "periodic_taper",
]
