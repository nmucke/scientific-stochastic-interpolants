"""The analytic linear--Gaussian system + its two CLASSICAL reference filters.

The generative posterior samplers (Ours + the deep baselines) all run through the
canonical ``src/scisi`` posterior models -- only the closed-form problem definition
and the two conventional filters live here, because they have no generative-model
analogue in ``src/scisi``:

* :class:`GaussianSystem` -- the linear--Gaussian system ``x1|x0 ~ N(x0, I)``,
  ``y = x1 + e``, ``e ~ N(0, R)``, with the exact posterior in closed form (the
  KL / sliced-W2 reference).
* :func:`enkf_posterior` -- stochastic ensemble Kalman filter (Evensen).
* :func:`particle_filter_posterior` -- bootstrap particle filter (Carrassi).

Both filters are exact up to ensemble noise on this linear--Gaussian system and are
reported as the classical references in ``tab:analytical_results``. Everything else
that used to live in the (now-archived) ``samplers.py`` -- the hand-coded SI/FM/DM
integrators and the generative baselines -- was replaced by the ``src/scisi``
posteriors (see ``driver.draw_interpolant_posterior``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import torch

Tensor = torch.Tensor
# Optional nonlinear observation map h: [E, d] -> [E, N_y] (row-wise). ``None``
# keeps the original H = I code path bit-for-bit (the linear table's numbers).
ObsFn = Optional[Callable[[Tensor], Tensor]]


@dataclass(frozen=True)
class GaussianSystem:
    """The analytic linear--Gaussian system (spec Section 4).

    ``H = I``, ``R = obs_var I``; prior transition ``x1 | x0 ~ N(x0, I)``.
    """

    d: int
    obs_var: float = 1.0  # R = obs_var * I
    prior_var: float = 1.0  # Cov(x1 | x0) = prior_var * I  (=1 in the spec)

    def kalman_gain(self) -> float:
        """Scalar gain ``K`` (isotropic): ``K = c / (c + R)`` with ``c = prior_var``."""
        return self.prior_var / (self.prior_var + self.obs_var)

    def exact_posterior_moments(
        self, x0: Tensor, y: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Exact posterior mean and (isotropic) covariance scale.

        Returns ``(mean[d], cov_scale)`` with covariance ``cov_scale * I``.
        """
        K = self.kalman_gain()
        mean = x0 + K * (y - x0)  # H = I
        cov_scale = (1.0 - K) * self.prior_var
        return mean, torch.tensor(cov_scale, dtype=x0.dtype, device=x0.device)

    def exact_posterior_samples(
        self, x0: Tensor, y: Tensor, n: int, generator: torch.Generator
    ) -> Tensor:
        """Draw ``n`` exact-posterior samples ``[n, d]``."""
        mean, cov_scale = self.exact_posterior_moments(x0, y)
        z = torch.randn(n, self.d, generator=generator, dtype=x0.dtype)
        return mean.unsqueeze(0) + cov_scale.sqrt() * z


def enkf_posterior(
    sys: GaussianSystem,
    x0: Tensor,
    y: Tensor,
    *,
    ensemble_size: int,
    generator: torch.Generator,
    obs_fn: ObsFn = None,
) -> Tensor:
    """Stochastic ensemble Kalman filter (Evensen) on the analytic system.

    Forecast ensemble ``x_f ~ N(x0, I)`` (the known prior transition), perturbed
    observations ``y + e``, ``e ~ N(0, R)``, sample-covariance gain. With ``H = I``
    this is the textbook stochastic EnKF and is exact up to ensemble noise.

    With a nonlinear ``obs_fn = h`` it is the standard stochastic EnKF with
    statistical linearisation (Evensen 2003 / Houtekamer & Mitchell): the
    predicted observations are ``h(x_f^(e))`` per member, the gain is
    ``K = C_xh (C_hh + R I)^{-1}`` from the ensemble cross-/auto-covariances of
    ``(x_f, h(x_f))``, and ``x_a = x_f + K (y + e - h(x_f))``. Exact only for
    linear ``h``; for nonlinear ``h`` it is the classical Gaussian-update baseline.
    """
    d = sys.d
    R = sys.obs_var
    x_f = x0.unsqueeze(0) + sys.prior_var**0.5 * torch.randn(
        ensemble_size, d, generator=generator
    )
    if obs_fn is not None:
        hx = obs_fn(x_f)
        dx = x_f - x_f.mean(0, keepdim=True)
        dh = hx - hx.mean(0, keepdim=True)
        C_xh = dx.T @ dh / (ensemble_size - 1)
        C_hh = dh.T @ dh / (ensemble_size - 1)
        n_y = hx.shape[1]
        K = C_xh @ torch.linalg.inv(C_hh + R * torch.eye(n_y, dtype=x_f.dtype))
        y_pert = y.reshape(1, -1) + (R**0.5) * torch.randn(
            ensemble_size, n_y, generator=generator
        )
        return x_f + (y_pert - hx) @ K.T
    # Sample forecast covariance.
    xf_mean = x_f.mean(0, keepdim=True)
    P = (x_f - xf_mean).T @ (x_f - xf_mean) / (ensemble_size - 1)
    # H = I: gain K = P (P + R I)^{-1}.
    K = P @ torch.linalg.inv(P + R * torch.eye(d))
    y_pert = y.unsqueeze(0) + (R**0.5) * torch.randn(
        ensemble_size, d, generator=generator
    )
    innov = y_pert - x_f  # H x_f = x_f
    return x_f + innov @ K.T


def particle_filter_posterior(
    sys: GaussianSystem,
    x0: Tensor,
    y: Tensor,
    *,
    ensemble_size: int,
    generator: torch.Generator,
    obs_fn: ObsFn = None,
) -> Tensor:
    """Bootstrap particle filter (Carrassi) on the analytic system.

    Propose ``x ~ N(x0, I)`` (prior transition), weight by the Gaussian
    likelihood ``N(y; H x, R)``, and resample (systematic). Exact in the
    ``E -> inf`` limit; reported with the same ``E`` as the other methods.
    With ``obs_fn = h`` the weights are the exact nonlinear likelihood
    ``N(y; h(x), R)`` (the PF needs no linearisation).
    """
    d = sys.d
    R = sys.obs_var
    x = x0.unsqueeze(0) + sys.prior_var**0.5 * torch.randn(
        ensemble_size, d, generator=generator
    )
    if obs_fn is None:
        log_w = -0.5 * ((y.unsqueeze(0) - x) ** 2).sum(1) / R  # H = I
    else:
        log_w = -0.5 * ((y.reshape(1, -1) - obs_fn(x)) ** 2).sum(1) / R
    w = torch.softmax(log_w, dim=0)
    # Systematic resampling.
    positions = (
        torch.arange(ensemble_size) + torch.rand(1, generator=generator)
    ) / ensemble_size
    cumsum = torch.cumsum(w, dim=0)
    idx = torch.searchsorted(cumsum, positions.clamp(max=1.0 - 1e-7))
    return x[idx.clamp(max=ensemble_size - 1)]


__all__ = ["GaussianSystem", "enkf_posterior", "particle_filter_posterior"]
