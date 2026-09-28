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

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


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
) -> Tensor:
    """Stochastic ensemble Kalman filter (Evensen) on the analytic system.

    Forecast ensemble ``x_f ~ N(x0, I)`` (the known prior transition), perturbed
    observations ``y + e``, ``e ~ N(0, R)``, sample-covariance gain. With ``H = I``
    this is the textbook stochastic EnKF and is exact up to ensemble noise.
    """
    d = sys.d
    R = sys.obs_var
    x_f = x0.unsqueeze(0) + sys.prior_var**0.5 * torch.randn(
        ensemble_size, d, generator=generator
    )
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
) -> Tensor:
    """Bootstrap particle filter (Carrassi) on the analytic system.

    Propose ``x ~ N(x0, I)`` (prior transition), weight by the Gaussian
    likelihood ``N(y; H x, R)``, and resample (systematic). Exact in the
    ``E -> inf`` limit; reported with the same ``E`` as the other methods.
    """
    d = sys.d
    R = sys.obs_var
    x = x0.unsqueeze(0) + sys.prior_var**0.5 * torch.randn(
        ensemble_size, d, generator=generator
    )
    log_w = -0.5 * ((y.unsqueeze(0) - x) ** 2).sum(1) / R  # H = I
    w = torch.softmax(log_w, dim=0)
    # Systematic resampling.
    positions = (
        torch.arange(ensemble_size) + torch.rand(1, generator=generator)
    ) / ensemble_size
    cumsum = torch.cumsum(w, dim=0)
    idx = torch.searchsorted(cumsum, positions.clamp(max=1.0 - 1e-7))
    return x[idx.clamp(max=ensemble_size - 1)]


__all__ = ["GaussianSystem", "enkf_posterior", "particle_filter_posterior"]
