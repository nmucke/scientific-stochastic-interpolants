"""Evaluation metrics for the observation-interpolant DA experiments.

Exact metric definitions follow ``archive/paper_new/EXPERIMENTS_IMPLEMENTATION_SPEC.md``
Section 3. Conventions:

* The ensemble dimension is the leading axis ``E`` of an ensemble tensor.
* Spatial metrics accept an optional boolean ``mask`` over the per-point
  dimensions where ``True`` keeps a cell and ``False`` excludes it (solid
  cells).
"""

from scisi.metrics.accuracy import ensemble_mean_rmse
from scisi.metrics.calibration import (
    crps,
    plot_rank_histogram,
    rank_histogram,
    spread_skill,
)
from scisi.metrics.cost import NFECounter, StepTimer
from scisi.metrics.distributional import (
    gaussian_kl_1d,
    kde_kl_1d,
    kl_at_points,
    sliced_wasserstein_w2,
)
from scisi.metrics.spectral import (
    energy_spectrum_rmse,
    radial_kinetic_energy_spectrum,
)

__all__ = [
    "ensemble_mean_rmse",
    "energy_spectrum_rmse",
    "radial_kinetic_energy_spectrum",
    "crps",
    "spread_skill",
    "rank_histogram",
    "plot_rank_histogram",
    "gaussian_kl_1d",
    "kde_kl_1d",
    "kl_at_points",
    "sliced_wasserstein_w2",
    "NFECounter",
    "StepTimer",
]
