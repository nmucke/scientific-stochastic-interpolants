"""The nonparametric distances of the Lorenz 96 scores (``_l96_divergence``)."""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cases.lorenz96._l96_divergence import energy_distance, knn_kl  # noqa: E402


def test_knn_kl_recovers_a_known_gaussian_kl():
    gen = torch.Generator().manual_seed(0)
    d = 2
    x = torch.randn(20000, d, generator=gen, dtype=torch.float64)
    y = 0.5 + 1.5 * torch.randn(20000, d, generator=gen, dtype=torch.float64)
    # KL(N(0, I) || N(0.5, 1.5^2 I)) in closed form
    s2 = 1.5**2
    exact = 0.5 * d * (1 / s2 + 0.25 / s2 - 1 + math.log(s2))
    assert abs(knn_kl(x, y) - exact) < 0.05


def test_knn_kl_near_zero_for_equal_laws_and_large_samples():
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(20000, 3, generator=gen, dtype=torch.float64)
    y = torch.randn(20000, 3, generator=gen, dtype=torch.float64)
    assert abs(knn_kl(x, y)) < 0.05


def test_knn_kl_duplicates_in_the_ensemble_stay_finite_but_not_in_x():
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(500, 4, generator=gen, dtype=torch.float64)
    y = torch.randn(64, 4, generator=gen, dtype=torch.float64)
    y[1] = y[0]  # SMC resampling duplicates members of the ensemble (y side)
    assert math.isfinite(knn_kl(x, y))
    x[1] = x[0]
    assert math.isnan(knn_kl(x, y))


def test_energy_distance_zero_for_equal_laws_positive_otherwise():
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(3000, 4, generator=gen, dtype=torch.float64)
    y = torch.randn(3000, 4, generator=gen, dtype=torch.float64)
    z = 1.0 + torch.randn(3000, 4, generator=gen, dtype=torch.float64)
    assert abs(energy_distance(x, y)) < 0.01
    assert energy_distance(x, z) > 0.5
    assert math.isnan(energy_distance(x[:1], y))
