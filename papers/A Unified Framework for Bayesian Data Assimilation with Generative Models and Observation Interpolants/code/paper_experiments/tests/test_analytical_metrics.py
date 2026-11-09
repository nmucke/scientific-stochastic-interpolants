"""Pin ``cases/analytical/_metrics.py`` to the archived ``analytical_utils``.

The expected values were computed on 2026-11-09 with
``archive/paper/scripts/analytical_utils/kl_divergence.py`` (numpy 2.2.6,
scipy 1.16.3) on the inputs below. Exact equality: the functions were moved, not
changed, so any difference means the analytical numbers would move too.
"""

import numpy as np
import pytest

from cases.analytical._metrics import gaussian_kl_divergence, wasserstein_distance

# d -> (gaussian KL, sliced W default, sliced W with 16 projections and seed 3)
ARCHIVED = {
    1: (0.09067379321483335, 0.4011236315070383, 0.4011236315070383),
    2: (0.10017342423155724, 0.25881094776264096, 0.21867911977566065),
    5: (0.3427745277209387, 0.2987024305458203, 0.26712896058930646),
}


def _samples(d):
    rng = np.random.default_rng(1234 + d)
    x = rng.normal(size=(2000, d))
    mix = np.eye(d) + 0.2 * np.tril(np.ones((d, d)), -1)
    y = 0.3 + 1.2 * rng.normal(size=(1500, d)) @ mix
    return x, y


@pytest.mark.parametrize("d", sorted(ARCHIVED))
def test_matches_archived_analytical_utils(d):
    x, y = _samples(d)
    kl, w_default, w_custom = ARCHIVED[d]
    assert gaussian_kl_divergence(x, y) == kl
    assert wasserstein_distance(x, y) == w_default
    assert wasserstein_distance(x, y, num_projections=16, seed=3) == w_custom


def test_driver_uses_case_metrics():
    from cases.analytical import _metrics, driver

    assert driver.gaussian_kl_divergence is _metrics.gaussian_kl_divergence
    assert driver.wasserstein_distance is _metrics.wasserstein_distance
