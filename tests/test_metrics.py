"""Unit tests for ``scisi.metrics`` (spec Section 3 metric definitions).

Sanity criteria asserted here:
* spread-skill of a perfectly calibrated synthetic ensemble ~ 1;
* rank histogram of a calibrated ensemble ~ flat;
* CRPS of a deterministic perfect forecast ~ 0;
* sliced-W2 between identical sets ~ 0.
"""

import numpy as np
import pytest
import torch

from scisi.metrics import (
    NFECounter,
    StepTimer,
    crps,
    energy_spectrum_rmse,
    ensemble_mean_rmse,
    gaussian_kl_1d,
    kde_kl_1d,
    kl_at_points,
    plot_rank_histogram,
    rank_histogram,
    sliced_wasserstein_w2,
    spread_skill,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


# --------------------------------------------------------------------------- #
# Accuracy
# --------------------------------------------------------------------------- #
def test_ensemble_mean_rmse_tight_ensemble():
    truth = torch.randn(16, 16)
    ens = truth.unsqueeze(0) + 0.01 * torch.randn(64, 16, 16)
    assert ensemble_mean_rmse(ens, truth).item() < 0.01


def test_ensemble_mean_rmse_is_mean_not_per_member():
    truth = torch.randn(16, 16)
    ens = truth.unsqueeze(0) + torch.randn(64, 16, 16)
    mean_rmse = ensemble_mean_rmse(ens, truth).item()
    per_member = torch.stack(
        [torch.sqrt(((ens[m] - truth) ** 2).mean()) for m in range(64)]
    ).mean().item()
    # RMSE of the mean is much smaller than the mean of per-member RMSEs.
    assert mean_rmse < 0.5 * per_member


def test_ensemble_mean_rmse_mask():
    truth = torch.randn(8, 8)
    ens = truth.unsqueeze(0) + torch.randn(64, 8, 8)
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[:, :4] = True
    val = ensemble_mean_rmse(ens, truth, mask=mask)
    assert torch.isfinite(val)


def test_energy_spectrum_rmse_identical_zero():
    field = torch.randn(128, 128)
    assert energy_spectrum_rmse(field, field).item() < 1e-9


def test_energy_spectrum_rmse_positive():
    val = energy_spectrum_rmse(torch.randn(128, 128), torch.randn(128, 128))
    assert val.item() > 0.0


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def test_crps_deterministic_perfect_zero():
    truth = torch.randn(16, 16)
    det = truth.unsqueeze(0).repeat(32, 1, 1)
    assert abs(crps(det, truth).item()) < 1e-6


def test_crps_unbiased_estimator_standard_normal():
    # E[CRPS] for N(0,1) forecast & truth is 1/sqrt(pi).
    ens = torch.randn(2000, 5000)
    truth = torch.randn(5000)
    assert abs(crps(ens, truth).item() - 1.0 / np.sqrt(np.pi)) < 0.02


def test_spread_skill_calibrated_ratio_one():
    ens = torch.randn(256, 4000)
    truth = torch.randn(4000)
    ss = spread_skill(ens, truth)
    assert abs(ss["ratio"].item() - 1.0) < 0.05
    assert ss["deviation"].item() < 0.05


def test_spread_skill_under_dispersed():
    ss = spread_skill(0.3 * torch.randn(256, 4000), torch.randn(4000))
    assert ss["ratio"].item() < 0.9


def test_rank_histogram_calibrated_flat():
    E = 20
    ens = torch.randn(E, 20000)
    truth = torch.randn(20000)
    counts = rank_histogram(ens, truth)
    assert len(counts) == E + 1
    expected = counts.sum().item() / (E + 1)
    assert (counts.float() - expected).abs().max().item() / expected < 0.1


def test_rank_histogram_under_dispersed_u_shape():
    E = 20
    counts = rank_histogram(0.3 * torch.randn(E, 20000), torch.randn(20000)).float()
    assert (counts[0] + counts[-1]).item() > 2 * counts[E // 2].item()


def test_plot_rank_histogram_runs():
    counts = rank_histogram(torch.randn(10, 1000), torch.randn(1000))
    plot_rank_histogram(counts, show=False)  # must not raise


# --------------------------------------------------------------------------- #
# Distributional fidelity
# --------------------------------------------------------------------------- #
def test_kl_at_points_same_distribution_zero():
    samp = torch.randn(1000, 6)
    ref = torch.randn(5000, 6)
    obs_mask = torch.tensor([True, True, True, False, False, False])
    kl = kl_at_points(samp, ref, observed_mask=obs_mask)
    assert abs(kl["mean"]) < 0.05
    assert "observed" in kl and "unobserved" in kl


def test_kl_at_points_shifted_positive():
    kl = kl_at_points(torch.randn(2000, 1), torch.randn(5000, 1) + 3.0)
    assert kl["mean"] > 0.5


def test_gaussian_and_kde_kl_small_for_same_dist():
    assert abs(gaussian_kl_1d(torch.randn(2000), torch.randn(5000))) < 0.05
    assert kde_kl_1d(torch.randn(2000), torch.randn(5000)) < 0.05


def test_sliced_w2_identical_sets_zero():
    a = torch.randn(1000, 10)
    assert sliced_wasserstein_w2(a, a.clone()) < 1e-9


def test_sliced_w2_shift_magnitude():
    a = torch.randn(1000, 10)
    b = torch.randn(1000, 10) + 2.0
    assert abs(sliced_wasserstein_w2(a, b) - 2.0) < 0.2


def test_sliced_w2_1d_scale():
    # W2(N(0,1), N(0,2^2)) = |1 - 2| = 1.
    val = sliced_wasserstein_w2(torch.randn(20000), 2.0 * torch.randn(20000))
    assert abs(val - 1.0) < 0.05


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #
def test_nfe_counter():
    nfe = NFECounter()
    with nfe:
        for _ in range(7):
            nfe.increment()
        nfe.increment(3)
    assert nfe.count == 10
    nfe.reset()
    assert nfe.count == 0


def test_step_timer():
    timer = StepTimer()
    with timer:
        sum(range(100000))
    assert timer.elapsed > 0.0
