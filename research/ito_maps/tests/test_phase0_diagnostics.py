"""Phase 0 diagnostic helpers: spectra conventions, SI steppers, shared paths, NS budget."""

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import ns_budget  # noqa: E402
import phase0_common as common  # noqa: E402
import rollout_diagnostics as rd  # noqa: E402
import uncertainty_spectrum as us  # noqa: E402
from spatial_enstrophy_spectra import shell_geometry  # noqa: E402

from scisi.models.follmer_ito_map_model import FollmerItoMapModel  # noqa: E402
from scisi.models.follmer_stochastic_interpolant import (  # noqa: E402
    FollmerStochasticInterpolant,
)
from scisi.models.interpolations import QuadraticStochasticInterpolation  # noqa: E402


# ------------------------------------------------------------------ bases


def test_fourier_basis_shells_parseval_and_white_level():
    basis = us.FourierBasis(32, 32)
    full = shell_geometry(32)["shell_index"]
    np.testing.assert_array_equal(basis.shell, full[:, : 32 // 2 + 1])
    counts = basis.shell_means(np.zeros((1, *basis.mode_shape)))[1]
    np.testing.assert_array_equal(counts, shell_geometry(32)["mode_counts"])
    rng = np.random.default_rng(0)
    x = 1.7 * rng.standard_normal((400, 3, 32, 32))
    power = basis.power(x)
    np.testing.assert_allclose((power * basis.multiplicity).sum((-2, -1)), (x**2).sum((-2, -1)))
    means, _ = basis.shell_means(power.mean(0))
    np.testing.assert_allclose(means[:, 1:17], 1.7**2, rtol=0.1)


def test_fourier_basis_non_square_odd_width():
    basis = us.FourierBasis(12, 9)
    x = np.random.default_rng(1).standard_normal((5, 2, 12, 9))
    power = basis.power(x)
    assert power.shape == (5, 2, 12, 5)
    np.testing.assert_allclose((power * basis.multiplicity).sum((-2, -1)), (x**2).sum((-2, -1)))
    assert basis.multiplicity[0, 0] == 1 and basis.multiplicity[0, -1] == 2


def test_increment_and_anomaly_power_units():
    basis = us.FourierBasis(16, 16)
    rng = np.random.default_rng(2)
    walk = np.cumsum(0.5 * rng.standard_normal((3, 1, 16, 16, 60)), -1)
    inc = us.increment_power(walk, basis, history_length=5)
    np.testing.assert_allclose(basis.shell_means(inc.mean(0))[0][0, 1:9], 0.25, rtol=0.15)
    truth = rng.standard_normal((300, 1, 16, 16))
    ens = rng.standard_normal((300, 8, 1, 16, 16))  # calibrated: truth exchangeable with members
    var, err = us.anomaly_power(ens, truth, basis)
    assert var.mean() == pytest.approx(1.0, rel=0.02)
    assert err.mean() / var.mean() == pytest.approx(1 + 1 / 8, rel=0.03)


def test_gate_logic():
    ratios = np.array([0.0, 0.31, 20, 190, 59, 9.8, 0.59, 0.066, 0.012])
    g = us.gate(ratios, np.arange(1, 9))
    assert not g["any_white_level_within_factor"] and g["argmax_shell"] == 3
    flat = np.array([0.0, 1.0, 2.0, 0.5, 1.5])
    g = us.gate(flat, np.arange(1, 5))
    assert g["within_factor_of_white_everywhere"] and g["any_white_level_within_factor"]


# ------------------------------------------------------------ SI steppers


class ZeroDrift(nn.Module):
    def __init__(self, value=0.0):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def forward(self, x, cond, field_history=None, field_cond=None, pars_cond=None):
        return self.value * torch.ones_like(x)


def si(a=1.0, drift=0.0):
    return FollmerStochasticInterpolant(
        QuadraticStochasticInterpolation(gamma_multiplier=a), ZeroDrift(drift)
    ).eval()


def test_em_stepper_reproduces_existing_solver():
    model = si(drift=0.3)
    history = torch.randn(4, 2, 5, 6, 3)
    torch.manual_seed(5)
    ref = model.sample(base=history[..., -1], field_history=history, batch_size=4, num_steps=7)
    torch.manual_seed(5)
    out = common.sample_next(model, history, 7, si_noise="em")
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_exact_noise_endpoint_variance_and_em_inflation():
    a, n = 1.0, 4
    model = si(a)
    history = torch.zeros(20000, 1, 2, 3, 1)
    g = torch.Generator().manual_seed(0)
    exact = common.sample_next(model, history, n, si_noise="exact", generator=g)
    em = common.sample_next(model, history, n, si_noise="em", generator=g)
    assert exact.var().item() == pytest.approx(a**2 / 3, rel=0.02)
    em_theory = a**2 * (1 / 3 + 1 / (2 * n) + 1 / (6 * n**2))
    assert em.var().item() == pytest.approx(em_theory, rel=0.02)
    shared = common.sample_next(model, history[:6], n, shared=True, generator=g)
    assert torch.all(shared == shared[:1])


def test_integrated_variance_matches_clock():
    gamma = lambda t: 1.3 * (1 - t)
    s, t = torch.tensor(0.2, dtype=torch.float64), torch.tensor(0.7, dtype=torch.float64)
    expected = 1.3**2 * ((1 - s) ** 3 - (1 - t) ** 3) / 3
    torch.testing.assert_close(common.integrated_variance(gamma, s, t), expected)


# ------------------------------------------------------------ map wrappers


class PathSensitive(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, cond, field_history=None, field_cond=None, pars_cond=None, path_features=None):
        f = path_features.reshape(x.shape[0], -1, *x.shape[1:]).sum(1)
        return self.scale * f


def ito_map():
    return FollmerItoMapModel(
        QuadraticStochasticInterpolation(gamma_multiplier=1.0, wiener_process=True),
        PathSensitive(),
        num_path_features=2,
    ).eval()


def test_shared_path_gives_identical_members_on_non_ns_shape():
    model = ito_map()
    history = torch.randn(1, 3, 4, 7, 2).repeat(5, 1, 1, 1, 1)
    g = torch.Generator().manual_seed(3)
    out = common.sample_next(model, history, 2, shared=True, generator=g)
    assert out.shape == (5, 3, 4, 7)
    assert torch.all(out == out[:1])
    free = common.sample_next(model, history, 2, generator=g)
    assert not torch.all(free == free[:1])


def test_zero_feature_path_removes_feature_spread_only():
    model = ito_map()
    history = torch.zeros(3, 3, 4, 7, 2)
    g = torch.Generator().manual_seed(4)
    state = g.get_state()
    full, path = common.sample_next(model, history, 1, generator=g, return_path=True)
    g.set_state(state)
    ablated, path2 = common.sample_next(
        model, history, 1, generator=g, path_wrapper=common.ZeroFeaturePath, return_path=True
    )
    m1 = path.increment(0.0, 1.0)
    torch.testing.assert_close(m1, path2.increment(0.0, 1.0))
    torch.testing.assert_close(ablated, m1)  # zero features -> zero drift here
    torch.testing.assert_close(full, 2 * m1)  # drift = sum of features = M1


# -------------------------------------------------------------- NS budget


def test_ns_budget_constants_and_operators():
    budgets = ns_budget.build_ns_budgets(height=32, width=32)
    z, e = budgets["enstrophy"], budgets["energy"]
    assert z.injection_rate == pytest.approx(2.0)
    assert e.injection_rate == pytest.approx(0.5 * (1 / 36 + 1 / 49 + 1 / 50 + 1 / 128))
    x = torch.arange(32, dtype=torch.float64) * 2 * math.pi / 32
    u = torch.cos(6 * x)[:, None].expand(32, 32).reshape(1, 1, 32, 32)
    torch.testing.assert_close(z.dissipation(u), (0.1 + 1e-3 * 36) * u)
    torch.testing.assert_close(e.quadratic(u), torch.tensor([0.25 / 36], dtype=torch.float64))
    torch.testing.assert_close(z.martingale_variance(u), torch.tensor([0.5 * 0.25], dtype=torch.float64))


CACHE = Path(
    os.environ.get(
        "PHASE0_DATA_CACHE",
        "/scratch/ntmucke/scientific-stochastic-interpolants/results_itomap/phase0/cache",
    )
) / "data_traj160_180.npy"


@pytest.mark.skipif(not CACHE.exists(), reason="validation-split archive cache absent (data test)")
def test_archive_budget_closes_on_validation_split():
    data = torch.from_numpy(np.load(CACHE))
    budgets = ns_budget.build_ns_budgets(height=data.shape[2], width=data.shape[3])
    for name in ("enstrophy", "energy"):
        stats = ns_budget.validate_archive(budgets[name], data)
        assert stats["stationary_balance_ratio"] == pytest.approx(1.0, abs=0.03)
        assert abs(stats["residual_mean_over_injection"]) < 0.03


# --------------------------------------------------------- rollout helpers


def test_member_member_corr_and_band():
    rng = np.random.default_rng(6)
    ens = rng.standard_normal((5, 8, 9))
    brute = np.mean([rd.pattern_corr(ens[i], ens[j]) for i in range(5) for j in range(5) if i != j])
    assert rd.member_member_corr(ens) == pytest.approx(brute)
    band = rd.boot_band([1.0, 1.0, 3.0, 3.0], [0, 0, 1, 1])
    assert band["mean"] == pytest.approx(2.0) and band["q05"] <= 2.0 <= band["q95"]
