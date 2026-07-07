"""Unit tests for the Brownian path simulation and encoding (plan Phase 5.2)."""

import math

import pytest
import torch

from scisi.models.interpolations import QuadraticStochasticInterpolation
from scisi.models.ito_maps.brownian import (
    BrownianPathSampler,
    DyadicEncoder,
    GammaMatchedSigmaSchedule,
    KLEncoder,
    PaperSigmaSchedule,
    ZeroSigmaSchedule,
)

BATCH = 4096
STATE_SHAPE = (BATCH, 1, 2, 2)


def _paper_sigma_sq_integral(t: float) -> float:
    """int_0^t sigma_u^2 du for sigma = sqrt(2(1-u))."""
    return 2 * t - t**2


def test_sigma_schedules():
    t = torch.linspace(0, 1, 5).reshape(-1, 1)

    assert ZeroSigmaSchedule().is_zero
    assert torch.all(ZeroSigmaSchedule()(t) == 0)

    paper = PaperSigmaSchedule()
    assert not paper.is_zero
    assert torch.allclose(paper(t) ** 2, 2 * (1 - t))

    interpolation = QuadraticStochasticInterpolation(gamma_multiplier=0.5)
    matched = GammaMatchedSigmaSchedule(interpolation)
    assert torch.allclose(matched(t), interpolation.gamma(t))


def test_grid_brownian_statistics():
    torch.manual_seed(0)
    sampler = BrownianPathSampler(PaperSigmaSchedule(), num_grid_points=64, mode="path")
    sample = sampler.sample(torch.Size(STATE_SHAPE), "cpu")

    # W(1) ~ N(0, 1)
    w_one = sample.w_at_times(torch.tensor([1.0]))[..., 0]
    assert abs(w_one.var().item() - 1.0) < 0.1
    assert abs(w_one.mean().item()) < 0.05

    # Per-sample evaluation agrees with the shared-time evaluation.
    t = torch.full((BATCH, 1), 0.5)
    w_half_per_sample = sample.w_at(t)
    w_half_shared = sample.w_at_times(torch.tensor([0.5]))[..., 0]
    assert torch.allclose(w_half_per_sample, w_half_shared)

    # Var(W(t) - W(s)) = t - s (independent increments)
    s = torch.full((BATCH, 1), 0.25)
    increment = sample.w_at(t) - sample.w_at(s)
    assert abs(increment.var().item() - 0.25) < 0.05


@pytest.mark.parametrize("mode", ["path", "kl"])
def test_martingale_variance_matches_sigma_sq_integral(mode):
    torch.manual_seed(1)
    sampler = BrownianPathSampler(
        PaperSigmaSchedule(), num_grid_points=64, mode=mode, num_kl_terms=64
    )
    sample = sampler.sample(torch.Size(STATE_SHAPE), "cpu")

    zeros = torch.zeros(BATCH, 1)
    for t_val in (0.5, 1.0):
        t = torch.full((BATCH, 1), t_val)
        m_t = sample.martingale_increment(zeros, t)
        expected = _paper_sigma_sq_integral(t_val)
        assert abs(m_t.mean().item()) < 0.05
        assert abs(m_t.var().item() - expected) / expected < 0.1


def test_kl_cumulative_integrals_parseval():
    """Sum_n I_n(t)^2 -> int_0^t sigma^2 du as the truncation grows (exact
    closed-form check of the kl-mode martingale variance, no sampling)."""
    sampler = BrownianPathSampler(
        PaperSigmaSchedule(), mode="kl", num_kl_terms=256, dense_grid_size=2048
    )
    integrals = sampler._cumulative_integrals  # [G+1, K]

    for t_val in (0.5, 1.0):
        row = integrals[int(t_val * (integrals.shape[0] - 1))]
        variance = (row**2).sum().item()
        expected = _paper_sigma_sq_integral(t_val)
        assert abs(variance - expected) / expected < 0.02


def test_kl_coefficients_of_grid_path_are_standard_normal():
    torch.manual_seed(2)
    sampler = BrownianPathSampler(
        PaperSigmaSchedule(), num_grid_points=128, mode="path"
    )
    sample = sampler.sample(torch.Size(STATE_SHAPE), "cpu")

    xi = sample.kl_coefficients(4)  # [B, 4, 1, 2, 2]
    assert xi.shape == (BATCH, 4, 1, 2, 2)
    variances = xi.var(dim=(0, 2, 3, 4))
    assert torch.all((variances - 1.0).abs() < 0.15)


@pytest.mark.parametrize("mode", ["path", "kl"])
def test_encoder_output_shapes(mode):
    torch.manual_seed(3)
    batch, channels, height, width = 3, 2, 4, 4
    sampler = BrownianPathSampler(
        PaperSigmaSchedule(), num_grid_points=32, mode=mode, num_kl_terms=8
    )
    sample = sampler.sample(torch.Size((batch, channels, height, width)), "cpu")

    kl_features = KLEncoder(num_coeffs=5)(sample)
    assert kl_features.shape == (batch, 5 * channels, height, width)
    assert KLEncoder(num_coeffs=5).num_features_per_channel == 5

    dyadic = DyadicEncoder(depth=3)
    dyadic_features = dyadic(sample)
    assert dyadic_features.shape == (batch, 8 * channels, height, width)
    assert dyadic.num_features_per_channel == 8
    assert torch.isfinite(dyadic_features).all()


@pytest.mark.parametrize(
    "mode,kwargs",
    [("path", {"num_grid_points": 64}), ("kl", {"num_kl_terms": 16})],
)
def test_standard_normal_at_is_unit_variance_at_small_times(mode, kwargs):
    """The coupled interpolant noise z = W_t / sqrt(t) must be exactly N(0, 1)
    marginally at EVERY t. Uncompensated, the truncated KL series (K=16) and
    the interpolated grid path under-disperse badly below t ~ 1/K (e.g.
    Var ~ 0.16 at t = 0.005 for kl mode); standard_normal_at compensates the
    representation's variance deficit with independent noise."""
    torch.manual_seed(5)
    num_samples = 20000
    sampler = BrownianPathSampler(PaperSigmaSchedule(), mode=mode, **kwargs)
    sample = sampler.sample(torch.Size((num_samples, 1, 1, 1)), "cpu")

    for t_val in (0.005, 0.02, 0.05, 0.5, 0.95):
        t = torch.full((num_samples, 1), t_val)
        z = sample.standard_normal_at(t)
        assert abs(z.var().item() - 1.0) < 0.05, f"t={t_val}: var={z.var():.3f}"
        assert abs(z.mean().item()) < 0.03

        # The representable variance is a valid lower bound (deficit >= 0).
        assert torch.all(sample.w_variance_at(t) <= t_val + 1e-6)


def test_kl_and_grid_martingale_increments_have_matching_statistics():
    """The two path representations must define the same increment law."""
    torch.manual_seed(4)
    s = torch.full((BATCH, 1), 0.2)
    t = torch.full((BATCH, 1), 0.7)
    expected = _paper_sigma_sq_integral(0.7) - _paper_sigma_sq_integral(0.2)

    for mode, kwargs in (("path", {"num_grid_points": 128}), ("kl", {"num_kl_terms": 64})):
        sampler = BrownianPathSampler(PaperSigmaSchedule(), mode=mode, **kwargs)
        sample = sampler.sample(torch.Size(STATE_SHAPE), "cpu")
        increment = sample.martingale_increment(s, t)
        assert abs(increment.var().item() - expected) / expected < 0.1
