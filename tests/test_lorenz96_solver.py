"""Lorenz 96 solver: fixed point, order of accuracy, chaos, noise model."""

import math

import pytest
import torch

from scisi.dynamics.lorenz96 import (
    StochasticLorenz96,
    integrate,
    integrate_state_major,
    lorenz96_rhs,
    lyapunov_exponents,
    rk4_step,
)


def test_rhs_matches_explicit_loop():
    x = torch.randn(7, dtype=torch.float64)
    d, forcing = 7, 8.0
    expected = torch.tensor(
        [
            (x[(i + 1) % d] - x[(i - 2) % d]) * x[(i - 1) % d] - x[i] + forcing
            for i in range(d)
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(lorenz96_rhs(x, forcing), expected)


def test_uniform_state_is_a_fixed_point():
    forcing = 8.0
    x = torch.full((40,), forcing, dtype=torch.float64)
    assert lorenz96_rhs(x, forcing).abs().max() == 0
    # Rounding only: the fixed point is unstable, so keep the horizon short.
    assert (integrate(x, 100, 0.01, forcing) - forcing).abs().max() < 1e-12


def test_energy_balance_of_the_quadratic_term():
    # The advection term conserves sum(x^2): d/dt (1/2)|x|^2 = -|x|^2 + F sum(x).
    x = torch.randn(40, dtype=torch.float64)
    forcing = 8.0
    lhs = (x * lorenz96_rhs(x, forcing)).sum()
    torch.testing.assert_close(lhs, -(x * x).sum() + forcing * x.sum())


def test_rk4_has_fourth_order_convergence():
    forcing, horizon, d = 8.0, 0.5, 40
    gen = torch.Generator().manual_seed(0)
    x0 = forcing + torch.randn(d, generator=gen, dtype=torch.float64)
    reference = integrate(x0, 8192, horizon / 8192, forcing)
    errors = []
    steps = [32, 64, 128, 256]
    for n in steps:
        errors.append((integrate(x0, n, horizon / n, forcing) - reference).norm().item())
    orders = [math.log2(a / b) for a, b in zip(errors, errors[1:])]
    assert all(3.7 < order < 4.3 for order in orders), orders


def test_ensemble_axis_is_vectorised_and_layout_independent():
    x = torch.randn(5, 11, dtype=torch.float64)
    batched = rk4_step(x, 0.01, 8.0)
    for i in range(5):
        torch.testing.assert_close(batched[i], rk4_step(x[i], 0.01, 8.0))
    torch.testing.assert_close(rk4_step(x.T, 0.01, 8.0, dim=0).T, batched)


@pytest.mark.parametrize("d", [5, 16, 40])
def test_state_major_path_matches_reference_rk4(d):
    gen = torch.Generator().manual_seed(8)
    x = 8.0 + 3.0 * torch.randn(d, 37, generator=gen, dtype=torch.float64)
    original = x.clone()
    fast = integrate_state_major(x, 40, 0.01, 8.0)
    torch.testing.assert_close(
        fast, integrate(x, 40, 0.01, 8.0, dim=0), atol=1e-9, rtol=0
    )
    assert torch.equal(x, original)  # the input must not be modified


def test_climatology_at_f8():
    # Long-run mean about 2.3 and standard deviation about 3.6 for F = 8.
    model = StochasticLorenz96(dim_state=40, dt=0.01, interval=0.05)
    gen = torch.Generator().manual_seed(1)
    x = model.spin_up(64, 400, gen)
    samples = model.trajectory(x, 200, gen)
    assert 2.1 < samples.mean().item() < 2.5
    assert 3.4 < samples.std().item() < 3.8


def test_lyapunov_spectrum_at_f8():
    # Published d = 40, F = 8 values: lambda_1 about 1.7, about 13 positive.
    exponents = lyapunov_exponents(40, total_time=200.0, spinup_time=20.0)
    assert 1.5 < exponents[0].item() < 1.9
    assert 11 <= int((exponents > 0.02).sum()) <= 15
    # d/dt log volume is exactly -d.
    assert exponents.sum().item() == pytest.approx(-40.0, abs=0.5)


def test_nearby_trajectories_separate_at_the_lyapunov_rate():
    forcing, dt = 8.0, 0.01
    gen = torch.Generator().manual_seed(2)
    model = StochasticLorenz96(dim_state=40, dt=dt, interval=0.05)
    x = model.spin_up(256, 200, gen)
    y = x + 1e-9 * torch.randn(x.shape, generator=gen, dtype=x.dtype)
    gap0 = (x - y).norm(dim=1).log().mean()
    horizon = 5.0
    steps = int(horizon / dt)
    gap = (integrate(x, steps, dt, forcing) - integrate(y, steps, dt, forcing))
    rate = (gap.norm(dim=1).log().mean() - gap0).item() / horizon
    assert 1.3 < rate < 2.0


def test_single_stage_kernel_is_exactly_gaussian_around_the_flow():
    model = StochasticLorenz96(dim_state=8, interval=0.1, noise_std=0.5, noise_stages=1)
    gen = torch.Generator().manual_seed(3)
    x0 = 8.0 + torch.randn(8, generator=gen, dtype=torch.float64)
    flow = integrate(x0, 10, 0.01, 8.0)
    draws = model.step(x0.expand(200_000, 8).clone(), gen)
    torch.testing.assert_close(draws.mean(0), flow, atol=5e-3, rtol=0)
    torch.testing.assert_close(
        draws.var(0), torch.full_like(flow, 0.25), atol=5e-3, rtol=0
    )


def test_stages_inject_the_stated_kick_variances():
    # With the flow removed (F = 0 from the origin stays at 0 up to the kicks, and a
    # tiny interval keeps it linear) the kicks simply add, so the summed variance is
    # noise_std^2 whatever the stage count.
    model = StochasticLorenz96(
        dim_state=8, forcing=0.0, dt=1e-5, interval=4e-5, noise_std=0.3, noise_stages=4
    )
    gen = torch.Generator().manual_seed(9)
    draws = model.step(torch.zeros(400_000, 8, dtype=torch.float64), gen)
    torch.testing.assert_close(
        draws.var(0), torch.full((8,), 0.09, dtype=torch.float64), atol=1e-3, rtol=0
    )


def test_multi_stage_kernel_is_non_gaussian_at_long_intervals():
    split = StochasticLorenz96(dim_state=8, interval=0.4, noise_std=1.0, noise_stages=4)
    gen = torch.Generator().manual_seed(4)
    x0 = split.spin_up(1, 200, gen)[0]
    draws = split.step(x0.expand(200_000, 8).clone(), gen)
    z = (draws - draws.mean(0)) / draws.std(0)
    excess_kurtosis = (z**4).mean(0) - 3.0
    assert excess_kurtosis.abs().max() > 0.05


def test_incompatible_stage_count_is_rejected():
    with pytest.raises(ValueError):
        StochasticLorenz96(interval=0.05, dt=0.01, noise_stages=3)
