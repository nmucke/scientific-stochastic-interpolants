"""Distributional and coupling checks for the Itô-map Gaussian driver."""

import pytest
import torch

from scisi.sampling.brownian_path import WeightedBrownianPath


def make_path(shape=(4, 2, 3, 3), amplitude=0.7, seed=19):
    return WeightedBrownianPath.sample_like(
        torch.zeros(shape, dtype=torch.float64),
        amplitude,
        generator=torch.Generator().manual_seed(seed),
    )


def test_cached_values_and_composed_increments_are_coherent():
    path = make_path()
    s = torch.tensor([[0.0], [0.1], [0.2], [0.3]], dtype=torch.float64)
    u, t = s + 0.2, s + 0.6
    whole = path.increment(s, t)
    left, right = path.increment(s, u), path.increment(u, t)
    torch.testing.assert_close(whole, left + right, atol=1e-14, rtol=1e-14)
    assert torch.equal(whole, path.increment(s, t))
    assert torch.equal(path.increment(s, s), torch.zeros_like(whole))
    features = path.features(s, t, 4)
    assert features.shape == (4, 8, 3, 3)
    torch.testing.assert_close(features.reshape(4, 4, 2, 3, 3).sum(1), whole)
    assert torch.equal(path.features(s, s, 4), torch.zeros_like(features))


def test_arbitrary_time_bridge_has_correct_covariance_and_independent_increments():
    # Many independent coordinates avoid thousands of Python batch cache entries.
    path = make_path((1, 30000), amplitude=0.7)
    end = path.value(1.0).flatten()
    late = path.value(0.83).flatten()  # Query out of temporal order deliberately.
    early = path.value(0.17).flatten()
    middle = path.value(0.46).flatten()
    values = torch.stack([early, middle, late, end])
    q = lambda t: 0.7**2 * (t - t * t + t * t * t / 3)
    times = [0.17, 0.46, 0.83, 1.0]
    expected = torch.tensor(
        [[q(min(s, t)) for t in times] for s in times], dtype=torch.float64
    )
    torch.testing.assert_close(torch.cov(values), expected, atol=0.0025, rtol=0.025)
    assert values.mean(1).abs().max() < 0.006
    increments = torch.stack([early, middle - early, late - middle, end - late])
    expected_increment_cov = torch.diag(
        torch.tensor([q(0.17), q(0.46) - q(0.17), q(0.83) - q(0.46), q(1) - q(0.83)])
    ).double()
    torch.testing.assert_close(
        torch.cov(increments), expected_increment_cov, atol=0.0025, rtol=0.025
    )


def test_tiny_interval_variance_and_zero_amplitude():
    path = make_path((1, 20000), amplitude=1.0)
    # Near t=1, subtraction of absolute clock values would lose precision.
    variance = path.variance(0.99999, 1.0)
    torch.testing.assert_close(
        variance, torch.tensor([[1e-15 / 3]], dtype=torch.float64), rtol=1e-9, atol=0
    )
    zero = make_path(amplitude=0.0)
    assert torch.count_nonzero(zero.features(0.123, 0.982, 4)) == 0


def test_fixed_generator_and_validation():
    a, b = make_path(), make_path()
    assert torch.equal(a.features(0.11, 0.92, 4), b.features(0.11, 0.92, 4))
    with pytest.raises(ValueError):
        a.value(-0.1)
    with pytest.raises(ValueError):
        a.value(torch.tensor([0.1, 0.2]))
    with pytest.raises(ValueError):
        a.increment(0.7, 0.3)
    with pytest.raises(ValueError):
        a.features(0.1, 0.9, 0)
    with pytest.raises(ValueError):
        make_path(amplitude=-1)
