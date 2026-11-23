"""The coloured-noise spectrum estimator recovers a known spectrum (non-NS shapes)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from scisi.sampling.noise_covariance import FourierDiagonalCovariance

SCRIPT = Path(__file__).parents[1] / "experiments" / "estimate_noise_spectrum.py"
spec = importlib.util.spec_from_file_location("estimate_noise_spectrum", SCRIPT)
estimator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(estimator)


@pytest.mark.parametrize("shape", [(3, 8, 6), (3, 6, 5)])
def test_recovers_the_spectrum_of_coloured_samples(shape):
    generator = torch.Generator().manual_seed(0)
    half = 0.2 + torch.rand(*shape[:-1], shape[-1] // 2 + 1, generator=generator, dtype=torch.float64)
    truth = FourierDiagonalCovariance(half).double()
    sigma_half = truth._symmetric_spectrum(shape[-1])
    samples = truth.sqrt(torch.randn(20000, *shape, generator=generator, dtype=torch.float64))
    power = estimator.full_mode_power(samples) / samples.shape[0]
    torch.testing.assert_close(estimator.half_to_full(power[..., : shape[-1] // 2 + 1], shape[-1]), power)
    q1 = 0.25
    spectrum = estimator.noise_spectrum(power, q1, kappa=2.0, floor=0.0, smoothing=0)
    torch.testing.assert_close(spectrum, 2.0 * sigma_half / q1, rtol=0.06, atol=0)
    # The written spectrum gives back q(1) C = kappa Sigma as driver covariance.
    covariance = FourierDiagonalCovariance(spectrum).double()
    driver = covariance.sqrt(torch.randn(20000, *shape, generator=generator, dtype=torch.float64))
    driver_power = estimator.full_mode_power(driver)[..., : shape[-1] // 2 + 1] / 20000
    torch.testing.assert_close(q1 * driver_power, 2.0 * sigma_half, rtol=0.1, atol=0)


def test_floor_smoothing_and_invariant_mode():
    sigma = torch.zeros(3, 6, 5, dtype=torch.float64)
    sigma[:, 0, 0] = 50.0  # mean mode, excluded by zero_mean
    sigma[:, 1, 1] = 1.0
    sigma[:, -1, -1] = 1.0  # its mirror
    spectrum = estimator.noise_spectrum(sigma, 1.0, floor=1e-3, smoothing=1, zero_mean=True)
    assert spectrum.shape == (3, 6, 3)
    assert (spectrum[:, 0, 0] == 0).all()
    assert (spectrum[:, 1:, :] >= 1e-3 * spectrum.max() * (1 - 1e-12)).all()
    # Smoothing spreads the single mode over its neighbours, not over k = 0's 50.
    assert spectrum.max() < 0.5
    constant = estimator.noise_spectrum(torch.ones(2, 4, 6, dtype=torch.float64), 0.5, smoothing=2)
    torch.testing.assert_close(constant, torch.full((2, 4, 4), 2.0, dtype=torch.float64))
    with pytest.raises(ValueError, match="kappa"):
        estimator.noise_spectrum(sigma, 1.0, kappa=0.5)


def test_anomaly_and_power_inputs(tmp_path):
    rng = np.random.default_rng(1)
    ensemble = rng.normal(size=(50, 8, 2, 6, 4)) * 3.0
    np.savez(tmp_path / "ensemble.npz", ensemble=ensemble)
    power, count = estimator.ensemble_power(tmp_path / "ensemble.npz", scale=3.0)
    assert count == 400 and power.shape == (2, 6, 4)
    assert power.mean().item() == pytest.approx(1.0, rel=0.05)
    np.savez(tmp_path / "power.npz", increment_train=power[..., :3].numpy() * 4.0)
    full, _ = estimator.precomputed_power(tmp_path / "power.npz", "increment_train", 2.0, None)
    torch.testing.assert_close(full, power)
