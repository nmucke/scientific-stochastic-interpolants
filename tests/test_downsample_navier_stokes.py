"""Spectral downsampling of periodic fields (scripts/downsample_navier_stokes.py)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "downsample_navier_stokes", ROOT / "scripts" / "downsample_navier_stokes.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def band_limited(shape, max_k, seed=0):
    """Random real field on [0, 2π)^2 with modes |k_x| < max_k[0], |k_y| < max_k[1]."""
    rng = np.random.default_rng(seed)
    x = 2 * np.pi * np.arange(shape[0])[:, None] / shape[0]
    y = 2 * np.pi * np.arange(shape[1])[None, :] / shape[1]
    field = np.full(shape, rng.normal())
    for kx in range(-max_k[0] + 1, max_k[0]):
        for ky in range(0, max_k[1]):
            a, b = rng.normal(size=2)
            field = field + a * np.cos(kx * x + ky * y) + b * np.sin(kx * x + ky * y)
    return field


@pytest.mark.parametrize("big,small", [((32, 32), (16, 16)), ((24, 40), (12, 20))])
def test_band_limited_field_is_exact_on_the_coarse_grid(big, small):
    field = band_limited(big, (small[0] // 2, small[1] // 2))
    coarse = module.spectral_downsample(field[None], small)[0]
    stride = (big[0] // small[0], big[1] // small[1])
    np.testing.assert_allclose(coarse, field[:: stride[0], :: stride[1]], atol=1e-10)


def test_mean_and_kept_amplitudes_are_preserved_and_high_modes_dropped():
    rng = np.random.default_rng(1)
    field = rng.normal(size=(3, 32, 32))
    coarse = module.spectral_downsample(field, (16, 16))
    assert coarse.shape == (3, 16, 16)
    np.testing.assert_allclose(coarse.mean(axis=(-2, -1)), field.mean(axis=(-2, -1)))
    big = np.fft.rfft2(field, norm="forward")
    small = np.fft.rfft2(coarse, norm="forward")
    np.testing.assert_allclose(small[..., :8, :8], big[..., :8, :8], atol=1e-12)
    np.testing.assert_allclose(small[..., 8, :], 0, atol=1e-12)  # Nyquist row
    np.testing.assert_allclose(small[..., :, 8], 0, atol=1e-12)  # Nyquist column


def test_identity_at_full_size_up_to_the_nyquist_mode():
    field = band_limited((16, 16), (8, 8), seed=2)
    np.testing.assert_allclose(module.spectral_downsample(field, (16, 16)), field, atol=1e-10)


def test_rejects_odd_or_larger_targets():
    field = np.zeros((16, 16))
    for size in [(15, 16), (32, 16)]:
        with pytest.raises(ValueError):
            module.spectral_downsample(field, size)


def test_discarded_power_counts_only_dropped_modes():
    field = band_limited((32, 32), (8, 8), seed=3)  # nothing above |k| = 7
    sums = module.discarded_power(field, (16, 16))
    assert sums["enstrophy_dropped"] < 1e-20 * sums["enstrophy_total"]
    x = 2 * np.pi * np.arange(32)[:, None] / 32
    high = field + np.cos(12 * x) * np.ones((1, 32))
    sums = module.discarded_power(high, (16, 16))
    assert sums["enstrophy_dropped"] == pytest.approx(0.5, rel=1e-10)  # (1/2)^2 * 2 modes


def test_main_writes_data_and_summary_and_refuses_to_overwrite(tmp_path):
    state = np.stack([band_limited((32, 32), (8, 8), seed=s) for s in range(12)])
    state = state.reshape(3, 4, 32, 32).astype(np.float32)
    np.savez(tmp_path / "in.npz", state=state)
    out = tmp_path / "out"
    argv = ["--input", str(tmp_path / "in.npz"), "--size", "16",
            "--output-dir", str(out), "--train-trajectories", "2"]
    module.main(argv)
    small = np.load(out / "data.npz")["state"]
    assert small.shape == (3, 4, 16, 16) and small.dtype == np.float32
    np.testing.assert_allclose(small, state[..., ::2, ::2], atol=1e-5)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["train_std"] == pytest.approx(float(small[:2].astype(np.float64).std()), rel=1e-6)
    assert summary["discarded_fraction_train"]["enstrophy"] < 1e-12
    assert not (out / "data.tmp.npz").exists()
    with pytest.raises(SystemExit):
        module.main(argv)
