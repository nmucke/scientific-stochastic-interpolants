import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "spatial_enstrophy_spectra",
    Path(__file__).parents[1]
    / "research/ito_maps/experiments/spatial_enstrophy_spectra.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_constant_field_keeps_dc_and_correct_absolute_units():
    k, spectrum, audit = module.radial_enstrophy_shells(np.full((32, 32), 3.0))
    assert spectrum[0] == pytest.approx(4.5)
    np.testing.assert_allclose(spectrum[1:], 0, atol=0)
    assert audit["direct_enstrophy"] == pytest.approx(4.5)
    assert audit["dc_enstrophy"] == pytest.approx(4.5)


def test_known_sine_lives_in_expected_integer_shell():
    n = 64
    x = np.arange(n) * 2 * np.pi / n
    field = 4 * np.sin(3 * x[:, None] + 4 * x[None, :])
    k, spectrum, audit = module.radial_enstrophy_shells(field)
    assert spectrum[5] == pytest.approx(4.0)  # A^2/4, no shell-mode-count averaging.
    assert np.sum(np.delete(spectrum, 5)) < 1e-25
    assert spectrum.sum() == pytest.approx(0.5 * np.mean(field**2))


def test_random_batch_parseval_and_amplitude_scaling():
    rng = np.random.default_rng(18)
    field = rng.normal(size=(2, 3, 32, 32))
    k, spectrum, audit = module.radial_enstrophy_shells(field)
    assert spectrum.shape == (2, 3, len(k))
    np.testing.assert_allclose(
        spectrum.sum(-1), 0.5 * np.mean(field**2, axis=(-2, -1)), rtol=1e-13
    )
    _, scaled, _ = module.radial_enstrophy_shells(3 * field)
    np.testing.assert_allclose(scaled, 9 * spectrum, rtol=1e-13)


def test_square_corner_nyquist_mode_is_not_dropped():
    n = 32
    indices = np.arange(n)
    field = 3.0 * (-1.0) ** (indices[:, None] + indices[None, :])
    k, spectrum, audit = module.radial_enstrophy_shells(field)
    corner_shell = int(np.floor(np.sqrt(2) * (n // 2) + 0.5))
    assert corner_shell > n // 2
    assert spectrum[corner_shell] == pytest.approx(4.5)
    assert spectrum.sum() == pytest.approx(4.5)
    geometry = module.shell_geometry(n)
    assert geometry["mode_counts"].sum() == n * n
    assert geometry["mode_counts"][0] == 1


def test_member_mean_spectrum_differs_from_spectrum_of_mean():
    n = 32
    x = np.arange(n) * 2 * np.pi / n
    field = np.sin(2 * x[:, None]) * np.ones((1, n))
    _, summary = module.spectrum_summary(np.stack([field, -field]))
    assert summary["spectrum_mean"][2] == pytest.approx(0.25)
    np.testing.assert_allclose(summary["spectrum_of_ensemble_mean"], 0, atol=0)


@pytest.mark.parametrize(
    "field",
    [np.zeros((3, 4)), np.ones((4, 4)) * np.nan, np.ones((4, 4), dtype=complex)],
)
def test_invalid_fields_rejected(field):
    with pytest.raises(ValueError):
        module.radial_enstrophy_shells(field)
