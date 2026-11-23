"""Coloured interpolant noise: covariance operators, drivers and samplers (WP1)."""

import pytest
import torch

from scisi.models.follmer_ito_map_model import FollmerItoMapModel
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
from scisi.models.interpolations import (
    AdditiveNoiseMixin,
    LinearStochasticInterpolation,
    MatchedStochasticInterpolation,
    QuadraticStochasticInterpolation,
)
from scisi.models.invariants import SpatialMeanInvariant
from scisi.sampling.brownian_path import (
    KarhunenLoeveBrownianPath,
    WeightedBrownianPath,
    make_brownian_path,
)
from scisi.sampling.noise_covariance import (
    FourierDiagonalCovariance,
    IdentityCovariance,
    ProjectedCovariance,
    same_noise_covariance,
)
from scisi.sampling.sde_solvers import euler_maruyama_step, heun_step

# Non-NS shapes: three channels, non-square grids, odd and even last axes,
# and spatial ranks 1 and 3.
SHAPES = [(3, 6, 5), (3, 8, 6), (2, 7), (2, 4, 3, 6)]


def random_spectrum(state_shape, seed=0, dtype=torch.float64):
    """A random non-negative rfft-grid spectrum, not Hermitian-symmetric."""
    rfft_shape = (*state_shape[:-1], state_shape[-1] // 2 + 1)
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(rfft_shape, generator=generator, dtype=dtype) * 2 + 0.05


def mode_power(x):
    """E|F x|^2 per orthonormal rfft mode over the leading batch dimension."""
    coefficients = torch.fft.rfftn(x, dim=tuple(range(2, x.ndim)), norm="ortho")
    return coefficients.abs().square().mean(0)


@pytest.mark.parametrize("state_shape", SHAPES)
def test_fourier_covariance_is_a_real_symmetric_square_root(state_shape):
    covariance = FourierDiagonalCovariance(random_spectrum(state_shape)).double()
    generator = torch.Generator().manual_seed(1)
    a, b = (
        torch.randn(4, *state_shape, generator=generator, dtype=torch.float64)
        for _ in range(2)
    )
    root = covariance.sqrt(a)
    assert root.shape == a.shape and not root.is_complex()
    # Symmetric: <a, C^{1/2} b> = <C^{1/2} a, b>.
    torch.testing.assert_close(
        (a * covariance.sqrt(b)).sum(), (root * b).sum(), rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(covariance.sqrt(root), covariance.apply(a))
    # All modes positive here, so solve inverts apply.
    torch.testing.assert_close(covariance.solve(covariance.apply(a)), a)


def test_fourier_covariance_colours_white_noise_with_its_spectrum():
    state_shape = (3, 6, 5)
    spectrum = random_spectrum(state_shape)
    covariance = FourierDiagonalCovariance(spectrum).double()
    symmetric = covariance._symmetric_spectrum(state_shape[-1])
    z = torch.randn(40000, *state_shape, generator=torch.Generator().manual_seed(2))
    power = mode_power(covariance.sqrt(z.double()))
    # With norm="ortho" every rfft mode of white noise has E|.|^2 = 1, so every
    # mode of C^{1/2} z has E|.|^2 = S.
    torch.testing.assert_close(power, symmetric, rtol=0.05, atol=0)
    # S was asymmetric inside the self-conjugate plane; elsewhere it is kept.
    assert not torch.equal(symmetric[:, :, 0], spectrum.float().double()[:, :, 0])
    torch.testing.assert_close(symmetric[:, :, 1:], spectrum[:, :, 1:])


def test_fourier_covariance_from_file_and_validation(tmp_path):
    spectrum = random_spectrum((3, 6, 5)).numpy()
    path = tmp_path / "spectrum.npz"
    import numpy as np

    np.savez(path, spectrum=spectrum)
    from_file = FourierDiagonalCovariance(spectrum_path=str(path))
    torch.testing.assert_close(from_file.spectrum, torch.as_tensor(spectrum).float())
    assert same_noise_covariance(
        LinearStochasticInterpolation(noise_covariance=from_file),
        LinearStochasticInterpolation(
            noise_covariance=FourierDiagonalCovariance(spectrum)
        ),
    )
    assert not same_noise_covariance(
        LinearStochasticInterpolation(noise_covariance=from_file),
        LinearStochasticInterpolation(),
    )
    with pytest.raises(ValueError, match="exactly one"):
        FourierDiagonalCovariance()
    with pytest.raises(ValueError, match="non-negative"):
        FourierDiagonalCovariance(-torch.ones(1, 4, 3))
    with pytest.raises(ValueError, match="does not match"):
        from_file.sqrt(torch.randn(2, 3, 6, 6))


def test_identity_returns_its_input_and_projection_removes_the_mean():
    z = torch.randn(2, 3, 6, 5)
    assert IdentityCovariance().sqrt(z) is z
    invariant = SpatialMeanInvariant()
    for base in (
        IdentityCovariance(),
        FourierDiagonalCovariance(random_spectrum((3, 6, 5))),
    ):
        projected = ProjectedCovariance(base, invariant).double()
        out = projected.sqrt(z.double())
        torch.testing.assert_close(
            invariant.value(out), torch.zeros(2, 3, dtype=torch.float64)
        )


class CubicGammaInterpolation(LinearStochasticInterpolation):
    """gamma = a (1 - t)^2: not affine, so drivers must use the quadrature clock."""

    variance_clock = AdditiveNoiseMixin.variance_clock

    def gamma(self, t):
        return self.gamma_multiplier * (1 - t) ** 2

    def gamma_diff(self, t):
        return -2 * self.gamma_multiplier * (1 - t)


@pytest.mark.parametrize("state_shape", [(3, 6, 5), (2, 8, 6)])
@pytest.mark.parametrize(
    "interpolation_class",
    [QuadraticStochasticInterpolation, MatchedStochasticInterpolation, CubicGammaInterpolation],
)
def test_driver_increment_covariance_is_clock_times_c(state_shape, interpolation_class):
    spectrum = random_spectrum(state_shape)
    covariance = FourierDiagonalCovariance(spectrum).double()
    interpolation = interpolation_class(gamma_multiplier=0.8, noise_covariance=covariance)
    # The quadrature clock runs per batch entry in Python: fewer samples there.
    cubic = interpolation_class is CubicGammaInterpolation
    x = torch.zeros(4000 if cubic else 20000, *state_shape, dtype=torch.float64)
    path = make_brownian_path(interpolation, x, torch.Generator().manual_seed(3))
    s, u, t = 0.2, 0.45, 0.9
    increment = path.increment(s, t)
    # Additivity over sub-intervals of one realization.
    torch.testing.assert_close(
        path.increment(s, u) + path.increment(u, t), increment, rtol=0, atol=1e-13
    )
    expected = interpolation.variance_clock(s, t) * covariance._symmetric_spectrum(
        state_shape[-1]
    )
    power = mode_power(increment)
    torch.testing.assert_close(power, expected, rtol=0.12 if cubic else 0.06, atol=0)
    assert (power / expected).mean().item() == pytest.approx(1.0, abs=0.02)
    if cubic:
        assert path.variance_clock is not None  # quadrature clock routed in
        # int_s^t a^2 (1-r)^4 dr = a^2 [(1-s)^5 - (1-t)^5] / 5
        exact = 0.64 * ((1 - s) ** 5 - (1 - t) ** 5) / 5
        assert interpolation.variance_clock(s, t) == pytest.approx(exact, rel=1e-12)


def test_karhunen_loeve_driver_is_coloured_and_forks_keep_the_covariance():
    state_shape = (3, 6, 5)
    covariance = FourierDiagonalCovariance(random_spectrum(state_shape)).double()
    interpolation = QuadraticStochasticInterpolation(
        gamma_multiplier=0.8, noise_covariance=covariance
    )
    x = torch.zeros(20000, *state_shape, dtype=torch.float64)
    path = make_brownian_path(
        interpolation, x, torch.Generator().manual_seed(4), "karhunen_loeve", 3
    )
    assert isinstance(path, KarhunenLoeveBrownianPath)
    fork = path.fork(torch.Generator().manual_seed(5))
    assert fork.covariance is covariance
    end = fork.increment(0.0, 1.0)
    expected = interpolation.variance_clock(0.0, 1.0) * covariance._symmetric_spectrum(5)
    torch.testing.assert_close(
        mode_power(end), expected, rtol=0.06, atol=0
    )
    with pytest.raises(ValueError, match="require gamma"):
        make_brownian_path(
            CubicGammaInterpolation(), x[:1], features="karhunen_loeve"
        )


def test_interpolant_and_drift_target_use_the_coloured_noise():
    state_shape = (3, 6, 5)
    covariance = FourierDiagonalCovariance(random_spectrum(state_shape)).double()
    interpolation = MatchedStochasticInterpolation(
        gamma_multiplier=0.6, noise_covariance=covariance
    )
    base, target, noise = (torch.randn(2, *state_shape, dtype=torch.float64) for _ in range(3))
    t = torch.tensor([[0.3], [0.8]], dtype=torch.float64)
    tt = t[:, :, None, None]
    coloured = covariance.sqrt(noise)
    expected = (1 - tt) * base + tt * torch.sqrt(3 - 2 * tt) * target
    expected = expected + 0.6 * (1 - tt) * tt.sqrt() * coloured
    torch.testing.assert_close(interpolation(base, target, t, noise), expected)
    expected_diff = -base + 3 * (1 - tt) / torch.sqrt(3 - 2 * tt) * target
    expected_diff = expected_diff - 0.6 * tt.sqrt() * coloured
    torch.testing.assert_close(interpolation.forward_diff(base, target, t, noise), expected_diff)
    with pytest.raises(NotImplementedError, match="white"):
        interpolation.velocity_score_coeff(t)


class ZeroDrift(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x, cond, field_history=None, field_cond=None, pars_cond=None, path_features=None):
        return self.scale * x


@pytest.mark.parametrize("stepper", [euler_maruyama_step, heun_step])
def test_solver_steppers_colour_the_sde_noise(stepper):
    state_shape = (3, 6, 5)
    covariance = FourierDiagonalCovariance(random_spectrum(state_shape))
    interpolation = LinearStochasticInterpolation(
        gamma_multiplier=0.7, noise_covariance=covariance
    )
    model = FollmerStochasticInterpolant(interpolation, ZeroDrift())
    history = torch.zeros(1, *state_shape, 2)
    torch.manual_seed(6)
    out = model.sample(
        base=history[..., -1], field_history=history, batch_size=20000,
        num_steps=4, stepper=stepper,
    )
    times = torch.linspace(0, 1, 5)[:-1]
    if stepper is euler_maruyama_step:  # left-point gamma^2 dt
        clock = (0.49 * (1 - times) ** 2 * 0.25).sum()
    else:  # Heun averages gamma over the step
        clock = (0.49 * (1 - times - 0.125) ** 2 * 0.25).sum()
    expected = clock * covariance._symmetric_spectrum(5)
    torch.testing.assert_close(
        mode_power(out), expected, rtol=0.06, atol=0
    )
    with pytest.raises(NotImplementedError, match="white"):
        model.sample(base=history[..., -1], field_history=history, diffusion_term=lambda t: t)
    # The exact driver of the SDE comes from the same interpolation.
    path = model.make_brownian_path(history[..., -1])
    assert isinstance(path, WeightedBrownianPath) and path.covariance is covariance


def test_ito_map_samples_with_coloured_driver_and_checks_the_invariant():
    state_shape = (3, 6, 5)
    spectrum = random_spectrum(state_shape)
    model = FollmerItoMapModel(
        MatchedStochasticInterpolation(
            gamma_multiplier=0.5, noise_covariance=FourierDiagonalCovariance(spectrum)
        ),
        ZeroDrift(),
        num_path_features=2,
    )
    history = torch.zeros(1, *state_shape, 3)
    out = model.sample(history, batch_size=20000, num_steps=3,
                       generator=torch.Generator().manual_seed(7))
    expected = model.interpolation.variance_clock(0.0, 1.0) * model.interpolation.noise_covariance._symmetric_spectrum(5)
    torch.testing.assert_close(
        mode_power(out), expected, rtol=0.06, atol=0
    )
    leaky = FollmerItoMapModel(
        MatchedStochasticInterpolation(noise_covariance=FourierDiagonalCovariance(spectrum)),
        ZeroDrift(),
        invariant=SpatialMeanInvariant(),
    )
    with pytest.raises(ValueError, match="vanish on the invariant"):
        leaky.sample(history)
