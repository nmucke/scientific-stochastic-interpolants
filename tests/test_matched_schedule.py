"""Variance clock, matched schedule and Gaussian exactness of the matched driver."""

import pytest
import torch
from torch import nn

from scisi.models.follmer_ito_map_model import FollmerItoMapModel
from scisi.models.interpolations import (
    LinearStochasticInterpolation,
    MatchedStochasticInterpolation,
    QuadraticStochasticInterpolation,
    matched_beta_quadrature,
    quadrature_variance_clock,
)
from scisi.sampling.noise_covariance import FourierDiagonalCovariance

TIMES = torch.linspace(0, 1, 101, dtype=torch.float64)


@pytest.mark.parametrize(
    "interpolation_class",
    [
        LinearStochasticInterpolation,
        QuadraticStochasticInterpolation,
        MatchedStochasticInterpolation,
    ],
)
def test_variance_clock_closed_form_equals_quadrature(interpolation_class):
    interpolation = interpolation_class(gamma_multiplier=0.7)
    s, t = TIMES[:-1], TIMES[1:]
    closed = interpolation.variance_clock(s, t)
    torch.testing.assert_close(
        closed, quadrature_variance_clock(interpolation.gamma, s, t), rtol=1e-12, atol=0
    )
    assert interpolation.variance_clock(0.0, 1.0) == pytest.approx(0.49 / 3, rel=1e-15)
    # Additive over sub-intervals.
    assert closed.sum().item() == pytest.approx(0.49 / 3, rel=1e-12)


@pytest.mark.parametrize("gamma_multiplier", [0.1, 1.0, 3.0])
def test_matched_beta_endpoints_monotone_and_equal_to_quadrature(gamma_multiplier):
    interpolation = MatchedStochasticInterpolation(gamma_multiplier=gamma_multiplier)
    beta = interpolation.beta(TIMES)
    assert beta[0] == 0 and beta[-1] == 1
    assert (beta.diff() > 0).all()
    quadrature, quadrature_diff = matched_beta_quadrature(
        interpolation.gamma, interpolation.gamma_diff, TIMES
    )
    torch.testing.assert_close(beta, quadrature, rtol=1e-10, atol=1e-12)
    inner = TIMES[1:-1]
    torch.testing.assert_close(
        interpolation.beta_diff(inner), quadrature_diff[1:-1], rtol=1e-8, atol=1e-10
    )
    assert interpolation.beta_diff(torch.tensor(0.0)) == pytest.approx(3**0.5)
    # beta^2 q(1) = q(t) - t gamma^2, the defining identity.
    q = interpolation.variance_clock(torch.zeros_like(TIMES), TIMES)
    torch.testing.assert_close(
        beta**2 * interpolation.variance_clock(0.0, 1.0),
        q - TIMES * interpolation.gamma(TIMES) ** 2,
        rtol=1e-12,
        atol=1e-15,
    )


class GaussianZeroFeatureMap(nn.Module):
    """Best average drift without path features for a Gaussian target.

    With y | h ~ N(m, Sigma), C = Sigma / q(1) and I_t = alpha c + beta y +
    gamma sqrt(t) C^{1/2} z, the exact drift is b_t(x) = mu'_t + K_t (x - mu_t),
    mu_t = alpha_t c + beta_t m, with the scalar gain
    K_t = (beta beta' q(1) + t gamma gamma') / (beta^2 q(1) + t gamma^2).
    Given only x, E[X_t | X_s = x] = mu_t + Phi(t, s) (x - mu_s) with
    Phi = exp(int_s^t K), so the zero-feature average drift is
    G_{s,t}(x) = (mu_t - mu_s + (Phi - 1)(x - mu_s)) / (t - s). The matched
    schedule gives K = 0; beta = t^2 does not.
    """

    def __init__(self, interpolation, mean):
        super().__init__()
        self.interpolation = interpolation
        self.register_buffer("mean", mean)
        self.dummy = nn.Parameter(torch.zeros((), dtype=mean.dtype))

    def gain(self, t):
        i = self.interpolation
        q1 = i.variance_clock(0.0, 1.0)
        numerator = i.beta(t) * i.beta_diff(t) * q1 + t * i.gamma(t) * i.gamma_diff(t)
        return numerator / (i.beta(t) ** 2 * q1 + t * i.gamma(t) ** 2)

    def log_propagator(self, s, t, nodes=64):
        r, w = (torch.from_numpy(a) for a in __import__("numpy").polynomial.legendre.leggauss(nodes))
        half = 0.5 * (t - s)
        times = (s + t) * 0.5 + half * r
        return half.squeeze(-1) * (w * self.gain(times)).sum(-1)

    def forward(self, x, cond, field_history=None, path_features=None, **kwargs):
        i = self.interpolation
        s, t = cond[:, :1], cond[:, 1:]
        c = field_history[..., -1]

        def mu(time):
            time = time.reshape(-1, 1, 1, 1)
            return i.alpha(time) * c + i.beta(time) * self.mean

        gap = (t - s).reshape(-1, 1, 1, 1)
        propagator = self.log_propagator(s, t).exp().reshape(-1, 1, 1, 1)
        return (mu(t) - mu(s) + (propagator - 1) * (x - mu(s))) / gap


def spectrum_of(samples):
    centred = samples - samples.mean(0)
    coefficients = torch.fft.rfftn(centred, dim=(2, 3), norm="ortho")
    return coefficients.abs().square().mean(0)


@pytest.mark.parametrize("schedule", ["matched", "quadratic"])
def test_gaussian_target_zero_feature_map_exactness(schedule):
    """Matched beta: right mean and covariance at every step count, no features.

    For beta = t^2 a zero-feature map gets the covariance wrong once it takes
    more than one step (from a point mass one step only sees M_1, whose law
    q(1) C = Sigma is right for any beta, though not pathwise).
    """
    state_shape = (3, 6, 5)  # non-NS: three channels, non-square
    gamma_multiplier = 0.9
    generator = torch.Generator().manual_seed(0)
    sigma = 0.2 + torch.rand(*state_shape[:-1], 3, generator=generator, dtype=torch.float64)
    sigma = FourierDiagonalCovariance(sigma)._symmetric_spectrum(5)
    mean = torch.randn(1, *state_shape, generator=generator, dtype=torch.float64)
    q1 = gamma_multiplier**2 / 3
    covariance = FourierDiagonalCovariance(sigma / q1).double()
    interpolation_class = (
        MatchedStochasticInterpolation
        if schedule == "matched"
        else QuadraticStochasticInterpolation
    )
    interpolation = interpolation_class(
        gamma_multiplier=gamma_multiplier, noise_covariance=covariance
    )
    network = GaussianZeroFeatureMap(interpolation, mean).double()
    model = FollmerItoMapModel(interpolation, network, num_path_features=1)
    history = torch.randn(1, *state_shape, 2, generator=generator, dtype=torch.float64)
    for num_steps in (1, 2, 4):
        samples = model.sample(
            history,
            batch_size=20000,
            num_steps=num_steps,
            generator=torch.Generator().manual_seed(num_steps),
        )
        torch.testing.assert_close(samples.mean(0), mean[0], rtol=0, atol=0.03)
        ratio = spectrum_of(samples) / sigma
        if schedule == "matched" or num_steps == 1:
            torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=0, atol=0.07)
        else:
            # Composed zero-feature steps: Var = sum_j Phi(1, t_j)^2 dq_j C.
            times = torch.linspace(0, 1, num_steps + 1, dtype=torch.float64)
            ones = torch.ones(1, 1, dtype=torch.float64)
            expected = sum(
                network.log_propagator(times[j + 1] * ones, ones).exp().item() ** 2
                * interpolation.variance_clock(times[j].item(), times[j + 1].item())
                for j in range(num_steps)
            ) / q1
            assert abs(expected - 1) > 0.15
            torch.testing.assert_close(
                ratio, torch.full_like(ratio, expected), rtol=0.07, atol=0
            )
