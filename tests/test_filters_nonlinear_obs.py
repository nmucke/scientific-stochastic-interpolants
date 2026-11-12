"""Filters with pointwise nonlinear observations y = g(H x) + eps.

Exact references: the Kalman filter when ``g`` is the identity, and a 1D grid
(quadrature) filter for a scalar linear-Gaussian model with arctan observations.
"""

import math
from dataclasses import dataclass

import pytest
import torch

from scisi.particle_filter.bootstrap import (
    BootstrapParticleFilter,
    ImportanceOneStepOracle,
    OneStepPosteriorOracle,
    selection_indices,
)
from scisi.particle_filter.enkf import EnsembleKalmanFilter, periodic_taper


@dataclass(frozen=True)
class LinearModel:
    """x' = A x + sqrt(q) xi, with the protocol the filters expect."""

    A: torch.Tensor
    stage_variance: float
    noise_stages: int = 1

    def forecast_to_last_kick(self, x, generator=None, dim=-1):
        return self.A @ x if dim == 0 else x @ self.A.T

    def step(self, x, generator=None, dim=-1):
        z = self.forecast_to_last_kick(x, generator, dim)
        return z + self.stage_variance**0.5 * torch.randn(
            z.shape, generator=generator, dtype=z.dtype
        )


def arctan(c):
    def g(w):
        return torch.atan(c * w) / c, 1.0 / (1.0 + (c * w) ** 2)

    return g


def identity(w):
    return w, torch.ones_like(w)


def kalman_filter(A, q, H, r, m0, P0, observations):
    d = A.shape[0]
    m, P = m0.clone(), P0.clone()
    means, covs, log_evidence = [], [], []
    eye_y = torch.eye(H.shape[0], dtype=A.dtype)
    for y in observations:
        m, P = A @ m, A @ P @ A.T + q * torch.eye(d, dtype=A.dtype)
        S = H @ P @ H.T + r * eye_y
        log_evidence.append(
            float(torch.distributions.MultivariateNormal(H @ m, S).log_prob(y))
        )
        K = P @ H.T @ torch.linalg.inv(S)
        m = m + K @ (y - H @ m)
        P = (torch.eye(d, dtype=A.dtype) - K @ H) @ P
        means.append(m)
        covs.append(P)
    return torch.stack(means), torch.stack(covs), torch.tensor(log_evidence)


def grid_filter(a, q, g, r, m0, p0, observations, lo=-12.0, hi=12.0, n=6001):
    """Scalar x' = a x + N(0, q), y = g(x) + N(0, r), by quadrature on a grid."""
    x = torch.linspace(lo, hi, n, dtype=torch.float64)
    dx = float(x[1] - x[0])
    density = torch.exp(-0.5 * (x - m0) ** 2 / p0) / math.sqrt(2 * math.pi * p0)
    kernel = torch.exp(-0.5 * (x[:, None] - a * x[None, :]) ** 2 / q) / math.sqrt(
        2 * math.pi * q
    )
    means, variances, log_evidence = [], [], []
    for y in observations:
        predicted = kernel @ density * dx
        likelihood = torch.exp(-0.5 * (y - g(x)[0]) ** 2 / r) / math.sqrt(2 * math.pi * r)
        joint = predicted * likelihood
        evidence = float(joint.sum() * dx)
        density = joint / evidence
        mean = float((x * density).sum() * dx)
        means.append(mean)
        variances.append(float(((x - mean) ** 2 * density).sum() * dx))
        log_evidence.append(math.log(evidence))
    return (
        torch.tensor(means)[:, None],
        torch.tensor(variances)[:, None],
        torch.tensor(log_evidence),
    )


def assert_unbiased(estimates, exact, name, rms_limit=1.8):
    """Replicate t-statistics against an exact value (see test_filters.py)."""
    estimates = torch.stack(estimates)
    t = (estimates.mean(0) - exact) / (estimates.std(0) / estimates.shape[0] ** 0.5)
    rms = t.pow(2).mean().sqrt().item()
    assert rms < rms_limit, f"{name}: RMS t = {rms:.2f}"


def replicates(H, r, proposal, model, m0, p0, observations, n, reps, g):
    runs = []
    for rep in range(reps):
        gen = torch.Generator().manual_seed(2000 + rep)
        particles = m0[:, None] + p0**0.5 * torch.randn(
            m0.numel(), n, generator=gen, dtype=torch.float64
        )
        pf = BootstrapParticleFilter(H, r, proposal=proposal, obs_nonlinearity=g)
        runs.append(pf.run(model, particles, observations, gen))
    return runs


@pytest.fixture
def linear_problem():
    gen = torch.Generator().manual_seed(0)
    d = 4
    A = 0.9 * torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))[0]
    H = torch.eye(d, dtype=torch.float64)[[0, 2]]
    q, r = 0.3, 0.5
    m0 = torch.ones(d, dtype=torch.float64)
    P0 = 0.5 * torch.eye(d, dtype=torch.float64)
    x = m0 + 0.5**0.5 * torch.randn(d, generator=gen, dtype=torch.float64)
    observations = []
    for _ in range(10):
        x = A @ x + q**0.5 * torch.randn(d, generator=gen, dtype=torch.float64)
        observations.append(H @ x + r**0.5 * torch.randn(2, generator=gen, dtype=torch.float64))
    return A, q, H, r, m0, P0, torch.stack(observations)


@pytest.mark.parametrize("g", [None, identity])
def test_linearised_proposal_with_linear_observations_matches_kalman(linear_problem, g):
    A, q, H, r, m0, P0, observations = linear_problem
    mean, cov, log_evidence = kalman_filter(A, q, H, r, m0, P0, observations)
    runs = replicates(H, r, "linearised", LinearModel(A, q), m0, 0.5, observations, 50_000, 12, g)
    assert_unbiased([x.mean for x in runs], mean, "mean")
    assert_unbiased([x.variance for x in runs], torch.diagonal(cov, dim1=1, dim2=2), "variance")
    assert_unbiased(
        [x.log_evidence.sum() for x in runs], log_evidence.sum(), "log evidence", 3.0
    )


def test_linearised_weights_reduce_to_the_optimal_proposal_weights(linear_problem):
    """For g = identity the weight is N(y; H z, (q + r) I) whatever the kick."""
    A, q, H, r, m0, _, observations = linear_problem
    gen = torch.Generator().manual_seed(3)
    z = torch.randn(4, 1000, generator=gen, dtype=torch.float64)
    pf = BootstrapParticleFilter(H, r, proposal="linearised")
    log_w = pf._linearised_kick(z.clone(), observations[0], q, gen)
    S = (q + r) * torch.eye(2, dtype=torch.float64)
    expected = pf._log_gaussian(observations[0][:, None] - H @ z, S)
    torch.testing.assert_close(log_w, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("proposal", ["bootstrap", "linearised"])
def test_nonlinear_observation_filter_matches_grid_filter(proposal):
    a, q, r, c = 0.95, 0.5, 0.6, 1.5
    g = arctan(c)
    gen = torch.Generator().manual_seed(5)
    x, observations = torch.tensor(0.5, dtype=torch.float64), []
    for _ in range(8):
        x = a * x + q**0.5 * torch.randn((), generator=gen, dtype=torch.float64)
        observations.append(g(x)[0] + r**0.5 * torch.randn((), generator=gen, dtype=torch.float64))
    observations = torch.stack(observations)[:, None]
    mean, variance, log_evidence = grid_filter(
        a, q, g, r, 0.5, 1.0, observations[:, 0]
    )
    H = torch.ones(1, 1, dtype=torch.float64)
    model = LinearModel(torch.tensor([[a]], dtype=torch.float64), q)
    runs = replicates(H, r, proposal, model, torch.tensor([0.5], dtype=torch.float64), 1.0,
                      observations, 50_000, 12, g)
    assert_unbiased([x.mean for x in runs], mean, "mean")
    assert_unbiased([x.variance for x in runs], variance, "variance")
    assert_unbiased(
        [x.log_evidence.sum() for x in runs], log_evidence.sum(), "log evidence", 3.0
    )
    assert (runs[0].mean - mean).abs().max() < 0.03


def test_optimal_proposal_refuses_a_nonlinearity_and_linearised_needs_a_selection():
    H = torch.eye(3, dtype=torch.float64)[[0, 2]]
    with pytest.raises(ValueError, match="linear observations"):
        BootstrapParticleFilter(H, 1.0, proposal="optimal", obs_nonlinearity=arctan(1.0))
    with pytest.raises(ValueError, match="selection"):
        BootstrapParticleFilter(0.5 * H, 1.0, proposal="linearised")
    assert selection_indices(H).tolist() == [0, 2]


def test_linearised_proposal_warns_when_weights_may_be_heavy_tailed():
    H = torch.ones(1, 1, dtype=torch.float64)
    pf = BootstrapParticleFilter(H, 0.1, proposal="linearised")
    z = torch.zeros(1, 10, dtype=torch.float64)
    with pytest.warns(RuntimeWarning, match="infinite variance"):
        pf._linearised_kick(z, torch.zeros(1, dtype=torch.float64), 1.0, None)


@pytest.mark.parametrize("radius", [None, 2.0])
def test_nonlinear_enkf_with_identity_equals_the_linear_enkf(radius):
    gen = torch.Generator().manual_seed(1)
    d, members = 8, 30
    H = torch.eye(d, dtype=torch.float64)[::2]
    forecast = 3.0 + torch.randn(members, d, generator=gen, dtype=torch.float64)
    y = torch.randn(H.shape[0], generator=gen, dtype=torch.float64)
    taper = None if radius is None else periodic_taper(d, radius)
    linear = EnsembleKalmanFilter(H, 0.7, inflation=1.1, localisation=taper)
    nonlinear = EnsembleKalmanFilter(
        H, 0.7, inflation=1.1, localisation=taper, obs_nonlinearity=identity
    )
    a = linear.analyse(forecast, y, torch.Generator().manual_seed(9))
    b = nonlinear.analyse(forecast, y, torch.Generator().manual_seed(9))
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_nonlinear_enkf_uses_predicted_observations():
    """A large ensemble with a strongly nonlinear g: the mean update is
    C_xy (C_yy + R)^-1 (y - mean g(H x)), not the update of the linear H."""
    gen = torch.Generator().manual_seed(2)
    H = torch.ones(1, 1, dtype=torch.float64)
    forecast = 2.0 + torch.randn(20_000, 1, generator=gen, dtype=torch.float64)
    g = arctan(2.0)
    y = torch.tensor([0.1], dtype=torch.float64)
    out = EnsembleKalmanFilter(H, 0.3, obs_nonlinearity=g).analyse(forecast, y, gen)
    pred = g(forecast)[0]
    cxy = torch.cov(torch.cat([forecast, pred], 1).T)[0, 1]
    cyy = pred.var()
    expected = forecast.mean() + cxy / (cyy + 0.3) * (y - pred.mean())
    assert float(out.mean()) == pytest.approx(float(expected), abs=1e-10)


def test_importance_oracle_converges_to_the_closed_form_oracle():
    """Linear g: the per-member target is the Gaussian of OneStepPosteriorOracle."""
    gen = torch.Generator().manual_seed(4)
    d = 3
    A = 0.8 * torch.eye(d, dtype=torch.float64)
    H = torch.eye(d, dtype=torch.float64)[[0, 2]]
    q, r = 1.0, 0.5
    model = LinearModel(A, q)
    x = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64).repeat(4000, 1)
    y = torch.tensor([2.0, -1.0], dtype=torch.float64)
    oracle = ImportanceOneStepOracle(H, r, proposals=2048)
    result = oracle.run(model, x, y[None], gen)
    draws = result.analysis[0]
    z = x[0] @ A.T
    S = q * H @ H.T + r * torch.eye(2, dtype=torch.float64)
    K = q * H.T @ torch.linalg.inv(S)
    mean = z + K @ (y - H @ z)
    var = torch.diagonal(q * (torch.eye(d, dtype=torch.float64) - K @ H))
    se = (var / draws.shape[0]).sqrt()
    assert ((draws.mean(0) - mean).abs() < 4 * se).all()
    torch.testing.assert_close(draws.var(0), var, rtol=0.08, atol=0.0)
    # The closed-form oracle draws from the same target.
    exact = OneStepPosteriorOracle(H, r).run(model, x, y[None], gen)[0]
    torch.testing.assert_close(exact.var(0), var, rtol=0.08, atol=0.0)
    assert result.ess.shape == (1, 4000)
    # Weights N(y; x_obs, r) against a unit-variance prior: ESS well below P.
    assert 1.0 < float(result.ess.min()) and float(result.ess.max()) < 2048


def test_importance_oracle_with_several_stages_uses_the_transition():
    @dataclass(frozen=True)
    class TwoStage:
        stage_variance: float = 0.5
        noise_stages: int = 2

        def step(self, x, generator=None, dim=-1):
            return x + torch.randn(x.shape, generator=generator, dtype=x.dtype)

        def forecast_to_last_kick(self, x, generator=None, dim=-1):  # pragma: no cover
            raise AssertionError("must not be used with several stages")

    H = torch.eye(2, dtype=torch.float64)[[0]]
    x = torch.zeros(500, 2, dtype=torch.float64)
    out = ImportanceOneStepOracle(H, 1.0, proposals=512).run(
        TwoStage(), x, torch.tensor([[1.0]], dtype=torch.float64), torch.Generator().manual_seed(0)
    )
    # Prior N(0, 1), r = 1, y = 1: posterior mean 0.5 on the observed site.
    assert float(out.analysis[0, :, 0].mean()) == pytest.approx(0.5, abs=0.1)
    assert float(out.analysis[0, :, 1].mean()) == pytest.approx(0.0, abs=0.15)
