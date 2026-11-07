"""EnKF and particle filters against the exact Kalman filter."""

from dataclasses import dataclass

import pytest
import torch

from scisi.dynamics.lorenz96 import StochasticLorenz96
from scisi.particle_filter.bootstrap import (
    BootstrapParticleFilter,
    OneStepPosteriorOracle,
    effective_sample_size,
    systematic_resample,
)
from scisi.particle_filter.enkf import (
    EnsembleKalmanFilter,
    gaspari_cohn,
    periodic_taper,
)


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


@dataclass(frozen=True)
class TwoStageLinearModel:
    """x' = B (B x + sqrt(q) xi_1) + sqrt(q) xi_2: Gaussian, with two kicks.

    The filters only see the protocol, so this drives exactly the multi-stage
    path used for Lorenz 96 while keeping a closed-form answer: it is the linear
    model with A = B^2 and process covariance q (B B^T + I).
    """

    B: torch.Tensor
    stage_variance: float
    noise_stages: int = 2

    def forecast_to_last_kick(self, x, generator=None, dim=-1):
        apply = (lambda v: self.B @ v) if dim == 0 else (lambda v: v @ self.B.T)
        kicked = apply(x) + self.stage_variance**0.5 * torch.randn(
            x.shape, generator=generator, dtype=x.dtype
        )
        return apply(kicked)

    def step(self, x, generator=None, dim=-1):
        z = self.forecast_to_last_kick(x, generator, dim)
        return z + self.stage_variance**0.5 * torch.randn(
            z.shape, generator=generator, dtype=z.dtype
        )


def kalman_filter(A, q, H, r, m0, P0, observations):
    """Exact moments and log evidence; ``q`` is a scalar or a full matrix."""
    d = A.shape[0]
    Q = q * torch.eye(d, dtype=A.dtype) if isinstance(q, float) else q
    m, P = m0.clone(), P0.clone()
    means, covs, log_evidence = [], [], []
    eye_y = torch.eye(H.shape[0], dtype=A.dtype)
    for y in observations:
        m, P = A @ m, A @ P @ A.T + Q
        S = H @ P @ H.T + r * eye_y
        innovation = y - H @ m
        log_evidence.append(
            float(
                torch.distributions.MultivariateNormal(H @ m, S).log_prob(y)
            )
        )
        K = P @ H.T @ torch.linalg.inv(S)
        m = m + K @ innovation
        P = (torch.eye(d, dtype=A.dtype) - K @ H) @ P
        means.append(m)
        covs.append(P)
    return torch.stack(means), torch.stack(covs), torch.tensor(log_evidence)


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
    for _ in range(12):
        x = A @ x + q**0.5 * torch.randn(d, generator=gen, dtype=torch.float64)
        observations.append(
            H @ x + r**0.5 * torch.randn(2, generator=gen, dtype=torch.float64)
        )
    observations = torch.stack(observations)
    exact = kalman_filter(A, q, H, r, m0, P0, observations)
    return A, q, H, r, m0, P0, observations, exact


def assert_unbiased(estimates, exact, name, rms_limit=1.8):
    """Replicate t-statistics against an exact value.

    A single fixed-seed tolerance either passes by luck or is too loose to mean
    anything. With R replicates t has RMS sqrt((R-1)/(R-3)), about 1.1 for R = 12.
    """
    estimates = torch.stack(estimates)
    t = (estimates.mean(0) - exact) / (estimates.std(0) / estimates.shape[0] ** 0.5)
    rms = t.pow(2).mean().sqrt().item()
    assert rms < rms_limit, f"{name}: RMS t = {rms:.2f}"


def replicate_filter(H, r, proposal, model, m0, observations, n, reps, dtype):
    runs = []
    for rep in range(reps):
        gen = torch.Generator().manual_seed(1000 + rep)
        particles = (
            m0[:, None]
            + 0.5**0.5 * torch.randn(m0.numel(), n, generator=gen, dtype=torch.float64)
        ).to(dtype)
        runs.append(
            BootstrapParticleFilter(H, r, proposal=proposal).run(
                model, particles, observations, gen
            )
        )
    return runs


@pytest.mark.parametrize(
    "proposal, dtype",
    [
        ("bootstrap", torch.float64),
        ("optimal", torch.float64),
        ("optimal", torch.float32),
    ],
)
def test_particle_filter_matches_kalman(linear_problem, proposal, dtype):
    A, q, H, r, m0, _, observations, (mean, cov, log_evidence) = linear_problem
    model = LinearModel(A.to(dtype), q)
    runs = replicate_filter(H, r, proposal, model, m0, observations, 100_000, 12, dtype)
    assert_unbiased([x.mean for x in runs], mean, "mean")
    assert_unbiased(
        [x.variance for x in runs], torch.diagonal(cov, dim1=1, dim2=2), "variance"
    )
    assert_unbiased([x.covariance for x in runs], cov, "covariance")
    assert_unbiased(
        [x.log_evidence.sum() for x in runs], log_evidence.sum(), "log evidence", 3.0
    )
    # The per-run error must be small as well as unbiased.
    assert (runs[0].mean - mean).pow(2).mean().sqrt() < 0.02


def test_particle_filter_with_non_selection_operator():
    gen = torch.Generator().manual_seed(21)
    d = 4
    A = 0.9 * torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))[0]
    H = torch.tensor([[0.5, 0.5, 0, 0], [0, 0.2, 0.3, 0.5]], dtype=torch.float64)
    q, r = 0.3, 0.5
    m0 = torch.ones(d, dtype=torch.float64)
    x = m0.clone()
    observations = []
    for _ in range(8):
        x = A @ x + q**0.5 * torch.randn(d, generator=gen, dtype=torch.float64)
        observations.append(
            H @ x + r**0.5 * torch.randn(2, generator=gen, dtype=torch.float64)
        )
    observations = torch.stack(observations)
    mean, cov, _ = kalman_filter(
        A, q, H, r, m0, 0.5 * torch.eye(d, dtype=torch.float64), observations
    )
    runs = replicate_filter(
        H, r, "optimal", LinearModel(A, q), m0, observations, 100_000, 12, torch.float64
    )
    assert_unbiased([x.mean for x in runs], mean, "mean")
    assert_unbiased([x.covariance for x in runs], cov, "covariance")


def test_saved_samples_are_draws_from_the_posterior(linear_problem):
    A, q, H, r, m0, _, observations, (mean, cov, _) = linear_problem
    gen = torch.Generator().manual_seed(22)
    particles = m0[:, None] + 0.5**0.5 * torch.randn(
        4, 200_000, generator=gen, dtype=torch.float64
    )
    result = BootstrapParticleFilter(H, r, num_saved_samples=50_000).run(
        LinearModel(A, q), particles, observations, gen
    )
    assert result.samples.shape == (observations.shape[0], 50_000, 4)
    torch.testing.assert_close(result.samples.mean(1), mean, atol=0.03, rtol=0)
    torch.testing.assert_close(
        result.samples.var(1), torch.diagonal(cov, dim1=1, dim2=2), atol=0.03, rtol=0
    )


@pytest.mark.parametrize("proposal", ["bootstrap", "optimal"])
def test_multi_stage_filter_matches_kalman(proposal):
    # Exact reference for the Rao--Blackwellised last kick when earlier kicks exist.
    gen = torch.Generator().manual_seed(11)
    d = 4
    B = 0.9 * torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))[0]
    H = torch.eye(d, dtype=torch.float64)[[0, 2]]
    q, r = 0.3, 0.5
    model = TwoStageLinearModel(B, q)
    m0 = torch.ones(d, dtype=torch.float64)
    x = m0 + 0.5**0.5 * torch.randn(d, generator=gen, dtype=torch.float64)
    observations = []
    for _ in range(12):
        x = model.step(x, gen)
        observations.append(
            H @ x + r**0.5 * torch.randn(2, generator=gen, dtype=torch.float64)
        )
    observations = torch.stack(observations)
    Q = q * (B @ B.T + torch.eye(d, dtype=torch.float64))
    mean, cov, log_evidence = kalman_filter(
        B @ B, Q, H, r, m0, 0.5 * torch.eye(d, dtype=torch.float64), observations
    )
    runs = replicate_filter(H, r, proposal, model, m0, observations, 100_000, 12, torch.float64)
    assert_unbiased([x.mean for x in runs], mean, "mean")
    assert_unbiased(
        [x.variance for x in runs], torch.diagonal(cov, dim1=1, dim2=2), "variance"
    )
    assert_unbiased(
        [x.log_evidence.sum() for x in runs], log_evidence.sum(), "log evidence", 3.0
    )


def test_optimal_proposal_has_higher_ess_than_bootstrap(linear_problem):
    A, q, H, r, m0, _, observations, _ = linear_problem
    ess = {}
    for proposal in ("bootstrap", "optimal"):
        gen = torch.Generator().manual_seed(2)
        particles = m0[:, None] + 0.5**0.5 * torch.randn(
            4, 50_000, generator=gen, dtype=torch.float64
        )
        result = BootstrapParticleFilter(H, r, proposal=proposal).run(
            LinearModel(A, q), particles, observations, gen
        )
        ess[proposal] = result.ess_fraction.mean().item()
    assert ess["optimal"] > ess["bootstrap"]


def test_enkf_matches_kalman(linear_problem):
    A, q, H, r, m0, P0, observations, (mean, cov, _) = linear_problem
    gen = torch.Generator().manual_seed(3)
    ensemble = m0 + 0.5**0.5 * torch.randn(
        400_000, 4, generator=gen, dtype=torch.float64
    )
    analysis = EnsembleKalmanFilter(H, r).run(
        LinearModel(A, q), ensemble, observations, gen
    ).analysis
    torch.testing.assert_close(analysis.mean(1), mean, atol=0.01, rtol=0)
    torch.testing.assert_close(
        analysis.var(1), torch.diagonal(cov, dim1=1, dim2=2), atol=0.01, rtol=0
    )


def test_enkf_inflation_equals_scaling_the_forecast_covariance():
    gen = torch.Generator().manual_seed(23)
    d, lam, r = 3, 1.3, 0.5
    H = torch.eye(d, dtype=torch.float64)[:2]
    y = torch.tensor([1.0, -1.0], dtype=torch.float64)
    forecast = torch.randn(1_000_000, d, generator=gen, dtype=torch.float64)
    forecast = forecast * torch.tensor([1.0, 0.7, 1.2], dtype=torch.float64)
    analysis = EnsembleKalmanFilter(H, r, inflation=lam).analyse(forecast, y, gen)
    P = lam**2 * torch.cov(forecast.T)
    K = P @ H.T @ torch.linalg.inv(H @ P @ H.T + r * torch.eye(2, dtype=torch.float64))
    mean = forecast.mean(0) + K @ (y - H @ forecast.mean(0))
    torch.testing.assert_close(analysis.mean(0), mean, atol=5e-3, rtol=0)
    torch.testing.assert_close(
        torch.cov(analysis.T),
        (torch.eye(d, dtype=torch.float64) - K @ H) @ P,
        atol=5e-3,
        rtol=0,
    )


def test_enkf_localisation_uses_the_tapered_covariance():
    gen = torch.Generator().manual_seed(24)
    d, r = 4, 0.5
    H = torch.eye(d, dtype=torch.float64)[:1]
    y = torch.tensor([2.0], dtype=torch.float64)
    mix = torch.randn(d, d, generator=gen, dtype=torch.float64)
    forecast = torch.randn(400_000, d, generator=gen, dtype=torch.float64) @ mix
    taper = periodic_taper(d, 1.0)
    analysis = EnsembleKalmanFilter(H, r, localisation=taper).analyse(forecast, y, gen)
    P = torch.cov(forecast.T) * taper
    K = P @ H.T / (H @ P @ H.T + r)
    mean = forecast.mean(0) + (K @ (y - H @ forecast.mean(0)))
    torch.testing.assert_close(analysis.mean(0), mean, atol=5e-3, rtol=0)


def test_oracle_sir_targets_the_one_step_posterior():
    # Two-stage linear kernel: p(x' | x, y) is Gaussian and known in closed form.
    gen = torch.Generator().manual_seed(25)
    d = 3
    B = 0.9 * torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))[0]
    H = torch.eye(d, dtype=torch.float64)[:2]
    q, r = 0.4, 0.5
    x = torch.tensor([1.0, -0.5, 0.3], dtype=torch.float64)
    Q = q * (B @ B.T + torch.eye(d, dtype=torch.float64))
    prior_mean = B @ B @ x
    y = H @ prior_mean + torch.tensor([0.5, -0.4], dtype=torch.float64)
    K = Q @ H.T @ torch.linalg.inv(H @ Q @ H.T + r * torch.eye(2, dtype=torch.float64))
    post_mean = prior_mean + K @ (y - H @ prior_mean)
    post_cov = (torch.eye(d, dtype=torch.float64) - K @ H) @ Q
    analysis = OneStepPosteriorOracle(H, r, inner_samples=256).run(
        TwoStageLinearModel(B, q), x.expand(20_000, d).clone(), y[None], gen
    )[0]
    torch.testing.assert_close(analysis.mean(0), post_mean, atol=0.03, rtol=0)
    torch.testing.assert_close(torch.cov(analysis.T), post_cov, atol=0.03, rtol=0)


def test_oracle_equals_filter_when_transition_ignores_the_past():
    # A = 0 makes the state independent of its past, so p(y | x^{n-1}) is constant:
    # the unweighted one-step scheme is then exactly the filter.
    d = 3
    H = torch.eye(d, dtype=torch.float64)[:2]
    q, r = 1.0, 0.5
    model = LinearModel(torch.zeros(d, d, dtype=torch.float64), q)
    gen = torch.Generator().manual_seed(4)
    observations = torch.randn(3, 2, generator=gen, dtype=torch.float64)
    ensemble = torch.randn(300_000, d, generator=gen, dtype=torch.float64)
    analysis = OneStepPosteriorOracle(H, r).run(model, ensemble, observations, gen)
    mean, cov, _ = kalman_filter(
        model.A, q, H, r, torch.zeros(d).double(), torch.eye(d).double(), observations
    )
    torch.testing.assert_close(analysis.mean(1), mean, atol=0.01, rtol=0)
    torch.testing.assert_close(
        analysis.var(1), torch.diagonal(cov, dim1=1, dim2=2), atol=0.01, rtol=0
    )


def test_oracle_differs_from_filter_when_past_matters(linear_problem):
    # With informative dynamics the unweighted scheme is NOT the filter: it cannot
    # reweight x^{n-1}, so it must retain more variance than the Kalman posterior.
    A, q, H, r, m0, _, observations, (_, cov, _) = linear_problem
    gen = torch.Generator().manual_seed(5)
    ensemble = m0 + 0.5**0.5 * torch.randn(
        300_000, 4, generator=gen, dtype=torch.float64
    )
    analysis = OneStepPosteriorOracle(H, r).run(
        LinearModel(A, q), ensemble, observations, gen
    )
    exact_var = torch.diagonal(cov, dim1=1, dim2=2)
    assert (analysis.var(1)[-1] - exact_var[-1]).sum() > 0.05


def test_multi_stage_filters_agree_on_lorenz96():
    # Both proposals target the same posterior, so they must agree on a nonlinear
    # model with non-Gaussian kernel (noise_stages > 1) as well.
    model = StochasticLorenz96(
        dim_state=6, dt=0.01, interval=0.1, noise_std=1.0, noise_stages=2
    )
    gen = torch.Generator().manual_seed(6)
    x = model.spin_up(1, 100, gen)[0]
    H = torch.eye(6, dtype=torch.float64)[::2]
    r = 1.0
    observations = []
    truth = x
    for _ in range(6):
        truth = model.step(truth, gen)
        observations.append(H @ truth + torch.randn(3, generator=gen, dtype=torch.float64))
    observations = torch.stack(observations)
    results = {}
    for proposal in ("bootstrap", "optimal"):
        particles = x[:, None] + torch.randn(6, 300_000, generator=gen, dtype=torch.float64)
        results[proposal] = BootstrapParticleFilter(H, r, proposal=proposal).run(
            model, particles, observations, gen
        )
    # RMS scale from replicates at this N: 0.011 for the mean, 0.03 for the variance.
    boot, opt = results["bootstrap"], results["optimal"]
    assert (boot.mean - opt.mean).pow(2).mean().sqrt() < 0.03
    assert (boot.variance - opt.variance).pow(2).mean().sqrt() < 0.08
    assert boot.log_evidence.sum().item() == pytest.approx(
        opt.log_evidence.sum().item(), abs=0.15
    )


def test_systematic_resample_counts_track_weights():
    gen = torch.Generator().manual_seed(7)
    w = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    idx = systematic_resample(w.log().repeat_interleave(1000), gen)
    counts = torch.bincount(idx // 1000, minlength=4).double() / 4000
    torch.testing.assert_close(counts, w, atol=1e-3, rtol=0)


def test_effective_sample_size_limits():
    assert effective_sample_size(torch.zeros(100)) == pytest.approx(100.0)
    degenerate = torch.full((100,), -1e9)
    degenerate[3] = 0.0
    assert effective_sample_size(degenerate) == pytest.approx(1.0)


def test_gaspari_cohn_is_compact_and_normalised():
    assert gaspari_cohn(torch.tensor(0.0), 2.0) == pytest.approx(1.0)
    assert gaspari_cohn(torch.tensor(4.0), 2.0) == pytest.approx(0.0, abs=1e-12)
    # Both polynomial branches, from the published formula, and their join at r = 1.
    assert gaspari_cohn(torch.tensor(1.5), 1.0) == pytest.approx(0.0164931, abs=1e-6)
    assert gaspari_cohn(torch.tensor(0.5), 1.0) == pytest.approx(0.6848958, abs=1e-6)
    below = gaspari_cohn(torch.tensor(1.0 - 1e-7, dtype=torch.float64), 1.0)
    above = gaspari_cohn(torch.tensor(1.0 + 1e-7, dtype=torch.float64), 1.0)
    assert below.item() == pytest.approx(above.item(), abs=1e-5)
    taper = periodic_taper(16, 2.0)
    torch.testing.assert_close(taper, taper.T)
    assert taper[0, 15].item() == pytest.approx(taper[0, 1].item())
    assert (taper.diagonal() == 1).all()


@pytest.mark.parametrize("d, radius", [(16, 5.0), (16, 8.0), (40, 12.0)])
def test_periodic_taper_is_positive_semidefinite_at_large_radius(d, radius):
    assert torch.linalg.eigvalsh(periodic_taper(d, radius)).min() > -1e-10
