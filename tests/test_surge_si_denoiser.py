"""SURGE on an SI prior: the reward's denoiser E[x_1 | x_t] (``si_denoiser``).

On an exact Gaussian stochastic interpolant (x_1 | x_0 ~ N(A x_0, Q) with the
closed-form Follmer drift) ``si_denoiser="follmer"`` gives E[x_1 | x_t] exactly,
``"velocity"`` (the default) keeps the historical values bit for bit, and the
unguided SMC with the exact denoiser keeps its weights near uniform at the start.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
from scisi.models.interpolations import LinearStochasticInterpolation
from scisi.posterior_models.surge_posterior import (
    MIN_TIME,
    SurgeStochasticInterpolantPosterior,
)

D = 16
G0 = 1.0


class ExactDrift(nn.Module):
    """Follmer drift of x_1 | x_0 ~ N(A x_0, diag(q)) on the linear SI
    (alpha = 1 - t, beta = t, gamma = g0 (1 - t)):
    b = (mu - x0) + (beta beta' Q + t gamma gamma' I) C_t^{-1} (x - m_t)."""

    def __init__(self, A: torch.Tensor, q: torch.Tensor, g0: float = G0) -> None:
        super().__init__()
        self.register_buffer("A", A)
        self.register_buffer("q", q)
        self.g0 = g0
        self._p = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x, t, field_history=None, field_cond=None, pars_cond=None):
        t = torch.as_tensor(t, dtype=x.dtype).reshape(-1, *([1] * (x.dim() - 1)))
        t = t.clamp(MIN_TIME, 1 - MIN_TIME)  # C_0 = 0: the first step evaluates t = 0
        x0 = field_history[..., -1]
        mu = x0 @ self.A.T
        gamma = self.g0 * (1 - t)
        cov = t**2 * self.q + t * gamma**2
        coef = (t * self.q - t * gamma * self.g0) / cov
        return (mu - x0) + coef * (x - ((1 - t) * x0 + t * mu))


def exact_posterior_mean(x, t, x0, A, q, g0=G0):
    """E[x_1 | x_t] = mu + beta Q C_t^{-1} (x - m_t)."""
    t = t.reshape(-1, *([1] * (x.dim() - 1)))
    mu = x0 @ A.T
    cov = t**2 * q + t * (g0 * (1 - t)) ** 2
    return mu + t * q / cov * (x - ((1 - t) * x0 + t * mu))


class Select(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x):
        return x.reshape(x.shape[0], -1)[:, self.idx]


class NullGuidance(nn.Module):
    def score(self, x, **_):
        return torch.zeros_like(x)

    def sde_weight(self, t, diffusion_term):
        return 0.0


@pytest.fixture
def system():
    gen = torch.Generator().manual_seed(0)
    A = 0.9 * torch.eye(D, dtype=torch.float64) + 0.05 * torch.randn(D, D, generator=gen, dtype=torch.float64)
    q = 0.2 + torch.rand(D, generator=gen, dtype=torch.float64)
    model = FollmerStochasticInterpolant(
        interpolation=LinearStochasticInterpolation(gamma_multiplier=G0),
        drift_model=ExactDrift(A, q),
    ).double()
    return A, q, model


def posterior(model, denoiser, **kw):
    return SurgeStochasticInterpolantPosterior(
        model=model, obs_operator=Select(torch.arange(0, D, 4)), variance=0.5,
        likelihood_model=NullGuidance(), si_denoiser=denoiser, **kw,
    )


@pytest.mark.parametrize("tau", [0.02, 0.1, 0.5, 0.9])
def test_follmer_denoiser_is_the_exact_conditional_mean(system, tau):
    A, q, model = system
    gen = torch.Generator().manual_seed(1)
    B = 32
    x0 = torch.randn(B, 1, 1, D, generator=gen, dtype=torch.float64)
    x = torch.randn(B, 1, 1, D, generator=gen, dtype=torch.float64) * 2
    t = torch.full((B,), tau, dtype=torch.float64)
    history = x0.unsqueeze(-1)
    exact = exact_posterior_mean(x, t, x0, A, q)
    follmer = posterior(model, "follmer")._denoise(x, t, history, None, None)
    torch.testing.assert_close(follmer, exact, rtol=1e-9, atol=1e-9)
    # The velocity denoiser treats b as the path velocity: biased.
    velocity = posterior(model, "velocity")._denoise(x, t, history, None, None)
    assert (velocity - exact).abs().max() > 1e-3


@pytest.mark.parametrize("tau", [0.02, 0.1, 0.5, 0.9])
def test_velocity_denoiser_is_the_historical_formula_bitwise(system, tau):
    _, _, model = system
    gen = torch.Generator().manual_seed(2)
    x0 = torch.randn(8, 1, 1, D, generator=gen, dtype=torch.float64)
    x = torch.randn(8, 1, 1, D, generator=gen, dtype=torch.float64)
    t = torch.full((8,), tau, dtype=torch.float64)
    history = x0.unsqueeze(-1)
    default = SurgeStochasticInterpolantPosterior(
        model=model, obs_operator=Select(torch.arange(4)), variance=0.5,
        likelihood_model=NullGuidance(),
    )
    assert default.si_denoiser == "velocity"
    out = default._denoise(x, t, history, None, None)
    # The formula before the option existed.
    tc = torch.clamp(t, min=MIN_TIME, max=1.0 - MIN_TIME).reshape(-1, 1, 1, 1)
    interp = model.interpolation
    v = model.drift(x, t, history, None, None)
    score = interp.score_from_velocity(x=x, v=v, t=tc, a0=x0)
    old = (x + interp.sigma(tc) ** 2 * score - interp.alpha(tc) * x0) / interp.beta(tc)
    assert torch.equal(out, old)


def test_unknown_denoiser_is_refused(system):
    with pytest.raises(ValueError, match="si_denoiser"):
        posterior(system[2], "tweedie")


def _ess_trace(model, denoiser, x0, y, steps=50):
    post = posterior(model, denoiser)
    trace = []
    original = post._maybe_resample

    def recorded(base, field_history, force=False):
        if post._log_w is not None:
            w = post._normalized_weights()
            trace.append(float(1.0 / (w**2).sum()) / base.shape[0])
        return original(base, field_history, force=force)

    post._maybe_resample = recorded
    torch.manual_seed(3)
    with torch.no_grad():
        post.sample(
            base=x0.clone(), batch_size=x0.shape[0], num_steps=steps,
            field_history=x0.unsqueeze(-1).clone(), observations=y,
        )
    return trace


def test_unguided_surge_with_the_exact_drift_keeps_ess_near_one_at_the_start(system):
    """From one anchor, every member's denoiser is near mu at small t, so the reward
    increments, hence the weights, are near uniform over the first pseudo-steps."""
    A, q, model = system
    E = 64
    gen = torch.Generator().manual_seed(4)
    x0 = torch.randn(1, 1, 1, D, generator=gen, dtype=torch.float64).repeat(E, 1, 1, 1)
    truth = x0[0, 0, 0] @ A.T + q.sqrt() * torch.randn(D, generator=gen, dtype=torch.float64)
    y = (truth[::4] + 0.5**0.5 * torch.randn(4, generator=gen, dtype=torch.float64))
    y = y.reshape(1, -1).repeat(E, 1)
    follmer = _ess_trace(model, "follmer", x0, y)
    assert len(follmer) >= 5
    assert min(follmer[:5]) > 0.95, follmer[:5]
    # The velocity denoiser's biased xhat_1 spreads the rewards from the start
    # (measured: ESS/N 0.35 after the first pseudo-step).
    assert _ess_trace(model, "velocity", x0, y)[0] < 0.9
