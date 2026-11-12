"""TMPD baseline (``TMPDLikelihood`` / ``TMPDPosterior``) on exact linear-Gaussian priors.

CPU / float64 / fast. The prior ``x1 | x0 ~ N(A x0, Q)`` is the closed-form FM
velocity of the analytical case (``cases/analytical/linear_system.py``), read as
a diffusion model exactly as the drivers do for SDA.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_PE = (
    Path(__file__).resolve().parents[1]
    / "papers"
    / "A Unified Framework for Bayesian Data Assimilation with Generative Models "
    "and Observation Interpolants"
    / "code"
    / "paper_experiments"
)
if str(_PE) not in sys.path:
    sys.path.insert(0, str(_PE))

from cases.analytical.linear_system import (  # noqa: E402
    AnalyticalLinearFMVelocityModel,
    DenseObservationOperator,
    FullFirstStepDiffusionModel,
)
from cases.analytical.prior_models import build_identity_obs_operator  # noqa: E402

from scisi.likelihood_models.observation_operators import (  # noqa: E402
    LinearObservationOperator,
    PointwiseObservationOperator,
)
from scisi.likelihood_models.tmpd import MIN_TIME, TMPDLikelihood  # noqa: E402
from scisi.models.flow_matching_model import FlowMatchingModel  # noqa: E402
from scisi.models.interpolations import LinearDeterministicInterpolation  # noqa: E402
from scisi.posterior_models.flow_matching_posterior import (  # noqa: E402
    endpoint_vanishing_diffusion,
)
from scisi.posterior_models.tmpd_posterior import TMPDPosterior  # noqa: E402


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _spd(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(d, d, generator=g)
    return M @ M.T / d + 0.5 * torch.eye(d)


def _dm(A: torch.Tensor, Q: torch.Tensor, g0: float = 1.0):
    fm = FlowMatchingModel(
        interpolation=LinearDeterministicInterpolation(),
        drift_model=AnalyticalLinearFMVelocityModel(A, Q),
    ).double()
    # Full first step (keeps v dt at tau = 0) so the exactness test is not
    # polluted by the stock DM first-step convention.
    return FullFirstStepDiffusionModel.from_flow_matching(
        fm, diffusion_term=endpoint_vanishing_diffusion(fm.interpolation, scale=g0)
    )


def _closed_form(A, Q, H, R, x0, x, tau, y):
    """Exact p(y | x_tau) moments and its score on x_tau = (1-tau) z + tau x1."""
    d = Q.shape[0]
    mu = A @ x0
    a, b = 1.0 - tau, tau
    C = a**2 * torch.eye(d) + b**2 * Q
    Ci = torch.linalg.inv(C)
    Jm = b * Q @ Ci  # grad_x E[x1 | x_tau] (symmetric)
    m = mu + (x - b * mu) @ Jm.T  # [B, d]
    Sig = Q - b**2 * Q @ Ci @ Q
    S = H @ Sig @ H.T + R
    r = y - m @ H.T
    grad = (r @ torch.linalg.inv(S)) @ H @ Jm  # Jm^T H^T S^-1 r (row form)
    return m, S, grad


def _setup(d=3, d_y=2, seed=0):
    A = 0.8 * torch.eye(d)
    Q = _spd(d, seed)
    g = torch.Generator().manual_seed(seed + 1)
    H = torch.randn(d_y, d, generator=g)
    x0 = torch.randn(d, generator=g)
    y = torch.randn(d_y, generator=g)
    return A, Q, H, x0, y


def _fh(x0: torch.Tensor, B: int) -> torch.Tensor:
    d = x0.numel()
    return x0.reshape(1, 1, 1, d, 1).expand(B, 1, 1, d, 1).clone()


# --------------------------------------------------------------------------- #
# (a) shapes / ensemble batching
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cov", ["full", "diagonal", "row_sum"])
def test_shapes_and_per_member_batching(cov):
    A, Q, H, x0, y = _setup(d=4, d_y=3)
    B = 5
    dm = _dm(A, Q)
    op = DenseObservationOperator(H)
    lik = TMPDLikelihood(dm, op, variance=0.1, ensemble_size=B, covariance=cov)
    x = torch.randn(B, 1, 1, 4, generator=torch.Generator().manual_seed(7))
    t = torch.full((1, 1), 0.4)
    obs = y.expand(B, -1)
    guid, ll = lik.score(obs, x, t, _fh(x0, B))
    assert guid.shape == x.shape and ll.shape == (B,)
    assert torch.isfinite(guid).all() and torch.isfinite(ll).all()
    # Each member equals its own single-member evaluation.
    for i in range(B):
        gi, li = lik.score(obs[i : i + 1], x[i : i + 1], t, _fh(x0, 1))
        torch.testing.assert_close(gi, guid[i : i + 1])
        torch.testing.assert_close(li, ll[i : i + 1])
    # The caller's tensor is not turned into a grad leaf.
    assert not x.requires_grad


def test_selection_operator_shapes():
    d = 6
    Q = _spd(d, 3)
    dm = _dm(torch.eye(d), Q)
    op = LinearObservationOperator(type="random", data_size=(1, 1, d), percent_obs=0.5, seed=0)
    lik = TMPDLikelihood(dm, op, variance=0.05, covariance="full")
    x = torch.randn(3, 1, 1, d)
    guid, ll = lik.score(torch.randn(3, op.num_obs), x, torch.full((3, 1), 0.7), _fh(torch.zeros(d), 3))
    assert guid.shape == x.shape and ll.shape == (3,)


# --------------------------------------------------------------------------- #
# (b) moments, guidance and exact posterior on a linear-Gaussian prior
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tau", [0.05, 0.3, 0.6, 0.95])
def test_moments_and_guidance_match_closed_form(tau):
    A, Q, H, x0, y = _setup()
    R = 0.1 * torch.eye(2)
    B = 4
    dm = _dm(A, Q)
    lik = TMPDLikelihood(dm, DenseObservationOperator(H), variance=0.1, covariance="full")
    x = torch.randn(B, 1, 1, 3, generator=torch.Generator().manual_seed(11))
    t = torch.full((1, 1), tau)
    out = lik.guided_terms(y.expand(B, -1), x, t, _fh(x0, B))
    m, S, grad = _closed_form(A, Q, H, R, x0, x.reshape(B, 3), tau, y)
    torch.testing.assert_close(out["xhat1"].reshape(B, 3), m, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out["guidance"].reshape(B, 3), grad, rtol=1e-5, atol=1e-6)
    r = y - m @ H.T
    ll = torch.distributions.MultivariateNormal(torch.zeros(2), S).log_prob(r)
    torch.testing.assert_close(out["log_likelihood"], ll, rtol=1e-5, atol=1e-6)


def test_full_posterior_matches_exact_gaussian_posterior():
    """TMPD Prop. 3: exact posterior sampling for Gaussian data."""
    torch.manual_seed(0)
    A, Q, H, x0, y = _setup()
    R = 0.1 * torch.eye(2)
    dm = _dm(A, Q)
    lik = TMPDLikelihood(dm, DenseObservationOperator(H), variance=0.1, covariance="full")
    post = TMPDPosterior(model=dm, likelihood_model=lik)
    n = 4000
    with torch.no_grad():
        out = post.sample(
            base=None, batch_size=n, num_steps=400,
            field_history=_fh(x0, n), observations=y.expand(n, -1),
        ).reshape(n, 3)
    mu = A @ x0
    K = Q @ H.T @ torch.linalg.inv(H @ Q @ H.T + R)
    m_post = mu + K @ (y - H @ mu)
    P_post = (torch.eye(3) - K @ H) @ Q
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.mean(0), m_post, atol=0.06, rtol=0.0)
    torch.testing.assert_close(torch.cov(out.T), P_post, atol=0.06, rtol=0.0)


# --------------------------------------------------------------------------- #
# (c) diagonal / row_sum == full when H C H^T is diagonal
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cov", ["diagonal", "row_sum"])
def test_diagonal_variants_equal_full_when_HCHt_diagonal(cov):
    d = 4
    Q = torch.diag(torch.tensor([0.5, 1.0, 2.0, 0.3]))
    H = torch.eye(d)[[0, 2, 3]]  # selection -> H C H^T diagonal
    dm = _dm(torch.eye(d), Q)
    op = DenseObservationOperator(H)
    x = torch.randn(3, 1, 1, d, generator=torch.Generator().manual_seed(5))
    y = torch.randn(3, 3)
    t = torch.full((1, 1), 0.45)
    fh = _fh(torch.randn(d), 3)
    full = TMPDLikelihood(dm, op, variance=0.2, covariance="full").score(y, x, t, fh)
    diag = TMPDLikelihood(dm, op, variance=0.2, covariance=cov).score(y, x, t, fh)
    torch.testing.assert_close(diag[0], full[0])
    torch.testing.assert_close(diag[1], full[1])


def test_diagonal_variants_differ_when_HCHt_not_diagonal():
    A, Q, H, x0, y = _setup()
    dm = _dm(A, Q)
    x = torch.randn(2, 1, 1, 3)
    t = torch.full((1, 1), 0.5)
    outs = {
        c: TMPDLikelihood(dm, DenseObservationOperator(H), 0.1, covariance=c).score(
            y.expand(2, -1), x, t, _fh(x0, 2)
        )[0]
        for c in ("full", "diagonal", "row_sum")
    }
    assert not torch.allclose(outs["full"], outs["diagonal"])
    assert not torch.allclose(outs["full"], outs["row_sum"])


# --------------------------------------------------------------------------- #
# (d) tau < MIN_TIME, (e) nonlinear operator, misc
# --------------------------------------------------------------------------- #


def test_zero_guidance_below_min_time():
    A, Q, H, x0, y = _setup()
    dm = _dm(A, Q)
    lik = TMPDLikelihood(dm, DenseObservationOperator(H), 0.1)
    x = torch.randn(2, 1, 1, 3)
    for tau in (0.0, 0.5 * MIN_TIME):
        g, ll = lik.score(y.expand(2, -1), x, torch.full((1, 1), tau), _fh(x0, 2))
        assert torch.equal(g, torch.zeros_like(x))
        assert torch.equal(ll, torch.zeros(2))


def test_nonlinear_operator_raises():
    d = 3
    dm = _dm(torch.eye(d), torch.eye(d))
    op = PointwiseObservationOperator(inner=build_identity_obs_operator(d=d), nonlinearity="arctan")
    with pytest.raises(NotImplementedError, match="LINEAR"):
        TMPDLikelihood(dm, op, 0.1)
    # Even the identity nonlinearity is flagged (is_linear False).
    op_id = PointwiseObservationOperator(inner=build_identity_obs_operator(d=d), nonlinearity="identity")
    with pytest.raises(NotImplementedError):
        TMPDLikelihood(dm, op_id, 0.1)


def test_invalid_options():
    dm = _dm(torch.eye(2), torch.eye(2))
    op = build_identity_obs_operator(d=2)
    with pytest.raises(ValueError):
        TMPDLikelihood(dm, op, 0.1, covariance="bogus")
    with pytest.raises(ValueError):
        TMPDLikelihood(dm, op, 0.1, model_class="si")
    with pytest.raises(ValueError):
        TMPDPosterior(dm, TMPDLikelihood(dm, op, 0.1), guidance_weight="bogus")


def test_reduces_to_prior_sampler_without_observation_information():
    """Huge noise variance: guidance -> 0 and TMPDPosterior reproduces the
    unguided DM reverse SDE (``model.sample``) path-by-path under the same seed."""
    A, Q, H, x0, y = _setup()
    dm = _dm(A, Q)
    lik = TMPDLikelihood(dm, DenseObservationOperator(H), variance=1e12)
    post = TMPDPosterior(dm, lik)
    n, M = 200, 50
    torch.manual_seed(1)
    with torch.no_grad():
        out = post.sample(
            base=None, batch_size=n, num_steps=M,
            field_history=_fh(x0, n), observations=y.expand(n, -1),
        )
    torch.manual_seed(1)
    with torch.no_grad():
        ref = dm.sample(field_history=_fh(x0, n), batch_size=n, num_steps=M)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)
