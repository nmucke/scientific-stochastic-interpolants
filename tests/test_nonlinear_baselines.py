"""Nonlinear observation operators in every BASELINE posterior sampler.

CPU / float64 / fast. Uses the closed-form analytical prior models of the paper's
analytical case (``cases/analytical/prior_models.py``) and constructs/invokes each
baseline exactly as that case's ``driver.py`` (``_sample_and_count``) does.

For every baseline:

(a) ``PointwiseObservationOperator(nonlinearity="identity")`` reproduces the
    linear operator's samples under fixed seeds;
(b) ``h = arctan`` (c = 1) and ``h = square`` (c = 1) run end to end on d = 2
    and return finite samples of the right shape.

Plus dense-Jacobian brute-force checks of ``ObservationOperator.gram``, the
per-member OT-ODE / SDA guidance, the SDA ``Gamma^{-1}`` solve, FIG's
linearization and the EnSF linearized forecast variance.
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

from cases.analytical.prior_models import (  # noqa: E402
    build_dm_model,
    build_fm_model,
    build_identity_obs_operator,
    build_si_model,
)

from scisi.likelihood_models.dflow import DFlowSGLDLikelihood  # noqa: E402
from scisi.likelihood_models.gaussian_likelihood import (  # noqa: E402
    FlowdasGaussianLikelihood,
)
from scisi.likelihood_models.guidance import (  # noqa: E402
    DPSGaussianLikelihood,
    FIGGaussianLikelihood,
    GuidanceGaussianLikelihood,
)
from scisi.likelihood_models.observation_operators import (  # noqa: E402
    LinearizedObservationOperator,
    LinearObservationOperator,
    ObservationOperator,
    PointwiseObservationOperator,
)
from scisi.likelihood_models.physics_observation_operators import (  # noqa: E402
    VelocityMagnitudeOperator,
)
from scisi.likelihood_models.sda import SDALikelihood  # noqa: E402
from scisi.particle_filter.ensemble_score_filter import (  # noqa: E402
    EnsembleScoreFilter,
)
from scisi.posterior_models.diffusion_posterior import DiffusionPosterior  # noqa: E402
from scisi.posterior_models.dflow_posterior import DFlowPosterior  # noqa: E402
from scisi.posterior_models.flow_matching_posterior import (  # noqa: E402
    FlowMatchingPosterior,
)
from scisi.posterior_models.stochastic_interpolant_posterior import (  # noqa: E402
    StochasticInterpolantPosterior,
)
from scisi.posterior_models.surge_posterior import (  # noqa: E402
    SurgeFlowMatchingPosterior,
    SurgeStochasticInterpolantPosterior,
)

D = 2
N = 16  # ensemble size
M = 12  # pseudo-time steps
OBS_VAR = 1.0
PRIOR_VAR = 1.0
G0 = 1.0
NONLINEAR = (("arctan", 1.0), ("square", 1.0))


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _op(kind: str):
    """Linear identity H, or h = g(H x) with H the analytical identity sensor map."""
    lin = build_identity_obs_operator(d=D)
    if kind == "linear":
        return lin
    if kind == "identity":
        return PointwiseObservationOperator(inner=lin, nonlinearity="identity")
    return PointwiseObservationOperator(
        inner=lin, nonlinearity=kind, strength=dict(NONLINEAR)[kind]
    )


def _inputs():
    g = torch.Generator().manual_seed(1234)
    x0 = torch.randn(D, generator=g)
    y = x0 + 0.5 * torch.randn(D, generator=g)
    x0_4d = x0.reshape(1, 1, 1, D)
    fh = x0_4d.unsqueeze(-1).repeat(N, 1, 1, 1, 1)
    base_si = x0_4d.repeat(N, 1, 1, 1)
    return fh, base_si, y.reshape(1, D)


# ----------------------------------------------------------------------
# Baseline builders (mirroring driver._sample_and_count)
# ----------------------------------------------------------------------
def _run(method: str, obs_op) -> torch.Tensor:
    fh, base_si, y_obs = _inputs()
    base = None
    if method == "flowdas":
        model = build_si_model(d=D, g0=G0, prior_var=PRIOR_VAR)
        lik = FlowdasGaussianLikelihood(
            model=model, obs_operator=obs_op, variance=OBS_VAR,
            num_mc_samples=5, guidance_scale=1.0,
        )
        post = StochasticInterpolantPosterior(model=model, likelihood_model=lik)
        base = base_si
    elif method == "dflow":
        model = build_fm_model(d=D, prior_var=PRIOR_VAR)
        kw = dict(num_optim_steps=M, step_size=5e-3, noise_scale=1e-3, lambda_reg=1e-4)
        lik = DFlowSGLDLikelihood(
            model=model, obs_operator=obs_op, variance=OBS_VAR, **kw
        )
        post = DFlowPosterior(
            model=model, likelihood_model=lik, variance=OBS_VAR, **kw
        )
    elif method == "sda":
        model = build_dm_model(d=D, prior_var=PRIOR_VAR, g0=G0)
        lik = SDALikelihood(
            model=model, obs_operator=obs_op, variance=OBS_VAR,
            model_class="fm", gamma_sda=1e-2,
        )
        post = DiffusionPosterior(model=model, likelihood_model=lik)
    elif method == "surge_sda":
        model = build_dm_model(d=D, prior_var=PRIOR_VAR, g0=G0)
        lik = SDALikelihood(
            model=model, obs_operator=obs_op, variance=OBS_VAR,
            model_class="fm", gamma_sda=1e-2,
        )
        post = SurgeFlowMatchingPosterior(
            model=model, obs_operator=obs_op, variance=OBS_VAR,
            likelihood_model=lik, ess_threshold=0.5, diffusion_term=None,
        )
    elif method == "surge_flowdas":
        model = build_si_model(d=D, g0=G0, prior_var=PRIOR_VAR)
        lik = FlowdasGaussianLikelihood(
            model=model, obs_operator=obs_op, variance=OBS_VAR,
            num_mc_samples=5, guidance_scale=1.0,
        )
        post = SurgeStochasticInterpolantPosterior(
            model=model, obs_operator=obs_op, variance=OBS_VAR,
            likelihood_model=lik, ess_threshold=0.5, diffusion_term=None,
        )
        base = base_si
    elif method in ("fig", "fig_noise"):
        model = build_fm_model(d=D, prior_var=PRIOR_VAR)
        lik = FIGGaussianLikelihood(
            model=model, obs_operator=obs_op, variance=OBS_VAR,
            guidance_steps=2, guidance_scale=1.0,
            interpolant_noise=0.3 if method == "fig_noise" else 0.0,
        )
        post = FlowMatchingPosterior(
            model=model, likelihood_model=lik, diffusion_term=None
        )
    elif method in ("ot_ode", "guided_simple"):
        model = build_fm_model(d=D, prior_var=PRIOR_VAR)
        lik = GuidanceGaussianLikelihood(
            model=model, obs_operator=obs_op, variance=OBS_VAR,
            weighting="simple" if method == "guided_simple" else "ot_ode",
        )
        post = FlowMatchingPosterior(
            model=model, likelihood_model=lik, diffusion_term=None
        )
    elif method == "dps_si":
        model = build_si_model(d=D, g0=G0, prior_var=PRIOR_VAR)
        lik = DPSGaussianLikelihood(
            model=model, obs_operator=obs_op, variance=OBS_VAR, model_class="si"
        )
        post = StochasticInterpolantPosterior(model=model, likelihood_model=lik)
        base = base_si
    elif method == "dps_dm":
        model = build_dm_model(d=D, prior_var=PRIOR_VAR, g0=G0)
        lik = DPSGaussianLikelihood(
            model=model, obs_operator=obs_op, variance=OBS_VAR, model_class="fm"
        )
        post = DiffusionPosterior(model=model, likelihood_model=lik)
    elif method == "ensf":
        g = torch.Generator().manual_seed(7)
        forecast = fh[..., -1] + torch.randn(N, 1, 1, D, generator=g)
        return EnsembleScoreFilter(analysis_steps=M)._analysis_update(
            forecast=forecast,
            observations=y_obs.expand(N, -1),
            obs_operator=obs_op,
            variance=OBS_VAR,
            generator=torch.Generator().manual_seed(11),
        ).reshape(N, D)
    else:
        raise ValueError(method)

    torch.manual_seed(0)
    with torch.no_grad():
        out = post.sample(
            base=base, batch_size=N, num_steps=M,
            field_history=fh, observations=y_obs,
        )
    return out.reshape(N, D)


METHODS = (
    "flowdas", "dflow", "sda", "surge_sda", "surge_flowdas",
    "fig", "fig_noise", "ot_ode", "guided_simple",
    "dps_si", "dps_dm", "ensf",
)


@pytest.mark.parametrize("method", METHODS)
def test_identity_matches_linear(method):
    """(a) h = identity o H reproduces the linear operator under fixed seeds."""
    lin = _run(method, _op("linear"))
    ident = _run(method, _op("identity"))
    assert torch.isfinite(lin).all()
    torch.testing.assert_close(ident, lin, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("kind", [k for k, _ in NONLINEAR])
@pytest.mark.parametrize("method", METHODS)
def test_nonlinear_runs(method, kind):
    """(b) arctan / square run end to end with finite samples of the right shape."""
    out = _run(method, _op(kind))
    assert out.shape == (N, D)
    assert torch.isfinite(out).all()


# ----------------------------------------------------------------------
# gram / per-member preconditioners vs dense Jacobians
# ----------------------------------------------------------------------
class _SinMix(ObservationOperator):
    """Generic (non-pointwise) nonlinear h: exercises the torch.func defaults."""

    def __init__(self, inner: LinearObservationOperator) -> None:
        super().__init__()
        self.inner = inner
        self.C, self.H, self.W = inner.C, inner.H, inner.W
        self.num_obs = inner.num_obs

    def forward(self, x):
        z = self.inner(x)
        return torch.sin(z) + 0.3 * z * z.sum(dim=1, keepdim=True)


def _dense_jac(op, x1):
    f = lambda v: op(v.reshape(1, op.C, op.H, op.W)).reshape(-1)  # noqa: E731
    return torch.autograd.functional.jacobian(f, x1.reshape(-1))


def _operators():
    sup = LinearObservationOperator(type="super_res", data_size=(1, 4, 4), low_res=2)
    rnd = LinearObservationOperator(
        type="random", data_size=(1, 4, 4), percent_obs=0.5, seed=3
    )
    return {
        "arctan_superres": PointwiseObservationOperator(
            inner=sup, nonlinearity="arctan", strength=1.5
        ),
        "square_random": PointwiseObservationOperator(inner=rnd, nonlinearity="square"),
        "identity_random": PointwiseObservationOperator(inner=rnd, nonlinearity="identity"),
        "sinmix_superres": _SinMix(sup),
        "speed_random": VelocityMagnitudeOperator(inner=rnd, norm_std=1.0),
    }


OP_NAMES = list(_operators())


@pytest.mark.parametrize("name", OP_NAMES)
def test_gram_matches_dense(name):
    op = _operators()[name]
    x = torch.randn(3, 1, 4, 4, generator=torch.Generator().manual_seed(len(name)))
    G = op.gram(x)
    G_small = op.gram(x, chunk_elems=16)  # several chunks of the default build
    assert G.shape == (3, op.num_obs, op.num_obs)
    for b in range(3):
        J = _dense_jac(op, x[b])
        torch.testing.assert_close(op.jacobian(x[b]), J, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(G[b], J @ J.T, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(G_small[b], J @ J.T, rtol=1e-12, atol=1e-12)


def test_identity_gram_is_exact_HHt():
    lin = LinearObservationOperator(type="super_res", data_size=(1, 4, 4), low_res=2)
    op = PointwiseObservationOperator(inner=lin, nonlinearity="identity")
    x = torch.randn(1, 1, 4, 4)
    H = lin.obs_matrix.to(torch.float64)
    assert torch.equal(op.gram(x)[0], H @ H.T)
    assert torch.equal(lin.gram(x), H @ H.T)


def test_linear_gram_cache_follows_obs_matrix(tmp_path):
    op = LinearObservationOperator(type="random", data_size=(1, 4, 4), percent_obs=0.5, seed=3)
    G1 = op.gram()
    assert op.gram() is G1  # cached
    other = LinearObservationOperator(type="random", data_size=(1, 4, 4), percent_obs=0.5, seed=8)
    other.save_mask(str(tmp_path / "m.pt"))
    op.load_mask(str(tmp_path / "m.pt"))
    H = op.obs_matrix.to(torch.float64)
    assert op.gram() is not G1 and torch.equal(op.gram(), H @ H.T)


@pytest.mark.parametrize("name", ["arctan_superres", "square_random", "sinmix_superres"])
def test_sda_gamma_solve_matches_dense(name):
    op = _operators()[name]
    model = build_dm_model(d=16)
    lik = SDALikelihood(model=model, obs_operator=op, variance=0.3, model_class="fm")
    g = torch.Generator().manual_seed(5)
    xhat = torch.randn(4, 1, 4, 4, generator=g)
    r = torch.randn(4, op.num_obs, generator=g)
    rho = torch.tensor(0.7)
    sol = lik._nonlinear_gamma_solve(xhat, rho, r)
    for b in range(4):
        J = _dense_jac(op, xhat[b])
        Gamma = 0.3 * torch.eye(op.num_obs) + rho * J @ J.T
        torch.testing.assert_close(
            sol[b], torch.linalg.solve(Gamma, r[b]), rtol=1e-10, atol=1e-10
        )


def test_sda_score_matches_dense():
    """SDA nonlinear score == autograd of 1/2 r_b^T Gamma_b^{-1} r_b, Gamma_b fixed,
    J_b = dense J_h(xhat1_b) per member."""
    op = PointwiseObservationOperator(
        inner=build_identity_obs_operator(d=D), nonlinearity="arctan"
    )
    model = build_dm_model(d=D, prior_var=PRIOR_VAR, g0=G0)
    lik = SDALikelihood(model=model, obs_operator=op, variance=0.5, model_class="fm",
                        gamma_sda=0.3)
    fh, _, y = _inputs()
    B = 5
    fh, y = fh[:B], y.expand(B, -1)
    x = torch.randn(B, 1, 1, D, generator=torch.Generator().manual_seed(3))
    t = torch.full((B, 1), 0.4)
    s, _ = lik.score(observations=y, x=x, t=t, field_history=fh)
    xg = x.clone().requires_grad_(True)
    xhat, rho = lik._denoise(xg, t, fh, None, None, None)
    total = 0.0
    for b in range(B):
        J = _dense_jac(op, xhat[b].detach())
        Gamma = 0.5 * torch.eye(D) + rho.detach() * J @ J.T
        r = y[b] - op(xhat[b : b + 1])[0]
        total = total + 0.5 * r @ torch.linalg.solve(Gamma, r)
    (grad,) = torch.autograd.grad(total, xg)
    torch.testing.assert_close(s, -grad, rtol=1e-10, atol=1e-10)


def test_ot_ode_preconditioner_matches_dense():
    """OT-ODE nonlinear guidance == brute force with per-member
    M_b = r^2 J_b J_b^T + s^2 I, J_b = dense J_h(xhat_1_b)."""
    op = PointwiseObservationOperator(
        inner=build_identity_obs_operator(d=D), nonlinearity="arctan"
    )
    model = build_fm_model(d=D)
    lik = GuidanceGaussianLikelihood(
        model=model, obs_operator=op, variance=0.5, weighting="ot_ode"
    )
    fh, _, y = _inputs()
    B = 5
    fh = fh[:B]
    y = y.expand(B, -1)
    x = torch.randn(B, 1, 1, D, generator=torch.Generator().manual_seed(2))
    t = torch.full((B, 1), 0.4)
    drift = model.drift(x, t, fh, None, None)
    guid, _ = lik.score(observations=y, x=x, t=t, field_history=fh, drift=drift)

    r2 = max(0.6**2 / (0.6**2 + 0.4**2), 1e-2)
    xhat_ref = x + 0.6 * drift
    xg = x.clone().requires_grad_(True)
    xhat = xg + 0.6 * model.drift(xg, t, fh, None, None)
    quad = 0.0
    for b in range(B):
        J = _dense_jac(op, xhat_ref[b])
        Minv = torch.linalg.inv(r2 * J @ J.T + 0.5 * torch.eye(D))
        res = y[b] - op(xhat[b : b + 1])[0]
        quad = quad + 0.5 * res @ Minv @ res
    (grad,) = torch.autograd.grad(quad, xg)
    torch.testing.assert_close(guid, -grad, rtol=1e-10, atol=1e-10)


def test_linearize_is_the_linear_map():
    """``linearize`` is J only (no offset): J x == jvp, obs_matrix == dense J, and
    FIG's offset b = h(xhat) - J xhat makes J x + b the Taylor model."""
    op = PointwiseObservationOperator(
        inner=LinearObservationOperator(
            type="super_res", data_size=(1, 4, 4), low_res=2
        ),
        nonlinearity="arctan",
    )
    g = torch.Generator().manual_seed(9)
    xhat = torch.randn(1, 1, 4, 4, generator=g)
    x = torch.randn(1, 1, 4, 4, generator=g)
    lin = op.linearize(xhat)
    assert isinstance(lin, LinearizedObservationOperator) and lin.is_linear
    torch.testing.assert_close(lin.obs_matrix, _dense_jac(op, xhat), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(lin(x), op.jvp(xhat, x), rtol=1e-12, atol=1e-12)
    J = lin.obs_matrix
    torch.testing.assert_close(lin.gram(), J @ J.T, rtol=0, atol=0)
    b = op(xhat) - op.jvp(xhat, xhat)
    torch.testing.assert_close(lin(xhat) + b, op(xhat), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", ["arctan_superres", "square_random"])
def test_ensf_forecast_variance_matches_dense(name):
    """Pointwise EnSF var term g'(L mu)^2 (L var) == diag(J diag(var) J^T) for selection L."""
    op = _operators()[name]
    g = torch.Generator().manual_seed(4)
    forecast = torch.randn(8, 1, 4, 4, generator=g)
    mu, var = EnsembleScoreFilter._empirical_gaussian(forecast)
    v = EnsembleScoreFilter._forecast_obs_variance(forecast, mu, var, op)
    J = _dense_jac(op, mu)
    dense = torch.diagonal(J @ torch.diag(var.reshape(-1)) @ J.T)
    if op.inner.type == "random":
        torch.testing.assert_close(v.reshape(-1), dense, rtol=1e-12, atol=1e-12)
    else:
        # Block average: same L-var rule as the linear path (H var, not (H.H) var).
        dg = op.dg(op.inner(mu)).reshape(-1)
        torch.testing.assert_close(
            v.reshape(-1), dg**2 * op.inner(var).reshape(-1), rtol=1e-12, atol=1e-12
        )


def test_ensf_obs_score_is_vjp():
    """EnSF nonlinear obs score uses J_h(z)^T u per member (dense check)."""
    op = _operators()["arctan_superres"]
    g = torch.Generator().manual_seed(6)
    z = torch.randn(3, 1, 4, 4, generator=g)
    u = torch.randn(3, op.num_obs, generator=g)
    out = op.vjp(z, u)
    for b in range(3):
        J = _dense_jac(op, z[b])
        torch.testing.assert_close(out[b].reshape(-1), J.T @ u[b])
