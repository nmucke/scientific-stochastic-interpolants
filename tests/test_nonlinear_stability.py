"""Regression tests for numerical failures exposed by the nonlinear-observation sweep.

CPU / float64 / fast. One test group per root cause:

1. ``inflated_shared`` + non-injective ``h`` (``square``): the ensemble-mean
   linearization ``J_bar`` gave members on the other branch an ASCENT step
   (``~ J_bar (y - c xhat_b^2)`` with fixed-sign ``J_bar`` -> Riccati blow-up to
   non-finite samples). Pointwise ``h`` now linearizes per member around the
   shared ``L Sigma_s L^T`` (fixed rule for a pointwise ``h``), which equals ``inflated`` whenever the network Jacobian is
   state-independent (the analytic prior); the ensemble-mean route (still used
   for non-pointwise ``h``) warns.
2. SDA nonlinear ``Gamma_b`` Cholesky: the reported "not positive-definite" was
   ``J J^T`` overflowing after the guided SDE had diverged (stiff explicit
   step); it is now factorised in float64 and a non-finite / non-PD ``Gamma``
   raises a descriptive ``FloatingPointError``.
3. Standalone SURGE dropped the paper's ``Sigma = g^2`` factor of the proposal
   drift (``w = guidance_scale`` instead of ``guidance_scale * g^2``): the SMC
   weights degenerated (ESS ~ 1) and the ensemble collapsed -- even for an
   identity ``h``.
"""

from __future__ import annotations

import sys
import warnings
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
pytestmark = pytest.mark.skipif(not _PE.exists(), reason="paper_experiments not present")
if str(_PE) not in sys.path:
    sys.path.insert(0, str(_PE))

from scisi.likelihood_models.gaussian_likelihood import (  # noqa: E402
    InterpolantGaussianLikelihood,
)
from scisi.likelihood_models.observation_operators import (  # noqa: E402
    LinearObservationOperator,
    PointwiseObservationOperator,
)
from scisi.likelihood_models.physics_observation_operators import (  # noqa: E402
    VelocityMagnitudeOperator,
)
from scisi.likelihood_models.sda import SDALikelihood  # noqa: E402

D = 2
N = 256
STEPS = 50
R_SMALL = 0.01
# Sweep cell nl:square_c1_R0.01_x00.5_d2, seed 0 (x0 near the + mode, y[0]
# small -> a sizeable fraction of the prior on the - branch).
X0 = (0.7076, 0.7483)
Y_SQUARE = (0.3403, 1.1384)


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _prior_models():
    from cases.analytical import prior_models

    return prior_models


def _pointwise(nl: str, d: int = D) -> PointwiseObservationOperator:
    lin = _prior_models().build_identity_obs_operator(d=d)
    return PointwiseObservationOperator(inner=lin, nonlinearity=nl, strength=1.0)


def _ours(sampler: str, mode: str, op, y, r=R_SMALL, n=N, seed=0, **kw) -> torch.Tensor:
    from scisi.posterior_models.flow_matching_posterior import (
        FlowMatchingPosterior,
        endpoint_vanishing_diffusion,
    )
    from scisi.posterior_models.stochastic_interpolant_posterior import (
        StochasticInterpolantPosterior,
    )

    pm = _prior_models()
    x0 = torch.tensor(X0).reshape(1, 1, 1, D)
    fh = x0.unsqueeze(-1).repeat(n, 1, 1, 1, 1)
    torch.manual_seed(seed)
    if sampler == "si_sde":
        model = pm.build_si_model(d=D, g0=1.0, prior_var=1.0)
        lik = InterpolantGaussianLikelihood(
            model=model, obs_operator=op, variance=r, model_class="si",
            likelihood_mode=mode, **kw,
        )
        post = StochasticInterpolantPosterior(model=model, likelihood_model=lik)
        base = x0.repeat(n, 1, 1, 1)
    else:
        model = pm.build_fm_model(d=D, prior_var=1.0)
        lik = InterpolantGaussianLikelihood(
            model=model, obs_operator=op, variance=r, model_class="fm",
            likelihood_mode=mode, **kw,
        )
        diff = (
            endpoint_vanishing_diffusion(model.interpolation, scale=1.0)
            if sampler == "dm_sde" else None
        )
        post = FlowMatchingPosterior(model=model, likelihood_model=lik, diffusion_term=diff)
        base = None
    with torch.no_grad():
        out = post.sample(
            base=base, batch_size=n, num_steps=STEPS, field_history=fh,
            observations=torch.tensor(y).reshape(1, D),
        )
    return out.reshape(n, D)


# ----------------------------------------------------------------------
# 1. inflated_shared with a non-injective pointwise h
# ----------------------------------------------------------------------
@pytest.mark.parametrize("sampler", ["si_sde", "dm_sde", "fm_ode"])
def test_shared_square_equals_inflated_for_state_independent_prior(sampler):
    """Analytic prior (network Jacobian state-independent): shared == inflated,
    and the bimodal square posterior stays finite and bounded (it diverged to
    non-finite / 1e15-scale samples with the ensemble-mean linearization)."""
    op = _pointwise("square")
    shared = _ours(sampler, "inflated_shared", op, Y_SQUARE)
    member = _ours(sampler, "inflated", op, Y_SQUARE)
    assert torch.isfinite(shared).all()
    assert float(shared.abs().max()) < 5.0
    torch.testing.assert_close(shared, member, rtol=1e-10, atol=1e-10)


def test_shared_linearization_rule_per_operator():
    """inflated_shared: pointwise h -> per-member J_h; other h -> ensemble mean."""
    pm = _prior_models()
    model = pm.build_fm_model(d=D, prior_var=1.0)
    lik = InterpolantGaussianLikelihood(
        model=model, obs_operator=_pointwise("square"), model_class="fm",
        likelihood_mode="inflated_shared",
    )
    assert lik._shared_member_linearization
    vel = VelocityMagnitudeOperator(
        inner=LinearObservationOperator(type="random", data_size=(1, 8, 8), percent_obs=0.25, seed=0),
    )
    model16 = pm.build_fm_model(d=64, prior_var=1.0)
    lik = InterpolantGaussianLikelihood(
        model=model16, obs_operator=vel, model_class="fm", likelihood_mode="inflated_shared",
    )
    assert not lik._shared_member_linearization


def _one_score(mean_linearization, x, t_val=0.6):
    pm = _prior_models()
    model = pm.build_fm_model(d=D, prior_var=1.0)
    lik = InterpolantGaussianLikelihood(
        model=model, obs_operator=_pointwise("square"), variance=R_SMALL,
        model_class="fm", likelihood_mode="inflated_shared",
    )
    if mean_linearization:  # force the ensemble-mean route used for non-pointwise h
        lik._shared_member_linearization = False
    b = x.shape[0]
    t = torch.full((b, 1), t_val)
    fh = torch.tensor(X0).reshape(1, 1, 1, D, 1).repeat(b, 1, 1, 1, 1)
    y = torch.tensor(Y_SQUARE).reshape(1, D).expand(b, D)
    s = model.score(x, t, fh)
    sbar, _ = lik.score(observations=y, x=x, t=t, field_history=fh, score=s)
    tg = t.reshape(b, 1, 1, 1)
    xhat = (x + model.interpolation.alpha(tg) ** 2 * s) / model.interpolation.beta(tg)
    return sbar, xhat, y


def test_shared_mean_linearization_ascent_mechanism_and_warning():
    """Root cause, isolated: member 0 sits on the NEGATIVE branch beyond
    -sqrt(y/c) while the ensemble mean is positive. The mean linearization
    steps it further away from its observation (and warns); the per-member
    route steps it toward the nearest mode."""
    x = torch.tensor([[-1.6, -1.6], [1.2, 1.2], [1.0, 1.0], [0.9, 0.9]]).reshape(4, 1, 1, D)

    def misfit_change(sbar, xhat, y, eps=1e-4):
        # d/d eps of ||y - g(xhat + eps sbar)||^2 for member 0 (xhat moves ~ +sbar).
        g = lambda z: z**2  # noqa: E731
        f0 = ((y[0] - g(xhat[0].reshape(-1))) ** 2).sum()
        f1 = ((y[0] - g(xhat[0].reshape(-1) + eps * sbar[0].reshape(-1))) ** 2).sum()
        return float(f1 - f0)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sb_mean, xhat, y = _one_score(True, x)
    assert any("ASCENT" in str(w.message) for w in rec)
    assert misfit_change(sb_mean, xhat, y) > 0  # ascent: moves away from y

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the per-member route must not warn
        sb_member, xhat, y = _one_score(False, x)
    assert misfit_change(sb_member, xhat, y) < 0  # descent


# ----------------------------------------------------------------------
# 2. SDA nonlinear Gamma solve
# ----------------------------------------------------------------------
def _sda(op, d):
    model = _prior_models().build_dm_model(d=d)
    return SDALikelihood(model=model, obs_operator=op, variance=1e-2, model_class="fm")


def test_sda_gamma_solve_float32_wide_dynamic_range():
    """rho J J^T >> sigma^2 (cubic h at |xhat| ~ 1e2..1e5) in float32: the solve
    matches a float64 dense reference and keeps the input dtype."""
    torch.set_default_dtype(torch.float32)
    L = LinearObservationOperator(type="super_res", data_size=(1, 4, 4), low_res=2, high_res=4)
    op = PointwiseObservationOperator(inner=L, nonlinearity="cubic", strength=1.0)
    lik = _sda(op, 16)
    for scale in (30.0, 1e3, 1e5):
        g = torch.Generator().manual_seed(0)
        xhat = scale * torch.randn(4, 1, 4, 4, generator=g)
        r = torch.randn(4, op.num_obs, generator=g)
        sol = lik._nonlinear_gamma_solve(xhat, torch.tensor(1e-2), r)
        assert sol.dtype == torch.float32 and torch.isfinite(sol).all()
        G = 1e-2 * op.gram(xhat.double()) + 1e-2 * torch.eye(op.num_obs, dtype=torch.float64)
        ref = torch.linalg.solve(G, r.double().unsqueeze(-1)).squeeze(-1)
        rel = float((sol.double() - ref).abs().max() / ref.abs().max())
        assert rel < 1e-5, (scale, rel)


def test_sda_gamma_solve_reports_divergence():
    """A diverged state (J J^T overflows float32) raises a descriptive
    FloatingPointError instead of an opaque Cholesky failure / silent garbage."""
    torch.set_default_dtype(torch.float32)
    op = _pointwise("cubic", d=4)
    lik = _sda(op, 4)
    xhat = torch.full((3, 1, 1, 4), 1e12)
    r = torch.ones(3, 4)
    with pytest.raises(FloatingPointError, match="diverged"):
        lik._nonlinear_gamma_solve(xhat, torch.tensor(1e-2), r)


# ----------------------------------------------------------------------
# 3. Standalone SURGE injection weight
# ----------------------------------------------------------------------
def _surge(nl, r, seed, n=512):
    from scisi.posterior_models.surge_posterior import SurgePosterior

    op = _pointwise(nl)
    torch.manual_seed(seed)
    dm = _prior_models().build_dm_model(d=D, prior_var=1.0, g0=1.0)
    post = SurgePosterior(
        model=dm, obs_operator=op, variance=r, guidance_scale=1.0,
        ess_threshold=0.5, diffusion_term=None,
    )
    x0 = torch.tensor(X0).reshape(1, 1, 1, D)
    fh = x0.unsqueeze(-1).repeat(n, 1, 1, 1, 1)
    y = op(torch.tensor([0.9, 0.2]).reshape(1, 1, 1, D))
    with torch.no_grad():
        out = post.sample(base=None, batch_size=n, num_steps=STEPS, field_history=fh, observations=y)
    return post, out.reshape(n, D)


def test_surge_standalone_injection_weight_is_g_squared():
    from scisi.posterior_models.surge_posterior import SurgePosterior

    post, _ = _surge("identity", R_SMALL, 0, n=4)
    assert isinstance(post, SurgePosterior)
    g = torch.tensor(0.3)
    assert float(post._injection_weight(torch.tensor([[0.5]]), g)) == pytest.approx(
        post.guidance_scale * 0.09
    )


@pytest.mark.parametrize("seed", [0, 1])
def test_surge_standalone_identity_not_collapsed(seed):
    """Identity h, R = 0.01: exact posterior variance R / (1 + R) per coordinate.
    With w = guidance_scale (no g^2) the SMC weights degenerated and the sample
    variance collapsed ~20x (~4e-4); with the paper's Sigma = g^2 it is O(R)."""
    _, out = _surge("identity", R_SMALL, seed)
    exact = R_SMALL / (1.0 + R_SMALL)
    ratio = out.var(0) / exact
    assert torch.isfinite(out).all()
    assert float(ratio.min()) > 0.3 and float(ratio.max()) < 2.0, ratio


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_surge_standalone_cubic_finite(seed):
    """cubic h, R = 0.25 (all-NaN on several sweep seeds before the fix)."""
    _, out = _surge("cubic", 0.25, seed)
    assert torch.isfinite(out).all()
    assert float(out.abs().max()) < 10.0
