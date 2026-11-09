"""Nonlinear observation-operator support of ``InterpolantGaussianLikelihood``.

CPU / float64 / fast. Covers:

- a frozen linear-operator regression fixture (``tests/fixtures/``) recorded
  from the pre-nonlinear implementation: linear operators must reproduce it;
- ``PointwiseObservationOperator(identity)`` == the linear operator;
- Sbar for a nonlinear ``h`` == a brute-force dense-Jacobian formula (pointwise,
  generic torch.func and velocity-magnitude operators; ensemble-mean
  linearization in ``inflated_shared``, per member otherwise);
- ``inflated_shared`` with one member == ``inflated``;
- statistical (Gauss-Hermite) linearization: identity == linear, and it matches
  a brute-force formula;
- 1-D end-to-end: SI-SDE / DM-SDE / FM-ODE posteriors with ``h = arctan`` vs
  1-D quadrature of the exact posterior.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from scisi.likelihood_models.gaussian_likelihood import (
    GH_NODES,
    InterpolantGaussianLikelihood,
)
from scisi.likelihood_models.observation_operators import (
    LinearObservationOperator,
    ObservationOperator,
    PointwiseObservationOperator,
)
from scisi.likelihood_models.physics_observation_operators import (
    VelocityMagnitudeOperator,
)
from scisi.models.flow_matching_model import FlowMatchingModel
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
from scisi.models.interpolations import (
    LinearDeterministicInterpolation,
    LinearStochasticInterpolation,
)

FIXTURE = Path(__file__).parent / "fixtures" / "nonlinear_likelihood_linear_reference.pt"

DATA_SIZE = (1, 4, 4)  # d = 16
B = 3
MODES = ("inflated", "inflated_shared", "dps_jacobian_free")
MODEL_CLASSES = ("si", "fm")
TIMES = (0.1, 0.5, 0.9)
OPERATORS = ("random", "super_res")

# Extra stabilisation-knob combos exercised by the regression fixture.
KNOB_CASES = {
    "plain": {},
    "floor_cap": dict(sigma_bar_eig_floor=True, residual_step_cap=0.3),
    "isofront_damp": dict(isotropic_front_factor=True, jacobian_damping=0.5),
    "lagged": dict(jacobian_refresh_every=2),
}


# ----------------------------------------------------------------------
# Tiny deterministic setup
# ----------------------------------------------------------------------
class TinyMLP(nn.Module):
    """``f(x, t)`` on ``[B, C, H, W]`` states: a small tanh MLP (nontrivial J)."""

    def __init__(self, d: int, hidden: int = 32, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w1 = nn.Parameter(torch.randn(hidden, d + 1, generator=g, dtype=torch.float64) / math.sqrt(d))
        self.b1 = nn.Parameter(0.1 * torch.randn(hidden, generator=g, dtype=torch.float64))
        self.w2 = nn.Parameter(torch.randn(d, hidden, generator=g, dtype=torch.float64) / math.sqrt(hidden))
        self.lin = nn.Parameter(-0.5 * torch.eye(d, dtype=torch.float64))

    def forward(self, x, t, field_history=None, field_cond=None, pars_cond=None):
        b = x.shape[0]
        xf = x.reshape(b, -1)
        tf = t.reshape(t.shape[0], -1)[:, :1].to(xf.dtype).expand(b, 1)
        h = torch.tanh(torch.cat([xf, tf], dim=1) @ self.w1.T + self.b1)
        out = h @ self.w2.T + xf @ self.lin.T
        return out.reshape(x.shape)


def build_model(model_class: str, d: int = 16, seed: int = 0):
    net = TinyMLP(d, seed=seed)
    if model_class == "si":
        return FollmerStochasticInterpolant(
            interpolation=LinearStochasticInterpolation(gamma_multiplier=1.0),
            drift_model=net,
        )
    return FlowMatchingModel(
        interpolation=LinearDeterministicInterpolation(), drift_model=net
    )


def build_linear_operator(kind: str) -> LinearObservationOperator:
    if kind == "random":
        return LinearObservationOperator(
            type="random", data_size=DATA_SIZE, percent_obs=0.5, seed=3
        )
    return LinearObservationOperator(
        type="super_res", data_size=DATA_SIZE, low_res=2, high_res=4
    )


def make_inputs(op: ObservationOperator, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, *DATA_SIZE, generator=g, dtype=torch.float64)
    x0 = torch.randn(B, *DATA_SIZE, generator=g, dtype=torch.float64)
    fh = x0.unsqueeze(-1)  # [B, C, H, W, 1]
    y = torch.randn(1, op.num_obs, generator=g, dtype=torch.float64).expand(B, -1)
    return x, fh, y.contiguous()


def call_score(lik, model, model_class, x, fh, y, t_val):
    t = torch.full((x.shape[0], 1), float(t_val), dtype=torch.float64)
    if model_class == "si":
        drift = model.drift(x, t, fh)
        return lik.score(observations=y, x=x, t=t, field_history=fh, drift=drift)
    score = model.score(x, t, fh)
    return lik.score(observations=y, x=x, t=t, field_history=fh, score=score)


def make_lik(model, op, mode, model_class, variance=0.05, **kw):
    return InterpolantGaussianLikelihood(
        model=model,
        obs_operator=op,
        variance=variance,
        likelihood_mode=mode,
        model_class=model_class,
        **kw,
    )


def compute_linear_reference() -> dict:
    """Sbar / log-lik of the linear operators over the fixture grid.

    The fixture stores this as ONE stacked dict ``{"keys": [K], "sbar": [K, B,
    C, H, W], "loglik": [K, B]}`` (see :func:`load_reference`). Regenerate it
    only from a trusted linear implementation::

        ref = compute_linear_reference(); keys = list(ref)
        torch.save({"keys": keys,
                    "sbar": torch.stack([ref[k][0] for k in keys]),
                    "loglik": torch.stack([ref[k][1] for k in keys])}, FIXTURE)
    """
    out = {}
    for opk in OPERATORS:
        for mc in MODEL_CLASSES:
            model = build_model(mc)
            for mode in MODES:
                for knob_name, knobs in KNOB_CASES.items():
                    if mode == "dps_jacobian_free" and knob_name != "plain":
                        continue
                    op = build_linear_operator(opk)
                    lik = make_lik(model, op, mode, mc, **knobs)
                    x, fh, y = make_inputs(op)
                    for t_val in TIMES:
                        sbar, ll = call_score(lik, model, mc, x, fh, y, t_val)
                        key = f"{opk}|{mc}|{mode}|{knob_name}|{t_val}"
                        out[key] = (sbar.detach().clone(), ll.detach().clone())
    return out


def load_reference() -> dict:
    """The fixture as ``{key: (sbar, loglik)}``."""
    p = torch.load(FIXTURE)
    return {k: (p["sbar"][i], p["loglik"][i]) for i, k in enumerate(p["keys"])}


# ----------------------------------------------------------------------
# Brute-force references (dense Jacobians, no code from the likelihood
# except the schedule coefficient helpers)
# ----------------------------------------------------------------------
def _dense_jac(fn, z):
    return torch.autograd.functional.jacobian(fn, z)


def brute_force_sbar(
    lik, model, mc, mode, op, x, fh, y, t_val, statistical=False, gh_order=GH_NODES,
):
    """Dense-matrix Sbar for a nonlinear ``h`` (spec formula). ``J_h`` at the
    ensemble mean of ``xhat1`` for ``inflated_shared``, per member otherwise."""
    b = x.shape[0]
    shape = x.shape[1:]
    t = torch.full((b, 1), float(t_val), dtype=torch.float64).clamp(1e-4, 1 - 1e-4)
    tg = t.reshape(b, 1, 1, 1)
    interp = lik.interpolant
    alpha, beta = interp.alpha(tg), interp.beta(tg)
    if mc == "si":
        drift = model.drift(x, t, fh)
        mu_s = lik._source_mean_si(x=x, t=tg, field_history=fh, drift=drift)
        rho = interp.gamma(tg) ** 2 * tg
        a0 = fh[..., -1]
        net = model.drift
    else:
        mu_s = -alpha**2 * model.score(x, t, fh)
        rho = alpha**2
        a0 = torch.zeros_like(x)
        net = model.score
    xhat1 = (x - alpha * a0 - mu_s) / beta
    beta_s, rho_s = float(beta.reshape(-1)[0]), float(rho.reshape(-1)[0])
    R = lik.original_variance
    n_x = x[0].numel()
    eye_x = torch.eye(n_x, dtype=torch.float64)

    def net_jac(xb, tb, fhb):
        return _dense_jac(
            lambda z: net(z.reshape(1, *shape), tb, fhb).reshape(-1), xb.reshape(-1)
        )

    if mode == "inflated_shared":
        c_iso, c_jac = lik._sigma_s_coeffs(tg[:1], rho[:1])
        Jn_shared = net_jac(x.mean(0, keepdim=True), t[:1], fh.mean(0, keepdim=True))
    else:
        c_iso, c_jac = lik._sigma_s_coeffs(tg, rho)
    c_jac = c_jac * lik.jacobian_damping
    def h_flat(z):
        return op(z.reshape(1, *shape)).reshape(-1)

    xbar1 = xhat1.mean(0)

    out = []
    for i in range(b):
        if mode == "dps_jacobian_free":
            Sig = rho_s * eye_x
        elif mode == "inflated_shared":
            Sig = float(c_iso.reshape(-1)[0]) * eye_x + float(c_jac.reshape(-1)[0]) * Jn_shared
        else:
            Jn = net_jac(x[i : i + 1], t[i : i + 1], fh[i : i + 1])
            Sig = float(c_iso.reshape(-1)[i]) * eye_x + float(c_jac.reshape(-1)[i]) * Jn
        if statistical:
            import numpy as np

            Lm = op.inner.obs_matrix.to(torch.float64)
            M = Lm @ Sig @ Lm.T
            s2 = torch.diagonal(M).clamp_min(0) / beta_s**2
            nodes, w = np.polynomial.hermite_e.hermegauss(gh_order)
            nodes = torch.as_tensor(nodes)
            w = torch.as_tensor(w / np.sqrt(2 * np.pi))
            Z = (Lm @ xhat1[i].reshape(-1))[:, None] + s2.sqrt()[:, None] * nodes
            gZ = op.g(Z)
            mh = (gZ * w).sum(-1)
            D = (op.dg(Z) * w).sum(-1)
            var = ((gZ - mh[:, None]) ** 2 * w).sum(-1)
            J = D[:, None] * Lm
            Sb = beta_s**2 * R * torch.eye(len(mh), dtype=torch.float64) + J @ Sig @ J.T
            Sb = Sb + torch.diag(beta_s**2 * (var - D**2 * s2).clamp_min(0))
            r = beta_s * (y[i] - mh)
        else:
            shared_mean = mode == "inflated_shared" and not lik._shared_member_linearization
            xl = xbar1 if shared_mean else xhat1[i]
            J = _dense_jac(h_flat, xl.reshape(-1))
            Sb = beta_s**2 * R * torch.eye(J.shape[0], dtype=torch.float64) + J @ Sig @ J.T
            r = beta_s * (y[i] - h_flat(xhat1[i].reshape(-1)))
        sb = (Sig / rho_s) @ J.T @ torch.linalg.solve(Sb, r)
        out.append(sb.reshape(shape))
    return torch.stack(out)


class GenericObs(ObservationOperator):
    """Non-pointwise nonlinear ``h(x) = atan(L x) + 0.2 L(x^2)`` with only a
    ``forward`` (``jvp``/``vjp``/``jacobian`` from the torch.func defaults)."""

    def __init__(self, inner: LinearObservationOperator) -> None:
        super().__init__()
        self.inner = inner
        self.C, self.H, self.W = inner.C, inner.H, inner.W
        self.num_obs, self.num_dofs = inner.num_obs, inner.num_dofs

    def forward(self, x):
        return torch.atan(self.inner(x)) + 0.2 * self.inner(x * x)


def _assert_close(a, b, rtol=1e-9):
    scale = max(1.0, float(b.detach().abs().max()))
    err = float((a - b).abs().max())
    assert err <= rtol * scale, f"max abs err {err:.3e} (scale {scale:.3e})"


# ----------------------------------------------------------------------
# 1. Linear regression fixture
# ----------------------------------------------------------------------
def test_linear_regression_fixture():
    ref = load_reference()
    new = compute_linear_reference()
    assert set(ref) == set(new)
    for k in ref:
        _assert_close(new[k][0], ref[k][0], rtol=1e-12)
        _assert_close(new[k][1], ref[k][1], rtol=1e-12)


def test_xhat1_reproduces_linear_residual():
    """beta (y - H xhat1) == ybar - mu_bar for the code's mu_s (SI and FM)."""
    for mc in MODEL_CLASSES:
        model = build_model(mc)
        op = build_linear_operator("super_res")
        lik = make_lik(model, op, "dps_jacobian_free", mc)
        x, fh, y = make_inputs(op)
        for t_val in TIMES:
            t = torch.full((B, 1), t_val, dtype=torch.float64)
            tg = t.reshape(B, 1, 1, 1)
            al, be = lik.interpolant.alpha(tg), lik.interpolant.beta(tg)
            if mc == "si":
                mu_s = lik._source_mean_si(x=x, t=tg, field_history=fh, drift=model.drift(x, t, fh))
                ybar = lik.interpolant.alpha(t) * op(fh[..., -1]) + lik.interpolant.beta(t) * y
                xhat1 = (x - al * fh[..., -1] - mu_s) / be
            else:
                mu_s = -al**2 * model.score(x, t, fh)
                ybar = lik.interpolant.beta(t) * y
                xhat1 = (x - mu_s) / be
            res_lin = ybar - (op(x) - op(mu_s))
            _assert_close(lik.interpolant.beta(t) * (y - op(xhat1)), res_lin, rtol=1e-12)


def test_hht_cache_invalidated_by_load_mask(tmp_path):
    model = build_model("si")
    op = build_linear_operator("random")
    lik = make_lik(model, op, "dps_jacobian_free", "si")
    x, fh, y = make_inputs(op)
    call_score(lik, model, "si", x, fh, y, 0.5)  # populate caches
    other = LinearObservationOperator(type="random", data_size=DATA_SIZE, percent_obs=0.5, seed=11)
    other.save_mask(str(tmp_path / "mask.pt"))
    op.load_mask(str(tmp_path / "mask.pt"))
    got, _ = call_score(lik, model, "si", x, fh, y, 0.5)
    fresh = make_lik(model, op, "dps_jacobian_free", "si")
    want, _ = call_score(fresh, model, "si", x, fh, y, 0.5)
    _assert_close(got, want, rtol=1e-13)


# ----------------------------------------------------------------------
# 2. Pointwise identity == linear (all modes / model classes / options)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("opk", OPERATORS)
@pytest.mark.parametrize("mc", MODEL_CLASSES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("method", ["first_order", "statistical"])
def test_pointwise_identity_equals_linear(opk, mc, mode, method):
    ref = load_reference()
    model = build_model(mc)
    for knob_name, knobs in KNOB_CASES.items():
        if mode == "dps_jacobian_free" and knob_name != "plain":
            continue
        op = PointwiseObservationOperator(inner=build_linear_operator(opk), nonlinearity="identity")
        lik = make_lik(model, op, mode, mc, obs_linearization_method=method, **knobs)
        x, fh, y = make_inputs(op)
        for t_val in TIMES:
            sbar, ll = call_score(lik, model, mc, x, fh, y, t_val)
            want_s, want_l = ref[f"{opk}|{mc}|{mode}|{knob_name}|{t_val}"]
            _assert_close(sbar, want_s, rtol=1e-9)
            _assert_close(ll, want_l, rtol=1e-9)


# ----------------------------------------------------------------------
# 3. Nonlinear h vs brute force (structured pointwise + generic routes)
# ----------------------------------------------------------------------
def _nonlinear_ops(opk):
    return {
        "arctan": PointwiseObservationOperator(
            inner=build_linear_operator(opk), nonlinearity="arctan", strength=1.5),
        "cubic": PointwiseObservationOperator(
            inner=build_linear_operator(opk), nonlinearity="cubic", strength=0.3),
        "generic": GenericObs(build_linear_operator(opk)),
        **(
            {"speed": VelocityMagnitudeOperator(inner=build_linear_operator(opk), norm_std=1.0)}
            if opk == "random" else {}
        ),
    }


@pytest.mark.parametrize("opk", OPERATORS)
@pytest.mark.parametrize("mc", MODEL_CLASSES)
@pytest.mark.parametrize("mode", MODES)
def test_nonlinear_matches_brute_force(opk, mc, mode):
    model = build_model(mc)
    for name, op in _nonlinear_ops(opk).items():
        x, fh, y = make_inputs(op)
        for t_val in TIMES:
            lik = make_lik(model, op, mode, mc)
            sbar, ll = call_score(lik, model, mc, x, fh, y, t_val)
            want = brute_force_sbar(lik, model, mc, mode, op, x, fh, y, t_val)
            assert torch.isfinite(sbar).all() and torch.isfinite(ll).all()
            _assert_close(sbar, want, rtol=1e-9)


@pytest.mark.parametrize("opk", OPERATORS)
@pytest.mark.parametrize("mc", MODEL_CLASSES)
def test_shared_mean_linearization_matches_brute_force(opk, mc):
    """inflated_shared, pointwise h, explicit ensemble-mean linearization
    (the route used for non-pointwise h, forced here on a pointwise h)."""
    import warnings

    model = build_model(mc)
    for name in ("arctan", "cubic"):
        op = _nonlinear_ops(opk)[name]
        x, fh, y = make_inputs(op)
        for t_val in TIMES:
            lik = make_lik(model, op, "inflated_shared", mc)
            lik._shared_member_linearization = False
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sbar, _ = call_score(lik, model, mc, x, fh, y, t_val)
            want = brute_force_sbar(lik, model, mc, "inflated_shared", op, x, fh, y, t_val)
            _assert_close(sbar, want, rtol=1e-9)


@pytest.mark.parametrize("mc", MODEL_CLASSES)
@pytest.mark.parametrize("mode", ["inflated", "inflated_shared"])
def test_nonlinear_member_knobs_finite_and_cap(mc, mode):
    """Floor / cap / isotropic-front knobs run on the nonlinear routes and
    the cap bounds ||J Sbar|| <= kappa ||residual||."""
    model = build_model(mc)
    for name, op in _nonlinear_ops("random").items():
        x, fh, y = make_inputs(op)
        for knobs in (KNOB_CASES["floor_cap"], KNOB_CASES["isofront_damp"]):
            lik = make_lik(model, op, mode, mc, **knobs)
            sbar, _ = call_score(lik, model, mc, x, fh, y, 0.5)
            assert torch.isfinite(sbar).all()


@pytest.mark.parametrize("mc", MODEL_CLASSES)
def test_shared_single_member_equals_inflated(mc):
    """One member: the ensemble-mean linearization / Jacobian is the member's."""
    model = build_model(mc)
    for name, op in _nonlinear_ops("random").items():
        x, fh, y = make_inputs(op)
        x, fh, y = x[:1], fh[:1], y[:1]
        a = make_lik(model, op, "inflated", mc)
        b_ = make_lik(model, op, "inflated_shared", mc)
        for t_val in TIMES:
            sa, la = call_score(a, model, mc, x, fh, y, t_val)
            sb, lb = call_score(b_, model, mc, x, fh, y, t_val)
            _assert_close(sa, sb, rtol=1e-11)
            _assert_close(la, lb, rtol=1e-11)


@pytest.mark.parametrize("opk", OPERATORS)
@pytest.mark.parametrize("mc", MODEL_CLASSES)
@pytest.mark.parametrize("mode", MODES)
def test_statistical_matches_brute_force(opk, mc, mode):
    model = build_model(mc)
    for name in ("arctan", "cubic"):
        op = _nonlinear_ops(opk)[name]
        x, fh, y = make_inputs(op)
        for t_val in TIMES:
            lik = make_lik(model, op, mode, mc, obs_linearization_method="statistical")
            sbar, _ = call_score(lik, model, mc, x, fh, y, t_val)
            # Same GH order on both sides (cubic: exact for any order).
            want = brute_force_sbar(lik, model, mc, mode, op, x, fh, y, t_val,
                                    statistical=True)
            _assert_close(sbar, want, rtol=1e-9)


def test_statistical_needs_pointwise_operator():
    model = build_model("si")
    with pytest.raises(ValueError):
        make_lik(model, GenericObs(build_linear_operator("random")), "inflated", "si",
                 obs_linearization_method="statistical")


def test_flowdas_nonlinear_operator_runs():
    """FlowDAS only uses h(x) (no obs_matrix / transpose)."""
    from scisi.likelihood_models.gaussian_likelihood import FlowdasGaussianLikelihood

    model = build_model("si")
    op = PointwiseObservationOperator(inner=build_linear_operator("random"), nonlinearity="arctan")
    lik = FlowdasGaussianLikelihood(model=model, obs_operator=op, variance=0.05, num_mc_samples=5)
    x, fh, y = make_inputs(op)
    t = torch.full((1, 1), 0.5, dtype=torch.float64)  # posterior passes [1, 1]
    torch.manual_seed(0)
    s, ll = lik.score(observations=y, x=x, t=t, field_history=fh)
    assert s.shape == x.shape and torch.isfinite(s).all() and torch.isfinite(ll).all()


def test_shared_linearization_cadence():
    """inflated_shared: the ensemble-mean linearization is frozen between
    shared-Jacobian refreshes (jacobian_refresh_every) and refreshed with it."""
    model = build_model("si")
    op = _nonlinear_ops("random")["generic"]
    lik = make_lik(model, op, "inflated_shared", "si", jacobian_refresh_every=2)
    x, fh, y = make_inputs(op)
    seen = []
    for k, t_val in enumerate((0.1, 0.3, 0.5, 0.7)):
        call_score(lik, model, "si", x + 0.1 * k, fh, y, t_val)
        seen.append(lik._shared_jac_cache["lin"])
    assert seen[0] is seen[1] and seen[2] is not seen[1] and seen[2] is seen[3]
    assert seen[0].is_linear


# ----------------------------------------------------------------------
# 4. End-to-end: SI-SDE / DM-SDE / FM-ODE posteriors vs 1-D quadrature
# ----------------------------------------------------------------------
_PE = (
    Path(__file__).resolve().parents[1]
    / "papers"
    / "A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants"
    / "code"
    / "paper_experiments"
)

E2E_D = 4
E2E_R = 0.1**2
E2E_PRIOR_VAR = 1.0
E2E_X0 = (0.0, 0.5, -1.0, 1.5)
E2E_X1_TRUE = (0.8, -0.3, 1.2, 2.5)


def _e2e_reference():
    """Exact posterior mean / variance per coordinate, y = arctan(x1_true)."""
    x0 = torch.tensor(E2E_X0, dtype=torch.float64)
    y = torch.atan(torch.tensor(E2E_X1_TRUE, dtype=torch.float64))
    grid = torch.linspace(-8.0, 8.0, 200001, dtype=torch.float64)
    out = []
    for k in range(E2E_D):
        lp = -0.5 * (grid - x0[k]) ** 2 / E2E_PRIOR_VAR - 0.5 * (y[k] - torch.atan(grid)) ** 2 / E2E_R
        w = torch.exp(lp - lp.max())
        w = w / w.sum()
        m = (w * grid).sum()
        out.append(((m).item(), (w * (grid - m) ** 2).sum().item()))
    return x0, y, torch.tensor(out, dtype=torch.float64)


def _run_e2e(sampler, mode, method, n=2000, steps=100, seed=0):
    import sys

    if str(_PE) not in sys.path:
        sys.path.insert(0, str(_PE))
    from cases.analytical.prior_models import (
        build_fm_model,
        build_identity_obs_operator,
        build_si_model,
    )
    from scisi.posterior_models.flow_matching_posterior import (
        FlowMatchingPosterior,
        endpoint_vanishing_diffusion,
    )
    from scisi.posterior_models.stochastic_interpolant_posterior import (
        StochasticInterpolantPosterior,
    )

    x0, y, _ = _e2e_reference()
    d = E2E_D
    op = PointwiseObservationOperator(
        inner=build_identity_obs_operator(d), nonlinearity="arctan", strength=1.0
    )
    x0_4d = x0.reshape(1, 1, 1, d)
    fh = x0_4d.unsqueeze(-1).repeat(n, 1, 1, 1, 1)
    kw = dict(likelihood_mode=mode, obs_linearization_method=method, variance=E2E_R)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(seed)
        if sampler == "si_sde":
            model = build_si_model(d=d, g0=1.0, prior_var=E2E_PRIOR_VAR)
            lik = InterpolantGaussianLikelihood(model=model, obs_operator=op, model_class="si", **kw)
            post = StochasticInterpolantPosterior(model=model, likelihood_model=lik)
            base = x0_4d.repeat(n, 1, 1, 1)
        else:
            model = build_fm_model(d=d, prior_var=E2E_PRIOR_VAR)
            lik = InterpolantGaussianLikelihood(model=model, obs_operator=op, model_class="fm", **kw)
            diff = (
                endpoint_vanishing_diffusion(model.interpolation, scale=1.0)
                if sampler == "dm_sde" else None
            )
            post = FlowMatchingPosterior(model=model, likelihood_model=lik, diffusion_term=diff)
            base = None
        with torch.no_grad():
            out = post.sample(
                base=base, batch_size=n, num_steps=steps, field_history=fh,
                observations=y.reshape(1, d),
            )
    finally:
        torch.set_default_dtype(prev)
    out = out.reshape(n, d)
    return torch.stack([out.mean(0), out.var(0)], dim=1)


_E2E_CASES = [
    (s, m, "first_order")
    for s in ("si_sde", "dm_sde", "fm_ode")
    for m in MODES
] + [(s, "inflated", "statistical") for s in ("si_sde", "dm_sde", "fm_ode")]


@pytest.mark.skipif(not _PE.exists(), reason="paper_experiments not present")
@pytest.mark.parametrize("sampler,mode,method", _E2E_CASES)
def test_end_to_end_arctan_vs_quadrature(sampler, mode, method):
    """Approximate agreement with the exact 1-D posterior (Gaussian
    linearization of h is an approximation, so the bounds are loose: recorded
    errors at n=2000, 100 steps are ~0.1-0.2 in the mean, dominated by the
    weakly-informative coordinate x0=1.5 where g' ~ 0.14)."""
    _, _, ref = _e2e_reference()
    got = _run_e2e(sampler, mode, method)
    post_std = ref[:, 1].sqrt()
    mean_err = (got[:, 0] - ref[:, 0]).abs()
    var_rel = ((got[:, 1] - ref[:, 1]) / ref[:, 1]).abs()
    print(
        f"\n{sampler} {mode} {method}: mean_err={mean_err.max():.3f} "
        f"(max {float((mean_err / post_std).max()):.2f} post-std) var_rel={var_rel.max():.3f}"
    )
    assert torch.isfinite(got).all()
    assert float((mean_err / post_std).max()) < 0.6
    assert float(var_rel.max()) < 0.6
