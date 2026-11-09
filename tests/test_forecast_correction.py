"""Forecast-residual correction (scisi.posterior_models.forecast_correction).

CPU-only and fast (one optional MPS test). The linear-Gaussian checks follow
``paper_experiments/tests/test_forecast_correction_lg.py`` but run through the real
``BasePosterior.sample_trajectory`` hook.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

import scisi.posterior_models.forecast_correction as fc
from scisi.likelihood_models.gaussian_likelihood import InterpolantGaussianLikelihood
from scisi.likelihood_models.observation_operators import LinearObservationOperator
from scisi.models.flow_matching_model import FlowMatchingModel
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
from scisi.models.interpolations import (
    LinearDeterministicInterpolation,
    LinearStochasticInterpolation,
)
from scisi.particle_filter.enkf import periodic_taper
from scisi.posterior_models.base_posterior import BasePosterior
from scisi.posterior_models.flow_matching_posterior import (
    FlowMatchingPosterior,
    endpoint_vanishing_diffusion,
)
from scisi.posterior_models.forecast_correction import (
    ForecastCorrection,
    ForecastCorrectionConfig,
    build_residual_bank,
    draw_unguided_forecast,
    grid_localisation,
)
from scisi.posterior_models.stochastic_interpolant_posterior import (
    StochasticInterpolantPosterior,
)
from scisi.posterior_models.surge_posterior import (
    SurgeFlowMatchingPosterior,
    SurgeStochasticInterpolantPosterior,
)
from scisi.sampling.ode_solvers import euler_step
from scisi.sampling.sde_solvers import euler_maruyama_step

torch.set_num_threads(min(3, torch.get_num_threads()))


def _loc1d(d, radius):
    return grid_localisation((1, 1, d), radius, periodic=True)


# --------------------------------------------------------------------------- #
# Linear-Gaussian system (as in test_forecast_correction_lg.py)
# --------------------------------------------------------------------------- #


def _system(d: int = 8, stride: int = 4):
    i = torch.arange(d)
    gap = (i[:, None] - i).abs()
    dist = torch.minimum(gap, d - gap).double()
    A = 0.95 * torch.roll(torch.eye(d, dtype=torch.float64), 1, dims=1)
    Q = 0.5 * torch.exp(-((dist / 1.5) ** 2)) + 1e-9 * torch.eye(d, dtype=torch.float64)
    rows = torch.arange(0, d, stride)
    return A, Q, rows, 0.25


def _selection_operator(d: int, rows: torch.Tensor) -> LinearObservationOperator:
    op = LinearObservationOperator(type="random", data_size=(1, 1, d), percent_obs=1.0, seed=0)
    op.obs_matrix = torch.eye(d)[rows]
    op.obs_indices = rows.clone()
    op.num_obs = rows.numel()
    return op


class _Likelihood(nn.Module):
    def __init__(self, operator, variance):
        super().__init__()
        self.obs_operator = operator
        self.original_variance = float(variance)


class ExactLinearGaussianPrior(nn.Module):
    """Prior ``x1 | x0 ~ N(A x0, Q)`` drawn exactly (stands in for a perfect model)."""

    def __init__(self, A, Q):
        super().__init__()
        self.register_buffer("A", A.float())
        self.register_buffer("L", torch.linalg.cholesky(Q).float())
        self._device_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _prepare_batch(self, batch_size=1, **kwargs):
        return tuple(None if v is None else v.repeat(batch_size, *[1] * (v.ndim - 1)) for v in kwargs.values())

    def sample(self, field_history, base=None, return_field_history=False, **_):
        n = field_history.shape[0]
        x0 = field_history[..., -1].reshape(n, -1)
        x1 = (x0 @ self.A.T + torch.randn_like(x0) @ self.L.T).reshape(field_history[..., -1].shape)
        if return_field_history:
            return x1, torch.cat([field_history[..., 1:], x1.unsqueeze(-1)], -1)
        return x1


class ExactMemberPosterior(BasePosterior):
    """Draws ``x' ~ p(x' | x_i, y)`` exactly for every member (no SDE)."""

    def __init__(self, A, Q, operator, variance):
        super().__init__(
            model=ExactLinearGaussianPrior(A, Q),
            likelihood_model=_Likelihood(operator, variance),
            diffusion_term=lambda t: torch.zeros_like(t),
            gaussian_base=True,
        )
        H = operator.obs_matrix.double()
        R = variance * torch.eye(H.shape[0], dtype=torch.float64)
        self.Kq = Q @ H.T @ torch.linalg.inv(H @ Q @ H.T + R)
        self.Lpost = torch.linalg.cholesky(Q - self.Kq @ H @ Q + 1e-12 * torch.eye(Q.shape[0], dtype=torch.float64))
        self.A64, self.H64 = A, H

    def _one_step(self, *a, **k):  # pragma: no cover - unused
        raise NotImplementedError

    def sample(self, base, batch_size, num_steps, field_history, observations,
               field_cond=None, pars_cond=None, stepper=None, return_field_history=False):
        n = field_history.shape[0]
        mu = field_history[..., -1].reshape(n, -1).double() @ self.A64.T
        y = observations.double().reshape(-1, self.H64.shape[0])
        x = mu + (y - mu @ self.H64.T) @ self.Kq.T + torch.randn_like(mu) @ self.Lpost.T
        x = x.float().reshape(field_history[..., -1].shape)
        # A subclass that overrides ``sample`` calls the hook itself.
        x = self._apply_forecast_correction(
            x, field_history, observations, num_steps, stepper, field_cond, pars_cond, batch_size)
        if return_field_history:
            return x, torch.cat([field_history[..., 1:], x.unsqueeze(-1)], -1)
        return x


def _kalman_reference(A, Q, rows, r, cycles, seed):
    g = torch.Generator().manual_seed(seed)
    d, m = A.shape[0], rows.numel()
    H = torch.eye(d, dtype=torch.float64)[rows]
    LQ = torch.linalg.cholesky(Q)
    truth = torch.randn(d, generator=g, dtype=torch.float64)
    ys, mean, cov = [], torch.zeros(d, dtype=torch.float64), torch.eye(d, dtype=torch.float64)
    for _ in range(cycles):
        truth = A @ truth + LQ @ torch.randn(d, generator=g, dtype=torch.float64)
        ys.append(H @ truth + math.sqrt(r) * torch.randn(m, generator=g, dtype=torch.float64))
        mf, Pf = A @ mean, A @ cov @ A.T + Q
        K = Pf @ H.T @ torch.linalg.inv(H @ Pf @ H.T + r * torch.eye(m, dtype=torch.float64))
        mean, cov = mf + K @ (ys[-1] - H @ mf), Pf - K @ H @ Pf
    return torch.stack(ys, -1).float().unsqueeze(0), mean, cov


def _kalman_run(post, A, Q, rows, r, members, cycles, seed, correction, num_steps=4, stepper=euler_maruyama_step):
    """Run ``sample_trajectory`` and the Kalman filter on the same observations."""
    obs, mean, cov = _kalman_reference(A, Q, rows, r, cycles, seed)
    torch.manual_seed(seed + 1)
    d = A.shape[0]
    x0 = torch.randn(members, 1, 1, d)
    traj = post.sample_trajectory(
        base=x0.clone(), field_history=x0.unsqueeze(-1).clone(), observations=obs,
        ensemble_size=members, batch_size=members, num_steps=num_steps,
        num_physical_steps=cycles + 1, stepper=stepper, forecast_correction=correction,
    )
    X = traj[..., -1].reshape(members, d).double()
    err = ((X.mean(0) - mean).abs() / cov.diagonal().sqrt()).max().item()
    return err, torch.cov(X.T).diagonal() / cov.diagonal()


@pytest.fixture(scope="module")
def lg():
    A, Q, rows, r = _system()
    return A, Q, rows, r, _selection_operator(A.shape[0], rows)


# --------------------------------------------------------------------------- #
# Kalman reproduction through the hook; bank, online and fixed Q
# --------------------------------------------------------------------------- #


def test_plain_chain_through_hook_is_biased(lg):
    A, Q, rows, r, op = lg
    err, ratio = _kalman_run(ExactMemberPosterior(A, Q, op, r), A, Q, rows, r, 20000, 8, 0, None)
    assert err > 0.3 or (ratio > 1.1).any()


def test_online_correction_through_hook_matches_kalman(lg):
    A, Q, rows, r, op = lg
    post = ExactMemberPosterior(A, Q, op, r)
    corr = ForecastCorrection(enabled=True, q_source="online")
    err, ratio = _kalman_run(post, A, Q, rows, r, 20000, 8, 0, corr)
    assert err < 0.06 and ((ratio - 1).abs() < 0.04).all()
    assert len(corr.diagnostics.records) == 8
    assert post.last_forecast_correction is corr


def test_bank_and_online_agree_in_linear_gaussian_case(lg, tmp_path):
    A, Q, rows, r, op = lg
    post = ExactMemberPosterior(A, Q, op, r)
    states = torch.randn(4000, 1, 1, A.shape[0], 1)
    bank = build_residual_bank(post, states, num_steps=4, stepper=euler_step, pairs_per_state=5, seed=3)
    assert bank.shape == (20000, 1, 1, A.shape[0])
    path = tmp_path / "bank.pt"
    torch.save({"residuals": bank, "meta": {"num_steps": 4}}, path)
    res = {}
    for source, kw in [("online", {}), ("bank", {"residual_bank": str(path)})]:
        corr = ForecastCorrection(enabled=True, q_source=source, **kw)
        res[source] = err, ratio = _kalman_run(post, A, Q, rows, r, 20000, 8, 0, corr)
        assert err < 0.06 and ((ratio - 1).abs() < 0.04).all(), source
    assert ((res["online"][1] - res["bank"][1]).abs() < 0.05).all()


# --------------------------------------------------------------------------- #
# The update against a dense float64 reference
# --------------------------------------------------------------------------- #


def _dense_reference(x, f, y, H, R, e, f2=None, Kq=None, txy=None, tyy=None, scale=1.0):
    X, F = x.reshape(len(x), -1).double(), f.reshape(len(f), -1).double()
    H, R = H.double(), torch.diag(torch.as_tensor(R, dtype=torch.float64).reshape(-1).expand(H.shape[0]))

    def gain(D, n):
        PH = D.T @ (D @ H.T) / n
        HPH = (D @ H.T).T @ (D @ H.T) / n
        if txy is not None:
            PH, HPH = PH * txy, HPH * tyy
        return PH @ torch.linalg.inv(HPH + R)

    K_ens = gain(F - F.mean(0), len(F) - 1)
    if Kq is None:
        Kq = gain((F - f2.reshape(len(f2), -1).double()) / math.sqrt(2), len(F))
    V = y.double().reshape(-1, H.shape[0]) + e.double() - F @ H.T
    return (X + scale * V @ (K_ens - Kq).T).reshape(x.shape)


def _inputs(N, d, m, seed=2):
    g = torch.Generator().manual_seed(seed)
    x, f, f2 = (torch.randn(N, 1, 1, d, generator=g) for _ in range(3))
    return x, f, f2, torch.randn(1, m, generator=g), torch.randn(N, m, generator=g)


@pytest.mark.parametrize("N", [6, 40])  # 40 * 40 > m * d: the m x d product order
def test_unlocalised_update_matches_dense_reference(lg, N):
    A, Q, rows, r, op = lg
    d, m = A.shape[0], rows.numel()
    x, f, f2, y, e = _inputs(N, d, m)
    out = ForecastCorrection(enabled=True, gain_scale=0.7).correct(
        x, f, y, op, r, f2=f2, perturbations=e)
    ref = _dense_reference(x, f, y, op.obs_matrix, r, e, f2=f2, scale=0.7)
    torch.testing.assert_close(out.double(), ref, rtol=1e-6, atol=1e-6)


def test_unlocalised_gain_avoids_e_by_e(monkeypatch):
    X, HX, Z = torch.randn(50, 4), torch.randn(50, 2), torch.randn(2, 50)
    torch.testing.assert_close(fc._unlocalised_gain(X, HX, Z), (HX @ Z).T @ X)
    seen = []
    real = torch.Tensor.__matmul__
    monkeypatch.setattr(torch.Tensor, "__matmul__", lambda a, b: seen.append(real(a, b).shape) or real(a, b))
    fc._unlocalised_gain(X, HX, Z)
    assert (50, 50) not in seen


@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize("source", ["online", "bank"])
def test_localised_update_matches_dense_reference(lg, monkeypatch, cache, source):
    A, Q, rows, r, op = lg
    d, m = A.shape[0], rows.numel()
    monkeypatch.setattr(fc, "ROW_CHUNK", 3)
    if not cache:
        monkeypatch.setattr(fc, "TAPER_CACHE_LIMIT", 0)
    x, f, f2, y, e = _inputs(30, d, m)
    loc = _loc1d(d, 2.0)
    bank = torch.randn(50, 1, 1, d, generator=torch.Generator().manual_seed(9))
    corr = ForecastCorrection(enabled=True, localisation=loc, q_source=source, residual_bank=bank)
    out = corr.correct(x, f, y, op, r, f2=f2, perturbations=e)
    full = loc.kernel(loc.state_coords, loc.state_coords)
    txy, tyy = full[:, rows], full[rows][:, rows]
    Kq = None
    if source == "bank":
        Bf, H = bank.reshape(50, -1).double(), op.obs_matrix.double()
        QH, HQH = Bf.T @ (Bf @ H.T) / 50 * txy, (Bf @ H.T).T @ (Bf @ H.T) / 50 * tyy
        Kq = QH @ torch.linalg.inv(HQH + r * torch.eye(m, dtype=torch.float64))
    ref = _dense_reference(x, f, y, op.obs_matrix, r, e, f2=f2, Kq=Kq, txy=txy, tyy=tyy)
    torch.testing.assert_close(out.double(), ref, rtol=1e-6, atol=1e-6)
    # The taper itself is float64 (no float32 cache round-off).
    assert loc.state_obs(op, slice(0, d), out.double()).dtype == torch.float64
    torch.testing.assert_close(loc.state_obs(op, slice(0, d), out.double()), txy, rtol=0, atol=1e-14)


def test_fixed_q_equals_closed_form(lg):
    A, Q, rows, r, op = lg
    d, m = A.shape[0], rows.numel()
    q = torch.linspace(0.2, 1.0, d)
    x, f, _, y, e = _inputs(12, d, m)
    corr = ForecastCorrection(enabled=True, q_source="fixed", q_variance=q)
    out = corr.correct(x, f, y, op, r, perturbations=e)
    H = op.obs_matrix.double()
    Qm = torch.diag(q.double())
    Kq = Qm @ H.T @ torch.linalg.inv(H @ Qm @ H.T + r * torch.eye(m, dtype=torch.float64))
    torch.testing.assert_close(out.double(), _dense_reference(x, f, y, H, r, e, Kq=Kq), rtol=1e-6, atol=1e-6)
    rec = corr.diagnostics.records[0]
    assert rec["q_rms"].item() == pytest.approx(q.double().mean().sqrt().item(), rel=1e-6)
    with pytest.raises(ValueError):
        ForecastCorrection(enabled=True, q_source="fixed").check_supported(ExactMemberPosterior(A, Q, op, r))


def test_fixed_q_matches_kalman_through_hook(lg):
    A, _, rows, r, op = lg
    d = A.shape[0]
    Q = 0.5 * torch.eye(d, dtype=torch.float64)
    post = ExactMemberPosterior(A, Q, op, r)
    corr = ForecastCorrection(enabled=True, q_source="fixed", q_variance=0.5)
    err, ratio = _kalman_run(post, A, Q, rows, r, 20000, 8, 0, corr)
    assert err < 0.06 and ((ratio - 1).abs() < 0.04).all()


# --------------------------------------------------------------------------- #
# Groups
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("q_source", ["online", "bank"])
@pytest.mark.parametrize("localised", [False, True])
@pytest.mark.parametrize("sizes", [(40, 40, 40), (40, 40, 23)])
def test_statistics_are_per_group(lg, q_source, localised, sizes):
    A, Q, rows, r, op = lg
    d, m = A.shape[0], rows.numel()
    G, N = len(sizes), sum(sizes)
    sizes_t = torch.tensor(sizes)
    gen = torch.Generator().manual_seed(1)
    shifts = torch.tensor([0.0, 5.0, -3.0]).repeat_interleave(sizes_t).reshape(-1, 1, 1, 1)
    scales = torch.tensor([1.0, 0.2, 3.0]).repeat_interleave(sizes_t).reshape(-1, 1, 1, 1)
    x = torch.randn(N, 1, 1, d, generator=gen)
    f = shifts + scales * torch.randn(N, 1, 1, d, generator=gen)
    f2 = f + torch.randn(N, 1, 1, d, generator=gen)
    y_groups = torch.randn(G, m, generator=gen)
    e = torch.randn(N, m, generator=gen)
    kw = dict(enabled=True, q_source=q_source, localisation=_loc1d(d, 2.0) if localised else None,
              residual_bank=torch.randn(100, 1, 1, d, generator=gen))
    batched = ForecastCorrection(group_size=40, **kw)
    out = batched.correct(x, f, y_groups, op, r, f2=f2, perturbations=e)
    per_member = ForecastCorrection(group_size=40, **kw).correct(
        x, f, y_groups.repeat_interleave(sizes_t, 0), op, r, f2=f2, perturbations=e)
    torch.testing.assert_close(out, per_member)
    start = 0
    for g, n in enumerate(sizes):
        sl = slice(start, start + n)
        start += n
        single = ForecastCorrection(**kw)
        ref = single.correct(x[sl], f[sl], y_groups[g : g + 1], op, r, f2=f2[sl], perturbations=e[sl])
        torch.testing.assert_close(out[sl], ref, rtol=1e-5, atol=1e-5)
        for key in ("innovation_mean", "spread_rms", "q_rms"):
            assert batched.diagnostics.records[0][key][g].item() == pytest.approx(
                single.diagnostics.records[0][key][0].item(), rel=1e-6, abs=1e-6)
    pooled = ForecastCorrection(**kw).correct(x, f, y_groups[:1], op, r, f2=f2, perturbations=e)
    assert not torch.allclose(out, pooled, atol=1e-3)


def test_group_slices_merge_a_single_leftover():
    assert fc._group_slices(10, 4) == [slice(0, 4), slice(4, 8), slice(8, 10)]
    assert fc._group_slices(9, 4) == [slice(0, 4), slice(4, 9)]
    assert fc._group_slices(5, None) == [slice(0, 5)]


# --------------------------------------------------------------------------- #
# Bit-for-bit identity when off
# --------------------------------------------------------------------------- #


class _SIDrift(nn.Module):
    """Closed-form SI drift for x1 | x0 ~ N(A x0, Q), Q = U diag(q) U'."""

    def __init__(self, A, Q, g0=1.0):
        super().__init__()
        q, U = torch.linalg.eigh(Q)
        self.register_buffer("A", A.float())
        self.register_buffer("q", q.float())
        self.register_buffer("U", U.float())
        self.g0 = g0
        self._device_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x, t, field_history=None, field_cond=None, pars_cond=None):
        t4 = torch.clamp(t, 1e-4, 1 - 1e-4).reshape(t.shape[0], 1, 1, 1)
        x0 = field_history[..., -1]
        mu = x0 @ self.A.T
        gamma = self.g0 * (1 - t4)
        coef = (t4 * self.q - t4 * gamma * self.g0) / (t4**2 * self.q + t4 * gamma**2)
        v = x - ((1 - t4) * x0 + t4 * mu)
        return (mu - x0) + (coef * (v @ self.U)) @ self.U.T


class _FMVelocity(_SIDrift):
    def forward(self, x, t, field_history=None, field_cond=None, pars_cond=None):
        t4 = torch.clamp(t, 1e-4, 1 - 1e-4).reshape(t.shape[0], 1, 1, 1)
        mu = field_history[..., -1] @ self.A.T
        alpha = 1 - t4
        cov = alpha**2 + t4**2 * self.q
        return mu + (((t4 * self.q - alpha) / cov) * ((x - t4 * mu) @ self.U)) @ self.U.T


def _real_posterior(kind: str, A, Q, op, r=0.25, shift=0.0):
    if kind == "si":
        model = FollmerStochasticInterpolant(
            interpolation=LinearStochasticInterpolation(gamma_multiplier=1.0), drift_model=_SIDrift(A, Q))
        lik = InterpolantGaussianLikelihood(model=model, obs_operator=op, variance=r,
                                            likelihood_mode="inflated", model_class="si")
        return StochasticInterpolantPosterior(model=model, likelihood_model=lik), euler_maruyama_step
    model = FlowMatchingModel(interpolation=LinearDeterministicInterpolation(), drift_model=_FMVelocity(A, Q))
    lik = InterpolantGaussianLikelihood(model=model, obs_operator=op, variance=r,
                                        likelihood_mode="inflated", model_class="fm")
    if kind == "fm_ode":
        return FlowMatchingPosterior(model=model, likelihood_model=lik, drift_time_shift=shift), euler_step
    g = endpoint_vanishing_diffusion(model.interpolation)
    post = FlowMatchingPosterior(model=model, likelihood_model=lik, diffusion_term=g, drift_time_shift=shift)
    return post, euler_maruyama_step


def _trajectory(post, stepper, d, members=6, steps=5, every=1, correction="omit", seed=0, rows_per_member=False):
    torch.manual_seed(seed)
    x0 = torch.randn(members, 1, 1, d)
    m = post.likelihood_model.obs_operator.num_obs
    y = torch.randn(members if rows_per_member else 1, m, steps)
    kw = {} if correction == "omit" else {"forecast_correction": correction}
    return post.sample_trajectory(
        base=x0.clone(), field_history=x0.unsqueeze(-1).clone(), observations=y,
        ensemble_size=members, batch_size=members, num_steps=6,
        num_physical_steps=steps, stepper=stepper, assimilate_every=every, **kw,
    )


@pytest.mark.parametrize("kind", ["si", "fm_ode", "dm_sde"])
@pytest.mark.parametrize("every", [1, 2])
def test_off_is_bit_for_bit_identical(lg, kind, every):
    A, Q, rows, r, op = lg
    post, stepper = _real_posterior(kind, A, Q, op)
    ref = _trajectory(post, stepper, A.shape[0], every=every)
    after_ref = torch.rand(3)
    for off in [None, ForecastCorrection(enabled=False)]:
        out = _trajectory(post, stepper, A.shape[0], every=every, correction=off)
        assert torch.equal(ref, out)
        assert torch.equal(after_ref, torch.rand(3))  # same global RNG state
    # Enabled with zero gain and a forecast seed (the "measure" mode): forecasts are
    # drawn from forked generators, so the trajectory and the stream are unchanged.
    measure = ForecastCorrection(enabled=True, gain_scale=0.0, forecast_seed=0, forecast_num_steps=2)
    out = _trajectory(post, stepper, A.shape[0], every=every, correction=measure)
    assert torch.equal(ref, out) and torch.equal(after_ref, torch.rand(3))
    assert len(measure.diagnostics.records) > 0
    with pytest.raises(TypeError):
        _trajectory(post, stepper, A.shape[0], every=every, correction=ForecastCorrectionConfig(enabled=True))


# --------------------------------------------------------------------------- #
# Real SI-SDE / FM-ODE / DM-SDE posteriors; forecast settings honoured
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["si", "fm_ode", "dm_sde"])
@pytest.mark.parametrize("q_source", ["online", "bank"])
def test_real_posteriors_run_and_forecast_steps_are_honoured(lg, monkeypatch, kind, q_source):
    A, Q, rows, r, op = lg
    d = A.shape[0]
    post, stepper = _real_posterior(kind, A, Q, op)
    drift = post.model.drift_model
    calls = {"stepper": 0, "inside": False, "drift": 0}

    def counting_stepper(**kw):
        calls["stepper"] += 1
        return stepper(**kw)

    original_forward = drift.forward

    def counted(*a, **k):
        calls["drift"] += calls["inside"]
        return original_forward(*a, **k)

    drift.forward = counted
    original_draw = fc.draw_unguided_forecast

    def draw(*a, **k):
        calls["inside"] = True
        try:
            return original_draw(*a, **k)
        finally:
            calls["inside"] = False

    monkeypatch.setattr(fc, "draw_unguided_forecast", draw)
    corr = ForecastCorrection(
        enabled=True, forecast_num_steps=3, forecast_stepper=counting_stepper, q_source=q_source,
        residual_bank=torch.randn(64, 1, 1, d) * 0.5, localisation=_loc1d(d, 2.0), update_cap=10.0,
    )
    traj = _trajectory(post, stepper, d, steps=5, every=2, correction=corr)  # assimilated: 1, 3
    assert traj.shape == (6, 1, 1, d, 5) and torch.isfinite(traj).all()
    assert calls["drift"] == 3 * 2 * (2 if q_source == "online" else 1)  # 3 drift evals per forecast
    assert calls["stepper"] > 0
    assert [rec["phys_idx"] for rec in corr.diagnostics.records] == [1, 3]
    assert corr.diagnostics.records[1]["innovation_lag1"] is not None
    assert corr.diagnostics.lag1_autocorrelation().shape == (1,)
    assert set(corr.diagnostics.as_dict()) >= {"q_rms", "innovation_msq", "innovation_lag1"}


def test_forecast_num_steps_defaults_to_sampler_steps(lg, monkeypatch):
    A, Q, rows, r, op = lg
    post, stepper = _real_posterior("fm_ode", A, Q, op)
    seen, original = [], fc.draw_unguided_forecast
    monkeypatch.setattr(fc, "draw_unguided_forecast", lambda *a, **k: (seen.append(a[2]), original(*a, **k))[1])
    _trajectory(post, stepper, A.shape[0], steps=3, correction=ForecastCorrection(enabled=True))
    assert seen and all(s == 6 for s in seen)


def test_field_history_holds_corrected_state(lg):
    """The SI init assert (base == field_history[..., -1]) holds over several steps."""
    A, Q, rows, r, op = lg
    post, stepper = _real_posterior("si", A, Q, op)
    corr = ForecastCorrection(enabled=True, forecast_num_steps=2)
    traj = _trajectory(post, stepper, A.shape[0], steps=5, correction=corr)
    assert torch.isfinite(traj).all() and len(corr.diagnostics.records) == 4


def test_hook_with_groups_and_one_row_per_member(lg):
    A, Q, rows, r, op = lg
    post, stepper = _real_posterior("si", A, Q, op)
    corr = ForecastCorrection(enabled=True, group_size=3, forecast_num_steps=2)
    traj = _trajectory(post, stepper, A.shape[0], members=7, steps=3, correction=corr, rows_per_member=True)
    assert torch.isfinite(traj).all()
    assert corr.diagnostics.records[0]["innovation_mean"].shape == (2,)  # 3 + 4 (ragged)


@pytest.mark.parametrize("kind", ["si", "dm_sde"])
def test_ode_stepper_with_an_sde_forecast_raises(lg, kind):
    A, Q, rows, r, op = lg
    post, stepper = _real_posterior(kind, A, Q, op)
    corr = ForecastCorrection(enabled=True, forecast_stepper=euler_step, forecast_num_steps=2)
    with pytest.raises(ValueError, match="ODE stepper"):
        _trajectory(post, stepper, A.shape[0], steps=2, correction=corr)
    fm, _ = _real_posterior("fm_ode", A, Q, op)  # an ODE forecast with an SDE stepper is fine
    corr = ForecastCorrection(enabled=True, forecast_stepper=euler_maruyama_step, forecast_num_steps=2)
    assert torch.isfinite(_trajectory(fm, euler_step, A.shape[0], steps=2, correction=corr)).all()


def test_batch_split_forecast_equals_unsplit(lg):
    A, Q, rows, r, op = lg
    post, _ = _real_posterior("fm_ode", A, Q, op)
    fh = torch.randn(11, 1, 1, A.shape[0], 1)
    out = []
    for batch_size in (3, 100):
        torch.manual_seed(5)
        out.append(draw_unguided_forecast(post, fh, 4, euler_step, batch_size=batch_size))
    torch.testing.assert_close(out[0], out[1])
    calls = []
    real = post.model._sample
    post.model._sample = lambda **kw: (calls.append(kw["field_history"].shape[0]), real(**kw))[1]
    draw_unguided_forecast(post, fh, 2, euler_step, batch_size=4)
    assert calls == [4, 4, 3]


def test_dm_sde_forecast_honours_drift_time_shift(lg):
    A, Q, rows, r, op = lg
    fh = torch.randn(5, 1, 1, A.shape[0], 1)
    out = []
    for shift in (0.0, 0.5):
        post, stepper = _real_posterior("dm_sde", A, Q, op, shift=shift)
        torch.manual_seed(1)
        out.append(draw_unguided_forecast(post, fh, 4, stepper))
    assert torch.isfinite(out[1]).all() and not torch.allclose(out[0], out[1])


def test_forecast_seed_is_decorrelated_from_the_driver_seed(lg, monkeypatch):
    A, Q, rows, r, op = lg
    d = A.shape[0]
    post, stepper = _real_posterior("fm_ode", A, Q, op)
    latents = []
    monkeypatch.setattr(fc, "draw_unguided_forecast",
                        lambda p, fh, *a, **k: latents.append(torch.randn_like(fh[..., -1])) or latents[-1])
    corr = ForecastCorrection(enabled=True, forecast_seed=0)
    _trajectory(post, stepper, d, steps=2, correction=corr, seed=0)
    first = latents[:2]
    corr.reset()
    _trajectory(post, stepper, d, steps=2, correction=corr, seed=0)
    torch.manual_seed(0)
    torch.randn(6, 1, 1, d)  # x0 in _trajectory
    guided = torch.randn(6, 1, 1, d)  # the guided sampler's latent
    for z in first:
        assert not torch.allclose(z, guided)
    assert not torch.allclose(first[0], first[1])
    assert not torch.allclose(first[0], latents[2])  # a new trajectory gets new noise


# --------------------------------------------------------------------------- #
# Cache, cap, post-update hook, unsupported posteriors
# --------------------------------------------------------------------------- #


def test_fixed_gain_is_cached_per_operator_and_noise(lg):
    A, Q, rows, r, op = lg
    d = A.shape[0]
    c = ForecastCorrection(enabled=True, q_source="bank", residual_bank=torch.randn(20, 1, 1, d))
    x = torch.randn(10, 1, 1, d)
    for var in [r, r, 2 * r]:
        c.correct(x, x + 0.1, torch.zeros(1, rows.numel()), op, var)
    assert len(c._cache._entries) == 2
    other = _selection_operator(d, torch.arange(0, d, 2))
    c.correct(x, x + 0.1, torch.zeros(1, other.num_obs), other, r)
    assert len(c._cache._entries) == 3
    other.obs_matrix = other.obs_matrix.clone()  # a reassigned matrix is not a stale hit
    c.correct(x, x + 0.1, torch.zeros(1, other.num_obs), other, r)
    assert len(c._cache._entries) == 3


def test_update_cap_binds_and_post_update_applies(lg):
    A, Q, rows, r, op = lg
    d = A.shape[0]
    gen = torch.Generator().manual_seed(4)
    x, f, f2 = (torch.randn(20, 1, 1, d, generator=gen) for _ in range(3))
    y = 50 + torch.randn(1, rows.numel(), generator=gen)  # large innovation
    zeros = torch.zeros(20, 2)
    free = ForecastCorrection(enabled=True)
    capped = ForecastCorrection(enabled=True, update_cap=1e-3, post_update=lambda new, old: new * 1.0)
    a = free.correct(x, f, y, op, r, f2=f2, perturbations=zeros)
    b = capped.correct(x, f, y, op, r, f2=f2, perturbations=zeros)
    assert capped.diagnostics.records[0]["cap_fraction"].item() == 1.0
    assert free.diagnostics.records[0]["cap_fraction"].item() == 0.0
    assert (b - x).norm() < (a - x).norm()
    mask = torch.ones(1, 1, 1, d)
    mask[..., 0] = 0
    out = ForecastCorrection(enabled=True, post_update=lambda new, old: new * mask).correct(x, f, y, op, r, f2=f2)
    assert (out[..., 0] == 0).all()


def test_unsupported_posteriors_raise(lg):
    A, Q, rows, r, op = lg
    si_post, _ = _real_posterior("si", A, Q, op)
    fm_post, _ = _real_posterior("dm_sde", A, Q, op)
    surge_si = SurgeStochasticInterpolantPosterior(
        model=si_post.model, obs_operator=op, variance=r, likelihood_model=si_post.likelihood_model)
    surge_fm = SurgeFlowMatchingPosterior(
        model=fm_post.model, obs_operator=op, variance=r, likelihood_model=fm_post.likelihood_model,
        diffusion_term=fm_post.diffusion_term)
    resampling_si = StochasticInterpolantPosterior(
        model=si_post.model, likelihood_model=si_post.likelihood_model, resample=True)

    class Reweighting(ExactMemberPosterior):  # overrides _post_sample, no resample flag
        def _post_sample(self, base, observations, field_history):
            return base, field_history

    for post in [surge_si, surge_fm, resampling_si, Reweighting(A, Q, op, r)]:
        with pytest.raises(NotImplementedError):
            _trajectory(post, euler_maruyama_step, A.shape[0], steps=2,
                        correction=ForecastCorrection(enabled=True))
    with pytest.raises(ValueError):
        _trajectory(si_post, euler_maruyama_step, A.shape[0], steps=2,
                    correction=ForecastCorrection(enabled=True, q_source="bank"))


# --------------------------------------------------------------------------- #
# Localisation
# --------------------------------------------------------------------------- #


def _full(loc):
    return loc.kernel(loc.state_coords, loc.state_coords) * (
        1.0 if loc.keep is None else (loc.keep[:, None] & loc.keep[None, :]).double())


def _min_eig(M):
    return torch.linalg.eigvalsh(M.double()).min().item()


def test_periodic_1d_matches_enkf_taper_and_is_psd():
    loc = _loc1d(40, 5.0)
    full = _full(loc)
    torch.testing.assert_close(full, periodic_taper(40, 5.0), rtol=1e-10, atol=1e-10)
    op = _selection_operator(40, torch.arange(0, 40, 3))
    ref = torch.zeros(1, dtype=torch.float64)
    torch.testing.assert_close(loc.state_obs(op, slice(0, 40), ref), full[:, ::3], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(loc.obs_obs(op, ref), full[::3][:, ::3], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("radius", [1.5, 3.0, 6.0])
def test_periodic_2d_taper_is_psd(radius):
    assert _min_eig(_full(grid_localisation((1, 16, 16), radius, periodic=True))) > -1e-8


def test_multichannel_3d_and_masked_tapers():
    loc = grid_localisation((3, 6, 6), 2.5)
    full = _full(loc)
    assert _min_eig(full) > -1e-8
    assert full[0, 36].item() == pytest.approx(1.0)  # same site, other channel
    loc3d = grid_localisation((4, 5, 5), 2.0, periodic=(False, True, True), channel_axis=False)
    assert loc3d.state_coords.shape == (100, 3) and _min_eig(_full(loc3d)) > -1e-8
    mask = torch.ones(6, 6, dtype=torch.bool)
    mask[2, 3] = False
    masked = grid_localisation((3, 6, 6), 2.5, mask=mask)
    op = _selection_operator(108, torch.arange(0, 108, 5))
    op.C, op.H, op.W = 3, 6, 6
    xy = masked.state_obs(op, slice(0, 108), torch.zeros(1, dtype=torch.float64))
    dead = torch.tensor([c * 36 + 2 * 6 + 3 for c in range(3)])
    assert (xy[dead] == 0).all() and (xy.sum(1) > 0).sum() == 105


def test_block_average_observations_sit_at_block_centres():
    op = LinearObservationOperator(type="super_res", data_size=(1, 8, 8), low_res=2)
    loc = grid_localisation((1, 8, 8), 2.0, periodic=True)
    torch.testing.assert_close(loc.obs_coords(op), torch.tensor(
        [[1.5, 1.5], [1.5, 5.5], [5.5, 1.5], [5.5, 5.5]], dtype=torch.float64))
    assert _min_eig(loc.obs_obs(op, torch.zeros(1, dtype=torch.float64))) > -1e-10


def test_super_resolution_operator_correction_runs():
    """m = 64 block-average observations of a 32 x 32 periodic field, ragged groups."""
    op = LinearObservationOperator(type="super_res", data_size=(1, 32, 32), low_res=8)
    gen = torch.Generator().manual_seed(5)
    x, f, f2 = (torch.randn(20, 1, 32, 32, generator=gen) for _ in range(3))
    c = ForecastCorrection(enabled=True, localisation=grid_localisation((1, 32, 32), 4.0, periodic=True),
                           group_size=8)
    out = c.correct(x, f, torch.randn(1, op.num_obs, generator=gen), op, 0.1, f2=f2)
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert c.diagnostics.records[0]["innovation_mean"].shape == (3,)  # 8 + 8 + 4


# --------------------------------------------------------------------------- #
# MPS (skipped elsewhere)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_correct_matches_cpu_and_seed_keeps_the_mps_stream(lg):
    A, Q, rows, r, op = lg
    d, m = A.shape[0], rows.numel()
    x, f, f2, y, e = _inputs(12, d, m)
    loc = _loc1d(d, 2.0)
    cpu = ForecastCorrection(enabled=True, localisation=loc).correct(x, f, y, op, r, f2=f2, perturbations=e)
    mps = ForecastCorrection(enabled=True, localisation=loc).correct(
        *(t.to("mps") for t in (x, f, y)), op, r, f2=f2.to("mps"), perturbations=e.to("mps"))
    assert mps.device.type == "mps" and mps.dtype == x.dtype
    torch.testing.assert_close(mps.cpu(), cpu, rtol=1e-5, atol=1e-5)
    torch.mps.manual_seed(0)
    ref = torch.randn(4, device="mps").cpu()
    torch.mps.manual_seed(0)
    with fc._forked_rng([torch.device("mps")], 123):
        torch.randn(4, device="mps")
    assert torch.equal(torch.randn(4, device="mps").cpu(), ref)


# --------------------------------------------------------------------------- #
# Real samplers against the Kalman filter (looser: the samplers are approximate)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["si", "fm_ode", "dm_sde"])
def test_real_samplers_with_correction_approach_kalman(kind):
    A, Q, rows, r = _system()
    post, stepper = _real_posterior(kind, A, Q, _selection_operator(A.shape[0], rows), r)
    run = lambda corr: _kalman_run(post, A, Q, rows, r, 3000, 6, 0, corr, num_steps=50, stepper=stepper)  # noqa: E731
    plain_err, _ = run(None)
    err, ratio = run(ForecastCorrection(enabled=True))
    assert plain_err > 0.4  # the per-member bias the correction removes
    assert err < 0.2 and ((ratio > 0.85) & (ratio < 1.12)).all()


# --------------------------------------------------------------------------- #
# The hook lives in sample(): direct sample() calls and D-Flow
# --------------------------------------------------------------------------- #


def test_direct_sample_call_is_corrected_and_attribute_is_restored(lg):
    A, Q, rows, r, op = lg
    d = A.shape[0]
    post, stepper = _real_posterior("fm_ode", A, Q, op)
    _trajectory(post, stepper, d, steps=3, correction=ForecastCorrection(enabled=True, forecast_num_steps=2))
    assert post.forecast_correction is None  # per-call setting restored
    torch.manual_seed(0)
    fh, y = torch.randn(6, 1, 1, d, 1), torch.randn(1, rows.numel())
    kw = dict(base=None, batch_size=6, num_steps=6, field_history=fh, observations=y, stepper=stepper)
    plain = post.sample(**kw)
    post.forecast_correction = corr = ForecastCorrection(enabled=True, forecast_num_steps=2)
    torch.manual_seed(0)
    out, new_fh = post.sample(**kw, return_field_history=True)
    assert len(corr.diagnostics.records) == 1
    assert not torch.equal(out, plain) and torch.equal(new_fh[..., -1], out)


def test_dflow_posterior_is_supported_and_uses_its_flow(lg, monkeypatch):
    from scisi.posterior_models.dflow_posterior import DFlowPosterior

    A, Q, rows, r, op = lg
    fm, _ = _real_posterior("fm_ode", A, Q, op)
    post = DFlowPosterior(model=fm.model, likelihood_model=_Likelihood(op, r), num_optim_steps=2, ode_steps=2, burn=0)
    flows = []
    real = post._flow
    monkeypatch.setattr(post, "_flow", lambda z0, n, *a, **k: (flows.append(n), real(z0, n, *a, **k))[1])
    corr = ForecastCorrection(enabled=True)
    traj = _trajectory(post, euler_step, A.shape[0], steps=3, correction=corr)
    assert torch.isfinite(traj).all() and len(corr.diagnostics.records) == 2
    assert flows and all(n == 2 for n in flows)  # forecasts use ode_steps by default


def test_sample_override_without_hook_raises(lg):
    A, Q, rows, r, op = lg

    class NoHook(ExactMemberPosterior):
        def _apply_forecast_correction(self, base, *a, **k):
            return base  # simulates an override of ``sample`` that skips the hook

    with pytest.raises(NotImplementedError):
        _kalman_run(NoHook(A, Q, op, r), A, Q, rows, r, 10, 2, 0, ForecastCorrection(enabled=True))


# --------------------------------------------------------------------------- #
# Nonlinear observation operators (EnKF-style gains; linearised bank / fixed Q)
# --------------------------------------------------------------------------- #


def _arctan(op, c=1.5):
    from scisi.likelihood_models.observation_operators import PointwiseObservationOperator

    return PointwiseObservationOperator(inner=op, nonlinearity="arctan", strength=c)


def _nl_dense_reference(x, f, y, h, R, e, f2=None, Kq=None, txy=None, tyy=None):
    """Dense EnKF-style reference: cross-covariances with ``h(f)``, innovation ``y + e - h(f)``."""
    X, F = x.reshape(len(x), -1).double(), f.reshape(len(f), -1).double()
    HF = h(f).double()
    m = HF.shape[1]
    Rm = torch.as_tensor(R, dtype=torch.float64) * torch.eye(m, dtype=torch.float64)

    def gain(D, HD, n):
        PH, HPH = D.T @ HD / n, HD.T @ HD / n
        if txy is not None:
            PH, HPH = PH * txy, HPH * tyy
        return PH @ torch.linalg.inv(HPH + Rm)

    K_ens = gain(F - F.mean(0), HF - HF.mean(0), len(F) - 1)
    if Kq is None:
        s = math.sqrt(2.0)
        Kq = gain((F - f2.reshape(len(f2), -1).double()) / s, (HF - h(f2).double()) / s, len(F))
    V = y.double().reshape(-1, m) + e.double() - HF
    return (X + V @ (K_ens - Kq).T).reshape(x.shape)


@pytest.mark.parametrize("localised", [False, True])
def test_nonlinear_online_matches_dense_enkf_reference(lg, localised):
    A, Q, rows, r, op = lg
    d, m = A.shape[0], rows.numel()
    nl = _arctan(op)
    x, f, f2, y, e = _inputs(30, d, m)
    f, f2 = 2 * f, 2 * f2  # well inside the nonlinear range of arctan(1.5 z)
    loc = _loc1d(d, 2.0) if localised else None
    out = ForecastCorrection(enabled=True, localisation=loc).correct(x, f, y, nl, r, f2=f2, perturbations=e)
    txy = tyy = None
    if localised:
        full = loc.kernel(loc.state_coords, loc.state_coords)
        txy, tyy = full[:, rows], full[rows][:, rows]
    ref = _nl_dense_reference(x, f, y, nl, r, e, f2=f2, txy=txy, tyy=tyy)
    torch.testing.assert_close(out.double(), ref, rtol=1e-5, atol=1e-5)
    # ... and it differs from the linear-H update (the nonlinearity matters here).
    lin = ForecastCorrection(enabled=True, localisation=loc).correct(x, f, y, op, r, f2=f2, perturbations=e)
    assert (lin - out).abs().max() > 1e-2


@pytest.mark.parametrize("source", ["bank", "fixed"])
@pytest.mark.parametrize("localised", [False, True])
def test_nonlinear_bank_and_fixed_linearise_at_the_group_mean(lg, source, localised):
    A, Q, rows, r, op = lg
    d, m = A.shape[0], rows.numel()
    nl = _arctan(op)
    x, f, _, y, e = _inputs(24, d, m)
    f = 2 * f
    bank = torch.randn(40, 1, 1, d, generator=torch.Generator().manual_seed(9))
    q = torch.linspace(0.2, 1.0, d)
    if source == "fixed" and localised:
        pytest.skip("fixed Q is not localised")
    loc = _loc1d(d, 2.0) if localised else None
    sizes = (12, 12)
    corr = ForecastCorrection(enabled=True, q_source=source, residual_bank=bank, q_variance=q,
                              localisation=loc, group_size=12)
    out = corr.correct(x, f, y, nl, r, perturbations=e)
    full = None if loc is None else loc.kernel(loc.state_coords, loc.state_coords)
    txy, tyy = (None, None) if full is None else (full[:, rows], full[rows][:, rows])
    refs, start = [], 0
    for n in sizes:
        sl = slice(start, start + n)
        J = nl.jacobian(f[sl].mean(0, keepdim=True)).double()  # [m, d] at the group forecast mean
        Rm = r * torch.eye(m, dtype=torch.float64)
        if source == "bank":
            B = bank.reshape(40, -1).double()
            QH, HQH = B.T @ (B @ J.T) / 40, (B @ J.T).T @ (B @ J.T) / 40
            if txy is not None:
                QH, HQH = QH * txy, HQH * tyy
        else:
            QH = torch.diag(q.double()) @ J.T
            HQH = J @ QH
        Kq = QH @ torch.linalg.inv(HQH + Rm)
        refs.append(_nl_dense_reference(x[sl], f[sl], y, nl, r, e[sl], Kq=Kq, txy=txy, tyy=tyy))
        start += n
    torch.testing.assert_close(out.double(), torch.cat(refs), rtol=1e-5, atol=1e-5)
    assert not corr._cache._entries  # state-dependent gains are not cached


def test_nonlinear_paths_reduce_to_linear_as_strength_vanishes(lg):
    """arctan(c z) / c -> z: every nonlinear path approaches its linear counterpart."""
    A, Q, rows, r, op = lg
    d, m = A.shape[0], rows.numel()
    x, f, f2, y, e = _inputs(20, d, m)
    y = 30 + y  # large innovation, so the cap binds
    bank = torch.randn(40, 1, 1, d, generator=torch.Generator().manual_seed(9))
    for source in ("online", "bank", "fixed"):
        kw = dict(enabled=True, q_source=source, residual_bank=bank, q_variance=0.4,
                  localisation=_loc1d(d, 2.0), update_cap=0.5)
        a = ForecastCorrection(**kw).correct(x, f, y, op, r, f2=f2, perturbations=e)
        corr = ForecastCorrection(**kw)
        b = corr.correct(x, f, y, _arctan(op, 1e-4), r, f2=f2, perturbations=e)
        assert corr.diagnostics.records[0]["cap_fraction"].item() == 1.0
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)


def test_nonlinear_update_cap_measures_the_observation_change(lg):
    A, Q, rows, r, op = lg
    d = A.shape[0]
    nl = _arctan(op, 0.5)
    gen = torch.Generator().manual_seed(4)
    x, f, f2 = (torch.randn(20, 1, 1, d, generator=gen) for _ in range(3))
    y = 3 + torch.randn(1, rows.numel(), generator=gen)
    zeros = torch.zeros(20, rows.numel())
    free = ForecastCorrection(enabled=True).correct(x, f, y, nl, r, f2=f2, perturbations=zeros)
    capped = ForecastCorrection(enabled=True, update_cap=1e-3)
    out = capped.correct(x, f, y, nl, r, f2=f2, perturbations=zeros)
    assert capped.diagnostics.records[0]["cap_fraction"].item() == 1.0
    assert (out - x).norm() < 0.1 * (free - x).norm()


def test_nonlinear_localisation_uses_the_sensor_map(lg):
    A, Q, rows, r, op = lg
    d = A.shape[0]
    loc = _loc1d(d, 2.0)
    nl = _arctan(op)
    torch.testing.assert_close(loc.obs_coords(nl), loc.obs_coords(op))
    ref = torch.zeros(1, dtype=torch.float64)
    torch.testing.assert_close(loc.obs_obs(nl, ref), loc.obs_obs(op, ref))

    class Opaque(nn.Module):  # nonlinear, no linear sensor map
        is_linear = False
        num_obs = rows.numel()

        def forward(self, x):
            return torch.sin(x.reshape(x.shape[0], -1)[:, rows])

    with pytest.raises(ValueError, match="obs_coords"):
        loc.obs_coords(Opaque())
    coords = torch.stack([torch.zeros(rows.numel()), rows.float()], 1).double()  # (H, W) of (1, 1, d)
    explicit = fc.CoordinateLocalisation(loc.state_coords, loc.kernel, (1, 1, d), obs_coords=coords)
    torch.testing.assert_close(explicit.obs_coords(Opaque()), loc.obs_coords(op))
    with pytest.raises(ValueError, match="rows"):
        explicit.obs_coords(_selection_operator(d, torch.arange(0, d, 2)))
    # Online needs only forward; bank / fixed need jvp.
    x, f, f2, y, e = _inputs(10, d, rows.numel())
    ForecastCorrection(enabled=True, localisation=explicit).correct(x, f, y, Opaque(), r, f2=f2)
    with pytest.raises(NotImplementedError, match="jvp"):
        ForecastCorrection(enabled=True, q_source="fixed", q_variance=0.5).correct(x, f, y, Opaque(), r)


def test_velocity_magnitude_operator_correction_matches_reference():
    """NS sensor speed (non-local in the vorticity) on a tiny periodic grid."""
    from scisi.likelihood_models.physics_observation_operators import VelocityMagnitudeOperator

    inner = LinearObservationOperator(type="random", data_size=(1, 8, 8), percent_obs=0.2, seed=3)
    nl = VelocityMagnitudeOperator(inner=inner, norm_std=1.0, velocity_scale=0.5)
    gen = torch.Generator().manual_seed(6)
    x, f, f2 = (torch.randn(16, 1, 8, 8, generator=gen) for _ in range(3))
    y, e = torch.randn(1, nl.num_obs, generator=gen).abs(), 0.1 * torch.randn(16, nl.num_obs, generator=gen)
    loc = grid_localisation((1, 8, 8), 2.0, periodic=True)
    out = ForecastCorrection(enabled=True, localisation=loc).correct(x, f, y, nl, 0.05, f2=f2, perturbations=e)
    full = loc.kernel(loc.state_coords, loc.state_coords)
    idx = inner.obs_indices.reshape(-1)
    ref = _nl_dense_reference(x, f, y, nl, 0.05, e, f2=f2, txy=full[:, idx], tyy=full[idx][:, idx])
    torch.testing.assert_close(out.double(), ref, rtol=1e-5, atol=1e-5)
    bank = torch.randn(20, 1, 8, 8, generator=gen)
    out_b = ForecastCorrection(enabled=True, localisation=loc, q_source="bank", residual_bank=bank,
                               update_cap=0.5).correct(x, f, y, nl, 0.05, perturbations=e)
    assert torch.isfinite(out_b).all()
