"""Forecast-residual correction (scisi.posterior_models.forecast_correction).

CPU-only and fast. The linear-Gaussian checks follow
``paper_experiments/tests/test_forecast_correction_lg.py`` but run through the real
``BasePosterior.sample_trajectory`` hook.
"""

from __future__ import annotations

import math
from typing import Optional

import pytest
import torch
import torch.nn as nn

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
    ResidualBank,
    build_residual_bank,
    grid_localisation,
    periodic_1d_localisation,
    periodic_2d_localisation,
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
        self.calls = 0

    def _prepare_batch(self, batch_size=1, **kwargs):
        out = []
        for v in kwargs.values():
            out.append(None if v is None else v.repeat(batch_size, *[1] * (v.ndim - 1)))
        return tuple(out)

    def sample(self, field_history, base=None, return_field_history=False, **_):
        self.calls += 1
        n = field_history.shape[0]
        x0 = field_history[..., -1].reshape(n, -1)
        x1 = x0 @ self.A.T + torch.randn_like(x0) @ self.L.T
        x1 = x1.reshape(field_history[..., -1].shape)
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
        x0 = field_history[..., -1].reshape(n, -1).double()
        mu = x0 @ self.A64.T
        y = observations.double().reshape(-1, self.H64.shape[0])
        x = mu + (y - mu @ self.H64.T) @ self.Kq.T + torch.randn_like(mu) @ self.Lpost.T
        x = x.float().reshape(field_history[..., -1].shape)
        # A subclass that overrides ``sample`` calls the hook itself.
        x = self._apply_forecast_correction(
            x, field_history, observations, num_steps, stepper, field_cond, pars_cond, batch_size)
        if return_field_history:
            return x, torch.cat([field_history[..., 1:], x.unsqueeze(-1)], -1)
        return x


def _kalman_run(posterior, A, Q, rows, r, members, cycles, seed, correction):
    """Run ``sample_trajectory`` and the Kalman filter on the same observations."""
    g = torch.Generator().manual_seed(seed)
    d, m = A.shape[0], rows.numel()
    H = torch.eye(d, dtype=torch.float64)[rows]
    LQ = torch.linalg.cholesky(Q)
    truth = torch.randn(d, generator=g, dtype=torch.float64)
    ys = []
    mean, cov = torch.zeros(d, dtype=torch.float64), torch.eye(d, dtype=torch.float64)
    for _ in range(cycles):
        truth = A @ truth + LQ @ torch.randn(d, generator=g, dtype=torch.float64)
        y = H @ truth + math.sqrt(r) * torch.randn(m, generator=g, dtype=torch.float64)
        ys.append(y)
        mf, Pf = A @ mean, A @ cov @ A.T + Q
        K = Pf @ H.T @ torch.linalg.inv(H @ Pf @ H.T + r * torch.eye(m, dtype=torch.float64))
        mean, cov = mf + K @ (y - H @ mf), Pf - K @ H @ Pf
    torch.manual_seed(seed + 1)
    x0 = torch.randn(members, 1, 1, d)
    obs = torch.stack(ys, -1).float().unsqueeze(0)  # [1, m, T]
    traj = posterior.sample_trajectory(
        base=x0.clone(), field_history=x0.unsqueeze(-1).clone(), observations=obs,
        ensemble_size=members, batch_size=members, num_steps=4,
        num_physical_steps=cycles + 1, forecast_correction=correction,
    )
    X = traj[..., -1].reshape(members, d).double()
    sd = cov.diagonal().sqrt()
    mean_error = ((X.mean(0) - mean).abs() / sd).max().item()
    var_ratio = torch.cov(X.T).diagonal() / cov.diagonal()
    return mean_error, var_ratio


# --------------------------------------------------------------------------- #
# Kalman reproduction through the hook; bank vs online
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def lg():
    A, Q, rows, r = _system()
    op = _selection_operator(A.shape[0], rows)
    return A, Q, rows, r, op


def test_plain_chain_through_hook_is_biased(lg):
    A, Q, rows, r, op = lg
    post = ExactMemberPosterior(A, Q, op, r)
    err, ratio = _kalman_run(post, A, Q, rows, r, 20000, 8, 0, None)
    assert err > 0.3 or (ratio > 1.1).any()


def test_online_correction_through_hook_matches_kalman(lg):
    A, Q, rows, r, op = lg
    post = ExactMemberPosterior(A, Q, op, r)
    corr = ForecastCorrection(ForecastCorrectionConfig(enabled=True, q_source="online"))
    err, ratio = _kalman_run(post, A, Q, rows, r, 20000, 8, 0, corr)
    assert err < 0.06
    assert ((ratio - 1).abs() < 0.04).all()
    assert len(corr.diagnostics.records) == 8
    assert post.last_forecast_correction is corr


def test_bank_and_online_agree_in_linear_gaussian_case(lg, tmp_path):
    A, Q, rows, r, op = lg
    post = ExactMemberPosterior(A, Q, op, r)
    states = torch.randn(4000, 1, 1, A.shape[0], 1)
    path = tmp_path / "bank.pt"
    bank = build_residual_bank(post, states, num_steps=4, stepper=euler_step,
                               pairs_per_state=5, path=path, seed=3)
    assert len(bank) == 20000 and bank.meta["num_steps"] == 4
    loaded = ResidualBank.load(path)
    assert torch.equal(loaded.residuals, bank.residuals)

    res = {}
    for source, kw in [("online", {}), ("bank", {"residual_bank": str(path)})]:
        corr = ForecastCorrection(enabled=True, q_source=source, **kw)
        res[source] = _kalman_run(post, A, Q, rows, r, 20000, 8, 0, corr)
        err, ratio = res[source]
        assert err < 0.06, source
        assert ((ratio - 1).abs() < 0.04).all(), source
    assert ((res["online"][1] - res["bank"][1]).abs() < 0.05).all()


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
        self.calls = 0

    def forward(self, x, t, field_history=None, field_cond=None, pars_cond=None):
        self.calls += 1
        t4 = torch.clamp(t, 1e-4, 1 - 1e-4).reshape(t.shape[0], 1, 1, 1)
        x0 = field_history[..., -1]
        mu = x0 @ self.A.T
        gamma = self.g0 * (1 - t4)
        coef = (t4 * self.q - t4 * gamma * self.g0) / (t4**2 * self.q + t4 * gamma**2)
        v = x - ((1 - t4) * x0 + t4 * mu)
        return (mu - x0) + (coef * (v @ self.U)) @ self.U.T


class _FMVelocity(_SIDrift):
    def forward(self, x, t, field_history=None, field_cond=None, pars_cond=None):
        self.calls += 1
        t4 = torch.clamp(t, 1e-4, 1 - 1e-4).reshape(t.shape[0], 1, 1, 1)
        mu = field_history[..., -1] @ self.A.T
        alpha = 1 - t4
        cov = alpha**2 + t4**2 * self.q
        return mu + (((t4 * self.q - alpha) / cov) * ((x - t4 * mu) @ self.U)) @ self.U.T


def _real_posterior(kind: str, A, Q, op, r=0.25):
    if kind == "si":
        model = FollmerStochasticInterpolant(
            interpolation=LinearStochasticInterpolation(gamma_multiplier=1.0),
            drift_model=_SIDrift(A, Q),
        )
        lik = InterpolantGaussianLikelihood(model=model, obs_operator=op, variance=r,
                                            likelihood_mode="inflated", model_class="si")
        return StochasticInterpolantPosterior(model=model, likelihood_model=lik), euler_maruyama_step
    model = FlowMatchingModel(interpolation=LinearDeterministicInterpolation(),
                              drift_model=_FMVelocity(A, Q))
    lik = InterpolantGaussianLikelihood(model=model, obs_operator=op, variance=r,
                                        likelihood_mode="inflated", model_class="fm")
    if kind == "fm_ode":
        return FlowMatchingPosterior(model=model, likelihood_model=lik, diffusion_term=None), euler_step
    g = endpoint_vanishing_diffusion(model.interpolation)
    return FlowMatchingPosterior(model=model, likelihood_model=lik, diffusion_term=g), euler_maruyama_step


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
    for off in [None, ForecastCorrectionConfig(), ForecastCorrection(enabled=False)]:
        out = _trajectory(post, stepper, A.shape[0], every=every, correction=off)
        assert torch.equal(ref, out)
    # And the global RNG is left in the same state.
    torch.manual_seed(0)
    _trajectory(post, stepper, A.shape[0], every=every, correction=None)
    after_off = torch.rand(3)
    _trajectory(post, stepper, A.shape[0], every=every)
    assert torch.equal(after_off, torch.rand(3))


# --------------------------------------------------------------------------- #
# Real SI-SDE / FM-ODE / DM-SDE posteriors; forecast_num_steps honoured
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["si", "fm_ode", "dm_sde"])
@pytest.mark.parametrize("q_source", ["online", "bank"])
def test_real_posteriors_run_and_forecast_steps_are_honoured(lg, kind, q_source):
    A, Q, rows, r, op = lg
    d = A.shape[0]
    post, stepper = _real_posterior(kind, A, Q, op)
    drift = post.model.drift_model
    bank = torch.randn(64, 1, 1, d) * 0.5 if q_source == "bank" else None
    stepper_calls = {"n": 0}

    def counting_stepper(**kw):
        stepper_calls["n"] += 1
        return stepper(**kw)

    corr = ForecastCorrection(
        enabled=True, forecast_num_steps=3, forecast_stepper=counting_stepper,
        q_source=q_source, residual_bank=bank, localisation=periodic_1d_localisation(d, 2.0),
        update_cap=10.0,
    )
    inside = {"on": False, "n": 0}
    original_forward = drift.forward

    def counted(*a, **k):
        if inside["on"]:
            inside["n"] += 1
        return original_forward(*a, **k)

    drift.forward = counted
    original_draw = corr.draw_forecast

    def draw(*a, **k):
        inside["on"] = True
        try:
            return original_draw(*a, **k)
        finally:
            inside["on"] = False

    corr.draw_forecast = draw
    steps, every = 5, 2  # assimilated physical steps 1, 3
    traj = _trajectory(post, stepper, d, steps=steps, every=every, correction=corr)
    assert traj.shape == (6, 1, 1, d, steps) and torch.isfinite(traj).all()
    n_assim = 2
    forecasts = n_assim * (2 if q_source == "online" else 1)
    assert inside["n"] == 3 * forecasts  # 3 drift evaluations per forecast
    assert stepper_calls["n"] > 0
    assert [rec["phys_idx"] for rec in corr.diagnostics.records] == [1, 3]
    assert corr.diagnostics.records[1]["innovation_lag1"] is not None
    assert corr.diagnostics.lag1_autocorrelation().shape == (1,)


def test_forecast_num_steps_defaults_to_sampler_steps(lg):
    A, Q, rows, r, op = lg
    post, stepper = _real_posterior("fm_ode", A, Q, op)
    seen = []
    corr = ForecastCorrection(enabled=True)
    original = corr.draw_forecast
    corr.draw_forecast = lambda *a, **k: (seen.append(k["num_steps"]), original(*a, **k))[1]
    _trajectory(post, stepper, A.shape[0], steps=3, correction=corr)
    corr_cfg = corr.config
    assert corr_cfg.forecast_num_steps is None and seen and all(s == 6 for s in seen)


def test_field_history_holds_corrected_state(lg):
    """The SI init assert (base == field_history[..., -1]) holds over several steps."""
    A, Q, rows, r, op = lg
    post, stepper = _real_posterior("si", A, Q, op)
    corr = ForecastCorrection(enabled=True, forecast_num_steps=2)
    traj = _trajectory(post, stepper, A.shape[0], steps=5, correction=corr)
    assert torch.isfinite(traj).all() and len(corr.diagnostics.records) == 4


# --------------------------------------------------------------------------- #
# Per-group statistics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("q_source", ["online", "bank"])
@pytest.mark.parametrize("localised", [False, True])
def test_statistics_are_per_group(lg, q_source, localised):
    A, Q, rows, r, op = lg
    d, m = A.shape[0], rows.numel()
    G, E = 3, 40
    gen = torch.Generator().manual_seed(1)
    shifts = torch.tensor([0.0, 5.0, -3.0]).repeat_interleave(E).reshape(-1, 1, 1, 1)
    scales = torch.tensor([1.0, 0.2, 3.0]).repeat_interleave(E).reshape(-1, 1, 1, 1)
    x = torch.randn(G * E, 1, 1, d, generator=gen)
    f = shifts + scales * torch.randn(G * E, 1, 1, d, generator=gen)
    res = torch.randn(G * E, 1, 1, d, generator=gen)
    y = torch.randn(G, m, generator=gen).repeat_interleave(E, 0)  # one row per member
    e = torch.randn(G * E, m, generator=gen)
    loc = periodic_1d_localisation(d, 2.0) if localised else None
    kw = dict(enabled=True, q_source=q_source, localisation=loc,
              residual_bank=torch.randn(100, 1, 1, d, generator=gen))

    batched = ForecastCorrection(group_size=E, **kw)
    out = batched(x, f, y, operator=op, obs_noise_var=r, residuals=res, perturbations=e)
    for g in range(G):
        sl = slice(g * E, (g + 1) * E)
        single = ForecastCorrection(**kw)
        ref = single(x[sl], f[sl], y[sl], operator=op, obs_noise_var=r,
                     residuals=res[sl], perturbations=e[sl])
        torch.testing.assert_close(out[sl], ref, rtol=1e-5, atol=1e-5)
        assert batched.diagnostics.records[0]["innovation_mean"][g] == pytest.approx(
            single.diagnostics.records[0]["innovation_mean"][0].item(), abs=1e-5)
    pooled = ForecastCorrection(**kw)(x, f, y, operator=op, obs_noise_var=r, residuals=res, perturbations=e)
    assert not torch.allclose(out, pooled, atol=1e-3)
    with pytest.raises(ValueError):
        ForecastCorrection(group_size=7, **kw)(x, f, y, operator=op, obs_noise_var=r,
                                               residuals=res, perturbations=e)


def test_hook_with_groups_and_one_row_per_member(lg):
    A, Q, rows, r, op = lg
    post, stepper = _real_posterior("si", A, Q, op)
    corr = ForecastCorrection(enabled=True, group_size=3, forecast_num_steps=2)
    traj = _trajectory(post, stepper, A.shape[0], members=6, steps=3, correction=corr,
                       rows_per_member=True)
    assert torch.isfinite(traj).all()
    assert corr.diagnostics.records[0]["innovation_mean"].shape == (2,)


# --------------------------------------------------------------------------- #
# Chunking, caching, cap, post-update hook
# --------------------------------------------------------------------------- #


def test_chunking_and_cache_do_not_change_the_update(lg):
    A, Q, rows, r, op = lg
    d = A.shape[0]
    gen = torch.Generator().manual_seed(2)
    x, f, res = (torch.randn(30, 1, 1, d, generator=gen) for _ in range(3))
    y = torch.randn(1, rows.numel(), generator=gen)
    e = torch.randn(30, rows.numel(), generator=gen)
    bank = torch.randn(50, 1, 1, d, generator=gen)
    outs = []
    for chunk, limit in [(4096, 2**24), (3, 2**24), (3, 0)]:
        loc = periodic_1d_localisation(d, 2.0, cache_limit=limit)
        for source in ["online", "bank"]:
            c = ForecastCorrection(enabled=True, q_source=source, residual_bank=bank,
                                   localisation=loc, row_chunk=chunk)
            outs.append(c(x, f, y, operator=op, obs_noise_var=r, residuals=res, perturbations=e))
    for i in range(2, len(outs)):
        torch.testing.assert_close(outs[i], outs[i % 2])


def test_bank_gain_is_cached_per_operator_and_noise(lg):
    A, Q, rows, r, op = lg
    d = A.shape[0]
    c = ForecastCorrection(enabled=True, q_source="bank", residual_bank=torch.randn(20, 1, 1, d))
    x = torch.randn(10, 1, 1, d)
    for var in [r, r, 2 * r]:
        c(x, x + 0.1, torch.zeros(1, rows.numel()), operator=op, obs_noise_var=var)
    assert len(c._kq_cache) == 2
    other = _selection_operator(d, torch.arange(0, d, 2))
    c(x, x + 0.1, torch.zeros(1, other.num_obs), operator=other, obs_noise_var=r)
    assert len(c._kq_cache) == 3


def test_update_cap_binds_and_post_update_applies(lg):
    A, Q, rows, r, op = lg
    d = A.shape[0]
    gen = torch.Generator().manual_seed(4)
    x, f, res = (torch.randn(20, 1, 1, d, generator=gen) for _ in range(3))
    y = 50 + torch.randn(1, rows.numel(), generator=gen)  # large innovation
    free = ForecastCorrection(enabled=True)
    capped = ForecastCorrection(enabled=True, update_cap=1e-3,
                                post_update=lambda new, old: new * 1.0)
    a = free(x, f, y, operator=op, obs_noise_var=r, residuals=res, perturbations=torch.zeros(20, 2))
    b = capped(x, f, y, operator=op, obs_noise_var=r, residuals=res, perturbations=torch.zeros(20, 2))
    assert capped.diagnostics.records[0]["cap_fraction"].item() == 1.0
    assert free.diagnostics.records[0]["cap_fraction"].item() == 0.0
    assert (b - x).norm() < (a - x).norm()
    mask = torch.ones(1, 1, 1, d)
    mask[..., 0] = 0
    masked = ForecastCorrection(enabled=True, post_update=lambda new, old: new * mask)
    out = masked(x, f, y, operator=op, obs_noise_var=r, residuals=res)
    assert (out[..., 0] == 0).all()


# --------------------------------------------------------------------------- #
# Unsupported subclasses
# --------------------------------------------------------------------------- #


def test_unsupported_subclasses_raise(lg):
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
    for post in [surge_si, surge_fm, resampling_si]:
        with pytest.raises(NotImplementedError):
            _trajectory(post, euler_maruyama_step, A.shape[0], steps=2,
                        correction=ForecastCorrectionConfig(enabled=True))
    with pytest.raises(ValueError):
        _trajectory(si_post, euler_maruyama_step, A.shape[0], steps=2,
                    correction=ForecastCorrectionConfig(enabled=True, q_source="bank"))


# --------------------------------------------------------------------------- #
# Localisation helpers
# --------------------------------------------------------------------------- #


def _min_eig(M):
    return torch.linalg.eigvalsh(M.double()).min().item()


def test_periodic_1d_matches_enkf_taper_and_is_psd():
    loc = periodic_1d_localisation(40, 5.0)
    full = loc.full_taper()
    torch.testing.assert_close(full, periodic_taper(40, 5.0), rtol=1e-10, atol=1e-10)
    assert _min_eig(full) > -1e-10
    op = _selection_operator(40, torch.arange(0, 40, 3))
    yy = loc.obs_obs(op, torch.zeros(1, dtype=torch.float64))
    xy = loc.state_obs(op, slice(0, 40), torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(xy, full[:, ::3], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(yy, full[::3][:, ::3], rtol=1e-6, atol=1e-6)
    assert _min_eig(yy) > -1e-10


@pytest.mark.parametrize("radius", [1.5, 3.0, 6.0])
@pytest.mark.parametrize("combine", ["euclidean", "product"])
def test_periodic_2d_taper_is_psd(radius, combine):
    loc = periodic_2d_localisation((1, 16, 16), radius, combine=combine)
    assert _min_eig(loc.full_taper()) > -1e-8


def test_nonperiodic_multichannel_and_3d_tapers_are_psd():
    ct = torch.tensor([[1.0, 0.5, 0.2], [0.5, 1.0, 0.5], [0.2, 0.5, 1.0]])
    loc2d = grid_localisation((3, 8, 8), 2.5, channel_taper=ct)
    assert _min_eig(loc2d.full_taper()) > -1e-8
    # 3D: two variables x two levels stacked into C = 4 channels.
    levels = torch.tensor([[0.0], [2.0], [0.0], [2.0]])
    var_taper = torch.tensor([[1.0, 0.3], [0.3, 1.0]])
    ct4 = var_taper[[0, 0, 1, 1]][:, [0, 0, 1, 1]]
    loc3d = grid_localisation((4, 6, 6), 2.0, channel_coords=levels, channel_taper=ct4)
    assert _min_eig(loc3d.full_taper()) > -1e-8
    # Same site, same variable, different level: tapered; same level: not.
    full = loc3d.full_taper()
    assert full[0, 36].item() < 1.0 and full[0, 0].item() == pytest.approx(1.0)


def test_block_average_observations_sit_at_block_centres():
    op = LinearObservationOperator(type="super_res", data_size=(1, 8, 8), low_res=2)
    loc = periodic_2d_localisation((1, 8, 8), 2.0)
    coords = loc.obs_coords(op)
    torch.testing.assert_close(coords[:, 1:], torch.tensor(
        [[1.5, 1.5], [1.5, 5.5], [5.5, 1.5], [5.5, 5.5]], dtype=torch.float64))
    yy = loc.obs_obs(op, torch.zeros(1, dtype=torch.float64))
    assert _min_eig(yy) > -1e-10


def test_super_resolution_operator_correction_runs():
    """m = 64 block-average observations of a 32 x 32 periodic field."""
    op = LinearObservationOperator(type="super_res", data_size=(1, 32, 32), low_res=8)
    loc = periodic_2d_localisation((1, 32, 32), 4.0)
    gen = torch.Generator().manual_seed(5)
    x, f, res = (torch.randn(16, 1, 32, 32, generator=gen) for _ in range(3))
    y = torch.randn(1, op.num_obs, generator=gen)
    c = ForecastCorrection(enabled=True, localisation=loc, row_chunk=256)
    out = c(x, f, y, operator=op, obs_noise_var=0.1, residuals=res)
    assert out.shape == x.shape and torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Real samplers against the Kalman filter (looser: the samplers are approximate)
# --------------------------------------------------------------------------- #


def _real_kalman(kind, correction, members=3000, cycles=6, seed=0):
    A, Q, rows, r = _system()
    d, m = A.shape[0], rows.numel()
    post, stepper = _real_posterior(kind, A, Q, _selection_operator(d, rows), r)
    H = torch.eye(d, dtype=torch.float64)[rows]
    g = torch.Generator().manual_seed(seed)
    LQ = torch.linalg.cholesky(Q)
    truth = torch.randn(d, generator=g, dtype=torch.float64)
    mean, cov, ys = torch.zeros(d, dtype=torch.float64), torch.eye(d, dtype=torch.float64), []
    for _ in range(cycles):
        truth = A @ truth + LQ @ torch.randn(d, generator=g, dtype=torch.float64)
        y = H @ truth + math.sqrt(r) * torch.randn(m, generator=g, dtype=torch.float64)
        ys.append(y)
        mf, Pf = A @ mean, A @ cov @ A.T + Q
        K = Pf @ H.T @ torch.linalg.inv(H @ Pf @ H.T + r * torch.eye(m, dtype=torch.float64))
        mean, cov = mf + K @ (y - H @ mf), Pf - K @ H @ Pf
    torch.manual_seed(seed + 1)
    x0 = torch.randn(members, 1, 1, d)
    traj = post.sample_trajectory(
        base=x0.clone(), field_history=x0.unsqueeze(-1).clone(),
        observations=torch.stack(ys, -1).float().unsqueeze(0), ensemble_size=members,
        batch_size=members, num_steps=50, num_physical_steps=cycles + 1, stepper=stepper,
        forecast_correction=correction,
    )
    X = traj[..., -1].reshape(members, d).double()
    err = ((X.mean(0) - mean).abs() / cov.diagonal().sqrt()).max().item()
    return err, torch.cov(X.T).diagonal() / cov.diagonal()


@pytest.mark.parametrize("kind", ["si", "fm_ode", "dm_sde"])
def test_real_samplers_with_correction_approach_kalman(kind):
    plain_err, _ = _real_kalman(kind, None)
    err, ratio = _real_kalman(kind, ForecastCorrection(enabled=True))
    assert plain_err > 0.4  # the per-member bias the correction removes
    assert err < 0.2
    assert ((ratio > 0.85) & (ratio < 1.12)).all()


# --------------------------------------------------------------------------- #
# The hook lives in sample(): direct sample() calls and D-Flow
# --------------------------------------------------------------------------- #


def test_direct_sample_call_is_corrected_and_attribute_is_restored(lg):
    A, Q, rows, r, op = lg
    d = A.shape[0]
    post, stepper = _real_posterior("fm_ode", A, Q, op)
    corr = ForecastCorrection(enabled=True, forecast_num_steps=2)
    _trajectory(post, stepper, d, steps=3, correction=corr)
    assert post.forecast_correction is None  # per-call setting restored
    torch.manual_seed(0)
    fh = torch.randn(6, 1, 1, d, 1)
    y = torch.randn(1, rows.numel())
    plain = post.sample(base=None, batch_size=6, num_steps=6, field_history=fh,
                        observations=y, stepper=stepper)
    post.forecast_correction = ForecastCorrectionConfig(enabled=True, forecast_num_steps=2)
    torch.manual_seed(0)
    out, new_fh = post.sample(base=None, batch_size=6, num_steps=6, field_history=fh,
                              observations=y, stepper=stepper, return_field_history=True)
    assert isinstance(post.forecast_correction, ForecastCorrection)
    assert len(post.forecast_correction.diagnostics.records) == 1
    assert not torch.equal(out, plain)
    assert torch.equal(new_fh[..., -1], out)


def test_dflow_posterior_is_supported(lg):
    from scisi.posterior_models.dflow_posterior import DFlowPosterior

    A, Q, rows, r, op = lg
    fm, _ = _real_posterior("fm_ode", A, Q, op)
    post = DFlowPosterior(model=fm.model, likelihood_model=_Likelihood(op, r),
                          num_optim_steps=2, ode_steps=2, burn=0)
    corr = ForecastCorrection(enabled=True, forecast_num_steps=2)
    traj = _trajectory(post, euler_step, A.shape[0], steps=3, correction=corr)
    assert torch.isfinite(traj).all() and len(corr.diagnostics.records) == 2


def test_sample_override_without_hook_raises(lg):
    A, Q, rows, r, op = lg

    class NoHook(ExactMemberPosterior):
        def _apply_forecast_correction(self, base, *a, **k):
            return base  # simulates an override of ``sample`` that skips the hook

    post = NoHook(A, Q, op, r)
    with pytest.raises(NotImplementedError):
        _kalman_run(post, A, Q, rows, r, 10, 2, 0, ForecastCorrection(enabled=True))
