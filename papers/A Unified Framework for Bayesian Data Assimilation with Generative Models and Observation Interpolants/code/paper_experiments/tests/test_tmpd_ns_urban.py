"""TMPD wired into the Navier--Stokes and urban (uDALES) cases (TMPD.md).

CPU tests on tiny random-weight priors (no data or trained weights), as in
test_forecast_correction_ns_urban.py / test_surge_ours.py. They check the
plumbing: the builder (FM prior, sqrt(alpha beta) diffusion, row_sum), that
TMPD's prior SDE is Ours (DM-SDE)'s, that a guided step takes 2 VJPs, the
``vjp`` metric, the sparse-assimilation schedule, the refusal of a nonlinear
operator, and that the default lineups and the other methods' rows are unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_forecast_correction_ns_urban import (  # noqa: E402
    G,
    L,
    _building_mask,
    _ns_ops,
    _prior,
    _scenario_op,
    _truth,
    _urban_op,
)
from cases.navier_stokes import driver as ns_driver  # noqa: E402
from cases.navier_stokes._ns_pipeline import (  # noqa: E402
    attach_nfe_counter,
    build_posterior,
    compute_metrics,
    run_assimilation,
)
from cases.urban import driver as urban_driver  # noqa: E402
from cases.urban import _urban_pipeline  # noqa: E402
from common.runner import RunContext  # noqa: E402
from results_schema import Case, Method, Metric, Scenario  # noqa: E402
from scisi.likelihood_models.tmpd import TMPDLikelihood  # noqa: E402
from scisi.posterior_models.flow_matching_posterior import FlowMatchingPosterior  # noqa: E402
from scisi.posterior_models.tmpd_posterior import TMPDPosterior  # noqa: E402

torch.set_num_threads(2)
EXP = Path(__file__).resolve().parents[1]


def _cfg(name: str):  # type: ignore[no-untyped-def]
    return OmegaConf.load(EXP / "configs" / "method" / f"{name}.yaml")


def _build(method: str, prior, op, variance=0.0025, cfg=None, M=4):  # type: ignore[no-untyped-def]
    name = {"TMPD": "tmpd", "Ours (DM-SDE)": "dm_sde"}[method]
    return build_posterior(
        method_name=method, method_cfg=cfg if cfg is not None else _cfg(name),
        prior=prior, obs_operator=op, variance=variance, likelihood_ensemble_size=4,
        likelihood_mode="dps_jacobian_free", case_key="navier_stokes",
        scenario_key="sparse_5", num_steps=M,
    )


def _rollout(method, prior, op, truth, T, E=4, M=4, every=None, batch=None, seed=0):  # type: ignore[no-untyped-def]
    model, post, stepper = _build(method, prior, op, M=M)
    nfe = attach_nfe_counter(model)
    torch.manual_seed(seed)
    return post, run_assimilation(
        posterior=post, model=model, truth_obs=truth, ensemble_size=E, num_steps=M,
        num_physical_steps=T, stepper=stepper, nfe_counter=nfe, gaussian_base=True,
        assimilate_every=every, batch_size=batch,
    )


# --------------------------------------------------------------------------- #
# Registry / defaults
# --------------------------------------------------------------------------- #


def test_tmpd_is_wired_but_off_the_default_lineups():
    assert Method.TMPD in ns_driver.WIRED_METHODS
    assert Method.TMPD not in ns_driver.NS_METHODS
    assert Method.TMPD not in ns_driver.VARIANT_METHODS
    assert ns_driver.METHOD_CONFIG_NAME[Method.TMPD] == "tmpd"
    assert Method.TMPD in urban_driver.WIRED_METHODS
    assert Method.TMPD not in urban_driver.URBAN_METHODS
    assert urban_driver.METHOD_CONFIG_NAME[Method.TMPD] == "tmpd"
    # The default lineups as before TMPD was wired.
    assert ns_driver.NS_METHODS == (
        Method.OURS_SI_SDE, Method.OURS_FM_ODE, Method.OURS_DM_SDE, Method.FLOWDAS,
        Method.SURGE_FLOWDAS, Method.SDA, Method.SURGE_SDA, Method.D_FLOW_SGLD,
        Method.GUIDED_FM_FIG, Method.ENKF, Method.PARTICLE_FILTER,
    )
    assert urban_driver.URBAN_METHODS == (
        Method.OURS_SI_SDE, Method.OURS_FM_ODE, Method.OURS_DM_SDE,
        Method.SURGE_FLOWDAS, Method.SURGE_SDA, Method.D_FLOW_SGLD, Method.GUIDED_FM_FIG,
    )
    ns = ns_driver.NavierStokesRunner(OmegaConf.create({"case": {}}), seeds=[0])
    assert tuple(ns.methods()) == ns_driver.NS_METHODS
    assert ns_driver.NavierStokesRunner(
        OmegaConf.create({"case": {}, "ns_methods": ["TMPD"]}), seeds=[0]
    ).methods() == (Method.TMPD,)
    ur = urban_driver.UrbanRunner(OmegaConf.create({"case": {}}), seeds=[0])
    assert tuple(ur.methods()) == urban_driver.URBAN_METHODS


def test_tmpd_config_keeps_the_analytical_default():
    cfg = _cfg("tmpd")
    # analytical / multistep read `covariance` (full); NS / urban read the other key.
    assert cfg.likelihood_model.covariance == "full"
    assert cfg.likelihood_model.covariance_ns_urban == "row_sum"
    assert cfg.posterior_model.guidance_weight == "score_substitution"
    assert cfg.stepper == "sde" == _cfg("dm_sde").stepper
    assert float(cfg.diffusion_term.multiplier) == float(_cfg("dm_sde").diffusion_term.multiplier)


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scen", ["sparse", "superres"])
def test_builder_matches_ours_dm_sde(scen):
    prior, op = _prior(1), _ns_ops()[scen]
    model, post, stepper = _build("TMPD", prior, op)
    m_dm, p_dm, s_dm = _build("Ours (DM-SDE)", prior, op)
    assert type(post) is TMPDPosterior and type(p_dm) is FlowMatchingPosterior
    lik = post.likelihood_model
    assert type(lik) is TMPDLikelihood and lik.covariance == "row_sum"
    assert lik.obs_operator is op and lik.original_variance == pytest.approx(0.0025)
    assert model is prior.fm_model is m_dm and post.model is prior.fm_model
    assert stepper is s_dm
    assert post.guidance_weight == "score_substitution"
    # Explicit sqrt(alpha beta) diffusion, the same as Ours (DM-SDE).
    assert post.default_diffusion_term is False and p_dm.default_diffusion_term is False
    t = torch.linspace(0.0, 1.0, 11).reshape(-1, 1)
    torch.testing.assert_close(post.diffusion_term(t), p_dm.diffusion_term(t))
    assert float(post.diffusion_term(torch.zeros(1, 1))) == 0.0
    assert post.gaussian_base


def test_builder_reads_the_ns_urban_covariance_and_multiplier():
    prior, op = _prior(1), _ns_ops()["sparse"]
    cfg = _cfg("tmpd")
    cfg.likelihood_model.covariance_ns_urban = "diagonal"
    cfg.diffusion_term.multiplier = 0.5
    _, post, _ = _build("TMPD", prior, op, cfg=cfg)
    assert post.likelihood_model.covariance == "diagonal"
    t = torch.full((1, 1), 0.3)
    _, p_dm, _ = _build("Ours (DM-SDE)", prior, op)
    torch.testing.assert_close(post.diffusion_term(t), 0.5 * p_dm.diffusion_term(t))


@pytest.mark.parametrize("scen", ["sparse_5_arctan", "sparse_5_speed"])
def test_nonlinear_operator_is_refused(scen):
    op = _scenario_op(scen, (1, G, G), strength=2.0 if scen.endswith("arctan") else None)
    assert not op.is_linear
    with pytest.raises(NotImplementedError, match="LINEAR"):
        _build("TMPD", _prior(1), op)


def test_urban_nonlinear_operator_is_refused():
    keep = _building_mask().unsqueeze(0).expand(4, G, G)
    op = _scenario_op("sparse_1p5_arctan_vel", (4, G, G), strength=2.0, fluid_mask=keep)
    with pytest.raises(NotImplementedError):
        _build("TMPD", _prior(4, [0.4] * 4), op)


# --------------------------------------------------------------------------- #
# One step
# --------------------------------------------------------------------------- #


def _step_inputs(op, E=3):  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(1)
    x = torch.randn(E, 1, G, G, generator=g)
    fh = torch.randn(E, 1, G, G, L, generator=g)
    y = op(torch.randn(1, 1, G, G, generator=g)).expand(E, -1).contiguous()
    return x, fh, y


def _one(post, x, fh, y, t, dt=0.05, seed=7):  # type: ignore[no-untyped-def]
    torch.manual_seed(seed)
    with torch.no_grad():
        return post._one_step(base=x.clone(), t=torch.full((1, 1), t), field_history=fh,
                              field_cond=None, pars_cond=None, observations=y,
                              dt=torch.tensor(dt))


@pytest.mark.parametrize("scen", ["sparse", "superres"])
def test_guided_step_takes_two_vjps_and_moves_the_state(scen):
    prior, op = _prior(1), _ns_ops()[scen]
    _, post, _ = _build("TMPD", prior, op)
    x, fh, y = _step_inputs(op)
    lik = post.likelihood_model
    lik.vjp_count = 0
    out = _one(post, x, fh, y, 0.37)
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert lik.vjp_count == 2
    # The guidance acts: a nearly uninformative likelihood gives another step.
    _, flat, _ = _build("TMPD", prior, op, variance=1e12)
    assert not torch.allclose(out, _one(flat, x, fh, y, 0.37), atol=1e-6)
    # The first node (tau = 0) is unguided: no VJP.
    lik.vjp_count = 0
    _one(post, x, fh, y, 0.0)
    assert lik.vjp_count == 0


def test_prior_part_of_the_step_is_ours_dm_sde():
    """With an uninformative likelihood (R -> inf) TMPD's step is the DM-SDE step:
    same drift v + 1/2 g^2 s, same g, same noise draw."""
    prior, op = _prior(1), _ns_ops()["sparse"]
    _, tmpd, _ = _build("TMPD", prior, op, variance=1e12)
    _, dm, _ = _build("Ours (DM-SDE)", prior, op, variance=1e12)
    x, fh, y = _step_inputs(op)
    for t in (0.2, 0.37, 0.9):
        torch.testing.assert_close(_one(tmpd, x, fh, y, t), _one(dm, x, fh, y, t),
                                   atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# Rollouts, vjp metric
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scen", ["sparse", "superres"])
def test_ns_rollout_and_vjp_metric(scen):
    prior, op = _prior(1), _ns_ops()[scen]
    T, M = L + 2, 4
    truth = _truth(op, 1, T)
    _, res = _rollout("TMPD", prior, op, truth, T, M=M)
    assert res.posterior_trajectory.shape == (4, 1, G, G, T)
    assert torch.isfinite(res.posterior_trajectory).all()
    # Guided nodes tau = 1/M .. (M-1)/M, 2 VJPs each, per generated step.
    assert res.vjp_per_step == pytest.approx(2 * (M - 1))
    assert res.nfe_per_step == pytest.approx(M)
    metrics = compute_metrics(res, op, L)
    assert metrics["vjp"] == pytest.approx(2 * (M - 1))
    # Other methods: no vjp anywhere.
    _, dm = _rollout("Ours (DM-SDE)", prior, op, truth, T, M=M)
    assert dm.vjp_per_step is None and "vjp" not in compute_metrics(dm, op, L)


def test_vjp_per_step_is_per_member_under_batch_chunking():
    prior, op = _prior(1), _ns_ops()["sparse"]
    T, M = L + 1, 3
    truth = _truth(op, 1, T)
    _, one = _rollout("TMPD", prior, op, truth, T, M=M)
    _, chunked = _rollout("TMPD", prior, op, truth, T, M=M, batch=2)
    assert chunked.vjp_per_step == pytest.approx(one.vjp_per_step) == pytest.approx(2 * (M - 1))
    assert chunked.nfe_per_step == pytest.approx(one.nfe_per_step)


def test_sparse_schedule_free_runs_like_ours_dm_sde():
    """Free-running steps use the prior with TMPD's own g (= DM-SDE's), no VJP."""
    prior, op = _prior(1), _ns_ops()["sparse"]
    T, M = L + 3, 3
    truth = _truth(op, 1, T)
    # every=T: only the first generated step is assimilated.
    _, res = _rollout("TMPD", prior, op, truth, T, M=M, every=T)
    assert res.vjp_per_step == pytest.approx(2 * (M - 1) / 3)
    # No assimilation at all (first update past the horizon): identical to DM-SDE.
    kw = dict(M=M, seed=3)
    model, post, stepper = _build("TMPD", prior, op, M=M)
    m2, p2, s2 = _build("Ours (DM-SDE)", prior, op, M=M)
    outs = []
    for mod, pst, stp in ((model, post, stepper), (m2, p2, s2)):
        torch.manual_seed(kw["seed"])
        outs.append(run_assimilation(
            posterior=pst, model=mod, truth_obs=truth, ensemble_size=4, num_steps=M,
            num_physical_steps=T, stepper=stp, nfe_counter=attach_nfe_counter(mod),
            gaussian_base=True, assimilate_from=T,
        ).posterior_trajectory)
    torch.testing.assert_close(outs[0], outs[1])


def test_urban_rollout_and_vjp_metric():
    prior = _prior(4, [0.4] * 4)
    keep = _building_mask().unsqueeze(0).expand(4, G, G)
    op = _urban_op(keep)
    T, M = L + 3, 3
    truth = _truth(op, 4, T)
    _, res = _rollout("TMPD", prior, op, truth, T, M=M, every=2)
    traj = res.posterior_trajectory
    assert traj.shape == (4, 4, G, G, T) and torch.isfinite(traj).all()
    # Assimilated at L and L + 2 of L .. L + 2.
    assert res.vjp_per_step == pytest.approx(2 * 2 * (M - 1) / 3)
    metrics = _urban_pipeline.compute_metrics(res, op, keep.to(torch.float32), L)
    assert metrics["vjp"] == pytest.approx(res.vjp_per_step)


# --------------------------------------------------------------------------- #
# Driver record emitters
# --------------------------------------------------------------------------- #


def _rows(runner, method, metrics):  # type: ignore[no-untyped-def]
    case = runner.case
    scen = Scenario.SPARSE_5 if case == Case.NAVIER_STOKES else Scenario.SPARSE_1p5
    ctx = RunContext(case=case, method=method, scenario=scen, seed=0, ensemble_size=4,
                     num_steps=3, extra={"test_index": 1})
    return list(runner._metric_rows(ctx, metrics))


@pytest.mark.parametrize("runner_cls,keys", [
    (ns_driver.NavierStokesRunner, [m.value for m in ns_driver.NS_FIELD_METRICS]),
    (urban_driver.UrbanRunner, [m.value for m in urban_driver.URBAN_METRICS]),
])
def test_vjp_row_only_for_tmpd(runner_cls, keys):
    case = "navier_stokes" if runner_cls is ns_driver.NavierStokesRunner else "urban"
    case_cfg = OmegaConf.load(EXP / "configs" / "case" / f"{case}.yaml")
    runner = runner_cls(OmegaConf.create({"case": case_cfg}), seeds=[0])
    base = {k: 0.5 for k in keys} | {"nfe": 4.0, "seconds": 1.0}
    plain = _rows(runner, Method.OURS_DM_SDE, base)
    assert Metric.VJP.value not in {r.metric for r in plain}
    assert len(plain) == len(keys) + 2
    tmpd = _rows(runner, Method.TMPD, base | {"vjp": 6.0})
    vjp = [r for r in tmpd if r.metric == Metric.VJP.value]
    assert len(tmpd) == len(keys) + 3 and len(vjp) == 1 and vjp[0].value == 6.0
    assert vjp[0].method == "TMPD" and vjp[0].variant is None
    assert runner._variant(Method.TMPD) is None
