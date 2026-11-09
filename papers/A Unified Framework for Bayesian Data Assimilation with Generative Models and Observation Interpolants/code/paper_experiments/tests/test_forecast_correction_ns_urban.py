"""Forecast-residual correction wired into the Navier-Stokes and urban cases.

CPU smoke tests on tiny synthetic inputs with RANDOM-weight networks: they check
the plumbing (switches, bit-for-bit identity when off, shapes, masks, variant
tags, the urban increment projection), not accuracy. No trained weights or data
are needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cases.navier_stokes import _forecast_correction as fcw  # noqa: E402
from cases.navier_stokes import _ns_pipeline  # noqa: E402
from cases.navier_stokes._ns_pipeline import (  # noqa: E402
    LoadedPrior,
    TruthAndObs,
    attach_nfe_counter,
    build_posterior,
    run_assimilation,
)
from cases.urban import _urban_constraints  # noqa: E402
from cases.urban._urban_constraints import IncrementProjection, derivative  # noqa: E402
from scisi.likelihood_models.observation_operators import (  # noqa: E402
    LinearObservationOperator,
)
from scisi.models.flow_matching_model import FlowMatchingModel  # noqa: E402
from scisi.models.follmer_stochastic_interpolant import (  # noqa: E402
    FollmerStochasticInterpolant,
)
from scisi.models.interpolations import (  # noqa: E402
    LinearDeterministicInterpolation,
    QuadraticStochasticInterpolation,
)
from scisi.posterior_models.forecast_correction import (  # noqa: E402
    ForecastCorrection,
    ForecastCorrectionConfig,
)
from scisi.preprocessing.preprocessor import Preprocesser  # noqa: E402

torch.set_num_threads(2)

EXP = Path(__file__).resolve().parents[1]
METHODS = {
    "Ours (SI-SDE)": "si_sde",
    "Ours (DM-SDE)": "dm_sde",
    "Ours (FM-ODE)": "fm_ode",
}
L = 2  # history length
G = 16  # grid side


# --------------------------------------------------------------------------- #
# Tiny random-weight priors
# --------------------------------------------------------------------------- #


class TinyDrift(nn.Module):
    """Random conv drift ``f(x, x_prev, t)``; stands in for the UNet."""

    def __init__(self, channels: int, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.conv = nn.Conv2d(2 * channels, channels, 3, padding=1, padding_mode="circular")
        with torch.no_grad():
            for p in self.conv.parameters():
                p.copy_(0.2 * torch.randn(p.shape, generator=g))

    def forward(self, x, t, field_history=None, field_cond=None, pars_cond=None):  # type: ignore[no-untyped-def]
        prev = field_history[..., -1]
        tt = t.reshape(t.shape[0], 1, 1, 1)
        return self.conv(torch.cat([x, prev], 1)) * (1 + tt) + (prev - x)


def _prior(channels: int, std=None) -> LoadedPrior:  # type: ignore[no-untyped-def]
    si = FollmerStochasticInterpolant(
        interpolation=QuadraticStochasticInterpolation(gamma_multiplier=1.0, wiener_process=True),
        drift_model=TinyDrift(channels, 0),
    )
    fm = FlowMatchingModel(
        interpolation=LinearDeterministicInterpolation(), drift_model=TinyDrift(channels, 1)
    )
    std = [1.0] * channels if std is None else std
    pre = Preprocesser(
        base={"mean": [0.0] * channels, "std": list(std)},
        target={"mean": [0.0] * channels, "std": list(std)},
    )
    ds = SimpleNamespace(num_channels=channels, height=G, width=G)
    return LoadedPrior(
        si_model=si.eval(), fm_model=fm.eval(), preprocesser=pre, test_dataset=ds,
        len_field_history=L, train_cfg=None, checkpoint_name="tiny",
        has_trained_weights=False,
    )


def _truth(op: LinearObservationOperator, channels: int, T: int, seed: int = 0) -> TruthAndObs:
    g = torch.Generator().manual_seed(seed)
    traj = torch.randn(1, channels, G, G, T, generator=g)
    obs = torch.stack([op(traj[..., t]) for t in range(T)], -1)
    obs = obs + 0.05 * torch.randn(obs.shape, generator=g)
    return TruthAndObs(
        init_base=traj[..., L - 1].clone(), field_history=traj[..., :L].clone(),
        field_cond=None, pars_cond=None, true_trajectory=traj, observations=obs,
        obs_operator=op,
    )


def _method_cfg(name: str):  # type: ignore[no-untyped-def]
    return OmegaConf.load(EXP / "configs" / "method" / f"{METHODS[name]}.yaml")


def _run(method, prior, op, truth, T, correction=None, E=4, M=3, every=None, pass_kw=True, seed=0):  # type: ignore[no-untyped-def]
    model, post, stepper = build_posterior(
        method_name=method, method_cfg=_method_cfg(method), prior=prior,
        obs_operator=op, variance=0.0025, likelihood_ensemble_size=4,
        likelihood_mode="dps_jacobian_free", case_key="navier_stokes",
        scenario_key="sparse_5", num_steps=M,
    )
    nfe = attach_nfe_counter(model)
    if correction is not None and not isinstance(correction, ForecastCorrection):
        correction = correction(post, stepper)
    kw = {"forecast_correction": correction} if pass_kw else {}
    torch.manual_seed(seed)
    res = run_assimilation(
        posterior=post, model=model, truth_obs=truth, ensemble_size=E, num_steps=M,
        num_physical_steps=T, stepper=stepper, nfe_counter=nfe,
        gaussian_base=method != "Ours (SI-SDE)", assimilate_every=every, **kw,
    )
    return res, post


def _ns_ops():  # type: ignore[no-untyped-def]
    return {
        "superres": LinearObservationOperator(type="super_res", data_size=(1, G, G), low_res=4, high_res=G),
        "sparse": LinearObservationOperator(type="random", data_size=(1, G, G), percent_obs=0.1, seed=3),
    }


def _settings(**kw):  # type: ignore[no-untyped-def]
    s = dict(fcw.DEFAULTS)
    s.update({"enabled": True, "localisation_radius": 3.0})
    s.update(kw)
    return s


def _ns_correction(settings, scen="sparse_5"):  # type: ignore[no-untyped-def]
    def make(post, stepper):  # type: ignore[no-untyped-def]
        loc = fcw.ns_localisation(settings, (1, G, G), scen)
        corr, mon = fcw.build_correction(
            settings, posterior=post, num_steps=3, stepper=stepper, localisation=loc,
            steppers=_ns_pipeline._STEPPERS, step_diagnostics=fcw.ns_step_diagnostics,
            constraint=fcw.zero_mean_increment,
        )
        make.monitor = mon  # type: ignore[attr-defined]
        make.correction = corr  # type: ignore[attr-defined]
        return corr
    return make


# --------------------------------------------------------------------------- #
# Config switches and variant tags
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", ["navier_stokes", "urban"])
def test_case_config_defaults_off_and_known_keys(case):
    cfg = OmegaConf.load(EXP / "configs" / "case" / f"{case}.yaml")
    s = fcw.fc_settings(cfg)
    assert s["enabled"] is False
    urban_only = {
        "periodic", "channel_taper_offdiag", "project_divergence", "mask_obstacles",
        "u_channel", "v_channel", "u_axis", "v_axis", "divergence_stencil",
        "cg_tol", "cg_max_iter", "zero_mean_increment",
    }
    unknown = set(cfg.forecast_correction) - set(fcw.DEFAULTS) - urban_only
    assert not unknown, unknown
    for m in fcw.DEFAULT_METHODS:
        assert not fcw.applies_to(s, m)
    assert fcw.variant_with_correction("jacfree", s, "Ours (SI-SDE)", "sparse_5") == "jacfree"
    assert fcw.variant_with_correction(None, s, "FlowDAS", "sparse_5") is None


def test_variant_tags_are_distinct():
    on = _settings(localisation_radius={"default": 8, "superres_16": 12})
    tag = fcw.variant_with_correction("jacfree", on, "Ours (SI-SDE)", "sparse_5")
    assert tag == "jacfree_fc_r8_g1_nM_qon_cap3"
    assert fcw.variant_with_correction("jacfree", on, "Ours (SI-SDE)", "superres_16").startswith(
        "jacfree_fc_r12_"
    )
    assert fcw.variant_with_correction(None, on, "Ours (FM-ODE)", "sparse_5").startswith("fc_")
    # Not in `methods` -> untouched.
    assert fcw.variant_with_correction(None, on, "FlowDAS", "sparse_5") is None
    meas = _settings(mode="measure", forecast_num_steps=10)
    assert fcw.variant_with_correction("shared_jac10", meas, "Ours (DM-SDE)", "x") == (
        "shared_jac10_fcdiag_n10"
    )
    other = _settings(gain_scale=0.5, update_cap=None, localisation_radius=None, q_source="bank")
    assert fcw.variant_suffix(other, None) == "fc_rglob_g0p5_nM_qbank_capnone"


def test_runner_variant_uses_case_block():
    from cases.navier_stokes.driver import NavierStokesRunner
    from cases.urban.driver import UrbanRunner
    from results_schema import Method, Scenario

    for runner_cls, case in ((NavierStokesRunner, "navier_stokes"), (UrbanRunner, "urban")):
        case_cfg = OmegaConf.load(EXP / "configs" / "case" / f"{case}.yaml")
        cfg = OmegaConf.create({"case": case_cfg, "likelihood_mode": "dps_jacobian_free"})
        r = runner_cls(cfg, seeds=[0])
        assert r._variant(Method.OURS_SI_SDE, Scenario.SPARSE_1p5) == "jacfree"
        cfg.case.forecast_correction.enabled = True
        r = runner_cls(cfg, seeds=[0])
        assert r._variant(Method.OURS_SI_SDE, Scenario.SPARSE_1p5) == "jacfree_fc_r8_g1_nM_qon_cap3"
        assert r._variant(Method.SURGE_SDA, Scenario.SPARSE_1p5) is None


# --------------------------------------------------------------------------- #
# NS: localisation geometry
# --------------------------------------------------------------------------- #


def test_ns_observation_sites_are_block_centres_and_sensor_cells():
    ops = _ns_ops()
    loc = fcw.ns_localisation(_settings(), (1, G, G), "superres_32")
    oc = loc.obs_coords(ops["superres"])  # type: ignore[union-attr]
    centres = torch.arange(1.5, G, 4.0, dtype=torch.float64)
    expect = torch.stack(torch.meshgrid(centres, centres, indexing="ij"), -1).reshape(-1, 2)
    torch.testing.assert_close(oc[:, 1:], expect)
    sp = ops["sparse"]
    oc = loc.obs_coords(sp)  # type: ignore[union-attr]
    idx = sp.obs_indices.reshape(-1)
    torch.testing.assert_close(oc[:, 1], (idx // G).double())
    torch.testing.assert_close(oc[:, 2], (idx % G).double())
    blk = loc.state_obs(sp, slice(0, G * G), torch.zeros(1))  # type: ignore[union-attr]
    assert blk.shape == (G * G, sp.num_obs)
    # Taper is 1 at the sensor cell itself and periodic (wraps around the edge).
    assert torch.allclose(blk[idx, torch.arange(sp.num_obs)], torch.ones(sp.num_obs))
    assert fcw.ns_localisation(_settings(localisation_radius=None), (1, G, G), "x") is None


# --------------------------------------------------------------------------- #
# NS: pipeline smoke
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("scen", ["superres", "sparse"])
def test_ns_off_is_bit_for_bit(method, scen):
    T = L + 3
    prior = _prior(1)
    op = _ns_ops()[scen]
    truth = _truth(op, 1, T)
    ref, _ = _run(method, prior, op, truth, T, pass_kw=False)
    none, _ = _run(method, prior, op, truth, T, correction=None)
    off, _ = _run(method, prior, op, truth, T,
                  correction=ForecastCorrection(ForecastCorrectionConfig(enabled=False)))
    assert torch.equal(ref.posterior_trajectory, none.posterior_trajectory)
    assert torch.equal(ref.posterior_trajectory, off.posterior_trajectory)
    assert ref.nfe_per_step == off.nfe_per_step


@pytest.mark.parametrize("method", list(METHODS))
def test_ns_measure_mode_leaves_trajectory_unchanged(method):
    T = L + 3
    prior = _prior(1)
    op = _ns_ops()["sparse"]
    truth = _truth(op, 1, T)
    ref, _ = _run(method, prior, op, truth, T)
    make = _ns_correction(_settings(mode="measure"))
    meas, _ = _run(method, prior, op, truth, T, correction=make)
    assert torch.equal(ref.posterior_trajectory, meas.posterior_trajectory)
    assert meas.nfe_per_step > ref.nfe_per_step  # the forecasts are counted
    payload = fcw.summary(make.correction, make.monitor)
    assert len(payload["innovation_msq"]) == T - L
    assert payload["lag1_autocorrelation"] is not None
    assert 0 < payload["q_share_obs_mean"] < 10
    assert all(r["increment_rms"] == 0.0 for r in payload["steps"])


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("scen", ["superres", "sparse"])
def test_ns_correction_runs_with_diagnostics(method, scen, tmp_path):
    T = L + 3
    prior = _prior(1)
    op = _ns_ops()[scen]
    truth = _truth(op, 1, T)
    plain, _ = _run(method, prior, op, truth, T)
    make = _ns_correction(_settings(diagnostics=True, forecast_num_steps=2, forecast_seed=0),
                          "superres_32" if scen == "superres" else "sparse_5")
    corr, _ = _run(method, prior, op, truth, T, correction=make)
    traj = corr.posterior_trajectory
    assert traj.shape == plain.posterior_trajectory.shape == (4, 1, G, G, T)
    assert torch.isfinite(traj).all()
    assert not torch.equal(traj, plain.posterior_trajectory)
    payload = fcw.summary(make.correction, make.monitor)
    steps = payload["steps"]
    assert len(steps) == T - L
    for rec in steps:
        assert len(rec["enstrophy_ratio_bands"]) == len(fcw.NS_BAND_EDGES) - 1
        assert rec["increment_rms"] > 0
        assert rec["increment_domain_mean_abs"] < 1e-6 * rec["increment_rms"] + 1e-7
        assert "q_share_obs" in rec
    path = fcw.write_diagnostics(tmp_path / "d.json", payload)
    assert path.stat().st_size > 0


def test_ns_forecast_steps_are_honoured():
    T = L + 2
    prior = _prior(1)
    op = _ns_ops()["sparse"]
    truth = _truth(op, 1, T)
    base, _ = _run("Ours (FM-ODE)", prior, op, truth, T, M=3)
    nfes = []
    for steps in (1, 4):
        make = _ns_correction(_settings(forecast_num_steps=steps))
        res, _ = _run("Ours (FM-ODE)", prior, op, truth, T, correction=make, M=3)
        nfes.append(res.nfe_per_step - base.nfe_per_step)
    # Online Q: two forecasts per member per assimilated step, `steps` passes each.
    assert nfes[1] - nfes[0] == pytest.approx(2 * 3, rel=1e-6)


# --------------------------------------------------------------------------- #
# Urban: projection
# --------------------------------------------------------------------------- #


def _building_mask() -> torch.Tensor:
    m = torch.ones(G, G, dtype=torch.bool)
    m[3:7, 4:9] = False
    m[10:14, 2:5] = False
    m[9:12, 11:15] = False
    return m


@pytest.mark.parametrize("stencil", ["central", "forward"])
@pytest.mark.parametrize("periodic", [False, True])
def test_derivative_adjoint(stencil, periodic):
    g = torch.Generator().manual_seed(0)
    a = torch.randn(3, G, G, generator=g, dtype=torch.float64)
    b = torch.randn(3, G, G, generator=g, dtype=torch.float64)
    for ax in (-1, -2):
        lhs = (derivative(a, ax, periodic, stencil) * b).sum()
        rhs = (a * derivative(b, ax, periodic, stencil, transpose=True)).sum()
        assert torch.allclose(lhs, rhs)


@pytest.mark.parametrize("stencil", ["central", "forward"])
@pytest.mark.parametrize("periodic", [(False, False), (True, True)])
def test_projection_removes_increment_divergence_and_obstacle_mass(stencil, periodic):
    mask = _building_mask()
    std = [0.45, 0.36, 0.13, 0.18]
    P = IncrementProjection(mask, std, periodic=periodic, stencil=stencil, tol=1e-10)
    g = torch.Generator().manual_seed(1)
    xg = torch.randn(3, 4, G, G, generator=g)
    xn = xg + 0.3 * torch.randn(3, 4, G, G, generator=g)
    xo = P(xn, xg)
    d = P.step_diagnostics(xg, xn, xo)
    assert d["increment_div_rms_raw"] > 1e-2
    assert d["increment_div_rms_out"] < 1e-5 * d["increment_div_rms_raw"]
    assert d["solid_increment_rms_raw"] > 0 and d["solid_increment_rms_out"] == 0.0
    assert abs(d["div_rms_out"] - d["div_rms_guided"]) < 1e-5 * d["div_rms_guided"]
    # Solid cells keep the guided values in every channel.
    assert torch.equal(xo[:, :, ~mask], xg[:, :, ~mask])
    # Idempotent: projecting the projected increment changes nothing.
    xo2 = P(xo, xg)
    assert torch.allclose(xo2, xo, atol=1e-5)
    # Divergence on ALL cells (incl. solid ones) of the projected increment.
    inc = (xo.double() - xg.double()) * torch.tensor(std, dtype=torch.float64)[None, :, None, None]
    div = P.divergence(inc[:, 0], inc[:, 1])
    assert div.abs().max() < 1e-5


def test_projection_from_settings_switches():
    mask = _building_mask()
    s = fcw.fc_settings(OmegaConf.load(EXP / "configs" / "case" / "urban.yaml"))
    P = _urban_constraints.from_settings(s, mask, [1.0] * 4)
    assert P is not None and P.project and P.mask_obstacles
    assert (P.u_axis, P.v_axis) == (-1, -2)
    s2 = dict(s, project_divergence=False, mask_obstacles=False)
    assert _urban_constraints.from_settings(s2, mask, [1.0] * 4) is None


def test_urban_localisation_zero_in_solids():
    mask = _building_mask()
    keep = mask.unsqueeze(0).expand(4, G, G)
    op = LinearObservationOperator(type="random", data_size=(4, G, G), percent_obs=0.05, seed=0)
    # Sensors in fluid only, as the urban pipeline draws them.
    from cases.urban._urban_pipeline import _restrict_sensors_to_fluid

    _restrict_sensors_to_fluid(op, keep, percent_obs=0.05, seed=0)
    s = dict(fcw.fc_settings(OmegaConf.load(EXP / "configs" / "case" / "urban.yaml")),
             enabled=True, localisation_radius=4.0, channel_taper_offdiag=0.5)
    loc = fcw.urban_localisation(s, (4, G, G), "sparse_1p5", mask)
    blk = loc.state_obs(op, slice(0, 4 * G * G), torch.zeros(1))  # type: ignore[union-attr]
    assert blk.shape == (4 * G * G, op.num_obs)
    assert blk[~keep.reshape(-1)].abs().max() == 0
    assert blk[keep.reshape(-1)].abs().max() > 0
    yy = loc.obs_obs(op, torch.zeros(1, dtype=torch.float64))  # type: ignore[union-attr]
    assert torch.linalg.eigvalsh(yy.double()).min() > -1e-8
    # Cross-channel taper 0.5 between a sensor and a different channel at its site.
    c0 = int(op.obs_indices[0]) // (G * G)
    site = int(op.obs_indices[0]) % (G * G)
    other = ((c0 + 1) % 4) * G * G + site
    assert blk[other, 0].item() == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Urban: pipeline smoke (4 channels, buildings, assimilate_every)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["Ours (SI-SDE)", "Ours (DM-SDE)"])
def test_urban_correction_with_projection(method):
    T = L + 4
    std = [0.45, 0.36, 0.13, 0.18]
    prior = _prior(4, std)
    mask = _building_mask()
    keep = mask.unsqueeze(0).expand(4, G, G)
    from cases.urban._urban_pipeline import build_obs_operator

    scen = OmegaConf.create({"obs_operator": {"type": "random", "percent_obs": 0.05}})
    op = build_obs_operator(scen, (4, G, G), mask_seed=0, fluid_mask=keep)
    assert bool(op.obs_indices_on_grid.reshape(-1)[~keep.reshape(-1)].eq(0).all())
    truth = _truth(op, 4, T)
    plain, _ = _run(method, prior, op, truth, T, every=2)
    off, _ = _run(method, prior, op, truth, T, every=2, pass_kw=False)
    assert torch.equal(plain.posterior_trajectory, off.posterior_trajectory)

    s = dict(fcw.fc_settings(OmegaConf.load(EXP / "configs" / "case" / "urban.yaml")),
             enabled=True, localisation_radius=4.0, diagnostics=True, forecast_num_steps=2)

    def make(post, stepper):  # type: ignore[no-untyped-def]
        loc = fcw.urban_localisation(s, (4, G, G), "sparse_1p5", mask)
        P = _urban_constraints.from_settings(s, mask, std)
        corr, mon = fcw.build_correction(
            s, posterior=post, num_steps=3, stepper=stepper, localisation=loc,
            steppers=_ns_pipeline._STEPPERS, constraint=P, step_diagnostics=P.step_diagnostics,
        )
        make.monitor, make.correction = mon, corr  # type: ignore[attr-defined]
        return corr

    res, _ = _run(method, prior, op, truth, T, correction=make, every=2)
    traj = res.posterior_trajectory
    assert traj.shape == (4, 4, G, G, T) and torch.isfinite(traj).all()
    steps = make.monitor.records  # type: ignore[attr-defined]
    # Only assimilated steps are corrected: steps L, L+2 of L..T-1.
    assert [r for r in make.correction.diagnostics.as_dict()["phys_idx"]] == [L, L + 2]  # type: ignore[attr-defined]
    assert len(steps) == 2
    for rec in steps:
        assert rec["increment_rms_raw"] > 0
        assert rec["increment_div_rms_out"] < 1e-5 * max(rec["increment_div_rms_raw"], 1e-12)
        assert rec["solid_increment_rms_out"] == 0.0
        assert abs(rec["div_rms_out"] - rec["div_rms_guided"]) <= 1e-5 * rec["div_rms_guided"]


# --------------------------------------------------------------------------- #
# Driver hooks (the code path run.py takes)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["correct", "measure"])
def test_driver_builds_and_writes_diagnostics(mode, tmp_path):
    from common.runner import RunContext
    from cases.navier_stokes.driver import NavierStokesRunner
    from cases.urban.driver import UrbanRunner
    from results_schema import Case, Method, Scenario

    for runner_cls, case, C, scenario in (
        (NavierStokesRunner, "navier_stokes", 1, Scenario.SPARSE_5),
        (UrbanRunner, "urban", 4, Scenario.SPARSE_1p5),
    ):
        case_cfg = OmegaConf.load(EXP / "configs" / "case" / f"{case}.yaml")
        fc = case_cfg.forecast_correction
        fc.enabled, fc.mode, fc.diagnostics = True, mode, True
        fc.localisation_radius, fc.forecast_num_steps = 4.0, 2
        fc.diagnostics_root = str(tmp_path / case)
        cfg = OmegaConf.create({"case": case_cfg, "likelihood_mode": "dps_jacobian_free"})
        runner = runner_cls(cfg, seeds=[0])
        prior = _prior(C, [0.4] * C)
        mask = _building_mask()
        keep = mask.unsqueeze(0).expand(C, G, G)
        if case == "urban":
            from cases.urban._urban_pipeline import _restrict_sensors_to_fluid

            op = LinearObservationOperator(type="random", data_size=(C, G, G), percent_obs=0.05, seed=0)
            _restrict_sensors_to_fluid(op, keep, percent_obs=0.05, seed=0)
        else:
            op = _ns_ops()["sparse"]
        T = L + 2
        truth = _truth(op, C, T)
        ctx = RunContext(
            case=Case(case), method=Method.OURS_DM_SDE, scenario=scenario, seed=0,
            ensemble_size=4, num_steps=3, extra={"test_index": 7},
        )
        holder = {}

        def make(post, stepper):  # type: ignore[no-untyped-def]
            args = (ctx, post, stepper, (C, G, G), prior)
            if case == "urban":
                args = args + (keep,)
            corr, mon = runner._build_forecast_correction(*args)
            holder.update(corr=corr, mon=mon)
            return corr

        plain, _ = _run("Ours (DM-SDE)", prior, op, truth, T)
        res, _ = _run("Ours (DM-SDE)", prior, op, truth, T, correction=make)
        if mode == "measure":
            assert torch.equal(res.posterior_trajectory, plain.posterior_trajectory)
        else:
            assert not torch.equal(res.posterior_trajectory, plain.posterior_trajectory)
        runner._write_fc_diagnostics(ctx, holder["corr"], holder["mon"])
        files = list((tmp_path / case).glob("*.json"))
        assert len(files) == 1
        name = files[0].name
        assert ("fcdiag" in name) == (mode == "measure")
        files[0].unlink()


@pytest.mark.parametrize("case", ["navier_stokes", "urban"])
def test_hydra_cli_overrides_reach_the_settings(case):
    from hydra import compose, initialize_config_dir

    import run  # noqa: F401  (registers the experiment_root resolver)

    with initialize_config_dir(config_dir=str(EXP / "configs"), version_base=None):
        cfg = compose(
            config_name="benchmark",
            overrides=[
                f"case={case}",
                "case.forecast_correction.enabled=true",
                "case.forecast_correction.forecast_num_steps=10",
                "case.forecast_correction.localisation_radius=null",
                "case.forecast_correction.bank_indices=[0,7]",
            ],
        )
    s = fcw.fc_settings(cfg.case)
    assert s["enabled"] and s["forecast_num_steps"] == 10
    assert s["localisation_radius"] is None and s["bank_indices"] == [0, 7]
    assert fcw.variant_suffix(s, "sparse_5") == "fc_rglob_g1_n10_qon_cap3"
