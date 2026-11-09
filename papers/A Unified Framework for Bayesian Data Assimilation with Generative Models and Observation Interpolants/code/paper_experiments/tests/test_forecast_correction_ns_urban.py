"""Forecast-residual correction wired into the Navier-Stokes and urban cases.

CPU smoke tests on tiny synthetic inputs with RANDOM-weight networks: they check
the plumbing (switches, bit-for-bit identity when off or at gain 0, shapes, masks,
variant tags), not accuracy. No trained weights or data are needed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cases.navier_stokes import _ns_correction  # noqa: E402
from cases.navier_stokes._ns_pipeline import (  # noqa: E402
    LoadedPrior,
    TruthAndObs,
    attach_nfe_counter,
    build_posterior,
    run_assimilation,
)
from common import forecast_correction as fcw  # noqa: E402
from common.runner import RunContext  # noqa: E402
from results_schema import Case, Method, Scenario  # noqa: E402
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
METHOD_ENUM = {
    "Ours (SI-SDE)": Method.OURS_SI_SDE,
    "Ours (DM-SDE)": Method.OURS_DM_SDE,
    "Ours (FM-ODE)": Method.OURS_FM_ODE,
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
    kw = {"forecast_correction": correction} if pass_kw else {}
    torch.manual_seed(seed)
    res = run_assimilation(
        posterior=post, model=model, truth_obs=truth, ensemble_size=E, num_steps=M,
        num_physical_steps=T, stepper=stepper, nfe_counter=nfe,
        gaussian_base=method != "Ours (SI-SDE)", assimilate_every=every, **kw,
    )
    return res


def _ns_ops():  # type: ignore[no-untyped-def]
    return {
        "superres": LinearObservationOperator(type="super_res", data_size=(1, G, G), low_res=4, high_res=G),
        "sparse": LinearObservationOperator(type="random", data_size=(1, G, G), percent_obs=0.1, seed=3),
    }


def _yaml(case: str) -> dict:
    return fcw.fc_settings(OmegaConf.load(EXP / "configs" / "case" / f"{case}.yaml"))


def _settings(case: str = "navier_stokes", **kw):  # type: ignore[no-untyped-def]
    return {**_yaml(case), "enabled": True, "localisation_radius": 3.0, **kw}


def _ctx(method="Ours (DM-SDE)", case=Case.NAVIER_STOKES, scenario=Scenario.SPARSE_5, test_index=7):  # type: ignore[no-untyped-def]
    return RunContext(case=case, method=METHOD_ENUM[method], scenario=scenario, seed=0,
                      ensemble_size=4, num_steps=3, extra={"test_index": test_index})


def _ns_fc(settings, scen="sparse_5", method="Ours (DM-SDE)"):  # type: ignore[no-untyped-def]
    return fcw.build_correction(
        settings, case="navier_stokes", ctx=_ctx(method), state_shape=(1, G, G),
        scenario_key=scen, periodic=True, constraint=_ns_correction.zero_mean_increment,
        step_diagnostics=_ns_correction.ns_step_diagnostics,
    )


def _building_mask() -> torch.Tensor:
    m = torch.ones(G, G, dtype=torch.bool)
    m[3:7, 4:9] = False
    m[10:14, 2:5] = False
    m[9:12, 11:15] = False
    return m


# --------------------------------------------------------------------------- #
# Config switches and variant tags
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", ["navier_stokes", "urban"])
def test_case_config_defaults_off(case):
    s = _yaml(case)
    assert s["enabled"] is False
    assert s["update_cap"] is None  # the cap is opt-in
    assert s["q_source"] == "online" and s["forecast_seed"] is None
    for m in s["methods"]:
        assert not fcw.applies_to(s, m)
    assert fcw.variant_with_correction("jacfree", s, "Ours (SI-SDE)") == "jacfree"
    assert fcw.variant_with_correction(None, s, "FlowDAS") is None
    assert fcw.fc_settings(OmegaConf.create({})) == {}  # no block -> off


def test_tag_changes_with_every_numeric_setting():
    base = _settings()
    tags = {fcw.fc_tag(base)}
    for key, value in [("gain_scale", 0.5), ("update_cap", 3.0), ("localisation_radius", None),
                       ("q_source", "bank"), ("residual_bank", "b.pt"), ("forecast_stepper", "sde"),
                       ("forecast_seed", 0), ("zero_mean_increment", False),
                       ("forecast_num_steps", 10), ("localisation_radius", {"default": 8})]:
        tags.add(fcw.fc_tag(dict(base, **{key: value})))
    assert len(tags) == 11
    # Settings that never change the numbers do not change the tag; 1 == 1.0.
    same = dict(base, diagnostics=True, diagnostics_root="/x", methods=["Ours (FM-ODE)"], gain_scale=1)
    assert fcw.fc_tag(same) == fcw.fc_tag(base)
    assert fcw.fc_tag(dict(base, forecast_num_steps=10)).startswith("fc10_")
    assert fcw.fc_tag(base).startswith("fcM_")


def test_tag_is_file_name_safe_for_mappings():
    s = _settings(localisation_radius={"default": 8, "superres_16": 12})
    tag = fcw.variant_with_correction("jacfree", s, "Ours (SI-SDE)")
    import re

    assert re.fullmatch(r"jacfree_fcM_[0-9a-f]{8}", tag), tag
    # Key order of the mapping does not matter.
    s2 = _settings(localisation_radius={"superres_16": 12, "default": 8})
    assert fcw.variant_with_correction("jacfree", s2, "Ours (SI-SDE)") == tag
    assert fcw.variant_with_correction(None, s, "Ours (FM-ODE)").startswith("fcM_")
    assert fcw.variant_with_correction(None, s, "FlowDAS") is None  # not in `methods`


def test_runner_variant_uses_case_block():
    from cases.navier_stokes.driver import NavierStokesRunner
    from cases.urban.driver import UrbanRunner

    for runner_cls, case in ((NavierStokesRunner, "navier_stokes"), (UrbanRunner, "urban")):
        case_cfg = OmegaConf.load(EXP / "configs" / "case" / f"{case}.yaml")
        cfg = OmegaConf.create({"case": case_cfg, "likelihood_mode": "dps_jacobian_free"})
        r = runner_cls(cfg, seeds=[0])
        assert r._variant(Method.OURS_SI_SDE, Scenario.SPARSE_1p5) == "jacfree"
        cfg.case.forecast_correction.enabled = True
        r = runner_cls(cfg, seeds=[0])
        expect = "jacfree_" + fcw.fc_tag(fcw.fc_settings(cfg.case))
        assert r._variant(Method.OURS_SI_SDE, Scenario.SPARSE_1p5) == expect
        assert r._variant(Method.SURGE_SDA, Scenario.SPARSE_1p5) is None


@pytest.mark.parametrize("case", ["navier_stokes", "urban"])
def test_hydra_overrides_and_cli_tag(case):
    from hydra import compose, initialize_config_dir

    overrides = [
        "case.forecast_correction.enabled=true",
        "case.forecast_correction.forecast_num_steps=10",
        "case.forecast_correction.localisation_radius={default:8,superres_16:12}",
    ]
    tag = fcw._cli_tag(case, overrides)
    with initialize_config_dir(config_dir=str(EXP / "configs"), version_base=None):
        cfg = compose(config_name="benchmark", overrides=[f"case={case}", *overrides])
    s = fcw.fc_settings(cfg.case)
    assert s["enabled"] and s["forecast_num_steps"] == 10
    assert s["localisation_radius"] == {"default": 8, "superres_16": 12}
    assert tag == fcw.fc_tag(s) and tag.startswith("fc10_")
    # The grid scripts' entry point prints the same tag.
    out = subprocess.run([sys.executable, "-m", "common.forecast_correction", "tag", case, *overrides],
                         cwd=EXP, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == tag


# --------------------------------------------------------------------------- #
# NS: localisation geometry
# --------------------------------------------------------------------------- #


def test_ns_observation_sites_are_block_centres_and_sensor_cells():
    ops = _ns_ops()
    corr, _ = _ns_fc(_settings(), "superres_32")
    loc = corr.config.localisation
    oc = loc.obs_coords(ops["superres"])
    centres = torch.arange(1.5, G, 4.0, dtype=torch.float64)
    expect = torch.stack(torch.meshgrid(centres, centres, indexing="ij"), -1).reshape(-1, 2)
    torch.testing.assert_close(oc, expect)
    sp = ops["sparse"]
    oc = loc.obs_coords(sp)
    idx = sp.obs_indices.reshape(-1)
    torch.testing.assert_close(oc[:, 0], (idx // G).double())
    torch.testing.assert_close(oc[:, 1], (idx % G).double())
    blk = loc.state_obs(sp, slice(0, G * G), torch.zeros(1, dtype=torch.float64))
    assert blk.shape == (G * G, sp.num_obs)
    assert torch.allclose(blk[idx, torch.arange(sp.num_obs)], torch.ones(sp.num_obs, dtype=torch.float64))
    corr, _ = _ns_fc(_settings(localisation_radius=None))
    assert corr.config.localisation is None
    corr, _ = _ns_fc(_settings(localisation_radius={"default": 3, "superres_32": 5}), "superres_32")
    assert corr.config.localisation is not None


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
    ref = _run(method, prior, op, truth, T, pass_kw=False)
    none = _run(method, prior, op, truth, T, correction=None)
    off = _run(method, prior, op, truth, T,
               correction=ForecastCorrection(ForecastCorrectionConfig(enabled=False)))
    assert torch.equal(ref.posterior_trajectory, none.posterior_trajectory)
    assert torch.equal(ref.posterior_trajectory, off.posterior_trajectory)
    assert ref.nfe_per_step == off.nfe_per_step


@pytest.mark.parametrize("method", list(METHODS))
def test_ns_gain_zero_with_seed_measures_without_changing_the_trajectory(method):
    T = L + 3
    prior = _prior(1)
    op = _ns_ops()["sparse"]
    truth = _truth(op, 1, T)
    ref = _run(method, prior, op, truth, T)
    corr, mon = _ns_fc(_settings(gain_scale=0.0, forecast_seed=0, diagnostics=True),
                               method=method)
    meas = _run(method, prior, op, truth, T, correction=corr)
    assert torch.equal(ref.posterior_trajectory, meas.posterior_trajectory)
    assert meas.nfe_per_step > ref.nfe_per_step  # the forecasts are counted
    payload = fcw.summary(corr, mon)
    assert len(payload["innovation_msq"]) == T - L
    assert payload["lag1_autocorrelation"] is not None
    assert 0 < payload["q_share_state_mean"] < 10
    assert all(r["increment_rms"] == 0.0 for r in payload["steps"])


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("scen", ["superres", "sparse"])
def test_ns_correction_runs_with_diagnostics(method, scen, tmp_path):
    T = L + 3
    prior = _prior(1)
    op = _ns_ops()[scen]
    truth = _truth(op, 1, T)
    plain = _run(method, prior, op, truth, T)
    corr, mon = _ns_fc(_settings(diagnostics=True, forecast_num_steps=2, forecast_seed=0),
                               "superres_32" if scen == "superres" else "sparse_5", method)
    res = _run(method, prior, op, truth, T, correction=corr)
    traj = res.posterior_trajectory
    assert traj.shape == plain.posterior_trajectory.shape == (4, 1, G, G, T)
    assert torch.isfinite(traj).all()
    assert not torch.equal(traj, plain.posterior_trajectory)
    payload = fcw.summary(corr, mon)
    assert len(payload["steps"]) == T - L
    assert len(payload["q_share_state"]) == T - L
    for rec in payload["steps"]:
        assert len(rec["enstrophy_ratio_bands"]) == len(_ns_correction.NS_BAND_EDGES) - 1
        assert rec["increment_rms"] > 0
        assert rec["increment_domain_mean_abs"] < 1e-6 * rec["increment_rms"] + 1e-7


def test_ns_forecast_steps_are_honoured():
    T = L + 2
    prior = _prior(1)
    op = _ns_ops()["sparse"]
    truth = _truth(op, 1, T)
    base = _run("Ours (FM-ODE)", prior, op, truth, T, M=3)
    nfes = []
    for steps in (1, 4):
        corr, _ = _ns_fc(_settings(forecast_num_steps=steps), method="Ours (FM-ODE)")
        res = _run("Ours (FM-ODE)", prior, op, truth, T, correction=corr, M=3)
        nfes.append(res.nfe_per_step - base.nfe_per_step)
    # Online Q: two forecasts per member per assimilated step, `steps` passes each.
    assert nfes[1] - nfes[0] == pytest.approx(2 * 3, rel=1e-6)


def test_forecast_seed_depends_on_the_cell():
    s = _settings(forecast_seed=0)
    seeds = set()
    for ti in (1, 2):
        for scen in (Scenario.SPARSE_5, Scenario.SPARSE_1p5):
            corr, _ = fcw.build_correction(s, case="navier_stokes", ctx=_ctx(scenario=scen, test_index=ti),
                                           state_shape=(1, G, G), scenario_key="x", periodic=True)
            seeds.add(corr.config.forecast_seed)
    assert len(seeds) == 4
    corr, _ = _ns_fc(_settings())
    assert corr.config.forecast_seed is None and corr.config.update_cap is None


# --------------------------------------------------------------------------- #
# Urban: masked localisation and pipeline smoke
# --------------------------------------------------------------------------- #


def _urban_op(keep):  # type: ignore[no-untyped-def]
    from cases.urban._urban_pipeline import _restrict_sensors_to_fluid

    op = LinearObservationOperator(type="random", data_size=(4, G, G), percent_obs=0.05, seed=0)
    _restrict_sensors_to_fluid(op, keep, percent_obs=0.05, seed=0)
    return op


def test_urban_localisation_zero_in_solids():
    keep = _building_mask().unsqueeze(0).expand(4, G, G)
    op = _urban_op(keep)
    corr, mon = fcw.build_correction(
        _settings("urban", localisation_radius=4.0), case="urban",
        ctx=_ctx(case=Case.URBAN, scenario=Scenario.SPARSE_1p5), state_shape=(4, G, G),
        scenario_key="sparse_1p5", periodic=(False, False), mask=keep)
    loc = corr.config.localisation
    blk = loc.state_obs(op, slice(0, 4 * G * G), torch.zeros(1, dtype=torch.float64))
    assert blk.shape == (4 * G * G, op.num_obs)
    assert blk[~keep.reshape(-1)].abs().max() == 0
    assert blk[keep.reshape(-1)].abs().max() > 0
    yy = loc.obs_obs(op, torch.zeros(1, dtype=torch.float64))
    assert torch.linalg.eigvalsh(yy.double()).min() > -1e-8
    # Channels at one site share a location (no cross-channel taper).
    c0 = int(op.obs_indices[0]) // (G * G)
    site = int(op.obs_indices[0]) % (G * G)
    assert blk[((c0 + 1) % 4) * G * G + site, 0].item() == pytest.approx(1.0)
    # post_update keeps the guided values in solid cells (also for a global gain).
    xg, xn = torch.randn(3, 4, G, G), torch.randn(3, 4, G, G)
    xo = mon.post_update(xn, xg)
    assert torch.equal(xo[:, ~keep], xg[:, ~keep]) and torch.equal(xo[:, keep], xn[:, keep])


@pytest.mark.parametrize("radius", [4.0, None])
@pytest.mark.parametrize("method", ["Ours (SI-SDE)", "Ours (DM-SDE)"])
def test_urban_correction_keeps_solids(method, radius):
    T = L + 4
    prior = _prior(4, [0.45, 0.36, 0.13, 0.18])
    keep = _building_mask().unsqueeze(0).expand(4, G, G)
    from cases.urban._urban_pipeline import build_obs_operator

    scen = OmegaConf.create({"obs_operator": {"type": "random", "percent_obs": 0.05}})
    op = build_obs_operator(scen, (4, G, G), mask_seed=0, fluid_mask=keep)
    truth = _truth(op, 4, T)
    plain = _run(method, prior, op, truth, T, every=2)
    off = _run(method, prior, op, truth, T, every=2, pass_kw=False)
    assert torch.equal(plain.posterior_trajectory, off.posterior_trajectory)

    corr, _ = fcw.build_correction(
        _settings("urban", localisation_radius=radius, forecast_num_steps=2, forecast_seed=0),
        case="urban", ctx=_ctx(method, Case.URBAN, Scenario.SPARSE_1p5), state_shape=(4, G, G),
        scenario_key="sparse_1p5", periodic=(False, False), mask=keep)
    res = _run(method, prior, op, truth, T, correction=corr, every=2)
    traj = res.posterior_trajectory
    assert traj.shape == (4, 4, G, G, T) and torch.isfinite(traj).all()
    # Only assimilated steps are corrected: steps L, L+2 of L..T-1.
    assert corr.diagnostics.as_dict()["phys_idx"] == [L, L + 2]
    assert not torch.equal(traj, plain.posterior_trajectory)


# --------------------------------------------------------------------------- #
# Driver hooks (the code path run.py takes)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("gain", [1.0, 0.0])
def test_driver_builds_and_writes_diagnostics(gain, tmp_path):
    from cases.navier_stokes.driver import NavierStokesRunner
    from cases.urban.driver import UrbanRunner

    for runner_cls, case, C, scenario in (
        (NavierStokesRunner, "navier_stokes", 1, Scenario.SPARSE_5),
        (UrbanRunner, "urban", 4, Scenario.SPARSE_1p5),
    ):
        case_cfg = OmegaConf.load(EXP / "configs" / "case" / f"{case}.yaml")
        fc = case_cfg.forecast_correction
        fc.enabled, fc.gain_scale, fc.forecast_seed, fc.diagnostics = True, gain, 0, True
        fc.localisation_radius, fc.forecast_num_steps = 4.0, 2
        fc.diagnostics_root = str(tmp_path / case)
        cfg = OmegaConf.create({"case": case_cfg, "likelihood_mode": "dps_jacobian_free"})
        runner = runner_cls(cfg, seeds=[0])
        prior = _prior(C, [0.4] * C)
        keep = _building_mask().unsqueeze(0).expand(C, G, G)
        op = _urban_op(keep) if case == "urban" else _ns_ops()["sparse"]
        T = L + 2
        truth = _truth(op, C, T)
        ctx = RunContext(case=Case(case), method=Method.OURS_DM_SDE, scenario=scenario, seed=0,
                         ensemble_size=4, num_steps=3, extra={"test_index": 7})
        args = (ctx, (C, G, G)) + ((keep,) if case == "urban" else ())
        corr, mon = runner._build_forecast_correction(*args)
        assert runner._build_forecast_correction(
            RunContext(case=Case(case), method=Method.SDA, scenario=scenario, seed=0,
                       ensemble_size=4, num_steps=3), *args[1:]) == (None, None)
        plain = _run("Ours (DM-SDE)", prior, op, truth, T)
        res = _run("Ours (DM-SDE)", prior, op, truth, T, correction=corr)
        same = torch.equal(res.posterior_trajectory, plain.posterior_trajectory)
        assert same == (gain == 0.0)
        path = runner._write_fc_diagnostics(ctx, corr, mon)
        assert path.parent == tmp_path / case
        assert runner._variant(ctx.method) in path.name
        assert fcw.fc_tag(runner._fc_settings()) in path.name
