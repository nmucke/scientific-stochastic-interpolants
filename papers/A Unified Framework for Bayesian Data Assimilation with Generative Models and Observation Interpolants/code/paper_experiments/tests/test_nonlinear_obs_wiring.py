"""Nonlinear observation operators in the NS / urban pipelines and the JAX EnKF h.

No checkpoints / data / jax-cfd needed: the scenario factory is exercised on the
scenario YAMLs, and the EnKF module is imported with its jax-cfd solver deps
stubbed (only the observation-operator port is tested here).
"""
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))
SCEN = EXPERIMENT / "configs" / "scenario"

from cases.navier_stokes import _ns_pipeline  # noqa: E402
from common.obs_scenario import (  # noqa: E402
    scenario_key,
    states_npz_fields,
    states_tag,
    strength_results_path,
)
from common.seeding import mask_seed, obs_seed  # noqa: E402
from scisi.likelihood_models.observation_operators import (  # noqa: E402
    LinearObservationOperator,
    PointwiseObservationOperator,
)
from scisi.likelihood_models.physics_observation_operators import (  # noqa: E402
    VelocityMagnitudeOperator,
)

DATA = (1, 128, 128)


def _load(name):
    return OmegaConf.load(SCEN / f"{name}.yaml")


def test_linear_scenarios_unchanged():
    for name in ("sparse_5", "sparse_1p5", "superres_32", "superres_16"):
        op = _ns_pipeline.build_obs_operator(_load(name), DATA, mask_seed=7)
        assert type(op) is LinearObservationOperator


def test_arctan_scenario_wraps_same_mask():
    lin = _ns_pipeline.build_obs_operator(_load("sparse_5"), DATA, mask_seed=7)
    nl = _ns_pipeline.build_obs_operator(_load("sparse_5_arctan"), DATA, mask_seed=7)
    assert isinstance(nl, PointwiseObservationOperator)
    assert nl.nonlinearity == "arctan" and nl.strength == 1.0
    assert torch.equal(nl.obs_indices, lin.obs_indices)
    assert torch.equal(nl.obs_indices_on_grid, lin.obs_indices_on_grid)
    x = torch.randn(2, *DATA)
    torch.testing.assert_close(nl(x), torch.atan(lin(x)))


def test_strength_override_and_seed_keys():
    cfg = _ns_pipeline.apply_nonlinearity_overrides(_load("sparse_5_arctan"), strength=4)
    op = _ns_pipeline.build_obs_operator(cfg, DATA, mask_seed=7)
    assert op.strength == 4.0
    assert states_tag(op) == "__c4"
    lin_cfg = _ns_pipeline.apply_nonlinearity_overrides(_load("sparse_5"), strength=4)
    assert "nonlinearity" not in lin_cfg
    assert scenario_key(cfg, "seed_scenario", "sparse 5% arctan") == "sparse 5%"
    assert scenario_key(cfg, "hparam_scenario", "sparse_5_arctan") == "sparse_5"
    # The strength override never touches a non-pointwise operator (no c).
    speed = _ns_pipeline.apply_nonlinearity_overrides(_load("sparse_5_speed"), strength=4)
    assert "strength" not in speed.nonlinearity


@pytest.mark.parametrize(
    "case, name, label, mask, obs",
    [
        ("navier_stokes", "sparse_5", "sparse 5%", 931617219, 1910179332),
        ("navier_stokes", "sparse_1p5", "sparse 1.5625%", 1426440050, 853784142),
        ("urban", "sparse_1p5", "sparse 1.5625%", 1281420277, 198563265),
    ],
)
def test_linear_scenarios_keys_pinned(case, name, label, mask, obs):
    """Linear scenarios: seeds, hparam key, states tag + npz fields unchanged."""
    cfg = _ns_pipeline.apply_nonlinearity_overrides(_load(name), strength=2)
    seed_label = scenario_key(cfg, "seed_scenario", label)
    assert seed_label == label
    assert mask_seed(case, seed_label) == mask
    assert obs_seed(case, seed_label, 1, 0) == obs
    assert scenario_key(cfg, "hparam_scenario", name, warn=True) == name
    op = _ns_pipeline.build_obs_operator(cfg, DATA, mask_seed=mask)
    assert type(op) is LinearObservationOperator
    assert states_tag(op) == ""
    f = states_npz_fields(op, hparam_scenario=name)
    assert f["obs_linear"] and f["obs_nonlinearity"] == "" and np.isnan(f["obs_strength"])


def test_ns_saved_states_filename_pinned(tmp_path):
    """The NS writer's filename for a linear scenario is unchanged, and the
    KL-reference glob (same tag helper) finds an arctan EnKF file."""
    from types import SimpleNamespace

    from cases.navier_stokes.driver import NavierStokesRunner
    from common.runner import RunContext
    from results_schema import Case, Method, Scenario

    def save(scenario, op, method=Method.ENKF):
        runner = NavierStokesRunner(
            {"states_root": str(tmp_path), "ns_scenarios": [scenario.value]}, seeds=[0]
        )
        ctx = RunContext(
            case=Case.NAVIER_STOKES, method=method, scenario=scenario, seed=3,
            ensemble_size=8, num_steps=5, extra={"test_index": 1},
        )
        T = 2
        result = SimpleNamespace(
            posterior_trajectory=torch.zeros(2, *DATA[:1], 4, 4, T),
            true_trajectory=torch.zeros(1, *DATA[:1], 4, 4, T),
            nfe_per_step=1.0, seconds_per_step=1.0,
        )
        truth = SimpleNamespace(observations=torch.zeros(1, 3, T))
        runner._save_states(ctx, result, truth, op)
        return runner, ctx

    lin = _ns_pipeline.build_obs_operator(_load("sparse_5"), DATA, mask_seed=7)
    save(Scenario.SPARSE_5, lin)
    assert (tmp_path / "navier_stokes__EnKF__sparse_5__seed3__E8_M5.npz").exists()
    d = np.load(tmp_path / "navier_stokes__EnKF__sparse_5__seed3__E8_M5.npz")
    assert str(d["hparam_scenario"]) == "sparse_5"

    nl = _ns_pipeline.build_obs_operator(_load("sparse_5_arctan"), DATA, mask_seed=7)
    runner, ctx = save(Scenario.SPARSE_5_ARCTAN, nl)
    path = tmp_path / "navier_stokes__EnKF__sparse_5_arctan__c1__seed3__E8_M5.npz"
    assert path.exists()
    assert str(np.load(path)["hparam_scenario"]) == "sparse_5"
    runner.config["kl_reference_states"] = str(tmp_path)
    ref = runner._reference_trajectory(ctx, None, None, nl, None)
    assert ref is not None and tuple(ref.shape) == (2, 1, 4, 4, 2)


def test_strength_results_path_and_aggregation(tmp_path):
    from common.aggregate_lib import aggregate_scalar
    from results_schema import ResultRecord, ResultsWriter

    p = tmp_path / "ns_arctan.csv"
    assert strength_results_path(p, None) == p
    assert strength_results_path(p, 2).name == "ns_arctan__c2.csv"
    assert strength_results_path(tmp_path / "x__c2.csv", 2).name == "x__c2.csv"
    for c in (None, 2):
        with ResultsWriter(strength_results_path(p, c)) as w:
            w.append(ResultRecord("navier_stokes", "EnKF", "sparse 5% arctan", "rmse", 1.0))
    rows = aggregate_scalar({1: sorted(tmp_path.glob("*.csv"))})
    assert sorted(r["scenario"] for r in rows) == ["sparse 5% arctan", "sparse 5% arctan c=2"]


def test_speed_metrics_report_totals_only():
    from cases.navier_stokes._ns_pipeline import AssimResult

    lin = LinearObservationOperator(type="random", data_size=DATA, percent_obs=0.05, seed=1)
    true = torch.randn(1, *DATA, 3)
    post = true + 0.1 * torch.randn(4, *DATA, 3)
    res = AssimResult(posterior_trajectory=post, true_trajectory=true,
                      nfe_per_step=1.0, seconds_per_step=1.0)
    m_lin = _ns_pipeline.compute_metrics(res, lin, len_field_history=1)
    m_spd = _ns_pipeline.compute_metrics(
        res, VelocityMagnitudeOperator(inner=lin), len_field_history=1
    )
    assert np.isfinite(m_lin["crps_observed"]) and np.isfinite(m_lin["crps_unobserved"])
    assert np.isnan(m_spd["crps_observed"]) and np.isnan(m_spd["crps_unobserved"])
    assert m_spd["crps"] == m_lin["crps"]


def test_speed_scenario():
    op = _ns_pipeline.build_obs_operator(_load("sparse_5_speed"), DATA, mask_seed=7)
    assert isinstance(op, VelocityMagnitudeOperator)
    assert op.velocity_scale == 0.25 and not op.is_linear


def test_urban_velocity_channel_sensors():
    from cases.urban import _urban_pipeline

    data = (4, 32, 32)
    fluid = torch.ones(data, dtype=torch.bool)
    fluid[:, 10:14, 10:14] = False
    op = _urban_pipeline.build_obs_operator(
        _load("sparse_1p5_arctan_vel"), data, mask_seed=3, fluid_mask=fluid
    )
    assert isinstance(op, PointwiseObservationOperator)
    chw = op.obs_indices_c_h_w
    assert set(chw[:, 0].tolist()) <= {0, 1, 2}
    assert op.num_obs == int(3 * 32 * 32 * 0.015625)
    assert fluid.reshape(-1)[op.obs_indices].all()
    # Linear control: SAME seed label -> same sensors (and noise draws), h = L.
    ctrl_cfg = _load("sparse_1p5_vel")
    arctan_label = "sparse 1.5625% arctan (uvw)"
    assert scenario_key(ctrl_cfg, "seed_scenario", "sparse 1.5625% (uvw)") == arctan_label
    assert scenario_key(_load("sparse_1p5_arctan_vel"), "seed_scenario", arctan_label) == arctan_label
    ctrl = _urban_pipeline.build_obs_operator(ctrl_cfg, data, mask_seed=3, fluid_mask=fluid)
    assert type(ctrl) is LinearObservationOperator
    assert torch.equal(ctrl.obs_indices, op.obs_indices)
    assert ctrl.num_obs == op.num_obs
    # Linear urban sparse path unchanged: budget over all 4 channels.
    lin = _urban_pipeline.build_obs_operator(_load("sparse_1p5"), data, 3, fluid)
    assert type(lin) is LinearObservationOperator
    assert lin.num_obs == int(4 * 32 * 32 * 0.015625)


# --------------------------------------------------------------------------- #
# JAX port in the EnKF / PF baseline.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def enkf_module():
    pytest.importorskip("jax")
    os.environ.setdefault("ENKF_JAX_PLATFORM", "cpu")
    os.environ["JAX_PLATFORMS"] = "cpu"
    stubs = {
        "jax_cfd_lib": {},
        "jax_cfd_lib.ENKF": {"LocalizedSpectralEnKF": object, "SpectralEnKF": object},
        "jax_cfd_lib.ETKF": {"LocalizedSpectralETKF": object},
        "jax_cfd_lib.navier_stokes_forward_model": {"set_up_forward_model": None},
        "jax_cfd": {},
        "jax_cfd.base": {},
        "jax_cfd.base.grids": {},
    }
    saved = {}
    for name, attrs in stubs.items():
        try:
            __import__(name)
            continue  # real package available: keep it
        except Exception:
            pass
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod
    sys.modules["jax_cfd.base"].grids = sys.modules["jax_cfd.base.grids"]
    sys.modules["jax_cfd"].base = sys.modules["jax_cfd.base"]
    from cases.navier_stokes import enkf_baseline

    return enkf_baseline


def _jax_vs_torch(enkf_module, op, x):
    import jax.numpy as jnp

    jh = enkf_module.TorchHObservationOperator(op)
    y_jax = np.stack([np.asarray(jh._apply_H_on_norm128(jnp.asarray(xi[0].numpy()))) for xi in x])
    y_torch = op(x).numpy()
    return y_jax, y_torch


@pytest.mark.parametrize("name", ["identity", "arctan", "tanh", "saturating", "cubic", "square", "abs", "exp"])
def test_jax_pointwise_matches_torch(enkf_module, name):
    lin = LinearObservationOperator(type="random", data_size=DATA, percent_obs=0.05, seed=1)
    op = PointwiseObservationOperator(inner=lin, nonlinearity=name, strength=2.0)
    x = 0.5 * torch.randn(2, *DATA)
    y_jax, y_torch = _jax_vs_torch(enkf_module, op, x)
    np.testing.assert_allclose(y_jax, y_torch, rtol=1e-5, atol=1e-5)


def test_jax_linear_path_unchanged(enkf_module):
    import jax.numpy as jnp

    lin = LinearObservationOperator(type="random", data_size=DATA, percent_obs=0.05, seed=1)
    jh = enkf_module.TorchHObservationOperator(lin)
    assert jh._pre is None and jh._post is None
    f = jnp.asarray(np.random.default_rng(0).standard_normal((128, 128)).astype(np.float32))
    H = jnp.asarray(lin.obs_matrix.numpy())
    np.testing.assert_array_equal(np.asarray(jh._apply_H_on_norm128(f)), np.asarray(H @ f.reshape(-1)))


def test_jax_speed_matches_torch(enkf_module):
    lin = LinearObservationOperator(type="random", data_size=DATA, percent_obs=0.05, seed=1)
    op = VelocityMagnitudeOperator(inner=lin, velocity_scale=0.25)
    x = torch.randn(2, *DATA)
    y_jax, y_torch = _jax_vs_torch(enkf_module, op, x)
    np.testing.assert_allclose(y_jax, y_torch, rtol=1e-4, atol=1e-4)
