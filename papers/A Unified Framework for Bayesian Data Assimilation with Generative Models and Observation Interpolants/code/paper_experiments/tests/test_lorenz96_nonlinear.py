"""The Lorenz 96 problem with arctan observations h(x) = arctan(c L x) / c.

Checks the operator and its derivatives, the c -> 0 limit against the linear
problem, the separation of the two problems (config, references, paths, labels),
the importance-sampling oracle and the arctan wiring on a tiny case (that every
cell runs on it is test_lorenz96_case.test_every_cell_runs_...). The filters' own
statistics (linearised proposal, importance oracle) are tested in the library's
tests/test_filters_nonlinear_obs.py.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cases.lorenz96._l96_generative as generative  # noqa: E402
from cases.lorenz96 import driver, pf_gate  # noqa: E402
from cases.lorenz96._l96_generative import (  # noqa: E402
    SamplerSettings,
    Standardisation,
    StandardisedArctanOperator,
    assimilate,
    build_obs_operator,
    build_posterior,
    observation_model,
    physical_obs_operator,
)
from cases.lorenz96._l96_system import (  # noqa: E402
    Observation,
    Problem,
    load_problem,
    make_sequence,
)
from common.paths import REPO_ROOT  # noqa: E402
from results_schema import SCENARIOS, Scenario  # noqa: E402
from scisi.likelihood_models.observation_operators import (  # noqa: E402
    NONLINEARITIES,
    ObservationOperator,
    PointwiseObservationOperator,
)

LINEAR_DICT = {
    "dim_state": 16,
    "forcing": 8.0,
    "dt": 0.01,
    "interval": 0.2,
    "noise_std": 1.0,
    "noise_stages": 1,
    "obs_stride": 4,
    "obs_variance": 4.0,
    "prior_std": 1.0,
    "num_steps": 50,
}
TINY_C = 1e-6  # arctan(c w) / c = w (1 - (c w)^2 / 3 + ...): linear to ~1e-12


def arctan_problem(strength=None, **changes) -> Problem:
    base = load_problem(obs="arctan")
    if strength is not None:
        base = replace(base, observation=Observation("arctan", strength))
    return replace(base, **changes)


# --------------------------------------------------------------------------- #
# Config, problem dictionary, labels and paths
# --------------------------------------------------------------------------- #


def test_linear_problem_dict_is_unchanged():
    assert Problem().to_dict() == LINEAR_DICT
    assert driver.problem_of(driver.case_config()).to_dict() == LINEAR_DICT
    assert driver.problem_of(driver.case_config("linear")).to_dict() == LINEAR_DICT
    assert load_problem() == Problem() == load_problem(obs="linear")


def test_arctan_problem_records_operator_and_strength_from_config():
    cfg = driver.case_config("arctan")
    problem = driver.problem_of(cfg)
    assert problem == load_problem(obs="arctan")
    c = float(cfg.observations.arctan.strength)
    assert problem.to_dict() == {**LINEAR_DICT, "observation": {"operator": "arctan", "strength": c}}
    # c = 1 / sigma_clim with the climatological std of the training data.
    assert 1.0 / c == pytest.approx(3.9866, abs=1e-3)
    assert problem.dynamics().to_dict() == LINEAR_DICT
    assert problem.obs_label == "arctan" and not problem.linear_obs


def test_strength_of_the_config_is_the_inverse_climatological_std():
    path = REPO_ROOT / "data/lorenz96/data.npz"
    if not path.exists():
        pytest.skip("data/lorenz96/data.npz not present")
    with np.load(path) as data:
        train = data["states"][: int(data["num_train"])].reshape(-1, 16).astype(np.float64)
    sigma = float(np.sqrt(train.var(0).mean()))
    assert load_problem(obs="arctan").observation.strength == pytest.approx(1.0 / sigma, rel=1e-6)


def test_observation_spec_is_validated():
    with pytest.raises(ValueError, match="strength"):
        Observation("arctan")
    with pytest.raises(ValueError, match="no strength"):
        Observation("linear", 0.5)
    with pytest.raises(ValueError, match="one of"):
        Observation("square", 1.0)
    with pytest.raises(ValueError, match="one of"):
        driver.case_config("square")


def test_each_operator_reads_its_own_methods_block():
    linear, arctan = driver.case_config(), driver.case_config("arctan")
    assert set(OmegaConf.to_container(linear.methods)) == set(
        OmegaConf.to_container(arctan.methods_arctan)
    )
    # Same keys, read from the right block: change one and only that cell moves.
    arctan.methods.enkf.inflation = 1.3
    assert driver.method_settings(arctan, "enkf")["inflation"] == 1.3
    assert driver.method_settings(linear, "enkf")["inflation"] == 1.0
    # A corrected cell inherits its parent's sampler settings from its own block.
    # (Set here, not read from the YAML: tuning may give both blocks the same mode.)
    arctan.methods.si_sde.likelihood_mode = "inflated_shared"
    assert driver.method_settings(arctan, "si_sde_fc")["likelihood_mode"] == "inflated_shared"
    assert driver.method_settings(linear, "si_sde_fc")["likelihood_mode"] == (
        linear.methods.si_sde.likelihood_mode
    )
    assert linear.methods.si_sde.likelihood_mode != "inflated_shared"
    with pytest.raises(ValueError, match="already set up"):
        driver.configure(linear, "arctan")


def test_paths_and_scenario_carry_the_operator(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "RESULTS", tmp_path)
    assert driver.reference_path("test", 0, 1, "arctan") == (
        tmp_path / "arctan" / "reference" / "test" / "sequence0_replicate1.pt"
    )
    assert driver.reference_path("test", 0, 1).parent.parent.parent == tmp_path / "linear"
    assert driver.cell_path("metrics", "enkf", 0, obs="arctan").parent == tmp_path / "arctan" / "metrics"
    assert driver.cell_path("metrics", "enkf", 0).parent == tmp_path / "linear" / "metrics"
    assert driver.SCENARIOS["arctan"] is Scenario.LORENZ96_ARCTAN
    assert Scenario.LORENZ96_ARCTAN.value == "lorenz96_arctan"
    # The linear scenario list of the other cases is unchanged.
    assert Scenario.LORENZ96_ARCTAN not in SCENARIOS
    assert pf_gate.gate_dir("arctan").parts[-3:] == ("lorenz96", "arctan", "gate")


def test_obs_option_parses_before_the_subcommand():
    args = driver.build_parser().parse_args(["--obs", "arctan", "cell", "--method", "enkf", "--seed", "0"])
    assert (args.obs, args.command, args.method, args.seed) == ("arctan", "cell", "enkf", 0)
    assert driver.build_parser().parse_args(["tune", "--method", "enkf", "--grid", "{}"]).obs == "linear"
    with pytest.raises(SystemExit):
        driver.build_parser().parse_args(["--obs", "cubic", "cell", "--method", "enkf", "--seed", "0"])
    gate = pf_gate.build_parser().parse_args(["--obs", "arctan", "converge"])
    assert gate.obs == "arctan" and gate.out is None


def test_runner_maps_the_arctan_scenario():
    cfg = OmegaConf.create(
        dict(
            case=OmegaConf.to_container(OmegaConf.load(driver.CASE_CONFIG)),
            scenario=dict(name="lorenz96_arctan"),
            method=dict(name="EnKF"),
        )
    )
    runner = driver.Lorenz96Runner(cfg)
    assert runner.obs == "arctan" and runner.scenarios() == [Scenario.LORENZ96_ARCTAN]


def test_standardisation_accepts_the_arctan_problem_and_refuses_other_dynamics(tmp_path):
    path = tmp_path / "data.npz"
    np.savez(path, mean=2.2, std=4.0, problem=json.dumps(LINEAR_DICT), states=np.zeros(1))
    assert Standardisation.from_data(path, arctan_problem()) == Standardisation(2.2, 4.0)
    assert Standardisation.from_data(path, Problem()) == Standardisation(2.2, 4.0)
    with pytest.raises(ValueError, match="different problem"):
        Standardisation.from_data(path, arctan_problem(forcing=6.0))


# --------------------------------------------------------------------------- #
# Sequences
# --------------------------------------------------------------------------- #


def test_arctan_sequences_share_truth_and_noise_with_the_linear_ones():
    linear, arctan = Problem(num_steps=6), arctan_problem(num_steps=6)
    a, b = make_sequence(linear, 11), make_sequence(arctan, 11)
    assert torch.equal(a.truth, b.truth) and torch.equal(a.prior_mean, b.prior_mean)
    noise = a.observations - a.truth[1:, linear.obs_indices]
    c = arctan.observation.strength
    torch.testing.assert_close(
        b.observations, torch.atan(c * a.truth[1:, linear.obs_indices]) / c + noise,
        rtol=0, atol=1e-12,
    )
    tiny = make_sequence(arctan_problem(TINY_C, num_steps=6), 11)
    torch.testing.assert_close(tiny.observations, a.observations, rtol=0, atol=1e-9)


# --------------------------------------------------------------------------- #
# The operator
# --------------------------------------------------------------------------- #


def test_problem_nonlinearity_is_the_library_arctan():
    problem = arctan_problem()
    c = problem.observation.strength
    w = torch.linspace(-15, 15, 31, dtype=torch.float64)
    g, dg = problem.obs_nonlinearity(w)
    lib_g, lib_dg = NONLINEARITIES["arctan"](w, c, 0.0)
    torch.testing.assert_close(g, lib_g, rtol=0, atol=0)
    torch.testing.assert_close(dg, lib_dg, rtol=0, atol=0)


def test_standardised_operator_is_arctan_of_physical_states_with_exact_derivatives():
    problem = arctan_problem()
    stats = Standardisation(2.23, 3.99)
    observed = observation_model(problem, stats)
    op = observed.operator
    assert isinstance(op, PointwiseObservationOperator) and not op.is_linear
    assert observed.variance == problem.obs_variance
    y = torch.randn(3, 4, dtype=torch.float64)
    assert observed.to_model(y) is y
    gen = torch.Generator().manual_seed(0)
    x = 2.0 + 5.0 * torch.randn(5, 16, generator=gen, dtype=torch.float64)
    z = stats.forward(x).reshape(5, 1, 1, 16)
    torch.testing.assert_close(op(z), problem.observe(x), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        physical_obs_operator(problem)(x.reshape(5, 1, 1, 16)), problem.observe(x),
        rtol=0, atol=0,
    )

    # Per-member Jacobian by autograd of the forward map.
    jac = torch.stack(
        [torch.autograd.functional.jacobian(op, z[b : b + 1])[0, :, 0].reshape(4, 16) for b in range(5)]
    )
    v = torch.randn(5, 1, 1, 16, generator=gen, dtype=torch.float64)
    u = torch.randn(5, 4, generator=gen, dtype=torch.float64)
    torch.testing.assert_close(op.jvp(z, v), torch.einsum("bij,bj->bi", jac, v.reshape(5, 16)))
    torch.testing.assert_close(op.vjp(z, u).reshape(5, 16), torch.einsum("bij,bi->bj", jac, u))
    torch.testing.assert_close(op.gram(z), jac @ jac.transpose(1, 2))
    torch.testing.assert_close(op.jacobian(z[0]), jac[0])
    torch.testing.assert_close(op.linearize(z[0:1]).obs_matrix, jac[0])
    # The base-class (torch.func) derivatives agree with the analytic ones.
    torch.testing.assert_close(ObservationOperator.jvp(op, z, v), op.jvp(z, v))
    torch.testing.assert_close(ObservationOperator.vjp(op, z, u), op.vjp(z, u))
    # The sensor map that places observations is the linear selection.
    assert op.inner.is_linear and torch.equal(op.inner.obs_indices, problem.obs_indices)


def test_unit_standardisation_is_the_library_pointwise_arctan():
    problem = arctan_problem()
    c = problem.observation.strength
    mine = StandardisedArctanOperator(build_obs_operator(problem), c)
    lib = PointwiseObservationOperator(build_obs_operator(problem), "arctan", c)
    x = 5.0 * torch.randn(4, 1, 1, 16)
    torch.testing.assert_close(mine(x), lib(x), rtol=0, atol=0)
    torch.testing.assert_close(mine.gram(x), lib.gram(x), rtol=0, atol=0)


class _ForwardOnlyArctan(ObservationOperator):
    """The same h with only ``forward``: derivatives by torch.func (base class)."""

    def __init__(self, op):
        super().__init__()
        self.op = op
        self.num_obs, self.C, self.H, self.W = op.num_obs, op.C, op.H, op.W

    def forward(self, x):
        return self.op.g(self.op.inner(x))


def _untrained_si():
    model_cfg = OmegaConf.load(REPO_ROOT / "config" / "lorenz96_stochastic_interpolant.yaml").model
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return hydra.utils.instantiate(model_cfg).eval()


def _one_sequence_run(problem, stats, mode, operator=None, members=4, steps=2):
    model = _untrained_si()
    settings = SamplerSettings(num_steps=3, likelihood_mode=mode)
    observed = observation_model(problem, stats)
    if operator is not None:
        observed = replace(observed, operator=operator)
    original = generative.observation_model
    generative.observation_model = lambda p, s: observed
    try:
        posterior, stepper = build_posterior(
            driver.Method.OURS_SI_SDE, model, problem, stats, settings
        )
    finally:
        generative.observation_model = original
    seq = make_sequence(replace(problem, num_steps=steps), 3)
    initial = driver.initial_ensemble(problem, seq, members, "test", 0, 0)[None]
    return assimilate(
        posterior, stepper, stats, initial, seq.observations[None], 3, 7,
        obs_to_model=observed.to_model,
    )


@pytest.mark.parametrize("mode", ["dps_jacobian_free", "inflated"])
def test_likelihood_uses_h_and_its_jacobian(mode):
    """The samplers' likelihood with the analytic operator equals the one with a
    forward-only h (autograd Jacobian), so they use h and J_h, never L alone."""
    torch.set_num_threads(2)
    problem, stats = arctan_problem(0.5), Standardisation(2.23, 3.99)
    op = observation_model(problem, stats).operator
    a = _one_sequence_run(problem, stats, mode)
    b = _one_sequence_run(problem, stats, mode, _ForwardOnlyArctan(op))
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
    # ... and differ from treating the observations as linear.
    lin = _one_sequence_run(problem.dynamics(), stats, mode)
    assert (a - lin).abs().max() > 1e-2


def test_inflated_shared_uses_the_overridden_pointwise_g():
    """inflated_shared calls g / dg / inner directly; with unit standardisation the
    case operator must give the library pointwise arctan's result exactly."""
    torch.set_num_threads(2)
    problem, unit = arctan_problem(0.5), Standardisation(0.0, 1.0)
    c = problem.observation.strength
    lib = PointwiseObservationOperator(build_obs_operator(problem), "arctan", c)
    a = _one_sequence_run(problem, unit, "inflated_shared")
    b = _one_sequence_run(problem, unit, "inflated_shared", lib)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# c -> 0 reproduces the linear problem
# --------------------------------------------------------------------------- #


def _sequences(problem, count=2):
    return driver.load_sequences(problem, "validation", count)


@pytest.fixture
def tiny_pair():
    return Problem(num_steps=4), arctan_problem(TINY_C, num_steps=4)


@pytest.mark.parametrize("radius", [None, 4])
def test_enkf_with_tiny_strength_equals_the_linear_enkf(tiny_pair, radius):
    linear, arctan = tiny_pair
    settings = {"inflation": 1.05, "localisation": radius}
    a = driver.run_enkf(None, "enkf", linear, _sequences(linear), "validation", 0, 12, settings)
    b = driver.run_enkf(None, "enkf", arctan, _sequences(arctan), "validation", 0, 12, settings)
    a, b = a.ensembles, b.ensembles
    torch.testing.assert_close(a, b, rtol=0, atol=1e-6)


def test_pf_with_tiny_strength_equals_the_linear_problem_filter(tiny_pair):
    linear, arctan = tiny_pair
    seq_l, seq_a = _sequences(linear, 1)[0], _sequences(arctan, 1)[0]
    a = pf_gate.run_pf(linear, seq_l, 2000, 5, proposal="linearised")
    b = pf_gate.run_pf(arctan, seq_a, 2000, 5)
    assert pf_gate.reference_proposal(arctan) == "linearised"
    assert pf_gate.reference_proposal(linear) == "optimal"
    torch.testing.assert_close(a.mean, b.mean, rtol=0, atol=1e-4)
    torch.testing.assert_close(a.log_evidence, b.log_evidence, rtol=0, atol=1e-6)
    torch.testing.assert_close(a.ess, b.ess, rtol=1e-6, atol=0)


def test_oracle_with_tiny_strength_equals_the_linear_importance_oracle(tiny_pair):
    linear, arctan = tiny_pair
    seq_l, seq_a = _sequences(linear, 1)[0], _sequences(arctan, 1)[0]
    x0 = driver.initial_ensemble(linear, seq_l, 8, "validation", 0, 0)
    cfg = OmegaConf.create({"oracle_arctan": {"proposals": 256}})
    a = driver.importance_oracle(linear, cfg).run(
        linear.model, x0, seq_l.observations, torch.Generator().manual_seed(1)
    )
    b = driver.run_oracle(cfg, "oracle", arctan, [seq_a], "validation", 0, 8, {})
    gen = torch.Generator().manual_seed(driver.derive_seed(driver.SEED_LABEL, "oracle", "validation", 0, 0))
    c = driver.importance_oracle(linear, cfg).run(linear.model, x0, seq_l.observations, gen)
    torch.testing.assert_close(b.ensembles[0], c.analysis, rtol=0, atol=1e-6)
    torch.testing.assert_close(b.oracle_ess[0], c.ess, rtol=1e-6, atol=0)
    assert not torch.equal(a.analysis, c.analysis)  # other seed, other draws


def test_innovations_with_tiny_strength_equal_the_linear_ones(tiny_pair):
    linear, arctan = tiny_pair
    gen = torch.Generator().manual_seed(0)
    members = 3.0 + 4.0 * torch.randn(20, 16, generator=gen, dtype=torch.float64)
    y = torch.randn(4, generator=gen, dtype=torch.float64)
    a = driver.normalised_innovations(linear, members, y)
    b = driver.normalised_innovations(arctan, members, y)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-8)


def test_arctan_innovations_use_the_exact_predictive_moments():
    """Gauss-Hermite moments against a large Monte Carlo of the forecast mixture."""
    from scisi.dynamics.lorenz96 import integrate

    problem = arctan_problem()
    gen = torch.Generator().manual_seed(1)
    members = 2.0 + 4.0 * torch.randn(10, 16, generator=gen, dtype=torch.float64)
    y = torch.randn(4, generator=gen, dtype=torch.float64)
    z = driver.normalised_innovations(problem, members, y)
    flow = integrate(members, problem.model.steps_per_stage, problem.dt, problem.forcing)
    kicks = torch.randn(10, 200_000, 16, generator=gen, dtype=torch.float64)
    h = problem.observe(flow[:, None] + kicks)  # [E, K, d_y]
    m_i, v_i = h.mean(1), h.var(1)
    mean, var = m_i.mean(0), m_i.var(0) + v_i.mean(0) + problem.obs_variance
    torch.testing.assert_close(z, (y - mean) / var.sqrt(), rtol=0, atol=5e-3)


# --------------------------------------------------------------------------- #
# A tiny arctan case: every cell runs, deterministically, and references refuse
# the other operator
# --------------------------------------------------------------------------- #


@pytest.fixture
def tiny_arctan(tmp_path, monkeypatch):
    from test_lorenz96_case import make_tiny

    return make_tiny("arctan", tmp_path, monkeypatch)


def test_arctan_samplers_take_the_arctan_observation_model(tiny_arctan, monkeypatch):
    seen = []
    real = generative.observation_model

    def spy(problem, stats):
        out = real(problem, stats)
        seen.append(out)
        return out

    monkeypatch.setattr(generative, "observation_model", spy)
    driver.evaluate_cell("si_sde_fc", 0, tiny_arctan, allow_other_m=True)
    assert seen and all(isinstance(o.operator, StandardisedArctanOperator) for o in seen)
    assert all(o.variance == 4.0 for o in seen)


def test_references_refuse_the_other_operator(tiny_arctan, tmp_path):
    arctan = driver.problem_of(tiny_arctan)
    linear = arctan.dynamics()
    assert driver.load_reference("test", 0, 0, arctan)["problem"] == arctan.to_dict()
    # A linear problem looks in linear/; with the arctan file copied there, it refuses it.
    target = driver.reference_path("test", 0, 0, "linear")
    target.parent.mkdir(parents=True)
    shutil.copy(driver.reference_path("test", 0, 0, "arctan"), target)
    with pytest.raises(ValueError, match="Stale"):
        driver.load_reference("test", 0, 0, linear)
    # And the other way round.
    driver._reference_task((linear, driver.load_sequences(linear, "test", 1)[0], "test", 1, 0, 64))
    shutil.copy(
        driver.reference_path("test", 1, 0, "linear"), driver.reference_path("test", 1, 0, "arctan")
    )
    with pytest.raises(ValueError, match="Stale"):
        driver.load_reference("test", 1, 0, arctan)


def test_cell_cli_writes_under_the_arctan_root(tiny_arctan, tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "case_config", lambda obs="linear": tiny_arctan)
    args = argparse.Namespace(
        obs="arctan", method="enkf", seed=0, set=None, tag="", threads=1, allow_other_m=False
    )
    driver.cmd_cell(args)
    assert (tmp_path / "arctan" / "metrics" / "lorenz96__enkf__traj0__.csv").exists()
    assert (tmp_path / "arctan" / "ensembles" / "enkf__seed0.pt").exists()


# --------------------------------------------------------------------------- #
# The gate on the arctan problem (tiny smoke runs)
# --------------------------------------------------------------------------- #


def test_gate_converge_and_cross_proposal_run_on_the_arctan_problem(tmp_path, monkeypatch):
    monkeypatch.setattr(pf_gate, "RESULTS_ROOT", tmp_path)
    common = dict(obs="arctan", problem='{"num_steps": 3}', workers=1, threads=1)
    pf_gate.cmd_converge(
        argparse.Namespace(**common, out=None, ladder="200,400", replicates=2, sequences=1, skip=0)
    )
    report = json.loads((tmp_path / "lorenz96" / "arctan" / "gate" / "convergence.json").read_text())
    assert report["proposal"] == "linearised"
    assert report["problem"]["observation"]["operator"] == "arctan"
    assert "passed" in report
    pf_gate.cmd_cross_proposal(
        argparse.Namespace(**common, out="cross.json", n=300, replicates=2, sequences=1)
    )
    cross = json.loads((tmp_path / "lorenz96" / "arctan" / "gate" / "cross.json").read_text())
    assert cross["proposals"] == ["linearised", "bootstrap"]
    assert math.isfinite(cross["sequences"][0]["t_mean_rms"])


def test_gate_calibrate_adds_the_linear_operator_control(tmp_path, monkeypatch):
    monkeypatch.setattr(pf_gate, "RESULTS_ROOT", tmp_path)
    pf_gate.cmd_calibrate(
        argparse.Namespace(
            obs="arctan", problem='{"num_steps": 3}', workers=1, threads=1,
            out=None, n=200, sequences=2, skip=0,
        )
    )
    report = json.loads((tmp_path / "lorenz96" / "arctan" / "gate" / "calibration.json").read_text())
    assert "wrong_obs_operator_linear" in report and "correct" in report


def test_gate_screen_is_linear_only():
    with pytest.raises(ValueError, match="linear"):
        pf_gate.cmd_screen(argparse.Namespace(obs="arctan"))
