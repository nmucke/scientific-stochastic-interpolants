"""Generative baselines on Lorenz 96 (FlowDAS, SDA, D-Flow, FIG) on the tiny case."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cases.lorenz96._l96_baselines as baselines  # noqa: E402
import cases.lorenz96._l96_generative as generative  # noqa: E402
import cases.navier_stokes._ns_pipeline as ns_pipeline  # noqa: E402
from cases.lorenz96 import driver  # noqa: E402
from cases.lorenz96.driver import load_sequences  # noqa: E402
from results_schema import Metric  # noqa: E402
from test_lorenz96_case import tiny_case  # noqa: E402,F401  (fixture)

CELLS = sorted(baselines.BASELINES)
STEPS = 3
ODE_STEPS = 2


@pytest.fixture
def small(tiny_case):  # noqa: F811
    """The tiny case with the baselines at a few steps."""
    for name in CELLS:
        tiny_case.methods[name].num_steps = STEPS
    tiny_case.methods.dflow.ode_steps = ODE_STEPS
    return tiny_case


def run(cfg, name, count=2, settings=None):
    problem = driver.problem_of(cfg)
    sequences = load_sequences(problem, "test", count)
    return driver.run_method(cfg, name, problem, sequences, "test", 0, settings)


def test_cells_are_registered_with_their_labels():
    for name, method in baselines.BASELINES.items():
        assert driver.CELLS[name] == (method, None)
        assert name in driver.TUNABLE
        assert name not in driver.GENERATIVE


def test_config_blocks_hold_every_knob_and_m50():
    cfg = driver.case_config()
    for name in CELLS:
        steps, knobs = baselines.cell_settings(name, driver.method_settings(cfg, name))
        assert steps == driver.FIXED_NUM_STEPS
        assert set(knobs) == set(baselines.KNOBS[name])


@pytest.mark.parametrize("name", CELLS)
def test_cell_runs_and_writes_the_generative_metrics(small, name):
    records = driver.evaluate_cell(name, 0, small, allow_other_m=True)
    reference = driver.evaluate_cell("fm_ode", 0, small, allow_other_m=True)
    assert {r.metric for r in records} == {r.metric for r in reference}
    expected = {
        Metric.MEAN_ERROR_VS_PF.value,
        Metric.VAR_ERROR_VS_PF.value,
        Metric.SLICED_W2.value,
        Metric.RMSE.value,
        Metric.SPREAD.value,
        Metric.CRPS.value,
        Metric.NFE.value,
        Metric.SECONDS.value,
        driver.INNOVATION_LAG1,
        driver.INNOVATION_MSQ,
        driver.DIVERGED_STEPS,
    }
    assert expected <= {r.metric for r in records}
    method, variant = driver.CELLS[name]
    assert all(r.method == method.value and r.variant == variant for r in records)
    assert all(r.M == STEPS and r.E == small.evaluation.ensemble_size for r in records)
    assert driver.cell_file("per_step", name, 0, small).exists()


@pytest.mark.parametrize("name", CELLS)
def test_cell_is_deterministic(small, name):
    a = run(small, name)
    b = run(small, name)
    assert a.ensembles.shape == (2, small.problem.num_steps, 6, small.problem.dim_state)
    torch.testing.assert_close(a.ensembles, b.ensembles, rtol=0, atol=0, equal_nan=True)
    assert a.nfe == b.nfe


# What runs per assimilation step at M steps (NFE_MEANING), counted on the network.
EXPECTED_NFE = {
    "flowdas": 1 + 2 * (STEPS - 1),
    "sda": 1 + 2 * (STEPS - 1),
    "fig": STEPS,
    "dflow": STEPS * 4 * ODE_STEPS + 2 * ODE_STEPS,
}


@pytest.mark.parametrize("name", CELLS)
def test_nfe_counts_the_calls_that_run(small, name):
    assert run(small, name).nfe == EXPECTED_NFE[name]


@pytest.mark.parametrize(
    "name, variable",
    [
        ("flowdas", "FLOWDAS_ZETA"),
        ("flowdas", "FLOWDAS_MC_CHUNK"),
        ("sda", "SDA_GAMMA"),
        ("dflow", "DFLOW_STEP_SIZE"),
        ("fig", "FIG_C"),
        ("fig", "FIG_ANYTHING_NEW"),
    ],
)
def test_env_override_is_refused(small, monkeypatch, name, variable):
    monkeypatch.setenv(variable, "")
    with pytest.raises(RuntimeError, match=variable):
        run(small, name)


@pytest.mark.parametrize("name", CELLS)
def test_build_posterior_gets_only_scalars_from_the_case(small, monkeypatch, name):
    seen = {}
    real = ns_pipeline.build_posterior

    def spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(ns_pipeline, "build_posterior", spy)
    run(small, name, count=1)
    method_cfg = seen["method_cfg"]
    assert isinstance(method_cfg, DictConfig)

    def leaves(node):
        if isinstance(node, dict):
            for value in node.values():
                yield from leaves(value)
        else:
            yield node

    plain = OmegaConf.to_container(method_cfg)
    assert all(
        value is None or isinstance(value, (bool, int, float, str)) for value in leaves(plain)
    )
    block = driver.method_settings(small, name)
    for key in baselines.KNOBS[name]:
        assert plain["likelihood_model"][key] == block[key]
    assert seen["num_steps"] == STEPS
    assert seen["method_name"] == baselines.BASELINES[name].value
    assert seen["likelihood_ensemble_size"] == small.evaluation.ensemble_size


@pytest.mark.parametrize("name", CELLS)
def test_unknown_and_missing_knobs_are_refused(small, name):
    settings = driver.method_settings(small, name)
    with pytest.raises(ValueError, match="unknown"):
        run(small, name, settings={**settings, "typo": 1})
    knob = next(iter(baselines.KNOBS[name]))
    with pytest.raises(ValueError, match="missing"):
        run(small, name, settings={k: v for k, v in settings.items() if k != knob})


def test_a_null_knob_that_would_fall_back_is_refused(small):
    settings = {**driver.method_settings(small, "sda"), "gamma_sda": None}
    with pytest.raises(ValueError, match="must be set"):
        run(small, "sda", settings=settings)


@pytest.mark.parametrize("name", CELLS)
def test_observations_come_from_the_observation_model(small, monkeypatch, name):
    seen = []
    real_model = generative.observation_model
    real_assimilate = generative.assimilate

    def model(problem, stats):
        observed = real_model(problem, stats)
        seen.append(observed)
        return observed

    def assimilate(*args, **kwargs):
        assert kwargs["obs_to_model"] is seen[-1].to_model
        return real_assimilate(*args, **kwargs)

    monkeypatch.setattr(generative, "observation_model", model)
    monkeypatch.setattr(generative, "assimilate", assimilate)
    run(small, name, count=1)
    assert len(seen) == 1


def test_fig_runs_each_sequence_on_its_own(small):
    """FIG's global residual norm pools a batch: sequences must not share one."""
    both = run(small, "fig", count=2)
    first = run(small, "fig", count=1)
    torch.testing.assert_close(both.ensembles[:1], first.ensembles, rtol=0, atol=0)


def test_only_the_sda_divergence_error_counts_as_diverged(small, monkeypatch):
    def failing(error):
        def assimilate(*args, **kwargs):
            raise error

        return assimilate

    diverged = FloatingPointError("SDA: Gamma ... the guided SDE has diverged -- ...")
    monkeypatch.setattr(generative, "assimilate", failing(diverged))
    out = run(small, "sda")
    assert torch.isnan(out.ensembles).all() and out.nfe is None
    records = driver.evaluate_cell("sda", 0, small, allow_other_m=True)
    steps = next(r.value for r in records if r.metric == driver.DIVERGED_STEPS)
    assert steps == 2 * small.problem.num_steps
    with pytest.raises(FloatingPointError):  # the same error from another cell
        run(small, "flowdas")
    monkeypatch.setattr(generative, "assimilate", failing(FloatingPointError("other")))
    with pytest.raises(FloatingPointError, match="other"):
        run(small, "sda")
