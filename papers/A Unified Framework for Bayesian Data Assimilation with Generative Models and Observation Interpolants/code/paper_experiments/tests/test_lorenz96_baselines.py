"""The NS-built cells on Lorenz 96 (FlowDAS, SDA, D-Flow, FIG and the two SURGE
baselines) on the tiny case. That every cell runs, writes its metrics and is
deterministic is test_lorenz96_case.test_every_cell_runs_..."""

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
from test_lorenz96_case import tiny_case  # noqa: E402,F401  (fixture)

CELLS = sorted(baselines.BASELINE_CELLS)
NS_CELLS = sorted(baselines.NS_CELLS)
STEPS = 3  # test_lorenz96_case.TINY_STEPS
ODE_STEPS = 2


@pytest.fixture
def small(tiny_case):  # noqa: F811
    """The tiny case: the baselines at STEPS pseudo-steps, D-Flow at ODE_STEPS."""
    return tiny_case


def run(cfg, name, count=2, settings=None):
    problem = driver.problem_of(cfg)
    sequences = load_sequences(problem, "test", count)
    return driver.run_method(cfg, name, problem, sequences, "test", 0, settings)


def test_cells_are_registered_with_their_labels():
    for name, (_, method, _, _) in baselines.NS_CELLS.items():
        assert driver.CELLS[name] == (method, None)
        assert name in driver.TUNABLE
        assert name not in driver.GENERATIVE


def test_config_blocks_hold_every_knob_and_m50():
    for obs in ("linear", "arctan"):
        cfg = driver.case_config(obs)
        for name in NS_CELLS:
            steps, knobs = baselines.ns_settings(name, driver.method_settings(cfg, name))
            assert steps == driver.FIXED_NUM_STEPS
            assert {b: set(k) for b, k in knobs.items()} == {
                b: set(k) for b, k in baselines.NS_CELLS[name][3].items()
            }


# What runs per assimilation step at M steps (module docstring), counted on the network.
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
        ("surge_sda", "FLOWDAS_ZETA"),
        ("surge_flowdas", "FLOWDAS_MAX_GRAD_NORM"),
    ],
)
def test_env_override_is_refused(small, monkeypatch, name, variable):
    monkeypatch.setenv(variable, "")
    with pytest.raises(RuntimeError, match=variable):
        run(small, name)


@pytest.mark.parametrize("name", NS_CELLS)
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
    for section, keys in baselines.NS_CELLS[name][3].items():
        for key in keys:
            assert plain[section][key] == block[key]
    assert seen["num_steps"] == STEPS
    assert seen["method_name"] == baselines.NS_CELLS[name][1].value
    assert seen["likelihood_ensemble_size"] == small.evaluation.ensemble_size
    # Only the network the method runs on is loaded.
    network = baselines.NS_CELLS[name][2]
    prior = seen["prior"]
    assert (prior.si_model is not None) == (network == "si")
    assert (prior.diffusion_model is not None) == (network == "diffusion")


@pytest.mark.parametrize("name", NS_CELLS)
def test_unknown_and_missing_knobs_are_refused(small, name):
    settings = driver.method_settings(small, name)
    with pytest.raises(ValueError, match="unknown"):
        run(small, name, settings={**settings, "typo": 1})
    knob = next(iter(next(iter(baselines.NS_CELLS[name][3].values()))))
    with pytest.raises(ValueError, match="missing"):
        run(small, name, settings={k: v for k, v in settings.items() if k != knob})
    with pytest.raises(ValueError, match="must be a number"):
        run(small, name, settings={**settings, knob: {"default": 1.0}})


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


@pytest.mark.parametrize("name", ["fig", "sda"])
def test_fig_and_sda_run_each_sequence_on_its_own(small, name):
    """FIG's global residual norm pools a batch, and SDA's divergence is a batch's:
    sequences must not share one."""
    both = run(small, name, count=2)
    first = run(small, name, count=1)
    torch.testing.assert_close(both.ensembles[:1], first.ensembles, rtol=0, atol=0)


def test_an_sda_divergence_nans_its_own_sequence_only(small, monkeypatch):
    real = generative.assimilate
    calls = []

    def assimilate(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:  # the second sequence
            raise FloatingPointError("SDA: Gamma ... the guided SDE has diverged -- ...")
        return real(*args, **kwargs)

    monkeypatch.setattr(generative, "assimilate", assimilate)
    out = run(small, "sda", count=2)
    assert torch.isfinite(out.ensembles[0]).all() and torch.isnan(out.ensembles[1]).all()
    assert out.nfe is None


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
