"""TMPD baseline wiring: analytical (single + multistep) and Lorenz 96 drivers.

Tiny settings only; the numerics of TMPD itself are covered by tests/test_tmpd.py
at the repository root.
"""
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from cases.analytical.driver import (  # noqa: E402
    ANALYTICAL_METHODS,
    AnalyticalRunner,
)
from cases.analytical.multistep import MultistepRunner  # noqa: E402
from cases.lorenz96 import driver as l96  # noqa: E402
import cases.lorenz96._l96_generative as generative  # noqa: E402
from results_schema import Method, Metric  # noqa: E402
from scisi.likelihood_models.tmpd import TMPDLikelihood  # noqa: E402
from scisi.posterior_models.tmpd_posterior import TMPDPosterior  # noqa: E402

# The Lorenz 96 fixture with untrained priors and a stored tiny reference.
from test_lorenz96_case import tiny_case  # noqa: E402,F401


def _analytical_config(**extra):
    return OmegaConf.create({
        "case": {"plot_dimension": 2, "n_eval_samples": 256, "adapted_variant": False},
        "ensemble_size": 64,
        "num_steps": 20,
        **extra,
    })


def test_tmpd_is_opt_in_on_the_analytical_lineup():
    assert Method.TMPD not in ANALYTICAL_METHODS
    runner = AnalyticalRunner(_analytical_config(), seeds=(0,))
    assert tuple(runner.methods()) == ANALYTICAL_METHODS
    picked = AnalyticalRunner(
        _analytical_config(analytical_methods=["TMPD", "SDA"]), seeds=(0,)
    )
    assert tuple(picked.methods()) == (Method.TMPD, Method.SDA)
    bad = AnalyticalRunner(_analytical_config(analytical_methods=["EnKF", "One-step oracle"]), seeds=(0,))
    with pytest.raises(ValueError, match="Not an analytical method"):
        bad.methods()


def test_analytical_tmpd_row_is_close_to_the_exact_posterior():
    """Linear-Gaussian: TMPD is exact up to discretisation (TMPD Prop. 3)."""
    runner = AnalyticalRunner(
        _analytical_config(analytical_methods=["TMPD"]), seeds=(0,)
    )
    rows = runner.run(aggregate=False)
    kl = [r.value for r in rows if r.metric == Metric.KL_POINTS.value]
    assert {r.method for r in rows} == {"TMPD"}
    assert len(kl) == 1 and kl[0] < 0.1


def test_multistep_tmpd_cycle_runs_on_a_general_system():
    system = dict(
        A=[[0.8, 0.4], [-0.2, 1.0]],
        Q=[[1.0, 0.3], [0.3, 0.5]],
        H=[[1.0, 0.5], [0.0, 1.0]],
        R=[[0.6, 0.2], [0.2, 0.4]],
    )
    cfg = OmegaConf.create({
        "case": {"plot_dimension": 2, "adapted_variant": False, "n_eval_samples": 64,
                 "system": system, "scenario_name": "multistep:test"},
        "ensemble_size": 64,
        "num_steps": 20,
    })
    runner = MultistepRunner(cfg, seeds=(0,))
    x = torch.randn(64, 2)
    torch.manual_seed(0)
    out, nfe = runner._baseline_cycle(
        Method.TMPD, runner.system(), x, torch.zeros(2), 20, torch.Generator(), 1
    )
    assert out.shape == (64, 2) and torch.isfinite(out).all() and nfe == 20


def test_lorenz96_tmpd_cell_is_registered():
    assert l96.CELLS["tmpd"] == (Method.TMPD, None)
    assert "tmpd" not in l96.GENERATIVE  # not a prior for prior_enkf, no _fc cell
    assert l96.Lorenz96Runner._NAMES[Method.TMPD] == "tmpd"
    assert "tmpd" in l96.TUNABLE


@pytest.mark.parametrize("covariance", ["full", "row_sum"])
def test_lorenz96_tmpd_builds_and_runs(tiny_case, covariance, monkeypatch):  # noqa: F811
    built = []
    real = generative.build_tmpd_posterior

    def spy(model, problem, stats, settings):
        posterior = real(model, problem, stats, settings)
        built.append(posterior)
        return posterior

    monkeypatch.setattr(generative, "build_tmpd_posterior", spy)
    problem = l96.problem_of(tiny_case)
    sequences = l96.load_sequences(problem, "test", 2)
    settings = {**l96.method_settings(tiny_case, "tmpd"), "covariance": covariance}
    out = l96.run_method(tiny_case, "tmpd", problem, sequences, "test", 0, settings)
    (posterior,) = built
    assert isinstance(posterior, TMPDPosterior)
    assert isinstance(posterior.likelihood_model, TMPDLikelihood)
    assert posterior.likelihood_model.covariance == covariance
    # Same endpoint-vanishing g as the DM-SDE cell: zero at both ends.
    g = posterior.diffusion_term(torch.tensor([0.0, 0.5, 1.0]))
    assert g[0] == 0 and g[2] == 0 and g[1] > 0
    assert out.ensembles.shape == (2, problem.num_steps, 6, problem.dim_state)
    assert torch.isfinite(out.ensembles).all()
    assert out.nfe >= settings["num_steps"]
