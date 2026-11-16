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
    """WIRING smoke test of the analytical TMPD row (not a numerics test).

    Linear-Gaussian: TMPD is exact up to discretisation (TMPD Prop. 3). At these
    tiny settings (256 samples, M = 20) the Monte-Carlo noise of the KL is too
    large to pin the guidance weight (a_tau alone scores about as well as the
    correct a_tau + 1/2 g^2); the weight is pinned by the Gaussian exactness test
    in tests/test_tmpd.py. What this test does catch: a missing / wrong-sign /
    g^2-only guidance (no guidance scores ~0.44 and g^2 ~0.28 against ~0.05),
    by requiring TMPD to beat SDA run in the SAME settings by a wide margin, and
    the backward-pass bookkeeping of the ``vjp`` row.
    """
    config = _analytical_config(analytical_methods=["TMPD", "SDA"])
    rows = AnalyticalRunner(config, seeds=(0,)).run(aggregate=False)
    kl = {
        r.method: r.value for r in rows if r.metric == Metric.KL_POINTS.value
    }
    assert set(kl) == {"TMPD", "SDA"}
    assert kl["TMPD"] < 0.1
    # Measured: TMPD 0.050, SDA 0.244 (seed 0).
    assert kl["TMPD"] < 0.5 * kl["SDA"]
    # Backward passes are reported beside the NFE: d_y = d = 2 VJPs on each of
    # the M - 1 guided steps (the tau = 0 step is unguided); NFE stays M.
    (vjp,) = [r for r in rows if r.metric == Metric.VJP.value]
    assert vjp.method == "TMPD" and vjp.value == 2 * (20 - 1) and vjp.nfe == 20
    assert all(r.metric != Metric.VJP.value for r in rows if r.method == "SDA")


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
    from lorenz96_lineup import ROW_BY_CELL, run_cells

    assert l96.CELLS["tmpd"] == (Method.TMPD, None)
    assert "tmpd" not in l96.GENERATIVE  # not a prior for prior_enkf, no _fc cell
    assert "tmpd" not in l96.CORRECTED.values()
    assert l96.RUNNERS["tmpd"] is l96.run_tmpd
    assert l96.Lorenz96Runner._NAMES[Method.TMPD] == "tmpd"
    assert "tmpd" in l96.TUNABLE
    # Linear h only: a baseline row of the linear sweep, absent from the arctan one.
    row = ROW_BY_CELL["tmpd"]
    assert row.group == "baselines" and row.generative and row.sweeps == ("linear",)
    assert "tmpd" in run_cells("generative", "linear")
    assert "tmpd" not in run_cells("generative", "arctan")
    assert "tmpd" in l96.case_config("linear").methods
    assert "tmpd" not in l96.case_config("arctan").methods


def test_lorenz96_tmpd_is_refused_in_the_arctan_sweep(tmp_path, monkeypatch):
    from test_lorenz96_case import make_tiny

    cfg = make_tiny("arctan", tmp_path, monkeypatch)
    problem = l96.problem_of(cfg)
    sequences = l96.load_sequences(problem, "test", 1)
    settings = l96.method_settings(l96.case_config("linear"), "tmpd")
    with pytest.raises(ValueError, match="linear h only"):
        l96.run_method(cfg, "tmpd", problem, sequences, "test", 0, settings)


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
    settings = {
        **l96.method_settings(tiny_case, "tmpd"),
        "covariance": covariance,
        # Non-default knobs: they must reach the likelihood / posterior.
        "vjp_chunk_size": 3,
        "guidance_weight": "g_squared",
    }
    out = l96.run_method(tiny_case, "tmpd", problem, sequences, "test", 0, settings)
    (posterior,) = built
    assert isinstance(posterior, TMPDPosterior)
    assert isinstance(posterior.likelihood_model, TMPDLikelihood)
    assert posterior.likelihood_model.covariance == covariance
    assert posterior.likelihood_model.vjp_chunk_size == 3
    assert posterior.guidance_weight == "g_squared"
    # Same endpoint-vanishing g as the DM-SDE cell: zero at both ends.
    g = posterior.diffusion_term(torch.tensor([0.0, 0.5, 1.0]))
    assert g[0] == 0 and g[2] == 0 and g[1] > 0
    assert out.ensembles.shape == (2, problem.num_steps, 6, problem.dim_state)
    assert torch.isfinite(out.ensembles).all()
    assert out.nfe >= settings["num_steps"]
    # Backward passes per assimilation step, beside the NFE: 2 per guided step
    # for row_sum, d_y per guided step for full.
    per_step = 2 if covariance == "row_sum" else problem_num_obs(problem)
    assert out.vjp == per_step * (settings["num_steps"] - 1)


def problem_num_obs(problem) -> int:
    return generative.build_obs_operator(problem).num_obs
