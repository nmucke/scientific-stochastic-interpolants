"""Statistics of the Lorenz 96 PF gate on synthetic inputs with known answers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cases.lorenz96 import pf_gate  # noqa: E402
from scisi.particle_filter.bootstrap import ParticleFilterResult  # noqa: E402

T, D = 6, 4


def fake_run(mean, variance=1.0, ess=1000.0, log_evidence=0.0, seed=0, n=10_000):
    """A PF result with constant moments; samples only feed the W2 diagnostic."""
    gen = torch.Generator().manual_seed(seed)
    mean = torch.full((T, D), float(mean), dtype=torch.float64)
    variance = torch.full((T, D), float(variance), dtype=torch.float64)
    return ParticleFilterResult(
        mean=mean,
        variance=variance,
        covariance=torch.diag_embed(variance),
        samples=torch.randn(T, 64, D, generator=gen, dtype=torch.float64),
        ess=torch.full((T,), float(ess), dtype=torch.float64),
        log_evidence=torch.full((T,), float(log_evidence) / T, dtype=torch.float64),
        num_particles=n,
    )


def test_fit_rate_of_exact_power_law():
    ladder = [1e4, 1e5, 1e6, 4e6]
    assert pf_gate.fit_rate(ladder, [3.0 * n**-0.5 for n in ladder]) == pytest.approx(
        -0.5, abs=1e-12
    )
    assert pf_gate.fit_rate([1e5, 4e6], [2.0, 2.0]) == pytest.approx(0.0, abs=1e-12)


def test_top_rung_leaves_each_run_out_of_its_own_reference():
    top = [fake_run(0.0), fake_run(0.0), fake_run(3.0)]
    stats = pf_gate.rung_stats(top, top, skip=1)
    # Leave-one-out references: 1.5, 1.5 and 0.0. Using all runs would give 1, 1, 2.
    errors = torch.tensor([1.5] * (2 * T * D) + [3.0] * (T * D), dtype=torch.float64)
    assert stats["mean_error"]["median"] == pytest.approx(1.5)
    assert stats["mean_error"]["max"] == pytest.approx(3.0)
    assert stats["mean_error"]["q95"] == pytest.approx(float(torch.quantile(errors, 0.95)))
    assert stats["mean_error_rms"] == pytest.approx(float(errors.pow(2).mean().sqrt()))
    assert stats["log_var_error"]["max"] == pytest.approx(0.0, abs=1e-12)


def test_lower_rung_is_scored_against_all_top_runs_with_known_q95():
    top = [fake_run(0.0), fake_run(2.0)]  # reference mean 1, variance 1
    offsets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    runs = [fake_run(1.0 + o, variance=2.0, ess=100 + i, log_evidence=i) for i, o in enumerate(offsets)]
    stats = pf_gate.rung_stats(runs, top, skip=1)
    expected = torch.tensor(offsets, dtype=torch.float64).repeat_interleave(T * D)
    assert stats["mean_error"]["q95"] == pytest.approx(float(torch.quantile(expected, 0.95)))
    assert stats["mean_error"]["median"] == pytest.approx(float(expected.median()))
    assert stats["log_var_error"]["q95"] == pytest.approx(float(torch.log(torch.tensor(2.0))))
    assert stats["ess_abs_min"] == 100.0
    assert stats["log_evidence_sd"] == pytest.approx(
        float(torch.arange(10, dtype=torch.float64).std())
    )


def passing_stats():
    return dict(
        mean_error=dict(q95=0.05),
        log_var_error=dict(q95=0.05),
        log_evidence_sd=0.2,
        ess_abs_min=1000.0,
    )


def test_verdict_passes_inside_all_criteria():
    result = pf_gate.verdict(passing_stats(), -0.5)
    assert result["passed"] and all(result["checks"].values())


@pytest.mark.parametrize(
    "criterion, change, rate",
    [
        ("mean_error_q95", lambda s: s["mean_error"].update(q95=0.11), -0.5),
        ("log_var_error_q95", lambda s: s["log_var_error"].update(q95=0.11), -0.5),
        ("log_evidence_sd", lambda s: s.update(log_evidence_sd=0.51), -0.5),
        ("ess_abs_min", lambda s: s.update(ess_abs_min=499.0), -0.5),
        ("rate", lambda s: None, -0.25),
        ("rate", lambda s: None, -0.75),
    ],
)
def test_verdict_fails_on_each_criterion_alone(criterion, change, rate):
    stats = passing_stats()
    change(stats)
    result = pf_gate.verdict(stats, rate)
    assert not result["passed"]
    assert [k for k, ok in result["checks"].items() if not ok] == [criterion]


def synthetic_filter(variance_factor, sequences=256, samples=200, seed=0):
    """Truths drawn from N(mean, var); the filter states ``variance_factor * var``."""
    gen = torch.Generator().manual_seed(seed)
    truths, runs = [], []
    for _ in range(sequences):
        mean = torch.randn(T, D, generator=gen, dtype=torch.float64)
        var = 0.5 + torch.rand(T, D, generator=gen, dtype=torch.float64)
        truths.append(mean + var.sqrt() * torch.randn(T, D, generator=gen, dtype=torch.float64))
        stated = variance_factor * var
        draws = mean[:, None] + stated.sqrt()[:, None] * torch.randn(
            T, samples, D, generator=gen, dtype=torch.float64
        )
        runs.append(SimpleNamespace(mean=mean, variance=stated, samples=draws))
    observed = torch.tensor([True, False, True, False])
    return pf_gate.calibration_stats(truths, runs, observed, skip=1)


def test_calibration_of_a_correct_posterior():
    stats = synthetic_filter(1.0)
    for block in ("all", "observed", "unobserved"):
        assert abs(stats[block]["z2"] - 1.0) < 4 * stats[block]["z2_se"] + 0.05
        assert stats[block]["rank_max_deviation"] < 0.03


@pytest.mark.parametrize("factor, z2", [(2.0, 0.5), (0.5, 2.0)])
def test_calibration_detects_a_wrong_variance(factor, z2):
    stats = synthetic_filter(factor)
    assert stats["all"]["z2"] == pytest.approx(z2, rel=0.1)
    assert stats["all"]["rank_max_deviation"] > 0.05


def test_screen_refuses_a_file_from_another_grid(tmp_path):
    out = tmp_path / "screen.csv"
    out.write_text("dim_state,N,seconds\n16,100,1.0\n")
    args = SimpleNamespace(
        grid='{"obs_variance": [4.0]}', out=str(out), n=10, sequences=1, enkf=8, skip=0, steps=2
    )
    with pytest.raises(ValueError, match="has columns"):
        pf_gate.cmd_screen(args)
    assert out.read_text() == "dim_state,N,seconds\n16,100,1.0\n"


def test_screen_writes_its_columns_and_appends_to_its_own_file(tmp_path):
    out = tmp_path / "screen.csv"
    args = SimpleNamespace(
        grid='{"obs_variance": [4.0]}', out=str(out), n=200, sequences=1, enkf=16, skip=0, steps=3
    )
    pf_gate.cmd_screen(args)
    pf_gate.cmd_screen(args)
    lines = out.read_text().splitlines()
    assert lines[0].split(",") == ["obs_variance", "N", *pf_gate.SCREEN_STATS, "seconds"]
    assert len(lines) == 3
