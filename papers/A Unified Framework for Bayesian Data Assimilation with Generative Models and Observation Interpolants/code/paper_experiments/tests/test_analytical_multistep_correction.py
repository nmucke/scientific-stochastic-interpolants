"""Forecast-residual correction on the multistep linear--Gaussian case.

The correction itself is tested in ``tests/test_forecast_correction.py`` (repository
root); these tests cover its wiring into ``cases/analytical/multistep.py``: the
corrected rows reach the Kalman filter, switching it off (or on, for the other
rows) changes nothing, and ``forecast_num_steps`` is what the forecasts use.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

import scisi.posterior_models.forecast_correction as fc_module  # noqa: E402
from cases.analytical.multistep import (  # noqa: E402
    FC_DEFAULT_BANK_SIZE,
    MultistepRunner,
    _groups,
    fc_variant,
    parse_fc_variant,
)
from results_schema import Method  # noqa: E402
from scisi.posterior_models.forecast_correction import (  # noqa: E402
    ForecastCorrection,
    ForecastCorrectionConfig,
)

PARTIAL = dict(
    A=[[0.9, 0.3], [-0.3, 0.9]],
    Q=[[1.0, 0.0], [0.0, 0.25]],
    H=[[1.0, 0.0]],
    R=[[0.5]],
)
OURS = [Method.OURS_SI_SDE, Method.OURS_DM_SDE, Method.OURS_FM_ODE]


def _config(**case):
    if case.get("system"):
        case.setdefault("scenario_name", "multistep:test")
    return OmegaConf.create({
        "case": {"plot_dimension": 2, "adapted_variant": False, **case},
        "ensemble_size": 64,
        "num_steps": 50,
    })


def _evaluate(method, *, num_steps, **case):
    runner = MultistepRunner(_config(**case), seeds=(0,))
    ctx = runner.make_context(method, None, 0)
    ctx.num_steps = num_steps
    torch.manual_seed(0)
    return runner, list(runner.evaluate(ctx))


def _curves(runner):
    """``{(method, variant, metric, step): value}`` of every rollout so far."""
    return {
        (r["method"], r["variant"], r["metric"], r["step"]): r["value"]
        for r in runner._per_step
    }


def _last(runner, metric):
    last = runner.num_cycles - 1
    return {
        (r["method"], r["variant"]): r["value"]
        for r in runner._per_step if r["metric"] == metric and r["step"] == last
    }


def test_variant_tags_round_trip():
    assert fc_variant("shared", 5) == "shared_fc5"
    assert fc_variant("jacfree", 2, "bank", 64) == "jacfree_fc2_bank_g64"
    assert parse_fc_variant("jacfree_fc2_bank_g64") == {
        "base": "jacfree", "steps": 2, "q_source": "bank", "group": 64, "bank_size": 4096,
    }
    assert parse_fc_variant("shared_fc50") == {
        "base": "shared", "steps": 50, "q_source": "online", "group": None,
        "bank_size": None,
    }
    for plain in ("shared", "jacfree", "adapted_shared", "surge_shared", None):
        assert parse_fc_variant(plain) is None


def test_bank_size_is_tagged_only_when_not_the_default():
    # The default size, or none given, leaves the stored tags as they were.
    assert fc_variant("shared", 5, "bank", None, FC_DEFAULT_BANK_SIZE) == "shared_fc5_bank"
    assert fc_variant("shared", 5, "bank") == "shared_fc5_bank"
    assert fc_variant("shared", 5, "online", 64, 1024) == "shared_fc5_g64"  # unused
    tag = fc_variant("shared", 5, "bank", 64, 1024)
    assert tag == "shared_fc5_bank1024_g64"
    assert parse_fc_variant(tag) == {
        "base": "shared", "steps": 5, "q_source": "bank", "group": 64, "bank_size": 1024,
    }
    # Stored tags of the existing grid keep their meaning.
    for stored, meaning in {
        "shared_fc5_bank": ("shared", 5, "bank", None, 4096),
        "jacfree_fc50_g64": ("jacfree", 50, "online", 64, None),
        "shared_fc10": ("shared", 10, "online", None, None),
    }.items():
        spec = parse_fc_variant(stored)
        assert (spec["base"], spec["steps"], spec["q_source"], spec["group"],
                spec["bank_size"]) == meaning
        assert fc_variant(*meaning[:4], meaning[4]) == stored


def test_bank_size_reaches_the_row_variant():
    runner = MultistepRunner(
        _config(fc_only=True, fc_steps=[3], fc_q_source="bank", fc_bank_size=128,
                fc_variants=["shared"]),
        seeds=(0,),
    )
    assert [v for v, _, _ in runner._fc_modes(10)] == ["shared_fc3_bank128"]
    default = MultistepRunner(
        _config(fc_only=True, fc_steps=[3], fc_q_source="bank", fc_variants=["shared"]),
        seeds=(0,),
    )
    assert [v for v, _, _ in default._fc_modes(10)] == ["shared_fc3_bank"]


def test_figure_group_sizes_come_from_the_data():
    import make_analytical_multistep_figures as figs

    assert figs.group_sizes(["shared", "g64", "adapted_shared_g64", None]) == [64]
    assert figs.group_sizes(["shared_fc5_g32", "g16", "shared_fc5_bank"]) == [16, 32]
    scalar = {
        ("s", "EnKF", None, "kl_points"): {},
        ("s", "EnKF", "g32", "kl_points"): {},
        ("s", "Ours (SI-SDE)", "adapted_shared_g32", "kl_points"): {},
        ("s", "Ours (SI-SDE)", "shared_fc5_g32", "kl_points"): {},
    }
    labels = [label for _, _, label in figs.correction_rows(scalar, "s")]
    assert "EnKF, E=32" in labels
    assert "Ours SI-SDE adapted_shared, E=32" in labels
    assert not any("E=64" in label for label in labels)


def _fake_python(tmp_path) -> Path:
    """A stand-in for ``run.py``: creates its output files and logs its arguments."""
    fake = tmp_path / "fake_python"
    fake.write_text(
        "#!/bin/sh\n"
        # The grid asks the case for its fc cell suffix: hand that to Python.
        f"[ \"$1\" = -m ] && exec \"{sys.executable}\" \"$@\"\n"
        "for a in \"$@\"; do case \"$a\" in results_file=*|+per_step_file=*) "
        ": > \"${a#*=}\" ;; esac; done\n"
        "echo \"$@\" >> \"$ROOT/args.txt\"\n"
    )
    fake.chmod(0o755)
    return fake


@pytest.mark.skipif(not Path("/bin/bash").exists(), reason="needs /bin/bash")
def test_grid_survives_an_empty_cell_list_and_names_bank_sizes(tmp_path):
    """Under /bin/bash (3.2 on macOS) with set -u an empty spec list must not abort."""
    env = {
        **os.environ, "PY": str(_fake_python(tmp_path)), "ROOT": str(tmp_path),
        "SEEDS": "0", "SETTINGS": "paper", "LINEUPS": "fc base", "FC_STEPS": " ",
    }
    script = str(EXPERIMENT / "run_analytical_multistep_grid.sh")
    run = subprocess.run(["/bin/bash", script], capture_output=True, text=True, env=env)
    assert run.returncode == 1, run.stderr
    assert "unbound" not in run.stderr
    assert "FAIL no cells for lineup fc" in run.stdout
    assert "failed cells: 1" in run.stdout
    # The lineup after the empty one still ran.
    assert (tmp_path / "metrics" / "paper__M50__K20__E100000__traj0.csv").exists()

    env.update(LINEUPS="fc", FC_STEPS="2", FC_Q_SOURCE="bank", FC_BANK_SIZE="1024")
    subprocess.run(["/bin/bash", script], check=True, capture_output=True, text=True, env=env)
    assert (tmp_path / "metrics" / "paper_fc2_bank1024__M50__K20__E100000__traj0.csv").exists()
    assert "+fc_bank_size=1024" in (tmp_path / "args.txt").read_text()
    env.update(FC_BANK_SIZE="4096", FC_STEPS="3")
    subprocess.run(["/bin/bash", script], check=True, capture_output=True, text=True, env=env)
    assert (tmp_path / "metrics" / "paper_fc3_bank__M50__K20__E100000__traj0.csv").exists()


@pytest.mark.parametrize("spec", [{}, PARTIAL], ids=["paper", "partial"])
@pytest.mark.parametrize("method", OURS)
def test_corrected_rows_approach_the_kalman_filter(spec, method):
    """The plain shared row targets the member chain; corrected, it hits Kalman."""
    runner, _ = _evaluate(
        method, num_steps=100, num_cycles=4, n_eval_samples=16384, system=spec,
        forecast_correction=True, fc_steps=[100], fc_variants=["shared"],
    )
    kl = _last(runner, "kl_exact")
    corrected = kl[(method.value, "shared_fc100")]
    assert corrected < 0.01
    if not spec:  # the paper system has the larger chain bias
        assert kl[(method.value, "shared")] > 3 * corrected
    ratio = _last(runner, "var_ratio")[(method.value, "shared_fc100")]
    assert ratio == pytest.approx(1.0, abs=0.06)


@pytest.mark.parametrize("method", [Method.OURS_SI_SDE, Method.OURS_FM_ODE])
def test_correction_off_and_on_leave_the_other_rows_identical(method):
    """Off is the default lineup; on adds rows and changes none of the others."""
    case = dict(num_cycles=3, n_eval_samples=512, adapted_variant=True)
    off, _ = _evaluate(method, num_steps=20, **case)
    default, _ = _evaluate(method, num_steps=20, **case, forecast_correction=False,
                           fc_steps=[2])
    on, _ = _evaluate(method, num_steps=20, **case, forecast_correction=True,
                      fc_steps=[2, 5])
    off_rows, on_rows = _curves(off), _curves(on)
    assert _curves(default) == off_rows
    assert {k[1] for k in off_rows} == {"jacfree", "shared", "adapted_shared"}
    assert {k[1] for k in on_rows} - {k[1] for k in off_rows} == {
        "jacfree_fc2", "shared_fc2", "jacfree_fc5", "shared_fc5",
    }
    assert {k: on_rows[k] for k in off_rows} == off_rows


def test_fc_only_cell_holds_only_corrected_rows_and_no_references():
    runner = MultistepRunner(
        _config(num_cycles=2, n_eval_samples=64, fc_only=True, fc_steps=[3]),
        seeds=(0,),
    )
    assert tuple(runner.methods()) == tuple(OURS)
    rows = runner.run(aggregate=False)
    assert {r.variant for r in rows} == {"jacfree_fc3", "shared_fc3"}
    with pytest.raises(ValueError, match="separate cells"):
        MultistepRunner(_config(fc_only=True, surge_only=True), seeds=(0,)).methods()


@pytest.mark.parametrize("q_source,calls_per_cycle", [("online", 2), ("bank", 1)])
@pytest.mark.parametrize("steps", [1, 7])
def test_forecast_num_steps_is_honoured(monkeypatch, q_source, calls_per_cycle, steps):
    """Every forecast (and the bank) uses ``fc_steps``; NFE counts them."""
    seen: list[int] = []
    original = fc_module.draw_unguided_forecast

    def spy(posterior, field_history, num_steps, *args, **kwargs):
        seen.append(int(num_steps))
        return original(posterior, field_history, num_steps, *args, **kwargs)

    monkeypatch.setattr(fc_module, "draw_unguided_forecast", spy)
    cycles = 3
    _, records = _evaluate(
        Method.OURS_SI_SDE, num_steps=20, num_cycles=cycles, n_eval_samples=128,
        fc_only=True, fc_steps=[steps], fc_variants=["shared"], fc_q_source=q_source,
        fc_bank_size=64,
    )
    assert set(seen) == {steps}
    bank_calls = 2 if q_source == "bank" else 0  # one pair of forecasts, built once
    assert len(seen) == cycles * calls_per_cycle + bank_calls
    tag = fc_variant("shared", steps, q_source, None, 64)  # fc_bank_size above
    assert {r.variant for r in records} == {tag}
    assert {r.nfe for r in records} == {20 + calls_per_cycle * steps}


def test_group_partition_rule():
    assert [s.stop - s.start for s in _groups(100, 32)] == [32, 32, 32, 4]
    assert [s.stop - s.start for s in _groups(97, 32)] == [32, 32, 33]
    assert [s.stop - s.start for s in _groups(129, 64)] == [64, 65]
    assert [s.stop - s.start for s in _groups(100, None)] == [100]
    assert [s.stop - s.start for s in _groups(50, 64)] == [50]


@pytest.mark.parametrize("n", [100, 129, 2000])
def test_corrected_groups_are_the_comparator_groups(n):
    """The library's ``group_size`` partition is ``_groups``, which the grouped
    EnKF / ``adapted_shared`` comparators use: the grouped correction of the batch
    equals one ungrouped correction per ``_groups`` slice."""
    torch.manual_seed(0)
    d, group = 2, 64
    x, f, f2 = (torch.randn(n, 1, 1, d) for _ in range(3))
    y, e = torch.randn(1, d), torch.randn(n, d)
    op = lambda z: z.reshape(z.shape[0], -1)  # noqa: E731  (H = I)
    grouped = ForecastCorrection(ForecastCorrectionConfig(enabled=True, group_size=group))
    out = grouped.correct(x, f, y, op, 1.0, f2=f2, perturbations=e)
    slices = _groups(n, group)
    # One diagnostics entry per group, in one record (one call, one step).
    assert len(grouped.diagnostics.records) == 1
    assert grouped.diagnostics.records[0]["spread_rms"].shape == (len(slices),)
    for sl in slices:
        alone = ForecastCorrection(ForecastCorrectionConfig(enabled=True))
        torch.testing.assert_close(
            out[sl], alone.correct(x[sl], f[sl], y, op, 1.0, f2=f2[sl], perturbations=e[sl]))


def test_make_correction_uses_the_library_groups():
    runner = MultistepRunner(
        _config(fc_only=True, fc_steps=[3], fc_q_source="bank", fc_group_size=64,
                fc_variants=["shared"]),
        seeds=(0,),
    )
    (_, _, spec), = runner._fc_modes(10)
    corr = runner._make_correction(spec)
    assert type(corr) is ForecastCorrection
    assert (corr.config.group_size, corr.config.q_source, corr.config.forecast_num_steps) == (
        64, "bank", 3)
    assert runner.fc_bank_size == FC_DEFAULT_BANK_SIZE


def test_group_cells_tag_enkf_and_adapted_rows():
    runner = MultistepRunner(
        _config(num_cycles=2, n_eval_samples=200, group_only=True, fc_group_size=64),
        seeds=(0,),
    )
    rows = runner.run(aggregate=False)
    assert {(r.method, r.variant) for r in rows} == {
        *((m.value, "adapted_shared_g64") for m in OURS), ("EnKF", "g64"),
    }
    with pytest.raises(ValueError, match="fc_group_size"):
        MultistepRunner(_config(group_only=True), seeds=(0,)).methods()


def test_grid_fc_lineup_names_cells_by_setting(tmp_path):
    env = {
        **os.environ, "PY": sys.executable, "ROOT": str(tmp_path), "SEEDS": "0",
        "N_EVAL": "64", "CYCLES": "2", "M": "10", "SETTINGS": "paper",
        "LINEUPS": "fc", "FC_STEPS": "1 3", "FC_VARIANTS": "shared",
        "FC_Q_SOURCE": "bank", "FC_GROUP": "32",
    }
    subprocess.run(
        ["bash", str(EXPERIMENT / "run_analytical_multistep_grid.sh")],
        check=True, capture_output=True, text=True, env=env,
    )
    names = sorted(p.name for p in (tmp_path / "metrics").glob("*.csv"))
    assert names == [
        "paper_fc1_shared_bank_g32__M10__K2__E64__traj0.csv",
        "paper_fc3_shared_bank_g32__M10__K2__E64__traj0.csv",
    ]
    text = (tmp_path / "metrics" / names[1]).read_text()
    assert "shared_fc3_bank_g32" in text and "Kalman filter" not in text
