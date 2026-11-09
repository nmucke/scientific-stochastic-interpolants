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
    GroupedForecastCorrection,
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
        "base": "jacfree", "steps": 2, "q_source": "bank", "group": 64,
    }
    assert parse_fc_variant("shared_fc50") == {
        "base": "shared", "steps": 50, "q_source": "online", "group": None,
    }
    for plain in ("shared", "jacfree", "adapted_shared", "surge_shared", None):
        assert parse_fc_variant(plain) is None


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
    tag = fc_variant("shared", steps, q_source)
    assert {r.variant for r in records} == {tag}
    assert {r.nfe for r in records} == {20 + calls_per_cycle * steps}


def test_grouped_correction_matches_the_library_groups_and_takes_a_remainder():
    torch.manual_seed(0)
    n, d = 96, 2
    x, f, res = torch.randn(n, 1, 1, d), torch.randn(n, 1, 1, d), torch.randn(n, 1, 1, d)
    y = torch.randn(1, d)
    e = torch.randn(n, d)
    op = lambda z: z.reshape(z.shape[0], -1)  # noqa: E731  (H = I)
    kw = dict(operator=op, obs_noise_var=1.0, perturbations=e)
    lib = ForecastCorrection(ForecastCorrectionConfig(enabled=True, group_size=32))
    ours = GroupedForecastCorrection(ForecastCorrectionConfig(enabled=True), group=32)
    torch.testing.assert_close(ours.correct(x, f, y, residuals=res, **kw),
                               lib.correct(x, f, y, residuals=res, **kw))
    assert [s.stop - s.start for s in _groups(100, 32)] == [32, 32, 32, 4]
    assert [s.stop - s.start for s in _groups(97, 32)] == [32, 32, 33]
    assert [s.stop - s.start for s in _groups(100, None)] == [100]
    remainder = GroupedForecastCorrection(ForecastCorrectionConfig(enabled=True), group=64)
    out = remainder.correct(x, f, y, residuals=res, **kw)
    whole = ForecastCorrection(ForecastCorrectionConfig(enabled=True))
    torch.testing.assert_close(out[64:], whole.correct(
        x[64:], f[64:], y, residuals=res[64:], operator=op, obs_noise_var=1.0,
        perturbations=e[64:],
    ))


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
