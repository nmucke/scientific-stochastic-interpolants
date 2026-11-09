"""Lorenz 96 tunes at M = 50 and runs the final grid at M in {25, 50, 100, 250}.

`tune` refuses any M but 50; `cell` refuses an M outside the sweep; both unless
--allow-other-m. A generative cell's output paths carry its M."""

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cases.lorenz96 import driver  # noqa: E402


def test_config_uses_fixed_m_everywhere():
    cfg = driver.case_config()
    for name in driver.CELLS:
        settings = driver.method_settings(cfg, name)
        driver.check_fixed_m(name, settings)  # does not raise
        if "num_steps" in settings:
            assert settings["num_steps"] == driver.FIXED_NUM_STEPS


@pytest.mark.parametrize(
    "settings",
    [
        {"num_steps": 25},
        {"num_steps": 200},
        {"num_steps": 50, "correction": {"forecast_num_steps": 5}},
    ],
)
def test_check_fixed_m_refuses_other_steps(settings):
    with pytest.raises(ValueError, match="allow-other-m"):
        driver.check_fixed_m("si_sde", settings)


@pytest.mark.parametrize(
    "settings",
    [{}, {"num_steps": 50}, {"num_steps": 50, "correction": {"forecast_num_steps": None}}],
)
def test_check_fixed_m_accepts_fixed_steps(settings):
    driver.check_fixed_m("si_sde", settings)


def test_cell_cli_refuses_other_m_before_running(monkeypatch):
    ran = []
    monkeypatch.setattr(driver, "evaluate_cell", lambda *a, **k: ran.append(1) or [])
    args = argparse.Namespace(
        method="si_sde", seed=0, set=["num_steps=30"], tag="m30", threads=1,
        allow_other_m=False,
    )
    with pytest.raises(ValueError, match="allow-other-m"):
        driver.cmd_cell(args)
    assert not ran


def test_cell_cli_override_flag_lets_other_m_through(monkeypatch, tmp_path):
    ran = []
    monkeypatch.setattr(driver, "evaluate_cell", lambda *a, **k: ran.append(1) or [])
    monkeypatch.setattr(driver, "RESULTS", tmp_path)
    args = argparse.Namespace(
        method="si_sde", seed=0, set=["num_steps=30"], tag="m30", threads=1,
        allow_other_m=True,
    )
    driver.cmd_cell(args)
    assert ran == [1]


def test_tune_cli_refuses_m_in_grid(monkeypatch, tmp_path):
    monkeypatch.setattr(driver, "RESULTS", tmp_path)
    monkeypatch.setattr(driver, "load_sequences", lambda *a, **k: [])
    ran = []
    monkeypatch.setattr(driver, "run_method", lambda *a, **k: ran.append(1))
    args = argparse.Namespace(
        method="si_sde", grid='{"num_steps": [25, 50]}', set=None, tag="", threads=1,
        innovations=False, allow_other_m=False,
    )
    with pytest.raises(ValueError, match="allow-other-m"):
        driver.cmd_tune(args)
    assert not ran


@pytest.mark.parametrize("m", driver.SWEEP_NUM_STEPS)
def test_cell_cli_accepts_sweep_m_and_names_files_by_m(monkeypatch, tmp_path, m):
    calls = []
    monkeypatch.setattr(
        driver, "evaluate_cell", lambda *a, **k: calls.append(k["num_steps"]) or []
    )
    monkeypatch.setattr(driver, "RESULTS", tmp_path)
    args = argparse.Namespace(
        method="si_sde", seed=1, set=None, tag="", threads=1, allow_other_m=False,
        num_steps=m,
    )
    driver.cmd_cell(args)
    assert calls == [m]
    assert (tmp_path / "linear" / "metrics" / f"lorenz96__si_sde__M{m}__traj1__.csv").exists()


def test_cell_cli_refuses_num_steps_for_cells_without_m(monkeypatch):
    monkeypatch.setattr(driver, "evaluate_cell", lambda *a, **k: [])
    args = argparse.Namespace(
        method="enkf", seed=0, set=None, tag="", threads=1, allow_other_m=False,
        num_steps=100,
    )
    with pytest.raises(ValueError, match="does not apply"):
        driver.cmd_cell(args)


def test_cell_cli_refuses_decoupled_forecast_steps(monkeypatch):
    monkeypatch.setattr(driver, "evaluate_cell", lambda *a, **k: [])
    args = argparse.Namespace(
        method="si_sde_fc", seed=0, set=["correction.forecast_num_steps=50"], tag="f50",
        threads=1, allow_other_m=False, num_steps=None,
    )
    with pytest.raises(ValueError, match="forecast_num_steps"):
        driver.cmd_cell(args)


def test_paths_carry_m_only_for_generative_cells():
    assert driver.cell_path("metrics", "si_sde", 0, num_steps=25).name == (
        "lorenz96__si_sde__M25__traj0__.csv"
    )
    assert driver.cell_path("metrics", "enkf", 0).name == "lorenz96__enkf__traj0__.csv"
    assert driver.cell_label("si_sde_fc", "g0.5", 100) == "si_sde_fc-g0.5__M100"
