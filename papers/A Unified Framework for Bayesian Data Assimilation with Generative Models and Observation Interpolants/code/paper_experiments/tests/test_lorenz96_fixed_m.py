"""Lorenz 96 tunes at M = 50 and runs the final grid at M in {25, 50, 100, 250}.

`tune` refuses any M but 50; `cell` refuses an M outside the sweep; both unless
--allow-other-m. A generative cell's output paths carry its M."""

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cases.lorenz96 import driver  # noqa: E402


class Ran(Exception):
    """Raised by the stubbed run_method: the cell got past its checks."""


@pytest.fixture
def no_run(monkeypatch, tmp_path):
    def run(*args, **kwargs):
        raise Ran

    monkeypatch.setattr(driver, "run_method", run)
    monkeypatch.setattr(driver, "load_sequences", lambda *a, **k: [])
    monkeypatch.setattr(driver, "RESULTS", tmp_path)


def test_config_uses_fixed_m_everywhere():
    cfg = driver.case_config()
    for name in driver.CELLS:
        settings = driver.method_settings(cfg, name)
        driver.check_m(name, settings, (driver.FIXED_NUM_STEPS,))  # does not raise
        if "num_steps" in settings:
            assert settings["num_steps"] == driver.FIXED_NUM_STEPS


@pytest.mark.parametrize(
    "settings",
    [
        {"num_steps": 25},
        {"num_steps": 200},
        {"num_steps": 50, "correction": {"forecast_num_steps": 5}},
        {"num_steps": 50, "correction": {"forecast_num_steps": 50}},
    ],
)
def test_check_m_refuses_other_steps(settings):
    with pytest.raises(ValueError, match="allow-other-m"):
        driver.check_m("si_sde", settings, (50,))


@pytest.mark.parametrize(
    "settings",
    [{}, {"num_steps": 50}, {"num_steps": 50, "correction": {"forecast_num_steps": None}}],
)
def test_check_m_accepts_fixed_steps(settings):
    driver.check_m("si_sde", settings, (50,))


def cell_args(**changes):
    args = dict(method="si_sde", seed=0, set=None, tag="", threads=1, allow_other_m=False,
                num_steps=None)
    return argparse.Namespace(**{**args, **changes})


def test_cell_cli_refuses_other_m_before_running(no_run):
    with pytest.raises(ValueError, match="allow-other-m"):
        driver.cmd_cell(cell_args(set=["num_steps=30"], tag="m30"))


def test_cell_cli_override_flag_lets_other_m_through(no_run):
    with pytest.raises(Ran):
        driver.cmd_cell(cell_args(set=["num_steps=30"], tag="m30", allow_other_m=True))


def test_tune_cli_refuses_m_in_grid(no_run):
    args = argparse.Namespace(
        method="si_sde", grid='{"num_steps": [25, 50]}', set=None, tag="", threads=1,
        innovations=False, allow_other_m=False,
    )
    with pytest.raises(ValueError, match="allow-other-m"):
        driver.cmd_tune(args)


@pytest.mark.parametrize("m", driver.M_VALUES)
def test_cell_cli_accepts_sweep_m_and_names_files_by_m(no_run, tmp_path, m):
    with pytest.raises(Ran):
        driver.cmd_cell(cell_args(seed=1, num_steps=m))
    path = driver.cell_file("metrics", "si_sde", 1, driver.case_config(), num_steps=m)
    assert path == tmp_path / "linear" / "metrics" / f"lorenz96__si_sde__M{m}__traj1__.csv"


def test_cell_cli_refuses_num_steps_for_cells_without_m(no_run):
    with pytest.raises(ValueError, match="does not apply"):
        driver.cmd_cell(cell_args(method="enkf", num_steps=100))


def test_cell_cli_refuses_decoupled_forecast_steps(no_run):
    args = cell_args(method="si_sde_fc", set=["correction.forecast_num_steps=50"], tag="f50")
    with pytest.raises(ValueError, match="forecast_num_steps"):
        driver.cmd_cell(args)


def test_paths_carry_m_only_for_generative_cells():
    assert driver.cell_path("metrics", "si_sde", 0, num_steps=25).name == (
        "lorenz96__si_sde__M25__traj0__.csv"
    )
    assert driver.cell_path("metrics", "enkf", 0).name == "lorenz96__enkf__traj0__.csv"
    assert driver.cell_label("si_sde_fc", "g0.5", 100) == "si_sde_fc-g0.5__M100"
