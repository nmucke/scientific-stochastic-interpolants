"""Regression checks for nested experiment locations and launch directories."""
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))
from common.paths import (  # noqa: E402
    EXPERIMENT_ROOT, REPO_ROOT, find_repo_root, output_path, resolve_checkpoint_paths,
)


@pytest.mark.parametrize("location", ["repo", "experiment", "elsewhere"])
def test_cli_defaults_do_not_depend_on_launch_directory(location, tmp_path):
    cwd = {"repo": REPO_ROOT, "experiment": EXPERIMENT_ROOT, "elsewhere": tmp_path}[location]
    proc = subprocess.run(
        [sys.executable, str(EXPERIMENT / "run.py"), "--cfg", "job", "--resolve"],
        cwd=cwd, capture_output=True, text=True, check=True,
    )
    cfg = OmegaConf.create(proc.stdout)
    assert Path(cfg.results_root) == EXPERIMENT / "results"
    assert Path(cfg.results_file) == EXPERIMENT / "results/navier_stokes_results.csv"


def test_root_discovery_allows_nested_directories_with_spaces(tmp_path):
    root = tmp_path / "research repo"
    (root / "src/scisi").mkdir(parents=True)
    (root / "pyproject.toml").touch()
    nested = root / "manuscripts/A long title/code/paper_experiments"
    nested.mkdir(parents=True)
    assert find_repo_root(nested) == root


def test_checkpoint_paths_preserve_science_and_original_config(tmp_path, monkeypatch):
    config = OmegaConf.create({
        "model": {"mask_path": "data/udales/mask.npz"},
        "test_data": {"paths": ["data/udales", str(tmp_path)],
                      "files": "(170, 178)", "starting_time": 50,
                      "cache_dir": "cache/udales", "use_exisiting_cache": False},
    })
    monkeypatch.chdir(tmp_path)
    resolved = resolve_checkpoint_paths(config)
    assert resolved.model.mask_path == str(REPO_ROOT / "data/udales/mask.npz")
    assert list(resolved.test_data.paths) == [str(REPO_ROOT / "data/udales"), str(tmp_path)]
    assert resolved.test_data.cache_dir == str(REPO_ROOT / "cache/udales")
    assert resolved.test_data.files == "(170, 178)"
    assert resolved.test_data.starting_time == 50
    assert resolved.test_data.use_exisiting_cache is False
    assert config.model.mask_path == "data/udales/mask.npz"
    assert list(config.test_data.paths) == ["data/udales", str(tmp_path)]


def test_relative_output_override_is_relative_to_launch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert output_path("custom figures/a.csv") == tmp_path / "custom figures/a.csv"
    assert output_path(tmp_path / "absolute.csv") == tmp_path / "absolute.csv"


def test_figure_defaults_stay_in_experiments():
    import make_analytical_figures
    import make_ablation_figures
    import temp_make_analytical_figures
    import make_ns_figures
    import make_urban_figures
    import make_sequential_da_tiles

    for module in [make_analytical_figures, make_ns_figures, make_urban_figures,
                   make_ablation_figures, temp_make_analytical_figures]:
        assert module.DEFAULT_OUT.is_relative_to(EXPERIMENT / "figures")
    assert make_sequential_da_tiles.OUT == EXPERIMENT / "figures/sequential_da"


def test_shell_root_discovery_from_unrelated_directory(tmp_path):
    proc = subprocess.run(
        ["bash", "-c", 'source "$1"; printf "%s\\n" "$PE_REPO_ROOT" "$PE_SCRIPT_DIR" "$PE_PYTHON"',
         "paths-test", str(EXPERIMENT / "common/paths.sh")],
        cwd=tmp_path, capture_output=True, text=True, check=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert proc.stdout.splitlines() == [str(REPO_ROOT), str(EXPERIMENT),
                                       str(REPO_ROOT / ".venv/bin/python")]
