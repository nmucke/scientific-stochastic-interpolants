"""Restart guards must fail before an existing training log is redirected."""

import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

EXPERIMENTS = (
    Path(__file__).resolve().parents[1] / "research/video_forecasting/experiments"
)
SPEC = importlib.util.spec_from_file_location(
    "video_stage_status", EXPERIMENTS / "training_stage_status.py"
)
STATUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STATUS)


def completed(path, updates):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text("{}")
    (path / "metadata.json").write_text("{}")
    (path / "model.pth").write_bytes(b"test weight placeholder")
    (path / "history.json").write_text(json.dumps([{"step": updates}]))


def test_stage_status_distinguishes_missing_complete_and_incomplete(tmp_path):
    directory = tmp_path / "run"
    assert STATUS.stage_status(directory, 10) == "missing"
    directory.mkdir()
    assert STATUS.stage_status(directory, 10) == "missing"
    (directory / "config.yaml").write_text("{}")
    with pytest.raises(RuntimeError, match="incomplete"):
        STATUS.stage_status(directory, 10)
    completed(directory, 10)
    assert STATUS.stage_status(directory, 10) == "complete"
    with pytest.raises(RuntimeError, match="unexpected updates"):
        STATUS.stage_status(directory, 20)


@pytest.mark.parametrize("state", ["missing", "incomplete", "complete"])
def test_shell_guard_precedes_redirection_and_never_launches_for_existing_stage(
    tmp_path, state
):
    manifest = tmp_path / "data/video_forecasting/kth_64/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    # Only the real CPU status helper is executed. Any attempted training is
    # intercepted and returns immediately without importing torch or CUDA.
    python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == research/video_forecasting/experiments/training_stage_status.py ]]; then\n'
        f'  exec {shlex.quote(sys.executable)} {shlex.quote(str(EXPERIMENTS / "training_stage_status.py"))} "$2" "$3"\n'
        "fi\n"
        "touch training_launch_attempted\nexit 42\n"
    )
    python.chmod(0o755)
    result_dir = tmp_path / "research/video_forecasting/results"
    result_dir.mkdir(parents=True)
    log = result_dir / "kth_follmer_stochastic_interpolant_train.log"
    log.write_text("preserve previous diagnostics\n")
    stage_root = tmp_path / "checkpoints/video/kth"
    if state == "incomplete":
        incomplete = stage_root / "follmer_stochastic_interpolant"
        incomplete.mkdir(parents=True)
        (incomplete / "config.yaml").write_text("{}")
    elif state == "complete":
        for stage, updates in [
            ("follmer_stochastic_interpolant", 20000),
            ("flow_matching", 20000),
            ("follmer_ito_map_psd", 2000),
            ("follmer_ito_map", 500),
            ("flow_map_psd", 2000),
            ("flow_map", 500),
        ]:
            completed(stage_root / stage, updates)
    result = subprocess.run(
        ["bash", str(EXPERIMENTS / "train_all_video_models.sh"), "kth"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    attempted = (tmp_path / "training_launch_attempted").exists()
    if state == "missing":
        assert result.returncode == 42 and attempted
    else:
        assert not attempted
        assert log.read_text() == "preserve previous diagnostics\n"
        if state == "complete":
            assert result.returncode == 0
        else:
            assert result.returncode != 0
            assert "incomplete" in result.stderr
