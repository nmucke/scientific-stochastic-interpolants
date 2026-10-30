"""Archiving preserves evidence and counts warm-start stages without teachers."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / "research/ito_maps/experiments/archive_training.py"
spec = importlib.util.spec_from_file_location("archive_training", SCRIPT)
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def write_run(directory, updates, seconds, initial=None):
    directory.mkdir()
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "seed": 3,
                "initial_student_checkpoint": str(initial) if initial else None,
                "teacher_checkpoint": "not-a-student-ancestor",
            }
        )
    )
    (directory / "config.yaml").write_text("model: {}\n")
    (directory / "history.json").write_text(
        json.dumps(
            [
                {
                    "epoch": 0,
                    "step": updates,
                    "val_loss": 0.02,
                    "elapsed_seconds": seconds,
                }
            ]
        )
    )
    (directory / "model.pth").write_bytes(b"checkpoint-bytes")


def test_archive_keeps_hashes_and_initialization_lineage(tmp_path):
    first, refined, final = tmp_path / "psd", tmp_path / "refined", tmp_path / "rollout"
    write_run(first, 2000, 100)
    refined.mkdir()
    (refined / "refinement_provenance.json").write_text(
        json.dumps(
            {
                "source_checkpoint_dir": str(first),
                "source_num_path_features": 4,
                "num_path_features": 16,
            }
        )
    )
    write_run(final, 500, 200, refined / "model.pth")
    output, record = archive.archive_run(final, tmp_path / "archive")
    assert record["cumulative_recorded_student_updates"] == 2500
    assert record["cumulative_recorded_student_stage_seconds"] == 300
    assert [stage["kind"] for stage in record["initialization_lineage"]] == [
        "training",
        "feature_refinement",
        "training",
    ]
    assert record["sha256"] == archive.hash_file(final / "model.pth")
    assert (output / "history.json").read_bytes() == (
        final / "history.json"
    ).read_bytes()
    assert not (output / "model.pth").exists()
    assert (
        json.loads((output / "checkpoint_provenance.json").read_text())[
            "weights_copied"
        ]
        is False
    )


def test_incomplete_run_is_not_archived(tmp_path):
    with pytest.raises(FileNotFoundError):
        archive.archive_run(tmp_path / "missing", tmp_path / "output")
    assert not (tmp_path / "output").exists()
