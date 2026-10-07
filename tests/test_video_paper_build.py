"""Dataset-free manuscript setup generation preserves frozen-manifest identity."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "video_case_for_build_test", ROOT / "research/ito_maps/paper/build_video_case.py"
)
case = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(case)


def preparation_summary(tmp_path):
    content = json.dumps(
        {"clips": [{"split": s} for s in ["train", "train", "validation", "test"]]}
    )
    manifest = {
        "path": "data/video_forecasting/kth_64/manifest.json",
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
    }
    summary = {
        "datasets": [
            {
                "dataset": "kth",
                "manifest_path": manifest["path"],
                "manifest_sha256": manifest["sha256"],
                "clips": 4,
                "split_summary": {
                    s: {"clips": n}
                    for s, n in [("train", 2), ("validation", 1), ("test", 1)]
                },
            }
        ]
    }
    path = tmp_path / "data_summary.json"
    path.write_text(json.dumps(summary))
    return {"manifest": manifest}, path, content


def test_no_dataset_needed_for_audited_split_counts(tmp_path):
    protocol, summary, _ = preparation_summary(tmp_path)
    assert case.split_counts_for_protocol("kth", protocol, tmp_path, summary) == {
        "train": 2,
        "validation": 1,
        "test": 1,
    }
    assert not (tmp_path / "data").exists()


def test_summary_must_match_frozen_manifest_hash(tmp_path):
    protocol, summary, _ = preparation_summary(tmp_path)
    protocol["manifest"]["sha256"] = "different"
    with pytest.raises(ValueError, match="summary differs from frozen"):
        case.split_counts_for_protocol("kth", protocol, tmp_path, summary)


def test_local_manifest_if_present_is_still_verified(tmp_path):
    protocol, summary, content = preparation_summary(tmp_path)
    manifest = tmp_path / protocol["manifest"]["path"]
    manifest.parent.mkdir(parents=True)
    manifest.write_text(content)
    assert (
        case.split_counts_for_protocol("kth", protocol, tmp_path, summary)["train"] == 2
    )
    manifest.write_text(content + " ")
    with pytest.raises(ValueError, match="Local manifest differs"):
        case.split_counts_for_protocol("kth", protocol, tmp_path, summary)


def test_local_manifest_checks_summary_count_consistency(tmp_path):
    protocol, summary, content = preparation_summary(tmp_path)
    manifest = tmp_path / protocol["manifest"]["path"]
    manifest.parent.mkdir(parents=True)
    manifest.write_text(content)
    data = json.loads(summary.read_text())
    data["datasets"][0]["split_summary"]["train"]["clips"] = 1
    data["datasets"][0]["split_summary"]["test"]["clips"] = 2
    summary.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="counts differ from local"):
        case.split_counts_for_protocol("kth", protocol, tmp_path, summary)
