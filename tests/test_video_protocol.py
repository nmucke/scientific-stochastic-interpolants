"""Immutable video protocol and artifact audit checks using small local fixtures."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

EXPERIMENTS = (
    Path(__file__).resolve().parents[1] / "research/video_forecasting/experiments"
)
sys.path.insert(0, str(EXPERIMENTS))
from freeze_video_protocol import (  # noqa: E402
    FAMILIES,
    RESULTS,
    TEACHERS,
    freeze_protocol,
    sha256,
    verify_frozen,
)
from verify_video_results import verify_evaluation  # noqa: E402


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def frozen_tree(tmp_path):
    manifest_dir = tmp_path / "data/video_forecasting/kth_64"
    manifest_dir.mkdir(parents=True)
    clip_path = manifest_dir / "clip.npy"
    np.save(clip_path, np.zeros((40, 4, 4, 1), dtype=np.uint8))
    clips = [
        {
            "clip_id": f"clip_{i}",
            "path": "clip.npy",
            "split": "test",
            "num_frames": 40,
            "source_member": f"source_{i // 2}",
            "sha256": sha256(clip_path),
        }
        for i in range(128)
    ]
    write_json(
        manifest_dir / "manifest.json",
        {
            "dataset": "kth",
            "resolution": 64,
            "channels": 1,
            "frame_stride": 1,
            "clips": clips,
        },
    )
    for stage in [*FAMILIES, "follmer_ito_map_psd", "flow_map_psd"]:
        family = stage.removesuffix("_psd")
        is_psd = stage.endswith("_psd")
        is_map = family in TEACHERS
        epochs, batches = (20, 100) if is_psd else ((10, 50) if is_map else (200, 100))
        directory = tmp_path / f"checkpoints/video/kth/{stage}"
        directory.mkdir(parents=True)
        config = {
            "seed": 0,
            "dataset_name": "kth",
            "channels": 1,
            "len_field_history": 10,
            "manifest_path": "data/video_forecasting/kth_64/manifest.json",
            "preprocesser": None,
            "model": {"_target_": "test." + FAMILIES[family]},
            "trainer": {
                "num_epochs": epochs,
                "max_train_batches": batches,
                "checkpoint_dir": f"checkpoints/video/kth/{stage}",
            },
            "train_data": {"dataset": {"split": "train"}},
            "val_data": {"dataset": {"split": "validation"}},
            "initial_student_checkpoint": None,
        }
        metadata = {"seed": 0, "initialization": "random"}
        if is_map:
            teacher = f"checkpoints/video/kth/{TEACHERS[family]}"
            config["teacher"] = {"checkpoint_dir": teacher, "initialize_student": True}
            config["trainer"].update(
                objective="progressive" if is_psd else "teacher_rollout",
                teacher_steps=32,
            )
            metadata["teacher_checkpoint"] = teacher
            if not is_psd:
                config["initial_student_checkpoint"] = (
                    f"checkpoints/video/kth/{family}_psd/model.pth"
                )
            metadata["initial_student_checkpoint"] = config[
                "initial_student_checkpoint"
            ]
        OmegaConf.save(OmegaConf.create(config), directory / "config.yaml")
        write_json(directory / "metadata.json", metadata)
        write_json(
            directory / "history.json",
            [
                {
                    "step": (epoch + 1) * batches,
                    "train_loss": 1 / (epoch + 1),
                    "val_loss": 1 / (epoch + 1),
                    "elapsed_seconds": epoch + 1,
                }
                for epoch in range(epochs)
            ],
        )
        (directory / "model.pth").write_bytes(stage.encode())
    metrics = {}
    for name in ["i3d_google_conversion", "alexnet_imagenet", "lpips_calibration"]:
        path = tmp_path / "cache" / (name + ".pth")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(name.encode())
        metrics[name] = {"path": str(path), "sha256": sha256(path)}
    write_json(
        tmp_path / "research/video_forecasting/notes/metric_weights.json",
        {"artifacts": metrics},
    )
    return tmp_path


def test_freeze_archives_metadata_and_is_immutable(frozen_tree):
    protocol = freeze_protocol("kth", root=frozen_tree)
    assert len(protocol["cases"]) == len(set(protocol["selected_indices"])) == 128
    assert len(protocol["training_stages"]) == 6
    assert protocol["models"]["flow_map"]["num_steps"] == [1, 8]
    assert protocol["models"]["flow_matching"]["num_steps"] == [16, 64]
    assert protocol["test_evaluation_started_at_freeze"] is False
    for stage in protocol["training_stages"]:
        archive = frozen_tree / RESULTS / "training" / f"kth_{stage}"
        assert (archive / "checkpoint_provenance.json").exists()
        assert (archive / "history.json").exists()
        assert not (archive / "model.pth").exists()
    with pytest.raises(FileExistsError, match="Immutable"):
        freeze_protocol("kth", root=frozen_tree)
    assert freeze_protocol("kth", root=frozen_tree, verify_only=True) == protocol


@pytest.mark.parametrize("corruption", ["incomplete", "teacher", "test_started"])
def test_freeze_rejects_unfinished_training_wrong_lineage_and_prior_tests(
    frozen_tree, corruption
):
    if corruption == "incomplete":
        history = frozen_tree / "checkpoints/video/kth/flow_matching/history.json"
        rows = json.loads(history.read_text())
        rows[-1]["step"] -= 1
        write_json(history, rows)
    elif corruption == "teacher":
        path = frozen_tree / "checkpoints/video/kth/flow_map/config.yaml"
        cfg = OmegaConf.load(path)
        cfg.teacher.checkpoint_dir = (
            "checkpoints/video/kth/follmer_stochastic_interpolant"
        )
        OmegaConf.save(cfg, path)
    else:
        write_json(frozen_tree / RESULTS / "test/kth/flow_map/forecasts.json", [])
    with pytest.raises(ValueError):
        freeze_protocol("kth", root=frozen_tree)
    assert not (frozen_tree / RESULTS / "kth_protocol.json").exists()


@pytest.mark.parametrize(
    "target", ["checkpoint", "manifest", "metric", "frames", "archive"]
)
def test_verify_detects_changes_to_frozen_artifacts(frozen_tree, target):
    protocol = freeze_protocol("kth", root=frozen_tree)
    paths = {
        "checkpoint": frozen_tree / "checkpoints/video/kth/flow_map/model.pth",
        "manifest": frozen_tree / "data/video_forecasting/kth_64/manifest.json",
        "metric": Path(protocol["metric_artifacts"]["lpips_calibration"]["path"]),
        "frames": frozen_tree / "data/video_forecasting/kth_64/clip.npy",
        "archive": frozen_tree / RESULTS / "training/kth_flow_map/history.json",
    }
    path = paths[target]
    if not path.is_absolute():
        path = frozen_tree / path
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError):
        verify_frozen(frozen_tree, protocol)


def test_evaluation_verifier_checks_nfe_metrics_and_saved_feature_hashes(tmp_path):
    from scisi.bin.main_video_model_test import evaluate_video_model
    from report_video_results import QUALITY

    class Dataset:
        len_field_history = 2
        normalization = "minus_one_one"
        clips = [
            {"clip_id": "a", "source_member": "a"},
            {"clip_id": "b", "source_member": "b"},
        ]

        def __len__(self):
            return 2

        def __getitem__(self, index):
            import torch

            return {"x": torch.zeros(1, 4, 4, 4)}

    item = evaluate_video_model(
        None,
        Dataset(),
        num_steps=[1],
        ensemble_size=2,
        num_clips=2,
        forecast_horizon=2,
        device="cpu",
        selected_indices=[0, 1],
        output_dir=tmp_path,
        save_qualitative=0,
        feature_extractor=lambda videos: np.zeros((len(videos), 400)),
        metric_fn=lambda *args: {
            key: (None if key == "spread_skill_ratio" else 0.0) for key in QUALITY
        },
    )
    protocol = {
        key: item[key]
        for key in [
            "history_length",
            "forecast_horizon",
            "num_clips",
            "ensemble_size",
            "seed",
            "selected_indices",
            "feedback_clipping",
        ]
    }
    protocol.update(
        dataset="kth",
        split="test",
        resolution=64,
        frame_stride=1,
        manifest={"path": "manifest.json", "sha256": "manifest-hash"},
        persistence_steps=[1],
        cases=Dataset.clips,
        metric_artifacts={"i3d_google_conversion": {"sha256": "i3d-hash"}},
        models={"flow_map": {"config": {"sha256": "config-hash"}}},
    )
    item.update(
        dataset="kth",
        split="test",
        resolution=64,
        frame_stride=1,
        manifest_sha256="manifest-hash",
        data_manifest="manifest.json",
        i3d_sha256="i3d-hash",
        lpips={"version": "0.1", "network": "alex", "package": "0.1.4"},
        checkpoint=None,
        checkpoint_sha256=None,
        config_sha256="config-hash",
    )
    check = verify_evaluation(tmp_path, protocol, "persistence", item, tmp_path)
    assert check["features_verified"] and check["fvd_recomputed"]
    item["rows"][0]["nfe"] = 1
    with pytest.raises(ValueError, match="network evaluation count"):
        verify_evaluation(tmp_path, protocol, "persistence", item, tmp_path)
    item["rows"][0]["nfe"] = 0
    item["rows"][0]["lpips_sample_mean"] = None
    with pytest.raises(ValueError, match="Uncomputed metric"):
        verify_evaluation(tmp_path, protocol, "persistence", item, tmp_path)
    item["rows"][0]["lpips_sample_mean"] = 0.0
    path = tmp_path / "features_steps_1.npz"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="Feature artifact changed"):
        verify_evaluation(tmp_path, protocol, "persistence", item, tmp_path)
