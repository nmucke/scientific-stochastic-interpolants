#!/usr/bin/env python
"""Freeze trained video checkpoints and held-out cases before evaluation begins."""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

from scisi.bin.main_video_model_test import select_cases
from scisi.data.video_datasets import VideoFrameDataset

ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path("research/video_forecasting/results")
SETTINGS = {
    "kth": {"history": 10, "horizon": 30, "channels": 1, "frame_stride": 1},
    "clevrer": {"history": 2, "horizon": 14, "channels": 3, "frame_stride": 5},
}
FAMILIES = {
    "follmer_stochastic_interpolant": "FollmerStochasticInterpolant",
    "flow_matching": "FlowMatchingModel",
    "follmer_ito_map": "FollmerItoMapModel",
    "flow_map": "FlowMapModel",
}
TEACHERS = {
    "follmer_ito_map": "follmer_stochastic_interpolant",
    "flow_map": "flow_matching",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(root, path):
    value = Path(path)
    return value if value.is_absolute() else Path(root) / value


def portable(root, path):
    path = Path(path).resolve()
    return (
        str(path.relative_to(Path(root).resolve()))
        if path.is_relative_to(Path(root).resolve())
        else str(path)
    )


def artifact(root, path):
    path = resolve(root, path)
    return {
        "path": portable(root, path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def inspect_stage(root, dataset, stage):
    """Validate the prespecified training budget and teacher/student lineage."""
    settings = SETTINGS[dataset]
    directory = Path(root) / "checkpoints/video" / dataset / stage
    config = OmegaConf.load(directory / "config.yaml")
    metadata = read_json(directory / "metadata.json")
    history = read_json(directory / "history.json")
    is_psd = stage.endswith("_psd")
    family = stage.removesuffix("_psd")
    is_map = family in TEACHERS
    expected_updates = 2000 if is_psd else (500 if is_map else 20000)
    per_epoch = 50 if is_map and not is_psd else 100
    require(
        history and history[-1]["step"] == expected_updates,
        f"Incomplete {dataset}/{stage}: expected {expected_updates} updates",
    )
    require(
        len(history) == expected_updates // per_epoch
        and [r["step"] for r in history]
        == list(range(per_epoch, expected_updates + 1, per_epoch)),
        f"Unexpected epoch/update sequence: {stage}",
    )
    for row in history:
        require(
            all(
                math.isfinite(float(row[key]))
                for key in ("train_loss", "val_loss", "elapsed_seconds")
            ),
            f"Nonfinite training history: {stage}",
        )
    require(
        all(
            b["elapsed_seconds"] >= a["elapsed_seconds"]
            for a, b in zip(history, history[1:])
        ),
        f"Nonmonotonic training cost: {stage}",
    )
    require(
        config.model._target_.endswith(FAMILIES[family]),
        f"Model family mismatch: {stage}",
    )
    require(
        int(config.seed) == int(metadata["seed"]) == 0,
        f"Training seed mismatch: {stage}",
    )
    require(
        config.dataset_name == dataset
        and int(config.channels) == settings["channels"]
        and int(config.len_field_history) == settings["history"],
        f"Dataset/history mismatch: {stage}",
    )
    expected_manifest = (
        Path(root) / f"data/video_forecasting/{dataset}_64/manifest.json"
    )
    require(
        resolve(root, config.manifest_path).resolve() == expected_manifest.resolve(),
        f"Manifest mismatch: {stage}",
    )
    require(config.preprocesser is None, f"Unexpected learned preprocessing: {stage}")
    require(
        config.train_data.dataset.split == "train"
        and config.val_data.dataset.split == "validation",
        f"Invalid train/validation split: {stage}",
    )
    require(
        int(config.trainer.num_epochs) * int(config.trainer.max_train_batches)
        == expected_updates,
        f"Configured budget mismatch: {stage}",
    )
    require(
        resolve(root, config.trainer.checkpoint_dir).resolve() == directory.resolve(),
        f"Checkpoint destination mismatch: {stage}",
    )
    if is_map:
        teacher = Path(root) / "checkpoints/video" / dataset / TEACHERS[family]
        require(
            resolve(root, config.teacher.checkpoint_dir).resolve() == teacher.resolve(),
            f"Wrong teacher lineage: {stage}",
        )
        require(
            config.teacher.initialize_student is True,
            f"Teacher initialization disabled: {stage}",
        )
        require(
            resolve(root, metadata["teacher_checkpoint"]).resolve()
            == teacher.resolve(),
            f"Metadata teacher lineage mismatch: {stage}",
        )
        require(
            config.trainer.objective
            == ("progressive" if is_psd else "teacher_rollout"),
            f"Wrong map objective: {stage}",
        )
        require(
            int(config.trainer.teacher_steps) == 32,
            f"Wrong distillation teacher budget: {stage}",
        )
        initial = config.get("initial_student_checkpoint")
        if is_psd:
            require(
                initial is None and metadata.get("initial_student_checkpoint") is None,
                f"Unexpected PSD student initialization: {stage}",
            )
        else:
            expected_initial = directory.parent / f"{family}_psd/model.pth"
            require(
                initial is not None
                and resolve(root, initial).resolve() == expected_initial.resolve(),
                f"Wrong PSD-to-refinement lineage: {stage}",
            )
            require(
                resolve(root, metadata["initial_student_checkpoint"]).resolve()
                == expected_initial.resolve(),
                f"Metadata refinement lineage mismatch: {stage}",
            )
    else:
        require(
            config.get("initial_student_checkpoint") is None
            and metadata.get("initialization") == "random",
            f"Teacher was not trained from scratch: {stage}",
        )
    names = ["config.yaml", "metadata.json", "history.json", "model.pth"]
    if (directory / "ema.pth").exists():
        names.append("ema.pth")
    snapshots = {name: artifact(root, directory / name) for name in names}
    archive = Path(root) / RESULTS / "training" / f"{dataset}_{stage}"
    archive.mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "metadata.json", "history.json"):
        data = (directory / name).read_bytes()
        require(
            hashlib.sha256(data).hexdigest() == snapshots[name]["sha256"],
            f"Training changed while archiving: {stage}",
        )
        (archive / name).write_bytes(data)
    record = {
        "stage": stage,
        "model_class": FAMILIES[family],
        "seed": 0,
        "recorded_updates": expected_updates,
        "recorded_epochs": len(history),
        "recorded_stage_seconds": history[-1]["elapsed_seconds"],
        "minimum_recorded_validation_loss": min(row["val_loss"] for row in history),
        "teacher_checkpoint": metadata.get("teacher_checkpoint"),
        "initial_student_checkpoint": metadata.get("initial_student_checkpoint"),
        "artifacts": snapshots,
        "weights_copied": False,
        "cost_scope": "This stage's training and validation wall time; excludes initialization and data loading before trainer creation.",
    }
    (archive / "checkpoint_provenance.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    return record


def verify_frozen(root, protocol, verify_frames=True):
    """Check frozen original artifacts and lightweight archive copies."""
    records = [protocol["manifest"], *protocol["metric_artifacts"].values()]
    for stage in protocol["training_stages"].values():
        records.extend(stage["artifacts"].values())
    for record in records:
        path = resolve(root, record["path"])
        require(
            path.exists() and sha256(path) == record["sha256"],
            f"Frozen artifact changed or missing: {path}",
        )
    for name, stage in protocol["training_stages"].items():
        archive = Path(root) / RESULTS / "training" / f"{protocol['dataset']}_{name}"
        for filename in ("config.yaml", "metadata.json", "history.json"):
            require(
                sha256(archive / filename) == stage["artifacts"][filename]["sha256"],
                f"Training archive changed: {archive / filename}",
            )
    if verify_frames:
        manifest_dir = resolve(root, protocol["manifest"]["path"]).parent
        for case in protocol["cases"]:
            require(
                sha256(manifest_dir / case["path"]) == case["sha256"],
                f"Selected video data changed: {case['clip_id']}",
            )
    return protocol


def freeze_protocol(dataset, root=ROOT, verify_only=False):
    root = Path(root).resolve()
    require(dataset in SETTINGS, f"Unknown dataset: {dataset}")
    destination = root / RESULTS / f"{dataset}_protocol.json"
    if verify_only:
        return verify_frozen(root, read_json(destination))
    if destination.exists():
        raise FileExistsError(
            f"Immutable protocol exists: {destination}; use --verify-only"
        )
    evaluation_dir = root / RESULTS / "test" / dataset
    require(
        not evaluation_dir.exists() or not any(evaluation_dir.rglob("*.json")),
        "Held-out evaluation artifacts already exist; cannot retroactively freeze the protocol",
    )
    settings = SETTINGS[dataset]
    manifest_path = root / f"data/video_forecasting/{dataset}_64/manifest.json"
    dataset_object = VideoFrameDataset(
        manifest_path,
        split="test",
        len_field_history=settings["history"],
        train_or_test="test",
        horizon=settings["horizon"],
    )
    manifest = dataset_object.manifest
    require(
        manifest["dataset"] == dataset
        and manifest["resolution"] == 64
        and manifest["frame_stride"] == settings["frame_stride"]
        and manifest["channels"] == settings["channels"],
        "Dataset preparation differs from prespecified pixel-space protocol",
    )
    indices = select_cases(dataset_object, 128, 20261007)
    stages = list(FAMILIES) + ["follmer_ito_map_psd", "flow_map_psd"]
    training = {stage: inspect_stage(root, dataset, stage) for stage in stages}
    metric_record = read_json(
        root / "research/video_forecasting/notes/metric_weights.json"
    )
    metric_artifacts = {}
    for name in ("i3d_google_conversion", "alexnet_imagenet", "lpips_calibration"):
        expected = metric_record["artifacts"][name]
        actual = artifact(root, expected["path"])
        require(
            actual["sha256"] == expected["sha256"],
            f"Metric weights differ from provenance: {name}",
        )
        metric_artifacts[name] = actual
    selected = {
        family: {
            "model_class": model_class,
            "num_steps": [1, 8] if family in TEACHERS else [16, 64],
            "checkpoint": training[family]["artifacts"]["model.pth"],
            "config": training[family]["artifacts"]["config.yaml"],
        }
        for family, model_class in FAMILIES.items()
    }
    protocol = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "study_type": "initial pixel-space pilot",
        "training_seed": 0,
        "split": "test",
        "resolution": 64,
        "frame_stride": settings["frame_stride"],
        "history_length": settings["history"],
        "forecast_horizon": settings["horizon"],
        "num_clips": 128,
        "ensemble_size": 4,
        "seed": 20261007,
        "normalization": "minus_one_one",
        "feedback_clipping": False,
        "manifest": artifact(root, manifest_path),
        "selected_indices": indices,
        "cases": [dataset_object.clips[i] for i in indices],
        "models": selected,
        "persistence_steps": [1],
        "metric_artifacts": metric_artifacts,
        "metric_versions": metric_record.get("versions", {}),
        "training_stages": training,
        "checkpoint_selection": "Each stage uses the best fixed-noise validation checkpoint under its early-stopping improvement threshold; no held-out test selection.",
        "test_evaluation_started_at_freeze": False,
    }
    verify_frozen(root, protocol)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation enforces immutability even if two callers race.
    with destination.open("x") as stream:
        json.dump(protocol, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=SETTINGS)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = freeze_protocol(args.dataset, verify_only=args.verify_only)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "status": "verified" if args.verify_only else "frozen",
                "cases": len(result["cases"]),
                "training_stages": len(result["training_stages"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
