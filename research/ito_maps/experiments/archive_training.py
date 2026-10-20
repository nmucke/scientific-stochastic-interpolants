#!/usr/bin/env python3
"""Archive completed map-run metadata and hashes without copying model weights.

Example:
 .venv/bin/python research/ito_maps/experiments/archive_training.py \
   --checkpoint-dirs checkpoints/stochastic_navier_stokes/ito_rollout_seed0 \
   checkpoints/stochastic_navier_stokes/flow_rollout_seed0

Run after training completes. Re-running refreshes the lightweight archive and
hash record. Training lineage follows initial_student_checkpoint and optional
function-preserving feature-refinement provenance, including prior-stage cost.
Teacher pretraining cost is not inferred from student-stage artifacts.
"""

import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def lineage(checkpoint_dir):
    """Record distinct initialization ancestors without counting teachers."""
    stages = []
    visited = set()
    current = checkpoint_dir.resolve()
    while current not in visited:
        visited.add(current)
        history = read_json(current / "history.json", [])
        metadata = read_json(current / "metadata.json", {})
        refinement = read_json(current / "refinement_provenance.json", {})
        item = {"checkpoint_dir": str(current), "artifacts_available": current.exists()}
        if history:
            item.update(
                {
                    "kind": "training",
                    "recorded_updates": history[-1]["step"],
                    "recorded_epochs": len(history),
                    "recorded_training_seconds": history[-1].get("elapsed_seconds"),
                    "seed": metadata.get("seed"),
                }
            )
        elif refinement:
            item.update(
                {
                    "kind": "feature_refinement",
                    "recorded_updates": 0,
                    "source_num_path_features": refinement.get(
                        "source_num_path_features"
                    ),
                    "num_path_features": refinement.get("num_path_features"),
                }
            )
        else:
            item.update({"kind": "unrecorded_initialization", "recorded_updates": None})
        stages.append(item)
        initial = metadata.get("initial_student_checkpoint")
        if initial is None and (current / "config.yaml").exists():
            initial = OmegaConf.load(current / "config.yaml").get(
                "initial_student_checkpoint"
            )
        if refinement.get("source_checkpoint_dir"):
            current = Path(refinement["source_checkpoint_dir"]).resolve()
        elif initial:
            current = Path(initial).resolve().parent
        else:
            break
    else:
        raise ValueError("Initialization provenance contains a cycle")
    return stages


def archive_run(checkpoint_dir, output_root):
    source = Path(checkpoint_dir)
    required = [
        source / name
        for name in ("metadata.json", "history.json", "config.yaml", "model.pth")
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete training artifacts: {missing}")
    artifacts = [path for path in required if path.suffix != ".pth"]
    for name in ("refinement_provenance.json",):
        if (source / name).exists():
            artifacts.append(source / name)
    weight_files = [source / "model.pth"]
    if (source / "ema.pth").exists():
        weight_files.append(source / "ema.pth")
    tracked = artifacts + weight_files
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in tracked}
    snapshots = {path.name: path.read_bytes() for path in artifacts}
    history = json.loads(snapshots["history.json"])
    if not history:
        raise ValueError("Cannot archive a run without completed epoch history")
    weights = {
        path.name: {"sha256": hash_file(path), "size_bytes": path.stat().st_size}
        for path in weight_files
    }
    stages = lineage(source)
    after = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in tracked}
    if before != after:
        raise RuntimeError(
            "Training artifacts changed during archiving; retry after training completes"
        )
    output = Path(output_root) / source.name
    output.mkdir(parents=True, exist_ok=True)
    for name, content in snapshots.items():
        (output / name).write_bytes(content)
    recorded_times = [
        row.get("recorded_training_seconds")
        for row in stages
        if row.get("kind") == "training"
    ]
    record = {
        "checkpoint": str(source / "model.pth"),
        "sha256": weights["model.pth"]["sha256"],
        "weight_artifacts": weights,
        "metadata_artifact_sha256": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in snapshots.items()
        },
        "completed_epochs": len(history),
        "recorded_final_step": history[-1]["step"],
        "recorded_best_validation_loss": min(row["val_loss"] for row in history),
        "initialization_lineage": stages,
        "cumulative_recorded_student_updates": sum(
            row.get("recorded_updates") or 0 for row in stages
        ),
        "cumulative_recorded_student_stage_seconds": sum(
            value for value in recorded_times if value is not None
        ),
        "cost_caveat": "Recorded stage times include their teacher evaluations. Teacher pretraining cost and absent ancestor records remain unavailable; listed totals include only available student-stage records.",
        "weights_copied": False,
    }
    (output / "checkpoint_provenance.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    return output, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dirs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("research/ito_maps/results/training")
    )
    args = parser.parse_args()
    for directory in args.checkpoint_dirs:
        output, record = archive_run(directory, args.output_root)
        print(
            f"Archived {directory} -> {output}: {record['recorded_final_step']} stage updates, {record['cumulative_recorded_student_updates']} recorded cumulative updates",
            flush=True,
        )


if __name__ == "__main__":
    main()
