#!/usr/bin/env python3
"""Audit the completed frozen three-seed confirmation, without running models.

Run from the repository root ONLY after run_confirmation.sh has finished:
  .venv/bin/python research/ito_maps/experiments/verify_confirmation.py

Checks scientific artifacts, not statistical superiority. The original source
hash list is not enforced: documented post-freeze API guards did not change NS
numerics. Original frozen model/config hashes are enforced without exceptions.
"""

import argparse
from datetime import datetime, timezone
from itertools import product
import json
import math
from pathlib import Path
import sys

from omegaconf import OmegaConf

# Support direct execution and importlib-based CPU tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from archive_training import hash_file, lineage
from report_results import METRICS, OPTIONAL_METRICS, load_evaluation, summarize


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def finite_numbers(value, location):
    """Reject NaN/inf even when hidden in raw rows or aggregate intervals."""
    if isinstance(value, float):
        require(math.isfinite(value), f"Nonfinite number at {location}")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_numbers(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_numbers(item, f"{location}[{index}]")


def validate_evaluation(data, specification, budgets, checkpoint, checkpoint_hash):
    """Check exact Cartesian cases, not merely a shared subset of cases."""
    path = data["source_path"]
    horizon = specification.get("horizon", 1)
    require(data["split"] == "test", f"Non-test evaluation: {path}")
    for key, expected in (
        ("ensemble_size", specification["ensemble_size"]),
        ("forecast_horizon", horizon),
        ("seed", specification["seed"]),
        ("coordinate_system", "dataset normalized coordinates"),
    ):
        require(data.get(key) == expected, f"{path}: {key} != {expected}")
    require(Path(data["checkpoint"]).resolve() == checkpoint.resolve(),
            f"Wrong evaluated checkpoint: {path}")
    require(data.get("checkpoint_sha256") == checkpoint_hash,
            f"Evaluated checkpoint hash no longer matches: {path}")
    expected = set(product(range(*specification["trajectories"]),
                           specification["origins"], range(1, horizon + 1), budgets))
    keys = [(r["trajectory"], r["origin"], r["lead"], r["num_steps"])
            for r in data["rows"]]
    require(len(keys) == len(set(keys)) and set(keys) == expected,
            f"Missing, duplicate, or extra forecast cases/budgets: {path}")
    finite_numbers(data["rows"], f"{path}.rows")
    finite_numbers(data.get("summary", []), f"{path}.summary")
    for row in data["rows"]:
        require(all(isinstance(row.get(m), (float, int))
                    for m in METRICS + OPTIONAL_METRICS),
                f"Missing/non-numeric forecast metric: {path}")
        require(row.get("ensemble_size") == specification["ensemble_size"],
                f"Per-case ensemble size mismatch: {path}")
        require(row["nfe"] == row["num_steps"], f"Unexpected NFE accounting: {path}")
        ranks = row.get("rank_counts", [])
        require(len(ranks) == specification["ensemble_size"] + 1
                and all(isinstance(n, int) and n >= 0 for n in ranks)
                and sum(ranks) == 128 * 128,
                f"Invalid NS pixel rank counts: {path}")
    return {"path": path, "sha256": data["source_sha256"],
            "checkpoint_sha256": checkpoint_hash, "rows": len(keys),
            "budgets": sorted(budgets), "horizon": horizon}


def compare_lead_one(one_step, rollout):
    """Identical per-context RNG implies exact score equality, excluding timing."""
    fields = tuple(m for m in METRICS + OPTIONAL_METRICS if m != "seconds") + ("rank_counts",)
    standalone = {(r["trajectory"], r["origin"], r["num_steps"]): r
                  for r in one_step["rows"] if r["lead"] == 1}
    matched = 0
    for row in rollout["rows"]:
        if row["lead"] != 1:
            continue
        key = row["trajectory"], row["origin"], row["num_steps"]
        require(key in standalone, f"Rollout lead-one case absent from standalone: {key}")
        for field in fields:
            require(standalone[key][field] == row[field],
                    f"Lead-one mismatch in {rollout['source_path']}: {key}, {field}")
        matched += 1
    return matched


def training_signature(config):
    """Settings allowed to differ across seeds are provenance/output paths only."""
    keys = ("len_field_history", "preprocesser", "train_data", "val_data",
            "test_data", "optimizer", "loss_fn", "scheduler", "model", "teacher")
    result = {key: config[key] for key in keys}
    result["trainer"] = {key: value for key, value in config["trainer"].items()
                         if key not in ("checkpoint_dir", "resume_from")}
    return result


def validate_training(directory, archive, seed, stage, teacher, reference, path_features):
    config = OmegaConf.to_container(OmegaConf.load(directory / "config.yaml"), resolve=True)
    metadata, history = read_json(directory / "metadata.json"), read_json(directory / "history.json")
    epochs, batches = (20, 100) if stage == "psd" else (10, 50)
    require(config["seed"] == metadata["seed"] == seed, f"Wrong training seed: {directory}")
    require(len(history) == epochs
            and [row["epoch"] for row in history] == list(range(epochs))
            and [row["step"] for row in history] == [batches * (i + 1) for i in range(epochs)],
            f"Incomplete/incorrect training updates: {directory}")
    finite_numbers(history, str(directory / "history.json"))
    require(all(isinstance(row.get(metric), (float, int))
                for row in history for metric in ("train_loss", "val_loss")),
            f"Missing/non-numeric training losses: {directory}")
    trainer = config["trainer"]
    require(Path(trainer["checkpoint_dir"]).resolve() == directory.resolve(),
            f"Training config records a different output directory: {directory}")
    expected_objective = "progressive" if stage == "psd" else "teacher_rollout"
    require(trainer["num_epochs"] == epochs and trainer["max_train_batches"] == batches
            and trainer["objective"] == expected_objective and trainer["teacher_steps"] == 32,
            f"Wrong stage budget/objective: {directory}")
    require(training_signature(config) == training_signature(reference),
            f"Training settings differ from seed zero: {directory}")
    require(Path(config["teacher"]["checkpoint_dir"]).resolve() == teacher.resolve()
            and Path(metadata["teacher_checkpoint"]).resolve() == teacher.resolve(),
            f"Wrong teacher lineage: {directory}")
    if directory.name.startswith("ito_"):
        require(config["model"]["num_path_features"] == path_features
                and config["model"]["map_model"]["path_channels"] == path_features,
                f"Wrong Brownian feature count: {directory}")
    initial = config.get("initial_student_checkpoint")
    expected_ancestors = [directory.resolve()]
    if stage == "rollout":
        parent = directory.with_name(directory.name.replace("_rollout_", "_psd_"))
        require(initial and Path(initial).resolve() == (parent / "model.pth").resolve()
                and metadata.get("initial_student_checkpoint") == initial,
                f"Wrong warm-start checkpoint: {directory}")
        expected_ancestors.append(parent.resolve())
    else:
        require(not initial and not metadata.get("initial_student_checkpoint"),
                f"PSD unexpectedly warm-started from a student: {directory}")
    provenance = read_json(archive / "checkpoint_provenance.json")
    require(Path(provenance["checkpoint"]).resolve() == (directory / "model.pth").resolve(),
            f"Archive records a different checkpoint: {archive}")
    actual_lineage = lineage(directory)
    require(provenance["initialization_lineage"] == actual_lineage
            and [Path(item["checkpoint_dir"]) for item in actual_lineage] == expected_ancestors,
            f"Stale or incorrect archived initialization lineage: {directory}")
    require(provenance["recorded_final_step"] == epochs * batches
            and provenance["completed_epochs"] == epochs
            and provenance["recorded_best_validation_loss"] == min(row["val_loss"] for row in history)
            and provenance["cumulative_recorded_student_updates"] == (2000 if stage == "psd" else 2500),
            f"Archived training budget mismatch: {directory}")
    for name in ("config.yaml", "metadata.json", "history.json"):
        require(hash_file(directory / name) == hash_file(archive / name)
                == provenance["metadata_artifact_sha256"][name],
                f"Stale or corrupted metadata archive: {archive / name}")
    for name in ("model.pth", "ema.pth"):
        weight = provenance["weight_artifacts"][name]
        require(hash_file(directory / name) == weight["sha256"]
                and (directory / name).stat().st_size == weight["size_bytes"],
                f"Stale archived checkpoint hash: {directory / name}")
    require(provenance["sha256"] == provenance["weight_artifacts"]["model.pth"]["sha256"],
            f"Inconsistent selected checkpoint hash: {archive}")
    return {"run": directory.name, "seed": seed, "stage": stage,
            "updates": epochs * batches, "cumulative_updates": provenance["cumulative_recorded_student_updates"],
            "checkpoint_sha256": provenance["sha256"], "config_sha256": hash_file(directory / "config.yaml"),
            "provenance_sha256": hash_file(archive / "checkpoint_provenance.json")}


def verify(protocol_path, results_root, checkpoint_root):
    protocol = read_json(protocol_path)
    require(protocol["training_seeds"] == [0, 1, 2], "Expected three frozen student seeds [0,1,2]")
    baseline_names = ("stochastic_interpolant", "flow_matching")
    student_names = [f"{kind}_rollout_seed{seed}" for seed in protocol["training_seeds"]
                     for kind in ("ito", "flow")]
    require(set(protocol["checkpoints"]) == set(baseline_names + tuple(student_names[:2])),
            "Frozen manifest must contain the original four selected checkpoints")
    frozen = []
    for name, item in protocol["checkpoints"].items():
        directory = Path(item["path"])
        require(directory.resolve() == (checkpoint_root / name).resolve(),
                f"Frozen checkpoint root mismatch: {name}")
        for filename, key in (("model.pth", "model_sha256"), ("config.yaml", "config_sha256")):
            require(hash_file(directory / filename) == item[key], f"Frozen artifact changed: {directory / filename}")
        frozen.append({"name": name, **item})
    evaluations, evaluation_records = {}, []
    for mode in ("one_step", "rollout"):
        specification = protocol[mode]
        for name in student_names + list(baseline_names):
            if name in baseline_names:
                budgets = specification["solver_steps"]
            elif mode == "one_step" and not name.endswith("seed0"):
                budgets = specification["replication_steps"]
            else:
                budgets = specification["map_steps"]
            checkpoint = checkpoint_root / name / "model.pth"
            data = load_evaluation(results_root / f"test_{mode}" / name / "evaluation.json")
            evaluation_records.append(validate_evaluation(data, specification, budgets, checkpoint, hash_file(checkpoint)))
            evaluations[mode, name] = data
        # Reuse the reporting pipeline's checks for matched comparison populations.
        for lead in range(1, specification.get("horizon", 1) + 1):
            names = student_names + list(baseline_names)
            summarize([evaluations[mode, name] for name in names], lead=lead,
                      labels=names, split="test", n_bootstrap=1)
    lead_matches = {name: compare_lead_one(evaluations["one_step", name], evaluations["rollout", name])
                    for name in student_names}
    training = []
    for kind, stage in product(("ito", "flow"), ("psd", "rollout")):
        reference = OmegaConf.to_container(OmegaConf.load(checkpoint_root / f"{kind}_{stage}_seed0" / "config.yaml"), resolve=True)
        teacher = checkpoint_root / ("stochastic_interpolant" if kind == "ito" else "flow_matching")
        for seed in protocol["training_seeds"]:
            name = f"{kind}_{stage}_seed{seed}"
            training.append(validate_training(checkpoint_root / name, results_root / "training" / name,
                                              seed, stage, teacher, reference, protocol["selected_path_features"]))
    return {"success": True, "checked_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_path": str(protocol_path.resolve()), "protocol_sha256": hash_file(protocol_path),
            "audit_script_sha256": hash_file(Path(__file__)),
            "frozen_artifacts": frozen, "evaluations": evaluation_records,
            "total_forecast_rows": sum(item["rows"] for item in evaluation_records),
            "exact_lead_one_matches_by_student": lead_matches, "training_runs": training,
            "scope": "Artifact integrity and matched frozen protocol; no test-informed model selection or assertion of statistical superiority.",
            "source_hash_policy": "Original model/config hashes enforced. Source hash list excluded because separately documented API guards were added after freezing without changing NS dynamics."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("research/ito_maps/results/frozen_protocol.json"))
    parser.add_argument("--results-root", type=Path, default=Path("research/ito_maps/results"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/stochastic_navier_stokes"))
    parser.add_argument("--output", type=Path, default=Path("research/ito_maps/results/confirmation_audit.json"))
    args = parser.parse_args()
    try:
        record = verify(args.protocol, args.results_root, args.checkpoint_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        record = {"success": False, "checked_utc": datetime.now(timezone.utc).isoformat(),
                  "error": f"{type(error).__name__}: {error}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(f"Confirmation audit {'passed' if record['success'] else 'FAILED'}: {args.output}")
    if not record["success"]:
        print(record["error"], file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
