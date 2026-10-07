"""Separate student-training variation from test-trajectory uncertainty.

Training seed IDs are read from checkpoint training metadata/configuration,
never from the evaluation sampling seed. Across-seed sample SD/range describes
student training with fixed teachers; it does not estimate teacher uncertainty.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from report_results import (
    METRICS,
    OPTIONAL_METRICS,
    MODEL_NAMES,
    load_evaluation,
    summarize,
    tex_escape,
)


def training_identity(data):
    checkpoint = Path(data["checkpoint"]).parent if data.get("checkpoint") else None
    metadata = {}
    if checkpoint is not None and (checkpoint / "metadata.json").exists():
        metadata = json.loads((checkpoint / "metadata.json").read_text())
    seed = data.get("training_seed", metadata.get("seed"))
    teacher = metadata.get("teacher_checkpoint")
    if (
        seed is None
        and checkpoint is not None
        and (checkpoint / "config.yaml").exists()
    ):
        cfg = OmegaConf.load(checkpoint / "config.yaml")
        seed = cfg.get("seed")
        teacher = teacher or cfg.get("teacher", {}).get("checkpoint_dir")
    if seed is None:
        raise ValueError(
            f"Training seed is missing for {data['source_path']}; do not substitute the evaluation sampling seed"
        )
    return int(seed), teacher


def generate(evaluations, output, *, steps=(1, 8), replicates=10000, seed=712):
    selected = [
        data
        for data in evaluations
        if data.get("model_class", "").split(".")[-1]
        in ("FlowMapModel", "FollmerItoMapModel")
    ]
    if not selected:
        raise ValueError(
            "Training-seed report needs at least one learned map evaluation"
        )
    identities = [training_identity(data) for data in selected]
    names = [MODEL_NAMES[data["model_class"].split(".")[-1]] for data in selected]
    labels = [
        f"{name} [student seed {identity[0]}]"
        for name, identity in zip(names, identities)
    ]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate model/training-seed inputs")
    for name in set(names):
        group = [data for data, n in zip(selected, names) if n == name]
        teacher_paths = {
            identity[1] for identity, n in zip(identities, names) if n == name
        }
        if len(teacher_paths) > 1:
            raise ValueError(
                f"Teacher checkpoints differ across student training seeds for {name}"
            )
        counts = {
            data.get("parameters")
            for data in group
            if data.get("parameters") is not None
        }
        if len(counts) > 1:
            raise ValueError(
                f"Parameter counts differ across seeds for {name}; freeze architecture before test"
            )
    horizon = selected[0]["forecast_horizon"]
    per_seed, across_seed = [], []
    sources = []
    for data, name, identity in zip(selected, names, identities):
        sources.append(
            {
                "model": name,
                "training_seed": identity[0],
                "fixed_teacher_checkpoint": identity[1],
                "source_path": data["source_path"],
                "checkpoint_sha256": data.get("checkpoint_sha256"),
                "sampling_seed": data.get("seed"),
            }
        )
        budgets = {row["num_steps"] for row in data["rows"]}
        if not set(steps) <= budgets:
            raise ValueError(
                f"Missing selected seed-comparison budgets in {data['source_path']}"
            )
    for lead in range(1, horizon + 1):
        summaries, metadata = summarize(
            selected,
            lead=lead,
            labels=labels,
            split="test",
            n_bootstrap=replicates,
            seed=seed,
        )
        for row in summaries:
            if row["num_steps"] not in steps:
                continue
            index = labels.index(row["model"])
            row = dict(row, model=names[index], training_seed=identities[index][0])
            per_seed.append(row)
        trajectories = metadata["trajectory_ids"]
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(trajectories), (replicates, len(trajectories)))
        for name in dict.fromkeys(names):
            group_indices = [index for index, n in enumerate(names) if n == name]
            for budget in steps:
                result = {
                    "model": name,
                    "num_steps": budget,
                    "lead": lead,
                    "num_training_seeds": len(group_indices),
                    "num_test_trajectories": len(trajectories),
                    "training_seeds": [identities[index][0] for index in group_indices],
                }
                for metric in METRICS + OPTIONAL_METRICS:
                    arrays = []
                    for index in group_indices:
                        rows = [
                            row
                            for row in selected[index]["rows"]
                            if row["lead"] == lead and row["num_steps"] == budget
                        ]
                        if not all(metric in row for row in rows):
                            break
                        arrays.append(
                            [
                                np.mean(
                                    [
                                        row[metric]
                                        for row in rows
                                        if row["trajectory"] == trajectory
                                    ]
                                )
                                for trajectory in trajectories
                            ]
                        )
                    if len(arrays) != len(group_indices):
                        continue
                    array = np.asarray(arrays)
                    means = array.mean(axis=1)
                    by_trajectory = array.mean(axis=0)
                    boot = by_trajectory[indices].mean(axis=1)
                    result[metric] = float(means.mean())
                    result[metric + "_student_seed_sd"] = (
                        float(means.std(ddof=1)) if len(means) > 1 else None
                    )
                    result[metric + "_student_seed_min"] = float(means.min())
                    result[metric + "_student_seed_max"] = float(means.max())
                    result[metric + "_data_cluster_ci95_lower"] = float(
                        np.quantile(boot, 0.025)
                    )
                    result[metric + "_data_cluster_ci95_upper"] = float(
                        np.quantile(boot, 0.975)
                    )
                across_seed.append(result)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for basename, rows in (
        ("per_student_seed", per_seed),
        ("student_seed_variation", across_seed),
    ):
        with (output / (basename + ".csv")).open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(dict.fromkeys(key for row in rows for key in row)),
            )
            writer.writeheader()
            writer.writerows(rows)
    provenance = {
        "sources": sources,
        "steps": list(steps),
        "forecast_horizon": horizon,
        "split": "test",
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "uncertainty": "Per-seed CIs and seed-mean data-cluster CIs resample test trajectories. Student-seed SD/range describes only these trained students with fixed teachers. No teacher-training uncertainty or simultaneous lead confidence is estimated.",
        "single_seed": "Student-seed SD is unreported (null) when only seed 0 is available.",
        "across_seed": across_seed,
    }
    (output / "training_seed_report.json").write_text(json.dumps(provenance, indent=2))
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & Steps & Lead & Seeds & Mean CRPS & Student-seed SD \\",
        r"\midrule",
    ]
    for row in across_seed:
        sd = row["crps_student_seed_sd"]
        sd_text = "--" if sd is None else f"{sd:.4f}"
        lines.append(
            f"{tex_escape(row['model'])} & {row['num_steps']} & {row['lead']} & {row['num_training_seeds']} & {row['crps']:.4f} & {sd_text}"
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "% Teachers are fixed; data-cluster confidence intervals are in the companion CSV.",
            "",
        ]
    )
    (output / "student_seed_variation.tex").write_text("\n".join(lines))
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=712)
    args = parser.parse_args()
    generate(
        [load_evaluation(path) for path in args.inputs],
        args.output,
        steps=args.steps,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
