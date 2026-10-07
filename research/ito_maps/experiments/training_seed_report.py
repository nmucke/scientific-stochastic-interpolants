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
from paired_comparison import compare

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


def fixed_teacher_comparisons(
    evaluations, selected, names, identities, steps, replicates, seed
):
    """Compare mean student scores, never pooled ensemble members, to fixed teachers."""
    teacher_classes = {
        "Föllmer Itô map": "FollmerStochasticInterpolant",
        "Flow map": "FlowMatchingModel",
    }
    results = []
    for name in dict.fromkeys(names):
        teachers = [
            d
            for d in evaluations
            if d.get("model_class", "").split(".")[-1] == teacher_classes[name]
        ]
        if not teachers:
            continue  # Map-only CLI inputs still support the original seed report.
        if len(teachers) != 1:
            raise ValueError(f"Expected one fixed teacher evaluation for {name}")
        teacher = teachers[0]
        if 64 not in {row["num_steps"] for row in teacher["rows"]}:
            raise ValueError(f"Fixed teacher comparison requires 64 steps for {name}")
        indices = [i for i, value in enumerate(names) if value == name]
        group = [selected[i] for i in indices]
        teacher_path = teacher.get("checkpoint")
        if teacher_path:
            actual_directory = Path(teacher_path).resolve().parent
            for index in indices:
                trained_with = identities[index][1]
                if trained_with:
                    expected_directory = Path(trained_with).resolve()
                    if expected_directory.suffix == ".pth":
                        expected_directory = expected_directory.parent
                    if expected_directory != actual_directory:
                        raise ValueError(
                            f"Evaluation does not use the training teacher for {name}"
                        )
        # summarize performs complete split/horizon/ensemble/case compatibility checks.
        for lead in range(1, group[0]["forecast_horizon"] + 1):
            summarize(
                group + [teacher],
                lead=lead,
                labels=[f"seed {identities[i][0]}" for i in indices] + ["teacher"],
                split="test",
                n_bootstrap=1,
                seed=seed,
            )
            for budget in steps:
                for metric in METRICS + OPTIONAL_METRICS:
                    if not all(
                        metric in row
                        for data in group + [teacher]
                        for row in data["rows"]
                        if row["lead"] == lead
                        and row["num_steps"] == (64 if data is teacher else budget)
                    ):
                        continue
                    # Pair each seed separately, then use its per-trajectory differences
                    # to average scores across training seeds before resampling data.
                    pairs = [
                        compare(
                            data,
                            teacher,
                            budget,
                            64,
                            metric=metric,
                            lead=lead,
                            replicates=1,
                            seed=seed,
                        )
                        for data in group
                    ]
                    differences = np.array(
                        [pair["per_trajectory_differences"] for pair in pairs]
                    )
                    seed_means = np.array([pair["candidate_mean"] for pair in pairs])
                    trajectory_mean = differences.mean(axis=0)
                    rng = np.random.default_rng(seed)
                    draws = rng.integers(
                        0, differences.shape[1], (replicates, differences.shape[1])
                    )
                    low, high = np.quantile(
                        trajectory_mean[draws].mean(axis=1), [0.025, 0.975]
                    )
                    reference = pairs[0]["reference_mean"]
                    result = {
                        "model": name,
                        "num_steps": budget,
                        "lead": lead,
                        "metric": metric,
                        "num_training_seeds": len(group),
                        "training_seeds": [identities[i][0] for i in indices],
                        "num_test_trajectories": differences.shape[1],
                        "student_mean": float(seed_means.mean()),
                        "student_seed_sd": (
                            float(seed_means.std(ddof=1)) if len(group) > 1 else None
                        ),
                        "student_seed_min": float(seed_means.min()),
                        "student_seed_max": float(seed_means.max()),
                        "reference_model": MODEL_NAMES[teacher_classes[name]],
                        "reference_steps": 64,
                        "reference_mean": reference,
                        "difference_student_mean_minus_reference": float(
                            trajectory_mean.mean()
                        ),
                        "difference_data_cluster_ci95_lower": float(low),
                        "difference_data_cluster_ci95_upper": float(high),
                        "relative_change_percent": (
                            float(100 * trajectory_mean.mean() / reference)
                            if reference
                            else None
                        ),
                        "reference_source": teacher["source_path"],
                        "reference_checkpoint_sha256": teacher.get("checkpoint_sha256"),
                        "student_sources": [data["source_path"] for data in group],
                        "student_checkpoint_sha256": [
                            data.get("checkpoint_sha256") for data in group
                        ],
                    }
                    results.append(result)
    return results


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
    comparisons = fixed_teacher_comparisons(
        evaluations, selected, names, identities, steps, replicates, seed
    )
    provenance["fixed_teacher_comparisons"] = comparisons
    if comparisons:
        with (output / "fixed_teacher_comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
        (output / "fixed_teacher_comparison.json").write_text(
            json.dumps(
                {
                    "comparisons": comparisons,
                    "bootstrap_replicates": replicates,
                    "bootstrap_seed": seed,
                    "uncertainty": "Paired trajectory bootstrap of the mean score over fixed trained students. Student-seed SD/range is reported separately. Ensembles are never pooled; teacher training uncertainty is not estimated.",
                },
                indent=2,
            )
        )
        table = [
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Model & Steps & Lead & Mean CRPS & Seed SD & Teacher CRPS & Paired difference [95\% data CI] \\",
            r"\midrule",
        ]
        for row in comparisons:
            if row["metric"] != "crps":
                continue
            sd = (
                "--"
                if row["student_seed_sd"] is None
                else f"{row['student_seed_sd']:.4f}"
            )
            table.append(
                f"{tex_escape(row['model'])} & {row['num_steps']} & {row['lead']} & {row['student_mean']:.4f} & {sd} & {row['reference_mean']:.4f} & {row['difference_student_mean_minus_reference']:+.4f} [{row['difference_data_cluster_ci95_lower']:+.4f}, {row['difference_data_cluster_ci95_upper']:+.4f}]"
                + r" \\"
            )
        table.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                "% Reference is the fixed 64-step teacher; data CIs exclude student and teacher training uncertainty.",
                "",
            ]
        )
        (output / "fixed_teacher_comparison.tex").write_text("\n".join(table))
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
