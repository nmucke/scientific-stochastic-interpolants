"""Orchestrate a frozen, matched one-step and ten-lead test report.

This reads completed JSON evaluations and never loads models or uses a GPU.
The default protocol is test trajectories 180--199, E=16, one-step origins
5/45/85, and a ten-lead rollout from origin 45. Incomplete protocols fail.

Example:
    python research/ito_maps/experiments/final_report.py \\
      --one-step RESULTS/one_step/*/evaluation.json \\
      --rollout RESULTS/rollout/*/evaluation.json --output RESULTS/final_report

Use --pair 'Föllmer Itô map' 8 'Föllmer SI' 64 to override the predeclared
comparisons. Labels are taken from model classes, with checkpoint names added
only to disambiguate duplicate classes. Freeze selections on validation first.
"""

import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paired_comparison import compare
from training_seed_report import generate as generate_seed_report, training_identity
from report_results import (
    COLORS,
    MODEL_NAMES,
    create_report,
    load_evaluation,
    summarize,
    tex_escape,
    write_table,
)


DEFAULT_PAIRS = [
    ("Föllmer Itô map", 1, "Föllmer SI", 16),
    ("Föllmer Itô map", 8, "Föllmer SI", 64),
    ("Föllmer Itô map", 8, "Föllmer SI", 128),
    ("Flow map", 1, "Flow matching", 64),
    ("Flow map", 8, "Flow matching", 64),
    ("Föllmer Itô map", 1, "Flow map", 1),
    ("Föllmer Itô map", 8, "Flow map", 8),
]


def labels_for(data):
    labels, used = [], set()
    for evaluation in data:
        name = evaluation.get("model_class", "").split(".")[-1]
        label = MODEL_NAMES.get(name, Path(evaluation["source_path"]).parent.name)
        if label in used:
            label += " (" + Path(evaluation["source_path"]).parent.name + ")"
        if label in used:
            raise ValueError("Cannot disambiguate duplicate model labels")
        labels.append(label)
        used.add(label)
    return labels


def check_protocol(
    evaluations,
    *,
    ensemble,
    trajectories,
    origins,
    horizon,
    rollout,
    additional_seed=False,
):
    expected_cases = {
        (trajectory, origin, lead)
        for trajectory in trajectories
        for origin in origins
        for lead in range(1, horizon + 1)
    }
    for data in evaluations:
        if (
            data["split"] != "test"
            or data["ensemble_size"] != ensemble
            or data["forecast_horizon"] != horizon
        ):
            raise ValueError(
                f"Input does not match frozen test split/ensemble/horizon: {data['source_path']}"
            )
        model_class = data.get("model_class", "").split(".")[-1]
        is_map = model_class in ("FollmerItoMapModel", "FlowMapModel")
        is_persistence = model_class == "PersistenceModel"
        if (
            not is_map
            and not is_persistence
            and model_class
            not in (
                "FollmerStochasticInterpolant",
                "FlowMatchingModel",
            )
        ):
            raise ValueError(f"Unrecognized benchmark model: {model_class}")
        expected_budgets = (
            ({1, 8} if is_map else {64})
            if rollout
            else ({1, 2, 4, 8} if is_map else {1, 4, 16, 64, 128})
        )
        if is_persistence:
            expected_budgets = {1}
        elif additional_seed:
            if not is_map:
                raise ValueError("Additional student-seed inputs must be learned maps")
            expected_budgets = {1, 8}
        budgets = {row["num_steps"] for row in data["rows"]}
        if budgets != expected_budgets:
            raise ValueError(
                f"Incomplete/unexpected budgets for {model_class}: {budgets}, expected {expected_budgets}"
            )
        for budget in budgets:
            rows = [row for row in data["rows"] if row["num_steps"] == budget]
            cases = {(row["trajectory"], row["origin"], row["lead"]) for row in rows}
            if cases != expected_cases or len(rows) != len(expected_cases):
                raise ValueError(
                    f"Incomplete, duplicate, or unexpected forecast cases for {data['source_path']}, steps={budget}"
                )
        if not is_persistence and not data.get("checkpoint_sha256"):
            raise ValueError(f"Missing frozen checkpoint hash in {data['source_path']}")


def cumulative_evaluations(evaluations):
    """Accumulate costs along each generated path before trajectory averaging."""
    result = copy.deepcopy(evaluations)
    for data in result:
        running = {}
        for row in sorted(data["rows"], key=lambda row: row["lead"]):
            key = (row["trajectory"], row["origin"], row["num_steps"])
            seconds, nfe = running.get(key, (0.0, 0.0))
            seconds += row["seconds"]
            nfe += row["nfe"]
            row["seconds"], row["nfe"] = seconds, nfe
            running[key] = seconds, nfe
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_rollout(rows, labels, metadata, output):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    ratios = (
        "rms_spread_skill"
        if all("rms_spread_skill" in row for row in rows)
        else "spread_skill"
    )
    metrics = [
        ("crps", "Fair CRPS ↓"),
        ("energy_score", "Fair energy score ↓"),
        ("rmse", "Ensemble-mean RMSE ↓"),
        ("coverage90", "Empirical 90% coverage"),
        (
            ratios,
            "RMS spread/skill" if ratios.startswith("rms") else "Legacy spread/skill",
        ),
        ("rank_extreme_fraction", "Extreme-rank fraction"),
    ]
    if not all("rank_extreme_fraction" in row for row in rows):
        metrics[-1] = ("spread", "Legacy ensemble spread")
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.0), layout="constrained")
    cost_fig, cost_axes = plt.subplots(1, 2, figsize=(7.2, 3.1), layout="constrained")
    for index, label in enumerate(labels):
        budgets = sorted({row["num_steps"] for row in rows if row["model"] == label})
        for budget in budgets:
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["model"] == label and row["num_steps"] == budget
                ],
                key=lambda row: row["lead"],
            )
            leads = np.array([row["lead"] for row in selected])
            color = COLORS[index % len(COLORS)]
            style = "--" if len(budgets) > 1 and budget == min(budgets) else "-"
            curve_label = (
                "Persistence (0 NFE)"
                if all(row["nfe"] == 0 for row in selected)
                else f"{label}, {budget} steps/transition"
            )
            for axis, (metric, ylabel) in zip(axes.flat, metrics):
                values = [row[metric] for row in selected]
                lower = [row[metric + "_ci95_lower"] for row in selected]
                upper = [row[metric + "_ci95_upper"] for row in selected]
                axis.plot(
                    leads, values, style, color=color, label=curve_label, linewidth=1.5
                )
                axis.fill_between(
                    leads, lower, upper, color=color, alpha=0.09, linewidth=0
                )
                axis.set(xlabel="Physical forecast lead", ylabel=ylabel, xticks=leads)
                axis.grid(alpha=0.2, linewidth=0.5)
            if all(row["nfe"] == 0 for row in selected):
                continue  # Persistence has no neural evaluations; do not compare its CPU latency.
            for axis, (metric, ylabel) in zip(
                cost_axes,
                [
                    ("cumulative_nfe", "Cumulative backbone evaluations"),
                    ("cumulative_seconds", "Cumulative ensemble latency (s)"),
                ],
            ):
                axis.plot(
                    leads,
                    [row[metric] for row in selected],
                    style,
                    color=color,
                    label=curve_label,
                )
                axis.fill_between(
                    leads,
                    [row[metric + "_ci95_lower"] for row in selected],
                    [row[metric + "_ci95_upper"] for row in selected],
                    color=color,
                    alpha=0.09,
                    linewidth=0,
                )
                axis.set(
                    xlabel="Physical forecast lead",
                    ylabel=ylabel,
                    xticks=leads,
                    yscale="log",
                )
                axis.grid(alpha=0.2, linewidth=0.5)
    for axis, (metric, _) in zip(axes.flat, metrics):
        if metric.endswith("spread_skill"):
            axis.axhline(1.0, color="0.5", linewidth=0.8, linestyle=":")
        if metric == "rank_extreme_fraction":
            axis.axhline(
                2 / (metadata["ensemble_size"] + 1),
                color="0.5",
                linewidth=0.8,
                linestyle=":",
            )
        # No 90% reference line: empirical finite-E quantiles are not population quantiles.
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        f"Test autoregression · E={metadata['ensemble_size']} · {len(metadata['trajectory_ids'])} trajectories · pointwise 95% CIs",
        fontsize=10,
    )
    cost_handles, cost_labels = cost_axes[0].get_legend_handles_labels()
    cost_fig.legend(
        cost_handles,
        cost_labels,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=7,
    )
    for figure, name in (
        (fig, "rollout_quality_calibration"),
        (cost_fig, "rollout_cumulative_cost"),
    ):
        figure.savefig(output / (name + ".pdf"), bbox_inches="tight")
        figure.savefig(output / (name + ".png"), bbox_inches="tight")
        plt.close(figure)


def paired_reports(
    evaluations,
    labels,
    pairs,
    leads,
    output,
    replicates,
    seed,
    skip_missing_budgets=False,
):
    lookup = dict(zip(labels, evaluations))
    result, skipped = [], []
    for candidate, candidate_steps, reference, reference_steps in pairs:
        if candidate not in lookup or reference not in lookup:
            raise ValueError(
                f"Paired comparison refers to an unavailable label: {candidate}, {reference}"
            )
        available_candidate = {row["num_steps"] for row in lookup[candidate]["rows"]}
        available_reference = {row["num_steps"] for row in lookup[reference]["rows"]}
        if (
            int(candidate_steps) not in available_candidate
            or int(reference_steps) not in available_reference
        ):
            if not skip_missing_budgets:
                raise ValueError(
                    f"Predeclared pair budget unavailable: {candidate} {candidate_steps}, {reference} {reference_steps}"
                )
            skipped.append(
                {
                    "candidate": candidate,
                    "candidate_steps": int(candidate_steps),
                    "reference": reference,
                    "reference_steps": int(reference_steps),
                    "reason": "One-step-only comparison: this budget is absent from the frozen rollout protocol",
                }
            )
            continue
        for lead in leads:
            pair = compare(
                lookup[candidate],
                lookup[reference],
                int(candidate_steps),
                int(reference_steps),
                metric="crps",
                lead=lead,
                replicates=replicates,
                seed=seed,
            )
            pair.update(candidate_label=candidate, reference_label=reference)
            result.append(pair)
    (output / "paired_crps.json").write_text(json.dumps(result, indent=2))
    (output / "paired_crps_skipped.json").write_text(json.dumps(skipped, indent=2))
    if result:
        write_csv(output / "paired_crps.csv", result)
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Comparison & Lead & CRPS difference & 95\% paired CI \\",
        r"\midrule",
    ]
    for row in result:
        label = f"{row['candidate_label']} ({row['candidate_steps']}) $-$ {row['reference_label']} ({row['reference_steps']})"
        # Escape labels separately so the intended mathematical minus survives.
        label = (
            tex_escape(f"{row['candidate_label']} ({row['candidate_steps']})")
            + " $-$ "
            + tex_escape(f"{row['reference_label']} ({row['reference_steps']})")
        )
        lower, upper = row["difference_ci95"]
        lines.append(
            f"{label} & {row['lead']} & {row['difference_candidate_minus_reference']:.4f} & [{lower:.4f}, {upper:.4f}]"
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "% Negative difference favors the first model; pointwise trajectory-bootstrap intervals.",
            "",
        ]
    )
    (output / "paired_crps.tex").write_text("\n".join(lines))
    return {"comparisons": result, "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-step", nargs="+", required=True)
    parser.add_argument("--rollout", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--seed-one-step",
        nargs="*",
        default=[],
        help="Additional student seeds 1/2, evaluated at budgets 1 and 8",
    )
    parser.add_argument(
        "--seed-rollout",
        nargs="*",
        default=[],
        help="Additional student-seed ten-lead evaluations",
    )
    parser.add_argument("--persistence-one-step")
    parser.add_argument("--persistence-rollout")
    parser.add_argument("--ensemble", type=int, default=16)
    parser.add_argument("--trajectory-start", type=int, default=180)
    parser.add_argument("--trajectory-count", type=int, default=20)
    parser.add_argument("--one-step-origins", nargs="+", type=int, default=[5, 45, 85])
    parser.add_argument("--rollout-origin", type=int, default=45)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument(
        "--pair",
        nargs=4,
        action="append",
        metavar=(
            "CANDIDATE_LABEL",
            "CANDIDATE_STEPS",
            "REFERENCE_LABEL",
            "REFERENCE_STEPS",
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=712)
    args = parser.parse_args()
    if args.horizon < 1 or args.trajectory_count < 1:
        parser.error("horizon and trajectory count must be positive")
    one = [load_evaluation(path) for path in args.one_step]
    rollout = [load_evaluation(path) for path in args.rollout]
    extra_one = [load_evaluation(path) for path in args.seed_one_step]
    extra_rollout = [load_evaluation(path) for path in args.seed_rollout]
    persistence_one = (
        load_evaluation(args.persistence_one_step)
        if args.persistence_one_step
        else None
    )
    persistence_rollout = (
        load_evaluation(args.persistence_rollout) if args.persistence_rollout else None
    )
    labels_one, labels_rollout = labels_for(one), labels_for(rollout)
    if set(labels_one) != set(labels_rollout):
        raise ValueError(
            "One-step and rollout reports must contain the same selected models"
        )
    trajectories = list(
        range(args.trajectory_start, args.trajectory_start + args.trajectory_count)
    )
    check_protocol(
        one,
        ensemble=args.ensemble,
        trajectories=trajectories,
        origins=args.one_step_origins,
        horizon=1,
        rollout=False,
    )
    check_protocol(
        rollout,
        ensemble=args.ensemble,
        trajectories=trajectories,
        origins=[args.rollout_origin],
        horizon=args.horizon,
        rollout=True,
    )
    check_protocol(
        extra_one,
        ensemble=args.ensemble,
        trajectories=trajectories,
        origins=args.one_step_origins,
        horizon=1,
        rollout=False,
        additional_seed=True,
    )
    check_protocol(
        extra_rollout,
        ensemble=args.ensemble,
        trajectories=trajectories,
        origins=[args.rollout_origin],
        horizon=args.horizon,
        rollout=True,
        additional_seed=True,
    )
    for persistence, origins, horizon, is_rollout in (
        (persistence_one, args.one_step_origins, 1, False),
        (persistence_rollout, [args.rollout_origin], args.horizon, True),
    ):
        if persistence is not None:
            check_protocol(
                [persistence],
                ensemble=args.ensemble,
                trajectories=trajectories,
                origins=origins,
                horizon=horizon,
                rollout=is_rollout,
            )
    map_hashes = {}
    for data in one + extra_one:
        if data.get("model_class", "").split(".")[-1] in (
            "FlowMapModel",
            "FollmerItoMapModel",
        ):
            map_hashes[(data["model_class"], training_identity(data)[0])] = data[
                "checkpoint_sha256"
            ]
    for data in rollout + extra_rollout:
        if data.get("model_class", "").split(".")[-1] in (
            "FlowMapModel",
            "FollmerItoMapModel",
        ):
            key = data["model_class"], training_identity(data)[0]
            if key in map_hashes and map_hashes[key] != data["checkpoint_sha256"]:
                raise ValueError(
                    f"Training seed checkpoint differs between one-step and rollout: {key}"
                )
    lookup_one = dict(zip(labels_one, one))
    for label, data in zip(labels_rollout, rollout):
        if lookup_one[label]["checkpoint_sha256"] != data["checkpoint_sha256"]:
            raise ValueError(
                f"Checkpoint changed between one-step and rollout evaluation: {label}"
            )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    one_output, rollout_output = output / "one_step", output / "rollout"
    rollout_output.mkdir(exist_ok=True)
    _, one_metadata = create_report(
        args.one_step,
        one_output,
        labels=labels_one,
        split="test",
        n_bootstrap=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    if persistence_one is not None:
        persistence_rows, _ = summarize(
            [persistence_one],
            labels=["Persistence"],
            split="test",
            n_bootstrap=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        )
        write_csv(one_output / "persistence_scores.csv", persistence_rows)
        write_table(
            one_output / "persistence_quality.tex",
            persistence_rows,
            one_metadata,
            [
                ("CRPS", "crps", 1.0),
                ("Energy score", "energy_score", 1.0),
                ("RMSE", "rmse", 1.0),
            ],
            "Deterministic persistence sanity baseline; no neural evaluations and no GPU speedup comparison.",
            "tab:persistence-quality",
        )
    seed_one_report = generate_seed_report(
        one + extra_one,
        one_output / "student_training_seeds",
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    seed_rollout_report = generate_seed_report(
        rollout + extra_rollout,
        rollout_output / "student_training_seeds",
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    cumulative = cumulative_evaluations(rollout)
    all_rows = []
    for lead in range(1, args.horizon + 1):
        rows, metadata = create_report(
            args.rollout,
            rollout_output / f"lead_{lead:02d}",
            lead=lead,
            labels=labels_rollout,
            split="test",
            n_bootstrap=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        )
        costs, _ = summarize(
            cumulative,
            lead=lead,
            labels=labels_rollout,
            split="test",
            n_bootstrap=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        )
        cost_lookup = {(row["model"], row["num_steps"]): row for row in costs}
        for row in rows:
            cost = cost_lookup[row["model"], row["num_steps"]]
            for metric in ("seconds", "nfe"):
                for suffix in ("", "_ci95_lower", "_ci95_upper"):
                    row["cumulative_" + metric + suffix] = cost[metric + suffix]
        if persistence_rollout is not None:
            persistence_rows, _ = summarize(
                [persistence_rollout],
                lead=lead,
                labels=["Persistence"],
                split="test",
                n_bootstrap=args.bootstrap_replicates,
                seed=args.bootstrap_seed,
            )
            # Kept in score plots/tables only; these CPU costs never enter the cost figure.
            rows.extend(persistence_rows)
        all_rows.extend(rows)
    write_csv(rollout_output / "all_leads.csv", all_rows)
    plot_rollout(
        all_rows,
        labels_rollout + (["Persistence"] if persistence_rollout is not None else []),
        metadata,
        rollout_output,
    )
    pairs = args.pair or DEFAULT_PAIRS
    one_pairs = paired_reports(
        one,
        labels_one,
        pairs,
        [1],
        one_output,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    rollout_pairs = paired_reports(
        rollout,
        labels_rollout,
        pairs,
        range(1, args.horizon + 1),
        rollout_output,
        args.bootstrap_replicates,
        args.bootstrap_seed,
        skip_missing_budgets=True,
    )
    manifest = {
        "protocol": vars(args),
        "selected_models": labels_one,
        "paired_comparisons": {"one_step": one_pairs, "rollout": rollout_pairs},
        "student_seed_sources": {
            "one_step": seed_one_report["sources"],
            "rollout": seed_rollout_report["sources"],
        },
        "persistence": {
            "one_step": args.persistence_one_step,
            "rollout": args.persistence_rollout,
            "excluded_from_latency_comparisons": True,
        },
        "one_step": one_metadata,
        "rollout": metadata,
        "rollout_cost_semantics": "Costs summed from lead 1 through each displayed lead within each origin before trajectory bootstrap",
        "interval_semantics": "Pointwise paired trajectory bootstrap, not simultaneous confidence bands or training-seed uncertainty",
        "checkpoint_hashes": {
            label: data["checkpoint_sha256"] for label, data in zip(labels_one, one)
        },
    }
    (output / "final_report_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"Prepared frozen test report with {len(one)} models and {args.horizon} autoregressive leads: {output}"
    )


if __name__ == "__main__":
    main()
