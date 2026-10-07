"""Paired trajectory-bootstrap differences for explicitly selected budgets.

Negative candidate-minus-reference differences favor the candidate for CRPS,
RMSE, and energy score. Selection must use validation before reporting test.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from report_results import load_evaluation, summarize, tex_escape


def compare(
    candidate,
    reference,
    candidate_steps,
    reference_steps,
    *,
    metric="crps",
    lead=1,
    replicates=10000,
    seed=712,
):
    # Apply the reporter's strict split/ensemble/horizon/case compatibility checks.
    _, metadata = summarize(
        [candidate, reference],
        lead=lead,
        labels=["Candidate", "Reference"],
        n_bootstrap=1,
    )

    def select(data, steps):
        rows = [
            row
            for row in data["rows"]
            if row["lead"] == lead and row["num_steps"] == steps
        ]
        if not rows:
            raise ValueError(f"Missing requested {steps}-step budget at lead {lead}")
        return rows

    candidate_rows, reference_rows = select(candidate, candidate_steps), select(
        reference, reference_steps
    )
    trajectories = metadata["trajectory_ids"]

    def means(rows, key):
        return np.asarray(
            [
                np.mean([row[key] for row in rows if row["trajectory"] == trajectory])
                for trajectory in trajectories
            ]
        )

    first, second = means(candidate_rows, metric), means(reference_rows, metric)
    if replicates < 1 or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Need positive bootstrap count and finite observed scores")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(trajectories), (replicates, len(trajectories)))
    candidate_boot, reference_boot = first[indices].mean(1), second[indices].mean(1)
    difference = first - second
    difference_boot = candidate_boot - reference_boot

    def ci(values):
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]

    result = {
        "metric": metric,
        "split": metadata["split"],
        "lead": lead,
        "ensemble_size": metadata["ensemble_size"],
        "trajectory_ids": trajectories,
        "num_trajectories": len(trajectories),
        "origins": sorted({row["origin"] for row in candidate_rows}),
        "candidate_steps": candidate_steps,
        "reference_steps": reference_steps,
        "candidate_mean": float(first.mean()),
        "candidate_ci95": ci(candidate_boot),
        "reference_mean": float(second.mean()),
        "reference_ci95": ci(reference_boot),
        "difference_candidate_minus_reference": float(difference.mean()),
        "difference_ci95": ci(difference_boot),
        "per_trajectory_differences": difference.tolist(),
        "candidate_nfe": float(means(candidate_rows, "nfe").mean()),
        "reference_nfe": float(means(reference_rows, "nfe").mean()),
        "candidate_seconds": float(means(candidate_rows, "seconds").mean()),
        "reference_seconds": float(means(reference_rows, "seconds").mean()),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "bootstrap_unit": "paired trajectory; origins averaged before resampling",
        "candidate_source": candidate["source_path"],
        "reference_source": reference["source_path"],
        "candidate_checkpoint_sha256": candidate.get("checkpoint_sha256"),
        "reference_checkpoint_sha256": reference.get("checkpoint_sha256"),
        "limitations": "Paired intervals exclude training-seed variability; small trajectory counts yield coarse intervals.",
    }
    if second.mean() > 0 and np.all(reference_boot > 0):
        result["relative_change_percent"] = float(
            100 * (first.mean() / second.mean() - 1)
        )
        result["relative_change_percent_ci95"] = ci(
            100 * (candidate_boot / reference_boot - 1)
        )
    if result["candidate_seconds"] > 0:
        result["latency_speedup_reference_over_candidate"] = (
            result["reference_seconds"] / result["candidate_seconds"]
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate-steps", type=int, required=True)
    parser.add_argument("--reference-steps", type=int, required=True)
    parser.add_argument("--candidate-label", default="Candidate")
    parser.add_argument("--reference-label", default="Reference")
    parser.add_argument(
        "--metric", choices=["crps", "energy_score", "rmse"], default="crps"
    )
    parser.add_argument("--lead", type=int, default=1)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=712)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(
        load_evaluation(args.candidate),
        load_evaluation(args.reference),
        args.candidate_steps,
        args.reference_steps,
        metric=args.metric,
        lead=args.lead,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    result.update(
        candidate_label=args.candidate_label, reference_label=args.reference_label
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_comparison.json").write_text(json.dumps(result, indent=2))
    with (output / "paired_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)

    def estimate(mean, interval):
        return f"{mean:.4f} [{interval[0]:.4f}, {interval[1]:.4f}]"

    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        f"Method & NFE & {tex_escape(args.metric.upper())}" + r" \\",
        r"\midrule",
        tex_escape(args.candidate_label)
        + f" & {result['candidate_nfe']:g} & "
        + estimate(result["candidate_mean"], result["candidate_ci95"])
        + r" \\",
        tex_escape(args.reference_label)
        + f" & {result['reference_nfe']:g} & "
        + estimate(result["reference_mean"], result["reference_ci95"])
        + r" \\",
        r"\midrule",
        r"Candidate $-$ reference & -- & "
        + estimate(
            result["difference_candidate_minus_reference"], result["difference_ci95"]
        )
        + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        f"% {result['split']} split; E={result['ensemble_size']}; lead={args.lead}; {result['num_trajectories']} trajectories.",
        "% 95% paired trajectory-bootstrap intervals; negative difference favors candidate.",
        "",
    ]
    (output / "paired_comparison.tex").write_text("\n".join(lines))
    print(
        f"{result['split']} {args.metric}: candidate-reference "
        + estimate(
            result["difference_candidate_minus_reference"], result["difference_ci95"]
        )
    )


if __name__ == "__main__":
    main()
