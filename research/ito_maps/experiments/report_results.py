"""Produce auditable figures and tables from completed forecast evaluations.

Example:
    python research/ito_maps/experiments/report_results.py \\
      --inputs research/ito_maps/results/validation_baselines/*/evaluation.json \\
      --output research/ito_maps/results/validation_baselines/report

Only raw observed forecast scores are used. Different evaluation cases, splits,
ensemble sizes, coordinates or horizons are rejected instead of pooled silently.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

METRICS = (
    "crps",
    "rmse",
    "energy_score",
    "spread",
    "spread_skill",
    "coverage90",
    "relative_enstrophy_error",
    "seconds",
    "nfe",
)
OPTIONAL_METRICS = ("rms_spread", "rms_spread_skill", "rank_extreme_fraction")

MODEL_NAMES = {
    "FollmerStochasticInterpolant": "Föllmer SI",
    "FlowMatchingModel": "Flow matching",
    "FollmerItoMapModel": "Föllmer Itô map",
    "FlowMapModel": "Flow map",
}
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
MARKERS = ("o", "s", "^", "D", "v", "P")


def load_evaluation(path):
    path = Path(path)
    if path.is_dir():
        path = path / "evaluation.json"
    data = json.loads(path.read_text())
    if not data.get("rows"):
        raise ValueError(f"No raw forecast rows in {path}")
    split = data.get("split")
    if split is None:
        command = path.parent.parent / "command.json"
        if command.exists():
            split = json.loads(command.read_text()).get("split")
    if split is None:
        # Exact standard split IDs are legitimate provenance, unlike filenames.
        ids = {int(row["trajectory"]) for row in data["rows"]}
        if ids and ids <= set(range(160, 180)):
            split = "validation"
        elif ids and ids <= set(range(180, 200)):
            split = "test"
    data["split"] = split or "unspecified"
    data["source_path"] = str(path.resolve())
    data["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if data.get("parameters") is None and data.get("checkpoint"):
        metadata = Path(data["checkpoint"]).parent / "metadata.json"
        if metadata.exists():
            data["parameters"] = json.loads(metadata.read_text()).get("parameters")
    return data


def summarize(
    evaluations, *, lead=1, labels=None, split=None, n_bootstrap=2000, seed=712
):
    """Cluster-bootstrap trajectories after averaging origins within each one."""
    if n_bootstrap < 1 or lead < 1:
        raise ValueError("lead and n_bootstrap must be positive")
    if labels is not None and len(labels) != len(evaluations):
        raise ValueError("Provide exactly one label per input")
    if not evaluations:
        raise ValueError("At least one evaluation is required")
    selected = []
    common_signature = None
    common_cases = None
    used_labels = set()
    for index, data in enumerate(evaluations):
        if split is not None and data["split"] != split:
            raise ValueError(
                f"Expected split {split}, found {data['split']} in {data['source_path']}"
            )
        signature = (
            data["split"],
            data.get("ensemble_size"),
            data.get("forecast_horizon"),
            data.get("coordinate_system"),
        )
        if signature[0] == "unspecified":
            raise ValueError(
                f"Split provenance is unspecified in {data['source_path']}"
            )
        if common_signature is None:
            common_signature = signature
        elif signature != common_signature:
            raise ValueError(
                f"Incompatible split/ensemble/horizon/coordinates: {signature} versus {common_signature}"
            )
        rows = [row for row in data["rows"] if row["lead"] == lead]
        if not rows:
            raise ValueError(f"Lead {lead} absent from {data['source_path']}")
        name = data.get("model_class", "").split(".")[-1]
        label = (
            labels[index]
            if labels is not None
            else MODEL_NAMES.get(name, Path(data["source_path"]).parent.name)
        )
        if label in used_labels:
            if labels is not None:
                raise ValueError("Explicit model labels must be unique")
            label += " (" + Path(data["source_path"]).parent.name + ")"
        if label in used_labels:
            raise ValueError("Ambiguous duplicate input labels; specify --labels")
        used_labels.add(label)
        for steps in sorted({row["num_steps"] for row in rows}):
            items = [row for row in rows if row["num_steps"] == steps]
            cases = {(row["trajectory"], row["origin"]) for row in items}
            if len(cases) != len(items):
                raise ValueError(f"Duplicate forecast cases for {label}, steps={steps}")
            if common_cases is None:
                common_cases = cases
            elif cases != common_cases:
                raise ValueError(
                    f"Different trajectory/origin cases for {label}, steps={steps}; select matched evaluations"
                )
            if any(
                row.get("ensemble_size", signature[1]) != signature[1] for row in items
            ):
                raise ValueError(
                    "Per-row ensemble size disagrees with evaluation metadata"
                )
            trajectories = sorted({row["trajectory"] for row in items})
            # Reuse the same resampling indices for every model, budget, metric.
            rng = np.random.default_rng(seed)
            draws = rng.integers(
                0, len(trajectories), size=(n_bootstrap, len(trajectories))
            )
            summary = {
                "model": label,
                "split": data["split"],
                "lead": lead,
                "ensemble_size": signature[1],
                "forecast_horizon": signature[2],
                "num_steps": steps,
                "num_trajectories": len(trajectories),
                "num_forecasts": len(items),
                "parameters": data.get("parameters"),
                "source_path": data["source_path"],
                "checkpoint": data.get("checkpoint"),
                "checkpoint_sha256": data.get("checkpoint_sha256"),
                "source_sha256": data["source_sha256"],
            }
            for metric in METRICS + tuple(
                metric
                for metric in OPTIONAL_METRICS
                if all(metric in row for row in items)
            ):
                values = np.asarray(
                    [
                        np.mean(
                            [
                                row[metric]
                                for row in items
                                if row["trajectory"] == trajectory
                            ]
                        )
                        for trajectory in trajectories
                    ]
                )
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"Nonfinite {metric} for {label}, steps={steps}; investigate before reporting"
                    )
                boot = values[draws].mean(axis=1)
                summary[metric] = float(values.mean())
                summary[metric + "_ci95_lower"] = float(np.quantile(boot, 0.025))
                summary[metric + "_ci95_upper"] = float(np.quantile(boot, 0.975))
            selected.append(summary)
    metadata = {
        "split": common_signature[0],
        "ensemble_size": common_signature[1],
        "forecast_horizon": common_signature[2],
        "coordinate_system": common_signature[3],
        "lead": lead,
        "trajectory_ids": sorted({case[0] for case in common_cases}),
        "forecast_cases": [list(case) for case in sorted(common_cases)],
        "bootstrap": {
            "unit": "trajectory",
            "origin_aggregation": "mean within trajectory",
            "confidence": 0.95,
            "replicates": n_bootstrap,
            "seed": seed,
            "paired_resampling_across_models": True,
        },
        "latency_unit": "seconds per ensemble forecast at this physical lead",
        "nfe_unit": "backbone evaluations per ensemble forecast",
        "interval_limitation": "Trajectory bootstrap intervals do not capture training-seed variation; small trajectory counts give coarse intervals.",
        "sources": [
            {
                key: data.get(key)
                for key in (
                    "source_path",
                    "source_sha256",
                    "checkpoint",
                    "checkpoint_sha256",
                    "parameters",
                    "seed",
                    "model_class",
                    "device",
                    "hardware",
                    "torch_version",
                )
            }
            for data in evaluations
        ],
    }
    return selected, metadata


def tex_escape(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def estimate_tex(row, metric, scale=1.0):
    if metric not in row:
        return "--"
    value = row[metric] * scale
    low = row[metric + "_ci95_lower"] * scale
    high = row[metric + "_ci95_upper"] * scale
    return f"{value:.3f} [{low:.3f}, {high:.3f}]"


def write_table(path, rows, metadata, columns, caption, label):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{"
        + caption
        + " "
        + tex_escape(metadata["split"].capitalize())
        + f" split; $E={metadata['ensemble_size']}$, physical lead {metadata['lead']}. "
        + "Brackets show 95\\% trajectory-bootstrap intervals.}",
        r"\label{" + label + "}",
        r"\begin{tabular}{lr" + "r" * len(columns) + "}",
        r"\toprule",
        "Model & NFE & " + " & ".join(col[0] for col in columns) + r" \\",
        r"\midrule",
    ]
    previous = None
    for row in rows:
        if previous is not None and previous != row["model"]:
            lines.append(r"\addlinespace")
        cells = [tex_escape(row["model"]), f"{row['nfe']:g}"]
        cells.extend(estimate_tex(row, metric, scale) for _, metric, scale in columns)
        lines.append(" & ".join(cells) + r" \\")
        previous = row["model"]
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines))


def plot_quality(rows, metadata, destination, xkey):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    models = list(dict.fromkeys(row["model"] for row in rows))
    fig, axes = plt.subplots(1, 3, figsize=(10.3, 3.2), layout="constrained")
    for axis, metric, ylabel in zip(
        axes,
        ("crps", "energy_score", "rmse"),
        ("Fair CRPS ↓", "Fair energy score ↓", "Ensemble-mean RMSE ↓"),
    ):
        for index, model in enumerate(models):
            selected = sorted(
                [row for row in rows if row["model"] == model],
                key=lambda row: row[xkey],
            )
            scale = 1000.0 if xkey == "seconds" else 1.0
            x = np.array([row[xkey] * scale for row in selected])
            y = np.array([row[metric] for row in selected])
            lower = np.array([row[metric + "_ci95_lower"] for row in selected])
            upper = np.array([row[metric + "_ci95_upper"] for row in selected])
            color = COLORS[index % len(COLORS)]
            axis.plot(
                x,
                y,
                label=model,
                color=color,
                marker=MARKERS[index % len(MARKERS)],
                markersize=4,
                linewidth=1.5,
            )
            axis.fill_between(x, lower, upper, color=color, alpha=0.13, linewidth=0)
        axis.set_xscale("log", base=2 if xkey == "nfe" else 10)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.2, linewidth=0.5)
        if xkey == "nfe":
            ticks = sorted({row["nfe"] for row in rows})
            axis.set_xticks(ticks)
            axis.xaxis.set_major_formatter(ScalarFormatter())
            axis.set_xlabel("Backbone evaluations (NFE)")
        else:
            axis.set_xlabel(
                f"Latency / {metadata['ensemble_size']}-member ensemble (ms)"
            )
    axes[0].legend(frameon=False)
    fig.suptitle(
        f"{metadata['split'].capitalize()} · E={metadata['ensemble_size']} · physical lead {metadata['lead']} · {len(metadata['trajectory_ids'])} trajectories",
        fontsize=10,
    )
    basename = "quality_vs_nfe" if xkey == "nfe" else "quality_vs_latency"
    fig.savefig(destination / (basename + ".pdf"), bbox_inches="tight")
    fig.savefig(destination / (basename + ".png"), bbox_inches="tight")
    plt.close(fig)


def create_report(
    inputs, output, *, lead=1, labels=None, split=None, n_bootstrap=2000, seed=712
):
    evaluations = [load_evaluation(path) for path in inputs]
    rows, metadata = summarize(
        evaluations,
        lead=lead,
        labels=labels,
        split=split,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "summary.csv").open("w", newline="") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (destination / "report_metadata.json").write_text(json.dumps(metadata, indent=2))
    write_table(
        destination / "quality_table.tex",
        rows,
        metadata,
        [
            ("CRPS", "crps", 1.0),
            ("Energy score", "energy_score", 1.0),
            ("RMSE", "rmse", 1.0),
        ],
        "Forecast quality in " + tex_escape(metadata["coordinate_system"]) + ".",
        "tab:map-quality",
    )
    write_table(
        destination / "calibration_table.tex",
        rows,
        metadata,
        [
            ("Spread", "spread", 1.0),
            ("Spread/skill", "spread_skill", 1.0),
            ("Coverage (\\%)", "coverage90", 100.0),
        ],
        "Ensemble calibration and spread. Nominal interval coverage is 90\\%.",
        "tab:map-calibration",
    )
    optional_columns = [
        ("RMS spread", "rms_spread", 1.0),
        ("RMS spread/skill", "rms_spread_skill", 1.0),
        ("Extreme ranks (\\%)", "rank_extreme_fraction", 100.0),
    ]
    optional_columns = [
        column for column in optional_columns if any(column[1] in row for row in rows)
    ]
    if optional_columns:
        write_table(
            destination / "rms_calibration_table.tex",
            rows,
            metadata,
            optional_columns,
            "Auxiliary calibration diagnostics using unbiased ensemble variance. "
            "A dash denotes an older evaluation without the metric.",
            "tab:map-rms-calibration",
        )
    write_table(
        destination / "cost_table.tex",
        rows,
        metadata,
        [
            ("Ensemble latency (ms)", "seconds", 1000.0),
            ("Relative spectrum error", "relative_enstrophy_error", 1.0),
        ],
        "Measured sampling cost and enstrophy-spectrum error.",
        "tab:map-cost",
    )
    provenance = [
        {
            key: row[key]
            for key in (
                "model",
                "parameters",
                "source_path",
                "checkpoint",
                "checkpoint_sha256",
            )
        }
        for row in rows
        if row["num_steps"]
        == min(r["num_steps"] for r in rows if r["model"] == row["model"])
    ]
    with (destination / "provenance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(provenance[0]))
        writer.writeheader()
        writer.writerows(provenance)
    lines = [r"\begin{tabular}{lr}", r"\toprule", r"Model & Parameters \\", r"\midrule"]
    for item in provenance:
        parameters = (
            f"{item['parameters']:,}"
            if item["parameters"] is not None
            else "unreported"
        )
        lines.append(tex_escape(item["model"]) + " & " + parameters + r" \\")
        lines.append("% Source: " + item["source_path"])
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (destination / "parameters_table.tex").write_text("\n".join(lines))
    plot_quality(rows, metadata, destination, "nfe")
    plot_quality(rows, metadata, destination, "seconds")
    return rows, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="evaluation.json files or their directories",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--lead", type=int, default=1)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        help="Require this split; never relabel the input",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=712)
    args = parser.parse_args()
    rows, metadata = create_report(
        args.inputs,
        args.output,
        lead=args.lead,
        labels=args.labels,
        split=args.split,
        n_bootstrap=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    print(
        f"Wrote {len(rows)} observed model/budget summaries to {args.output} ({metadata['split']}, E={metadata['ensemble_size']}, lead={metadata['lead']})."
    )


if __name__ == "__main__":
    main()
