#!/usr/bin/env python
"""Build matched-case pilot video reports; never compare FVD across datasets."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


FAMILIES = {
    "FollmerStochasticInterpolant": "Stochastic interpolant",
    "FlowMatchingModel": "Flow matching",
    "FollmerItoMapModel": "Itô map",
    "FlowMapModel": "Flow map",
    "persistence": "Persistence",
}
REQUIRED = set(FAMILIES) - {"persistence"}
MATCH_FIELDS = (
    "dataset", "split", "manifest_sha256", "selected_indices", "ensemble_size",
    "forecast_horizon", "history_length", "num_clips", "seed", "frame_stride",
    "resolution", "i3d_sha256", "lpips", "normalization", "image_metric_clipping",
    "fvd_clipping", "feedback_clipping", "aggregation",
)
QUALITY = ("crps", "energy_score", "rmse", "rms_spread", "coverage_90",
           "spread_skill_ratio", "out_of_range_fraction", "psnr_sample_mean",
           "psnr_oracle_best", "psnr_ensemble_mean", "ssim_sample_mean",
           "ssim_oracle_best", "ssim_ensemble_mean", "lpips_sample_mean",
           "lpips_oracle_best")
COLORS = dict(zip(FAMILIES, ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#666666"]))


def validate_comparable(evaluations, allow_incomplete=False):
    """Reject protocol mismatches and incomplete/duplicate forecast grids."""
    if not evaluations:
        raise ValueError("No evaluations supplied")
    reference = evaluations[0]
    seen = set()
    for item in evaluations:
        if item.get("status") != "complete":
            raise ValueError("Evaluation is not complete")
        for key in MATCH_FIELDS:
            if key not in item or item[key] != reference.get(key):
                raise ValueError(f"Unmatched or missing protocol field: {key}")
        family = item["model_class"]
        if family not in FAMILIES:
            raise ValueError(f"Unknown model family: {family}")
        if item["num_clips"] != len(item["selected_indices"]) or len(set(item["selected_indices"])) != item["num_clips"]:
            raise ValueError("Invalid selected case count or repeated indices")
        horizon, ensemble = item["forecast_horizon"], item["ensemble_size"]
        if horizon < 1 or ensemble < 2:
            raise ValueError("Invalid horizon or ensemble size")
        budgets = [r["num_steps"] for r in item["fvd"]]
        if len(budgets) != len(set(budgets)) or not budgets:
            raise ValueError("Empty or duplicate FVD budgets")
        for steps in budgets:
            if (family, steps) in seen:
                raise ValueError("Duplicate model family and budget")
            seen.add((family, steps))
        expected = {(steps, index, lead) for steps in budgets
                    for index in item["selected_indices"] for lead in range(1, horizon + 1)}
        actual = [(r["num_steps"], r["case_index"], r["lead"]) for r in item["rows"]]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Incomplete or duplicate per-case forecast grid")
        summary_keys = [(r["num_steps"], r["lead"]) for r in item["summary"]]
        if len(summary_keys) != len(set(summary_keys)) or set(summary_keys) != {
                (steps, lead) for steps in budgets for lead in range(1, horizon + 1)}:
            raise ValueError("Incomplete or duplicate per-lead summary grid")
        for row in item["fvd"]:
            if (row["real_clips"] != item["num_clips"] or
                    row["generated_clips"] != item["num_clips"] * ensemble or
                    len(row["fvd_member_values"]) != ensemble or row["feature_dimension"] != 400):
                raise ValueError("FVD feature/count mismatch")
            if not all(np.isfinite(v) and v >= 0 for v in [row["fvd_pooled"], *row["fvd_member_values"]]):
                raise ValueError("Invalid FVD values")
            if not np.isclose(row["fvd_member_mean"], np.mean(row["fvd_member_values"])):
                raise ValueError("FVD member mean inconsistent with raw values")
        for row in item["rows"]:
            if any(value is not None and not np.isfinite(value)
                   for key, value in row.items() if key not in {"clip_id", "source_video"}):
                raise ValueError("Nonfinite forecast metric")
    # Case identity is checked independently of case index: a different clip
    # ordering in a malformed evaluator must not pass via matching metadata.
    def identities(item):
        result = {}
        for row in item["rows"]:
            identity = (row["clip_id"], row["source_video"])
            if row["case_index"] in result and result[row["case_index"]] != identity:
                raise ValueError("Case identity changes inside an evaluation")
            result[row["case_index"]] = identity
        return result
    reference_ids = identities(reference)
    if any(identities(item) != reference_ids for item in evaluations):
        raise ValueError("Case identities differ across models")
    if not allow_incomplete:
        for family in REQUIRED:
            expected_steps = (1, 8) if "Map" in family else (16, 64)
            if not all((family, step) in seen for step in expected_steps):
                raise ValueError(f"Missing required model/budgets: {family} {expected_steps}")


def summarize_evaluation(item):
    """Mean per-lead quality and summed per-lead generation cost, per budget."""
    output = []
    for fvd in item["fvd"]:
        steps = fvd["num_steps"]
        leads = sorted((r for r in item["summary"] if r["num_steps"] == steps), key=lambda r: r["lead"])
        row = {"dataset": item["dataset"], "model_class": item["model_class"],
               "model": FAMILIES[item["model_class"]], "num_steps": steps,
               "history_length": item["history_length"], "horizon": item["forecast_horizon"],
               "num_clips": item["num_clips"], "ensemble_size": item["ensemble_size"],
               "real_fvd_clips": fvd["real_clips"], "pooled_generated_fvd_clips": fvd["generated_clips"],
               "member_generated_fvd_clips": fvd["real_clips"],
               "fvd_pooled": fvd["fvd_pooled"], "fvd_member_mean": fvd["fvd_member_mean"],
               "fvd_member_std": fvd["fvd_member_std"],
               "rollout_seconds": sum(r["seconds"] for r in leads),
               "rollout_nfe": sum(r["nfe"] for r in leads)}
        for key in QUALITY:
            values = [r.get(key) for r in leads]
            row["mean_" + key] = float(np.mean(values)) if all(v is not None for v in values) else None
            row["last_" + key] = leads[-1].get(key)
        output.append(row)
    return output


def render_rollout_gif(npz_path, output_path, title="", fps=7):
    """Truth and first two members, with observed prefix and labeled leads."""
    if fps <= 0:
        raise ValueError("GIF display fps must be positive")
    with np.load(npz_path, allow_pickle=False) as data:
        truth = data["truth"]
        predictions = data["predictions"][:2]
        history = int(data["history_length"])
    if len(predictions) != 2 or truth.shape[0] != history + predictions.shape[1]:
        raise ValueError("Qualitative artifact shape mismatch")
    sequences = [truth] + [np.concatenate([truth[:history], member], 0) for member in predictions]
    h, w = truth.shape[-2:]
    frames = []
    for time in range(len(truth)):
        canvas = Image.new("RGB", (2 * w + 110, 3 * 2 * h + 38), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((5, 3), title, fill="black")
        phase = f"Observed context {time + 1}/{history}" if time < history else f"Forecast lead {time - history + 1}"
        draw.text((5, 19), phase, fill="black")
        for row, (label, sequence) in enumerate(zip(["Truth", "Member 0", "Member 1"], sequences)):
            pixels = np.rint(np.clip((sequence[time] + 1) * 127.5, 0, 255)).astype(np.uint8)
            if pixels.shape[0] == 1:
                pixels = np.repeat(pixels, 3, axis=0)
            picture = Image.fromarray(pixels.transpose(1, 2, 0)).resize((2 * w, 2 * h), Image.Resampling.NEAREST)
            canvas.paste(picture, (110, 38 + row * 2 * h))
            draw.text((5, 38 + row * 2 * h + h), label, fill="black")
        frames.append(canvas)
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=round(1000 / fps), loop=0, optimize=False)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_dataset(evaluations, summary, destination):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    metrics = [("crps", "Fair CRPS ↓"), ("rmse", "Ensemble-mean RMSE ↓"),
               ("lpips_sample_mean", "Mean sample LPIPS ↓"),
               ("ssim_sample_mean", "Mean sample SSIM ↑"),
               ("psnr_sample_mean", "Mean sample PSNR (dB) ↑"),
               ("cumulative_seconds", "Cumulative generation seconds ↓")]
    for item in evaluations:
        for rank, steps in enumerate(sorted({r["num_steps"] for r in item["summary"]})):
            leads = sorted((r for r in item["summary"] if r["num_steps"] == steps), key=lambda r: r["lead"])
            label = f"{FAMILIES[item['model_class']]} ({steps})"
            for axis, (key, ylabel) in zip(axes.flat, metrics):
                values = [r.get(key) for r in leads]
                if all(v is not None for v in values):
                    axis.plot([r["lead"] for r in leads], values, label=label,
                              color=COLORS[item["model_class"]], linestyle="-" if rank == 0 else "--")
                axis.set(xlabel="Forecast lead (sampled frames)", ylabel=ylabel)
                axis.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle(f"{evaluations[0]['dataset'].upper()}: pixel-space pilot, {evaluations[0]['ensemble_size']} completions per case")
    fig.savefig(destination / "lead_metrics.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for row in summary:
        for axis, key in zip(axes, ["fvd_pooled", "fvd_member_mean"]):
            axis.errorbar(row["rollout_seconds"], row[key],
                          yerr=row["fvd_member_std"] if key == "fvd_member_mean" else None,
                          marker="o" if row["num_steps"] in (1, 16) else "s", linestyle="none",
                          color=COLORS[row["model_class"]], capsize=3,
                          label=f"{row['model']} ({row['num_steps']})")
        axes[0].set_title(f"Pooled FVD: {row['real_fvd_clips']} real / {row['pooled_generated_fvd_clips']} generated")
        axes[1].set_title(f"Member FVD: {row['real_fvd_clips']} real / {row['member_generated_fvd_clips']} generated")
    for axis in axes:
        axis.set(xscale="log", xlabel="Generation seconds per complete ensemble rollout", ylabel="Pilot FVD ↓")
        axis.grid(alpha=.2)
    fig.suptitle(f"{evaluations[0]['dataset'].upper()} — member error bars are completion SD, not confidence intervals")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.savefig(destination / "fvd_vs_cost.png", dpi=160)
    plt.close(fig)


def build_report(paths, output_dir, allow_incomplete=False, gif_fps=7):
    groups = {}
    for path in paths:
        path = Path(path)
        item = json.loads(path.read_text())
        groups.setdefault(item["dataset"], []).append((path, item))
    if not allow_incomplete and set(groups) != {"kth", "clevrer"}:
        raise ValueError("A complete report requires KTH and CLEVRER")
    # Validate everything before writing any apparently complete report.
    for entries in groups.values():
        validate_comparable([item for _, item in entries], allow_incomplete)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    combined = []
    for dataset, entries in sorted(groups.items()):
        destination = output / dataset
        destination.mkdir(parents=True, exist_ok=True)
        items = [item for _, item in entries]
        summary = [row for item in items for row in summarize_evaluation(item)]
        combined.extend(summary)
        write_csv(destination / "summary.csv", summary)
        lead_rows = [{"dataset": dataset, "model_class": item["model_class"], **row}
                     for item in items for row in item["summary"]]
        write_csv(destination / "per_lead.csv", lead_rows)
        plot_dataset(items, summary, destination)
        reference = items[0]
        lines = [f"# {dataset.upper()} pixel-space pilot", "",
                 "Preliminary end-to-end results, not a reproduction of the paper's latent-space training or evaluation budget.", "",
                 f"Matched protocol: {reference['num_clips']} clips, {reference['ensemble_size']} completions per clip; "
                 f"{reference['history_length']} observed + {reference['forecast_horizon']} forecast frames; stride {reference['frame_stride']}; "
                 f"resolution {reference['resolution']}; split `{reference['split']}`.", "",
                 "Quality columns average per-lead metrics, with each original source video equally weighted. "
                 "Generation cost sums those per-lead means for the entire ensemble. Per-lead confidence intervals in the CSV use source-video bootstrap; "
                 "they do not measure training-seed variation. FVD uses every selected clip and is not source-video reweighted.", "",
                 "| Model | Steps/frame | Mean CRPS ↓ | Mean LPIPS ↓ | Mean PSNR ↑ | Mean SSIM ↑ | Pooled FVD ↓ | Member FVD mean ± SD ↓ | Rollout seconds ↓ | Rollout NFE |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        def fmt(value):
            return "—" if value is None else f"{value:.4g}"
        for row in summary:
            lines.append("| " + " | ".join([
                row["model"], str(row["num_steps"]), fmt(row["mean_crps"]), fmt(row["mean_lpips_sample_mean"]),
                fmt(row["mean_psnr_sample_mean"]), fmt(row["mean_ssim_sample_mean"]), fmt(row["fvd_pooled"]),
                f"{fmt(row['fvd_member_mean'])} ± {fmt(row['fvd_member_std'])}", fmt(row["rollout_seconds"]), fmt(row["rollout_nfe"]),
            ]) + " |")
        lines += ["", f"Pooled FVD compares {reference['num_clips']} real clips with {reference['num_clips'] * reference['ensemble_size']} generated clips. "
                  f"Each member FVD compares {reference['num_clips']} real with {reference['num_clips']} generated clips. "
                  "Member SD describes completion randomness, not a confidence interval. Do not compare these values across datasets or against the paper's 256-real/25,600-generated table.",
                  "", "![Per-lead quality and cost](lead_metrics.png)", "", "![FVD versus complete-rollout cost](fvd_vs_cost.png)", "",
                  "## Fixed qualitative case", "",
                  f"Every animation uses the same first selected case and members 0 and 1. Display speed is {gif_fps} fps, not an assertion of physical frame rate. "
                  "Observed context is included; future truth is never fed to forecasts. No examples are selected by score.", ""]
        for path, item in entries:
            for fvd in item["fvd"]:
                steps = fvd["num_steps"]
                artifact = path.parent / f"case_000_steps_{steps}.npz"
                if artifact.exists():
                    name = f"{item['model_class']}_steps_{steps}.gif"
                    render_rollout_gif(artifact, destination / name,
                                       f"{dataset.upper()} {FAMILIES[item['model_class']]} ({steps})", gif_fps)
                    lines += [f"**{FAMILIES[item['model_class']]}: {steps} steps/frame**", "", f"![Fixed case]({name})", ""]
        (destination / "README.md").write_text("\n".join(lines).rstrip() + "\n")
        provenance = {"protocol": {key: reference[key] for key in MATCH_FIELDS},
                      "allow_incomplete": allow_incomplete,
                      "inputs": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "checkpoint_sha256": item.get("checkpoint_sha256")}
                                 for path, item in entries]}
        (destination / "report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    write_csv(output / "all_datasets.csv", combined)
    (output / "README.md").write_text("# Pixel-space video pilot results\n\n"
        "Four model families are evaluated using fixed rolling context, independently trained dataset-specific teachers, and matched forecast cases. "
        "These initial results establish an executable benchmark; they do not establish convergence or equivalence to the published latent-space results.\n\n"
        + "\n".join(f"- [{name.upper()} results]({name}/README.md)" for name in sorted(groups))
        + "\n\nFVD comparisons are valid only within a dataset and this recorded protocol. "
        "The combined CSV is for record keeping, not an aggregate score across datasets. "
        "Image metrics use clipped copies; proper scores and calibration retain out-of-range forecasts. "
        "Oracle best-member metrics in the CSV are explicitly separated from average-completion scores.\n")
    return combined


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-incomplete", action="store_true", help="Permit a development report missing datasets/model budgets")
    parser.add_argument("--gif-fps", default=7, type=float)
    args = parser.parse_args(argv)
    build_report(args.evaluations, args.output_dir, args.allow_incomplete, args.gif_fps)


if __name__ == "__main__":
    main()
