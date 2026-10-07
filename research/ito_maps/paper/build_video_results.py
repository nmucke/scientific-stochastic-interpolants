#!/usr/bin/env python
"""Turn complete audited video reports into manuscript tables and fixed figures.

The numerical aggregation belongs to report_video_results.py. This builder reads
its CSV outputs and verifies the corresponding source manifests/audits. Partial
reports are allowed only in explicitly separate development destinations.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[2]
VIDEO = ROOT / "research/video_forecasting"
DEFAULT_REPORT = VIDEO / "results/report"
DEFAULT_OUTPUT = PAPER / "figures/video_results"
DEFAULT_SNIPPET = PAPER / "generated/video_results.tex"
DEFAULT_MAIN_SNIPPET = PAPER / "generated/video_main_results.tex"
MAIN_VISUAL_SELECTION = VIDEO / "notes/paper_main_visual_selection.json"
VISUAL_SELECTION = VIDEO / "notes/paper_visual_selection.json"
SPEC = importlib.util.spec_from_file_location(
    "video_report_for_paper", VIDEO / "experiments/report_video_results.py"
)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)
FAMILY_ORDER = [
    "FollmerStochasticInterpolant",
    "FlowMatchingModel",
    "FollmerItoMapModel",
    "FlowMapModel",
    "persistence",
]
LABELS = {
    "FollmerStochasticInterpolant": "SI",
    "FlowMatchingModel": "FM",
    "FollmerItoMapModel": r"It\^o map",
    "FlowMapModel": "Flow map",
    "persistence": "Persistence",
}
# Declared before completed all-model video results are available, not selected by score.
VISUAL_BUDGETS = {
    "FollmerStochasticInterpolant": 64,
    "FlowMatchingModel": 64,
    "FollmerItoMapModel": 8,
    "FlowMapModel": 1,
    "persistence": 1,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve_source(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_dataset_report(report_dir, dataset, allow_incomplete=False, video_root=VIDEO):
    directory = Path(report_dir) / dataset
    manifest_path = directory / "report_manifest.json"
    manifest = read_json(manifest_path)
    require(
        allow_incomplete or manifest.get("allow_incomplete") is False,
        f"Refusing a development report as completed results: {dataset}",
    )
    evaluations, paths = [], {}
    for entry in manifest["inputs"]:
        path = resolve_source(entry["path"])
        require(
            sha256(path) == entry["sha256"],
            f"Evaluation changed since report generation: {path}",
        )
        value = read_json(path)
        require(value["dataset"] == dataset, "Mixed datasets in report manifest")
        require(value["model_class"] not in paths, "Duplicate model family in report")
        paths[value["model_class"]] = path
        evaluations.append(value)
    REPORT.validate_comparable(evaluations, allow_incomplete=allow_incomplete)
    for field in REPORT.MATCH_FIELDS:
        require(
            manifest["protocol"][field] == evaluations[0][field],
            f"Report protocol mismatch: {field}",
        )
    if not allow_incomplete:
        require(
            set(paths) == set(FAMILY_ORDER),
            f"Complete paper results need all four models and persistence: {dataset}",
        )
        frozen_path = Path(video_root) / f"results/{dataset}_protocol.json"
        audit = read_json(Path(video_root) / f"results/{dataset}_audit.json")
        require(
            audit["status"] == "passed"
            and audit["protocol_sha256"] == sha256(frozen_path),
            f"Missing/stale frozen-protocol audit: {dataset}",
        )
        for path in paths.values():
            require(
                audit["evaluation_artifacts"].get(path.parent.name) == sha256(path),
                f"Evaluation is not covered by the completed audit: {path}",
            )
    with (directory / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in row:
            if key not in {"dataset", "model_class", "model"}:
                row[key] = None if row[key] == "" else float(row[key])
        require(row["dataset"] == dataset, "Summary contains another dataset")
        require(
            all(
                value is None or math.isfinite(value)
                for key, value in row.items()
                if key not in {"dataset", "model_class", "model"}
            ),
            "Nonfinite report summary",
        )
    expected = {
        (item["model_class"], row["num_steps"])
        for item in evaluations
        for row in item["fvd"]
    }
    actual = [(row["model_class"], row["num_steps"]) for row in rows]
    require(
        len(actual) == len(set(actual)) and set(actual) == expected,
        "Summary family/budget grid mismatch",
    )
    rows.sort(
        key=lambda row: (FAMILY_ORDER.index(row["model_class"]), row["num_steps"])
    )
    return {
        "directory": directory,
        "rows": rows,
        "evaluations": evaluations,
        "paths": paths,
        "protocol": manifest["protocol"],
        "report_manifest_sha256": sha256(manifest_path),
        "summary_sha256": sha256(directory / "summary.csv"),
    }


def fmt(value, digits=3):
    return r"\textemdash" if value is None else f"{value:.{digits}f}"


def table(headers, rows):
    lines = [
        r"\begin{tabular}{l" + "r" * (len(headers) - 1) + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"]) + "\n"


def make_tables(rows):
    primary, image, calibration = [], [], []
    for row in rows:
        prefix = [LABELS[row["model_class"]], str(int(row["num_steps"]))]
        primary.append(
            prefix
            + [
                str(int(row["rollout_nfe"])),
                fmt(row["rollout_seconds"], 2),
                fmt(row["mean_crps"], 4),
                fmt(row["fvd_pooled"], 1),
                fmt(row["fvd_member_mean"], 1)
                + r" $\pm$ "
                + fmt(row["fvd_member_std"], 1),
            ]
        )
        image.append(
            prefix
            + [
                fmt(row["mean_psnr_sample_mean"], 2),
                fmt(row["mean_ssim_sample_mean"], 3),
                fmt(row["mean_lpips_sample_mean"], 3),
                fmt(row["mean_psnr_ensemble_mean"], 2),
                fmt(row["mean_ssim_ensemble_mean"], 3),
            ]
        )
        calibration.append(
            prefix
            + [
                fmt(row["mean_energy_score"], 4),
                fmt(row["mean_rmse"], 4),
                fmt(row["mean_rms_spread"], 4),
                fmt(row["mean_spread_skill_ratio"], 3),
                fmt(100 * row["mean_coverage_90"], 1),
                fmt(100 * row["mean_out_of_range_fraction"], 1),
            ]
        )
    return {
        "quality": table(
            [
                "Model",
                r"Steps/frame",
                "Rollout NFE",
                r"Time (s)$\downarrow$",
                r"CRPS$\downarrow$",
                r"FVD pooled$\downarrow$",
                r"FVD member mean $\pm$ SD",
            ],
            primary,
        ),
        "images": table(
            [
                "Model",
                "Steps/frame",
                r"Sample PSNR$\uparrow$",
                r"Sample SSIM$\uparrow$",
                r"Sample LPIPS$\downarrow$",
                r"Mean PSNR$\uparrow$",
                r"Mean SSIM$\uparrow$",
            ],
            image,
        ),
        "calibration": table(
            [
                "Model",
                "Steps/frame",
                r"Energy$\downarrow$",
                r"RMSE$\downarrow$",
                "RMS spread",
                "Spread/skill",
                r"Coverage (\%)",
                r"Out of range (\%)",
            ],
            calibration,
        ),
    }


def load_fixed_example(report, allow_incomplete=False):
    """Read the fixed case/member, enforcing common observations and truth."""
    selections = []
    reference = None
    case_id = None
    inputs = []
    for family in FAMILY_ORDER:
        if family not in report["paths"]:
            require(allow_incomplete, "Missing model for qualitative comparison")
            continue
        directory = report["paths"][family].parent
        budget = VISUAL_BUDGETS[family]
        path = directory / f"case_000_steps_{budget}.npz"
        require(path.exists(), f"Missing preselected qualitative artifact: {path}")
        sidecar = read_json(path.with_suffix(".json"))
        cases = read_json(directory / "cases.json")
        require(
            sidecar["clip_id"] == cases[0]["clip_id"],
            "Qualitative artifact is not the first selected case",
        )
        evaluation = next(
            item for item in report["evaluations"] if item["model_class"] == family
        )
        first_case = report["protocol"]["selected_indices"][0]
        require(
            all(
                row["clip_id"] == sidecar["clip_id"]
                for row in evaluation["rows"]
                if row["case_index"] == first_case
            ),
            "Qualitative case differs from the frozen evaluation ordering",
        )
        with np.load(path, allow_pickle=False) as data:
            truth = data["truth"].copy()
            predictions = data["predictions"].copy()
            history = int(data["history_length"])
        require(
            predictions.ndim == 5
            and predictions.shape[0] == report["protocol"]["ensemble_size"],
            "Invalid qualitative ensemble shape",
        )
        require(
            history == report["protocol"]["history_length"]
            and len(truth) == history + predictions.shape[1],
            "Qualitative history/horizon mismatch",
        )
        require(
            predictions.shape[1] == report["protocol"]["forecast_horizon"],
            "Wrong qualitative horizon",
        )
        require(
            truth.ndim == 4 and predictions.shape[2:] == truth.shape[1:],
            "Qualitative truth/forecast spatial shape mismatch",
        )
        require(np.isfinite(truth).all(), "Nonfinite qualitative truth")
        require(np.isfinite(predictions).all(), "Nonfinite raw qualitative forecast")
        if reference is None:
            reference, case_id = truth, sidecar["clip_id"]
        else:
            require(
                np.array_equal(reference, truth) and case_id == sidecar["clip_id"],
                "Qualitative models do not share the same observed case",
            )
        selections.append((f"{REPORT.FAMILIES[family]} ({budget})", predictions[0]))
        inputs.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "model_class": family,
                "steps": budget,
            }
        )
    require(reference is not None, "No qualitative artifacts available")
    return reference, selections, case_id, inputs


def fixed_example(dataset, report, output, allow_incomplete=False):
    """Always case_000 and member_0 at prespecified budgets and lead fractions."""
    reference, selections, case_id, inputs = load_fixed_example(
        report, allow_incomplete
    )
    horizon = report["protocol"]["forecast_horizon"]
    lead_indices = sorted(
        set([0, horizon // 4, horizon // 2, 3 * horizon // 4, horizon - 1])
    )
    sequences = [
        ("Truth", reference[report["protocol"]["history_length"] :]),
        *selections,
    ]
    figure_height = 1.45 * len(sequences) + 0.55
    fig, axes = plt.subplots(
        len(sequences),
        len(lead_indices),
        figsize=(10.3, figure_height),
        squeeze=False,
    )
    for row_index, (label, sequence) in enumerate(sequences):
        for column, lead in enumerate(lead_indices):
            axis = axes[row_index, column]
            pixels = np.clip((sequence[lead] + 1) / 2, 0, 1)
            axis.imshow(
                pixels[0] if pixels.shape[0] == 1 else pixels.transpose(1, 2, 0),
                cmap="gray" if pixels.shape[0] == 1 else None,
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(f"Future lead {lead + 1}", fontsize=10)
            if column == 0:
                axis.set_ylabel(
                    label, rotation=0, ha="right", va="center", fontsize=9, labelpad=9
                )
    fig.suptitle(
        f"{dataset.upper()} | first fixed case: {case_id} | member 0", fontsize=11
    )
    fig.subplots_adjust(
        left=0.22,
        right=0.995,
        bottom=0.015,
        top=1 - 0.65 / figure_height,
        wspace=0.025,
        hspace=0.025,
    )
    fig.savefig(output / f"{dataset}_fixed_forecast.pdf", bbox_inches="tight")
    fig.savefig(output / f"{dataset}_fixed_forecast.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    return {
        "case_index_in_selected_list": 0,
        "clip_id": case_id,
        "member": 0,
        "future_leads": [lead + 1 for lead in lead_indices],
        "inputs": inputs,
        "selection_rule": "Fixed before completed comparisons: first selected case/member0; SI64/FM64/Ito8/Flow1/persistence1",
    }


def main_table(reports):
    """Use report FVD values unchanged; display the same budgets for both datasets."""
    datasets = [name for name in ("kth", "clevrer") if name in reports]
    lookup = {
        dataset: {
            (r["model_class"], int(r["num_steps"])): r for r in reports[dataset]["rows"]
        }
        for dataset in datasets
    }
    grids = [set(value) for value in lookup.values()]
    require(all(grid == grids[0] for grid in grids), "Main-table budget grids differ")
    rows = []
    for family, steps in sorted(
        grids[0], key=lambda key: (FAMILY_ORDER.index(key[0]), key[1])
    ):
        counts = [
            lookup[name][family, steps]["rollout_nfe"]
            / reports[name]["protocol"]["forecast_horizon"]
            for name in datasets
        ]
        require(
            all(count == counts[0] for count in counts),
            "Main-table per-frame NFE differs across datasets",
        )
        values = [LABELS[family], str(int(counts[0]))]
        for dataset in datasets:
            row = lookup[dataset][family, steps]
            values.append(fmt(row["fvd_pooled"], 1))
            values.append(
                fmt(row["fvd_member_mean"], 1)
                + r" $\pm$ "
                + fmt(row["fvd_member_std"], 1)
            )
        rows.append(values)
    return table(
        [
            "Model",
            "NFE/frame",
            *[
                header
                for name in datasets
                for header in [
                    name.upper() + " pooled",
                    name.upper() + " member",
                ]
            ],
        ],
        rows,
    )


def main_frame_sequences(reports, output, allow_incomplete=False):
    """Side-by-side datasets: observed last frame, then first/middle/last forecast."""
    datasets = [name for name in ("kth", "clevrer") if name in reports]
    families = FAMILY_ORDER[:-1]
    labels = ["Truth", "SI (64)", "FM (64)", "Itô map (8)", "Flow map (1)"]
    fig = plt.figure(figsize=(5.6 * len(datasets), 6.0))
    outer = fig.add_gridspec(
        1,
        len(datasets),
        left=0.11 / len(datasets),
        right=0.998,
        bottom=0.015,
        top=0.88,
        wspace=0.07,
    )
    metadata = {}
    for index, dataset in enumerate(datasets):
        report = reports[dataset]
        truth, selections, case_id, inputs = load_fixed_example(
            report, allow_incomplete
        )
        predictions = {
            item["model_class"]: frames for item, (_, frames) in zip(inputs, selections)
        }
        history = report["protocol"]["history_length"]
        horizon = report["protocol"]["forecast_horizon"]
        stride = report["protocol"]["frame_stride"]
        leads = sorted(set([0, horizon // 2, horizon - 1]))
        columns = 1 + len(leads)
        grid = outer[index].subgridspec(5, columns, wspace=0.035, hspace=0.04)
        origin = read_json(
            report["paths"][next(iter(report["paths"]))].parent / "cases.json"
        )[0].get("source_frame_start", 0)
        for row in range(5):
            for column in range(columns):
                axis = fig.add_subplot(grid[row, column])
                if column == 0:
                    frame = truth[history - 1]
                elif row == 0:
                    frame = truth[history + leads[column - 1]]
                else:
                    family = families[row - 1]
                    if family not in predictions:
                        require(allow_incomplete, "Missing main figure family")
                        axis.set_axis_off()
                        continue
                    frame = predictions[family][leads[column - 1]]
                pixels = np.clip((frame + 1) / 2, 0, 1)
                axis.imshow(
                    pixels[0] if pixels.shape[0] == 1 else pixels.transpose(1, 2, 0),
                    cmap="gray" if pixels.shape[0] == 1 else None,
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
                axis.set_xticks([])
                axis.set_yticks([])
                if column == 0:
                    for spine in axis.spines.values():
                        spine.set_color("#0072B2")
                        spine.set_linewidth(1.6)
                if row == 0:
                    retained = (
                        history - 1 if column == 0 else history + leads[column - 1]
                    )
                    title = (
                        "Observed" if column == 0 else f"Lead {leads[column - 1] + 1}"
                    )
                    axis.set_title(
                        f"{title}\n$r={retained}$",
                        fontsize=11,
                        color="#0072B2" if column == 0 else "black",
                        pad=5,
                    )
                if column == 0 and index == 0:
                    axis.set_ylabel(
                        labels[row],
                        rotation=0,
                        ha="right",
                        va="center",
                        fontsize=11,
                        labelpad=6,
                    )
        box = outer[index].get_position(fig)
        fig.text(
            (box.x0 + box.x1) / 2,
            0.988,
            dataset.upper(),
            ha="center",
            va="top",
            fontsize=12,
            weight="bold",
        )
        metadata[dataset] = {
            "case_index_in_selected_list": 0,
            "clip_id": case_id,
            "member": 0,
            "observed_retained_index": history - 1,
            "future_leads": [lead + 1 for lead in leads],
            "displayed_retained_indices": [
                history - 1,
                *[history + lead for lead in leads],
            ],
            "displayed_source_frame_indices": [
                origin + retained * stride
                for retained in [history - 1, *[history + lead for lead in leads]]
            ],
            "source_frame_start": origin,
            "frame_stride": stride,
            "inputs": [item for item in inputs if item["model_class"] != "persistence"],
            "predictions_are": "individual sampled completion, never ensemble mean",
        }
    fig.savefig(output / "main_forecast_sequences.pdf", bbox_inches="tight")
    fig.savefig(output / "main_forecast_sequences.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    return metadata


def main_latex_snippet(output, development=False):
    prefix = (
        str(output.resolve().relative_to(PAPER))
        if output.resolve().is_relative_to(PAPER)
        else str(output.resolve())
    )
    warning = (
        r"\paragraph{Incomplete development preview.} Models or datasets are omitted; this is not a completed comparison."
        if development
        else ""
    )
    return (
        "\n".join(
            [
                warning,
                r"\begin{table*}[t]\centering\small",
                rf"\input{{{prefix}/main_fvd_table.tex}}",
                r"\caption{Pixel-video FVD (lower is better). Pooled FVD compares 128 real and 512 generated clips. Member FVD reports mean $\pm$ completion SD of four separate 128-versus-128 comparisons, not confidence intervals. Cases are shared; each model uses one checkpoint at all its budgets. NFE is per generated frame and member; persistence uses none. Other metrics are in Appendix~\ref{app:video-results}.}\label{tab:video-main-fvd}",
                r"\end{table*}",
                r"\begin{figure*}[t]\centering",
                rf"\includegraphics[width=\textwidth]{{{prefix}/main_forecast_sequences.pdf}}",
                r"\caption{Matched observed and predicted frame sequences for both datasets. Rows show truth and one sampled completion from each model, never an ensemble average. The blue-bordered column repeats the last observed frame; remaining columns are forecasts. $r$ is the zero-based retained-frame index within the clip: KTH retains every frame; CLEVRER raw-frame index is $5r$. Cases are the first frozen selections, member zero, at prespecified budgets (parentheses). Display clipping does not alter feedback. The other observed history frames condition each model but are omitted for space.}\label{fig:video-main-sequences}",
                r"\end{figure*}",
            ]
        )
        + "\n"
    )


def latex_snippet(output, reports, development=False):
    prefix = (
        str(output.resolve().relative_to(PAPER))
        if output.resolve().is_relative_to(PAPER)
        else str(output.resolve())
    )
    lines = [r"\subsection{Held-Out Pixel-Video Results}\label{app:video-results}"]
    if development:
        lines += [
            r"\paragraph{Incomplete development output.} This diagnostic fragment omits models or datasets and is not a completed comparison."
        ]
    else:
        lines += [
            r"Both frozen video evaluations and their artifact audits are complete. Results below use all four model families and persistence on the same cases; this single-training-seed pixel pilot is not a reproduction of the larger latent-space benchmark."
        ]
    lines += [
        r"Frame scores average over future leads and weight original source videos equally. FVD uses the selected clip distribution. Time is generation seconds for a complete four-member rollout, and rollout NFE counts neural calls per member over all future frames. Member-FVD SD measures completion randomness, not uncertainty over test cases or training seeds. SI denotes the stochastic interpolant and FM denotes flow matching."
    ]
    for dataset, report in reports.items():
        label = dataset.upper()
        protocol = report["protocol"]
        cases, ensemble = protocol["num_clips"], protocol["ensemble_size"]
        for suffix, caption in [
            (
                "quality",
                f"{label}: matched pilot distributional quality and generation cost. Pooled FVD uses {cases} real and {cases * ensemble} generated clips; member FVD uses {ensemble} separate {cases}-versus-{cases} comparisons. Lower CRPS and FVD are better. Persistence has zero neural evaluations.",
            ),
            (
                "images",
                f"{label}: future image quality, averaged across leads. Sample metrics average over completions. Mean PSNR/SSIM score the ensemble-mean image. Image/perceptual metrics use clipped copies; neither feedback nor proper scores are clipped.",
            ),
            (
                "calibration",
                f"{label}: additional future-only proper-score and calibration diagnostics. Spread/skill includes the finite-ensemble correction; 90\\% denotes the empirical central interval from four members and is not expected to attain 90\\% even under ideal calibration. Out-of-range values are measured before display clipping.",
            ),
        ]:
            lines += [
                r"\begin{table*}[t]\centering\small",
                rf"\input{{{prefix}/{dataset}_{suffix}_table.tex}}",
                rf"\caption{{{caption}}}\label{{tab:video-{dataset}-{suffix}}}",
                r"\end{table*}",
            ]
        lines += [
            r"\begin{figure*}[t]\centering",
            rf"\includegraphics[width=\textwidth]{{{prefix}/{dataset}_fvd_vs_cost.png}}",
            rf"\caption{{{label}: quality versus complete-ensemble generation cost, copied from the audited report without re-aggregating scores. The right panel uses equal-count member FVD; error bars are completion SD, not confidence intervals.}}\label{{fig:video-{dataset}-cost}}",
            r"\end{figure*}",
            r"\begin{figure*}[t]\centering",
            rf"\includegraphics[width=\textwidth,height=.78\textheight,keepaspectratio]{{{prefix}/{dataset}_fixed_forecast.pdf}}",
            rf"\caption{{{label}: the first preselected test case and ensemble member zero, at fixed future leads. Solver64, It\^o-map8, flow-map1 and persistence budgets were chosen before completed comparisons. All rows share the same observations and truth; no case or member was chosen by forecast quality. Display clips copies to the pixel range.}}\label{{fig:video-{dataset}-forecast}}",
            r"\end{figure*}",
        ]
    lines += [
        r"Full per-lead source-video bootstrap intervals, additional oracle-member metrics and the unrounded aggregate metrics remain in the accompanying machine-readable reports. These intervals do not quantify training-seed variation. The finite-member forecast examples are illustrative rather than evidence of calibrated conditional uncertainty."
    ]
    return "\n\n".join(lines) + "\n"


def build_results(
    report_dir=DEFAULT_REPORT,
    output_dir=DEFAULT_OUTPUT,
    snippet=DEFAULT_SNIPPET,
    allow_incomplete=False,
    video_root=VIDEO,
    main_snippet=None,
):
    report_dir, output, snippet = Path(report_dir), Path(output_dir), Path(snippet)
    main_snippet = (
        Path(main_snippet)
        if main_snippet is not None
        else snippet.with_name("video_main_results.tex")
    )
    if allow_incomplete and (
        output.resolve() == DEFAULT_OUTPUT.resolve()
        or snippet.resolve() == DEFAULT_SNIPPET.resolve()
        or main_snippet.resolve() == DEFAULT_MAIN_SNIPPET.resolve()
    ):
        raise ValueError(
            "Incomplete development output requires separate noncanonical asset and snippet destinations"
        )
    datasets = sorted(
        path.name
        for path in report_dir.iterdir()
        if path.is_dir() and (path / "report_manifest.json").exists()
    )
    require(
        set(datasets).issubset({"kth", "clevrer"}) and datasets,
        "No supported dataset reports",
    )
    require(
        allow_incomplete or set(datasets) == {"kth", "clevrer"},
        "Completed paper results require both datasets",
    )
    # Validate all inputs before creating any apparently complete paper results.
    reports = {
        dataset: load_dataset_report(report_dir, dataset, allow_incomplete, video_root)
        for dataset in datasets
    }
    selection = read_json(VISUAL_SELECTION)
    main_selection = read_json(MAIN_VISUAL_SELECTION)
    require(
        selection["budgets"] == VISUAL_BUDGETS
        and selection["case_index_in_frozen_selection"] == 0
        and selection["ensemble_member"] == 0,
        "Qualitative selection disagrees with its predeclared record",
    )
    require(
        main_selection["case_index_in_frozen_selection"] == 0
        and main_selection["ensemble_member"] == 0
        and main_selection["budgets"]
        == {
            key: value for key, value in VISUAL_BUDGETS.items() if key != "persistence"
        },
        "Main qualitative selection disagrees with its predeclared record",
    )
    for dataset, report in reports.items():
        require(
            (report["directory"] / "fvd_vs_cost.png").exists(),
            f"Missing report cost figure: {dataset}",
        )
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": (
            "incomplete_development_only"
            if allow_incomplete
            else "complete_audited_comparison"
        ),
        "report_root": str(report_dir.resolve()),
        "predeclared_visual_selection": {
            "path": str(VISUAL_SELECTION.relative_to(ROOT)),
            "sha256": sha256(VISUAL_SELECTION),
            "recorded_utc": selection["recorded_utc"],
        },
        "predeclared_main_visual_selection": {
            "path": str(MAIN_VISUAL_SELECTION.relative_to(ROOT)),
            "sha256": sha256(MAIN_VISUAL_SELECTION),
            "recorded_utc": main_selection["recorded_utc"],
        },
        "datasets": {},
    }
    for dataset, report in reports.items():
        for name, content in make_tables(report["rows"]).items():
            (output / f"{dataset}_{name}_table.tex").write_text(content)
        shutil.copy2(
            report["directory"] / "fvd_vs_cost.png",
            output / f"{dataset}_fvd_vs_cost.png",
        )
        shutil.copy2(
            report["directory"] / "summary.csv",
            output / f"{dataset}_unrounded_summary.csv",
        )
        qualitative = fixed_example(dataset, report, output, allow_incomplete)
        manifest["datasets"][dataset] = {
            "report_manifest_sha256": report["report_manifest_sha256"],
            "summary_sha256": report["summary_sha256"],
            "qualitative": qualitative,
            "protocol": report["protocol"],
        }
    (output / "main_fvd_table.tex").write_text(main_table(reports))
    manifest["main_frame_sequences"] = main_frame_sequences(
        reports, output, allow_incomplete
    )
    main_snippet.parent.mkdir(parents=True, exist_ok=True)
    main_snippet.write_text(main_latex_snippet(output, allow_incomplete))
    snippet.parent.mkdir(parents=True, exist_ok=True)
    snippet.write_text(latex_snippet(output, reports, allow_incomplete))
    (output / "paper_results_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snippet", type=Path, default=DEFAULT_SNIPPET)
    parser.add_argument(
        "--main-snippet",
        type=Path,
        default=None,
        help="Defaults to video_main_results.tex beside --snippet",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = build_results(
        args.report_dir,
        args.output_dir,
        args.snippet,
        args.allow_incomplete,
        main_snippet=args.main_snippet,
    )
    print(
        json.dumps(
            {"status": result["status"], "datasets": list(result["datasets"])}, indent=2
        )
    )


if __name__ == "__main__":
    main()
