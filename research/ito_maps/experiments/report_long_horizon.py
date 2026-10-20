#!/usr/bin/env python
"""Report autonomous NS boundedness separately from truth-supported skill."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MATCH_KEYS = ("trajectories", "origin", "history_length", "ensemble_size",
              "horizon", "seed", "data_sha256", "truth_leads", "inverse_affine",
              "initial_history_sha256", "truth_sha256", "snapshot_dt")
LABELS = {"FollmerStochasticInterpolant": "Stochastic interpolant",
          "FlowMatchingModel": "Flow matching", "FollmerItoMapModel": "Itô map",
          "FlowMapModel": "Flow map"}
COLORS = dict(zip(LABELS, ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]))


def load_runs(directories):
    """Validate matching initial conditions and absence of invented skill."""
    runs = []
    for directory in directories:
        directory = Path(directory)
        protocol = json.loads((directory / "protocol.json").read_text())
        result = json.loads((directory / "diagnostics.json").read_text())
        rows = result["rows"]
        if not rows:
            raise ValueError(f"No diagnostics: {directory}")
        for key in MATCH_KEYS:
            if key not in protocol:
                raise ValueError(f"Missing protocol key: {key}")
        family = protocol["model_class"].split(".")[-1]
        if family not in LABELS:
            raise ValueError(f"Unknown model class: {family}")
        if len(protocol["trajectories"]) != 1:
            raise ValueError("This diagnostic report expects the prespecified single initial condition")
        trajectory = protocol["trajectories"][0]
        seen = set()
        for row in rows:
            lead = row["lead"]
            if (row["trajectory"] != trajectory or row["origin"] != protocol["origin"]
                    or row["num_steps"] != protocol["num_steps"]
                    or row.get("ensemble_size", row.get("E")) != protocol["ensemble_size"]):
                raise ValueError("Row and protocol settings disagree")
            if lead in seen or lead < 1 or lead > protocol["horizon"]:
                raise ValueError("Duplicate or invalid lead")
            seen.add(lead)
            if lead > protocol["truth_leads"] and any(row.get(key) is not None for key in
                                                      ("rmse", "spread_skill_ratio", "truth_enstrophy")):
                raise ValueError("Truth-dependent diagnostics extend beyond available truth")
        rows.sort(key=lambda row: row["lead"])
        if [row["lead"] for row in rows] != list(range(1, rows[-1]["lead"] + 1)):
            raise ValueError("Diagnostic leads are not contiguous")
        run = {"directory": directory, "protocol": protocol, "result": result,
               "rows": rows, "family": family,
               "label": f"{LABELS[family]} ({protocol['num_steps']})"}
        if runs:
            for key in MATCH_KEYS:
                if protocol[key] != runs[0]["protocol"][key]:
                    raise ValueError(f"Unmatched long-rollout protocol: {key}")
        runs.append(run)
    if not runs:
        raise ValueError("No model directories provided")
    identities = [(run["family"], run["protocol"]["num_steps"]) for run in runs]
    if len(identities) != len(set(identities)):
        raise ValueError("Repeated model/budget")
    # Existing reference snapshots must agree whenever both runs have them.
    expected_truth = {}
    for run in runs:
        for row in run["rows"]:
            truth = row.get("truth_enstrophy")
            if truth is not None:
                if row["lead"] in expected_truth and not np.isclose(truth, expected_truth[row["lead"]], rtol=1e-7):
                    raise ValueError("Truth differs between models")
                expected_truth[row["lead"]] = truth
    return runs


def array(rows, key):
    return np.array([np.nan if row.get(key) is None else row[key] for row in rows], dtype=float)


def summarize_run(run):
    rows = run["rows"]
    finite = [row for row in rows if row.get("enstrophy_member_mean") is not None
              and np.isfinite(row["enstrophy_member_mean"]) and row["finite_member_fraction"] == 1]
    last = finite[-1] if finite else {}
    end = rows[-1]["lead"]
    late = [row for row in finite if row["lead"] > max(0, end - 512)]
    failures = [row["lead"] for row in rows if row["finite_member_fraction"] < 1]
    first_z = finite[0]["enstrophy_member_mean"] if finite else None
    final_z = last.get("enstrophy_member_mean")
    return {"model": run["label"], "model_class": run["family"],
            "status": run["result"].get("status"), "requested_horizon": run["protocol"]["horizon"],
            "recorded_last_lead": end, "last_all_finite_lead": last.get("lead"),
            "first_nonfinite_lead": min(failures) if failures else None,
            "last_finite_enstrophy": final_z,
            "last_finite_enstrophy_of_mean": last.get("enstrophy_of_mean"),
            "last_finite_rms_spread": last.get("rms_spread"),
            "last_finite_max_abs_vorticity": last.get("max_abs_vorticity"),
            "enstrophy_last_over_first_forecast": final_z / first_z if first_z else None,
            "last_window_mean_enstrophy": float(np.mean([row["enstrophy_member_mean"] for row in late])) if late else None,
            "last_window_first_lead": min(row["lead"] for row in late) if late else None,
            "cumulative_seconds": rows[-1].get("cumulative_seconds"),
            "cumulative_nfe": rows[-1].get("cumulative_nfe")}


def log_enstrophy_axis(axis, horizon):
    """Use true logarithmic axes; nonpositive values are omitted, never floored."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator

    axis.set_xscale("log", nonpositive="mask")
    axis.set_yscale("log", nonpositive="mask")
    axis.set_xlim(1, max(2, horizon))
    ticks = [10 ** power for power in range(int(np.log10(max(2, horizon))) + 1)]
    if horizon > 1 and horizon / ticks[-1] >= 2:
        ticks.append(horizon)
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.yaxis.set_minor_formatter(FuncFormatter(lambda value, _: ""))


def plot_runs(runs, destination, reference=None):
    protocol = runs[0]["protocol"]
    boundary = protocol["truth_leads"]
    horizon = max(run["rows"][-1]["lead"] for run in runs)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = [("enstrophy_member_mean", "Mean member enstrophy"),
               ("enstrophy_of_mean", "Enstrophy of ensemble mean"),
               ("rms_spread", "Unbiased RMS ensemble spread"),
               ("max_abs_vorticity", "Maximum |vorticity| across members/grid")]
    for run in runs:
        rows = run["rows"]
        x = array(rows, "lead")
        color = COLORS[run["family"]]
        for ax, (key, label) in zip(axes.flat, metrics):
            ax.plot(x, array(rows, key), label=run["label"], color=color, linewidth=1.3)
            ax.set(ylabel=label, xlabel="Forecast lead (stored snapshot intervals)")
        axes[0, 0].fill_between(x, array(rows, "enstrophy_member_q05"),
                                array(rows, "enstrophy_member_q95"), color=color, alpha=.12)
    truth = {row["lead"]: row["truth_enstrophy"] for run in runs for row in run["rows"]
             if row.get("truth_enstrophy") is not None}
    if truth:
        tx = sorted(truth)
        axes[0, 0].plot(tx, [truth[lead] for lead in tx], color="black", linewidth=1.6, label="Available truth")
    if reference is not None:
        axes[0, 0].axhspan(reference["enstrophy_q05"], reference["enstrophy_q95"],
                          color="gray", alpha=.15, label="Archive context 5–95% (not continuation)")
    for axis, (key, _) in zip(axes.flat, metrics):
        values = np.concatenate([array(run["rows"], key) for run in runs])
        positive = values[np.isfinite(values) & (values > 0)]
        if key.startswith("enstrophy"):
            log_enstrophy_axis(axis, max(horizon, boundary))
        elif len(positive) and positive.max() > 100 * max(1e-3, positive.min()):
            axis.set_yscale("symlog", linthresh=max(1e-3, positive.min()))
            axis.set_xlim(0, max(horizon, boundary))
        else:
            axis.set_xlim(0, max(horizon, boundary))
        axis.axvline(boundary, color="black", linestyle=":", linewidth=1)
        axis.grid(alpha=.2)
    axes[0, 0].text(boundary, .98, " End of available truth", rotation=90,
                    transform=axes[0, 0].get_xaxis_transform(), va="top", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    fig.suptitle(f"Autonomous NS rollout: one initial history, E={protocol['ensemble_size']}\n"
                 "Member bands are ensemble quantiles; bounded traces do not establish long-time accuracy")
    fig.savefig(destination / "autonomous_enstrophy_spread.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    correction = ((protocol["ensemble_size"] + 1) / protocol["ensemble_size"]) ** .5
    for run in runs:
        rows = [row for row in run["rows"] if row["lead"] <= boundary]
        x, color = array(rows, "lead"), COLORS[run["family"]]
        axes[0].plot(x, correction * array(rows, "rms_spread"), color=color,
                     linestyle="--", label=run["label"] + " corrected spread")
        axes[0].plot(x, array(rows, "rmse"), color=color, label=run["label"] + " RMSE")
        axes[1].plot(x, array(rows, "spread_skill_ratio"), color=color, label=run["label"])
    axes[0].set_ylabel("Vorticity units: corrected spread and RMSE")
    axes[1].set_ylabel("Corrected spread / RMSE")
    axes[1].axhline(1, color="black", linestyle=":", linewidth=1)
    for axis in axes:
        axis.set(xlabel="Forecast lead (stored snapshot intervals)", xlim=(1, max(2, boundary)))
        axis.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    fig.suptitle(f"Truth-supported spread and skill only: leads 1–{boundary}\n"
                 "One initial condition; no skill extrapolation beyond the stored reference")
    fig.savefig(destination / "truth_supported_spread_skill.png", dpi=160)
    plt.close(fig)


def plot_spatial_spectra(spectra, destination, extensions=("png",), figsize=(8, 4)):
    """Plot audited shell sums, separately for the first and last forecast step."""
    from matplotlib.ticker import FixedLocator, FuncFormatter

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    k = np.asarray(spectra["k"])
    if k.ndim != 1 or not np.array_equal(k, np.arange(len(k))):
        raise ValueError("Expected consecutive integer frequency shells including DC")
    entries = spectra["models"]
    leads = sorted({entry["lead"] for entry in entries})
    if len(leads) != 2 or leads[0] != 1:
        raise ValueError("Expected first and last step spectra")
    filenames = []
    for lead, stem in zip(leads, ("first_step_enstrophy_spectrum", "last_step_enstrophy_spectrum")):
        fig, axis = plt.subplots(figsize=figsize, layout="constrained")
        for entry in entries:
            if entry["lead"] != lead:
                continue
            member_spectra = np.asarray(entry["spectrum_members"], dtype=float)
            if member_spectra.shape != (entry["ensemble_size"], len(k)) or not np.isfinite(member_spectra).all() or np.any(member_spectra < 0):
                raise ValueError("Invalid member spectra")
            if not np.allclose(member_spectra.sum(axis=1), entry["enstrophy_members"], rtol=1e-10, atol=1e-12):
                raise ValueError("Spectral shell sum violates Parseval enstrophy identity")
            mean = np.asarray(entry["spectrum_mean"])
            if not np.allclose(mean, member_spectra.mean(axis=0), rtol=1e-10, atol=1e-14):
                raise ValueError("Spectrum mean does not match member spectra")
            family = entry["model_class"]
            color = COLORS[family]
            axis.plot(k[1:], mean[1:], color=color, label=f"{LABELS[family]} ({entry['num_steps']})", linewidth=1.3)
            axis.fill_between(k[1:], np.asarray(entry["spectrum_q05"])[1:],
                              np.asarray(entry["spectrum_q95"])[1:], color=color, alpha=.13)
        truth = spectra.get("truth")
        if truth and truth["lead"] == lead:
            axis.plot(k[1:], np.asarray(truth["spectrum"])[1:], color="black", label="Matching truth", linewidth=1.2)
        axis.set_xscale("log", nonpositive="mask")
        axis.set_yscale("log", nonpositive="mask")
        axis.set(xlim=(1, k[-1]), xlabel="Spatial frequency k (cycles per grid domain)",
                 ylabel="Shell-summed enstrophy", title=f"{'First' if lead == 1 else 'Last'} forecast step (lead {lead:,})")
        axis.xaxis.set_major_locator(FixedLocator([value for value in (1, 2, 4, 8, 16, 32, 64) if value <= k[-1]]))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.xaxis.set_minor_formatter(FuncFormatter(lambda value, _: ""))
        axis.axvline(spectra["axis_nyquist"], color="gray", linestyle=":", linewidth=.8, label="Axis Nyquist")
        axis.grid(alpha=.2)
        axis.legend(fontsize="small", frameon=False)
        for extension in extensions:
            filename = f"{stem}.{extension}"
            fig.savefig(destination / filename, dpi=180)
            filenames.append(filename)
        plt.close(fig)
    return filenames


def build_report(directories, output_dir, reference_path=None, spectra_path=None):
    runs = load_runs(directories)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference = json.loads(Path(reference_path).read_text()) if reference_path else None
    plot_runs(runs, output, reference)
    if spectra_path:
        spectra = json.loads(Path(spectra_path).read_text())
        expected = {(run["family"], lead, run["protocol"]["trajectories"][0],
                     run["protocol"]["ensemble_size"], run["protocol"]["num_steps"])
                    for run in runs for lead in (1, run["protocol"]["horizon"])}
        actual = {(entry["model_class"], entry["lead"], entry["trajectory"],
                   entry["ensemble_size"], entry["num_steps"]) for entry in spectra["models"]}
        if actual != expected:
            raise ValueError("Spatial spectra do not match the report runs")
        plot_spatial_spectra(spectra, output)
    summary = [summarize_run(run) for run in runs]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    combined = [{"model": run["label"], **row} for run in runs for row in run["rows"]]
    fields = list(dict.fromkeys(key for row in combined for key in row))
    with (output / "all_diagnostics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(combined)
    protocol = runs[0]["protocol"]
    lines = ["# Long autonomous Navier–Stokes diagnostics", "",
             f"One held-out initial condition (trajectory {protocol['trajectories'][0]}, origin {protocol['origin']}), "
             f"E={protocol['ensemble_size']}, requested horizon {protocol['horizon']} stored snapshot intervals. "
             f"Truth supports only the first {protocol['truth_leads']} forecast leads.", "",
             "The plots use denormalized archive vorticity. Enstrophy is half the spatial mean of squared vorticity; "
             "member-mean enstrophy and enstrophy of the ensemble mean are distinct. Member bands show 5–95% ensemble quantiles, not confidence intervals. "
             "Both enstrophy panels use log-log axes from lead 1, without positive floors or modified diagnostics; nonpositive values cannot be shown on logarithmic axes. "
             "The spread/skill ratio uses unbiased RMS spread with the sqrt((E+1)/E) finite-ensemble correction.", "",
             "| Model (steps/frame) | Status | Last recorded lead | Last all-finite lead | First nonfinite lead | Last finite enstrophy | Last finite RMS spread | Generation seconds |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    if any(run["result"].get("status") not in ("complete", "nonfinite") for run in runs):
        lines[2:2] = ["**Partial development snapshot:** at least one run is still in progress. "
                      "Do not interpret the recorded endpoint as the requested final horizon.", ""]
    if {run["family"] for run in runs} != set(LABELS):
        lines[2:2] = ["**Partial model comparison:** this development report omits one or more of the four "
                      "prespecified model families. Completed curves do not constitute the complete study.", ""]
    def fmt(value):
        return "—" if value is None else f"{value:.5g}" if isinstance(value, (float, int)) else str(value)
    for row in summary:
        lines.append("| " + " | ".join(fmt(row[key]) for key in
                      ("model", "status", "recorded_last_lead", "last_all_finite_lead", "first_nonfinite_lead",
                       "last_finite_enstrophy", "last_finite_rms_spread", "cumulative_seconds")) + " |")
    if spectra_path:
        lines += ["", "The requested spatial spectra use log-log spatial-frequency axes and shell-summed enstrophy, "
                  "not shell-averaged power. All square-grid modes enter the Parseval sum; DC is retained in the audit "
                  "but omitted on the logarithmic axis. Shells above axis Nyquist contain corner modes only. "
                  "The first-step truth is matched; no last-step truth is available.", "",
                  "![First-step spatial enstrophy spectrum](first_step_enstrophy_spectrum.png)", "",
                  "![Last-step spatial enstrophy spectrum](last_step_enstrophy_spectrum.png)", ""]
    lines += ["", "The following scalar-over-lead plots are supplemental stability diagnostics, not spatial spectra.", "",
              "![Autonomous scalar enstrophy and spread](autonomous_enstrophy_spread.png)", "",
              "![Truth-supported spread and skill](truth_supported_spread_skill.png)", "",
              "Numerical boundedness over this one rollout does not establish stability for all initial conditions, "
              "correct invariant statistics, or accuracy beyond available truth. Generated trajectories are never clipped, "
              "reset to truth, or rescaled for these diagnostics. Curves end at the last recorded lead; absent curves after failure are not surviving-member averages. "
              "Recorded generation wall time is diagnostic and does not establish a controlled speedup, particularly when another process shared the device.", ""]
    if reference:
        lines += [f"The gray enstrophy band pools {reference.get('n_fields', 'recorded')} held-out archive fields. "
                  "It is contextual data, not a reference continuation or a stationary confidence interval.", ""]
    (output / "README.md").write_text("\n".join(lines))
    inputs = []
    for run in runs:
        for filename in ["protocol.json", "diagnostics.json"]:
            path = run["directory"] / filename
            inputs.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if reference_path:
        path = Path(reference_path)
        inputs.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if spectra_path:
        path = Path(spectra_path)
        inputs.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (output / "report_manifest.json").write_text(json.dumps({"inputs": inputs, "summary": summary}, indent=2, allow_nan=False) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="Per-model run directories")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Optional archive-context enstrophy JSON")
    parser.add_argument("--spectra", type=Path, help="Audited first/last-step spatial enstrophy spectra JSON")
    args = parser.parse_args(argv)
    build_report(args.inputs, args.output, args.reference, args.spectra)


if __name__ == "__main__":
    main()
