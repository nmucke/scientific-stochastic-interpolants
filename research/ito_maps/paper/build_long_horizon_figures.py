"""Render audited long rollouts, retaining an explicit two-map preview until final."""

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
DATA = ROOT / "results/long_horizon"
spec = importlib.util.spec_from_file_location("long_report", ROOT / "experiments/report_long_horizon.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

EXPECTED_BUDGETS = {"FollmerStochasticInterpolant": 64, "FlowMatchingModel": 64,
                    "FollmerItoMapModel": 8, "FlowMapModel": 1}
TEX_LABELS = {"FollmerStochasticInterpolant": "Stochastic interpolant",
              "FlowMatchingModel": "Flow matching", "FollmerItoMapModel": r"It\^o map",
              "FlowMapModel": "Flow map"}


def load_verified_inputs(data=DATA, repo=PAPER.parents[2]):
    """Switch only on a complete final artifact pair; reject invalid final data."""
    data, repo = Path(data), Path(repo)
    final_audit = data / "independent_all_model_audit.json"
    final_spectra = data / "spatial_enstrophy_spectra_all_models.json"
    complete = final_audit.exists() and final_spectra.exists()
    audit_path = final_audit if complete else data / "independent_completed_map_audit.json"
    spectra_path = final_spectra if complete else data / "spatial_enstrophy_spectra_maps_preview.json"
    audit = json.loads(audit_path.read_text())
    expected_families = set(EXPECTED_BUDGETS) if complete else {"FollmerItoMapModel", "FlowMapModel"}
    if audit.get("status") != "passed":
        raise ValueError("Long-rollout audit has not passed")
    entries = audit["runs"]
    if len(entries) != len(expected_families) or {entry["model_class"] for entry in entries} != expected_families:
        raise ValueError("Audit does not contain the required model families")
    entries = sorted(entries, key=lambda entry: list(EXPECTED_BUDGETS).index(entry["model_class"]))
    directories = [repo / entry["run_directory"] for entry in entries]
    runs = report.load_runs(directories)
    sources = [audit_path, spectra_path]
    checks = {entry["model_class"]: entry for entry in entries}
    for run in runs:
        protocol = run["protocol"]
        if (run["result"]["status"] != "complete" or len(run["rows"]) != 4096
                or protocol["horizon"] != 4096 or protocol["truth_leads"] != 95
                or protocol["ensemble_size"] != 8 or protocol["trajectories"] != [180]
                or protocol["origin"] != 5
                or protocol["num_steps"] != EXPECTED_BUDGETS[run["family"]]
                or any(row["finite_member_fraction"] != 1 for row in run["rows"])):
            raise ValueError("Expected the prespecified complete, all-finite 4096-lead run")
        check = checks[run["family"]]
        for name in ("protocol", "diagnostics"):
            path = run["directory"] / f"{name}.json"
            if hashlib.sha256(path.read_bytes()).hexdigest() != check[f"{name}_sha256"]:
                raise ValueError(f"Audited {name} hash mismatch")
            sources.append(path)
    spectra = json.loads(spectra_path.read_text())
    expected = {(run["family"], lead, 180, 8, run["protocol"]["num_steps"])
                for run in runs for lead in (1, 4096)}
    actual = {(entry["model_class"], entry["lead"], entry["trajectory"], entry["ensemble_size"], entry["num_steps"])
              for entry in spectra["models"]}
    if len(spectra["models"]) != len(expected) or actual != expected or spectra.get("comparison_complete") != complete:
        raise ValueError("Spatial spectra do not match audited run scope")
    for entry in spectra["models"]:
        check = checks[entry["model_class"]]
        for name in ("protocol", "diagnostics"):
            if entry["provenance"][f"{name}_sha256"] != check[f"{name}_sha256"]:
                raise ValueError("Spatial spectra provenance differs from audited run")
    reference_path = data / "reference_enstrophy.json"
    reference = json.loads(reference_path.read_text())
    sources.append(reference_path)
    return runs, audit, spectra, reference, sources, complete


def main():
    runs, audit, spectra, reference, sources, complete = load_verified_inputs()
    FIGURES, GENERATED = PAPER / "figures", PAPER / "generated"
    FIGURES.mkdir(exist_ok=True)
    GENERATED.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    report.plot_spatial_spectra(spectra, FIGURES, extensions=("pdf", "png"), figsize=(6.7, 2.8))

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.65), layout="constrained")
    for run in runs:
        rows = run["rows"]
        x = report.array(rows, "lead")
        color = report.COLORS[run["family"]]
        axes[0].plot(x, report.array(rows, "enstrophy_member_mean"), color=color, label=run["label"], linewidth=.7)
        axes[0].fill_between(x, report.array(rows, "enstrophy_member_q05"), report.array(rows, "enstrophy_member_q95"), color=color, alpha=.10, linewidth=0)
        axes[1].plot(x, report.array(rows, "rms_spread"), color=color, label=run["label"], linewidth=.7)
    truth = runs[0]["rows"][:95]
    axes[0].plot(report.array(truth, "lead"), report.array(truth, "truth_enstrophy"), color="black", label="Available truth", linewidth=.7)
    axes[0].axhspan(reference["enstrophy_q05"], reference["enstrophy_q95"], color="gray", alpha=.13, label="Archive context 5–95%")
    for ax in axes:
        ax.axvline(95, color="black", linestyle=":", linewidth=.8)
        ax.set(xlabel="Forecast lead (snapshot intervals)", xlim=(0, 4096))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Mean member enstrophy")
    axes[1].set_ylabel("Unbiased RMS spread")
    report.log_enstrophy_axis(axes[0], 4096)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    for extension in ["pdf", "png"]:
        fig.savefig(FIGURES / f"long_horizon_maps.{extension}", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.65), layout="constrained")
    for run in runs:
        rows = run["rows"][:95]
        x = report.array(rows, "lead")
        color = report.COLORS[run["family"]]
        axes[0].plot(x, np.sqrt(9/8) * report.array(rows, "rms_spread"), color=color,
                     linestyle="--", label=run["label"] + " corrected spread", linewidth=1)
        axes[0].plot(x, report.array(rows, "rmse"), color=color, label=run["label"] + " RMSE", linewidth=1)
        axes[1].plot(x, report.array(rows, "spread_skill_ratio"), color=color, linewidth=1)
    axes[1].axhline(1, color="black", linestyle=":", linewidth=.8)
    for ax in axes:
        ax.set(xlabel="Forecast lead (snapshot intervals)", xlim=(1, 95))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Spread and RMSE (vorticity units)")
    axes[1].set_ylabel("Corrected spread / RMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    for extension in ["pdf", "png"]:
        fig.savefig(FIGURES / f"long_horizon_maps_skill.{extension}", dpi=200)
    plt.close(fig)

    scope = "All four prespecified model trajectories are complete." if complete else "Completed map trajectories only; solver references remain pending."
    table = [r"\begin{table*}[t]", r"\centering", r"\caption{" + scope + r" Late statistics average leads 3,585--4,096. $Q_{1:95}$ is the ratio of pooled corrected RMS spread to pooled RMSE over the available-reference interval; it is not the mean of per-lead ratios. All eight members remain finite through lead 4,096.}",
             r"\label{tab:long-maps}", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Model (steps/frame) & Mean $Z$, leads 1--95 & Late mean $Z$ & Late mean $S_{\mathrm{RMS}}$ & $Q_{1:95}$ & Max. $|\omega|$ \\", r"\midrule"]
    metrics = {}
    for run in runs:
        rows = run["rows"]
        z = report.array(rows, "enstrophy_member_mean")
        s = report.array(rows, "rms_spread")
        e = report.array(rows[:95], "rmse")
        q = np.sqrt(9/8) * np.sqrt(np.mean(s[:95]**2)) / np.sqrt(np.mean(e**2))
        label = TEX_LABELS[run["family"]] + f" ({run['protocol']['num_steps']})"
        maximum = max(row["max_abs_vorticity"] for row in rows)
        table.append(f"{label} & {z[:95].mean():.4f} & {z[-512:].mean():.4f} & {s[-512:].mean():.4f} & {q:.4f} & {maximum:.3f} " + r"\\")
        metrics[run["family"]] = {"enstrophy_first95": float(z[:95].mean()), "late_enstrophy": float(z[-512:].mean()),
                                  "late_spread": float(s[-512:].mean()), "pooled_spread_skill_first95": float(q)}
    table += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (GENERATED / "long_horizon_maps_table.tex").write_text("\n".join(table) + "\n")
    (GENERATED / "long_horizon_maps_provenance.json").write_text(json.dumps({
        "scope": scope,
        "all_models_complete": complete,
        "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sources],
        "metrics": metrics,
    }, indent=2) + "\n")

    records = sum(entry["all_scalar_rows_checked"] for entry in audit["runs"])
    snapshots = sum(entry["saved_raw_files_checked"] for entry in audit["runs"])
    teacher_text = "The pending teacher trajectories are needed to separate inherited teacher behavior from additional map error."
    if complete:
        si, fm = metrics["FollmerStochasticInterpolant"], metrics["FlowMatchingModel"]
        teacher_text = (
            f"For the completed 64-step references, late mean member enstrophy is {si['late_enstrophy']:.4f} "
            f"for the stochastic interpolant and {fm['late_enstrophy']:.4f} for flow matching; "
            f"their pooled corrected spread/skill ratios over leads 1--95 are {si['pooled_spread_skill_first95']:.4f} "
            f"and {fm['pooled_spread_skill_first95']:.4f}, respectively. "
            "These matched trajectories enable teacher comparisons, but one history and different numerical diffusion rules do not establish a causal attribution.")
    status = [r"\newif\iflongAllModels", r"\longAllModelstrue" if complete else r"\longAllModelsfalse",
              r"\newcommand{\longScopeCaption}{" + scope + "}",
              r"\newcommand{\longAuditSentence}{An independent CPU audit checks all " + f"{records:,}" +
              " scalar records and all " + f"{snapshots:,}" + " saved raw-field snapshots from these runs.}",
              r"\newcommand{\longTeacherComparison}{" + teacher_text + "}"]
    (GENERATED / "long_horizon_status.tex").write_text("\n".join(status) + "\n")


if __name__ == "__main__":
    main()
