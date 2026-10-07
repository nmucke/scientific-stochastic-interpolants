"""Render completed long-rollout maps without implying completed teacher tests."""

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
runs = report.load_runs([DATA / "follmer_ito_map", DATA / "flow_map"])
audit_path = DATA / "independent_completed_map_audit.json"
audit = json.loads(audit_path.read_text())
assert audit["status"] == "passed"
sources = [audit_path]
for run in runs:
    assert run["result"]["status"] == "complete" and len(run["rows"]) == 4096
    check = next(a for a in audit["runs"] if a["model_class"] == run["family"])
    for name in ["protocol", "diagnostics"]:
        path = run["directory"] / f"{name}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == check[f"{name}_sha256"]
        sources.append(path)
reference_path = DATA / "reference_enstrophy.json"
reference = json.loads(reference_path.read_text())
sources.append(reference_path)
FIGURES, GENERATED = PAPER / "figures", PAPER / "generated"
FIGURES.mkdir(exist_ok=True)
GENERATED.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

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

table = [r"\begin{table*}[t]", r"\centering", r"\caption{Completed map trajectories only; solver references remain pending. Late statistics average leads 3,585--4,096. $Q_{1:95}$ is the ratio of pooled corrected RMS spread to pooled RMSE over the available-reference interval; it is not the mean of per-lead ratios. All eight members remain finite through lead 4,096.}",
         r"\label{tab:long-maps}", r"\small", r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Model (steps/frame) & Mean $Z$, leads 1--95 & Late mean $Z$ & Late mean $S_{\mathrm{RMS}}$ & $Q_{1:95}$ & Max. $|\omega|$ \\", r"\midrule"]
metrics = {}
for run in runs:
    rows = run["rows"]
    z = report.array(rows, "enstrophy_member_mean")
    s = report.array(rows, "rms_spread")
    e = report.array(rows[:95], "rmse")
    q = np.sqrt(9/8) * np.sqrt(np.mean(s[:95]**2)) / np.sqrt(np.mean(e**2))
    label = r"It\^o map (8)" if run["family"] == "FollmerItoMapModel" else "Flow map (1)"
    maximum = max(row["max_abs_vorticity"] for row in rows)
    table.append(f"{label} & {z[:95].mean():.4f} & {z[-512:].mean():.4f} & {s[-512:].mean():.4f} & {q:.4f} & {maximum:.3f} " + r"\\")
    metrics[run["family"]] = {"enstrophy_first95": float(z[:95].mean()), "late_enstrophy": float(z[-512:].mean()),
                              "late_spread": float(s[-512:].mean()), "pooled_spread_skill_first95": float(q)}
table += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(GENERATED / "long_horizon_maps_table.tex").write_text("\n".join(table) + "\n")
(GENERATED / "long_horizon_maps_provenance.json").write_text(json.dumps({
    "scope": "two completed maps; 64-step solver references pending",
    "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sources],
    "metrics": metrics,
}, indent=2) + "\n")
