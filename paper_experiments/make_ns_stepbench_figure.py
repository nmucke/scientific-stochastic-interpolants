"""Navier--Stokes step-count benchmark figure (metrics vs number of SDE/ODE steps).

Reads the per-(M, trajectory) CSVs written by ``run_ns_stepbench.sh`` under
``paper_experiments/results/stepbench/csv/`` and makes a 4-panel figure (RMSE,
CRPS, spread-skill, KL-to-EnKF(E=1000)) versus the integration step count
M in {50,100,250,500}, one line per method, each metric averaged over the 5 test
trajectories. Mirrors the analytical KL-vs-steps figure (square panels, shared
legend below). Saves PDF + PNG to ``manuscript/figures/navier_stokes/``.

Run:  .venv/bin/python paper_experiments/make_ns_stepbench_figure.py
"""
from __future__ import annotations

import glob
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import re

CSV_DIR = "paper_experiments/results/stepbench/csv"
OUT_DIR = "manuscript/figures/navier_stokes"
# One figure per observation scenario (trajectory 1).
SCENARIOS = ["32^2->128^2", "16^2->128^2", "sparse 5%", "sparse 1.5625%"]

# (metric key in CSV, panel title). spread_skill is reported as |1 - spread/skill|.
METRICS = [
    ("rmse", "Vorticity RMSE"),
    ("crps", "CRPS"),
    ("spread_skill", "Spread--skill |1-ratio|"),
    ("kl_points", "KL to EnKF (E=1000)"),
]
# Method display order + consistent colours (tab10).
METHODS = [
    "Ours (SI-SDE)", "Ours (FM-SDE)", "Ours (FM-ODE)", "FlowDAS",
    "Guided FM (FIG)", "Guided FM (OT-ODE)", "SDA", "SURGE",
]
COLORS = {m: c for m, c in zip(METHODS, plt.cm.tab10(np.linspace(0, 1, 10)))}


def _load():
    """Return values[scenario][metric][method][M] = list of per-trajectory values."""
    vals = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"No CSVs in {CSV_DIR} -- run run_ns_stepbench.sh first.")
    for f in files:
        with open(f) as fh:
            header = fh.readline().strip().split(",")
            idx = {k: i for i, k in enumerate(header)}
            for line in fh:
                p = line.rstrip("\n").split(",")
                try:
                    scen = p[idx["scenario"]]
                    method = p[idx["method"]]
                    metric = p[idx["metric"]]
                    M = int(p[idx["M"]])
                    v = float(p[idx["value"]])
                except (ValueError, KeyError, IndexError):
                    continue
                if np.isfinite(v):
                    vals[scen][metric][method][M].append(v)
    return vals


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def main() -> None:
    vals = _load()
    os.makedirs(OUT_DIR, exist_ok=True)
    for scen in SCENARIOS:
        sv = vals.get(scen)
        if not sv:
            print(f"  (no data for scenario {scen!r} -- skipping)")
            continue
        fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0))
        for ax, (mkey, title) in zip(axes, METRICS):
            for method in METHODS:
                curve = sv.get(mkey, {}).get(method, {})
                if not curve:
                    continue
                Ms = sorted(curve)
                means = [float(np.mean(curve[M])) for M in Ms]
                ax.plot(Ms, means, "o-", color=COLORS[method], label=method, markersize=4)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("sampler steps $M$")
            ax.set_title(title)
            ax.set_xticks([50, 100, 250, 500])
            ax.set_xticklabels(["50", "100", "250", "500"])
            ax.grid(True, which="both", ls=":", alpha=0.4)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(f"Navier--Stokes step-count benchmark — {scen} (trajectory 1)", y=1.02)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        name = f"ns_stepbench_{_slug(scen)}"
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), bbox_inches="tight", dpi=180)
        plt.close(fig)
        print(f"wrote {OUT_DIR}/{name}.pdf (+png)")


if __name__ == "__main__":
    main()
