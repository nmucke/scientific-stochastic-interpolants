"""Experiment 1 (multistep linear-Gaussian, ``paper`` system): error vs the number of
SDE/ODE steps M.

Reads the per-seed metric files ``results/analytical_multistep/metrics/paper*__M<M>__K20__
E100000__traj<s>.csv`` directly (each value is already the average over the K = 20
cycles) and plots, against M, the mean +/- std over seeds of

  * ``kl_points``  -- KL to the Kalman posterior (the Exp 1 table metric), and
  * ``sliced_w2``  -- sliced W2 to the Kalman posterior.

Figures (``figures/analytical_multistep/``):

  * ``msweep_all_variants``: one column per sampler (SI-SDE, DM-SDE, FM-ODE), every
    variant of Ours -- shared / Jac-free Jacobian, + resampling (adapted), + forecast
    correction (S = M), inside SURGE (SDEs only) -- with the sampled Kalman floor and the
    particle filter as horizontal references (both independent of M).
  * ``msweep_fc_vs_nofc``: SI-SDE, DM-SDE, FM-ODE with the shared Jacobian, with
    (solid) and without (dashed) the forecast correction, the generative baselines
    (FlowDAS, SDA, their SURGE versions, D-Flow, FIG; single-step hyperparameters) and
    EnKF, against the same references.

KL here is the Gaussian (moment) KL of the Exp 1 tables (user 2026-11-11); sliced W2
is the nonparametric metric.

The plotted numbers go to ``results/analytical_multistep/aggregated/msweep.csv``.
Grouped (``_g<E>``) and residual-bank (``_bank``) variants are left out; the correction
is the one whose forecast uses S = M steps.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from figure_common import FIGURES_DIR, _save_fig, apply_style  # noqa: E402

ROOT = _here / "results" / "analytical_multistep"
OUT = FIGURES_DIR / "analytical_multistep"
K, E = 20, 100000
FILE = re.compile(rf"^paper[^_]*(?:_[a-z0-9]+)*__M(\d+)__K{K}__E{E}__traj(\d+)\.csv$")

METRICS = (("kl_points", "KL to Kalman posterior"), ("sliced_w2", r"Sliced $W_2$ to Kalman posterior"))
SAMPLERS = (("Ours (SI-SDE)", "SI-SDE"), ("Ours (DM-SDE)", "DM-SDE"), ("Ours (FM-ODE)", "FM-ODE"))
SAMPLER_COLOURS = {"Ours (SI-SDE)": "#0072B2", "Ours (DM-SDE)": "#009E73", "Ours (FM-ODE)": "#CC79A7"}
KALMAN, PF, ENKF = "Kalman filter (sampled)", "Particle filter", "EnKF"

# Baselines on the same prior (M-dependent), single-step hyperparameters (not re-tuned).
BASELINES = (
    ("FlowDAS", "FlowDAS", "#E69F00", "-.", "v"),
    ("SURGE (FlowDAS)", "SURGE (FlowDAS)", "#E69F00", ":", "v"),
    ("SDA", "SDA", "#D55E00", "-.", "P"),
    ("SURGE (SDA)", "SURGE (SDA)", "#D55E00", ":", "P"),
    ("D-Flow SGLD", "D-Flow (SGLD)", "#882255", "-.", "X"),
    ("Guided FM (FIG)", "FIG", "#56B4E9", "-.", "*"),
)

# Variant key (with "{M}" for the correction's forecast steps) -> label, colour, style, marker.
VARIANTS = (
    ("shared", "shared Jacobian", "#0072B2", "-", "o"),
    ("jacfree", "Jac-free", "#D55E00", "-", "s"),
    ("adapted_shared", "shared + resampling", "#56B4E9", ":", "o"),
    ("shared_fc{M}", "shared + FC", "#009E73", "--", "D"),
    ("jacfree_fc{M}", "Jac-free + FC", "#E69F00", "--", "v"),
    ("surge_shared", "SURGE, shared", "#CC79A7", "-.", "^"),
    ("surge_jacfree", "SURGE, Jac-free", "#882255", "-.", "<"),
    ("surge_unguided", "SURGE, prior only (no guidance)", "#777777", "-.", "x"),
)


def load(root: Path):
    """values[(method, variant, metric, M)] -> list of per-seed values."""
    values = defaultdict(list)
    seen = set()
    for path in sorted((root / "metrics").glob("paper*.csv")):
        match = FILE.match(path.name)
        if not match:
            continue
        m = int(match.group(1))
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if row["scenario"] != "multistep:paper" or int(row["M"]) != m:
                    continue
                variant = row["variant"] or ""
                if "_g" in variant or "bank" in variant:
                    continue
                key = (row["method"], variant, row["metric"], m)
                ident = (key, path.name)
                if ident in seen:
                    continue
                seen.add(ident)
                try:
                    values[key].append(float(row["value"]))
                except ValueError:
                    values[key].append(float("nan"))
    return values


def series(values, method, variant_template, metric, steps):
    """(Ms, means, stds, n) of one curve; the correction variant follows S = M."""
    out = []
    for m in steps:
        variant = variant_template.format(M=m)
        vals = np.asarray(values.get((method, variant, metric, m), []), dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out.append((m, vals.mean(), vals.std(), vals.size))
    if not out:
        return None
    ms, mean, std, n = map(np.asarray, zip(*out))
    return ms, mean, std, n


def reference(values, method, metric):
    """An M-independent reference: pooled over every M (one value per seed and M)."""
    # Plain rows only: e.g. EnKF also has a grouped variant (g64) in the group cells.
    vals = [v for (mt, var, me, _m), vs in values.items()
            if mt == method and var == "" and me == metric for v in vs]
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return (vals.mean(), vals.std()) if vals.size else None


def draw_references(ax, values, metric, enkf=False):
    refs = [(KALMAN, "Kalman (sampled floor)", "black", "--"), (PF, "Particle filter", "0.45", ":")]
    if enkf:
        refs.append((ENKF, "EnKF", "#999933", "-."))
    for method, label, colour, style in refs:
        ref = reference(values, method, metric)
        if ref is not None:
            ax.axhline(ref[0], color=colour, ls=style, lw=1.3, label=label, zorder=1)


def draw_curve(ax, s, label, colour, style, marker):
    ms, mean, std, _ = s
    ax.plot(ms, mean, ls=style, marker=marker, color=colour, lw=1.5, ms=5, label=label, zorder=3)
    lo = np.clip(mean - std, mean * 0.1, None)
    ax.fill_between(ms, lo, mean + std, color=colour, alpha=0.12, lw=0, zorder=2)


def style_axis(ax, steps, ylabel=None):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(steps)
    ax.set_xticklabels([str(m) for m in steps])
    ax.minorticks_off()
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.3)
    if ylabel:
        ax.set_ylabel(ylabel)


def fig_all_variants(values, steps, out: Path):
    fig, axes = plt.subplots(len(METRICS), len(SAMPLERS), figsize=(13.0, 7.4), sharey="row")
    handles = {}
    for col, (method, title) in enumerate(SAMPLERS):
        for row, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row, col]
            draw_references(ax, values, metric)
            for template, label, colour, style, marker in VARIANTS:
                s = series(values, method, template, metric, steps)
                if s is not None:
                    draw_curve(ax, s, label, colour, style, marker)
            style_axis(ax, steps, ylabel if col == 0 else None)
            if row == 0:
                ax.set_title(f"Ours {title}")
            if row == len(METRICS) - 1:
                ax.set_xlabel("SDE/ODE steps M")
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(r"Multistep linear-Gaussian ($A=Q=H=R=I$, $K=20$, $N=10^5$): "
                 "time-averaged error vs M (mean $\\pm$ std, 5 seeds)", y=0.995)
    fig.tight_layout(rect=(0, 0.09, 1, 0.98))
    return _save_fig(fig, out / "msweep_all_variants")


def fig_fc_vs_nofc(values, steps, out: Path):
    """Shared-Jacobian SI/DM/FM with and without FC, the generative baselines, EnKF."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(12.0, 4.8))
    for ax, (metric, ylabel) in zip(axes, METRICS):
        draw_references(ax, values, metric, enkf=True)
        for method, title in SAMPLERS:
            colour = SAMPLER_COLOURS[method]
            for template, suffix, style, marker in (
                ("shared_fc{M}", " shared + FC", "-", "o"),
                ("shared", " shared, no FC", "--", "s"),
            ):
                s = series(values, method, template, metric, steps)
                if s is not None:
                    draw_curve(ax, s, title + suffix, colour, style, marker)
        for method, label, colour, style, marker in BASELINES:
            s = series(values, method, "", metric, steps)
            if s is not None:
                draw_curve(ax, s, label, colour, style, marker)
        style_axis(ax, steps, ylabel)
        ax.set_xlabel("SDE/ODE steps M")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(r"Ours (shared Jacobian) with and without forecast correction (FC, $S=M$), "
                 "and baselines", y=1.0)
    fig.tight_layout(rect=(0, 0.2, 1, 0.97))
    return _save_fig(fig, out / "msweep_fc_vs_nofc")


def write_table(values, steps, path: Path):
    rows = []
    templates = sorted({t for t, *_ in VARIANTS})
    for method, _ in SAMPLERS:
        for template in templates:
            for metric, _ in METRICS:
                s = series(values, method, template, metric, steps)
                if s is None:
                    continue
                for m, mean, std, n in zip(*s):
                    rows.append(dict(method=method, variant=template.format(M=m), metric=metric,
                                     M=int(m), mean=mean, std=std, n_seeds=int(n)))
    for method, *_ in BASELINES:
        for metric, _ in METRICS:
            s = series(values, method, "", metric, steps)
            if s is not None:
                for m, mean, std, n in zip(*s):
                    rows.append(dict(method=method, variant="", metric=metric, M=int(m),
                                     mean=mean, std=std, n_seeds=int(n)))
    for method in (KALMAN, PF, ENKF):
        for metric, _ in METRICS:
            ref = reference(values, method, metric)
            if ref is not None:
                rows.append(dict(method=method, variant="", metric=metric, M="all",
                                 mean=ref[0], std=ref[1], n_seeds=""))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    apply_style()
    values = load(args.root)
    steps = sorted({k[3] for k in values})
    print("M values found:", steps)
    for method in (KALMAN, PF, ENKF):
        per_m = {m: np.mean(values[(method, "", "kl_points", m)]) for m in steps
                 if values.get((method, "", "kl_points", m))}
        print(f"{method} kl_points by M (should not depend on M):",
              {m: f"{v:.3g}" for m, v in per_m.items()})
    for path in (*fig_all_variants(values, steps, args.out), *fig_fc_vs_nofc(values, steps, args.out),
                 write_table(values, steps, args.root / "aggregated" / "msweep.csv")):
        print("wrote", path)


if __name__ == "__main__":
    main()
