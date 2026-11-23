#!/usr/bin/env python3
"""Metric vs number of SDE/ODE steps M for the multistep linear-Gaussian case and the
two Lorenz 96 cases (linear, arctan observations).

Four figures (pdf + png) in ./figures/:

    vs_M_ours_kl.*        our variants only, KL          (rows: case, cols: sampler)
    vs_M_ours_w2.*        our variants only, sliced W2
    vs_M_baselines_kl.*   baselines + all Ours (SI-SDE) variants, KL   (one panel per case)
    vs_M_baselines_w2.*   baselines + all Ours (SI-SDE) variants, sliced W2

"KL" is the Gaussian KL to the Kalman posterior (multistep) and the k-NN KL to the
particle-filter reference (Lorenz 96); "W2" is the sliced W2 to the same references.
Design follows the existing figures: msweep_all_variants (our variants, one panel per
sampler, mean +/- std bands) and lorenz96_vs_M_baselines (colours and line styles of
the Lorenz 96 lineup, M-independent methods as horizontal lines, a red ring on a point
whose mean excludes diverged runs). Values: mean +/- sample std over 5 seeds
(multistep) / 12 test sequences (Lorenz 96); a seed/sequence with any non-finite metric
is excluded (same rule as the tables).

    python3 make_vs_M_figures.py
"""

from __future__ import annotations

import glob
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import (FormatStrFormatter, LogFormatterMathtext, LogLocator,  # noqa: E402
                               NullFormatter)

RES = Path("/export/scratch2/ntm/delftblue_results/results")
OUT = Path(__file__).resolve().parent / "figures"
MS_M = (10, 25, 50, 100, 250)
L96_M = (25, 50, 100, 250)
CASES = (("ms", "Multistep linear\u2013Gaussian"), ("linear", "Lorenz 96, linear"),
         ("arctan", "Lorenz 96, arctan"))
METRIC = {  # figure key -> {case: (metric key, axis label)}
    "kl": {"ms": ("kl_points", "Gaussian KL to Kalman"),
           "linear": ("kl_knn", "$k$NN-KL to PF"), "arctan": ("kl_knn", "$k$NN-KL to PF")},
    "w2": {"ms": ("sliced_w2", "Sliced-$W_2$ to Kalman"),
           "linear": ("sliced_w2", "Sliced-$W_2$ to PF"), "arctan": ("sliced_w2", "Sliced-$W_2$ to PF")},
}
SAMPLERS = (("SI-SDE", "si_sde"), ("DM-SDE", "dm_sde"), ("FM-ODE", "fm_ode"))

# Our variants (msweep_all_variants palette): key -> label, colour, line style, marker
OURS_VARIANTS = (
    ("shared", "shared", "#0072B2", "-", "o"),
    ("jacfree", "Jac-free", "#D55E00", "-", "s"),
    ("shared_corr", "shared + corr.", "#009E73", "--", "D"),
    ("jacfree_corr", "Jac-free + corr.", "#E69F00", "--", "v"),
    ("surge_shared", "shared + SURGE", "#CC79A7", "-.", "^"),
    ("surge_jacfree", "Jac-free + SURGE", "#882255", "-.", "<"),
)
# Baselines (Lorenz 96 lineup colours, lorenz96_lineup.py): key -> label, colour, style
BASELINES = (
    ("flowdas", "FlowDAS", "#882255", "-"),
    ("surge_flowdas", "FlowDAS + SURGE", "#882255", ":"),
    ("sda", "SDA", "#44AA99", "-"),
    ("surge_sda", "SDA + SURGE", "#44AA99", ":"),
    ("dflow", "D-Flow SGLD", "#999933", "-"),
    ("fig", "Guided FM (FIG)", "#AA4499", "-"),
    ("tmpd", "TMPD", "#A6761D", "-"),
    ("prior_enkf", "Trained prior (SI) + EnKF", "#56B4E9", ":"),
    ("prior_enkf_fm", "Trained prior (FM) + EnKF", "#56B4E9", "-."),
)
# Baseline figures: only the SURGE version of a baseline that has one, and one prior + EnKF
BASELINE_FIG = ("surge_flowdas", "surge_sda", "dflow", "fig", "tmpd", "prior_enkf")
BASELINE_FIG_STYLE = {"prior_enkf": ("#332288", "-.")}   # off the blues of Ours (SI-SDE)
# All Ours (SI-SDE) variants on the baseline figures: key -> label, colour, style, marker, hollow
FM_SAMPLER = "si_sde"
FM_VARIANTS = (
    ("shared", "Ours (SI-SDE), shared", "#0072B2", "-", "o", False),
    ("jacfree", "Ours (SI-SDE), Jac-free", "#0072B2", "--", "s", True),
    ("shared_corr", "Ours (SI-SDE), shared + corr.", "#009E73", "-", "D", False),
    ("jacfree_corr", "Ours (SI-SDE), Jac-free + corr.", "#009E73", "--", "v", True),
)   # Ours + SURGE left out of the baseline figures (author decision)
# M-independent methods as horizontal lines: key -> label, colour, style
REFS = (
    ("floor", "Sampling floor", "#666666", "--"),
    ("enkf", "EnKF ($E=64$; multistep: $N=10^5$)", "#000000", ":"),
    ("enkf_large", "EnKF ($E=2000$)", "#000000", "-."),
    ("pf", "Particle filter", "#E69F00", ":"),
    ("oracle", "Oracle (exact one-step kernel)", "#D55E00", "--"),
    ("oracle_fc", "Oracle + corr.", "#D55E00", "-"),
)
OURS_REFS = ("floor", "enkf")


def mean_std(rows: list[dict], key: str) -> tuple[float, float, int, int]:
    """mean, std (ddof=1), n used, n excluded; a run with any non-finite metric is excluded."""
    ok = [r for r in rows if all(np.isfinite(v) for v in r.values())]
    xs = [r[key] for r in ok]
    if not xs:
        return (math.nan, math.nan, 0, len(rows))
    sd = float(np.std(xs, ddof=1)) if len(xs) > 1 else math.nan
    return (float(np.mean(xs)), sd, len(xs), len(rows) - len(ok))


# --------------------------------------------------------------------------- #
# loading: data[case][series key] = {M or None: per-run list of {metric: value}}
# --------------------------------------------------------------------------- #

MS_METRICS = ("kl_points", "sliced_w1", "sliced_w2")
L96_METRICS = ("sliced_w2", "kl_knn", "energy_distance")


def load_multistep() -> dict:
    frames = []
    for f in glob.glob(str(RES / "analytical_multistep/metrics/paper*__M*__K20__E100000__traj*.csv")):
        d = pd.read_csv(f)
        d["traj"] = int(re.search(r"__traj(\d+)\.csv", f).group(1))
        frames.append(d)
    d = pd.concat(frames)
    d["variant"] = d["variant"].fillna("")
    d = d[d.metric.isin(MS_METRICS)].drop_duplicates(["method", "variant", "M", "traj", "metric"])

    def runs(method, variant, M):
        s = d[(d.method == method) & (d.variant == variant) & (d.M == M)]
        if s.empty:
            return None
        w = s.pivot_table(index="traj", columns="metric", values="value", aggfunc="first")
        assert len(w) == 5, (method, variant, M)
        return [dict(r) for _, r in w.iterrows()]

    out: dict = {}
    meth = {a: f"Ours ({lab})" for lab, a in SAMPLERS}
    for lab, a in SAMPLERS:
        for key, var in (("shared", "shared"), ("jacfree", "jacfree"), ("shared_corr", "shared_fc{M}"),
                         ("jacfree_corr", "jacfree_fc{M}"), ("surge_shared", "surge_shared"),
                         ("surge_jacfree", "surge_jacfree")):
            out[(a, key)] = {M: r for M in MS_M if (r := runs(meth[a], var.format(M=M), M))}
    for key, method in (("flowdas", "FlowDAS"), ("surge_flowdas", "SURGE (FlowDAS)"), ("sda", "SDA"),
                        ("surge_sda", "SURGE (SDA)"), ("dflow", "D-Flow SGLD"), ("fig", "Guided FM (FIG)"),
                        ("tmpd", "TMPD")):
        out[key] = {M: r for M in MS_M if (r := runs(method, "", M))}
    # M-independent methods: rerun in every M file with the same seeds; take the M=250 file
    for key, method in (("floor", "Kalman filter (sampled)"), ("enkf", "EnKF"), ("pf", "Particle filter"),
                        ("oracle", "Exact per-member chain")):
        out[key] = {None: runs(method, "", 250)}
    return out


def load_l96(obs: str) -> dict:
    root = RES / "lorenz96" / obs / "per_sequence"

    def runs(cell, M):
        p = root / (f"lorenz96__{cell}__M{M}__traj0__.json" if M else f"lorenz96__{cell}__traj0__.json")
        if not p.exists():
            return None
        seq = json.load(open(p))
        n = len(seq["sliced_w2"])
        return [{k: float(seq[k][i]) for k in L96_METRICS} for i in range(n)]

    out: dict = {}
    for lab, a in SAMPLERS:
        for key, cell in (("shared", f"{a}-shared"), ("jacfree", a), ("shared_corr", f"{a}_fc-shared"),
                          ("jacfree_corr", f"{a}_fc"), ("surge_jacfree", f"surge_{a}")):
            out[(a, key)] = {M: r for M in L96_M if (r := runs(cell, M))}
    for key, _, _, _ in BASELINES:
        out[key] = {M: r for M in L96_M if (r := runs(key, M))}
    for key, cell in (("floor", "pf_subsample"), ("enkf", "enkf"), ("enkf_large", "enkf_large"),
                      ("pf", "pf_matched"), ("oracle", "oracle"), ("oracle_fc", "oracle_fc")):
        r = runs(cell, None)
        if r:
            out[key] = {None: r}
    return out


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #


def style() -> None:  # figure_common.apply_style
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.labelsize": 11,
                         "axes.titlesize": 11, "axes.linewidth": 0.9, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "legend.fontsize": 8.2, "figure.dpi": 150,
                         "mathtext.fontset": "dejavuserif"})


def log_axes(ax, Ms) -> None:
    """Log-log axes; y ticks at 1-2-5 x 10^k in plain numbers when the panel spans at
    most two decades, otherwise only at the decades, written as 10^k."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    if math.log10(hi / lo) > 2.0:
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
    else:
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(lo, hi)
    ax.set_xticks(Ms)
    ax.set_xticklabels([str(m) for m in Ms])
    ax.minorticks_off()
    ax.grid(True, which="major", alpha=0.25, lw=0.5)


def curve(ax, series: dict, key: str, colour, ls, marker, lw=1.4, ms=3.5, hollow=False,
          cap: float = math.inf) -> bool:
    """Mean line, +/- std band, red ring where runs were excluded; returns whether any ring.
    A mean above ``cap`` is drawn at the cap as an up-triangle labelled with its value."""
    Ms, mu, sd, ring = [], [], [], False
    for M in sorted(m for m in series if m is not None):
        m, s, n, nx = mean_std(series[M], key)
        if not np.isfinite(m) or m <= 0:
            continue
        y = m
        if m > cap:
            y, s = cap, 0.0
            ax.plot([M], [y], ls="none", marker="^", color=colour, ms=6, zorder=6, clip_on=False)
            e = int(math.floor(math.log10(m)))
            ax.annotate(rf"${m / 10**e:.1f}\cdot10^{{{e}}}$", (M, y), xytext=(-7, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=6.5, color=colour)
        Ms.append(M), mu.append(y), sd.append(s if np.isfinite(s) else 0.0)
        if nx:
            ring = True
    if not Ms:
        return False
    mu, sd = np.array(mu), np.array(sd)
    lo = np.where(mu - sd > 0, mu - sd, mu * 0.5)
    ax.fill_between(Ms, lo, mu + sd, color=colour, alpha=0.12, lw=0)
    ax.plot(Ms, mu, color=colour, ls=ls, lw=lw, marker=marker, ms=ms,
            mfc="white" if hollow else colour, mec=colour)
    return ring


def hline(ax, series: dict, key: str, colour, ls) -> None:
    m, _, _, _ = mean_std(series[None], key)
    if np.isfinite(m) and m > 0:
        ax.axhline(m, color=colour, ls=ls, lw=1.0, alpha=0.85)


def save(fig, name: str) -> list[Path]:
    OUT.mkdir(exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300 if ext == "png" else None)
        paths.append(p)
    plt.close(fig)
    return paths


def ours_figure(data: dict, fig_key: str) -> list[Path]:
    fig, axes = plt.subplots(3, 3, figsize=(7.4, 6.6), sharex="row")
    any_ring = False
    for i, (case, case_title) in enumerate(CASES):
        d = data[case]
        key, ylab = METRIC[fig_key][case]
        Ms = MS_M if case == "ms" else L96_M
        for j, (lab, a) in enumerate(SAMPLERS):
            ax = axes[i, j]
            for vkey, _, col, ls, mk in OURS_VARIANTS:
                s = d.get((a, vkey))
                if s:
                    any_ring |= curve(ax, s, key, col, ls, mk)
            for rkey, _, col, ls in REFS:
                if rkey in OURS_REFS and rkey in d:
                    hline(ax, d[rkey], key, col, ls)
            log_axes(ax, Ms)
            if i == 0:
                ax.set_title(f"Ours ({lab})")
            if i == 2:
                ax.set_xlabel("SDE/ODE steps $M$")
            if j == 0:
                ax.set_ylabel(f"{case_title}\n{ylab}", fontsize=9.5)
    handles = [Line2D([], [], color=c, ls=ls, marker=mk, ms=4, lw=1.4, label=lab)
               for _, lab, c, ls, mk in OURS_VARIANTS]
    handles += [Line2D([], [], color=c, ls=ls, lw=1.0, label=f"{lab} (no $M$)")
                for k, lab, c, ls in REFS if k in OURS_REFS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4,
               fontsize=8, frameon=False)
    fig.tight_layout()
    return save(fig, f"vs_M_ours_{fig_key}")


def baselines_figure(data: dict, fig_key: str) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    any_ring, used, used_refs, off_scale = False, set(), set(), False
    for j, (case, case_title) in enumerate(CASES):
        ax, d = axes[j], data[case]
        key, ylab = METRIC[fig_key][case]
        Ms = MS_M if case == "ms" else L96_M
        # off-scale cap: a mean more than 10^4 x the panel median (a collapsed ensemble)
        fm = [(d[(FM_SAMPLER, v)], v, *rest) for v, *rest in FM_VARIANTS if d.get((FM_SAMPLER, v))]
        means = [mean_std(s[M], key)[0]
                 for s in [d[b] for b in BASELINE_FIG if d.get(b)] + [f[0] for f in fm]
                 for M in s]
        means = [v for v in means if np.isfinite(v) and v > 0]
        med = float(np.median(means))
        cap = 3.0 * max(v for v in means if v <= 1e4 * med)
        for bkey, _, col, ls in BASELINES:
            if bkey in BASELINE_FIG and d.get(bkey):
                col, ls = BASELINE_FIG_STYLE.get(bkey, (col, ls))
                any_ring |= curve(ax, d[bkey], key, col, ls, "o", lw=1.2, ms=3, cap=cap)
                used.add(bkey)
        for s_, v, _, col, ls, mk, hollow in fm:
            any_ring |= curve(ax, s_, key, col, ls, mk, lw=1.4, ms=3.5, hollow=hollow, cap=cap)
        for rkey, _, col, ls in REFS:
            if rkey in d and not rkey.startswith("oracle") and rkey not in ("floor", "enkf_large"):
                hline(ax, d[rkey], key, col, ls)
                used_refs.add(rkey)
        log_axes(ax, Ms)
        if max(means) > cap:
            off_scale = True
            ax.set_ylim(top=cap * 4)   # headroom: the off-scale marker sits inside the panel
            log_axes(ax, Ms)
        ax.set_title(case_title, fontsize=10)
        ax.set_xlabel("SDE/ODE steps $M$")
        ax.set_ylabel(ylab, fontsize=9.5)
    handles = [Line2D([], [], color=c, ls=ls, marker=mk, ms=4, lw=1.4, mfc="white" if h else c,
                       label=lab) for _, lab, c, ls, mk, h in FM_VARIANTS]
    handles += [Line2D([], [], color=BASELINE_FIG_STYLE.get(k, (c, ls))[0],
                       ls=BASELINE_FIG_STYLE.get(k, (c, ls))[1], marker="o", ms=3, lw=1.2, label=lab)
                for k, lab, c, ls in BASELINES if k in used]
    handles += [Line2D([], [], color=c, ls=ls, lw=1.0, label=f"{lab} (no $M$)")
                for k, lab, c, ls in REFS if k in used_refs]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3,
               fontsize=8, frameon=False)
    fig.tight_layout()
    return save(fig, f"vs_M_baselines_{fig_key}")


def main() -> None:
    style()
    data = {"ms": load_multistep(), "linear": load_l96("linear"), "arctan": load_l96("arctan")}
    for fk in ("kl", "w2"):
        for p in ours_figure(data, fk) + baselines_figure(data, fk):
            print("wrote", p)


if __name__ == "__main__":
    main()
