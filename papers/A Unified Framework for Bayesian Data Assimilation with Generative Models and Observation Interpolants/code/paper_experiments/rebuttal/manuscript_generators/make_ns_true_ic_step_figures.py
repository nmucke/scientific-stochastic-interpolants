#!/usr/bin/env python3
"""Navier--Stokes from the TRUE initial condition: metric vs assimilation step, all four
scenarios, in the design of the poor-IC figures (make_ns_wrong_ic_step_figures.py).

Prototype copy (figure audit 2026-11-17); the generator in results_tables/ is untouched.

Per metric (rmse, crps, spread_skill) eight single panels + two legends in --out:

    ns_tic_<metric>_<set>_<scen>.pdf   set in {ours, baselines},
                                       scen in {32_2_128_2, 16_2_128_2, sparse_5, sparse_1_5625}
    ns_tic_legend_<set>.pdf

Sources (M=100, E=64, trajectories 10-14, 15 assimilation steps):
  * plain Ours, FlowDAS(+SURGE), SDA, D-Flow, FIG, EnKF, PF: results/navier_stokes/per_step
    (grouped CSVs, filtered by method/variant)
  * SURGE (SDA): results/navier_stokes/per_step/*__surge_sda.csv
  * Ours shared + corr.: results/navier_stokes_true_ic/ext  (tag = --fc-tag; r=20 by default,
    fc25_70a3134d = r=8). Jac-free + corr. exists for r=20 only.
  * TMPD: results/navier_stokes_true_ic/tmpd
  * SI-SDE / FM-ODE prior + EnKF (GC r=20): results/navier_stokes_true_ic/prior_enkf
A trajectory is diverged if its RMSE is NaN or exceeds 10 at any step (as in the tables);
the note above a panel lists every curve averaged over fewer than 5 trajectories.

    python3 make_ns_true_ic_step_figures.py [--fc-tag fc25_8797d2ae] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter  # noqa: E402

DASH = (0, (4.0, 1.8))                 # + corr.

RES = Path("/export/scratch2/ntm/delftblue_results/results")
OLD = RES / "navier_stokes/per_step"                       # original true-IC grid
NEW = RES / "navier_stokes_true_ic"
EXT, TM, PE = (NEW / "ext/navier_stokes/per_step", NEW / "tmpd/navier_stokes/per_step",
               NEW / "prior_enkf/navier_stokes/per_step")
FC_TAGS = {"fc25_8797d2ae": "r=20", "fc25_70a3134d": "r=8"}
TRAJS = (10, 11, 12, 13, 14)
GUARD = 10.0
SCENS = ("32_2_128_2", "16_2_128_2", "sparse_5", "sparse_1_5625")
METRICS = {"rmse": "Vorticity RMSE", "crps": "CRPS", "spread_skill": r"$|1-\mathrm{ratio}|$"}

_SI, _DM, _FM = "#0072B2", "#009E73", "#CC79A7"
SAMPLERS = (("si_sde", "SI-SDE", _SI, "o"), ("dm_sde", "DM-SDE", _DM, "s"), ("fm_ode", "FM-ODE", _FM, "^"))
SHORT = {"si_sde": "SI", "dm_sde": "DM", "fm_ode": "FM"}

# A source = (file pattern with {s}/{t}, method filter or None, variant filter or None).


def ours_series(fc_tag):
    """(key, label, source, colour, ls, marker, hollow, short note label)."""
    out = []
    for a, lab, col, mk in SAMPLERS:
        meth = f"Ours ({lab})"
        for cov, cl, hollow in (("shared_jac10", "shared", False), ("jacfree", "Jac-free", True)):
            ab = f"{SHORT[a]} {'sh.' if cov != 'jacfree' else 'JF'}"
            out.append((f"{a}_{cov}", f"{lab}, {cl}",
                        (f"{OLD}/{{s}}__M100__traj{{t}}__ours_{cov}.csv", meth, cov),
                        col, "-", mk, hollow, ab))
            if True:   # + corr. for both covariances (Jac-free + corr.: r=20 only)
                out.append((f"{a}_{cov}_fc", f"{lab}, {cl} + corr.",
                            (f"{EXT}/{{s}}__M100__traj{{t}}__ours_{cov}__{fc_tag}__{a}.csv", None, None),
                            col, DASH, mk, hollow, ab + " +corr."))
    return out


_B = f"{OLD}/{{s}}__M100__traj{{t}}__baselines.csv"
_C = f"{OLD}/{{s}}__traj{{t}}__classical.csv"
BASE = (  # figure_common.py / poor-IC palette
    ("flowdas", "FlowDAS", (_B, "FlowDAS", None), "#D55E00", "-", "D", False),
    ("surge_flowdas", "FlowDAS + SURGE", (_B, "SURGE (FlowDAS)", None), "#E69F00", "-", "v", False),
    ("sda", "SDA", (_B, "SDA", None), "#8C510A", "-", "P", False),
    ("surge_sda", "SDA + SURGE", (f"{OLD}/{{s}}__M100__traj{{t}}__surge_sda.csv", "SURGE (SDA)", None),
     "#6A3D9A", "-", "X", False),
    ("dflow", "D-Flow SGLD", (_B, "D-Flow SGLD", None), "#808000", "-", "h", False),
    ("fig", "Guided FM (FIG)", (_B, "Guided FM (FIG)", None), "#F0027F", "-", "*", False),
    ("tmpd", "TMPD", (f"{TM}/{{s}}__M100__traj{{t}}__tmpd.csv", None, None), "#A6761D", "-", "<", False),
    ("prior_si", "SI-SDE prior + EnKF", (f"{PE}/{{s}}__M100__traj{{t}}__prior_enkf_r20__si_sde.csv", None, None),
     _SI, ":", "o", True),
    ("prior_fm", "FM-ODE prior + EnKF", (f"{PE}/{{s}}__M100__traj{{t}}__prior_enkf_r20__fm_ode.csv", None, None),
     _FM, ":", "^", True),
    ("enkf", "EnKF", (_C, "EnKF", None), "#000000", ":", "p", False),
    ("pf", "Particle filter", (_C, "Particle filter", None), "#666666", ":", "d", False),
)
REF = ("ref", "Ours (FM-ODE), Jac-free, reference",
       (f"{OLD}/{{s}}__M100__traj{{t}}__ours_jacfree.csv", "Ours (FM-ODE)", "jacfree"))
BASE_SHORT = {"flowdas": "FlowDAS", "surge_flowdas": "FlowDAS+SURGE", "sda": "SDA", "surge_sda": "SDA+SURGE",
              "dflow": "D-Flow", "fig": "FIG", "tmpd": "TMPD", "prior_si": "SI prior+EnKF",
              "prior_fm": "FM prior+EnKF", "enkf": "EnKF", "pf": "PF"}


def load(src, slug: str, metric: str):
    """Mean per step over non-diverged trajectories; returns (steps, mean, n_ok, n_all)."""
    pattern, meth, var = src
    curves, n_all = [], 0
    for t in TRAJS:
        p = Path(pattern.format(s=slug, t=t))
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if meth is not None:
            d = d[d.method == meth]
        if var is not None:
            d = d[d.variant == var]
        if d.empty:
            continue
        n_all += 1
        r = d[d.metric == "rmse"].sort_values("step").value.to_numpy(float)
        if len(r) == 0 or np.isnan(r).any() or (r > GUARD).any():
            continue  # diverged trajectory: excluded whole
        v = d[d.metric == metric].sort_values("step")
        curves.append(v.set_index("step").value.astype(float))
    if not curves:
        return None, None, 0, n_all
    m = pd.concat(curves, axis=1).mean(axis=1)
    return m.index.to_numpy(), m.to_numpy(), len(curves), n_all


def style():
    plt.rcParams.update({"font.family": "serif", "font.size": 8.0, "axes.labelsize": 9.0,
                         "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "axes.linewidth": 0.55,
                         "legend.fontsize": 7.0, "mathtext.fontset": "dejavuserif"})


def panel(out, metric, set_, slug, series, ref=None):
    fig, ax = plt.subplots(figsize=(2.75, 1.6))
    notes = []
    if ref is not None:
        x, y, n, _ = load(ref[2], slug, metric)
        if x is not None:
            ax.plot(x, y, color="0.6", lw=2.6, alpha=0.8, zorder=1)
    for key, lab, src, col, ls, mk, hollow, short in series:
        x, y, n, n_all = load(src, slug, metric)
        if n < n_all or n_all < 5:
            notes.append(f"{short} {n}/5" + (" (all div.)" if n == 0 else ""))
        if x is None:
            continue
        ax.plot(x, y, color=col, ls=ls, lw=1.0, marker=mk, markevery=2, ms=3.2,
                mfc="white" if hollow else col, mec=col, mew=0.8, zorder=3)
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    subs = (1.0, 2.0, 5.0) if hi / lo < 30 else (1.0,)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(lo, hi)
    ax.set_xticks(range(0, 15, 2))
    ax.set_xlim(-0.3, 14.3)
    ax.set_xlabel("Assimilation step")
    ax.set_ylabel(METRICS[metric])
    ax.grid(True, alpha=0.25, lw=0.4)
    if notes:
        ax.text(0.99, 1.02, "; ".join(notes), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5.6, color="0.25")
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"ns_tic_{metric}_{set_}_{slug}.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return p


def legend(out, set_, series, ref=False, ncol=4):
    handles = [Line2D([], [], color=col, ls=ls, lw=1.0, marker=mk, ms=3.5,
                      mfc="white" if hollow else col, mec=col, label=lab)
               for _, lab, _, col, ls, mk, hollow, _ in series]
    if ref:
        handles.append(Line2D([], [], color="0.6", lw=2.6, alpha=0.8, label=REF[1]))
    fig = plt.figure(figsize=(5.6, 0.1))
    fig.legend(handles=handles, loc="center", ncol=ncol, frameon=False, fontsize=7,
               handlelength=2.6, columnspacing=1.0, handletextpad=0.5, labelspacing=0.3)
    p = out / f"ns_tic_legend_{set_}.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fc-tag", default="fc25_8797d2ae", choices=sorted(FC_TAGS),
                    help="forecast-correction run of the + corr. curves: fc25_8797d2ae (r=20, "
                         "default) or fc25_70a3134d (r=8)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figures" / "ns_tic")
    args = ap.parse_args()
    print(f"+ corr. = {args.fc_tag} ({FC_TAGS[args.fc_tag]})")
    style()
    ours = ours_series(args.fc_tag)
    base = [(k, lab, src, col, ls, mk, h, BASE_SHORT[k]) for k, lab, src, col, ls, mk, h in BASE]
    written = []
    for metric in METRICS:
        for slug in SCENS:
            written.append(panel(args.out, metric, "ours", slug, ours))
            written.append(panel(args.out, metric, "baselines", slug, base, ref=REF))
    written.append(legend(args.out, "ours", ours, ncol=3))   # rows: plain / + corr. / Jac-free
    written.append(legend(args.out, "baselines", base, ref=True, ncol=4))
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
