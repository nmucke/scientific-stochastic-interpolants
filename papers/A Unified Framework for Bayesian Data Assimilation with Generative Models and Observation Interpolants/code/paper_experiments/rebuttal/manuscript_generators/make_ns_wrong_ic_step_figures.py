#!/usr/bin/env python3
"""Navier--Stokes from a poor initial condition: metric vs assimilation step, for the
super-resolution scenarios 16^2 and 32^2 (no r=8 correction).

Per metric (rmse, crps, spread_skill) four single panels + two legends in ./figures/ns_wic/:

    ns_wic_<metric>_<set>_<scen>.pdf   set in {ours, baselines}, scen in {16_2_128_2, 32_2_128_2}
    ns_wic_legend_<set>.pdf

Design follows the existing manuscript figures (figure_common.py, the appendix
"long horizon" per-step figures): colour = sampler, filled/open marker = shared/Jac-free,
line style = variant for our samplers; the figure_common baseline palette, the EnKF in
black, the particle filter in grey, and our FM-ODE (Jac-free) as a thick grey reference
on the baseline panels. Curves are means over the non-diverged trajectories (a trajectory
is diverged if its RMSE is NaN or exceeds 10 at any step, as in the tables); a note above
a panel lists every curve averaged over fewer than the 5 trajectories.

    python3 make_ns_wrong_ic_step_figures.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter  # noqa: E402

DASH = (0, (4.0, 1.8))                 # + corr.
DASHDOT = (0, (5.0, 1.5, 1.2, 1.5))    # + SURGE

NS = Path("/export/scratch2/ntm/delftblue_results/results/navier_stokes_wrong_ic_np40")
MAIN, EXT = NS / "main/navier_stokes/per_step", NS / "ext/navier_stokes/per_step"
PE, TM = NS / "prior_enkf/navier_stokes/per_step", NS / "tmpd/navier_stokes/per_step"
L20 = NS / "enkf_loc20_superres/production/per_step"
OUT = Path(__file__).resolve().parent / "figures" / "ns_wic"
TRAJS = (10, 11, 12, 13, 14)
GUARD = 10.0
FC20 = "fc25_8797d2ae"
SCENS = (("16_2_128_2", "16^2->128^2"), ("32_2_128_2", "32^2->128^2"))
METRICS = {"rmse": "Vorticity RMSE", "crps": "CRPS", "spread_skill": r"$|1-\mathrm{ratio}|$"}

_SI, _DM, _FM = "#0072B2", "#009E73", "#CC79A7"
SAMPLERS = (("si_sde", "SI-SDE", _SI, "o"), ("dm_sde", "DM-SDE", _DM, "s"), ("fm_ode", "FM-ODE", _FM, "^"))
SHORT = {"si_sde": "SI", "dm_sde": "DM", "fm_ode": "FM"}


def ours_series():
    """(key, label, files-pattern, colour, ls, marker, hollow, short note label)."""
    out = []
    for a, lab, col, mk in SAMPLERS:
        for cov, cl, hollow in (("shared_jac10", "shared", False), ("jacfree", "Jac-free", True)):
            ab = f"{SHORT[a]} {'sh.' if cov != 'jacfree' else 'JF'}"
            out.append((f"{a}_{cov}", f"{lab}, {cl}", f"{MAIN}/{{s}}__M100__traj{{t}}__ours_{cov}__{a}.csv",
                        col, "-", mk, hollow, ab))
            out.append((f"{a}_{cov}_fc", f"{lab}, {cl} + corr.",
                        f"{EXT}/{{s}}__M100__traj{{t}}__ours_{cov}__{FC20}__{a}.csv",
                        col, DASH, mk, hollow, ab + " +corr."))
            if a != "fm_ode":
                out.append((f"{a}_{cov}_surge", f"{lab}, {cl} + SURGE",
                            f"{EXT}/{{s}}__M100__traj{{t}}__surge_ours_{cov}__{a}.csv",
                            col, DASHDOT, mk, hollow, ab + " +SURGE"))
    return out


# figure_common.py palette for the baselines
BASE = (
    ("flowdas", "FlowDAS", f"{MAIN}/{{s}}__M100__traj{{t}}__baselines__flowdas.csv", "#D55E00", "-", "D", False),
    ("surge_flowdas", "FlowDAS + SURGE", f"{MAIN}/{{s}}__M100__traj{{t}}__baselines__surge_flowdas.csv", "#E69F00", "-", "v", False),
    ("sda", "SDA", f"{MAIN}/{{s}}__M100__traj{{t}}__baselines__sda.csv", "#8C510A", "-", "P", False),
    ("surge_sda", "SDA + SURGE", f"{MAIN}/{{s}}__M100__traj{{t}}__baselines__surge_sda.csv", "#6A3D9A", "-", "X", False),
    ("dflow", "D-Flow SGLD", f"{MAIN}/{{s}}__M100__traj{{t}}__baselines__dflow_sgld.csv", "#808000", "-", "h", False),
    ("fig", "Guided FM (FIG)", f"{MAIN}/{{s}}__M100__traj{{t}}__baselines__fig.csv", "#F0027F", "-", "*", False),
    ("tmpd", "TMPD", f"{TM}/{{s}}__M100__traj{{t}}__baselines__tmpd.csv", "#A6761D", "-", "<", False),
    ("prior_si", "SI-SDE prior + EnKF", f"{PE}/{{s}}__M100__traj{{t}}__prior_enkf__si_sde.csv", _SI, ":", "o", True),
    ("prior_dm", "DM-SDE prior + EnKF", f"{PE}/{{s}}__M100__traj{{t}}__prior_enkf__dm_sde.csv", _DM, ":", "s", True),
    ("prior_fm", "FM-ODE prior + EnKF", f"{PE}/{{s}}__M100__traj{{t}}__prior_enkf__fm_ode.csv", _FM, ":", "^", True),
    ("enkf", "EnKF", f"{MAIN}/{{s}}__traj{{t}}__classical__enkf.csv", "#000000", ":", "p", False),
    ("enkf_r20", "EnKF ($r=20$)", f"{L20}/{{s}}__traj{{t}}__classical__enkf.csv", "#000000", "-", "o", True),
    ("pf", "Particle filter", f"{MAIN}/{{s}}__traj{{t}}__classical__particle_filter.csv", "#666666", ":", "d", False),
)
REF = ("ref", "Ours (FM-ODE), Jac-free, reference", f"{MAIN}/{{s}}__M100__traj{{t}}__ours_jacfree__fm_ode.csv")
BASE_SHORT = {"flowdas": "FlowDAS", "surge_flowdas": "FlowDAS+SURGE", "sda": "SDA", "surge_sda": "SDA+SURGE",
              "dflow": "D-Flow", "fig": "FIG", "tmpd": "TMPD", "prior_si": "SI prior+EnKF",
              "prior_dm": "DM prior+EnKF", "prior_fm": "FM prior+EnKF", "enkf": "EnKF",
              "enkf_r20": "EnKF r=20", "pf": "PF"}


def load(pattern: str, slug: str, metric: str):
    """Mean per step over non-diverged trajectories; returns (steps, mean, n_ok, n_all)."""
    curves, n_all = [], 0
    for t in TRAJS:
        p = Path(pattern.format(s=slug, t=t))
        if not p.exists():
            continue
        n_all += 1
        d = pd.read_csv(p)
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


def panel(metric, set_, slug, series, ref=None):
    fig, ax = plt.subplots(figsize=(2.75, 1.6))
    notes = []
    if ref is not None:
        x, y, n, _ = load(ref[2], slug, metric)
        if x is not None:
            ax.plot(x, y, color="0.6", lw=2.6, alpha=0.8, zorder=1)
    for key, lab, pat, col, ls, mk, hollow, short in series:
        x, y, n, n_all = load(pat, slug, metric)
        if n < n_all or n_all < 5:
            notes.append(f"{short} {n}/5" + (" (all div.)" if n == 0 else ""))
        if x is None:
            continue
        ax.plot(x, y, color=col, ls=ls, lw=1.0, marker=mk, markevery=4, ms=3.2,
                mfc="white" if hollow else col, mec=col, mew=0.8, zorder=3)
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    subs = (1.0, 2.0, 5.0) if hi / lo < 30 else (1.0,)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(lo, hi)
    ax.set_xticks(range(0, 35, 8))
    ax.set_xlim(-0.5, 34.5)
    ax.set_xlabel("Assimilation step")
    ax.set_ylabel(METRICS[metric])
    ax.grid(True, alpha=0.25, lw=0.4)
    if notes:
        ax.text(0.99, 1.02, "; ".join(notes), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5.6, color="0.25")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"ns_wic_{metric}_{set_}_{slug}.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return p


def legend(set_, series, ref=False, ncol=4):
    handles = [Line2D([], [], color=col, ls=ls, lw=1.0, marker=mk, ms=3.5,
                      mfc="white" if hollow else col, mec=col, label=lab)
               for _, lab, _, col, ls, mk, hollow, _ in series]
    if ref:
        handles.append(Line2D([], [], color="0.6", lw=2.6, alpha=0.8, label=REF[1]))
    if set_ == "ours":   # pad FM-ODE (no SURGE) so every sampler fills one column
        handles += [Line2D([], [], ls="none", label=" ")] * (-len(handles) % ncol)
    fig = plt.figure(figsize=(5.6, 0.1))
    fig.legend(handles=handles, loc="center", ncol=ncol, frameon=False, fontsize=7,
               handlelength=2.6, columnspacing=1.0, handletextpad=0.5, labelspacing=0.3)
    p = OUT / f"ns_wic_legend_{set_}.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return p


def main():
    style()
    ours = ours_series()
    base = [(k, lab, pat, col, ls, mk, h, BASE_SHORT[k]) for k, lab, pat, col, ls, mk, h in BASE]
    written = []
    for metric in METRICS:
        for slug, _ in SCENS:
            written.append(panel(metric, "ours", slug, ours))
            written.append(panel(metric, "baselines", slug, base, ref=REF))
    written.append(legend("ours", ours, ncol=3))   # one column per sampler
    written.append(legend("baselines", base, ref=True, ncol=4))
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
