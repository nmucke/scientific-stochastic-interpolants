"""Aggregate results/*.json into markdown tables and two figures."""
import glob, json, math, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from train import SETTINGS, CONFIGS

HERE = os.path.dirname(os.path.abspath(__file__))
NS = (1, 2, 4, 8, 16)
runs = defaultdict(list)
for f in sorted(glob.glob(os.path.join(HERE, "results", "*.json"))):
    r = json.load(open(f))
    runs[(r["setting"], r["name"])].append(r)
ref = json.load(open(os.path.join(HERE, "ref_check.json")))


def ms(vals, nd=3):
    v = np.array(vals)
    return f"{v.mean():.{nd}f}±{v.std(ddof=1) if len(v) > 1 else 0:.{nd}f}"


def monotone(r, tol=0.005):
    w = [r[str(n)]["w1"] for n in NS]
    return all(w[i + 1] <= w[i] + tol for i in range(len(w) - 1))


lines = []
order = [k for k in CONFIGS]
for setting in SETTINGS:
    names = [n for n in order if (setting, n) in runs]
    if not names:
        continue
    rf = ref[setting]
    lines.append(f"\n### {setting}  (EM-2048 reference: W1 {rf['ref']['w1']:.3f}, std ratio {rf['ref']['std_ratio']:.3f}; iid 50k sample W1 {rf['iid']['w1']:.3f})\n")
    for metric, title, nd in (("w1", "W1 to true p(y)", 3), ("std_ratio", "std ratio (sample / true)", 3),
                              ("strong", "pathwise RMSE vs EM-2048 on the same Brownian path", 3),
                              ("flip", "fraction of samples in a different mode than the same-path reference", 3),
                              ("mode_err", "max |mode weight - true|", 3)):
        if setting != "S0" and metric in ("flip", "mode_err"):
            continue
        lines.append(f"\n**{title}** (mean±std over seeds)\n")
        lines.append("| variant | seeds | " + " | ".join(f"{n} step" for n in NS) + (" | W1 monotone in steps (seeds) |" if metric == "w1" else " |"))
        lines.append("|---|---|" + "---|" * (len(NS) + (1 if metric == "w1" else 0)))
        for n_ in names:
            rs = runs[(setting, n_)]
            row = f"| {n_} | {len(rs)} | " + " | ".join(ms([r[str(n)][metric] for r in rs], nd) for n in NS)
            if metric == "w1":
                row += f" | {sum(monotone(r) for r in rs)}/{len(rs)}"
            lines.append(row + " |")
        for k, lab in (("EM", "EM, exact drift (no learning)"), ("EI", "exp. integrator, exact denoiser frozen at s")):
            if metric in rf[f"{k}1"]:
                lines.append(f"| _{lab}_ | - | " + " | ".join(f"{rf[f'{k}{n}'][metric]:.3f}" for n in NS) + (" | |" if metric == "w1" else " |"))
open(os.path.join(HERE, "results_table.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ------------------------------------------------------------------ figures
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e4dc"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 10})
main = ["V0", "V1", "V1s-rawfeat", "V0-legraw", "V0+ES1", "V0-legraw+ES1"]
main = [m for m in main if ("S0", m) in runs]
marks = ["o", "s", "^", "D", "v", "P"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (metric, title, logy) in zip(axes, (("w1", "W1 to true p(y)  (lower is better)", True),
                                             ("std_ratio", "std ratio  (1 = calibrated spread)", False),
                                             ("strong", "pathwise RMSE vs same-path EM-2048", False))):
    for i, n_ in enumerate(main):
        arr = np.array([[r[str(n)][metric] for n in NS] for r in runs[("S0", n_)]])
        m, sd = arr.mean(0), arr.std(0, ddof=1)
        ax.plot(NS, m, color=PAL[i], lw=2, marker=marks[i], ms=6, label=n_, markeredgecolor="white", markeredgewidth=0.8)
        ax.fill_between(NS, m - sd, m + sd, color=PAL[i], alpha=0.15, lw=0)
    ax.plot(NS, [ref["S0"][f"EM{n}"][metric] for n in NS], color=MUTED, lw=1.5, ls="--", label="EM with exact drift")
    if metric == "w1":
        ax.axhline(ref["S0"]["ref"]["w1"], color=MUTED, lw=1, ls=":")
        ax.text(16, ref["S0"]["ref"]["w1"] * 0.82, "EM-2048 reference", ha="right", va="top", color=MUTED, fontsize=8)
    if metric == "std_ratio":
        ax.axhline(1.0, color=MUTED, lw=1, ls=":")
        ax.set_ylim(0.9, 1.06)
    ax.set_xscale("log", base=2); ax.set_xticks(NS); ax.set_xticklabels(NS)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("map steps"); ax.set_title(title, loc="left", color=INK)
axes[0].legend(frameon=False, fontsize=8, ncol=2)
fig.suptitle("1D two-mode target (c=0, a=1, beta=t): mean ± std over 3 seeds, 50k samples", x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(HERE, "fig_metrics_vs_steps.png"), dpi=150)

st = SETTINGS["S0"]()
edges = np.linspace(-4, 4, 201); mid = 0.5 * (edges[1:] + edges[:-1])
g = st.gmm
dens = sum(float(w) * np.exp(-0.5 * ((mid - float(mu)) / float(sd)) ** 2) / (float(sd) * math.sqrt(2 * math.pi)) for w, mu, sd in zip(g.w, g.mu[:, 0], g.sd))
show = [m for m in ["V0", "V1", "V1s-rawfeat", "V0-legraw", "V1-ESonly", "V0+ES1", "V1+ES1", "V0-legraw+ES1"] if ("S0", m) in runs]
fig, axes = plt.subplots(2, len(show), figsize=(2.6 * len(show), 5.2), sharex=True, sharey=True)
for j, n_ in enumerate(show):
    r = sorted(runs[("S0", n_)], key=lambda r: r["seed"])[0]
    for i, key in enumerate(("hist1", "hist4")):
        ax = axes[i, j]
        h = np.array(r[key]) / (sum(r[key]) * (edges[1] - edges[0]))
        ax.fill_between(mid, h, step="mid", color="#2a78d6", alpha=0.55, lw=0, label="map samples")
        ax.plot(mid, dens, color=INK, lw=1.2, label="true p(y)")
        k = "1" if i == 0 else "4"
        ax.set_title(f"{n_}, {k} step{'s' if i else ''}\nW1 {r[k]['w1']:.3f}, std ratio {r[k]['std_ratio']:.2f}", loc="left", color=INK, fontsize=8.5)
        ax.set_xlim(-3, 3)
        if i == 1:
            ax.set_xlabel("y")
axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper center")
axes[0, 0].set_ylabel("density"); axes[1, 0].set_ylabel("density")
fig.suptitle("One-step (top) and four-step (bottom) samples vs truth, seed 0", x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(HERE, "fig_histograms.png"), dpi=150)
