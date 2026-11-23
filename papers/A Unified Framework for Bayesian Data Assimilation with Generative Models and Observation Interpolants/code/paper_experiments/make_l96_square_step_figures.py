#!/usr/bin/env python3
"""Lorenz 96 (square observations): metrics versus assimilation step for each method.

Reads the per-step files a scored (or ``--rescore``d) cell writes,
``<root>/lorenz96/<obs>/per_step/lorenz96__<cell>[__M<M>]__traj<seed>__.csv``
(mean over the run's test sequences per step), and adds the mode-mass error per step:
mean over observed sites of |P_method(x>0) - P_PF(x>0)|, from the saved ensembles and
the PF reference (replicate 0; replicate 1 vs 0 is drawn as the reference floor).

    python make_l96_square_step_figures.py --root results/_runs/sq1_quick/lorenz96 --M 100 \\
        --out results/figures/l96_square/quick1_M100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

# key, cell, uses M, label, colour, line style
METHODS = (
    ("enkf", "enkf", False, "EnKF", "#E69F00", "-"),
    ("shared", "si_sde-shared", True, "Ours SI-SDE (shared)", "#0072B2", "-"),
    ("shared_corr", "si_sde_fc-shared", True, "Ours SI-SDE (shared) + corr.", "#009E73", "--"),
    ("jacfree", "si_sde", True, "Ours SI-SDE (Jac-free)", "#CC79A7", "-."),
)
PANELS = (
    ("sliced_w2", "sliced $W_2$ to PF"),
    ("kl_knn", "$k$NN-KL to PF"),
    ("energy_distance", "energy distance to PF"),
    ("mode_mass", r"$|\Delta P(x>0)|$, observed sites"),
    ("rmse", "RMSE of the mean (truth)"),
    ("spread", "ensemble spread"),
)


def label(cell: str, uses_m: bool, m: int) -> str:
    return f"{cell}__M{m}" if uses_m else cell


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="the .../lorenz96 directory")
    ap.add_argument("--obs", default="square")
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-sequences", type=int, default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    base = a.root / a.obs
    a.out.mkdir(parents=True, exist_ok=True)

    # PF reference sign masses, replicates 0 and 1 ([S, T, d])
    def ref(i, r):
        return torch.load(base / f"reference/test/sequence{i}_replicate{r}.pt", weights_only=False)

    series, ens_s = {}, None
    for key, cell, uses_m, *_ in METHODS:
        p = base / "per_step" / f"lorenz96__{label(cell, uses_m, a.M)}__traj{a.seed}__.csv"
        e = base / "ensembles" / f"{label(cell, uses_m, a.M)}__seed{a.seed}.pt"
        if not p.exists() or not e.exists():
            print(f"skip {key}: {'per_step' if not p.exists() else 'ensembles'} missing")
            continue
        d = pd.read_csv(p)
        w = d.pivot_table(index="step", columns="metric", values="value")
        x = torch.load(e, weights_only=False).double()          # [S, T, E, d]
        S = x.shape[0] if a.num_sequences is None else min(a.num_sequences, x.shape[0])
        ens_s = S if ens_s is None else min(ens_s, S)
        series[key] = (w, x[:S])
    if not series:
        raise SystemExit("no scored method found")
    S = ens_s
    refs = [(ref(i, 0), ref(i, 1)) for i in range(S)]
    problem = refs[0][0]["problem"]
    stride = int(problem.get("obs_stride", 4))
    d = refs[0][0]["samples"].shape[-1]
    obs_idx = np.arange(0, d, stride)

    def pf_pos(r):
        if "positive_fraction" in r:
            return r["positive_fraction"].double().numpy()
        return (r["samples"] > 0).double().mean(1).numpy()

    p0 = np.stack([pf_pos(r0) for r0, _ in refs])[:, :, obs_idx]      # [S, T, n_obs]
    p1 = np.stack([pf_pos(r1) for _, r1 in refs])[:, :, obs_idx]
    floor = np.abs(p1 - p0).mean(axis=(0, 2))
    for key, (w, x) in series.items():
        pm = (x[:, :, :, obs_idx] > 0).double().mean(2).numpy()       # [S, T, n_obs]
        w["mode_mass"] = np.abs(pm - p0).mean(axis=(0, 2))[: len(w)]

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.2))
    style = {m[0]: m for m in METHODS}
    for ax, (metric, title) in zip(axes.flat, PANELS):
        for key, (w, _) in series.items():
            if metric not in w:
                continue
            _, _, _, lab, col, ls = style[key]
            ax.plot(w.index + 1, w[metric], color=col, ls=ls, lw=1.3, label=lab)
        if metric == "mode_mass":
            ax.plot(np.arange(1, len(floor) + 1), floor, color="0.4", ls=":", lw=1.2,
                    label="PF replicate 1 vs 0")
        if metric in ("kl_knn", "sliced_w2"):   # energy distance can be <= 0 (unbiased estimate)
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("assimilation step")
        ax.grid(alpha=0.25)
    handles, labels = [], []
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    seqs = f"{S} test sequence{'s' if S > 1 else ''}"
    op = (problem.get("observation") or {}).get("operator", a.obs)
    h_tex = {"square": "x^2", "abs": "|x|", "arctan": r"\arctan(x)", "linear": "x"}.get(op, op)
    fig.suptitle(a.title or f"Lorenz 96, $h(x)={h_tex}$: metrics per step, $M={a.M}$ ({seqs})", y=0.99)
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    stem = a.out / f"l96_{a.obs}_metrics_vs_step_M{a.M}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    print("wrote", stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
