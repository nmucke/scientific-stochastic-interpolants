#!/usr/bin/env python
"""Figures for the Phase 0 JSON outputs (no model or data access).

  plot_phase0.py --results research/ito_maps/results/phase0

Writes ``d1_ratio_to_white.png`` (D1), ``d5_by_lead.png`` (D5/D6 short) and
``d7_spectrum.png`` (D7) next to whichever JSON files exist.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_d1(path, out):
    d = json.loads(path.read_text())
    fig, ax = plt.subplots(figsize=(6, 4))
    last = d["incomplete_annuli_from_shell"]
    for name, t in d["tables"].items():
        r = np.array(t["ratio_all_shells"], dtype=float)
        k = np.arange(len(r))
        sel = (k >= 1) & (k < last)
        ax.plot(k[sel], r[sel], label=name)
        lo = np.array(t["ratio_q05_all_shells"], dtype=float)[sel]
        hi = np.array(t["ratio_q95_all_shells"], dtype=float)[sel]
        ax.fill_between(k[sel], lo, hi, alpha=0.2)
    for level, style in ((1, "-"), (3, ":"), (1 / 3, ":")):
        ax.axhline(level, color="k", lw=0.8, ls=style)
    ax.set(xscale="log", yscale="log", xlabel="shell k (cycles per domain)",
           ylabel="per-mode power / (a^2/3)",
           title=f"D1: uncertainty vs white driver (a={d['model']['gamma_multiplier']})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "d1_ratio_to_white.png", dpi=150)
    plt.close(fig)


def plot_short(path, out):
    d = json.loads(path.read_text())
    leads = sorted(int(k) for k in d["by_lead"])
    get = lambda key: np.array([[d["by_lead"][str(l)][key][q] for q in ("mean", "q05", "q95")] for l in leads])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for key in ("variance_over_climatology", "injected_variance_over_climatology",
                "propagated_variance_over_climatology", "mse_ensemble_mean_over_climatology"):
        v = get(key)
        axes[0].plot(leads, v[:, 0], label=key.replace("_over_climatology", ""))
        axes[0].fill_between(leads, v[:, 1], v[:, 2], alpha=0.2)
    axes[0].set(xlabel="lead", ylabel="/ climatological variance", title="D5 variance")
    for key in ("pattern_corr_member_member", "pattern_corr_member_truth"):
        v = get(key)
        axes[1].plot(leads, v[:, 0], label=key)
        axes[1].fill_between(leads, v[:, 1], v[:, 2], alpha=0.2)
    axes[1].set(xlabel="lead", title="D5 pattern correlation")
    for key in ("residual_enstrophy_mean", "truth_residual_enstrophy",
                "residual_energy_mean", "truth_residual_energy"):
        v = get(key)
        axes[2].plot(leads, v[:, 0], label=key)
        axes[2].fill_between(leads, v[:, 1], v[:, 2], alpha=0.2)
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set(xlabel="lead", title="D6 residual / injection")
    for ax in axes:
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "d5_by_lead.png", dpi=150)
    plt.close(fig)


def plot_long(path, out):
    d = json.loads(path.read_text())
    m, r = d["stationary_model"]["shell_spectrum"], d["stationary_data_validation"]["shell_spectrum"]
    k = np.arange(len(m["mean"]))
    sel = (k >= 1) & (k <= 64)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(k[sel], np.array(r["mean"])[sel], "k", label="validation data")
    axes[0].fill_between(k[sel], np.array(r["q05"])[sel], np.array(r["q95"])[sel], color="k", alpha=0.2)
    axes[0].plot(k[sel], np.array(m["mean"])[sel], label="model members (after burn-in)")
    axes[0].fill_between(k[sel], np.array(m["q05"])[sel], np.array(m["q95"])[sel], alpha=0.2)
    axes[0].set(xscale="log", yscale="log", xlabel="shell k", ylabel="enstrophy per shell", title="D7 spectrum")
    axes[0].legend(fontsize=7)
    leads = [row["lead"] for row in d["per_lead"]]
    for key in ("residual_enstrophy_mean", "residual_energy_mean"):
        axes[1].plot(leads, [row[key] for row in d["per_lead"]], lw=0.6, label=key)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set(xlabel="lead", ylabel="residual / injection", title="D6 long run")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "d7_spectrum.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default="research/ito_maps/results/phase0")
    args = p.parse_args()
    out = Path(args.results)
    for name, fn in (("d1_uncertainty_spectrum.json", plot_d1),
                     ("d5_d6_short_si128_exact.json", plot_short),
                     ("d5_d6_d7_long_si128_exact.json", plot_long)):
        if (out / name).exists():
            fn(out / name, out)
            print("plotted", name)


if __name__ == "__main__":
    main()
