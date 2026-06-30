"""Assess which large-ensemble conventional filter is the best ground-truth posterior (Case 2, NS).

Loads the E=1000 saved states written by the conventional baselines
(``+save_states=true``, one ``.npz`` per method x scenario) and, for each
scenario, produces:

  * a FIELD panel  (``gt_fields__<scenario>.png``)   -- truth / posterior mean /
    |error| / posterior spread at the final assimilated step, one row per method;
  * a DIAGNOSTIC panel (``gt_diagnostics__<scenario>.png``) -- ensemble-mean RMSE
    vs step, spread-skill ratio vs step, CRPS vs step, and the (coarse-binned)
    rank histogram at the final step, all methods overlaid;
  * a printed ASSESSMENT table -- final-step RMSE, spread-skill ratio (1 = well
    calibrated), CRPS, and calibration deviation, with a recommended candidate.

The "ground-truth posterior" we want is the filter whose ensemble is both
ACCURATE (low ensemble-mean RMSE) and CALIBRATED (spread-skill ratio near 1, flat
rank histogram). A particle filter that has collapsed shows a near-zero ratio and
a U-shaped rank histogram; an over-/under-dispersed Kalman filter shows a ratio
far from 1.

Usage:
    .venv/bin/python paper_experiments/analyze_groundtruth_posterior.py \
        --states paper_experiments/results/states_E1000 \
        --out paper_experiments/results/groundtruth_figures
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scisi.metrics.calibration import crps, rank_histogram, spread_skill

# Conventional filters under assessment, in a fixed display order.
METHOD_ORDER = ["EnKF", "LETKF", "Particle filter", "Ensemble score filter"]


def _load_cells(states_dir: Path) -> dict[str, dict[str, dict]]:
    """Map ``scenario -> method -> npz-contents`` for every saved cell found."""
    cells: dict[str, dict[str, dict]] = {}
    for npz_path in sorted(states_dir.glob("*.npz")):
        try:
            data = np.load(npz_path, allow_pickle=True)
            _ = data["posterior_trajectory"].shape  # force-read to catch partial writes
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            print(f"  (skipping {npz_path.name}: not readable yet -- {type(exc).__name__})")
            continue
        method = str(data["method"])
        scenario = str(data["scenario"])
        cells.setdefault(scenario, {})[method] = {
            "post": torch.from_numpy(data["posterior_trajectory"]).float(),  # [E,C,H,W,T]
            "true": torch.from_numpy(data["true_trajectory"]).float(),       # [1,C,H,W,T]
            "obs_indices": data["obs_indices"],
            "seconds": float(data["seconds_per_step"]),
            "E": int(data["E"]),
            "path": npz_path.name,
        }
    return cells


def _observed_mask(obs_indices: np.ndarray, c: int, h: int, w: int) -> torch.Tensor:
    """Boolean ``[C,H,W]`` keep-mask (True = observed sparse sensor location)."""
    mask = torch.zeros(c * h * w, dtype=torch.bool)
    idx = np.asarray(obs_indices).reshape(-1).astype(np.int64)
    if idx.size:
        mask[idx] = True
    return mask.reshape(c, h, w)


def _ordered(methods: list[str]) -> list[str]:
    known = [m for m in METHOD_ORDER if m in methods]
    return known + [m for m in methods if m not in METHOD_ORDER]


def _final_step(post: torch.Tensor) -> int:
    return post.shape[-1] - 1


def _len_history(post: torch.Tensor, true: torch.Tensor) -> int:
    """Recover the history prefix length: the first step where members disagree.

    Conventional savers pre-fill the history columns with the (broadcast) IC, so
    the assimilated columns are the ones with non-zero ensemble spread.
    """
    spread = post[:, 0].std(dim=0).flatten(1).mean(dim=1)  # [T]
    nonzero = torch.nonzero(spread > 0).flatten()
    return int(nonzero[0]) if nonzero.numel() else 0


def make_field_panel(scenario: str, by_method: dict[str, dict], out: Path) -> Path:
    methods = _ordered(list(by_method.keys()))
    n = len(methods)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.3 * n), squeeze=False)
    for r, m in enumerate(methods):
        post = by_method[m]["post"]
        true = by_method[m]["true"]
        t = _final_step(post)
        true_s = true[0, 0, :, :, t]
        mean_s = post[:, 0, :, :, t].mean(dim=0)
        std_s = post[:, 0, :, :, t].std(dim=0)
        err_s = (mean_s - true_s).abs()
        vmin, vmax = float(true_s.min()), float(true_s.max())
        axes[r][0].imshow(true_s, vmin=vmin, vmax=vmax, cmap="RdBu_r")
        axes[r][0].set_ylabel(m, fontsize=11)
        axes[r][1].imshow(mean_s, vmin=vmin, vmax=vmax, cmap="RdBu_r")
        im2 = axes[r][2].imshow(err_s, cmap="magma")
        plt.colorbar(im2, ax=axes[r][2], fraction=0.046)
        im3 = axes[r][3].imshow(std_s, cmap="viridis")
        plt.colorbar(im3, ax=axes[r][3], fraction=0.046)
        if r == 0:
            for c, title in enumerate(["Truth", "Posterior mean", "|mean - truth|", "Spread (std)"]):
                axes[0][c].set_title(title, fontsize=12)
        for c in range(4):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
    fig.suptitle(f"NS ground-truth posterior candidates -- {scenario} (final step, E=1000)", y=1.0)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def make_diagnostics(scenario: str, by_method: dict[str, dict], out: Path) -> tuple[Path, list[dict]]:
    methods = _ordered(list(by_method.keys()))
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    summary: list[dict] = []

    for m in methods:
        post = by_method[m]["post"]
        true = by_method[m]["true"]
        c, h, w = post.shape[1], post.shape[2], post.shape[3]
        L = _len_history(post, true)
        T = post.shape[-1]
        steps = list(range(L, T))

        rmse_curve, ratio_curve, crps_curve = [], [], []
        for t in steps:
            ens = post[:, 0, :, :, t]
            tgt = true[0, 0, :, :, t]
            ss = spread_skill(ens, tgt)
            rmse_curve.append(float(ss["skill"]))
            ratio_curve.append(float(ss["ratio"]))
            crps_curve.append(float(crps(ens, tgt)))

        axes[0].plot(steps, rmse_curve, marker="o", label=m)
        axes[1].plot(steps, ratio_curve, marker="o", label=m)
        axes[2].plot(steps, crps_curve, marker="o", label=m)

        # Rank histogram at the final step, coarse-binned (E=1000 -> 25 bins).
        t = _final_step(post)
        ranks = rank_histogram(post[:, 0, :, :, t], true[0, 0, :, :, t]).float().cpu().numpy()
        nbins = 25
        # E+1 rank bins (= 1001 at E=1000) -> nbins coarse groups (uneven split OK).
        binned = np.array([g.sum() for g in np.array_split(ranks, nbins)], dtype=float)
        binned = binned / binned.sum()
        axes[3].plot(np.linspace(0, 1, nbins), binned, marker=".", label=m)

        # Calibration deviation: how far the final-step rank histogram is from flat.
        cal_dev = float(np.abs(binned - 1.0 / nbins).sum())
        summary.append({
            "method": m,
            "rmse": rmse_curve[-1],
            "ratio": ratio_curve[-1],
            "crps": crps_curve[-1],
            "cal_dev": cal_dev,
            "seconds": by_method[m]["seconds"],
        })

    axes[0].set_title("ensemble-mean RMSE vs step")
    axes[0].set_xlabel("assimilation step")
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    axes[1].set_title("spread-skill ratio (1 = calibrated)")
    axes[1].set_xlabel("assimilation step")
    axes[2].set_title("CRPS vs step")
    axes[2].set_xlabel("assimilation step")
    axes[3].axhline(1.0 / 25, color="k", ls="--", lw=1)
    axes[3].set_title("rank histogram (final step)")
    axes[3].set_xlabel("normalized rank")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle(f"NS ground-truth posterior diagnostics -- {scenario} (E=1000)", y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="paper_experiments/results/states_E1000")
    ap.add_argument("--out", default="paper_experiments/results/groundtruth_figures")
    args = ap.parse_args()

    states_dir = Path(args.states)
    out_dir = Path(args.out)
    cells = _load_cells(states_dir)
    if not cells:
        print(f"No .npz states found in {states_dir} yet.")
        return

    for scenario, by_method in cells.items():
        slug = scenario.replace(" ", "_").replace("%", "pct").replace("^", "")
        fpath = make_field_panel(scenario, by_method, out_dir / f"gt_fields__{slug}.png")
        dpath, summary = make_diagnostics(scenario, by_method, out_dir / f"gt_diagnostics__{slug}.png")
        print(f"\n=== Scenario: {scenario} ===  (figures: {fpath.name}, {dpath.name})")
        print(f"{'method':<24}{'RMSE':>9}{'spread/skill':>14}{'CRPS':>9}{'cal_dev':>9}{'s/step':>9}")
        for s in summary:
            print(f"{s['method']:<24}{s['rmse']:>9.4f}{s['ratio']:>14.3f}"
                  f"{s['crps']:>9.4f}{s['cal_dev']:>9.3f}{s['seconds']:>9.1f}")
        # Recommend: accurate AND calibrated. Score = rmse + |1-ratio| + crps.
        ranked = sorted(summary, key=lambda s: s["rmse"] + abs(1 - s["ratio"]) + s["crps"])
        print(f"  -> best ground-truth-posterior candidate: {ranked[0]['method']} "
              f"(low RMSE + ratio near 1 + low CRPS)")


if __name__ == "__main__":
    main()
