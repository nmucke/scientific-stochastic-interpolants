"""Combined per-method state animations for ONE trajectory (Navier--Stokes).

For each observation scenario, builds ONE animation showing the truth and every
method's posterior-mean vorticity field side by side, animated over the
assimilation time steps -- so you can watch how each method evolves and compare
them at every step. Trajectory 1 only.

Panels (in fixed order, those present for the scenario):
  Truth, EnKF [GT E=1000], Ours (SI-SDE/FM-SDE/FM-ODE), FlowDAS, Guided FM,
  Guided diffusion, SDA, EnKF-local, Particle filter, EnSF.

Usage:
    .venv/bin/python paper_experiments/make_state_animations.py \
        --traj 1 --fps 2 --out paper_experiments/results/multitraj/animations
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

SCEN_DISPLAY = {
    "sparse_5": "sparse 5%",
    "sparse_1_5625": "sparse 1.5625%",
    "32_2_128_2": "32^2 -> 128^2",
    "16_2_128_2": "16^2 -> 128^2",
}
SCENARIOS = ["32_2_128_2", "16_2_128_2", "sparse_5", "sparse_1_5625"]

# (display label, subdir, method token, E-tag in filename or None=any). Order = panel order.
PANELS = [
    ("EnKF [GT, E=1000]", "gt", "EnKF", "E1000"),
    ("Ours (SI-SDE)", "gen", "Ours_SI_SDE", None),
    ("Ours (FM-SDE)", "gen", "Ours_FM_SDE", None),
    ("Ours (FM-ODE)", "gen", "Ours_FM_ODE", None),
    ("FlowDAS", "gen", "FlowDAS", None),
    ("Guided FM", "gen", "Guided_FM", None),
    ("Guided diffusion", "gen", "Guided_diffusion", None),
    ("SDA", "gen", "SDA", None),
    ("EnKF-local", "conv", "EnKF", "E64"),
    ("Particle filter", "conv", "Particle_filter", None),
    ("EnSF", "conv", "Ensemble_score_filter", None),
]


def _find(base, sub, token, scen, etag):
    pat = f"navier_stokes__{token}__{scen}__seed0__*.npz"
    for f in sorted(glob.glob(str(Path(base) / sub / pat))):
        if etag is None or etag in Path(f).name:
            return f
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=int, default=1)
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--out", default="paper_experiments/results/multitraj/animations")
    args = ap.parse_args()
    base = f"paper_experiments/results/multitraj/states/traj{args.traj}"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for scen in SCENARIOS:
        # Collect (label, posterior_mean[H,W,T]); truth from any present file.
        fields = []
        truth = None
        for label, sub, token, etag in PANELS:
            f = _find(base, sub, token, scen, etag)
            if f is None:
                continue
            d = np.load(f, allow_pickle=True)
            post = d["posterior_trajectory"]  # [E,C,H,W,T]
            if not np.isfinite(post[..., -1]).all():
                print(f"  {scen}: skip {label} (non-finite)")
                continue
            fields.append((label, post[:, 0].mean(axis=0)))  # [H,W,T]
            if truth is None:
                truth = d["true_trajectory"][0, 0]  # [H,W,T]
        if not fields or truth is None:
            print(f"  {scen}: no data, skip")
            continue

        panels = [("Truth", truth)] + fields
        T = truth.shape[-1]
        vmin, vmax = float(truth.min()), float(truth.max())
        n = len(panels)
        ncol = 4
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.1 * nrow))
        axes = np.atleast_1d(axes).ravel()
        ims = []
        for k, (label, fld) in enumerate(panels):
            ax = axes[k]
            im = ax.imshow(fld[..., 0], vmin=vmin, vmax=vmax, cmap="RdBu_r", animated=True)
            ax.set_title(label, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ims.append((im, fld))
        for k in range(n, len(axes)):
            axes[k].axis("off")
        sup = fig.suptitle("", fontsize=13)

        def update(t):
            sup.set_text(f"NS {SCEN_DISPLAY.get(scen, scen)}  --  trajectory {args.traj}, step {t+1}/{T}")
            for im, fld in ims:
                im.set_array(fld[..., t])
            return [im for im, _ in ims] + [sup]

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        anim = animation.FuncAnimation(fig, update, frames=T, blit=False)
        slug = scen
        outpath = out_dir / f"states_traj{args.traj}__{slug}.mp4"
        anim.save(str(outpath), writer="ffmpeg", fps=args.fps, dpi=110)
        plt.close(fig)
        print(f"  wrote {outpath.name} ({n} panels, {T} frames)")


if __name__ == "__main__":
    main()
