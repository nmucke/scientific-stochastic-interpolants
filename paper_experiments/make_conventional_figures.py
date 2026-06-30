"""Per-method + cross-method figures for the CONVENTIONAL filters (EnKF / PF / EnSF).

Same layout as ``make_method_figures.py`` (which it reuses), but the conventional
methods live in two state dirs and the two EnKF variants must be disambiguated:

  * ``states_E1000_noloc/``  -> EnKF (global, non-localized)  = the KL ground-truth
                                reference; available for all 4 scenarios.
  * ``states_E1000/``        -> EnKF (local, distance-localized), Particle filter,
                                Ensemble score filter (EnSF, fixed); sparse only.

KL-at-points is taken against the E=1000 non-localized EnKF (so the global EnKF's
own KL is ~0 by construction -- it IS the reference).

Usage:
    .venv/bin/python paper_experiments/make_conventional_figures.py \
        --seed 0 --out paper_experiments/results/method_figures

    # point at one trajectory's state dirs:
    .venv/bin/python paper_experiments/make_conventional_figures.py \
        --noloc paper_experiments/results/multitraj/states/traj1/gt \
        --loc   paper_experiments/results/multitraj/states/traj1/conv \
        --out   paper_experiments/results/multitraj/figures
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

# Reuse the figure builders + metric series from the generative figure script.
from make_method_figures import (
    SCEN_DISPLAY,
    _kl_point_split,
    _load,
    _series,
    compare_figure,
    per_method_figure,
    radial_kinetic_energy_spectrum,
)

DEFAULT_NOLOC = "paper_experiments/results/states_E1000_noloc"
DEFAULT_LOC = "paper_experiments/results/states_E1000"
SCENARIOS = ["sparse_5", "sparse_1_5625", "32_2_128_2", "16_2_128_2"]


def _sources(noloc, loc):
    """(display label, source dir, method token in filename). Order = legend order."""
    return [
        ("EnKF (global)", noloc, "EnKF"),
        ("EnKF (local)", loc, "EnKF"),
        ("Particle filter", loc, "Particle_filter"),
        ("EnSF", loc, "Ensemble_score_filter"),
    ]


def _find(directory, token, scen, seed):
    m = glob.glob(str(Path(directory) / f"navier_stokes__{token}__{scen}__seed{seed}__*.npz"))
    return m[0] if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noloc", default=DEFAULT_NOLOC,
                    help="E=1000 non-localized EnKF dir (global EnKF = KL reference)")
    ap.add_argument("--loc", default=DEFAULT_LOC,
                    help="E=1000 localized EnKF / PF / EnSF dir")
    ap.add_argument("--out", default="paper_experiments/results/method_figures")
    args = ap.parse_args()
    out_dir = Path(args.out)
    sources = _sources(args.noloc, args.loc)

    for scen in SCENARIOS:
        scen_disp = SCEN_DISPLAY.get(scen, scen)
        refp = _find(args.noloc, "EnKF", scen, args.seed)
        ref = torch.from_numpy(np.load(refp, allow_pickle=True)["posterior_trajectory"]).float() if refp else None

        compare = {}
        truth_final = None
        for label, directory, token in sources:
            fp = _find(directory, token, scen, args.seed)
            if fp is None:
                continue
            post, true, data = _load(fp)
            if not torch.isfinite(post[..., -1]).all():
                print(f"  skip {label}/{scen}: non-finite states")
                continue
            c, h, w = post.shape[1], post.shape[2], post.shape[3]
            pidx, pis = _kl_point_split(data["obs_indices"], c, h, w) if ref is not None else (None, None)
            series = _series(post, true, ref, pidx, pis)
            slug = label.replace(" ", "_").replace("(", "").replace(")", "")
            per_method_figure(label, scen_disp, post, true, series,
                              out_dir / f"conv__{slug}__{scen}.png")
            steps, rmse, cr, ss, kl = series
            kk, ek = radial_kinetic_energy_spectrum(post[:, 0, :, :, -1].mean(0))
            compare[label] = {"steps": steps, "rmse": rmse, "crps": cr, "ss": ss, "kl": kl, "spec": (kk, ek)}
            truth_final = true[0, 0, :, :, -1]
            print(f"  wrote conv__{slug}__{scen}.png")
        if compare:
            compare_figure(scen_disp, compare, truth_final, out_dir / f"compare_conventional__{scen}.png")
            print(f"  wrote compare_conventional__{scen}.png")


if __name__ == "__main__":
    main()
