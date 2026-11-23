#!/usr/bin/env python3
"""True-IC enstrophy spectra (fig:ns_enstrophy_spectra) with the new true-IC runs added.

Prototype (figure audit 2026-11-17). Re-uses, WITHOUT editing them:
  * make_ns_figures._spectrum_figures (paper_experiments) for the drawing, file names and legend;
  * make_ns_wic_fields.tic_methods (results_tables) for WHICH state file each method uses,
    so the spectra show exactly the runs of the true-IC field figures (M=100, traj 11,
    16^2 Jac-free / FlowDAS+SURGE / FIG from the local tic16_m100 re-runs, 16^2 D-Flow at M=250).
Adds: Ours shared + corr. (tag --fc-tag, default fc25_8797d2ae = r=20), TMPD (non-finite -> skipped
by _spectrum_figures), SI-SDE and FM-ODE prior + EnKF. Jac-free + corr. was not run (skipped).
Only the SERIES styling table is extended in memory (prior + EnKF entries, "+ FC" -> "+ corr.").

    .venv/bin/python make_ns_true_ic_spectra.py [--fc-tag fc25_8797d2ae] --out DIR
Writes DIR/singles/ns_enstrophy_spectrum{,_zoom}_<scen>.pdf and ns_enstrophy_spectrum_legend.pdf.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PAPER = Path("/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants/papers/A Unified Framework for "
             "Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments")
TABLES = Path("/export/scratch2/ntm/delftblue_results/results_tables")
sys.path[:0] = [str(PAPER), str(TABLES)]

import numpy as np  # noqa: E402

import figure_common as fc  # noqa: E402
import make_ns_figures as mnf  # noqa: E402
import make_ns_wic_fields as wf  # noqa: E402

PE_DIR = Path("/export/scratch2/ntm/delftblue_results/results/navier_stokes_true_ic/prior_enkf/"
              "navier_stokes/states/traj11")


def patch_series():
    s = []
    for row in fc.SERIES:
        row = (row[0], row[1], row[2].replace(" + FC", " + corr."), *row[3:])
        s.append(row)
        if row[0] == "TMPD":
            s += [("SI-SDE prior + EnKF", None, "SI-SDE prior + EnKF", fc._SI, ":", "o", False),
                  ("FM-ODE prior + EnKF", None, "FM-ODE prior + EnKF", fc._FM, ":", "^", False)]
    fc.SERIES = tuple(s)


def records(slug: str) -> list[dict]:
    files = [f for s, f in wf.tic_methods(slug) if f is not None and s != "fm_ode_prior_enkf"]
    files += [PE_DIR / f"navier_stokes__{A}_prior_EnKF__{slug}__seed0__E64_M100.npz" for A in ("SI_SDE", "FM_ODE")]
    recs = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        rec = {k: d[k] for k in d.files}
        rec["_path"] = f
        recs.append(rec)
        print(f"  {slug}: {rec['method']} {rec['variant']} M={rec['M']}  <- {f}")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc-tag", default="fc25_8797d2ae", choices=("fc25_8797d2ae", "fc25_70a3134d"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    wf.TIC_CORR_TAG = a.fc_tag          # read at call time by wf.tic_files
    patch_series()
    mnf.load_state_records = lambda case, scenario=None, traj=11, M=None: records(
        re.sub(r"[^A-Za-z0-9]+", "_", scenario).strip("_"))
    for p in mnf._spectrum_figures(a.out, state_M=100):
        print("wrote", p)


if __name__ == "__main__":
    main()
