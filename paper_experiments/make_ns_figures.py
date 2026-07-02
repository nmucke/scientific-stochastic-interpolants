"""Produce ALL figures for the Navier--Stokes case.

Metric-vs-sampler-steps figures, each with all methods and one panel per
observation scenario (legend below):

* ``ns_rmse_vs_M``          -- ensemble-mean vorticity RMSE vs. $M$.
* ``ns_crps_vs_M``          -- CRPS vs. $M$.
* ``ns_spread_skill_vs_M``  -- spread--skill $|1-\\mathrm{spread}/\\mathrm{skill}|$ vs. $M$.

Metric-vs-assimilation-step figures (per-step curves, trajectory-averaged):

* ``ns_rmse_vs_step`` / ``ns_crps_vs_step`` / ``ns_spread_skill_vs_step``.

Plus qualitative vorticity field maps for trajectory 1 (one figure per scenario):

* ``ns_states_<scenario>``  -- rows = methods, cols = Truth / Posterior mean /
                               $|$error$|$ / Spread at the final assimilated step,
                               from ``results/navier_stokes/states/traj1/*.npz``.

Reads the aggregates produced by ``aggregate_ns.py``: ``aggregated/all.csv``
(metric-vs-M) and ``aggregated/per_step.csv`` (metric-vs-step); the field maps
read the saved-state archives directly. Any figure with no data yet is skipped
with a message. Run order: ``run_ns_grid.sh`` -> ``aggregate_ns.py`` -> this.

    python paper_experiments/make_ns_figures.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from figure_common import (  # noqa: E402
    FIGURES_DIR,
    SCENARIO_LABEL,
    load_metric_vs_M,
    load_metric_vs_step,
    load_state_records,
    make_state_field_figure,
    make_vs_M_figure,
    make_vs_step_figure,
    mirror_to,
)

DEFAULT_OUT = _here.parent / "manuscript" / "figures" / "navier_stokes"
CASE = "navier_stokes"
SCENARIOS = ("16^2->128^2", "32^2->128^2", "sparse 5%", "sparse 1.5625%")


def SLUG(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")

# (metric key, y-axis label, output stem).
METRIC_FIGURES = (
    ("rmse", r"Vorticity RMSE", "ns_rmse_vs_M"),
    ("crps", r"CRPS", "ns_crps_vs_M"),
    ("spread_skill", r"Spread--skill $|1-\mathrm{spread}/\mathrm{skill}|$",
     "ns_spread_skill_vs_M"),
)


def _step_figures(out: Path) -> list[Path]:
    """Metric-vs-assimilation-step figures (one per metric, one panel per scenario).

    Reads the trajectory-aggregated per-step curves from
    ``results/navier_stokes/aggregated/per_step.csv`` (``aggregate_ns.py``);
    skipped with a message if that file does not exist yet.
    """
    written: list[Path] = []
    for metric, ylabel, _stem in METRIC_FIGURES:
        panels = [
            (SCENARIO_LABEL.get(sc, sc), load_metric_vs_step(CASE, metric, sc))
            for sc in SCENARIOS
        ]
        paths = make_vs_step_figure(panels, ylabel, out / f"ns_{metric}_vs_step", ncols=2)
        written += paths
    if not written:
        print("[ns] no per-step data; run run_ns_grid.sh + aggregate_ns.py")
    return written


def _state_figures(out: Path) -> list[Path]:
    """Qualitative vorticity field maps for trajectory 1: one figure per scenario,
    rows = methods, cols = Truth | Posterior mean | |Error| | Spread (final step).

    Reads the self-contained ``results/navier_stokes/states/traj1/*.npz`` archives
    saved by ``run_ns_grid.sh``; skipped with a message if none exist yet.
    """
    written: list[Path] = []
    # Vorticity is the single (signed) state channel -> diverging colour map.
    field_fn = lambda traj: traj[:, 0, :, :, -1]  # noqa: E731  [n, H, W] final step
    for sc in SCENARIOS:
        recs = load_state_records(CASE, scenario=sc)
        if not recs:
            continue
        stem = out / f"ns_states_{SLUG(sc)}"
        paths = make_state_field_figure(
            recs, field_fn, stem, cbar_label="vorticity",
            cmap="RdBu_r", diverging=True,
        )
        written += paths
    if not written:
        print("[ns] no saved states; run run_ns_grid.sh (save_states, traj1) first")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out)
    written: list[Path] = []

    for metric, ylabel, stem in METRIC_FIGURES:
        panels = [
            (SCENARIO_LABEL.get(sc, sc), load_metric_vs_M(CASE, metric, sc))
            for sc in SCENARIOS
        ]
        paths = make_vs_M_figure(panels, ylabel, out / stem, ncols=2)
        if paths:
            written += paths
        else:
            print(f"[ns] no data for {metric}; run run_ns_grid.sh + aggregate_ns.py")

    written += _step_figures(out)
    written += _state_figures(out)

    # Mirror every figure into the in-repo paper_experiments/figures/ tree too.
    written += mirror_to(written, FIGURES_DIR / CASE)

    for p in written:
        print(f"[fig] wrote {p}")
    if not written:
        print("[ns] no figures produced (the NS grid has not been run yet).")


if __name__ == "__main__":
    main()
