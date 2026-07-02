"""Produce ALL figures for the Navier--Stokes case.

Metric-vs-sampler-steps figures, each with all methods and one panel per
observation scenario (legend below):

* ``ns_rmse_vs_M``          -- ensemble-mean vorticity RMSE vs. $M$.
* ``ns_crps_vs_M``          -- CRPS vs. $M$.
* ``ns_spread_skill_vs_M``  -- spread--skill $|1-\\mathrm{spread}/\\mathrm{skill}|$ vs. $M$.

Reads the reduced-grid results from ``results/navier_stokes/`` (the aggregated
``aggregated/all.csv`` produced by ``aggregate_grid.py``, else the per-cell
``metrics/*.csv``). If the NS grid has not run yet there is no data and the
figures are skipped with a message.

    python paper_experiments/make_ns_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from figure_common import SCENARIO_LABEL, load_metric_vs_M, make_vs_M_figure  # noqa: E402

DEFAULT_OUT = _here.parent / "manuscript" / "figures" / "navier_stokes"
CASE = "navier_stokes"
SCENARIOS = ("16^2->128^2", "32^2->128^2", "sparse 5%", "sparse 1.5625%")

# (metric key, y-axis label, output stem).
METRIC_FIGURES = (
    ("rmse", r"Vorticity RMSE", "ns_rmse_vs_M"),
    ("crps", r"CRPS", "ns_crps_vs_M"),
    ("spread_skill", r"Spread--skill $|1-\mathrm{spread}/\mathrm{skill}|$",
     "ns_spread_skill_vs_M"),
)


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
            print(f"[ns] no data for {metric}; run run_ns_grid.sh + aggregate_grid.py")

    for p in written:
        print(f"[fig] wrote {p}")
    if not written:
        print("[ns] no figures produced (the NS grid has not been run yet).")


if __name__ == "__main__":
    main()
