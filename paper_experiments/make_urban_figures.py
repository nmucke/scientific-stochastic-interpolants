"""Produce ALL figures for the urban (uDALES) case.

Metric-vs-sampler-steps figures, each with all methods and one panel per
observation scenario / variable (legend below). Urban is generative-only
(no EnKF/PF) and sparse-only (two sensor densities):

* ``urban_rmse_vs_M``          -- per-variable RMSE vs. $M$ (velocity + temperature
                                  x the two sparse scenarios, 2x2 panels).
* ``urban_crps_vs_M``          -- CRPS vs. $M$.
* ``urban_spread_skill_vs_M``  -- spread--skill vs. $M$.

Reads ``results/urban/`` (aggregated ``aggregated/all.csv`` from
``aggregate_grid.py``, else the per-cell ``metrics/*.csv``). Skipped with a
message if the urban grid has not run yet.

    python paper_experiments/make_urban_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from figure_common import SCENARIO_LABEL, load_metric_vs_M, make_vs_M_figure  # noqa: E402

DEFAULT_OUT = _here.parent / "manuscript" / "figures" / "urban"
CASE = "urban"
SCENARIOS = ("sparse 5%", "sparse 1.5625%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out)
    written: list[Path] = []

    # (1) RMSE: velocity + temperature x the two sparse scenarios (2x2 panels).
    rmse_panels: list[tuple[str, dict]] = []
    for var_metric, var_name in (
        ("rmse_velocity", "Velocity"), ("rmse_temperature", "Temperature")
    ):
        for sc in SCENARIOS:
            title = f"{var_name} -- {SCENARIO_LABEL.get(sc, sc)}"
            rmse_panels.append((title, load_metric_vs_M(CASE, var_metric, sc)))
    paths = make_vs_M_figure(
        rmse_panels, r"RMSE (fluid cells)", out / "urban_rmse_vs_M", ncols=2
    )
    written += paths
    if not paths:
        print("[urban] no RMSE data; run run_urban_grid.sh + aggregate_grid.py")

    # (2) CRPS and (3) spread--skill: one panel per sparse scenario.
    for metric, ylabel, stem in (
        ("crps", r"CRPS", "urban_crps_vs_M"),
        ("spread_skill", r"Spread--skill $|1-\mathrm{spread}/\mathrm{skill}|$",
         "urban_spread_skill_vs_M"),
    ):
        panels = [
            (SCENARIO_LABEL.get(sc, sc), load_metric_vs_M(CASE, metric, sc))
            for sc in SCENARIOS
        ]
        paths = make_vs_M_figure(panels, ylabel, out / stem, ncols=2)
        written += paths
        if not paths:
            print(f"[urban] no data for {metric}; run the urban grid first")

    for p in written:
        print(f"[fig] wrote {p}")
    if not written:
        print("[urban] no figures produced (the urban grid has not been run yet).")


if __name__ == "__main__":
    main()
