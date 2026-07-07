"""Produce ALL figures for the urban (uDALES) case.

Metric-vs-sampler-steps figures, each with all methods and one panel per
observation scenario / variable (legend below). Urban is generative-only
(no EnKF/PF) and sparse-only (two sensor densities):

* ``urban_rmse_vs_M``          -- per-variable RMSE vs. $M$ (velocity + temperature
                                  x the two sparse scenarios, 2x2 panels).
* ``urban_crps_vs_M``          -- CRPS vs. $M$.
* ``urban_spread_skill_vs_M``  -- spread--skill vs. $M$.

Metric-vs-assimilation-step figures (per-step curves, trajectory-averaged):

* ``urban_rmse_vs_step`` (per-variable, 2x2) / ``urban_crps_vs_step`` /
  ``urban_spread_skill_vs_step``.

Plus qualitative field maps for trajectory 1 (velocity magnitude + temperature,
one figure each per scenario):

* ``urban_states_velocity_<scenario>`` / ``urban_states_temperature_<scenario>``
  -- rows = methods, cols = Truth / Posterior mean / $|$error$|$ / Spread at the
     final assimilated step, from ``results/urban/states/traj1/*.npz``.

Reads the aggregates produced by ``aggregate_urban.py``: ``aggregated/all.csv``
(metric-vs-M) and ``aggregated/per_step.csv`` (metric-vs-step); the field maps
read the saved-state archives directly. Any figure with no data yet is skipped
with a message. Run order: ``run_urban_grid.sh`` -> ``aggregate_urban.py`` ->
this.

    python paper_experiments/make_urban_figures.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

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

DEFAULT_OUT = _here.parent / "manuscript" / "figures" / "urban"
CASE = "urban"
SCENARIOS = ("sparse 5%", "sparse 1.5625%")


def SLUG(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


# Field extractors on a trajectory array [n, C, H, W, T] -> [n, H, W] final step.
# Urban state channels are (u, v, w, thl): velocity is the first three, the
# temperature variable ``thl`` is channel 3.
def _velocity_mag(traj):
    uvw = traj[:, 0:3, :, :, -1]
    return np.sqrt((uvw ** 2).sum(axis=1))


def _temperature(traj):
    return traj[:, 3, :, :, -1]


# (field_fn, output infix, colourbar label, cmap).
STATE_FIELDS = (
    (_velocity_mag, "velocity", "velocity magnitude", "viridis"),
    (_temperature, "temperature", r"temperature $\theta_\ell$", "inferno"),
)


def _step_figures(out: Path) -> list[Path]:
    """Metric-vs-assimilation-step figures (trajectory-averaged per-step curves).

    RMSE is per-variable (velocity + temperature x the two sparse scenarios, 2x2
    panels); CRPS and spread--skill get one panel per scenario. Reads
    ``results/urban/aggregated/per_step.csv`` (``aggregate_urban.py``); skipped
    with a message if that file does not exist yet.
    """
    written: list[Path] = []
    rmse_panels: list[tuple[str, dict]] = []
    for var_metric, var_name in (
        ("rmse_velocity", "Velocity"), ("rmse_temperature", "Temperature")
    ):
        for sc in SCENARIOS:
            title = f"{var_name} -- {SCENARIO_LABEL.get(sc, sc)}"
            rmse_panels.append((title, load_metric_vs_step(CASE, var_metric, sc)))
    written += make_vs_step_figure(
        rmse_panels, r"RMSE (fluid cells)", out / "urban_rmse_vs_step", ncols=2
    )
    for metric, ylabel, stem in (
        ("crps", r"CRPS", "urban_crps_vs_step"),
        ("spread_skill", r"Spread--skill $|1-\mathrm{spread}/\mathrm{skill}|$",
         "urban_spread_skill_vs_step"),
    ):
        panels = [
            (SCENARIO_LABEL.get(sc, sc), load_metric_vs_step(CASE, metric, sc))
            for sc in SCENARIOS
        ]
        written += make_vs_step_figure(panels, ylabel, out / stem, ncols=2)
    if not written:
        print("[urban] no per-step data; run run_urban_grid.sh + aggregate_urban.py")
    return written


def _state_figures(out: Path) -> list[Path]:
    """Qualitative field maps for trajectory 1: one figure per (variable, scenario),
    rows = methods, cols = Truth | Posterior mean | |Error| | Spread (final step).

    The state is multi-channel ``(u, v, w, thl)``, so it makes a velocity-magnitude
    figure and a temperature figure per scenario. Reads the self-contained
    ``results/urban/states/traj1/*.npz`` archives from ``run_urban_grid.sh``;
    skipped with a message if none exist yet.
    """
    written: list[Path] = []
    for sc in SCENARIOS:
        recs = load_state_records(CASE, scenario=sc)
        if not recs:
            continue
        for field_fn, infix, cbar, cmap in STATE_FIELDS:
            stem = out / f"urban_states_{infix}_{SLUG(sc)}"
            written += make_state_field_figure(
                recs, field_fn, stem, cbar_label=cbar, cmap=cmap
            )
    if not written:
        print("[urban] no saved states; run run_urban_grid.sh (save_states, traj1) first")
    return written


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
        print("[urban] no RMSE data; run run_urban_grid.sh + aggregate_urban.py")

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

    written += _step_figures(out)
    written += _state_figures(out)

    # Mirror every figure into the in-repo paper_experiments/figures/ tree too.
    written += mirror_to(written, FIGURES_DIR / CASE)

    for p in written:
        print(f"[fig] wrote {p}")
    if not written:
        print("[urban] no figures produced (the urban grid has not been run yet).")


if __name__ == "__main__":
    main()
