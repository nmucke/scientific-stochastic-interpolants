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
     final assimilated step, from ``results/urban/states/traj1/*.npz`` at $M=50$
     (``--state-M`` overrides).

and, into ``singles/``, the bare one-quantity-per-file panels the manuscript's
per-quantity grids are tiled from (velocity magnitude and temperature x posterior
mean / std / $|$error$|$, per method), plus the truth and sensor-location panels:

* ``urban_states_<var>_<scenario>__{mean,std,abserr}__<method>`` and the shared
  colourbars ``__cbar_{field,std,abserr}``.
* ``urban_truth_{velocity,temperature}`` and ``urban_obs_<scenario>``.

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
    make_state_panel_singles,
    make_vs_M_figure,
    make_vs_step_figure,
    mirror_figures,
    save_field_panel,
    save_series_legend,
)

DEFAULT_OUT = _here.parent / "manuscript" / "figures" / "urban"
CASE = "urban"
# 2026-08-04: urban lineup = the two sparsest regimes (1/64, 1/128); 5% is off it.
SCENARIOS = ("sparse 1.5625%", "sparse 0.78125%")
# Sampler-step ladder of the urban grid (run_urban_grid.sh STEPS). Must be passed
# explicitly: figure_common's default STEPS is the older (50, 100, 250, 500), which
# would silently drop the M=25 column and leave an empty M=500 one.
URBAN_STEPS = (25, 50, 100, 250)
# Trajectory whose saved posterior/truth fields the field maps are drawn from
# (run_urban_grid.sh SAVE_TRAJ).
STATE_TRAJ = 1
# Sampler-step count the field maps are drawn at. Fixed at M=50 -- the qualitative
# panels must all show the SAME M to be comparable, so this has to be a rung every
# method actually reached. Override with ``--state-M``.
STATE_M = 50


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


def _step_figures(out: Path, legend_keys: set) -> list[Path]:
    """Metric-vs-assimilation-step figures (trajectory-averaged per-step curves).

    RMSE is per-variable (velocity + temperature x the two sparse scenarios, 2x2
    panels); CRPS and spread--skill get one panel per scenario. Reads
    ``results/urban/aggregated/per_step.csv`` (``aggregate_urban.py``); skipped
    with a message if that file does not exist yet.
    """
    written: list[Path] = []
    rmse_panels: list[tuple[str, dict]] = []
    rmse_slugs: list[str] = []
    for var_metric, var_name in (
        ("rmse_velocity", "Velocity"), ("rmse_temperature", "Temperature")
    ):
        for sc in SCENARIOS:
            title = f"{var_name} -- {SCENARIO_LABEL.get(sc, sc)}"
            rmse_panels.append((title, load_metric_vs_step(CASE, var_metric, sc)))
            rmse_slugs.append(f"{var_name.lower()}_{SLUG(sc)}")
    for _t, series in rmse_panels:
        legend_keys.update(k for k, s in series.items() if s)
    written += make_vs_step_figure(
        rmse_panels, r"RMSE (fluid cells)", out / "urban_rmse_vs_step", ncols=2,
        panel_slugs=rmse_slugs,
    )
    for metric, ylabel, stem in (
        ("crps", r"CRPS", "urban_crps_vs_step"),
        # Short label on purpose -- see make_ns_figures.METRIC_FIGURES.
        ("spread_skill", r"$|1-\mathrm{spread}/\mathrm{skill}|$",
         "urban_spread_skill_vs_step"),
    ):
        panels = [
            (SCENARIO_LABEL.get(sc, sc), load_metric_vs_step(CASE, metric, sc))
            for sc in SCENARIOS
        ]
        for _t, series in panels:
            legend_keys.update(k for k, s in series.items() if s)
        written += make_vs_step_figure(
            panels, ylabel, out / stem, ncols=2,
            panel_slugs=[SLUG(sc) for sc in SCENARIOS],
        )
    if not written:
        print("[urban] no per-step data; run run_urban_grid.sh + aggregate_urban.py")
    return written


def _state_figures(out: Path, state_M: int | None = STATE_M) -> list[Path]:
    """Qualitative field maps for trajectory ``STATE_TRAJ``: one figure per
    (variable, scenario), rows = methods, cols = Truth | Posterior mean | |Error| |
    Spread (final step).

    The state is multi-channel ``(u, v, w, thl)``, so it makes a velocity-magnitude
    figure and a temperature figure per scenario. Reads the self-contained
    ``results/urban/states/traj<STATE_TRAJ>/*.npz`` archives from
    ``run_urban_grid.sh``; skipped with a message if none exist yet.

    ``state_M`` picks the sampler-step count to show (default ``STATE_M`` = 50;
    ``None`` falls back to the widest-coverage saved M). The grid saves one archive
    per M, so this must stay a single M -- otherwise the same method appears once
    per M as separate rows.

    Alongside each combined figure it also writes the bare per-(method, quantity)
    panels the manuscript's per-quantity grids are tiled from -- posterior mean,
    spread and $|$error$|$, one file each, on colour scales shared across methods
    (see :func:`figure_common.make_state_panel_singles`).
    """
    written: list[Path] = []
    for sc in SCENARIOS:
        recs = load_state_records(CASE, scenario=sc, traj=STATE_TRAJ, M=state_M)
        if not recs:
            continue
        Ms = sorted({int(r["M"]) for r in recs if "M" in r})
        print(f"[urban] states {sc}: {len(recs)} methods at M={Ms} (traj{STATE_TRAJ})")
        for field_fn, infix, cbar, cmap in STATE_FIELDS:
            stem = out / f"urban_states_{infix}_{SLUG(sc)}"
            written += make_state_field_figure(
                recs, field_fn, stem, cbar_label=cbar, cmap=cmap
            )
            written += make_state_panel_singles(
                recs, field_fn, stem, cbar_label=cbar, cmap=cmap
            )
    if not written:
        print(
            "[urban] no saved states; run run_urban_grid.sh "
            f"(save_states, traj{STATE_TRAJ}) first"
        )
    return written


def _truth_obs_figures(out: Path, state_M: int | None = STATE_M) -> list[Path]:
    """Truth + sensor-location panels (singles), for the manuscript's truth figure.

    Writes ``singles/urban_truth_{velocity,temperature}.pdf`` (the true fields at
    the final assimilation step -- shared by both scenarios, which assimilate the
    same trajectory) and ``singles/urban_obs_<scenario>.pdf``, the observed grid
    points of each sparse scenario drawn over a faint truth field. Both scenarios
    are sparse-sensor, so there is no super-resolution (coarse-field) case here as
    there is for Navier--Stokes.

    The panels reuse the same colour scales as the combined field maps, so
    ``urban_states_<var>_<scenario>__cbar_field`` serves this figure too.
    """
    written: list[Path] = []
    truth_done = False
    for sc in SCENARIOS:
        recs = load_state_records(CASE, scenario=sc, traj=STATE_TRAJ, M=state_M)
        if not recs:
            continue
        r = recs[0]  # truth + observations are shared across the methods of a scenario
        traj = np.asarray(r["true_trajectory"])
        vel = _velocity_mag(traj)[0]     # [H, W]
        thl = _temperature(traj)[0]      # [H, W]
        H, W = vel.shape
        if not truth_done:
            truth_done = True
            for field, infix, cmap in (
                (vel, "velocity", "viridis"), (thl, "temperature", "inferno")
            ):
                fin = field[np.isfinite(field)]
                written += save_field_panel(
                    out / "singles" / f"urban_truth_{infix}", field, cmap=cmap,
                    vmin=float(fin.min()) if fin.size else 0.0,
                    vmax=float(fin.max()) if fin.size else 1.0,
                )
        if "obs_indices" not in r:
            continue
        idx = np.asarray(r["obs_indices"]).reshape(-1)
        if idx.size >= H * W:  # not a sparse index set -> nothing to mark
            continue
        ys, xs = np.divmod(idx % (H * W), W)
        fin = vel[np.isfinite(vel)]
        written += save_field_panel(
            out / "singles" / f"urban_obs_{SLUG(sc)}", None, cmap="viridis",
            vmin=float(fin.min()) if fin.size else 0.0,
            vmax=float(fin.max()) if fin.size else 1.0,
            scatter=(xs.astype(float), ys.astype(float), vel[ys % H, xs % W]),
            background=vel, extent=(0.0, float(W), 0.0, float(H)),
        )
    if not written:
        print("[urban] no saved states; cannot draw truth/observation panels")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--state-M", type=int, default=STATE_M,
        help=f"sampler steps M for the field maps (default: {STATE_M}; "
             "0 = pick the widest-coverage saved M)",
    )
    args = ap.parse_args()
    out = Path(args.out)
    state_M = args.state_M or None  # --state-M 0 -> auto-pick
    written: list[Path] = []
    legend_keys: set = set()  # every (method, variant) with data, for the legend file

    # (1) RMSE: velocity + temperature x the two sparse scenarios (2x2 panels).
    rmse_panels: list[tuple[str, dict]] = []
    rmse_slugs: list[str] = []
    for var_metric, var_name in (
        ("rmse_velocity", "Velocity"), ("rmse_temperature", "Temperature")
    ):
        for sc in SCENARIOS:
            title = f"{var_name} -- {SCENARIO_LABEL.get(sc, sc)}"
            rmse_panels.append(
                (title, load_metric_vs_M(CASE, var_metric, sc, steps=URBAN_STEPS))
            )
            rmse_slugs.append(f"{var_name.lower()}_{SLUG(sc)}")
    for _t, series in rmse_panels:
        legend_keys.update(k for k, s in series.items() if s)
    paths = make_vs_M_figure(
        rmse_panels, r"RMSE (fluid cells)", out / "urban_rmse_vs_M", ncols=2,
        steps=URBAN_STEPS, panel_slugs=rmse_slugs,
    )
    written += paths
    if not paths:
        print("[urban] no RMSE data; run run_urban_grid.sh + aggregate_urban.py")

    # (2) CRPS and (3) spread--skill: one panel per sparse scenario.
    for metric, ylabel, stem in (
        ("crps", r"CRPS", "urban_crps_vs_M"),
        # Short label on purpose -- see make_ns_figures.METRIC_FIGURES.
        ("spread_skill", r"$|1-\mathrm{spread}/\mathrm{skill}|$",
         "urban_spread_skill_vs_M"),
    ):
        panels = [
            (
                SCENARIO_LABEL.get(sc, sc),
                load_metric_vs_M(CASE, metric, sc, steps=URBAN_STEPS),
            )
            for sc in SCENARIOS
        ]
        for _t, series in panels:
            legend_keys.update(k for k, s in series.items() if s)
        paths = make_vs_M_figure(
            panels, ylabel, out / stem, ncols=2, steps=URBAN_STEPS,
            panel_slugs=[SLUG(sc) for sc in SCENARIOS],
        )
        written += paths
        if not paths:
            print(f"[urban] no data for {metric}; run the urban grid first")

    written += _step_figures(out, legend_keys)
    written += _state_figures(out, state_M=state_M)
    written += _truth_obs_figures(out, state_M=state_M)

    # One shared legend file for all the single-panel metric figures of this case.
    written += save_series_legend(legend_keys, out / "singles" / "urban_legend")

    # Mirror every figure into the in-repo paper_experiments/figures/ tree too
    # (singles/ keeps its subfolder).
    written += mirror_figures(written, FIGURES_DIR / CASE)

    for p in written:
        print(f"[fig] wrote {p}")
    if not written:
        print("[urban] no figures produced (the urban grid has not been run yet).")


if __name__ == "__main__":
    main()
