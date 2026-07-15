"""Produce ALL figures for the Navier--Stokes case.

Metric-vs-sampler-steps figures, each with all methods and one panel per
observation scenario (legend below):

* ``ns_rmse_vs_M``          -- ensemble-mean vorticity RMSE vs. $M$.
* ``ns_crps_vs_M``          -- CRPS vs. $M$.
* ``ns_spread_skill_vs_M``  -- spread--skill $|1-\\mathrm{spread}/\\mathrm{skill}|$ vs. $M$.

Metric-vs-assimilation-step figures (per-step curves, trajectory-averaged):

* ``ns_rmse_vs_step`` / ``ns_crps_vs_step`` / ``ns_spread_skill_vs_step``.

Plus qualitative vorticity field maps for trajectory 11 (one figure per scenario):

* ``ns_states_<scenario>``  -- rows = methods, cols = Truth / Posterior mean /
                               $|$error$|$ / Spread at the final assimilated step,
                               from ``results/navier_stokes/states/traj11/*.npz``.

Reads the aggregates produced by ``aggregate_ns.py``: ``aggregated/all.csv``
(metric-vs-M) and ``aggregated/per_step.csv`` (metric-vs-step); the field maps
read the saved-state archives directly. Any figure with no data yet is skipped
with a message. Run order: ``run_ns_grid.sh`` -> ``aggregate_ns.py`` -> this.

    python paper_experiments/make_ns_figures.py
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import numpy as np  # noqa: E402

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

DEFAULT_OUT = _here.parent / "manuscript" / "figures" / "navier_stokes"
CASE = "navier_stokes"
SCENARIOS = ("16^2->128^2", "32^2->128^2", "sparse 5%", "sparse 1.5625%")
# Sampler-step ladder of the NS grid (run_ns_grid.sh): starts at M=25.
NS_STEPS = (25, 50, 100, 250)
# Trajectory whose saved posterior/truth fields the field maps are drawn from.
STATE_TRAJ = 11


def SLUG(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")

# (metric key, y-axis label, output stem).
METRIC_FIGURES = (
    ("rmse", r"Vorticity RMSE", "ns_rmse_vs_M"),
    ("crps", r"CRPS", "ns_crps_vs_M"),
    ("spread_skill", r"Spread--skill $|1-\mathrm{spread}/\mathrm{skill}|$",
     "ns_spread_skill_vs_M"),
)


def _step_figures(out: Path, legend_keys: set) -> list[Path]:
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
        for _t, series in panels:
            legend_keys.update(k for k, s in series.items() if s)
        paths = make_vs_step_figure(
            panels, ylabel, out / f"ns_{metric}_vs_step", ncols=2,
            panel_slugs=[SLUG(sc) for sc in SCENARIOS],
        )
        written += paths
    if not written:
        print("[ns] no per-step data; run run_ns_grid.sh + aggregate_ns.py")
    return written


def _state_figures(out: Path, state_M: int | None = None) -> list[Path]:
    """Qualitative vorticity field maps for trajectory ``STATE_TRAJ``: one figure per
    scenario, rows = methods, cols = Truth | Posterior mean | |Error| | Spread (final
    step).

    Reads the self-contained ``results/navier_stokes/states/traj<STATE_TRAJ>/*.npz``
    archives saved by ``run_ns_grid.sh``; skipped with a message if none exist yet.

    ``state_M`` picks the sampler-step count to show (default: the largest saved).
    The grid saves one archive per M, so this must stay a single M -- otherwise the
    same method appears once per M as separate rows.
    """
    written: list[Path] = []
    # Vorticity is the single (signed) state channel -> diverging colour map.
    field_fn = lambda traj: traj[:, 0, :, :, -1]  # noqa: E731  [n, H, W] final step
    for sc in SCENARIOS:
        recs = load_state_records(CASE, scenario=sc, traj=STATE_TRAJ, M=state_M)
        if not recs:
            continue
        Ms = sorted({int(r["M"]) for r in recs if "M" in r})
        print(f"[ns] states {sc}: {len(recs)} methods at M={Ms} (traj{STATE_TRAJ})")
        stem = out / f"ns_states_{SLUG(sc)}"
        paths = make_state_field_figure(
            recs, field_fn, stem, cbar_label="vorticity",
            cmap="RdBu_r", diverging=True,
        )
        # Bare one-quantity-per-file panels (truth / mean / std / |error| per
        # method) for the manuscript's per-quantity subfigure grids.
        paths += make_state_panel_singles(
            recs, field_fn, stem, cbar_label="vorticity",
            cmap="RdBu_r", diverging=True,
        )
        written += paths
    if not written:
        print(
            "[ns] no saved states; run run_ns_grid.sh "
            f"(save_states, traj{STATE_TRAJ}) first"
        )
    return written


def _truth_obs_figures(out: Path, state_M: int | None = None) -> list[Path]:
    """Truth + per-scenario observation panels (singles), for the truth figure.

    Writes ``singles/ns_truth.pdf`` (the full true vorticity field at the final
    assimilated step, shared by every scenario since they all assimilate the same
    trajectory) and one ``singles/ns_obs_<scenario>.pdf`` per scenario showing
    what that scenario actually sees:

    * super-resolution -- the observed low-resolution field ($16^2$ / $32^2$),
      drawn on the full-resolution extent so the panels stay the same size;
    * sparse sensors    -- the observed grid points (``obs_indices``) as markers
      coloured on the vorticity scale, over a faint grey truth field.

    All panels share the truth's symmetric colour scale, so ``ns_states_*__cbar_field``
    serves this figure too.
    """
    written: list[Path] = []
    truth2d = None
    for sc in SCENARIOS:
        recs = load_state_records(CASE, scenario=sc, traj=STATE_TRAJ, M=state_M)
        if not recs:
            continue
        r = recs[0]  # truth + observations are shared across methods of a scenario
        truth = np.asarray(r["true_trajectory"])[0, 0, :, :, -1]  # [H, W] final step
        obs = np.asarray(r["observations"])[0, :, -1]             # [n] final step
        idx = np.asarray(r["obs_indices"]).reshape(-1)
        H, W = truth.shape
        m = float(np.abs(truth[np.isfinite(truth)]).max()) or 1.0
        if truth2d is None:
            truth2d = truth
            written += save_field_panel(
                out / "singles" / "ns_truth", truth,
                cmap="RdBu_r", vmin=-m, vmax=m,
            )
        stem = out / "singles" / f"ns_obs_{SLUG(sc)}"
        if obs.size == idx.size and idx.size < H * W:
            # Sparse sensors: obs_indices are flat indices into the H x W grid.
            ys, xs = np.divmod(idx, W)
            written += save_field_panel(
                stem, None, cmap="RdBu_r", vmin=-m, vmax=m,
                scatter=(xs.astype(float), ys.astype(float), obs),
                background=truth, extent=(0.0, float(W), 0.0, float(H)),
            )
        else:
            # Super-resolution: the observation IS the coarse field (k^2 values).
            k = int(round(math.sqrt(obs.size)))
            written += save_field_panel(
                stem, obs.reshape(k, k), cmap="RdBu_r", vmin=-m, vmax=m,
                extent=(0.0, float(W), 0.0, float(H)),
            )
    if not written:
        print(f"[ns] no saved states; cannot draw truth/observation panels")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--state-M", type=int, default=None,
        help="sampler steps M for the field maps (default: largest saved)",
    )
    args = ap.parse_args()
    out = Path(args.out)
    written: list[Path] = []
    legend_keys: set = set()  # every (method, variant) with data, for the legend file

    for metric, ylabel, stem in METRIC_FIGURES:
        panels = [
            (SCENARIO_LABEL.get(sc, sc), load_metric_vs_M(CASE, metric, sc, steps=NS_STEPS))
            for sc in SCENARIOS
        ]
        for _t, series in panels:
            legend_keys.update(k for k, s in series.items() if s)
        paths = make_vs_M_figure(
            panels, ylabel, out / stem, ncols=2, steps=NS_STEPS,
            panel_slugs=[SLUG(sc) for sc in SCENARIOS],
        )
        if paths:
            written += paths
        else:
            print(f"[ns] no data for {metric}; run run_ns_grid.sh + aggregate_ns.py")

    written += _step_figures(out, legend_keys)
    written += _state_figures(out, state_M=args.state_M)
    written += _truth_obs_figures(out, state_M=args.state_M)

    # One shared legend file for all the single-panel metric figures of this case.
    written += save_series_legend(legend_keys, out / "singles" / "ns_legend")

    # Mirror every figure into the in-repo paper_experiments/figures/ tree too
    # (singles/ keeps its subfolder).
    written += mirror_figures(written, FIGURES_DIR / CASE)

    for p in written:
        print(f"[fig] wrote {p}")
    if not written:
        print("[ns] no figures produced (the NS grid has not been run yet).")


if __name__ == "__main__":
    main()
