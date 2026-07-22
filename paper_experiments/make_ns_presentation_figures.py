"""Presentation build of the Navier--Stokes figures -- reduced method lineup.

Same figures as ``make_ns_figures.py`` (metric-vs-$M$, metric-vs-assimilation-step,
and the trajectory-11 vorticity field maps at $M=250$), PLUS a cost figure
``ns_runtime_vs_M`` (runtime per assimilation step vs $M$, MINIMUM over
trajectories -- see :func:`_load_runtime_min`), restricted to six series:

* Ours SI-SDE / DM-SDE / FM-ODE -- SHARED-Jacobian variant only, and the
  "(shared)" qualifier dropped from the legend (there is no second variant left
  to distinguish it from).
* FlowDAS + SURGE, SDA + SURGE, D-Flow SGLD, Guided FM (FIG).

Dropped relative to ``make_ns_figures.py``: the Jac-free "Ours" variants, the
plain (non-SURGE) FlowDAS and SDA, and the classical filters (EnKF, particle
filter). The lineup is the single ``KEEP`` table below -- edit that to change it.

No ``singles/`` output (per-panel legend-free duplicates): every figure here is
the combined, legend-carrying version. The truth / observation panels are kept
(they are distinct content, not duplicates) and written at the top level.

Figures land in ``<repo root>/ns_presentation_figs/``; nothing is mirrored into
the in-repo ``paper_experiments/figures/`` tree.

    python paper_experiments/make_ns_presentation_figures.py
    python paper_experiments/make_ns_presentation_figures.py --out <dir>
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

import figure_common as fc  # noqa: E402
from figure_common import (  # noqa: E402
    SCENARIO_LABEL,
    _canon_variant,
    load_metric_vs_M,
    load_metric_vs_step,
    load_state_records,
    make_state_field_figure,
    make_vs_M_figure,
    make_vs_step_figure,
    save_field_panel,
)
from results_schema import load_records  # noqa: E402

DEFAULT_OUT = _here.parent / "ns_presentation_figs"
CASE = "navier_stokes"
SCENARIOS = ("16^2->128^2", "32^2->128^2", "sparse 5%", "sparse 1.5625%")
NS_STEPS = (25, 50, 100, 250)
STATE_TRAJ = 11
STATE_M = 250

# --------------------------------------------------------------------------- #
# The lineup. ``(method, variant) -> legend label`` -- the ONLY place the method
# selection lives. Styling (colour / linestyle / marker) still comes from the
# shared ``figure_common.SERIES`` table, so these curves look identical to the
# manuscript ones; only the membership and the two "(shared)" labels change.
# --------------------------------------------------------------------------- #
KEEP: dict[tuple[str, str | None], str] = {
    ("Ours (SI-SDE)", "shared"): "Ours SI-SDE",
    ("Ours (DM-SDE)", "shared"): "Ours DM-SDE",
    ("Ours (FM-ODE)", "shared"): "Ours FM-ODE",
    ("SURGE (FlowDAS)", None): "FlowDAS + SURGE",
    ("SURGE (SDA)", None): "SDA + SURGE",
    ("D-Flow SGLD", None): "D-Flow SGLD",
    ("Guided FM (FIG)", None): "Guided FM (FIG)",
}

# Restrict the shared SERIES table to KEEP, applying the new labels. Everything
# downstream (both plotters, the field-map row ordering) iterates SERIES, so this
# one patch drops the unwanted curves everywhere at once -- a series absent from
# SERIES is simply never drawn.
fc.SERIES = tuple(
    (m, v, KEEP[(m, v)], *rest) for m, v, _label, *rest in fc.SERIES if (m, v) in KEEP
)
_missing = set(KEEP) - {(s[0], s[1]) for s in fc.SERIES}
if _missing:  # a typo in KEEP would otherwise just silently drop the series
    raise SystemExit(f"[ns-pres] KEEP entries not in figure_common.SERIES: {_missing}")


def SLUG(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


METRIC_FIGURES = (
    ("rmse", r"Vorticity RMSE", "ns_rmse_vs_M"),
    ("crps", r"CRPS", "ns_crps_vs_M"),
    ("spread_skill", r"Spread--skill $|1-\mathrm{spread}/\mathrm{skill}|$",
     "ns_spread_skill_vs_M"),
)


def _keep_series(series: dict) -> dict:
    """Drop every ``(method, variant)`` outside :data:`KEEP` from a loaded series."""
    return {k: v for k, v in series.items() if (k[0], _canon_variant(k[1])) in KEEP}


def _keep_records(recs: list[dict]) -> list[dict]:
    """Same filter for saved-state archives.

    Needed as its own step: ``make_state_field_figure`` draws a row for EVERY
    record it is handed and only consults ``SERIES`` for ordering/labels, so a
    method left in ``recs`` would still get a row (under its raw name) despite
    being absent from the patched table.
    """
    return [
        r for r in recs
        if (str(r.get("method")), _canon_variant(str(r.get("variant", "")))) in KEEP
    ]


# Runtime axis EXCLUDES M=25. Those cells come from the CPU fill
# (run_ns_grid_M25.sh, DEVICE=cpu) while every M>=50 cell is from the cuda grid, so
# the two are not on the same hardware and must not share a cost axis: Ours SI-SDE
# reads 44.9 s at M=25 against 37.1 s at M=50, i.e. runtime FALLING as M doubles.
# Accuracy figures are unaffected (a metric is device-independent), which is why
# only this one figure narrows the ladder. Restore by using NS_STEPS here once the
# M=25 cells have been re-run on the GPU.
RUNTIME_STEPS = (50, 100, 250)


def _load_runtime_min(scenario: str, *, steps: tuple[int, ...] = RUNTIME_STEPS) -> dict:
    """Return ``{(method, variant) -> {M: seconds}}``, MINIMUM over trajectories.

    Two deliberate departures from :func:`load_metric_vs_M`:

    * It reads the PER-CELL files under ``results/navier_stokes/metrics/`` rather
      than ``aggregated/all.csv``. The aggregate has already collapsed the
      trajectories to their MEAN, so the minimum cannot be recovered from it.
    * It reduces with ``min`` instead of the mean. Runtime is a cost measurement on
      a shared, contended box: the slow tail is machine noise (other jobs, thread
      contention, swap), not a property of the method, so the fastest observed run
      is the cleaner estimate of what the method actually costs. Spread here is
      large -- SURGE (SDA) at $M=250$ ranges 186--385 s across trajectories.

    The ``seconds`` metric is ALREADY per assimilation step -- ``_ns_pipeline``
    stores ``timer.elapsed / n_assim`` -- so no further division is needed.
    """
    out: dict[tuple[str, str | None], dict[int, float]] = {}
    mdir = fc.RESULTS / CASE / "metrics"
    if not mdir.exists():
        return {}
    for f in sorted(mdir.glob("*.csv")):
        try:
            recs = load_records(f)
        except Exception:
            continue
        for r in recs:
            if r.metric != "seconds" or r.scenario != scenario or r.M not in steps:
                continue
            if r.value is None or (isinstance(r.value, float) and math.isnan(r.value)):
                continue
            key = (r.method, _canon_variant(r.variant))
            if key not in KEEP:
                continue
            per_M = out.setdefault(key, {})
            M, v = int(r.M), float(r.value)
            per_M[M] = min(per_M[M], v) if M in per_M else v
    return out


def _runtime_figure(out: Path) -> list[Path]:
    """Runtime per DA step vs $M$, one panel per scenario (minimum over trajectories).

    ``detect_off_scale=False``: on a cost axis a method sitting far above the rest
    is the result, not a divergence, so it must be drawn at its true value instead
    of being folded onto the "collapsed (off scale)" shelf.
    """
    panels = [
        (SCENARIO_LABEL.get(sc, sc), _load_runtime_min(sc)) for sc in SCENARIOS
    ]
    paths = make_vs_M_figure(
        panels, r"Runtime per DA step [s]", out / "ns_runtime_vs_M",
        ncols=2, steps=RUNTIME_STEPS, singles=False, detect_off_scale=False,
        xlabel_all=True,
        panel_slugs=[SLUG(sc) for sc in SCENARIOS],
    )
    if not paths:
        print("[ns-pres] no runtime data; run run_ns_grid.sh first")
    return paths


PER_STEP_MS = (50, 100, 250)
# One (linestyle, marker) per M -- the panel's colour already encodes the method,
# so M has to be carried by the line style alone.
_M_STYLE = {50: ("-", "o"), 100: ("--", "s"), 250: (":", "^")}


def _per_step_by_method_figures(out: Path) -> list[Path]:
    """Per-step curves at M=50/100/250, ONE PANEL PER METHOD, all metrics.

    Layout per file: rows = methods, cols = metrics (RMSE / CRPS / spread--skill),
    and inside each panel one curve per sampler-step count. So a method's row is
    "the figure for that method", and the whole lineup ships in a single file.

    One file PER SCENARIO (``ns_per_step_by_method_<scenario>``): the four
    scenarios are different assimilation problems whose metrics do not share an
    axis, and stacking all of them would make a 28-row sheet. Colour is the
    method's usual ``SERIES`` colour so panels stay tied to the other figures.
    """
    written: list[Path] = []
    fc.apply_style()
    style = {(s[0], s[1]): s for s in fc.SERIES}  # (method, variant) -> SERIES row

    for sc in SCENARIOS:
        # data[(metric, M)] -> {(method, variant): {step: value}}
        data = {
            (metric, M): _keep_series(load_metric_vs_step(CASE, metric, sc, M=M))
            for metric, _yl, _st in METRIC_FIGURES
            for M in PER_STEP_MS
        }
        # Keep SERIES order, and only methods that actually have a curve somewhere.
        methods = [
            k for k in ((s[0], s[1]) for s in fc.SERIES)
            if any(data[(metric, M)].get(k) for metric, _y, _s in METRIC_FIGURES
                   for M in PER_STEP_MS)
        ]
        if not methods:
            continue
        nrows, ncols = len(methods), len(METRIC_FIGURES)
        # No sharex: every panel carries its own tick labels AND its own x-label,
        # so a row lifted out of the sheet on its own slide still reads.
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), squeeze=False,
        )
        for i, key in enumerate(methods):
            colour = style[key][3]
            for j, (metric, ylabel, _stem) in enumerate(METRIC_FIGURES):
                ax = axes[i][j]
                for M in PER_STEP_MS:
                    s = data[(metric, M)].get(key)
                    if not s:
                        continue
                    xs = sorted(s)
                    ls, marker = _M_STYLE[M]
                    ax.plot(
                        xs, [s[x] for x in xs], color=colour, linestyle=ls,
                        marker=marker, markersize=4.5, markeredgewidth=1.0,
                        markeredgecolor=colour, markerfacecolor="white",
                        linewidth=1.8, zorder=3,
                    )
                ax.set_yscale("log")
                ax.margins(x=0.03)
                ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
                ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.15)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                ax.tick_params(which="both", direction="out", length=4, width=0.8)
                ax.tick_params(axis="x", labelbottom=True)
                ax.set_xlabel("Assimilation step")
                if i == 0:
                    ax.set_title(ylabel, fontsize=10)
                if j == 0:
                    ax.set_ylabel(style[key][2], fontsize=9)

        # One legend for the whole sheet: the three M, drawn neutral (black) since
        # colour means "method" here, not "M".
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], color="0.2", linestyle=_M_STYLE[M][0],
                   marker=_M_STYLE[M][1], markersize=4.5, markerfacecolor="white",
                   markeredgecolor="0.2", linewidth=1.8)
            for M in PER_STEP_MS
        ]
        fig.tight_layout()
        fig.legend(
            handles, [f"$M={M}$" for M in PER_STEP_MS], loc="upper center",
            bbox_to_anchor=(0.5, 0.0), ncol=len(PER_STEP_MS), frameon=False,
            handlelength=2.6, columnspacing=1.6, fontsize=10,
        )
        written += fc._save_fig(fig, out / f"ns_per_step_by_method_{SLUG(sc)}")
    if not written:
        print("[ns-pres] no per-step data; run run_ns_grid.sh + aggregate_ns.py")
    return written


def _step_figures(out: Path) -> list[Path]:
    """Metric-vs-assimilation-step figures (one per metric, one panel per scenario)."""
    written: list[Path] = []
    for metric, ylabel, _stem in METRIC_FIGURES:
        panels = [
            (SCENARIO_LABEL.get(sc, sc), _keep_series(load_metric_vs_step(CASE, metric, sc)))
            for sc in SCENARIOS
        ]
        written += make_vs_step_figure(
            panels, ylabel, out / f"ns_{metric}_vs_step", ncols=2, singles=False,
            xlabel_all=True,
            panel_slugs=[SLUG(sc) for sc in SCENARIOS],
        )
    if not written:
        print("[ns-pres] no per-step data; run run_ns_grid.sh + aggregate_ns.py")
    return written


def _state_figures(out: Path, state_M: int | None = STATE_M) -> list[Path]:
    """Vorticity field maps for trajectory ``STATE_TRAJ`` at ``state_M``, one per scenario.

    Rows = the kept methods, cols = Truth | Posterior mean | |Error| | Spread at the
    final assimilated step. Per-quantity ``singles`` are NOT written here.
    """
    written: list[Path] = []
    field_fn = lambda traj: traj[:, 0, :, :, -1]  # noqa: E731  [n, H, W] final step
    for sc in SCENARIOS:
        recs = _keep_records(load_state_records(CASE, scenario=sc, traj=STATE_TRAJ, M=state_M))
        if not recs:
            continue
        Ms = sorted({int(r["M"]) for r in recs if "M" in r})
        print(f"[ns-pres] states {sc}: {len(recs)} methods at M={Ms} (traj{STATE_TRAJ})")
        written += make_state_field_figure(
            recs, field_fn, out / f"ns_states_{SLUG(sc)}", cbar_label="vorticity",
            cmap="RdBu_r", diverging=True, singles=False,
        )
    if not written:
        print(f"[ns-pres] no saved states for traj{STATE_TRAJ}; run run_ns_grid.sh first")
    return written


def _truth_obs_figures(out: Path, state_M: int | None = STATE_M) -> list[Path]:
    """Truth field + one observation panel per scenario (what that scenario sees).

    Unlike ``make_ns_figures.py`` these go at the TOP level rather than under
    ``singles/`` -- they are distinct content (the truth and the observations),
    not legend-free copies of a combined figure, so they survive the no-singles
    rule. Colour scale is the truth's symmetric one, shared with the field maps.

    The method filter does not apply: truth and observations are properties of the
    scenario, so any surviving archive serves (``recs[0]`` unfiltered).
    """
    written: list[Path] = []
    truth2d = None
    for sc in SCENARIOS:
        recs = load_state_records(CASE, scenario=sc, traj=STATE_TRAJ, M=state_M)
        if not recs:
            continue
        r = recs[0]
        truth = np.asarray(r["true_trajectory"])[0, 0, :, :, -1]  # [H, W] final step
        obs = np.asarray(r["observations"])[0, :, -1]             # [n] final step
        idx = np.asarray(r["obs_indices"]).reshape(-1)
        H, W = truth.shape
        m = float(np.abs(truth[np.isfinite(truth)]).max()) or 1.0
        if truth2d is None:
            truth2d = truth
            written += save_field_panel(
                out / "ns_truth", truth, cmap="RdBu_r", vmin=-m, vmax=m,
            )
        stem = out / f"ns_obs_{SLUG(sc)}"
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
        print("[ns-pres] no saved states; cannot draw truth/observation panels")
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

    for metric, ylabel, stem in METRIC_FIGURES:
        panels = [
            (SCENARIO_LABEL.get(sc, sc),
             _keep_series(load_metric_vs_M(CASE, metric, sc, steps=NS_STEPS)))
            for sc in SCENARIOS
        ]
        paths = make_vs_M_figure(
            panels, ylabel, out / stem, ncols=2, steps=NS_STEPS, singles=False,
            xlabel_all=True,
            panel_slugs=[SLUG(sc) for sc in SCENARIOS],
        )
        if paths:
            written += paths
        else:
            print(f"[ns-pres] no data for {metric}; run run_ns_grid.sh + aggregate_ns.py")

    written += _runtime_figure(out)
    written += _per_step_by_method_figures(out)
    written += _step_figures(out)
    written += _state_figures(out, state_M=state_M)
    written += _truth_obs_figures(out, state_M=state_M)

    for p in written:
        print(f"[fig] wrote {p}")
    if not written:
        print("[ns-pres] no figures produced (the NS grid has not been run yet).")


if __name__ == "__main__":
    main()
