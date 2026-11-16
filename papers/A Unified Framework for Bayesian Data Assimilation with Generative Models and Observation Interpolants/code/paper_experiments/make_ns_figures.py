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
                               from ``results/navier_stokes/states/traj11/*.npz``
                               at $M=250$ (``--state-M`` overrides).

And the radially-averaged enstrophy spectra, two panels per scenario:

* ``ns_enstrophy_spectrum_<scenario>``      -- $Z(k)$ of the TRUE field and of
                               every method's posterior mean, from the same
                               saved-state archives. Enstrophy, not energy: it is
                               a functional of the stored vorticity alone.
* ``ns_enstrophy_spectrum_zoom_<scenario>`` -- the same panels restricted to
                               $k \\ge$ ``ZOOM_KMIN``, where the methods separate.

Reads the aggregates produced by ``aggregate_ns.py``: ``aggregated/all.csv``
(metric-vs-M) and ``aggregated/per_step.csv`` (metric-vs-step); the field maps
and the spectra read the saved-state archives directly. Any figure with no data
yet is skipped with a message. Run order: ``run_ns_grid.sh`` ->
``aggregate_ns.py`` -> this.

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
    DIVERGENCE_LEGEND,
    FIGURES_DIR,
    PARTIAL_KEY,
    SCENARIO_LABEL,
    TRUTH_KEY,
    TRUTH_LABEL,
    TRUTH_STYLE,
    load_metric_vs_M,
    load_metric_vs_step,
    load_state_records,
    make_spectrum_figure,
    make_state_field_figure,
    make_state_panel_singles,
    make_vs_M_figure,
    make_vs_step_figure,
    save_field_panel,
    save_series_legend,
    series_key,
    set_results_root,
)

# After figure_common, which selects the Agg backend first.
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

DEFAULT_OUT = FIGURES_DIR / "navier_stokes"
CASE = "navier_stokes"
SCENARIOS = ("16^2->128^2", "32^2->128^2", "sparse 5%", "sparse 1.5625%")
# Sampler-step ladder of the NS grid (run_ns_grid.sh): starts at M=25.
NS_STEPS = (25, 50, 100, 250)
# Trajectory whose saved posterior/truth fields the field maps are drawn from.
STATE_TRAJ = 11
# Sampler-step count the field maps are drawn at. Fixed at the top of the NS
# ladder (M=250) -- every method is re-run there, and the qualitative panels must
# all show the SAME M to be comparable. Override with ``--state-M``.
STATE_M = 250
# Lower wavenumber of the zoomed enstrophy-spectrum panels. The spectra are
# binned onto 60 radial shells over |k| <= sqrt(2)*64 ~ 90.5 (Nyquist is 64), so
# this keeps the small-scale end of the axis -- where the methods separate, and
# which the full-range panels compress into their last centimetre.
ZOOM_KMIN = 30.0


def SLUG(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")

# (metric key, y-axis label, output stem).
METRIC_FIGURES = (
    ("rmse", r"Vorticity RMSE", "ns_rmse_vs_M"),
    ("crps", r"CRPS", "ns_crps_vs_M"),
    # Short label on purpose: the print-size singles are ~1.4in tall, and the
    # full "Spread--skill |1-...|" wording overflows the panel height.
    ("spread_skill", r"$|1-\mathrm{spread}/\mathrm{skill}|$",
     "ns_spread_skill_vs_M"),
)


def _step_figures(
    out: Path, legend_keys: set, *, M: int | None = None,
    include_m_independent: bool = False, diverged: list | None = None,
) -> list[Path]:
    """Metric-vs-assimilation-step figures (one per metric, one panel per scenario).

    Reads the trajectory-aggregated per-step curves from
    ``results/navier_stokes/aggregated/per_step.csv`` (``aggregate_ns.py``);
    skipped with a message if that file does not exist yet.

    Diverged trajectories (NaN from the divergence guard on) are not allowed to
    drag the mean silently: each curve is solid only while every trajectory is
    finite, ends in an "x", and continues faded as the survivors' mean (see
    ``figure_common._plot_step_panel``). Every such cell is appended to
    ``diverged`` as ``(metric, scenario, method, variant, first_partial_step)``.
    """
    written: list[Path] = []
    for metric, ylabel, _stem in METRIC_FIGURES:
        panels = [
            (SCENARIO_LABEL.get(sc, sc), load_metric_vs_step(
                CASE, metric, sc, M=M, include_m_independent=include_m_independent,
                mark_partial=True,
            ))
            for sc in SCENARIOS
        ]
        any_partial = False
        for sc, (_t, series) in zip(SCENARIOS, panels):
            legend_keys.update(k for k, s in series.items() if s and k != PARTIAL_KEY)
            for (m, v), (p0, n_min, full, first) in (
                series.get(PARTIAL_KEY) or {}
            ).items():
                # Only a set that shrinks MID-rollout gets the x/faded styling
                # (and its legend entries); a set reduced throughout is footnoted.
                any_partial |= p0 > first
                if diverged is not None:
                    diverged.append((metric, sc, m, v, p0, n_min, full, p0 > first))
        paths = make_vs_step_figure(
            panels, ylabel, out / f"ns_{metric}_vs_step", ncols=2,
            panel_slugs=[SLUG(sc) for sc in SCENARIOS],
            extra_legend=DIVERGENCE_LEGEND if any_partial else (),
        )
        written += paths
    if not written:
        print("[ns] no per-step data; run run_ns_grid.sh + aggregate_ns.py")
    return written


# Reference methods kept in the reduced ("ours") FC lineup besides Ours.
FC_REFERENCES = ("SURGE (FlowDAS)", "EnKF")
# Full-text-width per-panel size of the FC per-step singles (ICLR text width
# 5.5in): four stacked panels + legend + caption fit on one page.
FC_SINGLE_FIGSIZE: tuple[float, float] = (5.6, 1.2)


def _fc_step_figures(
    out: Path, fc_path: Path, *, M: int | None, lineup: str = "all",
) -> list[Path]:
    """Per-step curves WITH the forecast-correction (FC) series added.

    The usual per-step series (from ``aggregated/per_step.csv``) plus the Ours
    ``*_fc*`` series from the separate FC aggregate ``fc_path`` (only its FC
    variants are kept, so e.g. SURGE-on-Ours cells there are ignored). Written as
    ``ns_<metric>_vs_step_fc`` (+ ``singles/ns_<metric>_vs_step_fc_<scen>``) and
    ``singles/ns_legend_fc``, next to -- never over -- the plain figures.

    ``lineup``: ``"all"`` = every usual method + FC; ``"ours"`` = Ours plain + FC
    plus the ``FC_REFERENCES`` baselines; ``"full"`` = every method including the
    classical filters + FC, singles at ``FC_SINGLE_FIGSIZE`` (full text width).
    """
    written: list[Path] = []
    keys: set = set()
    any_mid = False
    for metric, ylabel, _stem in METRIC_FIGURES:
        panels = []
        for sc in SCENARIOS:
            base = load_metric_vs_step(
                CASE, metric, sc, M=M, include_m_independent=(lineup in ("ours", "full")),
                mark_partial=True,
            )
            fc = load_metric_vs_step(
                CASE, metric, sc, M=M, mark_partial=True, path=fc_path,
            )
            series = {k: v for k, v in base.items() if k != PARTIAL_KEY}
            if lineup == "ours":
                series = {
                    k: v for k, v in series.items()
                    if k[0].startswith("Ours") or k[0] in FC_REFERENCES
                }
            fc_keys = [k for k in fc if k != PARTIAL_KEY and (k[1] or "").endswith("_fc")]
            series.update({k: fc[k] for k in fc_keys})
            partial = {
                k: v for k, v in (base.get(PARTIAL_KEY) or {}).items() if k in series
            }
            partial.update({
                k: v for k, v in (fc.get(PARTIAL_KEY) or {}).items() if k in fc_keys
            })
            if lineup in ("ours", "full") and not series.get(("EnKF", None)):
                # The EnKF reference diverged on every trajectory (16^2): the
                # aggregate has no rows at all, so footnote it explicitly.
                partial[("EnKF", None)] = (0, 0, 5, 0)
            if partial:
                series[PARTIAL_KEY] = partial
                any_mid |= any(p0 > first for p0, _n, _f, first in partial.values())
            keys.update(k for k, v in series.items() if v and k != PARTIAL_KEY)
            panels.append((SCENARIO_LABEL.get(sc, sc), series))
        written += make_vs_step_figure(
            panels, ylabel, out / f"ns_{metric}_vs_step_fc", ncols=2,
            panel_slugs=[SLUG(sc) for sc in SCENARIOS],
            extra_legend=DIVERGENCE_LEGEND if any_mid else (),
            single_figsize=FC_SINGLE_FIGSIZE if lineup == "full" else None,
        )
    written += save_series_legend(
        keys, out / "singles" / "ns_legend_fc", ncol=4 if lineup == "full" else 3,
        extra_after=DIVERGENCE_LEGEND if any_mid else (),
    )
    return written


# --------------------------------------------------------------------------- #
# Appendix "all runs" per-step figures (--appendix-all)
#
# Two full lineups on the unified aggregate ``aggregated_all/`` (aggregate_ns.py):
#   set "ours"      -- every variation of our samplers (plain, + FC r=8, + FC r=20,
#                      + SURGE) x {shared Jacobian, Jacobian-free};
#   set "baselines" -- every baseline, plus ONE reference curve of Ours.
# Unlike the figures above, series are keyed by the RAW (method, variant) of the
# CSVs (FC r=8 and FC r=20 differ only in their variant hash), and run status --
# pending / partial / diverged -- is read from ``all.csv`` + ``status.csv`` at
# every call, so rerunning after more runs finish updates legend and footnotes.
# Nothing here touches the styling of the figures above.
# --------------------------------------------------------------------------- #

APPX_SAMPLERS = (  # (sampler, colour); colours as in figure_common.SERIES
    ("SI-SDE", "#0072B2"), ("DM-SDE", "#009E73"), ("FM-ODE", "#CC79A7"),
)
APPX_JACS = (("shared_jac10", "shared", True), ("jacfree", "Jac-free", False))
# Variant groups of set "ours": (column header, short tag, CSV method template,
# CSV variant template, linestyle, marker); {s} = sampler, {j} = Jacobian mode.
APPX_OURS_GROUPS = (
    ("Plain", "", "Ours ({s})", "{j}", "-", "o"),
    ("+ FC ($r{{=}}8$)", "+FC8", "Ours ({s})", "{j}_fc25_70a3134d", (0, (4.0, 1.6)), "D"),
    ("+ FC ($r{{=}}20$)", "+FC20", "Ours ({s})", "{j}_fc25_8797d2ae", (0, (1.0, 1.1)), "s"),
    ("+ SURGE", "+SURGE", "SURGE ({s})", "surge_ours_{j}", (0, (5.0, 1.4, 1.0, 1.4)), "v"),
)
APPX_NO_SURGE = ("FM-ODE",)  # SURGE needs g > 0: no SURGE (FM-ODE)
APPX_REF_KEY = ("Ours (FM-ODE)", "jacfree")
APPX_REF_LABEL = "Ours FM-ODE (Jac-free), reference"
# Fixed axes geometry (inches) of the full-text-width appendix panels, so all
# 4 x 6 panels have the SAME width and aligned axes: four stacked + legend +
# caption fit on one page. Total height = bottom + axes + top + footnote lines.
APPX_WIDTH = 5.4
APPX_AX_LEFT, APPX_AX_RIGHT = 0.54, 0.06  # margins
APPX_AX_BOTTOM, APPX_AX_H, APPX_AX_TOP = 0.36, 1.0, 0.06
APPX_FOOT_LINE = 0.085
APPX_RC: dict = {
    "font.family": "serif", "font.size": 7.5, "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "axes.linewidth": 0.55,
}
APPX_FOOT_CHARS = 140  # wrap width of a panel footnote line (6pt serif ~ 5.3in)


def _appx_series(which: str) -> list[dict]:
    """Ordered series specs of one appendix set (legend order = list order)."""
    specs: list[dict] = []
    if which == "ours":
        for col, (head, tag, mtpl, vtpl, ls, marker) in enumerate(APPX_OURS_GROUPS):
            head = head.replace("{{", "{").replace("}}", "}")
            for s, colour in APPX_SAMPLERS:
                for jv, jname, filled in APPX_JACS:
                    if tag == "+SURGE" and s in APPX_NO_SURGE:
                        continue
                    specs.append(dict(
                        key=(mtpl.format(s=s), vtpl.format(j=jv)), column=col, header=head,
                        label=f"{s}, {jname}", short=f"{s[:2]} {'sh.' if filled else 'JF'} {tag}".strip(),
                        colour=colour, ls=ls, marker=marker, filled=filled,
                        lw=1.05 if not tag else 0.95, zorder=3,
                    ))
        return specs
    base = [  # (method, variant, label, short, colour, linestyle, marker, filled)
        ("FlowDAS", "", "FlowDAS", "FlowDAS", "#D55E00", "-", "D", True),
        ("SURGE (FlowDAS)", "", "FlowDAS + SURGE", "FlowDAS+SURGE", "#E69F00", (0, (4.0, 1.6)), "v", True),
        ("SDA", "", "SDA", "SDA", "#8C510A", "-", "P", True),
        ("SURGE (SDA)", "", "SDA + SURGE", "SDA+SURGE", "#6A3D9A", (0, (4.0, 1.6)), "X", True),
        ("D-Flow SGLD", "", "D-Flow SGLD", "D-Flow", "#808000", "-", "h", True),
        ("Guided FM (FIG)", "", "Guided FM (FIG)", "FIG", "#F0027F", "-", "*", True),
    ]
    base += [
        (m, v, f"{s} prior + EnKF", f"{s[:2]}+EnKF", colour, (0, (1.0, 1.1)), "o", False)
        for (m, v), (s, colour) in zip(APPX_PRIOR_ENKF_KEYS, APPX_SAMPLERS)
    ]
    base += [
        ("EnKF", "", r"EnKF, global on SR / $r{=}20$ on sparse", "EnKF", "#000000", ":", "p", True),
        (*APPX_ENKF_R20_SR_KEY, r"EnKF, $r{=}20$ on SR", "EnKF r20", "#000000",
         (0, (5.0, 1.4, 1.0, 1.4)), "p", False),
        ("Particle filter", "", "Particle filter", "PF", "#666666", ":", "d", True),
    ]
    for m, v, label, short, colour, ls, marker, filled in base:
        specs.append(dict(key=(m, v), label=label, short=short, colour=colour, ls=ls,
                          marker=marker, filled=filled, lw=0.95, zorder=3))
    specs.append(dict(key=APPX_REF_KEY, label=APPX_REF_LABEL, short="Ours ref.",
                      colour="0.62", ls="-", marker=None, filled=True, lw=2.6,
                      zorder=1, ref=True))
    return specs


# Keys of the series whose CSV names are fixed by the extension runs; adjusted
# to the unified aggregate's (method, variant) keys.
APPX_PRIOR_ENKF_KEYS: tuple[tuple[str, str], ...] = (
    ("SI-SDE prior + EnKF", ""), ("DM-SDE prior + EnKF", ""), ("FM-ODE prior + EnKF", ""),
)
APPX_ENKF_R20_SR_KEY: tuple[str, str] = ("EnKF", "loc20_superres")


def _appx_read_csv(path: Path) -> list[dict]:
    import csv as _csv

    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return list(_csv.DictReader(fh))


def _appx_int(x) -> int | None:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _appx_status(agg: Path) -> dict:
    """``{(method, variant, scenario): dict(n_traj, n_div, n_expected, status)}``.

    From ``all.csv`` (rows of the cells that produced metrics) and ``status.csv``
    (every expected cell, including those with no output yet); ``status.csv``
    wins where both have a field.
    """
    out: dict = {}
    for name in ("all.csv", "status.csv"):
        for r in _appx_read_csv(agg / name):
            if name == "all.csv" and r.get("metric") not in (None, "", "rmse"):
                continue
            k = (r.get("method", ""), r.get("variant") or "", r.get("scenario", ""))
            d = out.setdefault(k, {})
            for f in ("n_traj", "n_div", "n_expected", "n_finished"):
                v = _appx_int(r.get(f))
                if v is not None:
                    d[f] = v
            if r.get("status"):
                d["status"] = r["status"].strip()
    return out


def _appx_load_steps(agg: Path, metric: str) -> dict:
    """``{(method, raw variant, scenario): {step: value}}`` from ``per_step.csv``.

    Each series is a mean over its non-diverged trajectories. A series stored at
    several M (should not happen) keeps the largest.
    """
    best: dict = {}
    for r in _appx_read_csv(agg / "per_step.csv"):
        if r.get("metric") != metric:
            continue
        try:
            v = float(r["value"])
        except (TypeError, ValueError):
            continue
        if math.isnan(v):
            continue
        k = (r["method"], r.get("variant") or "", r["scenario"])
        M = _appx_int(r.get("M")) or 0
        cur = best.setdefault(k, [M, {}])
        if M > cur[0]:
            cur[0], cur[1] = M, {}
        if M == cur[0]:
            cur[1][int(float(r["step"]))] = v
    return {k: d for k, (_M, d) in best.items()}


def _appx_cell(st: dict, data: dict | None) -> dict:
    """Normalised status of one (series, scenario) cell.

    ``n_ok`` = trajectories in the plotted mean, ``n_div`` = diverged (excluded),
    ``n_exp`` = expected, ``n_left`` = not finished yet; ``expected`` False when
    the cell is not part of the grid at all (e.g. EnKF r=20 on a sparse scenario).
    """
    if not st and not data:
        return dict(expected=False, n_ok=0, n_div=0, n_exp=0, n_left=0)
    n_ok = st.get("n_traj", 5 if data else 0)
    n_div = st.get("n_div", 0)
    n_exp = st.get("n_expected", max(5, n_ok + n_div))
    n_left = max(0, n_exp - st.get("n_finished", n_ok + n_div))
    return dict(expected=True, n_ok=n_ok, n_div=n_div, n_exp=n_exp, n_left=n_left,
                status=st.get("status", ""))


def _appx_footnote(spec: dict, c: dict, has_data: bool) -> str | None:
    if not c["expected"] or (c["n_ok"] >= c["n_exp"] and has_data):
        return None
    if not has_data and c["n_div"] == 0:
        # Pending with nothing anywhere yet: the legend already says so.
        return f"{spec['short']} pending" if spec.get("has_any") else None
    bits = []
    if c["n_div"]:
        bits.append(f"{c['n_div']} div.")
    if c["n_left"]:
        bits.append(f"{c['n_left']} running")
    txt = f"{spec['short']} {c['n_ok']}/{c['n_exp']}"
    return txt + (f" ({', '.join(bits)})" if bits else "")


def _appx_wrap(items: list[str], width: int) -> str:
    lines, cur = [], ""
    for it in items:
        cand = it if not cur else f"{cur}; {it}"
        if cur and len(cand) > width:
            lines.append(cur)
            cur = it
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _appx_feet(specs, steps_by_key, cells) -> str:
    """Panel footnote: every cell whose mean is not over its full trajectory set."""
    feet = []
    for sp in specs:
        c = cells.get(sp["key"])
        if c is not None and not sp.get("ref"):
            f = _appx_footnote(sp, c, bool(steps_by_key.get(sp["key"])))
            if f:
                feet.append(f)
    return _appx_wrap(feet, APPX_FOOT_CHARS)


class _AppxLogFormatter(mticker.LogFormatterSciNotation):
    """Log-axis labels as plain decimals; same choice of labelled ticks."""

    def __call__(self, x, pos=None):
        return f"{x:g}" if super().__call__(x, pos) else ""


def _appx_draw(ax, specs, steps_by_key, feet: str) -> None:
    every = 5
    for i, sp in enumerate(specs):
        s = steps_by_key.get(sp["key"]) or {}
        if not s:
            continue
        xs = sorted(s)
        kw = dict(color=sp["colour"], linestyle=sp["ls"], linewidth=sp["lw"],
                  zorder=sp["zorder"], solid_capstyle="butt", dash_capstyle="butt")
        if sp["marker"]:
            kw.update(marker=sp["marker"], markevery=(i % every, every),
                      markersize=3.0, markeredgewidth=0.6,
                      markeredgecolor=sp["colour"],
                      markerfacecolor=sp["colour"] if sp["filled"] else "white")
        ax.plot(xs, [s[x] for x in xs], **kw)
    if feet:
        ax.text(1.0, 1.0 + 0.03 / APPX_AX_H, feet,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5.6, color="0.25", linespacing=1.1)
    ax.set_yscale("log")
    # Plain decimals ("0.6", not "6x10^-1") keep the tick labels narrow; which
    # ticks get a label is still decided by matplotlib's log formatter.
    ax.yaxis.set_major_formatter(_AppxLogFormatter())
    ax.yaxis.set_minor_formatter(_AppxLogFormatter(
        labelOnlyBase=False, minor_thresholds=(1.0, 0.5)))
    lo, hi = ax.get_ylim()
    if lo > 0 and math.log10(hi / lo) < 0.7:
        # Under ~half a decade the 1..9 x 10^k ticks leave one or two labels:
        # use evenly spaced decimal ticks instead (log scale kept).
        ticks = [v for v in mticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10])
                 .tick_values(lo, hi) if lo <= v <= hi]
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:g}"))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_ylim(lo, hi)
    ax.margins(x=0.01)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, which="major", linestyle="-", linewidth=0.3, alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.25, alpha=0.2)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    ax.tick_params(which="both", direction="out", length=2.2, width=0.5, pad=1.5)


def _appx_legend_label(sp: dict, cells_by_scen: dict, has_any: bool) -> str:
    """Legend label + global status: '(pending)' with no data anywhere yet,
    '(partial)' while any grid cell of it is still running."""
    if sp.get("ref"):
        return sp["label"]
    cs = [c for c in cells_by_scen if c["expected"]]
    if not has_any:
        return sp["label"] + " (pending)"
    if any(c["n_left"] for c in cs):
        return sp["label"] + " (partial)"
    return sp["label"]


def _appx_save_legend(specs, labels, out_stem: Path, which: str) -> list[Path]:
    from matplotlib.lines import Line2D

    def handle(sp):
        return Line2D(
            [0], [0], color=sp["colour"], linestyle=sp["ls"],
            linewidth=sp["lw"] * 1.1, marker=sp["marker"] or "", markersize=3.6,
            markeredgewidth=0.7, markeredgecolor=sp["colour"],
            markerfacecolor=sp["colour"] if sp["filled"] else "white",
            solid_capstyle="butt", dash_capstyle="butt",
        )

    blank = Line2D([], [], linestyle="none", marker="")
    with plt.rc_context(APPX_RC):
        fig = plt.figure()
        if which == "ours":
            # A grid: one column per variant group (bold header row), one row per
            # sampler x Jacobian mode; empty slots keep the rows aligned.
            ncol = len(APPX_OURS_GROUPS)
            nrow = 1 + len(APPX_SAMPLERS) * len(APPX_JACS)
            hs, ls_ = [], []
            for col, grp in enumerate(APPX_OURS_GROUPS):
                hs.append(blank)
                ls_.append(grp[0])
                colspecs = [(sp, lab) for sp, lab in zip(specs, labels) if sp["column"] == col]
                for sp, lab in colspecs:
                    hs.append(handle(sp))
                    ls_.append(lab)
                for _ in range(nrow - 1 - len(colspecs)):
                    hs.append(blank)
                    ls_.append("")
            leg = fig.legend(hs, ls_, loc="center", ncol=ncol, frameon=False,
                             handlelength=2.3, columnspacing=0.8, handletextpad=0.4,
                             labelspacing=0.28, fontsize=6.8)
            for i, t in enumerate(leg.get_texts()):
                if i % nrow == 0:
                    t.set_fontweight("bold")
                    t.set_position((-21, 0))  # header flush left over the handles
        else:
            fig.legend([handle(sp) for sp in specs], labels, loc="center", ncol=3,
                       frameon=False, handlelength=2.3, columnspacing=0.8,
                       handletextpad=0.4, labelspacing=0.28, fontsize=6.8)
        return _save_fig_appx(fig, out_stem)


def _save_fig_appx(fig, out_stem: Path, *, tight: bool = True) -> list[Path]:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_stem.with_suffix(f".{ext}")
        fig.savefig(p, bbox_inches="tight" if tight else None, pad_inches=0.02,
                    dpi=300 if ext == "png" else None)
        paths.append(p)
    plt.close(fig)
    return paths


def _appx_preview(pngs: list[Path], out: Path) -> Path:
    """Stack the panel PNGs + legend PNG vertically into one preview PNG."""
    import matplotlib.image as mpimg

    imgs = [mpimg.imread(p)[..., :3] for p in pngs]
    W = max(im.shape[1] for im in imgs)
    gap = np.ones((30, W, 3), dtype=imgs[0].dtype)
    rows = []
    for im in imgs:
        pad = np.ones((im.shape[0], W - im.shape[1], 3), dtype=im.dtype)
        rows += [np.concatenate([im, pad], axis=1), gap]
    mpimg.imsave(out, np.concatenate(rows[:-1], axis=0))
    return out


def _appendix_all_figures(out: Path, agg: Path) -> list[Path]:
    """Write the six appendix per-step figures (3 metrics x {ours, baselines}).

    Files (NEW names only): ``singles/ns_<metric>_vs_step_all_<set>_<scen>`` (one
    full-text-width panel per scenario), ``singles/ns_legend_all_<set>`` and a
    stacked preview ``ns_<metric>_vs_step_all_<set>_preview.png``. Status is
    recomputed from ``agg`` on every call (see ``_appx_status``).
    """
    if not (agg / "per_step.csv").exists():
        print(f"[ns] --appendix-all: no {agg / 'per_step.csv'}")
        return []
    status = _appx_status(agg)
    written: list[Path] = []
    sdir = out / "singles"
    data_keys_seen: set = set()
    for which in ("ours", "baselines"):
        specs = _appx_series(which)
        legend_done = False
        for metric, ylabel, _stem in METRIC_FIGURES:
            steps = _appx_load_steps(agg, metric)
            data_keys_seen |= {(m, v) for m, v, _sc in steps}
            for sp in specs:
                sp["has_any"] = any(steps.get((*sp["key"], sc)) for sc in SCENARIOS)
            pngs: list[Path] = []
            # Scenario order of the manuscript's wide-panel figures.
            for sc in ("32^2->128^2", "16^2->128^2", "sparse 5%", "sparse 1.5625%"):
                by_key = {sp["key"]: steps.get((*sp["key"], sc), {}) for sp in specs}
                cells = {
                    sp["key"]: _appx_cell(status.get((*sp["key"], sc), {}),
                                          by_key[sp["key"]])
                    for sp in specs
                }
                feet = _appx_feet(specs, by_key, cells)
                nfoot = feet.count("\n") + 1 if feet else 0
                top = APPX_AX_TOP + (0.04 + APPX_FOOT_LINE * nfoot if nfoot else 0)
                H = APPX_AX_BOTTOM + APPX_AX_H + top
                with plt.rc_context(APPX_RC):
                    fig = plt.figure(figsize=(APPX_WIDTH, H))
                    ax = fig.add_axes((
                        APPX_AX_LEFT / APPX_WIDTH, APPX_AX_BOTTOM / H,
                        (APPX_WIDTH - APPX_AX_LEFT - APPX_AX_RIGHT) / APPX_WIDTH,
                        APPX_AX_H / H,
                    ))
                    _appx_draw(ax, specs, by_key, feet)
                    if not any(by_key.values()):
                        ax.text(0.5, 0.5, "no finished runs yet", transform=ax.transAxes,
                                ha="center", va="center", fontsize=7, color="0.4")
                    ax.set_xlabel("Assimilation step", labelpad=1.5)
                    ax.set_ylabel(ylabel)
                    # Same label position on every panel, whatever the ticks.
                    ax.yaxis.set_label_coords(
                        -(APPX_AX_LEFT - 0.1)
                        / (APPX_WIDTH - APPX_AX_LEFT - APPX_AX_RIGHT), 0.5)
                    paths = _save_fig_appx(
                        fig, sdir / f"ns_{metric}_vs_step_all_{which}_{SLUG(sc)}",
                        tight=False)
                written += paths
                pngs.append(paths[1])
            if not legend_done:
                labels = []
                for sp in specs:
                    cs = [_appx_cell(status.get((*sp["key"], sc), {}),
                                     steps.get((*sp["key"], sc)))
                          for sc in SCENARIOS]
                    has_any = any(steps.get((*sp["key"], sc)) for sc in SCENARIOS)
                    labels.append(_appx_legend_label(sp, cs, has_any))
                leg_paths = _appx_save_legend(
                    specs, labels, sdir / f"ns_legend_all_{which}", which)
                written += leg_paths
                legend_done = True
            written.append(_appx_preview(
                pngs + [leg_paths[1]], out / f"ns_{metric}_vs_step_all_{which}_preview.png"))
    known = {sp["key"] for w in ("ours", "baselines") for sp in _appx_series(w)}
    unknown = sorted(k for k in data_keys_seen if k not in known)
    if unknown:
        print(f"[ns] --appendix-all: series in the aggregate NOT drawn: {unknown}")
    return written


def _state_figures(out: Path, state_M: int | None = STATE_M) -> list[Path]:
    """Qualitative vorticity field maps for trajectory ``STATE_TRAJ``: one figure per
    scenario, rows = methods, cols = Truth | Posterior mean | |Error| | Spread (final
    step).

    Reads the self-contained ``results/navier_stokes/states/traj<STATE_TRAJ>/*.npz``
    archives saved by ``run_ns_grid.sh``; skipped with a message if none exist yet.

    ``state_M`` picks the sampler-step count to show (default ``STATE_M`` = 250;
    ``None`` falls back to the widest-coverage saved M). The grid saves one archive
    per M, so this must stay a single M -- otherwise the same method appears once
    per M as separate rows.
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


def _mean_field_spectrum(fields: np.ndarray):
    """Radially-averaged enstrophy spectrum of vorticity fields ``[T, H, W]``.

    The enstrophy spectrum ``Z(k)`` (shell average of $\\tfrac12|\\hat\\omega|^2$)
    is a functional of the vorticity alone -- no stream-function inversion and no
    velocity reconstruction, unlike the kinetic-energy spectrum.

    Returns ``(k, Zk)`` as numpy arrays, with ``Zk`` averaged over the leading
    (assimilation-step) axis -- radial binning is linear, so binning the
    step-averaged modal enstrophy is identical to averaging the per-step spectra.
    Returns ``None`` for a field stack that is not finite (a diverged sampler),
    whose spectrum would be all-NaN and would silently vanish from the panel.
    """
    import torch  # local: the metric figures must not pay for a torch import

    from scisi.metrics.spectral import radial_enstrophy_spectrum

    if not np.isfinite(fields).all():
        return None
    k, zk = radial_enstrophy_spectrum(
        torch.as_tensor(np.ascontiguousarray(fields), dtype=torch.float64)
    )
    return np.asarray(k), np.asarray(zk)


def _assimilated_mean_fields(rec: dict, key: str) -> np.ndarray:
    """The ensemble-mean vorticity over the ASSIMILATED steps: ``[T', H, W]``.

    ``key`` is ``posterior_trajectory`` ``[E, C, H, W, T]`` or
    ``true_trajectory`` ``[1, C, H, W, T]``. The archives store the whole
    trajectory, whose leading ``len_field_history`` steps are the conditioning
    window and were never assimilated; the number of assimilated steps is
    recovered from the saved per-step metric curves (they have one entry per
    assimilated step), falling back to the final step alone.
    """
    traj = np.asarray(rec[key])[:, 0, :, :, :]  # [n, H, W, T]
    n_steps = int(np.asarray(rec.get("per_step_rmse", ())).size) or 1
    fields = traj[..., -n_steps:].mean(axis=0)  # [H, W, T'] (ensemble mean)
    return np.moveaxis(fields, -1, 0)           # [T', H, W]


def _spectrum_figures(out: Path, state_M: int | None = STATE_M) -> list[Path]:
    """Enstrophy-spectrum figures: two panels/files per scenario, all methods + truth.

    Reads the same self-contained saved-state archives as the field maps
    (``results/navier_stokes/states/traj<STATE_TRAJ>/*.npz``) and plots, per
    scenario, the radially-averaged enstrophy spectrum $Z(k)$ of the TRUE
    vorticity field against that of every method's POSTERIOR MEAN, averaged over
    the assimilated steps of trajectory ``STATE_TRAJ`` at ``state_M`` sampler
    steps. Each scenario gets TWO panels: the full spectrum, and a companion
    zoomed to $k \\ge$ ``ZOOM_KMIN``, where the methods actually separate (the
    full range spans several decades, which flattens the differences).

    Because $Z = |k|^2 E$, the log discrepancy against the truth is the same on
    either spectrum up to the spread of $|k|$ within a shell, so these curves
    localise in $k$ what the tabulated ``energy_spec_rmse`` summarises (that
    metric takes the log-spectrum RMSE PER STEP and averages, so the two are
    companions rather than the same number).

    Writes the combined figures, the per-scenario singles, and -- because the
    panels carry a curve the shared method legend has no entry for -- its own
    legend file with Truth prepended.
    """
    written: list[Path] = []
    keys: set = set()
    panels: list[tuple[str, dict]] = []
    slugs: list[str] = []
    for sc in SCENARIOS:
        recs = load_state_records(CASE, scenario=sc, traj=STATE_TRAJ, M=state_M)
        if not recs:
            continue
        series: dict = {}
        truth = None
        for r in recs:
            if truth is None:
                truth = _mean_field_spectrum(_assimilated_mean_fields(r, "true_trajectory"))
            key = series_key(str(r["method"]), str(r.get("variant", "")))
            spec = _mean_field_spectrum(
                _assimilated_mean_fields(r, "posterior_trajectory")
            )
            if spec is None:
                print(f"[ns] spectrum {sc}: {key[0]} diverged (non-finite); skipped")
                continue
            series[key] = spec
            keys.add(key)
        if not series or truth is None:
            continue
        series[TRUTH_KEY] = truth
        panels.append((SCENARIO_LABEL.get(sc, sc), series))
        slugs.append(SLUG(sc))
        print(f"[ns] spectrum {sc}: {len(series) - 1} methods (traj{STATE_TRAJ})")
    if not panels:
        print(
            "[ns] no saved states; cannot draw enstrophy spectra "
            f"(run run_ns_grid.sh with save_states for traj{STATE_TRAJ})"
        )
        return written
    # Full spectrum + the high-wavenumber zoom, from the SAME panel data: 4
    # scenarios x 2 views = the 8 single-panel files the manuscript tiles.
    written += make_spectrum_figure(
        panels, out / "ns_enstrophy_spectrum", ncols=2, panel_slugs=slugs,
    )
    written += make_spectrum_figure(
        panels, out / "ns_enstrophy_spectrum_zoom", ncols=2, panel_slugs=slugs,
        kmin=ZOOM_KMIN,
    )
    written += save_series_legend(
        keys, out / "singles" / "ns_enstrophy_spectrum_legend",
        extra=((TRUTH_LABEL, dict(TRUTH_STYLE, linewidth=1.6)),),
    )
    return written


def _truth_obs_figures(out: Path, state_M: int | None = STATE_M) -> list[Path]:
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
        "--state-M", type=int, default=STATE_M,
        help=f"sampler steps M for the field maps (default: {STATE_M}; "
             "0 = pick the widest-coverage saved M)",
    )
    ap.add_argument(
        "--results-root", type=Path, default=None,
        help="directory holding navier_stokes/ (default: results/ beside this script)",
    )
    ap.add_argument(
        "--curves-only", action="store_true",
        help="draw only the aggregate-based curves and the legend; skip the "
             "field maps, spectra and truth/observation panels (which load "
             "every saved-state archive)",
    )
    ap.add_argument(
        "--fc-per-step", type=Path, default=None,
        help="per-step aggregate of the forecast-correction cells: also write the "
             "ns_<metric>_vs_step_fc figures and singles/ns_legend_fc",
    )
    ap.add_argument(
        "--fc-lineup", choices=("all", "ours", "full"), default="all",
        help="FC figures: every usual method + FC (all); Ours plain + FC plus "
             "FlowDAS+SURGE and EnKF as references (ours); or every method incl. "
             "the classical filters + FC, as full-text-width panels (full)",
    )
    ap.add_argument(
        "--fc-only", action="store_true",
        help="write only the FC per-step figures (needs --fc-per-step)",
    )
    ap.add_argument(
        "--classical", action="store_true",
        help="with --single-M: also draw the classical filters (EnKF, particle "
             "filter; saved at another M) on the per-step curves. Off by default, "
             "matching the manuscript's method set.",
    )
    ap.add_argument(
        "--single-M", type=int, default=None,
        help="the tree holds this one sampler-step count: skip the metric-vs-M "
             "figures (one point per curve says nothing) and draw the per-step "
             "curves and field maps at this M. Takes precedence over --state-M.",
    )
    ap.add_argument(
        "--appendix-all", type=Path, nargs="?", const=Path("-"), default=None,
        help="write ONLY the appendix all-runs per-step figures (set ours + set "
             "baselines, singles/ns_<metric>_vs_step_all_<set>_<scen>) from the "
             "unified aggregate directory given (default: "
             "<results-root>/navier_stokes/aggregated_all)",
    )
    args = ap.parse_args()
    if args.appendix_all is not None:
        if args.results_root is not None:
            set_results_root(args.results_root)
        import figure_common as _fcm

        agg = (args.appendix_all if str(args.appendix_all) != "-"
               else _fcm.RESULTS / CASE / "aggregated_all")
        for p in _appendix_all_figures(Path(args.out), agg):
            print(f"[fig] wrote {p}")
        return
    if args.results_root is not None:
        set_results_root(args.results_root)
    out = Path(args.out)
    state_M = args.single_M or args.state_M or None  # --state-M 0 -> auto-pick
    written: list[Path] = []
    legend_keys: set = set()  # every (method, variant) with data, for the legend file

    for metric, ylabel, stem in () if args.single_M else METRIC_FIGURES:
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

    if args.fc_per_step is not None:
        written += _fc_step_figures(
            out, args.fc_per_step, M=args.single_M, lineup=args.fc_lineup,
        )
        if args.fc_only:
            for p in written:
                print(f"[fig] wrote {p}")
            return
    diverged: list = []
    written += _step_figures(
        out, legend_keys, M=args.single_M,
        include_m_independent=args.single_M is not None and args.classical,
        diverged=diverged,
    )
    for metric, sc, m, v, p0, n_min, full, mid in diverged:
        if metric != "rmse":
            continue
        if mid:
            print(f"[ns] per-step {sc}: {m} {v or ''} -- a trajectory diverged at "
                  f"step {p0}; solid up to step {p0 - 1}, survivors' mean after")
        else:
            print(f"[ns] per-step {sc}: {m} {v or ''} -- mean over {n_min}/{full} "
                  "trajectories (diverged ones excluded by the aggregation)")
    if not args.curves_only:
        written += _state_figures(out, state_M=state_M)
        written += _spectrum_figures(out, state_M=state_M)
        written += _truth_obs_figures(out, state_M=state_M)

    # One shared legend file for all the single-panel metric figures of this case.
    written += save_series_legend(
        legend_keys, out / "singles" / "ns_legend",
        extra_after=DIVERGENCE_LEGEND if any(d[-1] for d in diverged) else (),
    )

    for p in written:
        print(f"[fig] wrote {p}")
    if not written:
        print("[ns] no figures produced (the NS grid has not been run yet).")


if __name__ == "__main__":
    main()
