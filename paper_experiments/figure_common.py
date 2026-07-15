"""Shared styling + plotting for the per-case figure scripts.

``make_analytical_figures.py`` / ``make_ns_figures.py`` / ``make_urban_figures.py``
all import from here so every "metric vs. sampler steps $M$" figure across the
three cases looks identical: the same method colours/markers, the same
publication style, a legend placed BELOW the axes (there are many methods), and
the same graceful handling of a *collapsed* method whose metric blows up off the
top of the axis.

Data source: the reduced-grid tidy results. For each case a metric-vs-M series is
read from ``results/<case>/aggregated/all.csv`` (mean over trajectories/seeds per
``(method, variant, scenario, metric, M)`` -- produced by ``aggregate_<case>.py``);
if that file is absent it falls back to the per-M metric files
``results/<case>/metrics/*.csv``.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
from results_schema import load_records  # noqa: E402

RESULTS = _here / "results"
# In-repo mirror of every generated figure (the make scripts write their primary
# copy under manuscript/figures/<case> and also mirror into here, so the figures
# live next to the experiment code as well as in the manuscript tree).
FIGURES_DIR = _here / "figures"
STEPS: tuple[int, ...] = (50, 100, 250, 500)

# --------------------------------------------------------------------------- #
# Method styling -- one shared ordered series list (cases that lack a method,
# e.g. urban has no EnKF/PF, simply have no data for it and it is skipped).
# Colour-blind-friendly Okabe--Ito base palette + a few extensions. Each entry:
# (csv_method, variant, legend label, colour, linestyle, marker, filled_marker).
# --------------------------------------------------------------------------- #
_SI, _DM, _FM = "#0072B2", "#009E73", "#CC79A7"  # Ours sampler families
SERIES: tuple[tuple, ...] = (
    ("Ours (SI-SDE)", "shared", "Ours SI-SDE (shared)", _SI, "-", "o", True),
    ("Ours (DM-SDE)", "shared", "Ours DM-SDE (shared)", _DM, "-", "s", True),
    ("Ours (FM-ODE)", "shared", "Ours FM-ODE (shared)", _FM, "-", "^", True),
    ("Ours (SI-SDE)", "jacfree", "Ours SI-SDE (Jac-free)", _SI, "--", "o", False),
    ("Ours (DM-SDE)", "jacfree", "Ours DM-SDE (Jac-free)", _DM, "--", "s", False),
    ("Ours (FM-ODE)", "jacfree", "Ours FM-ODE (Jac-free)", _FM, "--", "^", False),
    ("FlowDAS", None, "FlowDAS", "#D55E00", "-", "D", True),
    ("SURGE (FlowDAS)", None, "FlowDAS + SURGE", "#E69F00", "-", "v", True),
    ("SDA", None, "SDA", "#8C510A", "-", "P", True),
    ("SURGE (SDA)", None, "SDA + SURGE", "#6A3D9A", "-", "X", True),
    ("D-Flow SGLD", None, "D-Flow SGLD", "#808000", "-", "h", True),
    ("Guided FM (FIG)", None, "Guided FM (FIG)", "#F0027F", "-", "*", True),
    ("EnKF", None, "EnKF", "#000000", ":", "p", True),
    ("Particle filter", None, "Particle filter", "#666666", ":", "d", True),
)

# Compact scenario labels (canonical Scenario value -> LaTeX label).
SCENARIO_LABEL: dict[str, str] = {
    "16^2->128^2": r"$16^2\!\to\!128^2$",
    "32^2->128^2": r"$32^2\!\to\!128^2$",
    "sparse 5%": r"sparse $5\%$",
    "sparse 1.5625%": r"sparse $1.5625\%$",
    "analytical": "analytical",
}


def slugify(s: str) -> str:
    """Filename slug: runs of non-alphanumerics -> ``_`` (lowercased)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 11,
        "axes.linewidth": 0.9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8.2,
        "figure.dpi": 150,
    })


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def _canon_variant(variant: str | None) -> str | None:
    """Map raw CSV variant names onto the canonical ``SERIES`` variants.

    The grid scripts stamp shared-Jacobian runs as ``shared_jac<refresh_every>``
    (e.g. ``shared_jac5``); all of them are the one "shared" series regardless
    of the refresh interval. Empty strings become ``None`` (no variant).
    """
    if not variant:
        return None
    if variant.startswith("shared"):
        return "shared"
    return variant


def load_metric_vs_M(
    case: str, metric: str, scenario: str, *, steps: tuple[int, ...] = STEPS
) -> dict[tuple[str, str | None], dict[int, float]]:
    """Return {(method, variant) -> {M: value}} for one (case, metric, scenario).

    Prefers ``results/<case>/aggregated/all.csv``; falls back to the per-M metric
    files. ``steps`` restricts the sampler-step axis (a case whose grid uses a
    different ladder than the default ``STEPS`` passes its own). Returns ``{}``
    when no results exist yet (the case's grid has not run).
    """
    agg = RESULTS / case / "aggregated" / "all.csv"
    files: list[Path]
    if agg.exists():
        files = [agg]
    else:
        files = sorted((RESULTS / case / "metrics").glob("*.csv")) if (
            RESULTS / case / "metrics"
        ).exists() else []
    out: dict[tuple[str, str | None], dict[int, float]] = {}
    for f in files:
        try:
            recs = load_records(f)
        except Exception:
            continue
        for r in recs:
            if r.metric != metric or r.scenario != scenario or r.M not in steps:
                continue
            if r.value is None or (isinstance(r.value, float) and math.isnan(r.value)):
                continue
            out.setdefault((r.method, _canon_variant(r.variant)), {})[int(r.M)] = float(r.value)
    return out


def load_metric_vs_step(
    case: str, metric: str, scenario: str, *, M: int | None = None
) -> dict[tuple[str, str | None], dict[int, float]]:
    """Return {(method, variant) -> {step: value}} for one (case, metric, scenario).

    Reads the trajectory-aggregated per-step curves from
    ``results/<case>/aggregated/per_step.csv`` (produced by ``aggregate_<case>.py``);
    ``value`` is the mean over trajectories at that assimilation step. Returns
    ``{}`` when the file is absent (analytical, or the grid/aggregation has not
    run). When several ``M`` are present, restrict to ``M`` if given, else use the
    largest ``M`` available so the curve is well-defined.
    """
    import csv as _csv

    path = RESULTS / case / "aggregated" / "per_step.csv"
    if not path.exists():
        return {}
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            if row.get("metric") != metric or row.get("scenario") != scenario:
                continue
            rows.append(row)
    if not rows:
        return {}
    if M is None:
        Ms = {int(float(r["M"])) for r in rows if r.get("M") not in (None, "")}
        M = max(Ms) if Ms else None
    out: dict[tuple[str, str | None], dict[int, float]] = {}
    for r in rows:
        if M is not None and r.get("M") not in (None, "") and int(float(r["M"])) != M:
            continue
        val = r.get("value")
        if val in (None, "") or (isinstance(val, str) and val.strip().lower() == "nan"):
            continue
        v = float(val)
        if math.isnan(v):
            continue
        variant = _canon_variant(r.get("variant"))
        out.setdefault((r["method"], variant), {})[int(float(r["step"]))] = v
    return out


# --------------------------------------------------------------------------- #
# The shared "metric vs M" plotter (one or more scenario panels; legend below)
# --------------------------------------------------------------------------- #


def _plot_panel(
    ax, series: dict, *, logy: bool, steps: tuple[int, ...] = STEPS,
    ycap: float | None = None,
) -> list:
    """Draw one panel; return the (handle, label) list in SERIES order."""
    # Auto-detect COLLAPSED / off-scale series (min value >> the rest) so one
    # blown-up method (e.g. FIG) doesn't stretch the axis over many empty decades.
    finite = {k: [v for v in s.values()] for k, s in series.items() if s}
    all_vals = [v for vs in finite.values() for v in vs]
    off_scale: set = set()
    y_hi = y_lo = None
    if all_vals:
        on_scale_vals = list(all_vals)
        for k, vs in finite.items():
            others = [v for kk, vv in finite.items() if kk != k for v in vv]
            if others and min(vs) > 50 * max(others):
                off_scale.add(k)
        on_scale_vals = [
            v for k, vs in finite.items() if k not in off_scale for v in vs
        ]
        # An explicit cap excludes individual DIVERGING points (a series that
        # blows up at large M) from the axis range; the line simply exits
        # through the top of the axis.
        if ycap is not None:
            on_scale_vals = [v for v in on_scale_vals if v <= ycap] or on_scale_vals
        if on_scale_vals and logy:
            y_lo = min(on_scale_vals) * 0.4
            y_hi = max(on_scale_vals) * 6.0
        elif on_scale_vals:
            span = max(on_scale_vals) - min(on_scale_vals)
            y_lo = min(on_scale_vals) - 0.08 * span
            y_hi = max(on_scale_vals) + 0.12 * span
    y_shelf = (y_hi / 2.2) if (y_hi is not None and logy) else None

    handles: list = []
    for method, variant, label, colour, ls, marker, filled in SERIES:
        s = series.get((method, variant))
        if not s:
            continue
        xs = [m for m in steps if m in s]
        ys = [s[m] for m in xs]
        if not xs:
            continue
        is_off = (method, variant) in off_scale and y_shelf is not None
        if is_off:
            ys = [y_shelf] * len(xs)
        (h,) = ax.plot(
            xs, ys, color=colour, linestyle=ls, marker=marker,
            markersize=7.5 if is_off else 6.5, markeredgewidth=1.3,
            markeredgecolor=colour, markerfacecolor=(colour if filled else "white"),
            linewidth=2.1 if (variant is not None or is_off) else 1.7,
            zorder=3, clip_on=True,
        )
        handles.append((h, label))
        if is_off:
            ax.text(
                math.sqrt(xs[0] * xs[-1]), y_shelf * 2.2, "collapsed (off scale)",
                ha="center", va="bottom", fontsize=8.0, color=colour,
                fontstyle="italic",
            )
            ax.axhline(y_hi / 4.5, color="0.6", lw=0.7, ls=(0, (2, 3)), zorder=1)

    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if y_lo is not None:
        ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(list(steps))
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(steps[0] * 0.9, steps[-1] * 1.1)
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.15)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(which="both", direction="out", length=4, width=0.8)
    return handles


def _save_fig(fig, out_stem: Path, *, png_dpi: int = 300) -> list[Path]:
    """Save ``fig`` as ``<out_stem>.pdf`` + ``.png`` (tight bbox) and close it."""
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        p = out_stem.with_suffix(f".{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=png_dpi if ext == "png" else None)
        written.append(p)
    plt.close(fig)
    return written


def _filter_panels(
    panels: list[tuple[str, dict]], panel_slugs: list[str] | None
) -> tuple[list[tuple[str, dict]], list[str]]:
    """Drop empty panels, keeping the (optional) per-panel filename slugs in sync.

    Without explicit slugs they are derived from the panel titles, which works
    for plain-text titles but NOT for LaTeX ones (``$16^2\\!\\to\\!128^2$`` slugs
    as ``16_2_to_128_2``); the make scripts pass slugs built from the raw
    scenario names so the singles match the state-figure stems.
    """
    slugs = panel_slugs if panel_slugs is not None else [None] * len(panels)
    pairs = [(p, sl) for p, sl in zip(panels, slugs) if any(p[1].values())]
    return (
        [p for p, _ in pairs],
        [sl if sl is not None else slugify(p[0]) for p, sl in pairs],
    )


def _save_panel_singles(
    panels: list[tuple[str, dict]],
    slugs: list[str],
    draw_panel,
    ylabel: str,
    xlabel: str,
    out_stem: Path,
) -> list[Path]:
    """Save each panel as its own standalone file (no title, no legend).

    The files land in ``<out_stem's dir>/singles/<out_stem's name>_<slug>.pdf``
    (+ ``.png``) -- a folder that can be copied straight into the manuscript
    figures tree. The manuscript caption describes the panel content and one
    shared legend file (see :func:`save_series_legend`) serves a whole cluster
    of such subfigures, so the singles themselves carry neither.
    """
    written: list[Path] = []
    sdir = out_stem.parent / "singles"
    for (_title, series), slug in zip(panels, slugs):
        fig, ax = plt.subplots(figsize=(4.6, 3.5))
        draw_panel(ax, series)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        name = out_stem.name + (f"_{slug}" if slug else "")
        written += _save_fig(fig, sdir / name)
    return written


def save_series_legend(
    keys, out_stem: Path, *, ncol: int = 4
) -> list[Path]:
    """Save a standalone legend-only figure for the given ``(method, variant)`` keys.

    Entries follow the shared ``SERIES`` order/styling, so one legend file serves
    every cluster of single-panel figures of a case (the singles carry no legend
    of their own). ``keys`` is any iterable of ``(method, variant)`` pairs as
    found in the loaded series dicts (raw variants are canonicalised). Writes
    ``<out_stem>.pdf`` + ``.png``; returns the written paths ([] if no key matches).
    """
    from matplotlib.lines import Line2D

    canon = {(m, _canon_variant(v)) for m, v in keys}
    entries = [s for s in SERIES if (s[0], s[1]) in canon]
    if not entries:
        return []
    apply_style()
    handles = [
        Line2D(
            [0], [0], color=colour, linestyle=ls, marker=marker,
            markersize=6.5, markeredgewidth=1.3, markeredgecolor=colour,
            markerfacecolor=(colour if filled else "white"), linewidth=2.0,
        )
        for _m, _v, _label, colour, ls, marker, filled in entries
    ]
    fig = plt.figure()
    fig.legend(
        handles, [e[2] for e in entries], loc="center", ncol=ncol,
        frameon=False, handlelength=2.3, columnspacing=1.3,
        handletextpad=0.5, labelspacing=0.4, fontsize=9,
    )
    return _save_fig(fig, out_stem)


def make_vs_M_figure(
    panels: list[tuple[str, dict]],
    ylabel: str,
    out_stem: Path,
    *,
    ncols: int | None = None,
    logy: bool = True,
    steps: tuple[int, ...] = STEPS,
    ycap: float | None = None,
    panel_slugs: list[str] | None = None,
    singles: bool = True,
) -> list[Path]:
    """Render a (multi-panel) metric-vs-M figure with a shared legend below.

    ``ycap`` (optional) excludes values above it from the y-axis range so a
    single diverging series doesn't stretch the axis; its line exits the top.

    ``panels`` is a list of ``(panel_title, series_dict)`` where ``series_dict``
    maps ``(method, variant) -> {M: value}``. One panel -> a single plot; several
    panels (e.g. one per observation scenario) -> a subplot grid sharing one
    legend. ``steps`` sets the sampler-step ticks/limits and must match the
    ``steps`` passed to :func:`load_metric_vs_M`. Saves ``<out_stem>.pdf`` and
    ``.png``; returns the written paths.

    With ``singles`` (default) each panel is ALSO saved standalone -- no title,
    no legend -- into ``singles/<stem>_<slug>.pdf`` next to the combined file
    (``panel_slugs`` names them; see :func:`_save_panel_singles`). The combined
    figure remains the overview; the singles are the manuscript subfigures.
    """
    apply_style()
    panels, slugs = _filter_panels(panels, panel_slugs)
    if not panels:
        return []
    n = len(panels)
    ncols = ncols or (1 if n == 1 else (2 if n <= 4 else 3))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows),
        squeeze=False, sharex=True,
    )
    flat = axes.flatten()
    legend_pairs: dict[str, object] = {}
    for i, (title, series) in enumerate(panels):
        ax = flat[i]
        for h, lab in _plot_panel(ax, series, logy=logy, steps=steps, ycap=ycap):
            legend_pairs.setdefault(lab, h)
        if n > 1:
            ax.set_title(title)
        ax.set_xlabel(r"Number of sampler steps $M$")
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
    for j in range(n, len(flat)):
        flat[j].set_visible(False)

    # One shared legend directly below the whole figure, in SERIES order. It is
    # anchored just under the canvas (no reserved whitespace); bbox_inches="tight"
    # at save time grows the saved bbox to include it, so the gap stays minimal.
    labels = [s[2] for s in SERIES if s[2] in legend_pairs]
    handles = [legend_pairs[l] for l in labels]
    if len(labels) <= 4:
        ncol_leg = max(2, len(labels))
    else:
        ncol_leg = 4 if ncols > 1 else 3  # match legend width to figure width
    fig.tight_layout()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
        ncol=ncol_leg, frameon=False, handlelength=2.3, columnspacing=1.3,
        handletextpad=0.5, labelspacing=0.4, fontsize=9,
    )

    written = _save_fig(fig, out_stem)
    if singles:
        written += _save_panel_singles(
            panels, slugs,
            lambda ax, s: _plot_panel(ax, s, logy=logy, steps=steps, ycap=ycap),
            ylabel, r"Number of sampler steps $M$", out_stem,
        )
    return written


# --------------------------------------------------------------------------- #
# The shared "metric vs assimilation step" plotter (per-step curves; legend below)
# --------------------------------------------------------------------------- #


def _plot_step_panel(ax, series: dict, *, logy: bool) -> list:
    """Draw one metric-vs-assimilation-step panel; return (handle, label) pairs."""
    handles: list = []
    for method, variant, label, colour, ls, marker, filled in SERIES:
        s = series.get((method, variant))
        if not s:
            continue
        xs = sorted(s)
        ys = [s[x] for x in xs]
        (h,) = ax.plot(
            xs, ys, color=colour, linestyle=ls, marker=marker,
            markersize=6.0, markeredgewidth=1.2, markeredgecolor=colour,
            markerfacecolor=(colour if filled else "white"),
            linewidth=2.1 if variant is not None else 1.7, zorder=3,
        )
        handles.append((h, label))
    if logy:
        ax.set_yscale("log")
    ax.margins(x=0.03)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.15)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(which="both", direction="out", length=4, width=0.8)
    return handles


def make_vs_step_figure(
    panels: list[tuple[str, dict]],
    ylabel: str,
    out_stem: Path,
    *,
    ncols: int | None = None,
    logy: bool = True,
    panel_slugs: list[str] | None = None,
    singles: bool = True,
) -> list[Path]:
    """Render a (multi-panel) metric-vs-assimilation-step figure, legend below.

    Mirrors :func:`make_vs_M_figure` but the x-axis is the assimilation-step index
    (linear), one line per method. ``panels`` is ``[(panel_title, series_dict)]``
    with ``series_dict`` mapping ``(method, variant) -> {step: value}`` (from
    :func:`load_metric_vs_step`). Saves ``<out_stem>.pdf`` and ``.png``, plus
    (with ``singles``, default) one standalone title-/legend-free file per panel
    under ``singles/`` (see :func:`_save_panel_singles`).
    """
    apply_style()
    panels, slugs = _filter_panels(panels, panel_slugs)
    if not panels:
        return []
    n = len(panels)
    ncols = ncols or (1 if n == 1 else (2 if n <= 4 else 3))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows), squeeze=False, sharex=True,
    )
    flat = axes.flatten()
    legend_pairs: dict[str, object] = {}
    for i, (title, series) in enumerate(panels):
        ax = flat[i]
        for h, label in _plot_step_panel(ax, series, logy=logy):
            legend_pairs.setdefault(label, h)
        if n > 1:
            ax.set_title(title)
        ax.set_xlabel("Assimilation step")
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
    for j in range(n, len(flat)):
        flat[j].set_visible(False)

    labels = [s[2] for s in SERIES if s[2] in legend_pairs]
    handles = [legend_pairs[l] for l in labels]
    ncol_leg = 4 if len(labels) > 8 else max(2, len(labels))
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02),
        ncol=ncol_leg, frameon=False, handlelength=2.3, columnspacing=1.3,
        handletextpad=0.5, labelspacing=0.5,
    )
    fig.tight_layout(rect=(0, 0.06 + 0.03 * math.ceil(len(labels) / ncol_leg), 1, 1))

    written = _save_fig(fig, out_stem)
    if singles:
        written += _save_panel_singles(
            panels, slugs,
            lambda ax, s: _plot_step_panel(ax, s, logy=logy),
            ylabel, "Assimilation step", out_stem,
        )
    return written


# --------------------------------------------------------------------------- #
# Saved-state field maps (fig: qualitative posterior vs. truth for trajectory 1)
#
# The grids save the raw posterior ensemble + truth for trajectory 1 to
# ``results/<case>/states/traj1/*.npz`` (self-contained: ``posterior_trajectory``
# ``[E, C, H, W, T]``, ``true_trajectory`` ``[1, C, H, W, T]``, plus method /
# scenario / variant metadata). These helpers turn those archives into
# side-by-side field maps at the final assimilated step.
# --------------------------------------------------------------------------- #


def load_state_records(
    case: str, scenario: str | None = None, *, traj: int = 1, M: int | None = None
) -> list[dict]:
    """Load the saved-state ``.npz`` archives for one case and trajectory.

    Reads ``results/<case>/states/traj<traj>/``. Returns a list of dicts (one per
    method/variant cell), each carrying the archive's arrays plus a ``_path`` key.
    Filtered to ``scenario`` when given.

    The grids save one archive per ``(method, variant, scenario, M)``, so a tree
    covering several sampler-step counts holds several archives per method. A
    field map must show ONE M or it silently plots the same method twice at
    different M: restrict to ``M`` if given, else pick the M with the widest
    method coverage (ties -> the largest such M). Coverage, not size, is the right
    default: only a few methods are re-run at the top of the M ladder, so keying
    on the largest M would silently drop most methods from the field maps.
    Returns ``[]`` when the case's grid has not saved states for that trajectory.
    """
    sdir = RESULTS / case / "states" / f"traj{traj}"
    if not sdir.exists():
        return []
    recs: list[dict] = []
    for f in sorted(sdir.glob("*.npz")):
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:
            continue
        rec = {k: d[k] for k in d.files}
        rec["_path"] = f
        if scenario is not None and str(rec.get("scenario")) != scenario:
            continue
        recs.append(rec)

    def _M(rec: dict) -> int | None:
        try:
            return int(rec["M"])
        except (KeyError, TypeError, ValueError):
            return None

    if M is None:
        coverage: dict[int, set] = {}
        for r in recs:
            m = _M(r)
            if m is None:
                continue
            key = (str(r.get("method")), _canon_variant(str(r.get("variant", ""))))
            coverage.setdefault(m, set()).add(key)
        if coverage:
            M = max(coverage, key=lambda m: (len(coverage[m]), m))
    if M is not None:
        recs = [r for r in recs if _M(r) in (M, None)]
    return recs


def _state_series_key(method: str, variant: str | None) -> tuple[int, str]:
    """Return ``(order_index, pretty_label)`` for a state record's method/variant.

    Orders rows by the shared ``SERIES`` list so the field maps match the
    metric-vs-M figures; unknown methods sort last under their raw name.
    """
    variant = _canon_variant(variant)
    for i, (m, v, label, *_rest) in enumerate(SERIES):
        if m == method and v == variant:
            return i, label
    return len(SERIES), method + (f" ({variant})" if variant else "")


def _state_rows_and_scales(records: list[dict], field_fn, diverging: bool):
    """Shared reduction behind the state figures.

    Returns ``(rows, truth2d, scales)`` where ``rows`` is
    ``[(label, mean, |error|, spread)]`` in ``SERIES`` order and ``scales`` holds
    the colour limits shared across methods: the Truth/Mean field range plus the
    common ``|error|`` and spread maxima. A diverged method can leave NaN/inf in
    its posterior; those must not poison the shared scales (a NaN vmax blanks the
    column for every method), so every reduction runs over finite values only.
    """
    records = sorted(
        records, key=lambda r: _state_series_key(str(r["method"]), str(r.get("variant", "")))
    )
    rows = []
    truth2d = None
    for r in records:
        label = _state_series_key(str(r["method"]), str(r.get("variant", "")))[1]
        post = field_fn(np.asarray(r["posterior_trajectory"]))  # [E, H, W]
        truth = field_fn(np.asarray(r["true_trajectory"]))[0]    # [H, W]
        mean = post.mean(axis=0)
        spread = post.std(axis=0)
        err = np.abs(mean - truth)
        rows.append((label, mean, err, spread))
        if truth2d is None:
            truth2d = truth

    def _finite_max(a: np.ndarray) -> float:
        finite = a[np.isfinite(a)]
        return float(finite.max()) if finite.size else 0.0

    def _robust_vmax(fields: list[np.ndarray]) -> float:
        """A shared 0..vmax scale that one blown-up method cannot destroy.

        A method that partially diverges can have an |error| or spread an order of
        magnitude above every other method; keying the shared scale on the raw
        maximum then pushes all the well-behaved methods into the bottom few
        percent of the colour map, and their panels read as uniformly black. So
        take each method's 99th percentile and cap the scale at a small multiple
        of the median of those -- the outlier simply saturates at the top of the
        map, which is exactly the message its panel should carry.
        """
        p99 = [
            float(np.percentile(f[np.isfinite(f)], 99))
            for f in fields if np.isfinite(f).any()
        ]
        if not p99:
            return 1.0
        med = float(np.median(p99))
        return (min(max(p99), 4.0 * med) if med > 0 else max(p99)) or 1.0

    if diverging:
        m = _finite_max(np.abs(truth2d)) or 1.0
        f_vmin, f_vmax = -m, m
    else:
        finite_truth = truth2d[np.isfinite(truth2d)]
        f_vmin = float(finite_truth.min()) if finite_truth.size else 0.0
        f_vmax = _finite_max(truth2d) or 1.0
    scales = {
        "field": (f_vmin, f_vmax),
        "err": (0.0, _robust_vmax([r[2] for r in rows])),
        "spread": (0.0, _robust_vmax([r[3] for r in rows])),
    }
    return rows, truth2d, scales


def make_state_panel_singles(
    records: list[dict],
    field_fn,
    out_stem: Path,
    *,
    cbar_label: str = "",
    cmap: str = "viridis",
    diverging: bool = False,
    panel_size: float = 2.6,
) -> list[Path]:
    """Save ONE bare image file per (method, quantity) at the final assimilated step.

    These are the atoms of the manuscript's per-quantity field figures: each file
    is a single field map with no title, no axes, no colourbar and no method label
    (the LaTeX subcaption names the method), so a cluster of them tiles into a
    subfigure grid. Files land next to the combined figure, under ``singles/``:

    * ``<stem>__truth.pdf``            -- the shared truth field
    * ``<stem>__mean__<method>.pdf``   -- posterior ensemble mean
    * ``<stem>__std__<method>.pdf``    -- posterior ensemble std (spread)
    * ``<stem>__abserr__<method>.pdf`` -- $|$posterior mean $-$ truth$|$

    Colour scales are shared across methods per quantity (identical to the
    combined figure's), so panels are directly comparable; each quantity's scale
    is also written out once as a standalone horizontal colourbar
    ``<stem>__cbar_{field,std,abserr}.pdf`` for the figure to carry a single bar.
    Truth and mean share the field scale (and hence ``cbar_field``).
    """
    if not records:
        return []
    rows, truth2d, scales = _state_rows_and_scales(records, field_fn, diverging)
    apply_style()
    sdir = out_stem.parent / "singles"
    written: list[Path] = []

    def _panel(field, kw, name: str) -> None:
        nonlocal written
        fig, ax = plt.subplots(figsize=(panel_size, panel_size))
        ax.imshow(field, origin="lower", **kw)
        # A sampler that blew up leaves an all-NaN field, which imshow renders as
        # blank white -- indistinguishable from a legitimately near-zero panel.
        # Say so on the panel instead.
        if not np.isfinite(field).any():
            ax.set_facecolor("0.9")
            ax.text(0.5, 0.5, "diverged\n(NaN)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="0.25")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        written += _save_fig(fig, sdir / name, png_dpi=200)

    field_kw = dict(cmap=cmap, vmin=scales["field"][0], vmax=scales["field"][1])
    err_kw = dict(cmap="magma", vmin=0.0, vmax=scales["err"][1])
    std_kw = dict(cmap="magma", vmin=0.0, vmax=scales["spread"][1])

    _panel(truth2d, field_kw, f"{out_stem.name}__truth")
    for label, mean, err, spread in rows:
        slug = slugify(label)
        _panel(mean, field_kw, f"{out_stem.name}__mean__{slug}")
        _panel(spread, std_kw, f"{out_stem.name}__std__{slug}")
        _panel(err, err_kw, f"{out_stem.name}__abserr__{slug}")

    # Standalone horizontal colourbars: one per shared scale, so each manuscript
    # figure shows its scale once instead of repeating it under every panel.
    for key, kw, label in (
        ("field", field_kw, cbar_label or "field"),
        ("std", std_kw, f"{cbar_label} std" if cbar_label else "std"),
        ("abserr", err_kw, f"$|$error$|$ ({cbar_label})" if cbar_label else "$|$error$|$"),
    ):
        fig, ax = plt.subplots(figsize=(5.0, 0.5))
        norm = matplotlib.colors.Normalize(vmin=kw["vmin"], vmax=kw["vmax"])
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=kw["cmap"])
        fig.colorbar(sm, cax=ax, orientation="horizontal", label=label)
        written += _save_fig(fig, sdir / f"{out_stem.name}__cbar_{key}")
    return written


def save_field_panel(
    out_stem: Path,
    field: np.ndarray | None = None,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    scatter: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    background: np.ndarray | None = None,
    extent: tuple[float, float, float, float] | None = None,
    panel_size: float = 2.6,
) -> list[Path]:
    """Save one bare field map (no title/axes/colourbar) as ``<out_stem>.pdf`` + ``.png``.

    The atom the manuscript's subfigure grids are tiled from -- the LaTeX
    subcaption carries the label, and one standalone colourbar file serves the
    whole grid (see :func:`make_state_panel_singles`).

    ``field`` is drawn with ``imshow``. ``scatter`` is an ``(x, y, values)``
    triple drawn as points on the ``cmap``/``vmin``/``vmax`` scale -- used for the
    sparse-sensor observation panels, where the observed locations are what
    matters; pass ``background`` to show the underlying field behind them in
    faint grey. ``extent`` sets the imshow extent, so a low-resolution observed
    field (e.g. $16^2$) can be drawn on the full-resolution axes and stay
    visually comparable in size to the other panels.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(panel_size, panel_size))
    if background is not None:
        ax.imshow(background, origin="lower", cmap="Greys", alpha=0.28,
                  extent=extent, aspect="equal")
    if field is not None:
        ax.imshow(field, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                  extent=extent, aspect="equal", interpolation="nearest")
    if scatter is not None:
        xs, ys, vals = scatter
        ax.scatter(xs, ys, c=vals, cmap=cmap, vmin=vmin, vmax=vmax, s=3.0,
                   linewidths=0.0, marker="s")
        ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    return _save_fig(fig, out_stem, png_dpi=200)


def make_state_field_figure(
    records: list[dict],
    field_fn,
    out_stem: Path,
    *,
    cbar_label: str = "",
    cmap: str = "viridis",
    diverging: bool = False,
    singles: bool = True,
) -> list[Path]:
    """Render final-step field maps: rows = methods, cols = Truth | Mean | |Error| | Spread.

    ``field_fn`` maps a trajectory array ``[n, C, H, W, T]`` to a stack of 2D
    fields ``[n, H, W]`` at the final step (e.g. pick a channel or take velocity
    magnitude). Truth is shared across methods, so the Truth/Mean columns share
    one colour scale; ``|Error|`` and ``Spread`` each share their own scale so
    methods are directly comparable. Writes ``<out_stem>.pdf`` + ``.png``.

    With ``singles`` (default) each method row is ALSO saved as its own 1x4
    strip -- same shared colour scales and colourbars, but no method label (the
    manuscript subcaption names the method) -- under
    ``singles/<stem>__<method_slug>.pdf``.
    """
    if not records:
        return []
    # Column colour scales. Truth/Mean share a scale from the truth field
    # (symmetric about 0 for signed / diverging fields); |Error| and Spread each
    # span 0..max over all methods so rows are comparable.
    rows, truth2d, scales = _state_rows_and_scales(records, field_fn, diverging)
    (f_vmin, f_vmax) = scales["field"]
    err_max = scales["err"][1]
    spread_max = scales["spread"][1]

    apply_style()
    col_titles = ("Truth", "Posterior mean", r"$|\mathrm{error}|$", "Spread")
    nrows = len(rows)
    fig, axes = plt.subplots(
        nrows, 4, figsize=(3.0 * 4, 3.0 * nrows), squeeze=False
    )

    for i, (label, mean, err, spread) in enumerate(rows):
        panels = (
            (truth2d, dict(cmap=cmap, vmin=f_vmin, vmax=f_vmax)),
            (mean, dict(cmap=cmap, vmin=f_vmin, vmax=f_vmax)),
            (err, dict(cmap="magma", vmin=0.0, vmax=err_max)),
            (spread, dict(cmap="magma", vmin=0.0, vmax=spread_max)),
        )
        for j, (field, kw) in enumerate(panels):
            ax = axes[i][j]
            ax.imshow(field, origin="lower", **kw)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(col_titles[j])
            if j == 0:
                ax.set_ylabel(label, fontsize=9)

    # One colourbar per distinct scale, under its column(s). Column scales are
    # shared across rows, so the top row's mappables are representative.
    im_field = axes[0][0].images[0]
    im_err = axes[0][2].images[0]
    im_spread = axes[0][3].images[0]
    fig.colorbar(im_field, ax=axes[:, :2], location="bottom", shrink=0.5,
                 pad=0.02, label=cbar_label or "field")
    fig.colorbar(im_err, ax=axes[:, 2], location="bottom", shrink=0.8, pad=0.02)
    fig.colorbar(im_spread, ax=axes[:, 3], location="bottom", shrink=0.8, pad=0.02)

    written = _save_fig(fig, out_stem, png_dpi=200)

    # Per-method single-row strips (same shared scales, so subfigures built from
    # them stay directly comparable across methods).
    if singles:
        sdir = out_stem.parent / "singles"
        for label, mean, err, spread in rows:
            fig1, axs1 = plt.subplots(1, 4, figsize=(3.0 * 4, 3.6), squeeze=False)
            panels = (
                (truth2d, dict(cmap=cmap, vmin=f_vmin, vmax=f_vmax)),
                (mean, dict(cmap=cmap, vmin=f_vmin, vmax=f_vmax)),
                (err, dict(cmap="magma", vmin=0.0, vmax=err_max)),
                (spread, dict(cmap="magma", vmin=0.0, vmax=spread_max)),
            )
            for j, (field, kw) in enumerate(panels):
                ax = axs1[0][j]
                ax.imshow(field, origin="lower", **kw)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(col_titles[j])
            fig1.colorbar(axs1[0][0].images[0], ax=axs1[:, :2], location="bottom",
                          shrink=0.5, pad=0.02, label=cbar_label or "field")
            fig1.colorbar(axs1[0][2].images[0], ax=axs1[:, 2], location="bottom",
                          shrink=0.8, pad=0.02)
            fig1.colorbar(axs1[0][3].images[0], ax=axs1[:, 3], location="bottom",
                          shrink=0.8, pad=0.02)
            written += _save_fig(
                fig1, sdir / f"{out_stem.name}__{slugify(label)}", png_dpi=200
            )
    return written


# --------------------------------------------------------------------------- #
# Mirror generated figures into a second location (in-repo figures/ tree)
# --------------------------------------------------------------------------- #


def mirror_to(written: list[Path], mirror_dir: Path) -> list[Path]:
    """Copy already-written figure files into ``mirror_dir`` (by basename).

    Lets the make scripts keep one primary copy (``manuscript/figures/<case>``)
    and a second in-repo copy (``paper_experiments/figures/<case>``) without
    re-rendering. Skips any file already at its destination (e.g. when the primary
    ``--out`` IS the mirror dir). Returns the paths actually copied.
    """
    import shutil

    if not written:
        return []
    mirror_dir.mkdir(parents=True, exist_ok=True)
    copies: list[Path] = []
    for p in written:
        dest = mirror_dir / p.name
        if dest.resolve() == p.resolve():
            continue
        shutil.copy2(p, dest)
        copies.append(dest)
    return copies


def mirror_figures(written: list[Path], mirror_root: Path) -> list[Path]:
    """:func:`mirror_to` that keeps the ``singles/`` subfolder structure.

    Combined figures go to ``mirror_root``, per-panel singles (anything written
    into a ``singles/`` directory) to ``mirror_root/singles``.
    """
    combined = [p for p in written if p.parent.name != "singles"]
    singles = [p for p in written if p.parent.name == "singles"]
    return mirror_to(combined, mirror_root) + mirror_to(singles, mirror_root / "singles")


__all__ = [
    "STEPS", "SERIES", "SCENARIO_LABEL", "RESULTS", "FIGURES_DIR",
    "apply_style", "slugify", "load_metric_vs_M", "make_vs_M_figure",
    "load_metric_vs_step", "make_vs_step_figure", "save_series_legend",
    "load_state_records", "make_state_field_figure", "make_state_panel_singles",
    "mirror_to", "mirror_figures",
]
