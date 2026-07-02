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


def load_metric_vs_M(
    case: str, metric: str, scenario: str
) -> dict[tuple[str, str | None], dict[int, float]]:
    """Return {(method, variant) -> {M: value}} for one (case, metric, scenario).

    Prefers ``results/<case>/aggregated/all.csv``; falls back to the per-M metric
    files. Returns ``{}`` when no results exist yet (the case's grid has not run).
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
            if r.metric != metric or r.scenario != scenario or r.M not in STEPS:
                continue
            if r.value is None or (isinstance(r.value, float) and math.isnan(r.value)):
                continue
            out.setdefault((r.method, r.variant), {})[int(r.M)] = float(r.value)
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
        variant = r.get("variant") or None
        out.setdefault((r["method"], variant), {})[int(float(r["step"]))] = v
    return out


# --------------------------------------------------------------------------- #
# The shared "metric vs M" plotter (one or more scenario panels; legend below)
# --------------------------------------------------------------------------- #


def _plot_panel(ax, series: dict, *, logy: bool) -> list:
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
        xs = [m for m in STEPS if m in s]
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
    ax.set_xticks(list(STEPS))
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(STEPS[0] * 0.9, STEPS[-1] * 1.1)
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.15)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(which="both", direction="out", length=4, width=0.8)
    return handles


def make_vs_M_figure(
    panels: list[tuple[str, dict]],
    ylabel: str,
    out_stem: Path,
    *,
    ncols: int | None = None,
    logy: bool = True,
) -> list[Path]:
    """Render a (multi-panel) metric-vs-M figure with a shared legend below.

    ``panels`` is a list of ``(panel_title, series_dict)`` where ``series_dict``
    maps ``(method, variant) -> {M: value}``. One panel -> a single plot; several
    panels (e.g. one per observation scenario) -> a subplot grid sharing one
    legend. Saves ``<out_stem>.pdf`` and ``.png``; returns the written paths.
    """
    apply_style()
    panels = [p for p in panels if any(p[1].values())]
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
        for h, lab in _plot_panel(ax, series, logy=logy):
            legend_pairs.setdefault(lab, h)
        if n > 1:
            ax.set_title(title)
        ax.set_xlabel(r"Number of sampler steps $M$")
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
    for j in range(n, len(flat)):
        flat[j].set_visible(False)

    # One shared legend below the whole figure, in SERIES order.
    labels = [s[2] for s in SERIES if s[2] in legend_pairs]
    handles = [legend_pairs[l] for l in labels]
    ncol_leg = 4 if len(labels) > 8 else max(2, len(labels))
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02),
        ncol=ncol_leg, frameon=False, handlelength=2.3, columnspacing=1.3,
        handletextpad=0.5, labelspacing=0.5,
    )
    fig.tight_layout(rect=(0, 0.06 + 0.03 * math.ceil(len(labels) / ncol_leg), 1, 1))

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        p = out_stem.with_suffix(f".{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=300 if ext == "png" else None)
        written.append(p)
    plt.close(fig)
    return written


# --------------------------------------------------------------------------- #
# The shared "metric vs assimilation step" plotter (per-step curves; legend below)
# --------------------------------------------------------------------------- #


def make_vs_step_figure(
    panels: list[tuple[str, dict]],
    ylabel: str,
    out_stem: Path,
    *,
    ncols: int | None = None,
    logy: bool = True,
) -> list[Path]:
    """Render a (multi-panel) metric-vs-assimilation-step figure, legend below.

    Mirrors :func:`make_vs_M_figure` but the x-axis is the assimilation-step index
    (linear), one line per method. ``panels`` is ``[(panel_title, series_dict)]``
    with ``series_dict`` mapping ``(method, variant) -> {step: value}`` (from
    :func:`load_metric_vs_step`). Saves ``<out_stem>.pdf`` and ``.png``.
    """
    apply_style()
    panels = [p for p in panels if any(p[1].values())]
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
            legend_pairs.setdefault(label, h)
        if logy:
            ax.set_yscale("log")
        ax.margins(x=0.03)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.15)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(which="both", direction="out", length=4, width=0.8)
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

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        p = out_stem.with_suffix(f".{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=300 if ext == "png" else None)
        written.append(p)
    plt.close(fig)
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


def load_state_records(case: str, scenario: str | None = None) -> list[dict]:
    """Load the trajectory-1 saved-state ``.npz`` archives for one case.

    Returns a list of dicts (one per method/variant cell), each carrying the
    archive's arrays plus a ``_path`` key. Filtered to ``scenario`` when given.
    Returns ``[]`` when the case's grid has not saved states yet.
    """
    sdir = RESULTS / case / "states" / "traj1"
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
    return recs


def _state_series_key(method: str, variant: str | None) -> tuple[int, str]:
    """Return ``(order_index, pretty_label)`` for a state record's method/variant.

    Orders rows by the shared ``SERIES`` list so the field maps match the
    metric-vs-M figures; unknown methods sort last under their raw name.
    """
    variant = variant or None
    for i, (m, v, label, *_rest) in enumerate(SERIES):
        if m == method and v == variant:
            return i, label
    return len(SERIES), method + (f" ({variant})" if variant else "")


def make_state_field_figure(
    records: list[dict],
    field_fn,
    out_stem: Path,
    *,
    cbar_label: str = "",
    cmap: str = "viridis",
    diverging: bool = False,
) -> list[Path]:
    """Render final-step field maps: rows = methods, cols = Truth | Mean | |Error| | Spread.

    ``field_fn`` maps a trajectory array ``[n, C, H, W, T]`` to a stack of 2D
    fields ``[n, H, W]`` at the final step (e.g. pick a channel or take velocity
    magnitude). Truth is shared across methods, so the Truth/Mean columns share
    one colour scale; ``|Error|`` and ``Spread`` each share their own scale so
    methods are directly comparable. Writes ``<out_stem>.pdf`` + ``.png``.
    """
    if not records:
        return []
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

    # Column colour scales. Truth/Mean share a scale from the truth field
    # (symmetric about 0 for signed / diverging fields); |Error| and Spread each
    # span 0..max over all methods so rows are comparable.
    if diverging:
        m = float(np.abs(truth2d).max()) or 1.0
        f_vmin, f_vmax = -m, m
    else:
        f_vmin, f_vmax = float(truth2d.min()), float(truth2d.max())
    err_max = max((float(r[2].max()) for r in rows), default=1.0) or 1.0
    spread_max = max((float(r[3].max()) for r in rows), default=1.0) or 1.0

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

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        p = out_stem.with_suffix(f".{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=200 if ext == "png" else None)
        written.append(p)
    plt.close(fig)
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


__all__ = [
    "STEPS", "SERIES", "SCENARIO_LABEL", "RESULTS", "FIGURES_DIR",
    "apply_style", "load_metric_vs_M", "make_vs_M_figure",
    "load_metric_vs_step", "make_vs_step_figure",
    "load_state_records", "make_state_field_figure", "mirror_to",
]
