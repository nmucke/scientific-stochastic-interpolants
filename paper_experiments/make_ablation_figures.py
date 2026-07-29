"""Ablation figures + table source for the Navier--Stokes case (NS ONLY).

The ablation axis is the shared-covariance Jacobian refresh cadence $k$ at one
fixed sampler-step count (default M=100). Data comes from TWO trees:

* ``results/navier_stokes/ablation/metrics/`` -- the cadence cells run by
  ``run_ns_ablation.sh`` (variants ``shared_jac1`` / ``shared_jac5`` / any
  other ``shared_jac<k>`` present), kept OUT of the main tree on purpose (the
  paper pipeline folds every shared cadence onto one "shared" series).
* ``results/navier_stokes/aggregated/all.csv`` -- the MAIN grid's rows at the
  same M: ``shared_jac10`` supplies the k=10 point and ``jacfree`` the
  Jacobian-free reference lines. (Run ``aggregate_ns.py`` first.)

Outputs (into ``manuscript/figures/navier_stokes`` + the in-repo mirror):

* ``ns_ablation_<metric>_vs_k``  -- metric vs. cadence $k$, one panel per
  scenario, one line per Ours sampler (mean +/- std over trajectories), with
  the Jacobian-free value as a dashed reference line per sampler. Metrics:
  rmse, crps, spread_skill, kl_points, seconds (per-step cost).
* ``singles/ns_ablation_<metric>_vs_k_<scenario>`` -- print-size per-panel
  singles (no title/legend) + one shared ``singles/ns_ablation_legend``.

Table source: ``results/navier_stokes/ablation/aggregated.csv`` (tidy
mean +/- std over trajectories for the ablation-tree cells) plus a console
table of every (scenario, method, k) cell -- paste-ready for ``tab:ablation``.

    python paper_experiments/make_ablation_figures.py [--M 100]
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

from common.aggregate_lib import (  # noqa: E402
    aggregate_scalar,
    discover_by_traj,
    write_scalar_csv,
)
from figure_common import (  # noqa: E402
    FIGURES_DIR,
    RESULTS,
    SCENARIO_LABEL,
    SERIES,
    SINGLE_FIGSIZE,
    SINGLE_RC,
    SINGLE_SCALE,
    _save_fig,
    apply_style,
    mirror_figures,
)

CASE = "navier_stokes"
DEFAULT_OUT = _here.parent / "manuscript" / "figures" / CASE
ABLATION_ROOT = RESULTS / CASE / "ablation"
MAIN_AGG = RESULTS / CASE / "aggregated" / "all.csv"
ABLATION_M = 100  # the single sampler-step count of the cadence ablation
SCENARIOS = ("16^2->128^2", "32^2->128^2", "sparse 5%", "sparse 1.5625%")

# The three shared-covariance samplers, styled exactly as in the main figures
# (colour/marker from figure_common.SERIES so the ablation reads consistently).
METHODS = ("Ours (SI-SDE)", "Ours (DM-SDE)", "Ours (FM-ODE)")
_STYLE = {m: (c, mk) for m, v, _l, c, _ls, mk, _f in SERIES
          if v == "shared" and m in METHODS}
METHOD_LABEL = {
    "Ours (SI-SDE)": "SI-SDE", "Ours (DM-SDE)": "DM-SDE", "Ours (FM-ODE)": "FM-ODE",
}

# (metric key, y-axis label). ``seconds`` is the per-step cost row.
METRIC_FIGURES = (
    ("rmse", r"Vorticity RMSE"),
    ("crps", r"CRPS"),
    ("spread_skill", r"$|1-\mathrm{spread}/\mathrm{skill}|$"),
    ("kl_points", r"KL divergence (pointwise)"),
    ("seconds", r"Wall-clock cost [s/step]"),
)


def SLUG(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def _cadence(variant: str | None) -> int | None:
    """Cadence k from a variant tag: ``shared`` -> 1, ``shared_jac<k>`` -> k."""
    if not variant:
        return None
    if variant == "shared":
        return 1
    m = re.fullmatch(r"shared_jac(\d+)", variant)
    return int(m.group(1)) if m else None


def load_cells(M: int):
    """Merge ablation-tree + main-tree rows at sampler-step count ``M``.

    Returns ``(shared, ref, agg_rows)``:

    * ``shared[(scenario, metric, method)][k] = (mean, std)`` -- one entry per
      cadence, from the ablation tree (shared_jac1/5/...) and the main
      aggregate (shared_jac10). On a collision the ablation tree wins.
    * ``ref[(scenario, metric, method)] = mean`` -- the jacfree reference from
      the main aggregate.
    * ``agg_rows`` -- the ablation tree's aggregated tidy rows (mean +/- std
      over trajectories), for ``aggregated.csv``.
    """
    agg_rows = aggregate_scalar(discover_by_traj(ABLATION_ROOT / "metrics"))

    def _rows_from_main():
        if not MAIN_AGG.exists():
            print(f"[nsabl] {MAIN_AGG} missing -- run aggregate_ns.py for the "
                  "k=10 point and the jacfree reference")
            return []
        with open(MAIN_AGG, encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    shared: dict[tuple, dict[int, tuple[float, float]]] = defaultdict(dict)
    ref: dict[tuple, float] = {}

    def _f(row, key, default=float("nan")):
        v = row.get(key)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    # Main tree first, so ablation-tree entries overwrite on the same k.
    for row in _rows_from_main() + agg_rows:
        if row.get("case") != CASE or row.get("method") not in METHODS:
            continue
        m_row = _f(row, "M")
        if not (m_row == m_row) or int(m_row) != M:  # NaN-safe M filter
            continue
        metric, scen = str(row.get("metric")), str(row.get("scenario"))
        value, std = _f(row, "value"), _f(row, "std", 0.0)
        if not (value == value):
            continue
        variant = str(row.get("variant") or "")
        key = (scen, metric, str(row.get("method")))
        if variant == "jacfree":
            ref[key] = value
        else:
            k = _cadence(variant)
            if k is not None:
                shared[key][k] = (value, std)
    return shared, ref, agg_rows


def _panel(ax, shared, ref, scen: str, metric: str, *, scale: float = 1.0) -> list:
    """One metric-vs-cadence panel; returns (handle, label) legend pairs."""
    handles: list = []
    ks_all: set[int] = set()
    for method in METHODS:
        cells = shared.get((scen, metric, method), {})
        if not cells:
            continue
        colour, marker = _STYLE[method]
        ks = sorted(cells)
        ks_all.update(ks)
        means = [cells[k][0] for k in ks]
        stds = [cells[k][1] for k in ks]
        h = ax.errorbar(
            ks, means, yerr=stds, color=colour, marker=marker,
            markersize=6.5 * scale, markeredgewidth=1.3 * scale,
            markeredgecolor=colour, markerfacecolor=colour,
            linewidth=2.1 * scale, capsize=3.0 * scale,
            elinewidth=1.1 * scale, zorder=3,
        )
        handles.append((h, METHOD_LABEL[method]))
        r = ref.get((scen, metric, method))
        if r is not None:
            ax.axhline(r, color=colour, linestyle=(0, (4, 3)),
                       linewidth=1.3 * scale, alpha=0.75, zorder=2)
    if not handles:
        return []
    # One neutral proxy for the per-method dashed jacfree reference lines.
    if any(ref.get((scen, metric, m)) is not None for m in METHODS):
        from matplotlib.lines import Line2D

        proxy = Line2D([0], [0], color="0.35", linestyle=(0, (4, 3)),
                       linewidth=1.3 * scale)
        handles.append((proxy, "Jac-free (reference)"))
    ks_sorted = sorted(ks_all)
    ax.set_xscale("log")
    ax.set_xticks(ks_sorted)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(min(ks_sorted) * 0.8, max(ks_sorted) * 1.25)
    ax.grid(True, which="major", linestyle="-", linewidth=0.6 * scale, alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(which="both", direction="out", length=4 * scale,
                   width=0.8 * scale)
    return handles


def make_figures(shared, ref, out: Path) -> list[Path]:
    written: list[Path] = []
    legend_pairs: dict[str, object] = {}
    xlabel = r"Jacobian refresh cadence $k$"
    for metric, ylabel in METRIC_FIGURES:
        scens = [s for s in SCENARIOS
                 if any(shared.get((s, metric, m)) for m in METHODS)]
        if not scens:
            print(f"[nsabl] no data for {metric}; skipped")
            continue
        apply_style()
        ncols = min(2, len(scens))
        nrows = math.ceil(len(scens) / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.2 * ncols, 4.4 * nrows),
                                 squeeze=False)
        flat = axes.flatten()
        for i, scen in enumerate(scens):
            ax = flat[i]
            for h, lab in _panel(ax, shared, ref, scen, metric):
                legend_pairs.setdefault(lab, h)
            if len(scens) > 1:
                ax.set_title(SCENARIO_LABEL.get(scen, scen))
            ax.set_xlabel(xlabel)
            if i % ncols == 0:
                ax.set_ylabel(ylabel)
        for j in range(len(scens), len(flat)):
            flat[j].set_visible(False)
        labels = list(legend_pairs)
        fig.tight_layout()
        fig.legend([legend_pairs[l] for l in labels], labels,
                   loc="upper center", bbox_to_anchor=(0.5, 0.0),
                   ncol=min(4, len(labels)), frameon=False, fontsize=9)
        stem = out / f"ns_ablation_{metric}_vs_k"
        written += _save_fig(fig, stem)
        # Print-size singles (no title/legend), one per scenario panel.
        for scen in scens:
            with plt.rc_context(SINGLE_RC):
                fig1, ax1 = plt.subplots(figsize=SINGLE_FIGSIZE)
                _panel(ax1, shared, ref, scen, metric, scale=SINGLE_SCALE)
                ax1.set_xlabel(xlabel)
                ax1.set_ylabel(ylabel)
                written += _save_fig(
                    fig1, out / "singles" / f"{stem.name}_{SLUG(scen)}"
                )
    # One shared legend file for the singles.
    if legend_pairs:
        apply_style()
        fig = plt.figure()
        labels = list(legend_pairs)
        fig.legend([legend_pairs[l] for l in labels], labels, loc="center",
                   ncol=min(4, len(labels)), frameon=False, fontsize=8,
                   handlelength=1.9, columnspacing=1.2, handletextpad=0.5)
        written += _save_fig(fig, out / "singles" / "ns_ablation_legend")
    return written


def print_table(shared, ref, M: int) -> None:
    """Console table of every (scenario, method, k) cell -- tab:ablation source."""
    metrics = [m for m, _ in METRIC_FIGURES]
    header = f"{'scenario':<16} {'method':<8} {'k':>7} " + "".join(
        f"{m:>14}" for m in metrics
    )
    print(f"\n[nsabl] cadence ablation at M={M} "
          "(mean over trajectories; 'jacfree' = reference)")
    print(header)
    print("-" * len(header))
    for scen in SCENARIOS:
        rows_scen = False
        ks = sorted({k for m in METHODS
                     for k in shared.get((scen, "rmse", m), {})})
        for method in METHODS:
            for k in ks:
                cells = [shared.get((scen, met, method), {}).get(k)
                         for met in metrics]
                if not any(cells):
                    continue
                rows_scen = True
                vals = "".join(
                    f"{c[0]:>14.4g}" if c else f"{'--':>14}" for c in cells
                )
                print(f"{SLUG(scen):<16} {METHOD_LABEL[method]:<8} {k:>7d} {vals}")
            r = [ref.get((scen, met, method)) for met in metrics]
            if any(v is not None for v in r):
                vals = "".join(
                    f"{v:>14.4g}" if v is not None else f"{'--':>14}" for v in r
                )
                print(f"{SLUG(scen):<16} {METHOD_LABEL[method]:<8} {'jacfree':>7} {vals}")
        if rows_scen:
            print("-" * len(header))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--M", type=int, default=ABLATION_M,
                    help=f"sampler-step count of the ablation (default {ABLATION_M})")
    args = ap.parse_args()
    out = Path(args.out)

    shared, ref, agg_rows = load_cells(args.M)
    if agg_rows:
        agg_path = ABLATION_ROOT / "aggregated.csv"
        write_scalar_csv(agg_path, agg_rows)
        print(f"[nsabl] wrote {agg_path} ({len(agg_rows)} rows)")
    if not shared:
        print("[nsabl] no cadence cells found; run run_ns_ablation.sh first "
              f"(and aggregate_ns.py for the main-grid k=10/jacfree rows at M={args.M})")
        return

    written = make_figures(shared, ref, out)
    written += mirror_figures(written, FIGURES_DIR / CASE)
    print_table(shared, ref, args.M)
    for p in written:
        print(f"[fig] wrote {p}")


if __name__ == "__main__":
    main()
