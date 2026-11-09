"""Produce the figures of the Lorenz 96 case.

* ``lorenz96_<metric>_vs_step`` -- each metric against the assimilation step, from
  ``results/lorenz96/aggregated/per_step.csv``.
* ``lorenz96_ess``              -- effective sample size of the reference over time.

Only the default variants in ``SERIES`` are drawn; tagged sweep cells
(``driver.py cell --tag``, variant ``<variant>_<tag>``) are in the aggregated CSVs
but not in these figures.

    python paper_experiments/make_lorenz96_figures.py
    python paper_experiments/make_lorenz96_figures.py --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from figure_common import (  # noqa: E402
    FIGURES_DIR,
    RESULTS,
    SINGLE_FIGSIZE,
    _save_fig,
    apply_style,
)

CASE = "lorenz96"
# (method, variant, label, colour, linestyle); colours follow figure_common.
SERIES = (
    ("Ours (SI-SDE)", "", "Ours SI-SDE", "#0072B2", "-"),
    ("Ours (DM-SDE)", "", "Ours DM-SDE", "#009E73", "-"),
    ("Ours (FM-ODE)", "", "Ours FM-ODE", "#CC79A7", "-"),
    ("Ours (SI-SDE)", "forecast_correction", "Ours SI-SDE + correction", "#0072B2", "--"),
    ("Ours (DM-SDE)", "forecast_correction", "Ours DM-SDE + correction", "#009E73", "--"),
    ("Ours (FM-ODE)", "forecast_correction", "Ours FM-ODE + correction", "#CC79A7", "--"),
    ("One-step oracle", "", "One-step oracle", "#D55E00", "--"),
    ("One-step oracle", "forecast_correction", "Oracle + correction", "#D55E00", "-"),
    ("EnKF", "trained_prior_si", "SI prior + EnKF", "#56B4E9", ":"),
    ("EnKF", "trained_prior_fm", "FM prior + EnKF", "#56B4E9", "-."),
    ("EnKF", "matched", "EnKF (matched E)", "#000000", ":"),
    ("EnKF", "large", "EnKF (large E)", "#000000", "-."),
    ("Particle filter", "matched", "PF (matched E)", "#E69F00", ":"),
    ("Particle filter", "reference_subsample", "PF reference, E draws", "#666666", "--"),
    ("Particle filter", "reference", "PF reference", "#666666", "-"),
)
METRICS = {
    "mean_error_vs_pf": "mean error vs PF (post. std)",
    "var_error_vs_pf": "log-variance error vs PF",
    "sliced_w2": "sliced $W_2$ to PF",
    "rmse": "RMSE vs truth",
    "spread": "spread",
    "crps": "CRPS",
    "diverged_fraction": "diverged members",
}


def load_curves(metric: str) -> dict[tuple[str, str], dict[int, float]]:
    curves: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    path = RESULTS / CASE / "aggregated" / "per_step.csv"
    if not path.exists():
        return curves
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["metric"] == metric:
                key = (row["method"], row.get("variant") or "")
                curves[key][int(float(row["step"]))] = float(row["value"])
    return curves


def plot_metric(metric: str, label: str, out: Path) -> None:
    curves = load_curves(metric)
    if not curves:
        print(f"no {metric} curves, skipped")
        return
    known = {(method, variant) for method, variant, *_ in SERIES}
    if not known & set(curves):
        # Data present but no label matches: a renamed method would otherwise
        # give an empty figure without any warning.
        raise ValueError(f"No SERIES entry matches the {metric} curves {sorted(curves)}.")
    fig, ax = plt.subplots(figsize=(2 * SINGLE_FIGSIZE[0], 2 * SINGLE_FIGSIZE[1]))
    for method, variant, name, colour, style in SERIES:
        curve = curves.get((method, variant))
        if curve:
            steps = sorted(curve)
            ax.plot(
                steps, [curve[s] for s in steps], color=colour, ls=style, label=name
            )
    ax.set_xlabel("assimilation step")
    ax.set_ylabel(label)
    ax.legend(fontsize=6, ncol=2, frameon=False)
    _save_fig(fig, out / f"{CASE}_{metric}_vs_step")
    plt.close(fig)


def plot_ess(out: Path) -> None:
    runs = sorted((RESULTS / CASE / "reference" / "test").glob("*_replicate0.pt"))
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(2 * SINGLE_FIGSIZE[0], 2 * SINGLE_FIGSIZE[1]))
    for path in runs:
        ref = torch.load(path, weights_only=False)
        ax.plot(ref["ess"].numpy(), color="#666666", lw=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("assimilation step")
    ax.set_ylabel("effective sample size")
    _save_fig(fig, out / f"{CASE}_ess")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIGURES_DIR / CASE)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    apply_style()
    for metric, label in METRICS.items():
        plot_metric(metric, label, args.out)
    plot_ess(args.out)
    print(f"wrote figures to {args.out}")


if __name__ == "__main__":
    main()
