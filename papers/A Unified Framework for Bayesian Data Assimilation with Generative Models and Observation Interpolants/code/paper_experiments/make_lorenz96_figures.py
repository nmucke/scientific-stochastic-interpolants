"""Produce the figures of the Lorenz 96 case, per observation sweep.

* ``lorenz96_<metric>_vs_step`` -- each metric against the assimilation step, from
  ``results/lorenz96/<obs>/aggregated/per_step.csv``; the ``smc_*`` figures hold the
  SMC cells only (the others do not write those metrics). Generative cells are drawn
  at M=50 (``lorenz96_lineup.HEADLINE_M``).
* ``lorenz96_<metric>_vs_M``    -- the cycle-averaged metric (``all.csv``, mean +/- std
  over seeds) against M on a log axis, one line per generative cell, with the
  PF E-draw floor, EnKF and the oracle as horizontal lines (``VS_M_METRICS``).
* ``lorenz96_ess``              -- effective sample size of the reference over time.

Figures go to ``figures/lorenz96/<obs>/`` (``--out`` replaces that directory for a
single sweep). Only the default variants in ``SERIES`` are drawn; tagged sweep cells
(``driver.py cell --tag``, variant ``<variant>_<tag>``) are in the aggregated CSVs
but not in these figures.

    python paper_experiments/make_lorenz96_figures.py [--obs linear|arctan]
    python paper_experiments/make_lorenz96_figures.py --obs linear --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import math
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
from lorenz96_lineup import (  # noqa: E402
    CASE,
    ENERGY_DISTANCE,
    KL_KNN,
    M_REFERENCE_LINES,
    M_VALUES,
    OBS_CHOICES,
    ROW_OF,
    ROWS,
    SMC_ESS_FRACTION,
    SMC_ESS_MIN_FRACTION,
    SMC_RESAMPLES,
    as_m,
    norm_variant,
    pick_m,
    sweep_root,
)

ROW_BY_CELL = {row.cell: row for row in ROWS}


# Legend labels of the cells that were in the figures before the full lineup.
LEGEND = {
    "si_sde": "Ours SI-SDE",
    "dm_sde": "Ours DM-SDE",
    "fm_ode": "Ours FM-ODE",
    "si_sde_fc": "Ours SI-SDE + correction",
    "dm_sde_fc": "Ours DM-SDE + correction",
    "fm_ode_fc": "Ours FM-ODE + correction",
    "oracle": "One-step oracle",
    "oracle_fc": "Oracle + correction",
    "prior_enkf": "SI prior + EnKF",
    "prior_enkf_fm": "FM prior + EnKF",
    "enkf": "EnKF (matched E)",
    "enkf_large": "EnKF (large E)",
    "pf_matched": "PF (matched E)",
    "pf_subsample": "PF reference, E draws",
    "pf_reference": "PF reference",
    "surge_si_sde": "Ours SI-SDE + SURGE",
    "surge_dm_sde": "Ours DM-SDE + SURGE",
    "surge_unguided": "SURGE, unguided",
}


def _series_label(row) -> str:
    return LEGEND.get(row.cell, row.method)


# (method, variant, label, colour, linestyle), from lorenz96_lineup.ROWS.
SERIES = tuple(
    (row.method, row.variant, _series_label(row), row.colour, row.style) for row in ROWS
)
METRICS = {
    "mean_error_vs_pf": "mean error vs PF (post. std)",
    "var_error_vs_pf": "log-variance error vs PF",
    "sliced_w2": "sliced $W_2$ to PF",
    KL_KNN: "k-NN KL to PF",
    ENERGY_DISTANCE: "energy distance to PF",
    "rmse": "RMSE vs truth",
    "spread": "spread",
    "crps": "CRPS",
    "diverged_fraction": "diverged members",
    SMC_ESS_FRACTION: "ESS / particles (SMC)",
    SMC_ESS_MIN_FRACTION: "min ESS / particles (SMC)",
    SMC_RESAMPLES: "resampling events (SMC)",
}
# Metrics of the metric-vs-M figures (the cycle-averaged values of all.csv).
VS_M_METRICS = (KL_KNN, ENERGY_DISTANCE, "sliced_w2", "mean_error_vs_pf")


def load_curves(metric: str, obs: str = "linear") -> dict[tuple[str, str], dict[int, float]]:
    """Per-step curves at the headline M (generative rows) or the one M recorded."""
    by_m: dict[tuple[str, str], dict] = defaultdict(lambda: defaultdict(dict))
    path = sweep_root(RESULTS, obs) / "aggregated" / "per_step.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["metric"] == metric:
                key = (row["method"], norm_variant(row.get("variant")))
                by_m[key][as_m(row.get("M"))][int(float(row["step"]))] = float(row["value"])
    curves = {}
    for key, entries in by_m.items():
        row = ROW_OF.get(key)
        if row is None:  # not drawn; kept for the renamed-method check
            curves[key] = next(iter(entries.values()))
        elif (curve := pick_m(entries, row)) is not None:
            curves[key] = curve
    return curves


def load_vs_m(metric: str, obs: str = "linear") -> dict[tuple[str, str], dict]:
    """``{(method, variant): {M: (mean, std)}}`` of the cycle-averaged scalar."""
    out: dict[tuple[str, str], dict] = defaultdict(dict)
    path = sweep_root(RESULTS, obs) / "aggregated" / "all.csv"
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["metric"] == metric:
                key = (row["method"], norm_variant(row.get("variant")))
                out[key][as_m(row.get("M"))] = (float(row["value"]), float(row["std"] or "nan"))
    return out


def plot_vs_m(metric: str, label: str, out: Path, obs: str = "linear") -> None:
    """Metric against M, one line per generative cell, M-independent rows as lines."""
    data = load_vs_m(metric, obs)
    lines = [r for r in ROWS if r.generative and any(m in M_VALUES for m in data.get((r.method, r.variant), {}))]
    if not lines:
        print(f"[{obs}] no {metric} values at several M, skipped")
        return
    fig, ax = plt.subplots(figsize=(2 * SINGLE_FIGSIZE[0], 2 * SINGLE_FIGSIZE[1]))
    for row in lines:
        entries = data[(row.method, row.variant)]
        ms = [m for m in M_VALUES if m in entries]
        means = [entries[m][0] for m in ms]
        stds = [entries[m][1] if math.isfinite(entries[m][1]) else 0.0 for m in ms]
        ax.errorbar(ms, means, yerr=stds, color=row.colour, ls=row.style, marker="o",
                    ms=3, capsize=2, label=_series_label(row))
    for cell in M_REFERENCE_LINES:
        row = ROW_BY_CELL[cell]
        entry = pick_m(data.get((row.method, row.variant), {}), row)
        if entry is not None and math.isfinite(entry[0]):
            ax.axhline(entry[0], color=row.colour, ls=row.style, lw=0.8, label=_series_label(row))
    ax.set_xscale("log")
    ax.set_xticks(list(M_VALUES))
    ax.set_xticklabels([str(m) for m in M_VALUES])
    ax.minorticks_off()
    ax.set_xlabel("steps per assimilation step $M$")
    ax.set_ylabel(label)
    ax.legend(fontsize=6, ncol=2, frameon=False)
    _save_fig(fig, out / f"{CASE}_{metric}_vs_M")
    plt.close(fig)


def plot_metric(metric: str, label: str, out: Path, obs: str = "linear") -> None:
    curves = load_curves(metric, obs)
    if not curves:
        print(f"[{obs}] no {metric} curves, skipped")
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


def plot_ess(out: Path, obs: str = "linear") -> None:
    runs = sorted((sweep_root(RESULTS, obs) / "reference" / "test").glob("*_replicate0.pt"))
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


def make_sweep(obs: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for metric, label in METRICS.items():
        plot_metric(metric, label, out, obs)
    for metric in VS_M_METRICS:
        plot_vs_m(metric, METRICS[metric], out, obs)
    plot_ess(out, obs)
    print(f"[{obs}] wrote figures to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--obs",
        choices=OBS_CHOICES,
        action="append",
        help="observation sweep (repeatable); default: every sweep",
    )
    parser.add_argument("--out", type=Path, help="output directory (one sweep only)")
    args = parser.parse_args()
    sweeps = args.obs or list(OBS_CHOICES)
    if args.out is not None and len(sweeps) != 1:
        parser.error("--out needs exactly one --obs, or the sweeps would overwrite each other")
    apply_style()
    for obs in sweeps:
        make_sweep(obs, args.out or FIGURES_DIR / CASE / obs)


if __name__ == "__main__":
    main()
