"""Aggregate the Lorenz 96 per-cell metric files into one tidy file.

``cases/lorenz96/driver.py --obs <obs> cell`` writes one file per (cell, M, seed),
already averaged over the held-out sequences. This reduces them to mean +/- std over
seeds, in ``results/lorenz96/<obs>/aggregated/{all,per_step}.csv``, separately for
each observation sweep (``--obs linear|arctan``, repeatable; default every sweep,
and a sweep without files is skipped). A file whose tidy scenario belongs to the
other sweep is refused (``lorenz96_lineup.check_scenario``). Rows are grouped by the
tidy ``M`` column too, so the M sweep of a generative cell
(``lorenz96__<cell>__M<M>__traj<seed>__.csv``) gives one row per M; in
``sequence_spread.csv`` such a cell is ``<cell>__M<M>``. Seeds share the sequences, so that
std is sampler noise only; ``sequence_spread.csv`` holds the spread over sequences.
A tagged cell (``driver.py cell --tag``) is its own row: its variant is
``<variant>_<tag>`` and its files ``lorenz96__<method>-<tag>__traj<seed>__.csv``.

    .venv/bin/python paper_experiments/aggregate_lorenz96.py [--obs linear|arctan]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from common.aggregate_lib import (  # noqa: E402
    aggregate_per_step,
    aggregate_scalar,
    discover_by_traj,
    print_scalar_table,
    write_per_step_csv,
    write_scalar_csv,
)
from lorenz96_lineup import (  # noqa: E402
    CASE,
    OBS_CHOICES,
    ENERGY_DISTANCE,
    KL_KNN,
    SMC_METRICS,
    check_scenario,
    sweep_root,
)

RESULTS = _here / "results"
KEY_METRICS = (
    "mean_error_vs_pf",
    "var_error_vs_pf",
    "sliced_w2",
    KL_KNN,
    ENERGY_DISTANCE,
    "rmse",
    "spread",
    "crps",
    "diverged_fraction",
    "diverged_steps",
    "innovation_lag1",
    "nfe",
    "seconds",
    *SMC_METRICS,
)


def sequence_spread(directory: Path) -> list[dict[str, object]]:
    """Std over sequences of the seed-averaged value, ignoring diverged entries."""
    cells: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(directory.glob("*.json")):
        # ``lorenz96__<cell>[__M<M>]__traj<seed>__.json``: the cell is ``<cell>`` or
        # ``<cell>__M<M>``, so the step counts of one cell stay apart.
        cell = path.name.removeprefix(f"{CASE}__").split("__traj")[0]
        cells[cell].append(json.loads(path.read_text()))
    rows: list[dict[str, object]] = []
    for name, seeds in cells.items():
        # A resumable grid can mix files written before and after a metric was
        # added (e.g. the innovation diagnostics): keep the metrics every seed has.
        shared = [m for m in seeds[0] if all(m in seed for seed in seeds)]
        dropped = sorted(set().union(*seeds) - set(shared))
        if dropped:
            print(
                f"[{CASE}] warning: {name}: not every seed has {dropped}; "
                "dropped from sequence_spread (re-run those seeds to include them)"
            )
        for metric in shared:
            means = [
                statistics.fmean(kept)
                for kept in (
                    [v for v in values if math.isfinite(v)]
                    for values in zip(*(seed[metric] for seed in seeds))
                )
                if kept
            ]
            spread = statistics.stdev(means) if len(means) > 1 else float("nan")
            rows.append(dict(cell=name, metric=metric, std=spread, n=len(means)))
    return rows


def undefined_std_for_one_value(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """NaN instead of the shared library's 0.0 for a std over a single value.

    The ``pf_reference`` cell runs for one seed only, and a std of 0 would read
    as "no seed noise" rather than "not measured".
    """
    for row in rows:
        if row["n_traj"] == 1:
            row["std"] = math.nan
    return rows


def aggregate_sweep(obs: str) -> bool:
    """Aggregate one sweep's root; False if it holds no metric files."""
    root = sweep_root(RESULTS, obs)
    by_seed = discover_by_traj(root / "metrics")
    if not by_seed:
        print(f"[{CASE}/{obs}] no per-cell metric files in {root / 'metrics'}")
        return False
    per_step = discover_by_traj(root / "per_step")
    for paths in (*by_seed.values(), *per_step.values()):
        check_scenario(paths, obs)
    rows = undefined_std_for_one_value(aggregate_scalar(by_seed))
    write_scalar_csv(root / "aggregated" / "all.csv", rows)
    write_per_step_csv(
        root / "aggregated" / "per_step.csv",
        undefined_std_for_one_value(aggregate_per_step(per_step)),
    )
    print(f"\n##### {CASE} sweep: {obs} #####")
    # One printed column per M (the reduction itself already groups by E and M).
    printed = [
        dict(row, scenario=f"{row['scenario']} M={row['M']}" if row.get("M") not in (None, "") else row["scenario"])
        for row in rows
    ]
    print_scalar_table(printed, KEY_METRICS)
    spread = sequence_spread(root / "per_sequence")
    with open(root / "aggregated" / "sequence_spread.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, ["cell", "metric", "std", "n"])
        writer.writeheader()
        writer.writerows(spread)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--obs",
        choices=OBS_CHOICES,
        action="append",
        help="observation sweep (repeatable); default: every sweep",
    )
    args = parser.parse_args()
    for obs in args.obs or OBS_CHOICES:
        aggregate_sweep(obs)


if __name__ == "__main__":
    main()
