"""Aggregate the Lorenz 96 per-cell metric files into one tidy file.

``cases/lorenz96/driver.py cell`` writes one file per (method, seed), already averaged
over the held-out sequences. This reduces them to mean +/- std over seeds, in
``results/lorenz96/aggregated/{all,per_step}.csv``. Seeds share the sequences, so that
std is sampler noise only; ``sequence_spread.csv`` holds the spread over sequences.

    .venv/bin/python paper_experiments/aggregate_lorenz96.py
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

RESULTS = _here / "results"
CASE = "lorenz96"
KEY_METRICS = (
    "mean_error_vs_pf",
    "var_error_vs_pf",
    "sliced_w2",
    "rmse",
    "spread",
    "crps",
    "diverged_fraction",
    "nfe",
    "seconds",
)


def sequence_spread(directory: Path) -> list[dict[str, object]]:
    """Std over sequences of the seed-averaged value, ignoring diverged entries."""
    cells: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(directory.glob("*.json")):
        cells[path.name.split("__")[1]].append(json.loads(path.read_text()))
    rows: list[dict[str, object]] = []
    for name, seeds in cells.items():
        for metric in seeds[0]:
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


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    root = RESULTS / CASE
    by_seed = discover_by_traj(root / "metrics")
    if not by_seed:
        print(f"[{CASE}] no per-cell metric files in {root / 'metrics'}")
        return
    rows = aggregate_scalar(by_seed)
    write_scalar_csv(root / "aggregated" / "all.csv", rows)
    write_per_step_csv(
        root / "aggregated" / "per_step.csv",
        aggregate_per_step(discover_by_traj(root / "per_step")),
    )
    print_scalar_table(rows, KEY_METRICS)
    spread = sequence_spread(root / "per_sequence")
    with open(root / "aggregated" / "sequence_spread.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, ["cell", "metric", "std", "n"])
        writer.writeheader()
        writer.writerows(spread)


if __name__ == "__main__":
    main()
