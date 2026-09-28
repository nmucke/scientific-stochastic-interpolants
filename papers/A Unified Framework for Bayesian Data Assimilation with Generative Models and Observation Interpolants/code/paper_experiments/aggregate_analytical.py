"""Aggregate the analytical (Case 1) per-cell metric files into one tidy file.

The analytical grid (``run_analytical_grid.sh``) writes one seed-aggregated
per-M file ``results/analytical/metrics/analytical__M<M>.csv`` (each row already
carries the across-seed mean in ``value`` and std in ``std``; there are no test
trajectories and no time dimension). This script concatenates them into the
canonical tidy file ``results/analytical/aggregated/all.csv`` (schema +
``n_traj``), which ``make_analytical_figures.py`` (KL / sliced-W2 vs M) and
``make_tables.py`` consume. Every metric present (``kl_points``, ``sliced_w2``,
plus the ``nfe`` / ``seconds`` cost rows) flows through.

There is a single trajectory bucket, so ``n_traj = 1`` -- effectively a
concatenation, matching the previous ``aggregate_grid.py`` behaviour for this
case. Analytical has no per-assimilation-step curves, so no ``per_step.csv`` is
written.

    .venv/bin/python paper_experiments/aggregate_analytical.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from common.aggregate_lib import (  # noqa: E402
    aggregate_scalar,
    discover_by_traj,
    print_scalar_table,
    write_scalar_csv,
)

RESULTS = _here / "results"
CASE = "analytical"
KEY_METRICS = ("kl_points", "sliced_w2")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    metrics_dir = RESULTS / CASE / "metrics"
    by_traj = discover_by_traj(metrics_dir)
    if not by_traj:
        print(f"[{CASE}] no per-cell metric files in {metrics_dir}")
        return
    print(f"[{CASE}] buckets: {sorted(by_traj)} "
          f"({sum(len(v) for v in by_traj.values())} files)")

    rows = aggregate_scalar(by_traj)
    out = RESULTS / CASE / "aggregated" / "all.csv"
    write_scalar_csv(out, rows)
    print(f"[{CASE}] wrote {len(rows)} aggregated rows -> {out}")
    print_scalar_table(rows, KEY_METRICS)


if __name__ == "__main__":
    main()
