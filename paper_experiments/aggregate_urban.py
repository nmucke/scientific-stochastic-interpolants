"""Aggregate the urban (Case 3, uDALES) per-cell results over trajectories.

The urban grid (``run_urban_grid.sh``) writes, per ``(scenario, M, trajectory,
group)`` cell, two tidy files:

* ``results/urban/metrics/<...>__traj<N>__<group>.csv`` -- the scalar metrics,
  each already averaged over assimilation steps IN-RUN (one value per trajectory).
* ``results/urban/per_step/<...>__traj<N>__<group>.csv`` -- the
  per-assimilation-step metric curves for that trajectory.

This produces the two aggregates the user asked for, both reduced ACROSS
trajectories (trajectory identity comes from the ``traj<N>`` filename token):

* ``aggregated/all.csv``      -- SCALAR metrics, mean +/- std over trajectories
                                 (metric already time-averaged). Canonical schema
                                 + ``n_traj``; consumed by ``make_urban_figures``
                                 (metric-vs-M) and ``make_tables``.
* ``aggregated/per_step.csv`` -- PER-STEP metrics, mean +/- std over trajectories
                                 at each assimilation step; consumed by
                                 ``make_urban_figures`` (metric-vs-step).

Urban is multi-variable, so the metric set differs from NS: every urban metric
present flows through both files (``rmse_velocity``, ``rmse_temperature``,
``crps``, ``crps_observed``, ``crps_unobserved``, ``spread_skill``, plus ``nfe``
/ ``seconds``).

    .venv/bin/python paper_experiments/aggregate_urban.py
"""

from __future__ import annotations

import argparse
import sys
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
CASE = "urban"
KEY_METRICS = ("rmse_velocity", "rmse_temperature", "crps", "spread_skill")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    root = RESULTS / CASE

    # (1) SCALAR metrics, aggregated over trajectories (time already averaged).
    scalar_by_traj = discover_by_traj(root / "metrics")
    if not scalar_by_traj:
        print(f"[{CASE}] no per-cell metric files in {root / 'metrics'}")
    else:
        print(f"[{CASE}] scalar trajectories: {sorted(scalar_by_traj)} "
              f"({sum(len(v) for v in scalar_by_traj.values())} files)")
        rows = aggregate_scalar(scalar_by_traj)
        out = root / "aggregated" / "all.csv"
        write_scalar_csv(out, rows)
        print(f"[{CASE}] wrote {len(rows)} scalar rows -> {out}")
        print_scalar_table(rows, KEY_METRICS)

    # (2) PER-STEP curves, aggregated over trajectories at each step.
    ps_by_traj = discover_by_traj(root / "per_step")
    if not ps_by_traj:
        print(f"[{CASE}] no per-step files in {root / 'per_step'} (skipping per_step.csv)")
    else:
        print(f"[{CASE}] per-step trajectories: {sorted(ps_by_traj)} "
              f"({sum(len(v) for v in ps_by_traj.values())} files)")
        ps_rows = aggregate_per_step(ps_by_traj)
        ps_out = root / "aggregated" / "per_step.csv"
        write_per_step_csv(ps_out, ps_rows)
        print(f"[{CASE}] wrote {len(ps_rows)} per-step rows -> {ps_out}")


if __name__ == "__main__":
    main()
