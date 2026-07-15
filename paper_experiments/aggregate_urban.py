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
* ``tables.tex``              -- LaTeX results tables, ONE PER SAMPLER-STEP COUNT
                                 ``M``, built from the same aggregated rows. The
                                 columns are the metrics urban actually computes
                                 (velocity RMSE, temperature RMSE, CRPS,
                                 spread--skill) x its two sparse scenarios, plus
                                 per-step cost.

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
from common.latex_tables import TableSpec, write_latex_tables  # noqa: E402

RESULTS = _here / "results"
CASE = "urban"
KEY_METRICS = ("rmse_velocity", "rmse_temperature", "crps", "spread_skill")

# The LaTeX tables. Urban runs only the two sparse scenarios (run_urban_grid.sh)
# and has no true solver, hence no classical filters -- so the row lineup is our
# samplers + the solver-free baselines only. Columns are the metrics urban
# actually computes: RMSE is per-variable (velocity, temperature).
TABLE_SPEC = TableSpec(
    case=CASE,
    case_label="Urban (uDALES)",
    label_stem="tab:urban_accuracy",
    source="paper_experiments/aggregate_urban.py",
    scenarios=(
        ("sparse 5%", r"$5\%$"),
        ("sparse 1.5625%", r"$\tfrac{1}{64}$"),
    ),
    metric_groups=(
        ("rmse_velocity", "Velocity RMSE"),
        ("rmse_temperature", "Temperature RMSE"),
        ("crps", "CRPS"),
        ("spread_skill", "Spread--skill"),
    ),
    ours=(
        ("Ours (SI-SDE)", "Ours (SI-SDE)"),
        ("Ours (DM-SDE)", "Ours (DM-SDE)"),
        ("Ours (FM-ODE)", "Ours (FM-ODE)"),
    ),
    baselines=(
        ("FlowDAS", "FlowDAS"),
        ("SURGE (FlowDAS)", "FlowDAS + SURGE"),
        ("SDA", "SDA"),
        ("SURGE (SDA)", "SDA + SURGE"),
        ("D-Flow SGLD", "D-Flow SGLD"),
        ("Guided FM (FIG)", "Guided FM (FIG)"),
    ),
    metrics_phrase=(
        r"velocity RMSE, temperature RMSE, CRPS, and spread--skill "
        r"($|1-\mathrm{spread}/\mathrm{skill}|$, $0=$ calibrated), "
        "all over fluid cells"
    ),
    notes=(
        "uDALES has no differentiable solver, so the lineup is solver-free "
        "throughout: no EnKF / particle-filter reference."
    ),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--std", action="store_true",
        help="print each table cell as mean +/- across-trajectory std",
    )
    args = ap.parse_args()
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

        # (1b) LaTeX tables, one per M, from those same aggregated rows.
        tex = root / "tables.tex"
        Ms = write_latex_tables(TABLE_SPEC, rows, tex, with_std=args.std)
        if Ms:
            print(f"[{CASE}] wrote {len(Ms)} LaTeX tables (M={Ms}) -> {tex}")
        else:
            print(f"[{CASE}] no rows carry an M; no LaTeX tables written")

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
