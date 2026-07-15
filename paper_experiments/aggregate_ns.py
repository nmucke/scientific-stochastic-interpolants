"""Aggregate the Navier--Stokes (Case 2) per-cell results over trajectories.

The NS grid (``run_ns_grid.sh``) writes, per ``(scenario, M, trajectory, group)``
cell, two tidy files:

* ``results/navier_stokes/metrics/<...>__traj<N>__<group>.csv`` -- the scalar
  metrics, each already averaged over assimilation steps IN-RUN (one value per
  trajectory).
* ``results/navier_stokes/per_step/<...>__traj<N>__<group>.csv`` -- the
  per-assimilation-step metric curves for that trajectory.

This produces the two aggregates the user asked for, both reduced ACROSS
trajectories (trajectory identity comes from the ``traj<N>`` filename token):

* ``aggregated/all.csv``      -- SCALAR metrics, mean +/- std over trajectories
                                 (metric already time-averaged). Canonical schema
                                 + ``n_traj``; consumed by ``make_ns_figures``
                                 (metric-vs-M) and ``make_tables``.
* ``aggregated/per_step.csv`` -- PER-STEP metrics, mean +/- std over trajectories
                                 at each assimilation step; consumed by
                                 ``make_ns_figures`` (metric-vs-step).
* ``tables.tex``              -- LaTeX results tables, ONE PER SAMPLER-STEP COUNT
                                 ``M`` (RMSE / CRPS / spread--skill x the four
                                 scenarios + per-step cost), built from the same
                                 aggregated rows so they can never drift from the
                                 CSVs. ``\\input`` straight into the manuscript.

Every NS metric present flows through both files (``rmse``, ``energy_spec_rmse``,
``kl_points``, ``crps``, ``crps_observed``, ``crps_unobserved``,
``spread_skill``, plus ``nfe`` / ``seconds``).

    .venv/bin/python paper_experiments/aggregate_ns.py
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
CASE = "navier_stokes"
KEY_METRICS = ("rmse", "energy_spec_rmse", "crps", "spread_skill", "kl_points")

# The LaTeX tables: metric groups x the four observation scenarios, + cost.
TABLE_SPEC = TableSpec(
    case=CASE,
    case_label="Navier--Stokes",
    label_stem="tab:ns_accuracy",
    source="paper_experiments/aggregate_ns.py",
    scenarios=(
        ("16^2->128^2", r"$16^2$"),
        ("32^2->128^2", r"$32^2$"),
        ("sparse 5%", r"$5\%$"),
        ("sparse 1.5625%", r"$\tfrac{1}{64}$"),
    ),
    metric_groups=(
        ("rmse", "Vorticity RMSE"),
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
    classical=(
        ("EnKF", "EnKF"),
        ("Particle filter", "Particle filter"),
    ),
    metrics_phrase=(
        r"vorticity RMSE, CRPS, and spread--skill "
        r"($|1-\mathrm{spread}/\mathrm{skill}|$, $0=$ calibrated)"
    ),
    notes=(
        "The solver-free baselines share the prior; the conventional filters "
        "(EnKF, particle filter) use the true solver."
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
