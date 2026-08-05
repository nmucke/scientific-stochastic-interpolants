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
* ``tables_scaled.tex``       -- the SAME tables in per-metric units of $10^{-n}$
                                 (the factor is named once in each metric header);
                                 pick whichever notation reads better.

Every NS metric present flows through both files (``rmse``, ``energy_spec_rmse``,
``kl_points``, ``crps``, ``crps_observed``, ``crps_unobserved``,
``spread_skill``, plus ``nfe`` / ``seconds``).

One further input, produced separately because it is computed from saved states
rather than from the grid's per-cell CSVs: ``reference/metrics.csv``, the
large-ensemble (E=1000) EnKF reference, written by
``compute_ns_reference_metrics.py``. It is loaded as-is, reduced over trajectories
like everything else, and rendered as an out-of-competition table row.

    .venv/bin/python paper_experiments/compute_ns_reference_metrics.py   # once
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
    aggregate_scalar_records,
    discover_by_traj,
    load_records_by_traj,
    print_scalar_table,
    write_per_step_csv,
    write_scalar_csv,
)
from common.latex_tables import TableSpec, write_latex_tables  # noqa: E402
from common.timing_lib import apply_timing, describe, load_timing  # noqa: E402

RESULTS = _here / "results"
CASE = "navier_stokes"

# Clean per-DA-step wall-clock, measured on an idle GPU by measure_timing.py. The
# ``seconds`` the grid runs recorded is contaminated: many cells were timed while
# another CUDA job shared the card (measured 2026-07-30: one D-Flow cell went
# 160 s -> 302 s, 1.9x, from contention alone) and some fell back to CPU. When this
# file exists its numbers REPLACE the ``seconds`` column and the ``metric=seconds``
# rows; every other metric is untouched. Absent -> the grid's own seconds are kept
# and a warning is printed, so the tables never silently mix the two sources.
TIMING_CSV = RESULTS / "timing" / "seconds_per_step.csv"
KEY_METRICS = ("rmse", "energy_spec_rmse", "crps", "spread_skill", "kl_points")

# The large-ensemble reference EnKF (E=1000 vs the graded E=64 -- the same run that
# provides the KL reference distribution), as computed from its saved states by
# ``compute_ns_reference_metrics.py``. That script owns the metric computation and
# the method name; this one only LOADS its CSV and reduces it over trajectories
# like any other cell. Absent -> no reference row, with a pointer to the script.
REF_METRICS_CSV = RESULTS / CASE / "reference" / "metrics.csv"
REF_METHOD = "EnKF (reference)"

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
    # The energy-spectrum RMSE is still aggregated into all.csv (and printed by
    # print_scalar_table below), but is deliberately NOT a table column: the
    # enstrophy-spectrum figures in make_ns_figures.py carry that comparison.
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
    # The bare FlowDAS / SDA rows are dropped from the table: their SURGE-augmented
    # counterparts are what the paper compares against. Both still run and are
    # aggregated into all.csv.
    baselines=(
        ("SURGE (FlowDAS)", "FlowDAS + SURGE"),
        ("SURGE (SDA)", "SDA + SURGE"),
        ("D-Flow SGLD", "D-Flow SGLD"),
        ("Guided FM (FIG)", "Guided FM (FIG)"),
    ),
    classical=(
        ("EnKF", "EnKF"),
        ("Particle filter", "Particle filter"),
    ),
    reference=((REF_METHOD, "EnKF reference"),),
    metrics_phrase=(
        r"vorticity RMSE, CRPS, and spread--skill "
        r"($|1-\mathrm{spread}/\mathrm{skill}|$, $0=$ calibrated)"
    ),
    notes=(
        "The solver-free baselines share the prior; the conventional filters "
        "(EnKF, particle filter) use the true solver and have no sampler-step "
        "count, so their rows are the same run repeated in every $M$ table."
    ),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--std", action="store_true",
        help="print each table cell as mean +/- across-trajectory std",
    )
    ap.add_argument(
        "--timing-csv", type=Path, default=TIMING_CSV,
        help="clean per-DA-step timings from measure_timing.py (default: "
             "results/timing/seconds_per_step.csv)",
    )
    ap.add_argument(
        "--no-timing-override", action="store_true",
        help="keep the grid's own (contended) seconds instead of splicing in the "
             "clean measurements",
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

        # Replace the contended grid timings with the clean measurements BEFORE
        # anything is written, so all.csv and tables.tex can never disagree about
        # cost. Only `seconds` moves; every accuracy metric is left alone.
        if args.no_timing_override:
            print(f"[{CASE}] timing override DISABLED -- keeping the grid's own "
                  f"seconds (contended; see measure_timing.py)")
        else:
            timing = load_timing(args.timing_csv)
            if timing is None:
                print(f"[{CASE}] WARNING: no timing file at {args.timing_csv} -- "
                      f"keeping the grid's own seconds, which are CONTENDED and "
                      f"should not be quoted. Run measure_timing.py.")
            else:
                stats = apply_timing(rows, timing)
                print(f"[{CASE}] timings <- {args.timing_csv.name} "
                      f"({timing.n_rows} measured rows): {describe(stats)}")
                for cell in stats.get("_unmatched_preview", []) or []:
                    print(f"[{CASE}]   unmatched: {cell}")

        # (1a) The large-ensemble reference EnKF, appended AFTER the timing splice:
        # measure_timing.py never re-measured it, and the E-agnostic fallback in
        # apply_timing would otherwise hand it the 64-member EnKF's cost (~7 s
        # against its own ~100 s). It therefore keeps the seconds its own run
        # recorded -- contended like the rest of the grid's raw timings, so it is
        # an order-of-magnitude figure, not a clean measurement.
        if REF_METRICS_CSV.exists():
            ref_rows = aggregate_scalar_records(load_records_by_traj(REF_METRICS_CSV))
            rows.extend(ref_rows)
            Es = sorted({r["E"] for r in ref_rows if r.get("E") is not None})
            n_traj = max((int(r["n_traj"]) for r in ref_rows), default=0)
            print(f"[{CASE}] reference <- {REF_METRICS_CSV.name}: {len(ref_rows)} rows "
                  f"(E={Es}, {n_traj} trajectories); seconds are the run's own "
                  f"(not re-measured)")
        else:
            print(f"[{CASE}] no reference metrics at {REF_METRICS_CSV} -- no "
                  f"reference row. Build it with compute_ns_reference_metrics.py")

        out = root / "aggregated" / "all.csv"
        write_scalar_csv(out, rows)
        print(f"[{CASE}] wrote {len(rows)} scalar rows -> {out}")
        print_scalar_table(rows, KEY_METRICS)

        # (1b) LaTeX tables, one per M, from those same aggregated rows -- in two
        # interchangeable notations, so the manuscript can \input whichever reads
        # better without either drifting from the CSVs:
        #   tables.tex         plain fixed-point (0.034)
        #   tables_scaled.tex  per-column units of 10^-n named in the header (3.400)
        # Same numbers, same bolding, distinct \label suffix (_scaled).
        for tex, scaled in ((root / "tables.tex", False),
                            (root / "tables_scaled.tex", True)):
            Ms = write_latex_tables(
                TABLE_SPEC, rows, tex, with_std=args.std, scaled=scaled
            )
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
