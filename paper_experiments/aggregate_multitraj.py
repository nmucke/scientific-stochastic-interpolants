"""Aggregate per-trajectory NS results into mean +/- std over trajectories.

The multi-trajectory NS study runs the case over several test trajectories
(``+test_index=1..5``), one seed each. For each trajectory N a separate setup
writes three tidy per-trajectory CSVs::

    paper_experiments/results/multitraj/csv/{gt,gen,conv}_traj<N>.csv

Each holds that ONE trajectory's metrics (the ``seed = SEED_AGGREGATED`` rows
carry the trajectory's value; columns are the canonical
``case, method, scenario, metric, value, std, E, M, seed, NFE, seconds`` schema
of :mod:`results_schema`).

This script globs all of them (tolerating missing trajectories), and for every
``(case, method, scenario, metric)`` cell collects the per-trajectory ``value``s
and reduces them to the mean and std *across trajectories*. It writes a tidy CSV
``multitraj/aggregated/all_trajectories.csv`` with the same schema where
``value`` = mean-over-trajectories, ``std`` = std-over-trajectories, plus an
extra column ``n_traj`` = number of trajectories averaged. It is NaN-safe
(nanmean / nanstd semantics; all-NaN groups are skipped).

Usage::

    .venv/bin/python paper_experiments/aggregate_multitraj.py
    .venv/bin/python paper_experiments/aggregate_multitraj.py \
        --csv-dir paper_experiments/results/multitraj/csv \
        --out paper_experiments/results/multitraj/aggregated/all_trajectories.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from results_schema import SEED_AGGREGATED, ResultRecord, load_records

# Per-trajectory CSVs are named {gt,gen,conv}_traj<N>.csv.
_TRAJ_RE = re.compile(r"^(gt|gen|conv)_traj(\d+)\.csv$")

# Output column order: the canonical schema + the extra n_traj column.
OUT_FIELDNAMES = (*ResultRecord.FIELDNAMES, "n_traj")

# Metrics shown in the compact human-readable table.
KEY_METRICS = ("rmse", "crps", "spread_skill", "kl_points")


def discover(csv_dir: Path) -> dict[int, list[Path]]:
    """Map trajectory index -> list of its per-trajectory CSV paths."""
    by_traj: dict[int, list[Path]] = defaultdict(list)
    for p in sorted(csv_dir.glob("*.csv")):
        m = _TRAJ_RE.match(p.name)
        if m:
            by_traj[int(m.group(2))].append(p)
    return dict(sorted(by_traj.items()))


def _per_traj_value(records: list[ResultRecord]) -> float:
    """The single per-trajectory value for one cell.

    A per-trajectory CSV holds that trajectory's value in the
    ``seed == SEED_AGGREGATED`` (-1) rows. Prefer those; fall back to the mean
    of any present rows so a differently-seeded file still aggregates.
    """
    agg = [r.value for r in records if r.seed == SEED_AGGREGATED]
    vals = agg if agg else [r.value for r in records]
    finite = [v for v in vals if v is not None and not math.isnan(v)]
    if not finite:
        return float("nan")
    return statistics.fmean(finite)


def _mean_opt(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None and not math.isnan(v)]
    return statistics.fmean(present) if present else None


def aggregate(by_traj: dict[int, list[Path]]) -> list[dict[str, object]]:
    """Reduce per-trajectory cells to mean +/- std over trajectories.

    Grouping key is (case, method, scenario, metric, E, M). For each group we
    collect ONE value per trajectory, then take nanmean / nanstd across them.
    All-NaN groups are skipped.
    """
    # (key) -> {traj_index -> list[ResultRecord]}
    cells: dict[tuple, dict[int, list[ResultRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for traj, paths in by_traj.items():
        for path in paths:
            for r in load_records(path):
                key = (r.case, r.method, r.scenario, r.metric, r.E, r.M, r.variant)
                cells[key][traj].append(r)

    rows: list[dict[str, object]] = []
    for (case, method, scenario, metric, E, M, variant), per_traj in cells.items():
        traj_vals = [_per_traj_value(recs) for recs in per_traj.values()]
        finite = [v for v in traj_vals if not math.isnan(v)]
        if not finite:  # skip all-NaN groups
            continue
        mean = statistics.fmean(finite)
        std = statistics.stdev(finite) if len(finite) > 1 else 0.0
        all_recs = [r for recs in per_traj.values() for r in recs]
        rec = ResultRecord(
            case=case,
            method=method,
            scenario=scenario,
            metric=metric,
            value=mean,
            std=std,
            E=E,
            M=M,
            seed=SEED_AGGREGATED,
            nfe=_mean_opt([r.nfe for r in all_recs]),
            seconds=_mean_opt([r.seconds for r in all_recs]),
            variant=variant,
        )
        row = rec.to_row()
        row["n_traj"] = len(finite)
        rows.append(row)
    return rows


def write_csv(out: Path, rows: list[dict[str, object]]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(OUT_FIELDNAMES))
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})


def print_table(rows: list[dict[str, object]]) -> None:
    """Compact method x scenario table for the key metrics."""
    # (method, scenario, metric) -> (value, std, n_traj)
    look = {
        (r["method"], r["scenario"], r["metric"]): (r["value"], r["std"], r["n_traj"])
        for r in rows
    }
    methods = sorted({r["method"] for r in rows})
    scenarios = sorted({r["scenario"] for r in rows})
    for metric in KEY_METRICS:
        present = [
            (m, s) for m in methods for s in scenarios if (m, s, metric) in look
        ]
        if not present:
            continue
        scen_here = [s for s in scenarios if any(s2 == s for (_, s2) in present)]
        print(f"\n=== {metric} (mean +/- std over trajectories) ===")
        header = f"{'method':<22}" + "".join(f"{s:>22}" for s in scen_here)
        print(header)
        for m in methods:
            if not any((m, s, metric) in look for s in scen_here):
                continue
            cells = []
            for s in scen_here:
                if (m, s, metric) in look:
                    v, sd, n = look[(m, s, metric)]
                    cells.append(f"{v:.4f}+/-{sd:.4f}(n{n})")
                else:
                    cells.append("-")
            print(f"{m:<22}" + "".join(f"{c:>22}" for c in cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv-dir",
        default="paper_experiments/results/multitraj/csv",
        help="dir holding {gt,gen,conv}_traj<N>.csv",
    )
    ap.add_argument(
        "--out",
        default="paper_experiments/results/multitraj/aggregated/all_trajectories.csv",
    )
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    by_traj = discover(csv_dir)
    if not by_traj:
        print(f"No per-trajectory CSVs found in {csv_dir} "
              "(expected {{gt,gen,conv}}_traj<N>.csv).")
        return

    print(f"Found {len(by_traj)} trajectory(ies): {sorted(by_traj)}")
    for traj, paths in by_traj.items():
        print(f"  traj{traj}: {', '.join(p.name for p in paths)}")

    rows = aggregate(by_traj)
    write_csv(Path(args.out), rows)
    print(f"\nWrote {len(rows)} aggregated rows -> {args.out}")
    print_table(rows)


if __name__ == "__main__":
    main()
