"""Shared reduction + IO for the per-case aggregation scripts.

``aggregate_analytical.py`` / ``aggregate_ns.py`` / ``aggregate_urban.py`` all
build on these helpers so the three cases reduce their per-cell tidy files
identically. Two reductions, both *across trajectories*:

* SCALAR ("full") metrics -- one value per
  ``(case, method, variant, scenario, metric, E, M)`` cell, reduced to the
  mean +/- std ACROSS trajectories. Each NS / urban per-cell metrics file already
  holds ONE trajectory's value (the metric is time-averaged in-run); analytical
  files are seed-aggregated with no trajectories (a single bucket, ``n_traj=1``).
  Written with the canonical :class:`results_schema.ResultRecord` columns plus an
  ``n_traj`` column -- exactly what ``make_*_figures.load_metric_vs_M`` and
  ``make_tables.py`` consume from ``results/<case>/aggregated/all.csv``.

* PER-STEP metrics (NS / urban only) -- the per-assimilation-step metric curves,
  reduced to the mean +/- std ACROSS trajectories AT EACH STEP. Written in the
  :mod:`common.per_step_io` columns plus ``std`` and ``n_traj`` to
  ``results/<case>/aggregated/per_step.csv``; consumed by the make-figure scripts'
  ``load_metric_vs_step`` for the metric-vs-assimilation-step figures.

Trajectory identity comes from the ``__traj<N>__`` token in the FILENAME (as the
grids name their per-cell files), NOT from any in-file ``test_index`` column --
the run scripts encode the trajectory in the path, so the filename is the
reliable key. Files with no ``traj<N>`` token (analytical) fall into a single
bucket 0.
"""

from __future__ import annotations

import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from common.per_step_io import load_per_step
from results_schema import SEED_AGGREGATED, ResultRecord, load_records

# Files that are NOT per-cell metric results (KL-reference bookkeeping etc.).
SKIP_PREFIX: tuple[str, ...] = ("ref_",)

# Output schema: the canonical scalar schema + n_traj.
SCALAR_FIELDNAMES: tuple[str, ...] = (*ResultRecord.FIELDNAMES, "n_traj")

# Per-step output schema: per_step_io columns + std (after value) + n_traj (last).
PER_STEP_AGG_FIELDNAMES: tuple[str, ...] = (
    "case", "method", "scenario", "variant", "metric", "step",
    "value", "std", "E", "M", "seed", "test_index", "NFE", "seconds", "n_traj",
)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def discover_by_traj(metrics_dir: Path) -> dict[int, list[Path]]:
    """Map trajectory index -> its per-cell CSV paths.

    Trajectory is parsed from ``__traj<N>__`` in the filename; files without it
    (analytical, already seed-aggregated) fall into bucket 0. Skips the
    ``ref_*`` bookkeeping files.
    """
    by_traj: dict[int, list[Path]] = defaultdict(list)
    if not metrics_dir.exists():
        return {}
    for p in sorted(metrics_dir.glob("*.csv")):
        if p.name.startswith(SKIP_PREFIX):
            continue
        m = re.search(r"traj(\d+)", p.name)
        by_traj[int(m.group(1)) if m else 0].append(p)
    return dict(sorted(by_traj.items()))


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #


def _finite(values: list[float | None]) -> list[float]:
    return [v for v in values if v is not None and not math.isnan(v)]


def _mean_opt(values: list[float | None]) -> float | None:
    present = _finite(values)
    return statistics.fmean(present) if present else None


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample std over already-finite values (std=0 for a single value)."""
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def _as_float(v: object) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _as_opt_float(v: object) -> float | None:
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_opt_int(v: object) -> int | None:
    f = _as_opt_float(v)
    return None if f is None else int(f)


def _norm_variant(v: object) -> str | None:
    """Empty / ``None`` / ``"None"`` variant cells -> ``None`` (a single-mode row)."""
    if v is None:
        return None
    s = str(v).strip()
    return None if s in ("", "None") else s


# --------------------------------------------------------------------------- #
# SCALAR ("full") aggregation over trajectories
# --------------------------------------------------------------------------- #


def _per_traj_value(records: list[ResultRecord]) -> float:
    """The single per-trajectory value for one cell.

    A per-cell file holds that trajectory's value in the ``seed ==
    SEED_AGGREGATED`` (-1) rows; prefer those, else the mean of any present rows
    so a differently-seeded file still aggregates.
    """
    agg = [r.value for r in records if r.seed == SEED_AGGREGATED]
    vals = agg if agg else [r.value for r in records]
    finite = _finite(vals)
    return statistics.fmean(finite) if finite else float("nan")


def aggregate_scalar(by_traj: dict[int, list[Path]]) -> list[dict[str, object]]:
    """Reduce per-trajectory scalar cells to mean +/- std over trajectories.

    Grouping key is ``(case, method, scenario, metric, E, M, variant)``. For each
    group we collect ONE value per trajectory bucket, then take nanmean / nanstd
    across trajectories. All-NaN groups are skipped. Every metric present in the
    inputs flows through (the reduction is metric-agnostic), so no case-specific
    metric is dropped.
    """
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
        traj_vals = _finite([_per_traj_value(recs) for recs in per_traj.values()])
        if not traj_vals:  # skip all-NaN groups
            continue
        mean, std = _mean_std(traj_vals)
        all_recs = [r for recs in per_traj.values() for r in recs]
        row = ResultRecord(
            case=case, method=method, scenario=scenario, metric=metric,
            value=mean, std=std, E=E, M=M, seed=SEED_AGGREGATED,
            nfe=_mean_opt([r.nfe for r in all_recs]),
            seconds=_mean_opt([r.seconds for r in all_recs]),
            variant=variant,
        ).to_row()
        row["n_traj"] = len(traj_vals)
        rows.append(row)
    return rows


def write_scalar_csv(out: Path, rows: list[dict[str, object]]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(SCALAR_FIELDNAMES))
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in SCALAR_FIELDNAMES})


# --------------------------------------------------------------------------- #
# PER-STEP aggregation over trajectories
# --------------------------------------------------------------------------- #


def aggregate_per_step(by_traj: dict[int, list[Path]]) -> list[dict[str, object]]:
    """Reduce per-trajectory per-step curves to mean +/- std over trajectories.

    Grouping key is ``(case, method, variant, scenario, metric, step, E, M)``.
    One value per trajectory bucket is collected at each step, then reduced with
    nanmean / nanstd across trajectories. All-NaN groups are skipped. ``NFE`` /
    ``seconds`` are carried as the mean over the contributing rows. Output rows
    carry ``seed = test_index = -1`` (aggregated) plus ``std`` and ``n_traj``.
    """
    # key -> {traj -> value}   and   key -> {"nfe": [...], "seconds": [...]}
    cells: dict[tuple, dict[int, float]] = defaultdict(dict)
    cost: dict[tuple, dict[str, list[float | None]]] = defaultdict(
        lambda: {"nfe": [], "seconds": []}
    )
    for traj, paths in by_traj.items():
        for path in paths:
            for row in load_per_step(path):
                key = (
                    str(row["case"]), str(row["method"]), _norm_variant(row.get("variant")),
                    str(row["scenario"]), str(row["metric"]), _as_opt_int(row.get("step")),
                    _as_opt_int(row.get("E")), _as_opt_int(row.get("M")),
                )
                cells[key][traj] = _as_float(row.get("value"))
                cost[key]["nfe"].append(_as_opt_float(row.get("NFE")))
                cost[key]["seconds"].append(_as_opt_float(row.get("seconds")))

    rows: list[dict[str, object]] = []
    for key, traj_vals in cells.items():
        case, method, variant, scenario, metric, step, E, M = key
        finite = _finite(list(traj_vals.values()))
        if not finite:
            continue
        mean, std = _mean_std(finite)
        rows.append({
            "case": case, "method": method, "scenario": scenario,
            "variant": variant, "metric": metric, "step": step,
            "value": mean, "std": std, "E": E, "M": M,
            "seed": SEED_AGGREGATED, "test_index": SEED_AGGREGATED,
            "NFE": _mean_opt(cost[key]["nfe"]),
            "seconds": _mean_opt(cost[key]["seconds"]),
            "n_traj": len(finite),
        })
    # Stable order: (method, variant, scenario, metric, step).
    rows.sort(key=lambda r: (
        str(r["method"]), str(r["variant"] or ""), str(r["scenario"]),
        str(r["metric"]), r["step"] if r["step"] is not None else -1,
    ))
    return rows


def write_per_step_csv(out: Path, rows: list[dict[str, object]]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(PER_STEP_AGG_FIELDNAMES))
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in PER_STEP_AGG_FIELDNAMES})


# --------------------------------------------------------------------------- #
# Human-readable summary
# --------------------------------------------------------------------------- #


def print_scalar_table(rows: list[dict[str, object]], key_metrics: tuple[str, ...]) -> None:
    """Compact method x scenario table for ``key_metrics`` (mean +/- std over traj)."""
    look = {
        (r["method"], r["scenario"], r["metric"], r.get("variant") or ""):
            (r["value"], r["std"], r["n_traj"])
        for r in rows
    }
    methods = sorted({str(r["method"]) for r in rows})
    scenarios = sorted({str(r["scenario"]) for r in rows})
    variants = sorted({str(r.get("variant") or "") for r in rows})
    for metric in key_metrics:
        present = [
            (m, s, v) for m in methods for s in scenarios for v in variants
            if (m, s, metric, v) in look
        ]
        if not present:
            continue
        scen_here = sorted({s for (_, s, _) in present})
        print(f"\n=== {metric} (mean +/- std over trajectories) ===")
        print(f"{'method [variant]':<26}" + "".join(f"{s:>24}" for s in scen_here))
        for m in methods:
            for v in variants:
                if not any((m, s, metric, v) in look for s in scen_here):
                    continue
                label = m + (f" [{v}]" if v else "")
                cells = []
                for s in scen_here:
                    if (m, s, metric, v) in look:
                        val, sd, n = look[(m, s, metric, v)]
                        cells.append(f"{val:.4f}+/-{sd:.4f}(n{n})")
                    else:
                        cells.append("-")
                print(f"{label:<26}" + "".join(f"{c:>24}" for c in cells))


__all__ = [
    "SKIP_PREFIX", "SCALAR_FIELDNAMES", "PER_STEP_AGG_FIELDNAMES",
    "discover_by_traj", "aggregate_scalar", "write_scalar_csv",
    "aggregate_per_step", "write_per_step_csv", "print_scalar_table",
]
