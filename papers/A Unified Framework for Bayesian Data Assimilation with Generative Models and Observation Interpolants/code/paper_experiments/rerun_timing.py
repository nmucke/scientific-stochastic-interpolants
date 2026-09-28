"""Re-measure per-DA-step wall-clock time and overwrite the ``seconds`` column.

WHY
---
The ``seconds`` recorded in ``results/<case>/metrics/*.csv`` by the grid runs is
NOT trustworthy: some cells were timed while the GPU was partly occupied by other
work, and some fell back to CPU. Every *other* column (rmse, crps, kl, NFE, ...)
is deterministic and fine -- only the wall-clock is contaminated. This script
re-times every cell on a clean machine and splices the fresh number back into the
``seconds`` column (and the ``metric==seconds`` row's ``value``), leaving all
other columns untouched.

WHAT IT MEASURES -- warmup-free marginal cost of ONE DA step
------------------------------------------------------------
``seconds`` in the tidy files is already a *per-step* number: the pipeline records
``seconds_per_step = elapsed / n_assim`` with ``n_assim = num_physical_steps -
len_field_history`` (5). The timer wraps only ``sample_trajectory``, but each
``run.py`` is a fresh process with a COLD CUDA context, so a single timed step
absorbs one-time cuDNN/kernel autotune that the original 15-step runs amortised
away. To get a warmup-free per-step number we time each cell TWICE and subtract:

    run A: num_physical_steps = L+1  -> n_assim = 1 -> elapsed_A = secA * 1
    run B: num_physical_steps = L+2  -> n_assim = 2 -> elapsed_B = secB * 2

    marginal_per_step = elapsed_B - elapsed_A = 2*secB - secA

The fixed per-process overhead (CUDA init, cuDNN autotune, the first-step warmup)
cancels in the subtraction, leaving the incremental cost of exactly one DA step.
If the subtraction comes out non-positive (measurement noise on a very cheap
cell), we fall back to ``secB`` (the stable 2-step average) and warn.

HOW A CELL IS RECONSTRUCTED
---------------------------
Each metrics CSV is exactly one ``run.py`` call. We read the file's own rows to
recover (scenario, M, E, methods, variant) and the trajectory index from the
``__traj<N>__`` in the filename, then rebuild the invocation the grid used
(``run_ns_grid.sh`` / ``run_urban_grid.sh``): same case / scenario / M / E /
test_index / likelihood_mode, so every per-case, per-scenario, per-M
hyperparameter resolves identically -- cost is reproduced, only the clock is new.

SCOPE
-----
NS + urban only (the GPU cases the complaint is about). Analytical is CPU-only,
timed consistently already, and has no autoregressive DA loop to reduce, so it is
deliberately left alone. Reference files (``ref_*.csv``) are skipped.

USAGE
-----
    # From the repository root, on an OTHERWISE-IDLE GPU:
    EXPERIMENT_DIR="$PWD/manuscripts/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments"
    .venv/bin/python "$EXPERIMENT_DIR/rerun_timing.py"
    .venv/bin/python "$EXPERIMENT_DIR/rerun_timing.py" --case urban
    .venv/bin/python "$EXPERIMENT_DIR/rerun_timing.py" --dry-run
    .venv/bin/python "$EXPERIMENT_DIR/rerun_timing.py" --limit 2

Then re-aggregate so the new seconds flow into the tables/figures:
    .venv/bin/python "$EXPERIMENT_DIR/aggregate_ns.py"
    .venv/bin/python "$EXPERIMENT_DIR/aggregate_urban.py"

NOTE: this rewrites metrics/*.csv IN PLACE (seconds only). It does NOT touch the
paired ``per_step/*.csv`` curves -- those still carry the old per-step seconds; if
the figures read timing from per_step, re-time those separately.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shlex
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from common.paths import EXPERIMENT_ROOT, REPO_ROOT, RESULTS_ROOT

DEFAULT_PY = REPO_ROOT / ".venv" / "bin" / "python"

# case -> the hydra keys the driver reads for the method / scenario subsets
# (cases/<case>/driver.py: ``{case}_methods`` / ``{case}_scenarios``).
CASE_KEYS = {
    "navier_stokes": ("ns_methods", "ns_scenarios"),
    "urban": ("urban_methods", "urban_scenarios"),
}

# len_field_history for the seeded history (configs/case/{navier_stokes,urban}.yaml
# both say 5). num_physical_steps = L + n_assim, so L+1 -> 1 DA step, L+2 -> 2.
DEFAULT_FIELD_HISTORY = 5

_TRAJ_RE = re.compile(r"__traj(\d+)__")


# --------------------------------------------------------------------------- #
# Cell reconstruction from a metrics CSV
# --------------------------------------------------------------------------- #


def _norm_variant(v: str | None) -> str:
    """Normalise the variant cell to a plain string ('' for the empty/None case)."""
    return "" if v is None or str(v).strip() == "" else str(v).strip()


def _likelihood_args(variant: str) -> list[str]:
    """CLI overrides that reproduce this variant's likelihood mode.

    Mirrors run_{ns,urban}_grid.sh:
      ''            (baselines / classical / dflow) -> dps_jacobian_free (ignored
                    by non-Ours methods, exactly as the grid passes it).
      'jacfree'     -> dps_jacobian_free
      'shared'      -> inflated_shared, k=1
      'shared_jac<k>' -> inflated_shared, k=<k>, variant_override stamped in
    """
    if variant in ("", "jacfree"):
        return ["likelihood_mode=dps_jacobian_free"]
    if variant == "shared":
        return [
            "likelihood_mode=inflated_shared",
            "+jacobian_refresh_every=1",
            "+variant_override=shared",
        ]
    m = re.fullmatch(r"shared_jac(\d+)", variant)
    if m:
        return [
            "likelihood_mode=inflated_shared",
            f"+jacobian_refresh_every={m.group(1)}",
            f"+variant_override={variant}",
        ]
    # Unknown variant -- treat as an Ours shared cell but do not guess k; run k=1
    # and stamp the label through so the temp file's variant matches for splicing.
    return [
        "likelihood_mode=inflated_shared",
        "+jacobian_refresh_every=1",
        f"+variant_override={variant}",
    ]


class Cell:
    """One ``run.py`` invocation-worth of rows, grouped by variant."""

    def __init__(self, case: str, scenario: str, num_steps: int, ensemble: int,
                 variant: str, methods: list[str]) -> None:
        self.case = case
        self.scenario = scenario
        self.num_steps = num_steps
        self.ensemble = ensemble
        self.variant = variant
        self.methods = methods


def _hydra_list(values: list[str]) -> str:
    """Render a hydra list literal exactly like the grid scripts do:
    ``["a","b"]`` -- double-quoted items, no spaces (safe as a single argv token)."""
    return "[" + ",".join(f'"{v}"' for v in values) + "]"


def parse_metrics_file(path: Path) -> tuple[list[dict], list[str], list[Cell], int | None]:
    """Load a metrics CSV and split it into per-variant :class:`Cell` groups.

    Returns ``(rows, fieldnames, cells, test_index)``. ``rows``/``fieldnames`` are
    kept verbatim so the file can be rewritten with only ``seconds`` changed.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    m = _TRAJ_RE.search(path.name)
    test_index = int(m.group(1)) if m else None

    # Group (method) by variant; scenario / M / E are uniform within a grid cell,
    # but we read them per group defensively (first row wins).
    by_variant: dict[str, dict] = {}
    for r in rows:
        var = _norm_variant(r.get("variant"))
        g = by_variant.setdefault(
            var,
            {"methods": [], "scenario": r["scenario"], "M": r.get("M"),
             "E": r.get("E"), "case": r["case"]},
        )
        if r["method"] not in g["methods"]:
            g["methods"].append(r["method"])

    cells: list[Cell] = []
    for var, g in by_variant.items():
        cells.append(
            Cell(
                case=g["case"],
                scenario=g["scenario"],
                num_steps=int(float(g["M"])) if g["M"] not in (None, "") else 50,
                ensemble=int(float(g["E"])) if g["E"] not in (None, "") else 64,
                variant=var,
                methods=g["methods"],
            )
        )
    return rows, fieldnames, cells, test_index


# --------------------------------------------------------------------------- #
# Running one cell at a given n_assim and reading back per-method seconds
# --------------------------------------------------------------------------- #


def _build_cmd(py: str, cell: Cell, test_index: int | None, num_physical: int,
               device: str, require_weights: bool, results_file: Path,
               hydra_run_dir: Path) -> list[str]:
    methods_key, scenarios_key = CASE_KEYS[cell.case]
    cmd = [
        py, "-u", str(EXPERIMENT_ROOT / "run.py"),
        f"case={cell.case}", "seeds=[0]",
        f"ensemble_size={cell.ensemble}",
        f"case.num_physical_steps={num_physical}",
        f"case.require_weights={'true' if require_weights else 'false'}",
        f"case.device={device}",
        f"+{methods_key}={_hydra_list(cell.methods)}",
        f"+{scenarios_key}={_hydra_list([cell.scenario])}",
        f"num_steps={cell.num_steps}",
        *_likelihood_args(cell.variant),
        f"results_file={results_file}",
        f"hydra.run.dir={hydra_run_dir}",
    ]
    if test_index is not None:
        cmd.append(f"+test_index={test_index}")
    return cmd


def _read_seconds_by_method(results_file: Path) -> dict[tuple[str, str], float]:
    """(method, variant) -> seconds_per_step from a freshly written temp CSV."""
    out: dict[tuple[str, str], float] = {}
    if not results_file.exists():
        return out
    with open(results_file, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (r["method"], _norm_variant(r.get("variant")))
            if key in out:
                continue
            raw = r.get("seconds")
            if raw in (None, ""):
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            if math.isfinite(val):
                out[key] = val
    return out


def _run_once(py: str, cell: Cell, test_index: int | None, num_physical: int,
              device: str, require_weights: bool, workdir: Path, log_dir: Path,
              tag: str) -> dict[tuple[str, str], float]:
    """One ``run.py`` invocation; returns per-(method,variant) seconds_per_step."""
    results_file = workdir / f"{tag}.csv"
    if results_file.exists():
        results_file.unlink()
    hydra_run_dir = workdir / f"hydra_{tag}"
    cmd = _build_cmd(py, cell, test_index, num_physical, device,
                     require_weights, results_file, hydra_run_dir)
    log_path = log_dir / f"{tag}.log"
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("# " + shlex.join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"run.py failed (rc={proc.returncode}); see {log_path}")
    return _read_seconds_by_method(results_file)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def process_file(path: Path, *, py: str, device: str, require_weights: bool,
                 field_history: int, workdir: Path, log_dir: Path,
                 dry_run: bool) -> dict:
    """Re-time every cell in one metrics CSV and rewrite its seconds column."""
    rows, fieldnames, cells, test_index = parse_metrics_file(path)
    np_a = field_history + 1  # n_assim = 1
    np_b = field_history + 2  # n_assim = 2

    # (method, variant) -> new marginal seconds/step
    new_seconds: dict[tuple[str, str], float] = {}
    fallbacks: list[str] = []
    for cell in cells:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", f"{path.stem}__{cell.variant or 'none'}")
        if dry_run:
            for num_physical in (np_a, np_b):
                cmd = _build_cmd(
                    py, cell, test_index, num_physical, device, require_weights,
                    workdir / "DRYRUN.csv", workdir / "DRYRUN_hydra",
                )
                print(f"    [dry-run] NP={num_physical}: {shlex.join(cmd)}")
            continue

        sec_a = _run_once(py, cell, test_index, np_a, device, require_weights,
                          workdir, log_dir, f"{slug}__a")
        sec_b = _run_once(py, cell, test_index, np_b, device, require_weights,
                          workdir, log_dir, f"{slug}__b")

        for m in cell.methods:
            key = (m, cell.variant)
            a = sec_a.get(key)
            b = sec_b.get(key)
            if a is None or b is None:
                continue  # method produced no timing this run; leave old value
            # elapsed_A = a*1, elapsed_B = b*2 -> marginal one-step cost.
            marginal = 2.0 * b - a
            if not math.isfinite(marginal) or marginal <= 0.0:
                marginal = b  # noise-dominated cheap cell: stable 2-step average
                fallbacks.append(m)
            new_seconds[key] = marginal

    if dry_run:
        return {"file": path.name, "cells": len(cells), "updated": 0,
                "unmatched": 0, "fallbacks": 0, "dry_run": True}

    # Splice: overwrite the seconds column on every row, and the value cell on the
    # ``metric==seconds`` rows; everything else (NFE, accuracy metrics) untouched.
    updated = 0
    unmatched: set[tuple[str, str]] = set()
    for r in rows:
        key = (r["method"], _norm_variant(r.get("variant")))
        if key not in new_seconds:
            unmatched.add(key)
            continue
        sec = new_seconds[key]
        r["seconds"] = repr(sec)
        if r["metric"] == "seconds":
            r["value"] = repr(sec)
        updated += 1

    # Rewrite atomically.
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)

    return {"file": path.name, "cells": len(cells), "updated": updated,
            "unmatched": len(unmatched), "fallbacks": len(set(fallbacks)),
            "dry_run": False}


def discover_files(results_root: Path, cases: list[str]) -> list[Path]:
    files: list[Path] = []
    for case in cases:
        met = results_root / case / "metrics"
        if not met.is_dir():
            print(f"[warn] no metrics dir for case '{case}': {met}")
            continue
        for p in sorted(met.glob("*.csv")):
            if p.name.startswith("ref_"):
                continue  # KL reference, not a timed method cell
            files.append(p)
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", choices=["navier_stokes", "urban", "both"],
                    default="both", help="which case(s) to re-time (default: both)")
    ap.add_argument("--results-root", type=Path,
                    default=RESULTS_ROOT,
                    help="root of the results tree (default: this paper_experiments/results)")
    ap.add_argument("--python", type=str, default=str(DEFAULT_PY),
                    help="python interpreter used to launch run.py")
    ap.add_argument("--device", type=str, default="cuda",
                    help="torch device for the reruns (default: cuda)")
    ap.add_argument("--field-history", type=int, default=DEFAULT_FIELD_HISTORY,
                    help="len_field_history L; NP=L+1 and L+2 give 1 and 2 DA steps")
    ap.add_argument("--require-weights", action="store_true", default=True,
                    help="fail if trained weights are missing (default: on)")
    ap.add_argument("--no-require-weights", dest="require_weights",
                    action="store_false", help="allow random weights (smoke tests only)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N files (smoke test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands that would run; change nothing")
    args = ap.parse_args()

    cases = ["navier_stokes", "urban"] if args.case == "both" else [args.case]
    files = discover_files(args.results_root, cases)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        print("Nothing to do: no metrics files found.")
        return

    workdir = (Path(tempfile.gettempdir()) / "rerun_timing_dry_run"
               if args.dry_run else Path(tempfile.mkdtemp(prefix="rerun_timing_")))
    log_dir = args.results_root / "_timing_rerun_logs"
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Re-timing {len(files)} metrics file(s) across {cases}")
    print(f"  marginal per-step = 2*sec(n_assim=2) - sec(n_assim=1)  "
          f"[NP={args.field_history + 1} vs {args.field_history + 2}]")
    print(f"  device={args.device}  require_weights={args.require_weights}"
          f"{'  [DRY RUN]' if args.dry_run else ''}")
    print(f"  scratch={workdir}\n  logs={log_dir}\n")

    summaries = []
    try:
        for i, path in enumerate(files, 1):
            print(f"[{i}/{len(files)}] {path.name}")
            try:
                s = process_file(
                    path, py=args.python, device=args.device,
                    require_weights=args.require_weights,
                    field_history=args.field_history, workdir=workdir,
                    log_dir=log_dir, dry_run=args.dry_run,
                )
            except Exception as exc:  # noqa: BLE001 -- one bad cell must not kill the sweep
                print(f"    FAILED: {exc}")
                summaries.append({"file": path.name, "error": str(exc)})
                continue
            if not args.dry_run:
                print(f"    updated {s['updated']} rows "
                      f"({s['cells']} cell(s)"
                      + (f", {s['unmatched']} method(s) left unchanged"
                         if s["unmatched"] else "")
                      + (f", {s['fallbacks']} fallback(s)" if s["fallbacks"] else "")
                      + ")")
            summaries.append(s)
    finally:
        if not args.dry_run:
            shutil.rmtree(workdir, ignore_errors=True)

    n_fail = sum(1 for s in summaries if "error" in s)
    n_updated = sum(s.get("updated", 0) for s in summaries)
    print(f"\nDone. {len(files) - n_fail}/{len(files)} files processed, "
          f"{n_updated} rows re-timed, {n_fail} failed.")
    if not args.dry_run:
        print("Next: re-run aggregate_ns.py / aggregate_urban.py to propagate "
              "the new seconds into the aggregated tables and figures.")


if __name__ == "__main__":
    main()
