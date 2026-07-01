"""Aggregate the reduced-grid per-cell metric files into one tidy file per case.

Reads every per-cell CSV under ``results/<case>/metrics/`` (as written by the
``run_*_grid.sh`` master scripts), groups by
``(case, method, variant, scenario, metric, E, M)``, and reduces the
per-trajectory values to the mean and std ACROSS trajectories -- writing
``results/<case>/aggregated/all.csv`` with the canonical schema plus an
``n_traj`` column. Variant-aware, so the two "Ours" modes (jacfree / shared)
stay distinct rows.

* NS / urban: each per-cell file holds ONE trajectory's value (the trajectory
  index is in the filename, ``__traj<N>__``); the mean/std is across trajectories.
* Analytical: files are already seed-aggregated in-run (no trajectories), so each
  cell has a single value and ``n_traj = 1`` -- effectively a concatenation.

    .venv/bin/python paper_experiments/aggregate_grid.py --case navier_stokes
    .venv/bin/python paper_experiments/aggregate_grid.py            # all cases

Reuses the (now variant-aware) reduction in ``aggregate_multitraj.aggregate``.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from aggregate_multitraj import aggregate, print_table, write_csv  # noqa: E402

RESULTS = _here / "results"
CASES = ("navier_stokes", "urban", "analytical")

# Files that are NOT per-cell metric results (KL-reference bookkeeping etc.).
_SKIP_PREFIX = ("ref_",)


def discover_by_traj(metrics_dir: Path) -> dict[int, list[Path]]:
    """Map trajectory index -> its per-cell CSV paths.

    Trajectory parsed from ``__traj<N>__`` in the filename; files without it
    (analytical, already seed-aggregated) fall into bucket 0.
    """
    by_traj: dict[int, list[Path]] = defaultdict(list)
    if not metrics_dir.exists():
        return {}
    for p in sorted(metrics_dir.glob("*.csv")):
        if p.name.startswith(_SKIP_PREFIX):
            continue
        m = re.search(r"traj(\d+)", p.name)
        traj = int(m.group(1)) if m else 0
        by_traj[traj].append(p)
    return dict(sorted(by_traj.items()))


def run_case(case: str) -> None:
    metrics_dir = RESULTS / case / "metrics"
    out = RESULTS / case / "aggregated" / "all.csv"
    by_traj = discover_by_traj(metrics_dir)
    if not by_traj:
        print(f"[{case}] no per-cell metric files in {metrics_dir}")
        return
    print(f"[{case}] trajectories/buckets: {sorted(by_traj)} "
          f"({sum(len(v) for v in by_traj.values())} files)")
    rows = aggregate(by_traj)
    write_csv(out, rows)
    print(f"[{case}] wrote {len(rows)} aggregated rows -> {out}")
    print_table(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", choices=CASES, default=None)
    args = ap.parse_args()
    for case in ([args.case] if args.case else CASES):
        run_case(case)


if __name__ == "__main__":
    main()
