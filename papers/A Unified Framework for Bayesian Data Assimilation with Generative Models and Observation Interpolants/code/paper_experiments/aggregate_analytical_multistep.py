"""Reduce the sequential linear--Gaussian grid over seeds.

Reads ``results/analytical_multistep/{metrics,per_step}/*__traj<seed>.csv`` and
writes ``aggregated/all.csv`` (cycle-averaged, canonical tidy schema) and
``aggregated/per_step.csv`` (one row per method, metric and cycle). Both carry the
mean and the across-seed std over the seeds with a finite value; ``n_traj`` is how
many those were. A mean with ``n_traj`` below the seed count covers survivors only.

    "$PY" "$EXPERIMENT_DIR/aggregate_analytical_multistep.py"
"""

from __future__ import annotations

import argparse
import re
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

ROOT = _here / "results" / "analytical_multistep"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    root = ap.parse_args().root

    scalar_files = discover_by_traj(root / "metrics")
    if not scalar_files:
        print(f"[multistep] no metric files in {root / 'metrics'}")
        return
    names = {path.name for paths in scalar_files.values() for path in paths}
    per_step_names = {path.name for path in (root / "per_step").glob("*.csv")}
    if names != per_step_names:
        raise SystemExit(
            f"[multistep] metrics/ and per_step/ differ: {sorted(names ^ per_step_names)}"
        )
    # The per-step key has E and M but not K: two horizons would overwrite each other.
    horizons = {match.group(1) for name in names if (match := re.search(r"__K(\d+)__", name))}
    if len(horizons) > 1:
        raise SystemExit(f"[multistep] several horizons in one root: K={sorted(horizons)}")
    rows = aggregate_scalar(scalar_files)
    write_scalar_csv(root / "aggregated" / "all.csv", rows)
    per_step = aggregate_per_step(discover_by_traj(root / "per_step"))
    write_per_step_csv(root / "aggregated" / "per_step.csv", per_step)
    print(f"[multistep] wrote {len(rows)} scalar and {len(per_step)} per-step rows")
    print_scalar_table(rows, ("kl_points", "kl_exact", "sliced_w2", "kl_chain"))


if __name__ == "__main__":
    main()
