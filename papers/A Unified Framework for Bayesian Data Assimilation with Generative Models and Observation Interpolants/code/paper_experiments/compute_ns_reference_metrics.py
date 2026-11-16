"""Collect the large-ensemble EnKF reference metrics into one tidy CSV.

``run_ns_reference.sh`` writes one ``metrics/ref_<scenario>__traj<N>.csv`` per cell,
with ``method=EnKF``. ``aggregate_ns.py`` keeps those files out of the graded EnKF
rows and instead reads ``reference/metrics.csv``, under the method name
``EnKF (reference)`` and with one ``test_index`` column. This writes that file.

    .venv/bin/python paper_experiments/compute_ns_reference_metrics.py \
        [--results-root paper_experiments/results]
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

REF_METHOD = "EnKF (reference)"
COLUMNS = ("case", "method", "scenario", "metric", "value", "std", "E", "M", "seed",
           "NFE", "seconds", "variant", "test_index")


def collect(metrics_dir: Path) -> list[dict[str, str]]:
    """Return the tidy rows of every ``ref_*__traj<N>.csv`` in ``metrics_dir``."""
    rows: list[dict[str, str]] = []
    for path in sorted(metrics_dir.glob("ref_*.csv")):
        match = re.search(r"traj(\d+)", path.name)
        if match is None:
            raise ValueError(f"no traj<N> token in {path.name}")
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                # The reference run skips its own KL draw, so kl_points is NaN.
                if row["metric"] == "kl_points":
                    continue
                row.update(method=REF_METHOD, seed="0", test_index=match.group(1))
                rows.append({c: row.get(c, "") for c in COLUMNS})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=Path,
                    default=Path(__file__).resolve().parent / "results",
                    help="directory holding navier_stokes/")
    args = ap.parse_args()
    root = args.results_root / "navier_stokes"
    rows = collect(root / "metrics")
    if not rows:
        raise SystemExit(f"no ref_*.csv cells in {root / 'metrics'}")
    out = root / "reference" / "metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[nsref] wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
