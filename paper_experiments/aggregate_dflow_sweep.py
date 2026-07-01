#!/usr/bin/env python3
"""Aggregate the D-Flow SGLD hyperparameter sweep into ranked per-scenario tables.

Scrapes ``paper_experiments/results/dflow_sweep/dflow_<scen>_lam<L>_s<S>_eta<E>_K<K>.csv``,
parses (scenario, lambda, s, eta, K) from each filename, pulls the key metrics, and
prints one table per scenario sorted by CRPS (primary; a proper probabilistic score),
flagging the best lambda / s / eta. LOWER IS BETTER for every reported metric: the
stored ``spread_skill`` is the calibration deviation |1 - spread/skill| (0 = perfectly
calibrated), not the raw ratio.

Usage:  .venv/bin/python paper_experiments/aggregate_dflow_sweep.py
        [--dir paper_experiments/results/dflow_sweep] [--sort crps|rmse]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict

FNAME_RE = re.compile(
    r"dflow_(?P<scen>.+)_lam(?P<lam>[^_]+)_s(?P<s>[^_]+)_eta(?P<eta>[^_]+)_K(?P<K>\d+)\.csv$"
)
SCEN_LABEL = {
    "32_2_128_2": "superres_32 (32^2->128^2)",
    "16_2_128_2": "superres_16 (16^2->128^2)",
    "sparse_5": "sparse_5 (5%)",
    "sparse_1_5625": "sparse_1p5 (1.5625%)",
}
SCEN_ORDER = ["32_2_128_2", "16_2_128_2", "sparse_5", "sparse_1_5625"]
METRICS = ["rmse", "crps", "spread_skill", "energy_spec_rmse", "kl_points"]


def _f(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def read_cell(path: str) -> dict:
    """Return {metric: value} for one per-cell CSV (metric rows: name -> value)."""
    out: dict[str, float] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["metric"]] = _f(row["value"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="paper_experiments/results/dflow_sweep")
    ap.add_argument("--sort", default="crps", choices=["crps", "rmse"])
    args = ap.parse_args()

    rows_by_scen: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.dir, "dflow_*.csv"))):
        m = FNAME_RE.search(os.path.basename(path))
        if not m:
            continue
        met = read_cell(path)
        rows_by_scen[m["scen"]].append(
            {
                "lam": m["lam"], "s": m["s"], "eta": m["eta"], "K": m["K"],
                "lamf": _f(m["lam"]),
                **{k: met.get(k, float("nan")) for k in METRICS},
                "nfe": met.get("nfe", float("nan")),
                "seconds": met.get("seconds", float("nan")),
            }
        )

    n_total = sum(len(v) for v in rows_by_scen.values())
    print(f"\nD-Flow SGLD sweep -- {n_total} cells across {len(rows_by_scen)} scenarios "
          f"(sorted by {args.sort}; LOWER IS BETTER for every column, "
          f"spread_skill = |1-ratio| calibration deviation)\n")

    for scen in SCEN_ORDER + [s for s in rows_by_scen if s not in SCEN_ORDER]:
        rows = rows_by_scen.get(scen)
        if not rows:
            continue
        rows.sort(key=lambda r: (r[args.sort] != r[args.sort], r[args.sort]))  # NaN last
        print(f"=== {SCEN_LABEL.get(scen, scen)}  ({len(rows)} cells) ===")
        print(f"  {'lambda':>7} {'s':>6} {'eta':>6} | {'rmse':>7} {'crps':>7} "
              f"{'sk|1-r|':>7} {'espec':>7} {'kl':>9}")
        for r in rows:
            print(f"  {r['lam']:>7} {r['s']:>6} {r['eta']:>6} | "
                  f"{r['rmse']:>7.4f} {r['crps']:>7.4f} {r['spread_skill']:>7.3f} "
                  f"{r['energy_spec_rmse']:>7.3f} {r['kl_points']:>9.2f}")
        # Best per metric = minimum (lower is better for all, incl. spread_skill dev).
        def best(key):
            valid = [r for r in rows if r[key] == r[key]]
            if not valid:
                return "n/a"
            b = min(valid, key=lambda r: r[key])
            return f"lam={b['lam']} s={b['s']} eta={b['eta']} ({b[key]:.4f})"
        print(f"  -> best rmse         : {best('rmse')}")
        print(f"  -> best crps         : {best('crps')}")
        print(f"  -> best calibration  : {best('spread_skill')}")
        print(f"  -> best kl_points    : {best('kl_points')}\n")


if __name__ == "__main__":
    main()
