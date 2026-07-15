"""Run-status tracker for the reduced paper grid.

Scans ``results/<case>/metrics/*.csv`` (the per-cell tidy files written by the
``run_*_grid.sh`` master scripts) and reports coverage of the expected grid:
which (case, method, variant, scenario, M, trajectory/seed) cells are present,
which are missing, and which produced non-finite (NaN) values. Also checks the
paired ``per_step/`` curves and the ``states/traj1/`` ensembles.

Writes a human-readable summary to ``results/STATUS.md`` and prints it. Read-only
over the results tree apart from that one file.

    .venv/bin/python paper_experiments/status.py
    .venv/bin/python paper_experiments/status.py --case navier_stokes
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from results_schema import load_records  # noqa: E402

RESULTS = _here / "results"

# --------------------------------------------------------------------------- #
# Expected reduced grid (mirrors the run_*_grid.sh master scripts + README).
# --------------------------------------------------------------------------- #

STEPS = (50, 100, 250)#, 500)
# Slice rows into test_data (data[180:200]); 10..14 == global trajectories 190-194.
TRAJ = (10, 11, 12, 13, 14)
SEEDS = (0) #, 1, 2, 3, 4)

OURS = ("Ours (SI-SDE)", "Ours (DM-SDE)", "Ours (FM-ODE)")
OURS_VARIANTS = ("jacfree", "shared")
GEN_BASELINES = ("FlowDAS", "SURGE (FlowDAS)", "SDA", "SURGE (SDA)", "D-Flow SGLD")
CLASSICAL = ("EnKF", "Particle filter")

NS_SCENARIOS = ("16^2->128^2", "32^2->128^2", "sparse 5%", "sparse 1.5625%")
URBAN_SCENARIOS = ("sparse 5%", "sparse 1.5625%")
ANALYTICAL_SCENARIOS = ("analytical",)

CASES = {
    "navier_stokes": {
        "scenarios": NS_SCENARIOS,
        "classical": CLASSICAL,
        "axis": "traj",
        "axis_vals": TRAJ,
    },
    "urban": {
        "scenarios": URBAN_SCENARIOS,
        "classical": (),
        "axis": "traj",
        "axis_vals": TRAJ,
    },
    "analytical": {
        "scenarios": ANALYTICAL_SCENARIOS,
        "classical": CLASSICAL,
        # Analytical averages its 5-seed list IN-RUN (rows are seed-aggregated,
        # seed=-1), so coverage is a single "present?" bucket, not 5 seeds.
        "axis": "agg",
        "axis_vals": (0,),
    },
}


def _cell_key(method: str, variant: str | None) -> str:
    return f"{method} [{variant}]" if variant else method


def expected_cells(case: str) -> set[tuple[str, str, int | None]]:
    """Return the expected {(cell_label, scenario, M)} set for a case.

    M is ``None`` for classical rows (no step sweep).
    """
    spec = CASES[case]
    cells: set[tuple[str, str, int | None]] = set()
    for scen in spec["scenarios"]:
        for m in STEPS:
            for method in OURS:
                for var in OURS_VARIANTS:
                    cells.add((_cell_key(method, var), scen, m))
            for method in GEN_BASELINES:
                cells.add((_cell_key(method, None), scen, m))
        for method in spec["classical"]:
            cells.add((_cell_key(method, None), scen, None))
    return cells


def scan(case: str) -> dict:
    """Load all metric rows for a case; return coverage + NaN + axis presence."""
    spec = CASES[case]
    metrics_dir = RESULTS / case / "metrics"
    # (cell_label, scenario, M) -> {axis_val -> finite?}
    present: dict[tuple[str, str, int | None], dict[int, bool]] = defaultdict(dict)
    files = sorted(metrics_dir.glob("*.csv")) if metrics_dir.exists() else []
    for f in files:
        try:
            recs = load_records(f)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  ! failed to read {f.name}: {exc}")
            continue
        for r in recs:
            if r.metric in ("nfe", "seconds"):
                continue
            label = _cell_key(r.method, r.variant)
            m = r.M if (r.M in STEPS) else None
            # classical rows may carry an arbitrary placeholder M -> normalise.
            if r.method in spec["classical"]:
                m = None
            if spec["axis"] == "agg":
                axis_val = 0  # single "present?" bucket (seed-aggregated in-run)
            elif spec["axis"] == "seed":
                axis_val = r.seed
            else:
                axis_val = _traj_from_name(f.name, r.seed)
            finite = r.value is not None and not (
                isinstance(r.value, float) and math.isnan(r.value)
            )
            key = (label, r.scenario, m)
            prev = present[key].get(axis_val, False)
            present[key][axis_val] = prev or finite
    return {"present": present, "n_files": len(files)}


def _traj_from_name(fname: str, seed: int) -> int:
    """Recover the trajectory index from a per-cell filename (``__traj<N>__``)."""
    import re

    m = re.search(r"traj(\d+)", fname)
    return int(m.group(1)) if m else seed


def render(case: str, data: dict) -> list[str]:
    spec = CASES[case]
    axis_vals = spec["axis_vals"]
    exp = expected_cells(case)
    present = data["present"]
    lines: list[str] = []
    done = missing = partial = nan_cells = 0
    detail: list[str] = []
    for key in sorted(exp):
        label, scen, m = key
        got = present.get(key, {})
        n_ok = sum(1 for v in got.values() if v)
        n_have = len(got)
        want = len(axis_vals)
        if n_ok >= want:
            done += 1
        elif n_ok == 0 and n_have == 0:
            missing += 1
            detail.append(f"    MISSING  {label:24s} {scen:16s} M={m}")
        elif n_ok == 0:
            nan_cells += 1
            detail.append(f"    ALL-NaN  {label:24s} {scen:16s} M={m}  ({n_have} runs)")
        else:
            partial += 1
            detail.append(
                f"    PARTIAL  {label:24s} {scen:16s} M={m}  {n_ok}/{want} {spec['axis']}s"
            )
    total = len(exp)
    lines.append(f"## {case}")
    lines.append("")
    lines.append(
        f"- metric files: {data['n_files']} | expected cells: {total} | "
        f"complete: {done} | partial: {partial} | all-NaN: {nan_cells} | "
        f"missing: {missing}"
    )
    if detail:
        lines.append("")
        lines.append("```")
        lines.extend(detail[:200])
        if len(detail) > 200:
            lines.append(f"    ... (+{len(detail) - 200} more)")
        lines.append("```")
    # states / per_step presence
    states = RESULTS / case / "states"
    per_step = RESULTS / case / "per_step"
    n_states = len(list(states.rglob("*.npz"))) if states.exists() else 0
    n_ps = len(list(per_step.glob("*.csv"))) if per_step.exists() else 0
    lines.append("")
    lines.append(f"- states files (traj1): {n_states} | per_step files: {n_ps}")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", choices=sorted(CASES), default=None)
    ap.add_argument("--out", default=str(RESULTS / "STATUS.md"))
    args = ap.parse_args()

    cases = [args.case] if args.case else list(CASES)
    out_lines = ["# Reduced-grid run status", "", "_(auto-written by `status.py`)_", ""]
    for case in cases:
        data = scan(case)
        out_lines.extend(render(case, data))

    text = "\n".join(out_lines)
    print(text)
    Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
