"""Measure clean per-DA-step wall-clock for every method, on an idle GPU.

WHY
---
The ``seconds`` column in ``results/<case>/metrics/*.csv`` is not trustworthy: many
of those cells were timed while other jobs shared the GPU (measured contention on
2026-07-30: a D-Flow cell went 160 s -> 302 s, 1.9x, purely from a second CUDA job),
and some fell back to CPU. Every other column is deterministic and fine.

This script re-measures the timing from scratch on an idle machine and writes it to
a SEPARATE csv. It does NOT touch the existing metrics files -- unlike
``rerun_timing.py``, which splices new numbers into them in place. Use this one to
produce the paper's cost column; use ``rerun_timing.py`` only if you specifically
want the old files rewritten.

WHAT IT MEASURES
----------------
One ``run.py`` per (case, scenario, M, method-group) at::

    num_physical_steps = len_field_history(5) + 3   ->   n_assim = 3 DA steps
    ONE trajectory, ONE seed

The pipeline already records ``seconds_per_step = elapsed / n_assim``
(``_ns_pipeline.py:1193``), so the number written out IS the average over the 3 DA
steps, per method, exactly as requested.

CAVEAT -- WARMUP IS NOT SUBTRACTED. Each ``run.py`` is a fresh process with a cold
CUDA context, so the first of the 3 DA steps carries one-time cuDNN/kernel autotune
that the grid's 15-step runs amortised away. Averaging over 3 steps dilutes that to
~1/3 of its full weight but does not remove it, so these numbers are a slight
OVER-estimate of the marginal steady-state step -- most visibly on the cheapest
cells, where warmup is the largest fraction. (``rerun_timing.py`` cancels it by
running each cell twice at n_assim=1 and 2 and subtracting; that costs 2x the runs.
The 3-step average was chosen here deliberately as the cheaper option.)

ONLY M=25 IS MEASURED
---------------------
Per-DA-step cost is essentially linear in M -- the sampler loop is M steps, and for
the two methods where M is not a sampler-step count the mapping is still linear:
D-Flow's M drives ``num_optim_steps`` (its Langevin chain length), and
``inflated_shared`` does ``M/k`` Jacobian refreshes per DA step. So the other rungs
of the ladder are obtained as ``seconds_per_step(M) = seconds_per_step(25) * M/25``
rather than measured, which cuts the sweep from 84 cells to 24 and removes the
expensive high-M shared cells (an urban M=250 shared cell alone is ~9-10 h).

Two honest limits on that extrapolation, both worth checking before the numbers go
in a table:
  * It is linear but not perfectly PROPORTIONAL. Measured on 2026-07-30, SDA's
    sampling time per cell was 5.7 / 11.2 / 23 / 76 s at M = 25 / 50 / 100 / 250,
    i.e. ~0.23 s per M-step at M=25 but ~0.30 at M=250 -- a ~30% superlinear drift
    at the top of the ladder. FIG, FlowDAS and D-Flow were much closer to
    proportional over the same range. So an x10 extrapolation to M=250 may
    UNDER-estimate by up to ~30% for some methods.
  * It does not apply to ``classical`` (EnKF / Particle filter) at all -- M is inert
    there, they propagate with the true solver. That group is already timed once per
    scenario with no M axis, so nothing needs extrapolating; do NOT scale it.
Add ``--steps 25 250`` to spot-check the two ends on whichever methods matter most.

METHOD GROUPS mirror the grid scripts exactly, so each method is timed under the
same batching and the same resolved per-cell hyperparameters it runs with in
production -- and one process startup is shared across the methods in a group:

  navier_stokes   ours_jacfree, ours_shared_k10, baselines, classical(no M axis)
  urban           ours_jacfree, ours_shared_k10, baselines, dflow

(NS folds D-Flow into ``baselines`` because run_ns_grid.sh does; urban splits it out
because run_urban_grid.sh does. Urban has no EnKF/Particle filter -- generative-only,
no in-repo solver -- so it has no ``classical`` group.)

+skip_kl_reference=true IS PASSED FOR NS AND IS NOT OPTIONAL. With neither
``skip_kl_reference`` nor ``kl_reference_states`` set, the NS driver falls through to
``build_reference_trajectory`` (driver.py:236-247) and draws an SI-SDE reference at
``reference_ensemble_size``, which navier_stokes.yaml sets to **1024**. That does not
corrupt the timing (the timer wraps only ``sample_trajectory``) but it costs hours per
cell -- the same pathology that made run_ns_reference.sh 93% dead time.

OUTPUT
------
``results/timing/seconds_per_step.csv`` -- one row per (case, scenario, M, method,
variant), appended incrementally so an interrupted run keeps what it measured.
Re-running SKIPS rows already present (resume), so it is safe to relaunch verbatim.

USAGE
-----
    # on an OTHERWISE-IDLE GPU:
    .venv/bin/python paper_experiments/measure_timing.py
    .venv/bin/python paper_experiments/measure_timing.py --case urban
    .venv/bin/python paper_experiments/measure_timing.py --dry-run
    .venv/bin/python paper_experiments/measure_timing.py --steps 25 --limit 3   # smoke

    # cheap cells first, so a partial run is still useful:
    .venv/bin/python paper_experiments/measure_timing.py --order cost
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_here = Path(__file__).resolve().parent
REPO_ROOT = _here.parent
DEFAULT_PY = REPO_ROOT / ".venv" / "bin" / "python"

OURS = ["Ours (SI-SDE)", "Ours (DM-SDE)", "Ours (FM-ODE)"]

# Per case: hydra keys, scenarios, trajectory, and the method groups. Groups and
# lineups are copied from run_{ns,urban}_grid.sh so timing matches production.
CASES = {
    "navier_stokes": {
        "keys": ("ns_methods", "ns_scenarios"),
        "scenarios": ["16^2->128^2", "32^2->128^2", "sparse 5%", "sparse 1.5625%"],
        "traj": 10,          # grid uses 10..14; any one of them times the same
        "skip_kl": True,     # see the module docstring -- NOT optional
        "groups": [
            ("ours_jacfree",    OURS, "jacfree"),
            ("ours_shared_k10", OURS, "shared_jac10"),
            ("baselines", ["FlowDAS", "SURGE (FlowDAS)", "SDA", "SURGE (SDA)",
                           "D-Flow SGLD", "Guided FM (FIG)"], ""),
            ("classical", ["EnKF", "Particle filter"], ""),   # no M axis
        ],
    },
    "urban": {
        "keys": ("urban_methods", "urban_scenarios"),
        "scenarios": ["sparse 1.5625%", "sparse 0.78125%"],
        "traj": 1,           # grid uses 1..5
        "skip_kl": False,    # urban driver draws no reference at all
        "groups": [
            ("ours_jacfree",    OURS, "jacfree"),
            ("ours_shared_k10", OURS, "shared_jac10"),
            ("baselines", ["SURGE (FlowDAS)", "SURGE (SDA)", "Guided FM (FIG)"], ""),
            ("dflow", ["D-Flow SGLD"], ""),
        ],
    },
}

# `classical` filters propagate with the TRUE solver -- there is no sampler-step
# axis, so they are timed once per scenario rather than once per M.
NO_M_GROUPS = {"classical"}

FIELD_HISTORY = 5
N_ASSIM = 3                      # -> num_physical_steps = 5 + 3 = 8

FIELDS = ["case", "scenario", "M", "group", "method", "variant", "E", "n_assim",
          "seconds_per_step", "seconds_total", "nfe", "traj", "device", "wall_s"]


def hydra_list(values: list[str]) -> str:
    """``["a","b"]`` -- double-quoted, no spaces, safe as one argv token."""
    return "[" + ",".join(f'"{v}"' for v in values) + "]"


def likelihood_args(variant: str) -> list[str]:
    """CLI overrides reproducing the grid's likelihood mode for this group."""
    if variant == "shared_jac10":
        return ["likelihood_mode=inflated_shared",
                "+jacobian_refresh_every=10",
                "+variant_override=shared_jac10"]
    # jacfree, and the baselines/classical groups (which ignore it, exactly as the
    # grid scripts pass it).
    return ["likelihood_mode=dps_jacobian_free"]


def build_cmd(py: str, case: str, scenario: str, M: int | None, methods: list[str],
              variant: str, E: int, device: str, results_file: Path,
              hydra_dir: Path) -> list[str]:
    cfg = CASES[case]
    mkey, skey = cfg["keys"]
    cmd = [
        py, "-u", "paper_experiments/run.py",
        f"case={case}", "seeds=[0]",
        f"+test_index={cfg['traj']}",
        f"ensemble_size={E}",
        f"case.num_physical_steps={FIELD_HISTORY + N_ASSIM}",
        "case.require_weights=true",
        f"case.device={device}",
        f"+{mkey}={hydra_list(methods)}",
        f"+{skey}={hydra_list([scenario])}",
        f"num_steps={M if M is not None else 50}",
        *likelihood_args(variant),
        "+save_states=false",
        f"results_file={results_file}",
        f"hydra.run.dir={hydra_dir}",
    ]
    if cfg["skip_kl"]:
        cmd.append("+skip_kl_reference=true")
    return cmd


def read_seconds(path: Path) -> list[dict]:
    """Pull one (method, variant, seconds, nfe) record per method from a cell CSV."""
    out: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (r["method"], (r.get("variant") or "").strip())
            if key in out:
                continue
            try:
                sec = float(r.get("seconds", ""))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(sec):
                continue
            try:
                nfe = float(r.get("NFE", "") or "nan")
            except ValueError:
                nfe = float("nan")
            out[key] = {"method": r["method"], "variant": key[1],
                        "seconds_per_step": sec, "nfe": nfe}
    return list(out.values())


def load_done(out_csv: Path) -> set[tuple[str, str, str, str]]:
    """(case, scenario, M, group) already measured -- used to resume."""
    done: set[tuple[str, str, str, str]] = set()
    if not out_csv.exists():
        return done
    with open(out_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            done.add((r["case"], r["scenario"], r["M"], r["group"]))
    return done


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", choices=["navier_stokes", "urban", "both"],
                    default="both")
    ap.add_argument("--steps", type=int, nargs="+", default=[25],
                    help="M values to time (default: 25 only -- per-DA-step cost is "
                         "linear in M, so the other rungs are extrapolated; see the "
                         "module docstring)")
    ap.add_argument("--groups", nargs="+", default=None,
                    help="subset of group names to run (default: all for the case)")
    ap.add_argument("-E", "--ensemble", type=int, default=64,
                    help="ensemble size; default 64 = the grid's, so the numbers are "
                         "the production cost")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--python", default=str(DEFAULT_PY))
    ap.add_argument("--out", type=Path,
                    default=_here / "results" / "timing" / "seconds_per_step.csv")
    ap.add_argument("--order", choices=["grid", "cost"], default="cost",
                    help="'cost' runs cheap cells first so a partial run is useful")
    ap.add_argument("--limit", type=int, default=None, help="only the first N cells")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cases = ["navier_stokes", "urban"] if args.case == "both" else [args.case]

    # Build the cell list: (case, scenario, M, group, methods, variant).
    cells = []
    for case in cases:
        cfg = CASES[case]
        for gname, methods, variant in cfg["groups"]:
            if args.groups and gname not in args.groups:
                continue
            Ms = [None] if gname in NO_M_GROUPS else list(args.steps)
            for scenario in cfg["scenarios"]:
                for M in Ms:
                    cells.append((case, scenario, M, gname, methods, variant))

    if args.order == "cost":
        # Cheap first: small M, and the shared-Jacobian group last (its cost grows
        # with M/k refreshes x one JVP per observation -- by far the dearest).
        cells.sort(key=lambda c: (c[3] == "ours_shared_k10", c[2] or 0))

    out_csv: Path = args.out
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_csv)
    todo = [c for c in cells
            if (c[0], c[1], "" if c[2] is None else str(c[2]), c[3]) not in done]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"cells: {len(cells)} total, {len(cells) - len(todo)} already in "
          f"{out_csv.name}, {len(todo)} to run")
    print(f"  n_assim={N_ASSIM} (num_physical_steps={FIELD_HISTORY + N_ASSIM}), "
          f"E={args.ensemble}, device={args.device}")
    print(f"  seconds_per_step = elapsed / {N_ASSIM}  (average over the 3 DA steps)")
    if args.dry_run:
        for c in todo:
            case, scenario, M, gname, methods, variant = c
            print(f"  [dry-run] {case:14s} {scenario:16s} M={str(M):4s} {gname}"
                  f"  ({len(methods)} method(s))")
        print(f"\n{len(todo)} run.py invocation(s) would be made.")
        return

    workdir = Path(tempfile.mkdtemp(prefix="measure_timing_"))
    log_dir = out_csv.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    new_file = not out_csv.exists()
    fh_out = open(out_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh_out, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()
        fh_out.flush()

    ok = fail = 0
    for i, (case, scenario, M, gname, methods, variant) in enumerate(todo, 1):
        slug = re.sub(r"[^A-Za-z0-9]+", "_",
                      f"{case}__{scenario}__M{M}__{gname}").strip("_")
        res = workdir / f"{slug}.csv"
        cmd = build_cmd(args.python, case, scenario, M, methods, variant,
                        args.ensemble, args.device, res, workdir / f"hydra_{slug}")
        print(f"[{i}/{len(todo)}] {case} / {scenario} / M={M} / {gname} ...",
              flush=True)
        t0 = time.time()
        with open(log_dir / f"{slug}.log", "w", encoding="utf-8") as log:
            log.write("# " + " ".join(cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=log,
                                  stderr=subprocess.STDOUT)
        wall = time.time() - t0
        if proc.returncode != 0:
            print(f"    FAILED rc={proc.returncode}; see {log_dir / (slug + '.log')}")
            fail += 1
            continue
        recs = read_seconds(res)
        if not recs:
            print("    FAILED: no timing rows produced")
            fail += 1
            continue
        for rec in recs:
            writer.writerow({
                "case": case, "scenario": scenario,
                "M": "" if M is None else M, "group": gname,
                "method": rec["method"], "variant": rec["variant"],
                "E": args.ensemble, "n_assim": N_ASSIM,
                "seconds_per_step": rec["seconds_per_step"],
                "seconds_total": rec["seconds_per_step"] * N_ASSIM,
                "nfe": rec["nfe"], "traj": CASES[case]["traj"],
                "device": args.device, "wall_s": round(wall, 1),
            })
        fh_out.flush()
        ok += 1
        print(f"    ok ({wall/60:.1f} min wall): "
              + ", ".join(f"{r['method']}={r['seconds_per_step']:.2f}s/step"
                          for r in recs))

    fh_out.close()
    print(f"\nDone. {ok} cell(s) measured, {fail} failed. -> {out_csv}")
    print("Existing results/<case>/metrics/*.csv were NOT modified.")


if __name__ == "__main__":
    main()
