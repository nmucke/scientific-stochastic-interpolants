"""Metrics for the Navier--Stokes REFERENCE posterior, straight from its states.

The reference is a large-ensemble true-solver EnKF (``E=1000``) run once per
(scenario, trajectory) by ``run_ns_reference.sh``. Its posterior ensembles are
saved as ``.npz`` state files -- ``results/navier_stokes/reference/traj<N>/gt/``
-- because they double as the KL-at-points reference distribution for every
graded cell (``+kl_reference_states=...``; see ``cases/navier_stokes/driver.py``).

This script turns those states into ONE tidy metrics CSV::

    results/navier_stokes/reference/metrics.csv

so ``aggregate_ns.py`` only has to LOAD that file to give the reference its own
table row, instead of re-deriving anything from the run's own bookkeeping CSVs.

WHY RECOMPUTE RATHER THAN READ THE RUN'S OWN CSVs
-------------------------------------------------
The reference run wrote ``metrics/ref_*.csv`` as a side effect, but those files
are the run's bookkeeping: they carry ``method="EnKF"`` (colliding with the
graded 64-member EnKF in every downstream lookup) and whatever metric set that
run happened to compute. Recomputing here from the saved ensembles gives one
authoritative file, under the reference's own method name, computed by the SAME
``_ns_pipeline.compute_metrics`` the graded cells use -- same estimators, same
step window, same observed/unobserved split -- so the reference row is comparable
to the rows above it rather than merely adjacent to them.

KL is deliberately absent (NaN): KL-at-points scores a posterior AGAINST this
reference, so the reference has nothing to be scored against but itself, which
would read as a spurious ~0.

USAGE
-----
    .venv/bin/python paper_experiments/compute_ns_reference_metrics.py
    .venv/bin/python paper_experiments/compute_ns_reference_metrics.py --limit 2
    .venv/bin/python paper_experiments/compute_ns_reference_metrics.py \\
        --states-root paper_experiments/results/navier_stokes/reference \\
        --out paper_experiments/results/navier_stokes/reference/metrics.csv

Then re-aggregate::

    .venv/bin/python paper_experiments/aggregate_ns.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

CASE = "navier_stokes"
DEFAULT_STATES_ROOT = _here / "results" / CASE / "reference"
DEFAULT_OUT = DEFAULT_STATES_ROOT / "metrics.csv"

# The method name the reference rows are written under. Distinct from the graded
# "EnKF" (E=64) on purpose -- aggregate_ns.py keys its reference table row on this
# exact string, and every other consumer (figures, tables) keys on the graded name.
REF_METHOD = "EnKF (reference)"

# The tidy scalar schema + the trajectory column. ``test_index`` is what lets the
# aggregation reduce ACROSS trajectories (mean +/- std), exactly as it does for the
# graded cells, whose trajectory lives in their filename instead.
FIELDNAMES: tuple[str, ...] = (
    "case", "method", "scenario", "metric", "value", "std",
    "E", "M", "seed", "NFE", "seconds", "variant", "test_index",
)

# Written as metric rows, in this order. ``kl_points`` is excluded (see above);
# ``nfe`` / ``seconds`` come from the state file's own recorded cost.
SCALAR_METRICS: tuple[str, ...] = (
    "rmse", "energy_spec_rmse", "crps", "crps_observed", "crps_unobserved",
    "spread_skill",
)


class _MaskOperator:
    """The slice of the observation-operator API that ``compute_metrics`` uses.

    ``compute_metrics`` reads exactly one thing off the operator --
    ``obs_indices_on_grid``, the 0/1 grid mask marking observed points -- to split
    CRPS (and the KL point set) into observed and unobserved. Rebuilding that mask
    from the ``obs_indices`` stored in the state file reproduces the property
    verbatim (``LinearObservationOperator.obs_indices_on_grid`` is literally
    ``zeros(num_dofs)[obs_indices] = 1``), which keeps this script from having to
    reconstruct scenario configs and operators just to recover a mask that is
    already on disk.

    For the super-resolution scenarios the stored indices cover every grid point
    (block averaging observes all of them), so the mask is all-ones and the
    unobserved CRPS is NaN -- the same degenerate case the pipeline reports.
    """

    def __init__(self, obs_indices, shape) -> None:  # type: ignore[no-untyped-def]
        import torch

        C, H, W = shape
        flat = torch.zeros(C * H * W)
        idx = torch.as_tensor(obs_indices).reshape(-1).long()
        if idx.numel():
            flat[idx] = 1.0
        self._mask = flat.view(C, H, W)

    @property
    def obs_indices_on_grid(self):  # type: ignore[no-untyped-def]
        return self._mask


def _traj_of(path: Path) -> int:
    """Trajectory index from the ``traj<N>`` path component (-1 if absent)."""
    m = re.search(r"traj(\d+)", str(path))
    return int(m.group(1)) if m else -1


def _as_py(x):  # type: ignore[no-untyped-def]
    """0-d numpy array -> Python scalar/str."""
    return x.item() if getattr(x, "ndim", 1) == 0 else x


def compute_for_state(path: Path, *, len_field_history: int | None):  # type: ignore[no-untyped-def]
    """Recompute the full metric set for one saved reference ensemble.

    Returns ``(rows, info)``: the tidy rows for this state file and a short
    dict describing what was scored (for the progress log).
    """
    import numpy as np
    import torch

    from cases.navier_stokes import _ns_pipeline

    data = np.load(path, allow_pickle=True)
    post = torch.from_numpy(data["posterior_trajectory"]).float()  # [E,C,H,W,T]
    true = torch.from_numpy(data["true_trajectory"]).float()       # [1,C,H,W,T]
    E, C, H, W, T = post.shape

    # Steps scored = T - len_field_history: the run seeds the first
    # ``len_field_history`` slots with the field history and only assimilates
    # after them. The state file records the per-step curves it scored, so their
    # length recovers the window exactly; --len-field-history is the fallback for
    # a state file saved without them.
    stored_steps = data["per_step_rmse"].shape[0] if "per_step_rmse" in data else None
    if len_field_history is not None:
        lfh = len_field_history
    elif stored_steps is not None:
        lfh = T - int(stored_steps)
    else:
        raise SystemExit(
            f"{path.name}: no per_step_rmse to infer the history length from -- "
            f"pass --len-field-history explicitly"
        )

    operator = _MaskOperator(data["obs_indices"], (C, H, W))
    result = _ns_pipeline.AssimResult(
        posterior_trajectory=post,
        true_trajectory=true,
        nfe_per_step=float(data["nfe_per_step"]),
        seconds_per_step=float(data["seconds_per_step"]),
    )
    metrics = _ns_pipeline.compute_metrics(
        result,
        operator,  # type: ignore[arg-type]
        len_field_history=lfh,
        # No reference for the reference: KL would be a self-comparison (~0).
        reference_trajectory=None,
    )

    scenario = str(_as_py(data["scenario"]))
    seconds = float(data["seconds_per_step"])
    nfe = float(data["nfe_per_step"])
    common = {
        "case": CASE,
        "method": REF_METHOD,
        "scenario": scenario,
        # std is the ACROSS-TRAJECTORY spread and is formed by the aggregation;
        # a single state file is one trajectory and carries none of its own.
        "std": "",
        "E": int(_as_py(data["E"])),
        "M": int(_as_py(data["M"])),
        "seed": int(_as_py(data["seed"])),
        "NFE": nfe,
        "seconds": seconds,
        "variant": "",
        "test_index": _traj_of(path),
    }
    rows = [
        {**common, "metric": name, "value": float(metrics[name])}
        for name in SCALAR_METRICS
    ]
    # Cost as its own metric rows, like every other cell: this is what the LaTeX
    # cost column reads.
    rows.append({**common, "metric": "nfe", "value": nfe})
    rows.append({**common, "metric": "seconds", "value": seconds})
    info = {"scenario": scenario, "E": E, "steps": T - lfh, "traj": common["test_index"]}
    return rows, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--states-root", type=Path, default=DEFAULT_STATES_ROOT,
        help=f"directory holding the reference .npz states, searched recursively "
             f"(default: {DEFAULT_STATES_ROOT})",
    )
    ap.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"tidy metrics CSV to write (default: {DEFAULT_OUT})",
    )
    ap.add_argument(
        "--len-field-history", type=int, default=None,
        help="number of seeded history steps to skip before scoring; default is "
             "inferred per file from its stored per-step curves",
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="only process the first N state files (smoke test)",
    )
    args = ap.parse_args()

    states = sorted(args.states_root.rglob("*.npz"))
    if args.limit:
        states = states[: args.limit]
    if not states:
        raise SystemExit(f"no .npz state files under {args.states_root}")
    print(f"[ns-ref] {len(states)} state files under {args.states_root}")

    rows: list[dict[str, object]] = []
    for i, path in enumerate(states, 1):
        t0 = time.time()
        new, info = compute_for_state(path, len_field_history=args.len_field_history)
        rows.extend(new)
        print(f"[ns-ref] {i}/{len(states)} {path.name}: traj{info['traj']} "
              f"{info['scenario']!r} E={info['E']} over {info['steps']} steps "
              f"({time.time() - t0:.1f}s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(FIELDNAMES))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in FIELDNAMES})
    trajs = sorted({r["test_index"] for r in rows})
    scens = sorted({r["scenario"] for r in rows})
    print(f"[ns-ref] wrote {len(rows)} rows -> {args.out}")
    print(f"[ns-ref]   method={REF_METHOD!r} trajectories={trajs} scenarios={scens}")
    print("[ns-ref] now re-run: .venv/bin/python paper_experiments/aggregate_ns.py")


if __name__ == "__main__":
    main()
