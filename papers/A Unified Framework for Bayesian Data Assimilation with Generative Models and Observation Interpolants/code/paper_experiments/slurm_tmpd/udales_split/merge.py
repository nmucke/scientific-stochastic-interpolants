#!/usr/bin/env python3
"""Merge the ensemble parts of uDALES TMPD cells into standard per-cell files.

A cell (scenario, trajectory) of E members runs as P independent parts of E/P
members each (``+ensemble_part=p +ensemble_parts=P``; TMPD updates every member on
its own given the observations, see TMPD.md). Each part writes, under
``<root>/parts/<stem>/part<p>of<P>/``, its own ``metrics.csv`` / ``per_step.csv``
(published only on success by run_task.sh) and its full states
``states/*.npz`` (every physical step of every member, plus the truth, the fluid
mask, the sensor grid, cost and the guard flag; urban driver ``_save_states``).

For every cell whose P parts are all done, this script

1. VERIFIES each part: its metrics recomputed from its states with
   ``_urban_pipeline.compute_metrics`` (the pipeline's own function) equal the
   metrics the run itself wrote (rtol ``--rtol``), and the parts agree on truth,
   observations, sensors, settings, and have distinct sampler seeds;
2. concatenates the members and recomputes every metric and per-step curve with
   the same function, writing ``<root>/metrics/<stem>.csv`` and
   ``<root>/per_step/<stem>.csv`` exactly as run.py + the urban driver do
   (``UrbanRunner._metric_rows`` -> ``aggregate_over_seeds`` -> ``ResultsWriter``;
   ``per_step_rows``), so aggregate_urban.py reads them like any other cell;
3. optionally writes the merged states (``--save-states-traj``) and, only after
   1-2 succeeded, deletes the PART state files (``--delete-part-states``; nothing
   else is ever deleted).

Cost columns of the merged cell: ``nfe`` and ``vjp`` are per member-batch pass
(the pipeline's normalisation), identical in every part, so they are the parts'
mean; ``seconds`` is the SUM over parts, i.e. the per-step cost of the whole
E-member ensemble on one GPU slice (parts run back to back).

Divergence: the divergence guard acts per part. The merged cell is DIVERGED if
any part's guard tripped (``diverged`` flag in its states; this also catches a
trip on the last step, which leaves no NaN padding) or any state is non-finite.
Then every field metric of the cell (rmse, crps, spread_skill and their splits)
is written as NaN, keeping nfe / seconds / vjp: the convention of
aggregate_ns._is_diverged ("rmse NaN or absent" -> dropped and counted as
div k/n), and aggregate_urban's nanmean then leaves the cell out of every
accuracy column. The per-step curves are kept as computed (finite up to the trip).

    python merge.py <root> [--parts 4] [--save-states-traj 1] [--delete-part-states]
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PE = Path(__file__).resolve().parents[2]
if str(PE) not in sys.path:
    sys.path.insert(0, str(PE))

# Must agree across parts (else they are not parts of one cell).
SHARED_FIELDS = ("true_trajectory", "observations", "obs_indices", "fluid_mask",
                 "obs_indices_on_grid", "method", "scenario", "test_index", "seed",
                 "M", "assimilate_every", "assimilate_from", "ic_index",
                 "len_field_history", "ensemble_parts", "E")
PART_RE = re.compile(r"part(\d+)of(\d+)$")


@dataclass
class Part:
    index: int
    count: int
    dir: Path
    states: Path
    data: dict


def part_dirs(cell_dir: Path, parts: int) -> list[Path]:
    return [cell_dir / f"part{p}of{parts}" for p in range(parts)]


def load_part(d: Path) -> Part:
    m = PART_RE.search(d.name)
    if m is None:
        raise ValueError(f"not a part directory: {d}")
    files = sorted((d / "states").glob("*.npz"))
    if len(files) != 1:
        raise ValueError(f"{d}: expected 1 states file, found {len(files)}")
    with np.load(files[0], allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    return Part(int(m.group(1)), int(m.group(2)), d, files[0], data)


def _scalar(x):  # type: ignore[no-untyped-def]
    x = np.asarray(x)
    return x.item() if x.ndim == 0 else x


def assim_result(post: np.ndarray, data: dict, nfe: float, seconds: float,
                 vjp: float | None, diverged: bool):  # type: ignore[no-untyped-def]
    import torch

    from cases.navier_stokes._ns_pipeline import AssimResult

    return AssimResult(
        posterior_trajectory=torch.from_numpy(np.ascontiguousarray(post)).float(),
        true_trajectory=torch.from_numpy(np.ascontiguousarray(data["true_trajectory"])).float(),
        nfe_per_step=float(nfe), seconds_per_step=float(seconds), diverged=bool(diverged),
        vjp_per_step=None if vjp is None or math.isnan(vjp) else float(vjp),
    )


def metrics_from_states(post: np.ndarray, data: dict, nfe: float, seconds: float,
                        vjp: float | None, diverged: bool) -> dict:
    """The pipeline's compute_metrics on saved states (no data or model needed)."""
    import torch

    from cases.urban._urban_pipeline import compute_metrics

    op = SimpleNamespace(obs_indices_on_grid=torch.from_numpy(data["obs_indices_on_grid"]))
    return compute_metrics(
        assim_result(post, data, nfe, seconds, vjp, diverged), op,
        torch.from_numpy(data["fluid_mask"]), int(data["len_field_history"]))


def read_metrics_csv(path: Path) -> dict[str, float]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["metric"]: float(r["value"]) for r in csv.DictReader(fh)}


def _close(a: float, b: float, rtol: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-12)


def verify_part(part: Part, rtol: float) -> dict:
    """Recompute the part's metrics from its states; they must equal its own CSV."""
    d = part.data
    vjp = float(d["vjp_per_step"]) if "vjp_per_step" in d else None
    got = metrics_from_states(d["posterior_trajectory"], d, float(d["nfe_per_step"]),
                              float(d["seconds_per_step"]), vjp, bool(int(d["diverged"])))
    want = read_metrics_csv(part.dir / "metrics.csv")
    bad = [k for k, v in want.items() if k in got and not isinstance(got[k], dict)
           and not _close(float(got[k]), v, rtol)]
    if bad:
        raise ValueError(f"{part.dir}: states do not reproduce its metrics for {bad}: "
                         + ", ".join(f"{k} {got[k]} vs {want[k]}" for k in bad))
    if int(_scalar(d["ensemble_part"])) != part.index or int(_scalar(d["ensemble_parts"])) != part.count:
        raise ValueError(f"{part.dir}: states say part {d['ensemble_part']}/{d['ensemble_parts']}")
    return got


def check_consistent(parts: list[Part]) -> None:
    ref = parts[0].data
    for p in parts[1:]:
        for k in SHARED_FIELDS:
            a, b = ref.get(k), p.data.get(k)
            if a is None or b is None or not np.array_equal(np.asarray(a), np.asarray(b)):
                raise ValueError(f"parts disagree on {k!r}: {parts[0].dir} vs {p.dir}")
    seeds = [int(p.data["part_seed"]) for p in parts]
    if len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError(f"part sampler seeds not distinct/set: {seeds}")


def merge_cell(cell_dir: Path, parts: int, root: Path, rtol: float = 1e-4,
               save_states_traj: int | None = None, delete_part_states: bool = False,
               force: bool = False) -> str:
    """Merge one cell; returns a status word (merged / exists / incomplete)."""
    from cases.urban.driver import URBAN_METRICS, UrbanRunner
    from common.aggregation import aggregate_over_seeds
    from common.per_step_io import append_per_step, per_step_rows
    from common.runner import RunContext
    from omegaconf import OmegaConf
    from results_schema import Case, Method, Scenario

    stem = cell_dir.name
    out = root / "metrics" / f"{stem}.csv"
    ps_out = root / "per_step" / f"{stem}.csv"
    if out.exists() and not force:
        return "exists"
    dirs = part_dirs(cell_dir, parts)
    if not all((d / "metrics.csv").is_file() for d in dirs):
        return "incomplete"
    loaded = [load_part(d) for d in dirs]
    for p in loaded:
        verify_part(p, rtol)
    check_consistent(loaded)

    ref = loaded[0].data
    post = np.concatenate([p.data["posterior_trajectory"] for p in loaded], axis=0)
    E = post.shape[0]
    if E != int(ref["E"]) * parts:
        raise ValueError(f"{stem}: {E} members, expected {int(ref['E'])} x {parts}")
    diverged = any(bool(int(p.data["diverged"])) for p in loaded) or not np.isfinite(post).all()
    nfe = float(np.mean([float(p.data["nfe_per_step"]) for p in loaded]))
    seconds = float(np.sum([float(p.data["seconds_per_step"]) for p in loaded]))
    vjps = [float(p.data["vjp_per_step"]) for p in loaded if "vjp_per_step" in p.data]
    vjp = float(np.mean(vjps)) if vjps else None
    metrics = metrics_from_states(post, ref, nfe, seconds, vjp, diverged)
    if diverged:
        for m in URBAN_METRICS:
            metrics[m.value] = float("nan")

    method, scenario = str(_scalar(ref["method"])), str(_scalar(ref["scenario"]))
    seed, test_index, M = int(_scalar(ref["seed"])), int(_scalar(ref["test_index"])), int(_scalar(ref["M"]))
    runner = UrbanRunner(OmegaConf.create({"case": OmegaConf.load(
        PE / "configs" / "case" / "urban.yaml")}), seeds=[seed])
    ctx = RunContext(case=Case.URBAN, method=Method(method), scenario=Scenario(scenario),
                     seed=seed, ensemble_size=E, num_steps=M, extra={"test_index": test_index})
    tmp_dir = root / ".tmp" / "merge"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp, ps_tmp = tmp_dir / f"{stem}.metrics.csv", tmp_dir / f"{stem}.per_step.csv"
    for f in (tmp, ps_tmp):
        f.unlink(missing_ok=True)
    runner.write(aggregate_over_seeds(list(runner._metric_rows(ctx, metrics))), tmp)
    append_per_step(ps_tmp, per_step_rows(
        case=Case.URBAN.value, method=method, scenario=scenario, variant=runner._variant(ctx.method),
        E=E, M=M, seed=seed, test_index=test_index, nfe=metrics.get("nfe"),
        seconds=metrics.get("seconds"), per_step=metrics["per_step"]))
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "per_step").mkdir(parents=True, exist_ok=True)

    if save_states_traj is not None and test_index == save_states_traj:
        sdir = root / "states" / f"traj{test_index}"
        sdir.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"__E\d+_", f"__E{E}_", loaded[0].states.name)
        keep = {k: v for k, v in ref.items() if k not in (
            "posterior_trajectory", "ensemble_part", "part_seed", "E", "diverged",
            "nfe_per_step", "seconds_per_step", "seconds_total", "vjp_per_step")
            and not k.startswith("per_step_")}
        np.savez_compressed(
            sdir / name, posterior_trajectory=post.astype(np.float32), E=E, diverged=int(diverged),
            nfe_per_step=nfe, seconds_per_step=seconds,
            vjp_per_step=float("nan") if vjp is None else vjp,
            part_seeds=np.array([int(p.data["part_seed"]) for p in loaded]),
            **{f"per_step_{k}": np.asarray(v, dtype=np.float64) for k, v in metrics["per_step"].items()},
            **keep)
    ps_tmp.replace(ps_out)
    tmp.replace(out)
    if delete_part_states:
        for p in loaded:
            p.states.unlink()
    return "merged (diverged)" if diverged else "merged"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--parts", type=int, default=4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--save-states-traj", type=int, default=None)
    ap.add_argument("--delete-part-states", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-merge cells already merged")
    ap.add_argument("--cell", action="append", help="only these cell stems")
    a = ap.parse_args()
    cells = sorted(d for d in (a.root / "parts").glob("*") if d.is_dir())
    if a.cell:
        cells = [c for c in cells if c.name in set(a.cell)]
    fail = 0
    for c in cells:
        try:
            status = merge_cell(c, a.parts, a.root, a.rtol, a.save_states_traj,
                                a.delete_part_states, a.force)
        except Exception as exc:  # report and continue with the other cells
            status, fail = f"FAILED: {exc}", 1
        print(f"{c.name}: {status}", flush=True)
    sys.exit(fail)


if __name__ == "__main__":
    main()
