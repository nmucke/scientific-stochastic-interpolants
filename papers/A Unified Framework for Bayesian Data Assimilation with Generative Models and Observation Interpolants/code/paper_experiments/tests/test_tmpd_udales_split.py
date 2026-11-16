"""uDALES TMPD cells split into ensemble parts and merged (slurm_tmpd/udales_split).

CPU tests on the tiny random-weight priors of test_forecast_correction_ns_urban.
Parts are written by the REAL urban driver code (``_seed_ensemble_part``,
``_save_states``, ``_save_per_step``, ``_metric_rows`` + ``write``), so the merge
is tested on the files production writes.
"""

from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_forecast_correction_ns_urban import G, L, _building_mask, _prior, _truth, _urban_op  # noqa: E402
from cases.navier_stokes._ns_pipeline import attach_nfe_counter, build_posterior, run_assimilation  # noqa: E402
from cases.urban import _urban_pipeline  # noqa: E402
from cases.urban.driver import URBAN_METRICS, UrbanRunner  # noqa: E402
from common.aggregation import aggregate_over_seeds  # noqa: E402
from common.runner import RunContext  # noqa: E402
from results_schema import Case, Method, Scenario  # noqa: E402

torch.set_num_threads(2)
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "slurm_tmpd" / "udales_split"))
import merge  # noqa: E402

SCEN = "sparse 1.5625%"
STEM = "sparse_1_5625__M3__traj1__tmpd"
M, T = 3, L + 3


def _setup():  # type: ignore[no-untyped-def]
    prior = _prior(4, [0.4] * 4)
    keep = _building_mask()
    prior.test_dataset.mask = keep.unsqueeze(0).float()
    op = _urban_op(keep.unsqueeze(0).expand(4, G, G))
    return prior, op, _truth(op, 4, T)


def _runner(extra: dict) -> UrbanRunner:
    cfg = OmegaConf.create({
        "case": OmegaConf.load(EXP / "configs" / "case" / "urban.yaml"),
        "urban_scenarios": [SCEN], **extra,
    })
    return UrbanRunner(cfg, seeds=[0])


def _ctx(E: int) -> RunContext:
    return RunContext(case=Case.URBAN, method=Method.TMPD, scenario=Scenario(SCEN), seed=0,
                      ensemble_size=E, num_steps=M, extra={"test_index": 1})


def _make_part(root: Path, p: int, P: int, setup, E: int = 2, diverged: bool = False,
               truth=None) -> Path:  # type: ignore[no-untyped-def]
    """One part exactly as the driver writes it (metrics, per-step, states)."""
    prior, op, tr = setup
    tr = truth or tr
    d = root / "parts" / STEM / f"part{p}of{P}"
    (d / "states").mkdir(parents=True, exist_ok=True)
    runner = _runner({"ensemble_part": p, "ensemble_parts": P, "states_root": str(d / "states"),
                      "per_step_file": str(d / "per_step.csv"), "assimilate_every": 2})
    runner._prior = prior
    ctx = _ctx(E)
    runner._part_seed = runner._seed_ensemble_part(ctx)
    model, post, stepper = build_posterior(
        method_name="TMPD", method_cfg=OmegaConf.load(EXP / "configs" / "method" / "tmpd.yaml"),
        prior=prior, obs_operator=op, variance=0.0025, likelihood_ensemble_size=4,
        case_key="urban", scenario_key="sparse_1p5", num_steps=M)
    res = run_assimilation(posterior=post, model=model, truth_obs=tr, ensemble_size=E,
                           num_steps=M, num_physical_steps=T, stepper=stepper,
                           nfe_counter=attach_nfe_counter(model), gaussian_base=True,
                           assimilate_every=2)
    res.diverged = diverged
    keep = _urban_pipeline.fluid_keep_mask(prior, 4, "cpu")
    metrics = _urban_pipeline.compute_metrics(res, op, keep, L)
    runner.write(aggregate_over_seeds(list(runner._metric_rows(ctx, metrics))), d / "metrics.csv")
    runner._save_per_step(ctx, metrics)
    runner._save_states(ctx, res, tr, op, metrics=metrics)
    return d


def _csv(path: Path) -> dict[str, float]:
    with open(path, newline="") as fh:
        return {r["metric"]: float(r["value"]) for r in csv.DictReader(fh)}


def _eq(a: dict, b: dict, rtol: float = 1e-12) -> None:
    assert a.keys() == b.keys()
    for k in a:
        assert (math.isnan(a[k]) and math.isnan(b[k])) or a[k] == pytest.approx(b[k], rel=rtol), k


# --------------------------------------------------------------------------- #


def test_part_seed_scheme():
    ctx = _ctx(16)
    assert _runner({})._seed_ensemble_part(ctx) is None  # default: RNG untouched
    seeds = {p: _runner({"ensemble_part": p, "ensemble_parts": 4})._seed_ensemble_part(ctx)
             for p in range(4)}
    assert len(set(seeds.values())) == 4
    assert seeds[0] == _runner({"ensemble_part": 0, "ensemble_parts": 4})._seed_ensemble_part(ctx)
    torch.manual_seed(0)
    a = torch.randn(3)
    _runner({"ensemble_part": 1, "ensemble_parts": 4})._seed_ensemble_part(ctx)
    b = torch.randn(3)
    _runner({"ensemble_part": 1, "ensemble_parts": 4})._seed_ensemble_part(ctx)
    assert torch.equal(b, torch.randn(3)) and not torch.equal(a, b)
    other = RunContext(case=Case.URBAN, method=Method.TMPD, scenario=Scenario(SCEN), seed=0,
                       ensemble_size=16, num_steps=M, extra={"test_index": 2})
    assert _runner({"ensemble_part": 0, "ensemble_parts": 4})._seed_ensemble_part(other) != seeds[0]
    with pytest.raises(ValueError):
        _runner({"ensemble_part": 4, "ensemble_parts": 4})._seed_ensemble_part(ctx)


def test_one_part_merge_reproduces_the_run(tmp_path):
    d = _make_part(tmp_path, 0, 1, _setup())
    assert merge.merge_cell(tmp_path / "parts" / STEM, 1, tmp_path) == "merged"
    _eq(_csv(tmp_path / "metrics" / f"{STEM}.csv"), _csv(d / "metrics.csv"))
    assert (tmp_path / "per_step" / f"{STEM}.csv").read_text() == (d / "per_step.csv").read_text()


def test_parts_merge_to_the_metrics_of_the_concatenated_states(tmp_path):
    setup = _setup()
    d0, d1 = (_make_part(tmp_path, p, 2, setup) for p in range(2))
    s0, s1 = (np.load(next((d / "states").glob("*.npz"))) for d in (d0, d1))
    assert int(s0["part_seed"]) != int(s1["part_seed"])
    assert not np.array_equal(s0["posterior_trajectory"], s1["posterior_trajectory"])
    assert merge.merge_cell(tmp_path / "parts" / STEM, 2, tmp_path, save_states_traj=1) == "merged"
    got = _csv(tmp_path / "metrics" / f"{STEM}.csv")
    post = np.concatenate([s0["posterior_trajectory"], s1["posterior_trajectory"]])
    want = merge.metrics_from_states(
        post, dict(s0), float(s0["nfe_per_step"]),
        float(s0["seconds_per_step"]) + float(s1["seconds_per_step"]),
        float(s0["vjp_per_step"]), False)
    for k, v in got.items():
        assert v == pytest.approx(want[k], rel=1e-12), k
    assert got["seconds"] == pytest.approx(float(s0["seconds_per_step"]) + float(s1["seconds_per_step"]))
    assert got["vjp"] == float(s0["vjp_per_step"]) and got["nfe"] == float(s0["nfe_per_step"])
    with open(tmp_path / "metrics" / f"{STEM}.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["E"] for r in rows} == {"4"} and {r["method"] for r in rows} == {"TMPD"}
    merged = np.load(next((tmp_path / "states" / "traj1").glob("*__E4_M3.npz")))
    assert merged["posterior_trajectory"].shape[0] == 4 and int(merged["E"]) == 4
    # Already merged: skipped unless forced.
    assert merge.merge_cell(tmp_path / "parts" / STEM, 2, tmp_path) == "exists"


def test_a_diverged_part_marks_the_cell_diverged(tmp_path):
    setup = _setup()
    _make_part(tmp_path, 0, 2, setup)
    _make_part(tmp_path, 1, 2, setup, diverged=True)  # guard flag only: states finite
    assert merge.merge_cell(tmp_path / "parts" / STEM, 2, tmp_path) == "merged (diverged)"
    got = _csv(tmp_path / "metrics" / f"{STEM}.csv")
    assert all(math.isnan(got[m.value]) for m in URBAN_METRICS)
    assert all(math.isfinite(got[k]) for k in ("nfe", "seconds", "vjp"))
    import aggregate_ns

    assert aggregate_ns._is_diverged(tmp_path / "metrics" / f"{STEM}.csv")[0]


def test_incomplete_inconsistent_and_tampered_parts_are_refused(tmp_path):
    setup = _setup()
    _make_part(tmp_path, 0, 2, setup)
    assert merge.merge_cell(tmp_path / "parts" / STEM, 2, tmp_path) == "incomplete"
    prior, op, tr = setup
    other = _truth(op, 4, T, seed=5)
    d1 = _make_part(tmp_path, 1, 2, setup, truth=other)
    with pytest.raises(ValueError, match="disagree"):
        merge.merge_cell(tmp_path / "parts" / STEM, 2, tmp_path)
    import shutil

    shutil.rmtree(d1)  # the writers append; a rerun starts from a fresh file
    d1 = _make_part(tmp_path, 1, 2, setup)
    text = (d1 / "metrics.csv").read_text().splitlines()
    i = next(j for j, l in enumerate(text) if ",rmse," in l)
    f = text[i].split(",")
    f[4] = str(float(f[4]) + 0.1)
    text[i] = ",".join(f)
    (d1 / "metrics.csv").write_text("\n".join(text) + "\n")
    with pytest.raises(ValueError, match="do not reproduce"):
        merge.merge_cell(tmp_path / "parts" / STEM, 2, tmp_path)
    assert not (tmp_path / "metrics" / f"{STEM}.csv").exists()


def test_delete_part_states_only_after_merge(tmp_path):
    setup = _setup()
    dirs = [_make_part(tmp_path, p, 2, setup) for p in range(2)]
    assert merge.merge_cell(tmp_path / "parts" / STEM, 2, tmp_path, delete_part_states=True) == "merged"
    for d in dirs:
        assert not list((d / "states").glob("*.npz"))
        assert (d / "metrics.csv").exists() and (d / "per_step.csv").exists()
    assert (tmp_path / "metrics" / f"{STEM}.csv").exists()


def test_manifest_splits_udales_cells(tmp_path):
    out = subprocess.run(
        [sys.executable, str(EXP / "slurm_tmpd" / "make_manifest.py"), "udales", str(tmp_path / "r"),
         str(tmp_path / "m"), "--parts", "4", "--tmpd-batch", "8"],
        check=True, capture_output=True, text=True).stdout
    assert "40 cells" in out and "parts=4 x 16" in out
    lines = (tmp_path / "m" / "cells.tsv").read_text().splitlines()
    assert len(lines) == 40 and len({l.split("\t")[2] for l in lines}) == 40
    f = lines[1].split("\t")
    assert f[2].endswith("parts/sparse_1_5625__M50__traj1__tmpd/part1of4/metrics.csv")
    for a in ("ensemble_size=16", "+ensemble_part=1", "+ensemble_parts=4", "+save_states=true",
              "+posterior_batch_size=8", "+ic_index=8", "case.num_physical_steps=150",
              "+assimilate_every=3", "+assimilate_from=5", "num_steps=50"):
        assert a in f, a
    bad = subprocess.run(
        [sys.executable, str(EXP / "slurm_tmpd" / "make_manifest.py"), "ns_wrong_ic",
         str(tmp_path / "r"), str(tmp_path / "m2"), "--parts", "4"], capture_output=True, text=True)
    assert bad.returncode != 0
