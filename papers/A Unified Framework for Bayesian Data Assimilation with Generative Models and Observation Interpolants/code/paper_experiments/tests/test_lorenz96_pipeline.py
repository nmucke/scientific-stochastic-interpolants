"""The Lorenz 96 pipeline on synthetic files: aggregation, figures, tables, tuning
selection and apply, and dry runs of run_lorenz96.sh with a stub interpreter.

Nothing here runs a cell: the result files are written by hand in the formats the
driver writes (tidy metrics CSV, per-step CSV, per-sequence JSON, tuning CSV).
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PE))

import aggregate_lorenz96  # noqa: E402
import make_lorenz96_figures  # noqa: E402
import make_lorenz96_tables  # noqa: E402
import select_lorenz96_tuning  # noqa: E402
from common.per_step_io import append_per_step, per_step_rows  # noqa: E402
from lorenz96_lineup import M_INDEPENDENT, M_VALUES, ROWS, SCENARIO_OF, SMC_METRICS  # noqa: E402
from results_schema import ResultRecord, ResultsWriter  # noqa: E402

ROW = {row.cell: row for row in ROWS}
BASE_METRICS = (
    "mean_error_vs_pf", "var_error_vs_pf", "sliced_w2", "kl_knn", "energy_distance",
    "rmse", "spread", "crps", "diverged_fraction",
)
SEQUENCE_ONLY = ("innovation_lag1", "innovation_msq")
CELLS = ("pf_reference", "pf_subsample", "enkf", "oracle", "si_sde", "si_sde_fc",
         "surge_si_sde", "pf_matched", "flowdas", "surge_sda")
STEPS = 4


def write_cell(root: Path, obs: str, cell: str, seed: int, offset: float = 0.0,
               m: int | None = None) -> None:
    """One cell's metrics, per-step and per-sequence files, as driver.py cell writes them.

    A generative cell at step count ``m`` gets the ``__M<m>__`` token and the tidy M;
    its values fall with M (1/M), so the M = 50 row differs from the others.
    """
    row = ROW[cell]
    smc = row.smc
    common = dict(
        case="lorenz96", method=row.method, scenario=SCENARIO_OF[obs], E=64, M=m,
        seed=seed, nfe=2.0 * m if m else None, seconds=0.5, variant=row.variant or None,
    )
    level = 1.0 + 0.1 * CELLS.index(cell) + 0.01 * seed + offset + (10.0 / m if m else 0.0)
    per_step = {m: [level + 0.01 * s for s in range(STEPS)] for m in BASE_METRICS}
    if smc:
        per_step.update({m: [0.5 + 0.01 * s for s in range(STEPS)] for m in SMC_METRICS})
    per_sequence = {m: [sum(v) / STEPS] * 2 for m, v in per_step.items()}
    if cell != "pf_reference":
        per_sequence.update({m: [0.1 * level, 0.1 * level] for m in SEQUENCE_ONLY})
    label = f"lorenz96__{cell}__M{m}__traj{seed}__" if m else f"lorenz96__{cell}__traj{seed}__"
    path = root / "per_step" / f"{label}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    append_per_step(path, per_step_rows(test_index=-1, per_step=per_step, **common))
    seq = root / "per_sequence" / f"{label}.json"
    seq.parent.mkdir(parents=True, exist_ok=True)
    seq.write_text(json.dumps(per_sequence))
    records = [ResultRecord(metric=m, value=v[0], **common) for m, v in per_sequence.items()]
    records.append(ResultRecord(metric="seconds", value=0.5, **common))
    if common["nfe"] is not None:
        records.append(ResultRecord(metric="nfe", value=common["nfe"], **common))
    records.append(ResultRecord(metric="diverged_steps", value=0.0, **common))
    out = root / "metrics" / f"{label}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with ResultsWriter(out) as writer:
        writer.extend(records)


@pytest.fixture
def sweeps(tmp_path, monkeypatch):
    results = tmp_path / "results"
    for module in (aggregate_lorenz96, make_lorenz96_figures, make_lorenz96_tables,
                   select_lorenz96_tuning):
        monkeypatch.setattr(module, "RESULTS", results)
    monkeypatch.setattr(make_lorenz96_tables, "GENERATED", tmp_path / "generated")
    for obs, offset in (("linear", 0.0), ("arctan", 0.5)):
        root = results / "lorenz96" / obs
        for cell in CELLS:
            for seed in ((0,) if cell == "pf_reference" else (0, 1, 2)):
                for m in (M_VALUES if ROW[cell].generative else (None,)):
                    write_cell(root, obs, cell, seed, offset, m)
    return results


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_sweeps_aggregate_plot_and_tabulate_apart(sweeps, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["aggregate_lorenz96.py"])
    aggregate_lorenz96.main()
    tables = {}
    for obs in ("linear", "arctan"):
        aggregated = sweeps / "lorenz96" / obs / "aggregated"
        rows = read_csv(aggregated / "all.csv")
        assert {r["scenario"] for r in rows} == {SCENARIO_OF[obs]}
        smc_methods = {(r["method"], r["variant"]) for r in rows if r["metric"] in SMC_METRICS}
        assert smc_methods == {
            (ROW[c].method, ROW[c].variant) for c in CELLS if ROW[c].smc
        }
        per_step = read_csv(aggregated / "per_step.csv")
        assert any(r["metric"] == "smc_ess_fraction" for r in per_step)
        reference = [r for r in rows if r["variant"] == "reference"]
        assert reference and all(math.isnan(float(r["std"])) for r in reference)
        # One row per M for a generative cell, none for an M-independent one.
        ms = {r["M"] for r in rows if r["method"] == "Ours (SI-SDE)" and r["variant"] == ""
              and r["metric"] == "kl_knn"}
        assert ms == {str(m) for m in M_VALUES}
        assert {r["M"] for r in rows if r["method"] == "EnKF"} == {""}
        cells = {r["cell"] for r in read_csv(aggregated / "sequence_spread.csv")}
        assert {"si_sde__M50", "si_sde__M250", "enkf"} <= cells

        out = tmp_path / "figures" / obs
        make_lorenz96_figures.apply_style()
        make_lorenz96_figures.make_sweep(obs, out)
        for name in ("rmse_vs_step", "smc_ess_fraction_vs_step", "kl_knn_vs_step",
                     "kl_knn_vs_M", "energy_distance_vs_M", "sliced_w2_vs_M",
                     "mean_error_vs_pf_vs_M"):
            assert (out / f"lorenz96_{name}.pdf").exists(), name
        # The per-step curves of a generative cell are those at M = 50.
        curve = make_lorenz96_figures.load_curves("rmse", obs)[("Ours (SI-SDE)", "")]
        level = 1.0 + 0.1 * CELLS.index("si_sde") + 0.01 + (0.5 if obs == "arctan" else 0) + 10 / 50
        assert curve[0] == pytest.approx(level)

        tex, md, tex_m, md_m = make_lorenz96_tables.make_sweep(obs)
        assert tex.name == f"tab_lorenz96_{obs}.tex" and md.exists()
        assert tex_m.name == f"tab_lorenz96_{obs}_vs_M.tex" and md_m.exists()
        vs_m = [l for l in tex_m.read_text().splitlines() if l.startswith(("Ours", "FlowDAS", "SURGE"))]
        assert len(vs_m) == sum(ROW[c].generative for c in CELLS)
        assert not any(l.startswith(("EnKF", "Particle", "One-step")) for l in tex_m.read_text().splitlines())
        tables[obs] = tex.read_text()
        # Main table: the M = 50 value of si_sde's mean error.
        si = next(l for l in tables[obs].splitlines() if l.startswith("Ours (SI-SDE) &  &"))
        assert float(si.split("&")[2].split()[0]) == pytest.approx(level + 0.015, abs=0.006)
    # The sweeps differ in values only: the arctan files are offset by 0.5.
    assert tables["linear"] != tables["arctan"]
    header = [
        [line for line in text.splitlines() if line.startswith("Method &")]
        for text in tables.values()
    ]
    assert header[0] == header[1] and len(header[0]) == 1
    ncols = header[0][0].count("&") + 1
    for text in tables.values():
        body = [
            line for line in text.splitlines()
            if line.endswith(r"\\") and not line.startswith((r"\multicolumn", " &", "Method"))
        ]
        assert body and all(line.count("&") + 1 == ncols for line in body)
        enkf = next(line for line in body if line.startswith("EnKF & matched E"))
        surge = next(line for line in body if line.startswith(r"Ours (SI-SDE) & + SURGE"))
        # SMC columns are the last two: blank for EnKF, filled for SURGE.
        assert not any(c.strip() for c in enkf.rstrip(r"\ ").split("&")[-2:])
        assert all(c.strip() for c in surge.rstrip(r"\ ").split("&")[-2:])
        # The reference is not bold; the lowest non-reference mean error is (EnKF).
        assert r"\textbf" in enkf.split("&")[2]
        assert r"\textbf" not in next(l for l in body if l.startswith("Particle filter & reference &"))
    assert "$\\pm$" in tables["linear"]


def test_a_file_of_the_other_sweep_is_refused(sweeps, tmp_path, monkeypatch):
    stray = tmp_path / "stray"
    write_cell(stray, "arctan", "enkf", 7)
    shutil.copy(
        stray / "metrics" / "lorenz96__enkf__traj7__.csv",
        sweeps / "lorenz96" / "linear" / "metrics",
    )
    with pytest.raises(ValueError, match="must not mix"):
        aggregate_lorenz96.aggregate_sweep("linear")
    assert aggregate_lorenz96.aggregate_sweep("arctan")  # the other sweep is unaffected


def test_unlisted_rows_are_left_out_of_the_table_with_a_note(sweeps, capsys):
    aggregate_lorenz96.aggregate_sweep("linear")
    path = sweeps / "lorenz96" / "linear" / "aggregated" / "all.csv"
    rows = read_csv(path)
    extra = dict(rows[0], variant="matched_sweep1")
    with open(path, "a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, list(rows[0])).writerow(extra)
    tex, md, *_ = make_lorenz96_tables.make_sweep("linear")
    assert "matched_sweep1" not in tex.read_text()
    assert "matched_sweep1" in md.read_text()
    assert "not in lorenz96_lineup.ROWS" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Tuning selection
# --------------------------------------------------------------------------- #

TUNING_TAIL = ("base", "seconds", "diverged_steps", "mean_error_vs_pf", "var_error_vs_pf",
               "sliced_w2", "rmse", "spread", "diverged_fraction")


def write_tuning(path: Path, grid_rows: list[tuple[dict, float, int]], base=None) -> None:
    """Rows as driver.py tune writes them: (override, sliced_w2, diverged_steps).

    ``base`` defaults to the cell's current base (its config block, no --set), so
    the file counts as up to date.
    """
    if base is None:
        obs, cell = path.parent.parent.name, path.name.split("__")[0]
        base = select_lorenz96_tuning.current_base(obs, cell)
    keys = list(grid_rows[0][0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, [*keys, *TUNING_TAIL])
        writer.writeheader()
        for override, sw2, diverged in grid_rows:
            score = math.inf if diverged else sw2
            writer.writerow(dict(
                override, base=json.dumps(base, sort_keys=True), seconds=1.0,
                diverged_steps=diverged, mean_error_vs_pf=score, var_error_vs_pf=score,
                sliced_w2=score, rmse=score, spread=1.0, diverged_fraction=0.0,
            ))


def tuning_dir(results: Path, obs: str) -> Path:
    return results / "lorenz96" / obs / "tuning"


def test_selection_picks_lowest_finite_sliced_w2_and_skips_diverged(sweeps):
    tdir = tuning_dir(sweeps, "linear")
    write_tuning(tdir / "enkf__inflation_localisation__s1.csv", [
        ({"inflation": 1.0, "localisation": None}, 0.50, 0),
        ({"inflation": 1.05, "localisation": 4}, 0.30, 0),
        ({"inflation": 1.2, "localisation": 8}, 0.10, 3),  # diverged: inf, ineligible
        ({"inflation": 1.1, "localisation": 8}, math.nan, 0),
        ({"inflation": 1.02, "localisation": 4}, 0.35, 0),
    ])
    # A diverged row that (wrongly) kept a finite score is still ineligible.
    path = tdir / "flowdas__guidance_scale__s1.csv"
    write_tuning(path, [({"guidance_scale": 0.25}, 0.4, 0), ({"guidance_scale": 1.0}, 0.2, 0)])
    rows = read_csv(path)
    rows[1]["diverged_steps"] = "2"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_tuning(tdir / "sda__gamma_sda__s1.csv", [({"gamma_sda": 1.0}, 0.1, 5)])
    # Another sweep's and another stage's files are not read.
    write_tuning(tuning_dir(sweeps, "arctan") / "enkf__inflation_localisation__s1.csv",
                 [({"inflation": 1.1, "localisation": 8}, 0.01, 0)])
    write_tuning(tdir / "enkf__inflation_localisation__s2.csv",
                 [({"inflation": 1.1, "localisation": 8}, 0.01, 0)])

    out = select_lorenz96_tuning.write_selection("linear", 1)
    picks = yaml.safe_load(out.read_text())
    assert picks["enkf"]["tuned"] == {"inflation": 1.05, "localisation": 4}
    assert picks["enkf"]["set"] == ["inflation=1.05", "localisation=4"]
    assert picks["enkf"]["full"] == {"inflation": 1.05, "localisation": 4}
    assert picks["enkf"]["runner_up_gap"] == pytest.approx(0.05)
    assert picks["flowdas"]["tuned"] == {"guidance_scale": 0.25}
    assert "sda" not in picks
    summary = out.with_suffix(".md").read_text()
    assert "No eligible row" in summary and "5 rows" in summary
    assert "methods:" in out.read_text()
    assert "methods_arctan:" in select_lorenz96_tuning.write_selection("arctan", 1).read_text()


def test_resolve_builds_stage2_grids_from_stage1_picks(sweeps):
    tdir = tuning_dir(sweeps, "linear")
    write_tuning(tdir / "flowdas__guidance_scale__s1.csv",
                 [({"guidance_scale": g}, abs(g - 0.064), 0) for g in (0.016, 0.064, 0.25)])
    write_tuning(tdir / "si_sde__likelihood_mode_jacobian_damping__s1.csv", [
        ({"likelihood_mode": "inflated", "jacobian_damping": 0.9}, 0.2, 0),
        ({"likelihood_mode": "dps_jacobian_free", "jacobian_damping": 1.0}, 0.3, 0),
    ])
    with pytest.raises(FileNotFoundError):
        select_lorenz96_tuning.resolve("linear", "flowdas", {"guidance_scale": {"times": [1]}})
    select_lorenz96_tuning.write_selection("linear", 1)
    grid, sets = select_lorenz96_tuning.resolve(
        "linear", "flowdas",
        {"guidance_scale": {"times": [0.25, 1, 4]}, "ess_threshold": [0.5, 1.0]},
    )
    assert grid["guidance_scale"] == pytest.approx([0.016, 0.064, 0.256])
    assert grid["ess_threshold"] == [0.5, 1.0] and sets == []
    grid, sets = select_lorenz96_tuning.resolve(
        "linear", "si_sde", {"correction.gain_scale": [0.5]}
    )
    assert sets == ["jacobian_damping=0.9", 'likelihood_mode="inflated"']
    with pytest.raises(ValueError, match="No stage-1 pick"):
        select_lorenz96_tuning.resolve("linear", "sda", {"x": [1]})
    with pytest.raises(ValueError, match="need a parent"):
        select_lorenz96_tuning.resolve("linear", "", {"x": {"times": [1]}})


def test_status_sees_complete_partial_and_missing_jobs(sweeps):
    grid = {"inflation": [1.0, 1.05], "localisation": [None, 4]}
    state, path = select_lorenz96_tuning.status("linear", "enkf", grid, 1)
    assert state == "missing" and path.name == "enkf__inflation_localisation__s1.csv"
    write_tuning(path, [({"inflation": 1.0, "localisation": None}, 0.1, 0)])
    assert select_lorenz96_tuning.status("linear", "enkf", grid, 1)[0] == "partial"
    write_tuning(path, [
        ({"inflation": i, "localisation": r}, 0.1, 0) for i in (1.0, 1.05) for r in (None, 4)
    ])
    assert select_lorenz96_tuning.status("linear", "enkf", grid, 1)[0] == "complete"


def test_status_treats_rows_of_another_base_as_partial(sweeps):
    """Stage 2 reruns when the stage-1 picks it got by --set (or the config) moved."""
    grid = {"correction.gain_scale": [0.5, 1.0]}
    sets = ["likelihood_mode=\"dps_jacobian_free\"", "jacobian_damping=0.9"]
    path = select_lorenz96_tuning.job_csv("linear", "si_sde_fc", grid, 2)
    base = select_lorenz96_tuning.current_base("linear", "si_sde_fc", sets)
    assert base["jacobian_damping"] == 0.9 and "correction" in base
    rows = [({"correction.gain_scale": g}, 0.1, 0) for g in (0.5, 1.0)]
    write_tuning(path, rows, base=base)
    assert select_lorenz96_tuning.status("linear", "si_sde_fc", grid, 2, sets)[0] == "complete"
    # Other stage-1 picks, or none: the rows are stale.
    other = ["likelihood_mode=\"inflated\"", "jacobian_damping=0.9"]
    assert select_lorenz96_tuning.status("linear", "si_sde_fc", grid, 2, other)[0] == "partial"
    assert select_lorenz96_tuning.status("linear", "si_sde_fc", grid, 2)[0] == "partial"
    # One stale row is enough.
    table = read_csv(path)
    table[1]["base"] = json.dumps(select_lorenz96_tuning.current_base("linear", "si_sde_fc", other))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    assert select_lorenz96_tuning.status("linear", "si_sde_fc", grid, 2, sets)[0] == "partial"


# --------------------------------------------------------------------------- #
# Shell script, dry run with a stub interpreter
# --------------------------------------------------------------------------- #

SCRIPTS = ("run_lorenz96.sh", "select_lorenz96_tuning.py", "lorenz96_lineup.py")
STUB = r"""#!/usr/bin/env bash
# Logs every call; runs the selection helper and the lineup for real; a `cell` call
# writes its metrics file unless its method is $FAIL (then it fails).
echo "$*" >> "$STUB_LOG"
if [[ "$1" == select_lorenz96_tuning.py || "$1" == lorenz96_lineup.py ]]; then
  exec "$REAL_PY" "$@"
fi
obs=linear; method=; seed=; sub=; m=
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[i]}" in
    --obs) obs=${args[i+1]} ;;
    --method) method=${args[i+1]} ;;
    --seed) seed=${args[i+1]} ;;
    --num-steps) m=${args[i+1]} ;;
    cell|tune|reference|prior-quality) [[ -z "$sub" ]] && sub=${args[i]} ;;
  esac
done
if [[ "$sub" == cell ]]; then
  [[ "$method" == "${FAIL:-}" ]] && exit 1
  mkdir -p "results/lorenz96/$obs/metrics"
  token=${m:+__M$m}
  echo x > "results/lorenz96/$obs/metrics/lorenz96__${method}${token}__traj${seed}__.csv"
fi
exit 0
"""
CONFIG = Path("configs/case/lorenz96.yaml")


@pytest.fixture
def dry(tmp_path):
    """A copy of the script, its helpers and the case config in a scratch directory,
    and a runner using the stub interpreter."""
    work = tmp_path / "pe"
    work.mkdir()
    for name in SCRIPTS:
        shutil.copy2(PE / name, work / name)
    (work / CONFIG).parent.mkdir(parents=True)
    shutil.copy2(PE / CONFIG, work / CONFIG)
    (work / "results").mkdir()
    stub = tmp_path / "stub_py"
    stub.write_text(STUB)
    stub.chmod(0o755)
    log = tmp_path / "calls.log"

    def run(*argv: str, **env):
        log.write_text("")
        full = dict(os.environ, PY=str(stub), STUB_LOG=str(log), REAL_PY=sys.executable,
                    PYTHONPATH=str(PE), **env)
        proc = subprocess.run(["bash", str(work / "run_lorenz96.sh"), *argv], env=full,
                              capture_output=True, text=True, cwd=tmp_path)
        return proc, log.read_text().splitlines()

    run.work = work
    return run


def cell_calls(calls):
    return [c for c in calls if " cell " in f" {c} "]


def _grids() -> list[list[str]]:
    text = (PE / "run_lorenz96.sh").read_text()
    body = text.split("GRIDS=(")[1].split("\n)")[0]
    return [line.strip().strip("'").split("|") for line in body.splitlines()
            if line.strip().startswith("'")]


def test_final_runs_every_cell_once_per_m_skips_done_and_stops_on_failure(dry):
    from lorenz96_lineup import run_cells

    gen, fixed = run_cells("generative"), run_cells("fixed")
    assert set(gen) | set(fixed) | {"pf_reference"} == {row.cell for row in ROWS}
    assert set(fixed) | {"pf_reference"} == set(M_INDEPENDENT)

    root = dry.work / "results/lorenz96/arctan"
    done = root / "metrics/lorenz96__si_sde__M50__traj0__.csv"
    done.parent.mkdir(parents=True)
    done.write_text("x")
    proc, calls = dry("final", "arctan", SEEDS="0 1")
    assert proc.returncode == 0, proc.stderr
    ran = cell_calls(calls)
    # Every generative cell at 4 M and 2 seeds, minus the done one; the fixed cells
    # once per seed; pf_reference once.
    assert len(ran) == 2 * (4 * len(gen) + len(fixed)) - 1 + 1
    assert not any("--method si_sde --seed 0 --num-steps 50" in c for c in ran)
    assert sum("--method si_sde --seed 0 --num-steps" in c for c in ran) == 3
    assert sum("pf_reference" in c for c in ran) == 1
    for cell in fixed + ["pf_reference"]:
        assert all("--num-steps" not in c for c in ran if f"--method {cell} " in c)
    # M = 50 first (with the fixed cells), then 25, 100, 250.
    order = [c.split("--num-steps ")[1] if "--num-steps" in c else "-" for c in ran[:-1]]
    first = 2 * (len(gen) + len(fixed)) - 1
    assert set(order[:first]) == {"50", "-"}
    assert order[first:] == ["25"] * 2 * len(gen) + ["100"] * 2 * len(gen) + ["250"] * 2 * len(gen)
    driver_calls = [c for c in calls if c.startswith("cases/")]
    assert all("--obs arctan" in c for c in driver_calls)
    assert calls[-3:] == [
        "aggregate_lorenz96.py --obs arctan",
        "make_lorenz96_figures.py --obs arctan",
        "make_lorenz96_tables.py --obs arctan",
    ]
    assert "reference --split test" in driver_calls[0]
    assert "prior-quality" in driver_calls[1]
    logs = sorted((root / "logs").glob("final_*.log"))
    assert logs and Path(f"{logs[-1]}.exit").read_text().strip() == "0"

    (root / "prior_quality.json").write_text("{}")
    proc, calls = dry("final", "arctan", SEEDS="0 1")
    assert proc.returncode == 0 and cell_calls(calls) == []
    assert not any("prior-quality" in c for c in calls)

    proc, calls = dry("final", "linear", SEEDS="0", M_VALUES="50", FAIL="fm_ode")
    assert proc.returncode != 0
    assert len(cell_calls(calls)) == len(gen) + len(fixed)  # the other cells still ran
    assert not any("aggregate" in c or "pf_reference" in c for c in calls)

    assert dry("final", "quadratic")[0].returncode == 2
    assert dry("grid", "linear")[0].returncode == 2


def test_tune_skips_complete_jobs_moves_partial_ones_and_never_tunes_m(dry):
    rows = _grids()
    stage1 = [r for r in rows if r[0] == "1"]
    assert all("num_steps" not in r[3] for r in rows)
    tdir = dry.work / "results/lorenz96/linear/tuning"
    # Every stage-1 job complete except enkf_large (partial).
    for _, cell, _, grid in stage1:
        grid = json.loads(grid)
        points = list(itertools.product(*grid.values()))
        write_tuning(tdir / f"{cell}__{'_'.join(grid)}__s1.csv",
                     [(dict(zip(grid, p)), 0.1 + 0.01 * i, 0)
                      for i, p in enumerate(points if cell != "enkf_large" else points[:1])])
    proc, calls = dry("tune", "linear")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "STOP" in proc.stdout
    assert calls[0] == "cases/lorenz96/driver.py --obs linear reference --split validation"
    tunes = [c for c in calls if " tune " in f" {c} "]
    methods = [c.split("--method ")[1].split()[0] for c in tunes]
    assert methods == ["enkf_large"] + [r[1] for r in rows if r[0] == "2"]
    assert list(tdir.glob("enkf_large__*.partial.*"))
    assert "skip linear stage 1 enkf" in proc.stdout
    assert all("--obs linear" in c and "num_steps" not in c for c in tunes)
    assert not cell_calls(calls)  # never the final grid
    assert (tdir / "selected_stage1.yaml").exists() and (tdir / "selected_stage2.md").exists()
    latest = sorted((dry.work / "results/lorenz96/linear/logs").glob("tune_*.log"))[-1]
    assert Path(f"{latest}.exit").read_text().strip() == "0"

    proc, calls = dry("tune", "linear", ONLY="nosuchcell")
    assert proc.returncode != 0 and "no tuning job matched" in proc.stdout  # the log has stderr


def test_tune_stage2_uses_the_stage1_picks(dry):
    tdir = dry.work / "results/lorenz96/arctan/tuning"
    proc, calls = dry("tune", "arctan")
    # The stub writes no tuning CSV: stage 2 cannot resolve its parents.
    assert proc.returncode != 0 and "selected_stage1.yaml" in proc.stdout + proc.stderr

    # Complete stage-1 files whose best rows are these picks.
    best = {"flowdas": {"guidance_scale": 0.064}, "sda": {"gamma_sda": 0.01},
            **{c: {"likelihood_mode": "inflated", "jacobian_damping": 0.95}
               for c in ("si_sde", "fm_ode", "dm_sde")}}
    for _, cell, _, grid in (r for r in _grids() if r[0] == "1"):
        grid = json.loads(grid)
        rows = [dict(zip(grid, p)) for p in itertools.product(*grid.values())]
        write_tuning(tdir / f"{cell}__{'_'.join(grid)}__s1.csv",
                     [(r, 0.1 if r == best.get(cell) else 0.5, 0) for r in rows])
    proc, calls = dry("tune", "arctan")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    tunes = {c.split("--method ")[1].split()[0]: c for c in calls if " tune " in f" {c} "
             and "--tag s2" in c}
    assert set(tunes) == {"si_sde_fc", "fm_ode_fc", "dm_sde_fc", "surge_si_sde", "surge_dm_sde",
                          "surge_unguided_si", "surge_unguided_dm", "surge_flowdas", "surge_sda"}
    for cell in ("si_sde_fc", "surge_si_sde"):
        assert "--set jacobian_damping=0.95 --set likelihood_mode=\"inflated\"" in tunes[cell]
    assert "--set" not in tunes["surge_unguided_si"] and "--set" not in tunes["surge_flowdas"]
    grid = json.loads(tunes["surge_flowdas"].split("--grid ")[1].split()[0])
    assert grid["guidance_scale"] == pytest.approx([0.064 * k for k in (0.004, 0.016, 0.064, 0.25, 1, 4)])
    grid = json.loads(tunes["surge_sda"].split("--grid ")[1].split()[0])
    assert grid["gamma_sda"] == pytest.approx([0.001, 0.01, 0.1])

    # A complete stage-2 CSV made with other stage-1 picks is stale: moved aside, rerun.
    fc_grid = json.loads(tunes["si_sde_fc"].split("--grid ")[1].split()[0])
    path = tdir / f"si_sde_fc__{'_'.join(fc_grid)}__s2.csv"
    points = [dict(zip(fc_grid, p)) for p in itertools.product(*fc_grid.values())]
    stale = select_lorenz96_tuning.current_base("arctan", "si_sde_fc")  # no stage-1 --set
    write_tuning(path, [(pt, 0.1, 0) for pt in points], base=stale)
    proc, calls = dry("tune", "arctan", ONLY="si_sde_fc")
    assert list(tdir.glob("si_sde_fc__*__s2.csv.partial.*"))
    assert sum("--method si_sde_fc" in c for c in calls if " tune " in f" {c} ") == 1


def test_apply_merges_the_picks_into_the_config_and_keeps_everything_else(dry):
    from omegaconf import OmegaConf

    tdir = dry.work / "results/lorenz96/arctan/tuning"
    config = dry.work / CONFIG
    original = config.read_text()
    picks1 = {"si_sde": {"tuned": {"likelihood_mode": "dps_jacobian_free",
                                   "jacobian_damping": 0.9}},
              "flowdas": {"tuned": {"guidance_scale": 0.016}}}
    # Values that differ from every block of the committed config (the tuned picks
    # included), so each of the 4 cells changes exactly one line.
    picks2 = {"si_sde_fc": {"tuned": {"correction": {"gain_scale": 0.3, "q_source": "online"}}},
              "surge_si_sde": {"tuned": {"ess_threshold": 0.25}}}
    tdir.mkdir(parents=True)
    (tdir / "selected_stage1.yaml").write_text(yaml.safe_dump(picks1))
    (tdir / "selected_stage2.yaml").write_text(yaml.safe_dump(picks2))
    proc, _ = dry("apply", "arctan")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    old, new = OmegaConf.load(PE / CONFIG), OmegaConf.load(config)
    old, new = OmegaConf.to_container(old), OmegaConf.to_container(new)
    block = new["methods_arctan"]
    assert block["si_sde"] == {**old["methods_arctan"]["si_sde"], **picks1["si_sde"]["tuned"]}
    assert block["flowdas"]["guidance_scale"] == 0.016
    # Nested: the correction block keeps its other keys (localisation, forecast steps).
    assert block["si_sde_fc"] == {"correction": {
        **old["methods_arctan"]["si_sde_fc"]["correction"], **picks2["si_sde_fc"]["tuned"]["correction"]}}
    assert block["surge_si_sde"] == {"ess_threshold": 0.25}
    # Nothing else moved: the linear block, every other cell, every comment line.
    assert new["methods"] == old["methods"]
    for key in old:
        if key != "methods_arctan":
            assert new[key] == old[key]
    changed = [(a, b) for a, b in zip(original.splitlines(), config.read_text().splitlines())
               if a != b]
    assert len(original.splitlines()) == len(config.read_text().splitlines())
    assert len(changed) == 4 and all(a.startswith("  ") and not a.lstrip().startswith("#")
                                     for a, _ in changed)
