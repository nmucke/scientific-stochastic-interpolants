"""ONE aggregate + ONE appendix table holding EVERY Navier--Stokes wrong-IC result.

Setting: foreign initial condition from test trajectory 15, truth trajectories
10--14, 40 physical steps (5 history + 35 assimilation), E=64, M=100.

Result groups (each its own tree; see ``GROUPS`` below), with the EXPECTED cells
spelled out as filenames so finished / pending counts come from ``ls``, never
from hard-coded text -- rerunning this script while arrays finish updates both
the aggregate and the table:

  main        results_wrong_ic_np40/            Ours x {shared_jac10, jacfree},
                                                generative baselines, EnKF, PF
  surge_ours  results_wrong_ic_np40_ext/        SURGE on Ours SI/DM-SDE
  fc_r8       results_wrong_ic_np40_ext/        forecast correction, radius 8
  fc_r20      results_wrong_ic_np40_ext/        forecast correction, radius 20
  prior_enkf  results_wrong_ic_np40_prior_enkf/ unguided Ours prior + EnKF analysis
  enkf_loc20  ENKF_LOC20_ROOT                   true-solver EnKF, r=20 on super-res

Outputs (none of the existing aggregated/, aggregated_fc/ or tables/ns_accuracy*
files is touched):

  results_wrong_ic_np40/navier_stokes/aggregated_all/
      all.csv       one row per (method, variant, scenario, metric) EXPECTED cell
      per_step.csv  per-step mean/std over non-diverged trajectories
      status.csv    expected / finished / diverged per group
      diverged.csv  every diverged trajectory-cell (excluded whole)
      READY         written last, after every file above is complete
  results_wrong_ic_np40/tables/ns_all_wrong_ic_scaled.{tex,md}

The prior + EnKF method names exist only in the isolated code copy's
``results_schema.py``; nothing here needs the Method enum -- method strings are
read straight from the per-trajectory CSVs.

    .venv/bin/python paper_experiments/aggregate_ns_all.py
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from common.aggregate_lib import (  # noqa: E402
    aggregate_per_step,
    aggregate_scalar,
    write_per_step_csv,
)
from common.latex_tables import (  # noqa: E402
    BOLD_UNDERLINE_SENTENCE,
    DIV_ALL,
    DIV_SOME,
    STD_SENTENCE,
    _best_in_column,
    _fmt_number,
    _second_in_column,
    std_suffix,
)

CASE = "navier_stokes"
SCRATCH_PE = Path("/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments")

# The production output root of the r=20 super-resolution EnKF (run_cpu.sh's
# <out_root>: it holds metrics/ and per_step/ directly, or under navier_stokes/).
# ONE constant -- point it at the real production subdir; overridable with
# --enkf-loc20-root or the ENKF_LOC20_ROOT environment variable.
ENKF_LOC20_ROOT = SCRATCH_PE / "enkf_loc20_superres" / "production"

FC_TAG_R8 = "fc25_70a3134d"
FC_TAG_R20 = "fc25_8797d2ae"
TRAJS = (10, 11, 12, 13, 14)
M = 100

# (tidy scenario key, filename slug, LaTeX column header)
SCENARIOS = (
    ("16^2->128^2", "16_2_128_2", r"$16^2$"),
    ("32^2->128^2", "32_2_128_2", r"$32^2$"),
    ("sparse 5%", "sparse_5", r"$5\%$"),
    ("sparse 1.5625%", "sparse_1_5625", r"$\tfrac{1}{64}$"),
)
SUPERRES = ("16^2->128^2", "32^2->128^2")
SAMPLERS = (("si_sde", "SI-SDE"), ("dm_sde", "DM-SDE"), ("fm_ode", "FM-ODE"))
COVS = ("shared_jac10", "jacfree")
TABLE_METRICS = (("rmse", "Vorticity RMSE"), ("crps", "CRPS"),
                 ("spread_skill", "Spread--skill"))
# Metrics written to all.csv for every expected cell (also when pending).
ALL_METRICS = ("rmse", "energy_spec_rmse", "crps", "crps_observed",
               "crps_unobserved", "spread_skill", "nfe", "seconds")
TABLE_EXP = 1          # x10^-1, as tab:ns_accuracy (scaled)
FMT = "{:.3f}"
DIVERGENCE_METRIC = "rmse"


# --------------------------------------------------------------------------- #
# Expected cells
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Cell:
    """One expected (method, variant) x scenario; its 5 trajectory files."""
    group: str
    method: str          # unified key (as in the CSVs)
    variant: str         # unified key ("" = none); may differ from the CSV's
    scenario: str
    files: tuple[Path, ...]  # expected metrics/ files, one per trajectory


@dataclass
class Group:
    name: str
    description: str
    root: Path                                  # dir holding metrics/ + per_step/
    # (method, unified variant, filename-stem(slug, traj), scenarios)
    keys: list[tuple[str, str, object, tuple[str, ...]]] = field(default_factory=list)


def _ns_dir(root: Path) -> Path:
    """Accept both <root>/metrics and <root>/navier_stokes/metrics layouts."""
    return root / CASE if (root / CASE / "metrics").exists() else root


def build_groups(base: Path, loc20_root: Path) -> list[Group]:
    all_scen = tuple(s for s, _, _ in SCENARIOS)
    main = Group("main", "main grid: Ours, generative baselines, EnKF, PF",
                 base / "results_wrong_ic_np40" / CASE)
    for samp, lbl in SAMPLERS:
        for cov in COVS:
            main.keys.append((f"Ours ({lbl})", cov,
                              lambda s, t, c=cov, a=samp: f"{s}__M100__traj{t}__ours_{c}__{a}",
                              all_scen))
    for meth, tok in (("FlowDAS", "flowdas"), ("SURGE (FlowDAS)", "surge_flowdas"),
                      ("SDA", "sda"), ("SURGE (SDA)", "surge_sda"),
                      ("D-Flow SGLD", "dflow_sgld"), ("Guided FM (FIG)", "fig")):
        main.keys.append((meth, "", lambda s, t, k=tok: f"{s}__M100__traj{t}__baselines__{k}",
                          all_scen))
    for meth, tok in (("EnKF", "enkf"), ("Particle filter", "particle_filter")):
        main.keys.append((meth, "", lambda s, t, k=tok: f"{s}__traj{t}__classical__{k}",
                          all_scen))

    ext = base / "results_wrong_ic_np40_ext" / CASE
    surge = Group("surge_ours", "SURGE on Ours (SI/DM-SDE)", ext)
    for samp, lbl in SAMPLERS[:2]:
        for cov in COVS:
            surge.keys.append((f"SURGE ({lbl})", f"surge_ours_{cov}",
                               lambda s, t, c=cov, a=samp:
                               f"{s}__M100__traj{t}__surge_ours_{c}__{a}", all_scen))
    fcs = []
    for name, tag, r in (("fc_r8", FC_TAG_R8, 8), ("fc_r20", FC_TAG_R20, 20)):
        g = Group(name, f"forecast correction, radius {r} (tag {tag})", ext)
        for samp, lbl in SAMPLERS:
            for cov in COVS:
                g.keys.append((f"Ours ({lbl})", f"{cov}_{tag}",
                               lambda s, t, c=cov, a=samp, tg=tag:
                               f"{s}__M100__traj{t}__ours_{c}__{tg}__{a}", all_scen))
        fcs.append(g)
    prior = Group("prior_enkf", "generative prior (unguided Ours, M=100) + EnKF analysis",
                  base / "results_wrong_ic_np40_prior_enkf" / CASE)
    for samp, lbl in SAMPLERS:
        prior.keys.append((f"{lbl} prior + EnKF", "",
                           lambda s, t, a=samp: f"{s}__M100__traj{t}__prior_enkf__{a}",
                           all_scen))
    loc20 = Group("enkf_loc20", "true-solver EnKF, Gaspari-Cohn r=20 on super-resolution",
                  _ns_dir(loc20_root))
    loc20.keys.append(("EnKF", "loc20_superres",
                       lambda s, t: f"{s}__traj{t}__classical__enkf", SUPERRES))
    return [main, surge, *fcs, prior, loc20]


def expected_cells(groups: list[Group]) -> list[Cell]:
    slug = {s: sl for s, sl, _ in SCENARIOS}
    out = []
    for g in groups:
        for method, variant, stem, scens in g.keys:
            for s in scens:
                files = tuple(g.root / "metrics" / f"{stem(slug[s], t)}.csv" for t in TRAJS)
                out.append(Cell(g.name, method, variant, s, files))
    return out


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def _read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _is_diverged(rows: list[dict[str, str]]) -> bool:
    for r in rows:
        if r.get("metric") == DIVERGENCE_METRIC:
            try:
                if not math.isnan(float(r["value"])):
                    return False
            except (TypeError, ValueError):
                pass
    return True


def _traj_of(path: Path) -> int:
    return int(path.name.split("__traj")[1].split("__")[0])


@dataclass
class CellState:
    cell: Cell
    finished: list[Path]
    diverged: list[Path]
    ok: list[Path]
    csv_method: str = ""
    csv_variant: str = ""

    @property
    def n_expected(self) -> int:
        return len(self.cell.files)

    @property
    def status(self) -> str:
        n = len(self.finished)
        return "done" if n == self.n_expected else ("partial" if n else "pending")


def scan(cells: list[Cell]) -> list[CellState]:
    out = []
    for c in cells:
        fin = [p for p in c.files if p.exists()]
        div, ok = [], []
        st = CellState(c, fin, div, ok)
        for p in fin:
            rows = _read_rows(p)
            if rows:
                st.csv_method = rows[0]["method"]
                st.csv_variant = rows[0].get("variant", "") or ""
                if rows[0]["scenario"] != c.scenario:
                    raise SystemExit(f"scenario mismatch in {p}: {rows[0]['scenario']}")
            (div if _is_diverged(rows) else ok).append(p)
        if fin and st.csv_method != c.method:
            raise SystemExit(f"method mismatch for {c}: CSV has {st.csv_method!r}")
        out.append(st)
    return out


def _unexpected_files(groups: list[Group], cells: list[Cell]) -> dict[str, list[str]]:
    """Files in a group's metrics/ that match no expected cell of ANY group sharing
    that directory -- reported, so a renamed run never silently goes missing."""
    expected = {p for c in cells for p in c.files}
    out: dict[str, list[str]] = {}
    for root in sorted({g.root for g in groups}):
        d = root / "metrics"
        if d.exists():
            extra = sorted(p.name for p in d.glob("*.csv") if p not in expected)
            if extra:
                out[str(d)] = extra
    return out


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def aggregate(states: list[CellState]):
    """Scalar + per-step reduction over non-diverged trajectories (aggregate_lib's
    reducers, so the main-grid numbers match aggregated/all.csv), relabelled to the
    unified (method, variant) keys."""
    scalar: dict[tuple, dict[str, tuple[float, float, int]]] = {}
    ps_rows: list[dict[str, object]] = []
    for st in states:
        c = st.cell
        key = (c.method, c.variant, c.scenario)
        scalar[key] = {}
        if not st.ok:
            continue
        by_traj = {_traj_of(p): [p] for p in st.ok}
        for r in aggregate_scalar(by_traj):
            scalar[key][str(r["metric"])] = (float(r["value"]), float(r["std"]),
                                            int(r["n_traj"]))
        ps_by_traj = {}
        for p in st.ok:
            q = p.parent.parent / "per_step" / p.name
            if q.exists():
                ps_by_traj[_traj_of(p)] = [q]
        for r in aggregate_per_step(ps_by_traj):
            r["method"], r["variant"] = c.method, c.variant
            r["group"] = c.group
            ps_rows.append(r)
    ps_rows.sort(key=lambda r: (str(r["method"]), str(r["variant"] or ""),
                                str(r["scenario"]), str(r["metric"]), r["step"]))
    return scalar, ps_rows


def write_outputs(out_dir: Path, groups, states, scalar, ps_rows, unexpected) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "READY").unlink(missing_ok=True)

    fields = ["method", "variant", "scenario", "metric", "value", "std", "n_traj",
              "n_div", "n_expected", "status", "group", "n_finished"]
    with open(out_dir / "all.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for st in states:
            c = st.cell
            vals = scalar[(c.method, c.variant, c.scenario)]
            for m in ALL_METRICS:
                v = vals.get(m)
                w.writerow({
                    "method": c.method, "variant": c.variant, "scenario": c.scenario,
                    "metric": m,
                    "value": "" if v is None else repr(v[0]),
                    "std": "" if v is None else repr(v[1]),
                    "n_traj": len(st.ok) if v is None else v[2],
                    "n_div": len(st.diverged), "n_expected": st.n_expected,
                    "status": st.status, "group": c.group,
                    "n_finished": len(st.finished),
                })

    ps_fields = ["case", "method", "variant", "scenario", "metric", "step", "value",
                 "std", "E", "M", "seed", "test_index", "NFE", "seconds", "n_traj",
                 "group"]
    with open(out_dir / "per_step.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ps_fields)
        w.writeheader()
        for r in ps_rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in ps_fields})

    with open(out_dir / "diverged.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "method", "variant", "scenario", "traj", "file"])
        for st in states:
            for p in st.diverged:
                c = st.cell
                w.writerow([c.group, c.method, c.variant, c.scenario, _traj_of(p), str(p)])

    srows = []
    for g in groups:
        gs = [s for s in states if s.cell.group == g.name]
        n_exp = sum(s.n_expected for s in gs)
        n_fin = sum(len(s.finished) for s in gs)
        srows.append({
            "group": g.name, "description": g.description, "root": str(g.root),
            "n_cells": len(gs), "n_expected": n_exp, "n_finished": n_fin,
            "n_diverged": sum(len(s.diverged) for s in gs), "n_pending": n_exp - n_fin,
            "complete": n_fin == n_exp,
            "keys": " | ".join(f"{m} [{v}]" for m, v, _, _ in g.keys),
        })
    with open(out_dir / "status.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0]))
        w.writeheader()
        w.writerows(srows)

    all_done = all(r["complete"] for r in srows)
    lines = [
        "aggregated_all written completely by aggregate_ns_all.py "
        "(all.csv, per_step.csv, status.csv, diverged.csv).",
        f"all_runs_finished: {str(all_done).lower()}",
    ]
    for r in srows:
        lines.append(f"{r['group']}: {r['n_finished']}/{r['n_expected']} finished, "
                     f"{r['n_diverged']} diverged, {r['n_pending']} pending")
    if unexpected:
        lines.append("unexpected files (not in any group): "
                     + "; ".join(f"{d}: {len(v)}" for d, v in unexpected.items()))
    lines.append("Rerun after more cells finish: PE/refresh_ns_appendix.sh")
    (out_dir / "READY").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines[1:]:
        print(f"[ns_all] {line}")


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #

PENDING = "pending"
# The table has 12 metric columns over ~34 rows; with the std inline it would be
# shrunk to ~55% by \resizebox. Stacking the std (and any superscript) UNDER the
# value halves the column width, so the page is read at ~full size.
STACK_STD = True
SAME_AS_ENKF = r"$=$\,EnKF"


@dataclass
class Row:
    label: str
    method: str | None      # None -> block header row
    variant: str = ""
    same_as: tuple[str, str] | None = None   # sparse cells printed as "= EnKF"


def table_rows() -> list[Row]:
    rows: list[Row] = []
    heads = {
        "shared_jac10": r"Ours -- ensemble-shared inflated covariance (Jacobian "
                        r"refreshed every $k=10$ steps)",
        "jacfree": "Ours -- Jacobian-free covariance",
    }
    for cov in COVS:
        rows.append(Row(heads[cov], None))
        for samp, lbl in SAMPLERS:
            m = f"Ours ({lbl})"
            rows.append(Row(m, m, cov))
            rows.append(Row(rf"\quad + FC ($r=8$)", m, f"{cov}_{FC_TAG_R8}"))
            rows.append(Row(rf"\quad + FC ($r=20$)", m, f"{cov}_{FC_TAG_R20}"))
            if samp != "fm_ode":
                rows.append(Row(r"\quad + SURGE", f"SURGE ({lbl})", f"surge_ours_{cov}"))
    rows.append(Row("Generative baselines", None))
    for m, lbl in (("FlowDAS", "FlowDAS"), ("SURGE (FlowDAS)", "FlowDAS + SURGE"),
                   ("SDA", "SDA"), ("SURGE (SDA)", "SDA + SURGE"),
                   ("D-Flow SGLD", "D-Flow SGLD"), ("Guided FM (FIG)", "Guided FM (FIG)")):
        rows.append(Row(lbl, m))
    rows.append(Row("Generative prior + EnKF analysis", None))
    for _, lbl in SAMPLERS:
        rows.append(Row(f"{lbl} prior + EnKF", f"{lbl} prior + EnKF"))
    rows.append(Row("Classical filters (true solver)", None))
    rows.append(Row("EnKF", "EnKF"))
    rows.append(Row(r"EnKF ($r=20$ everywhere)\textsuperscript{$\ast$}", "EnKF",
                    "loc20_superres", same_as=("EnKF", "")))
    rows.append(Row("Particle filter", "Particle filter"))
    return rows


def _cell_text(st: CellState | None, val, *, bold: bool, md: bool,
               underline: bool = False) -> str:
    if st is None:
        return "--"
    n_fin, n_div, n_exp = len(st.finished), len(st.diverged), st.n_expected
    if n_fin == 0:
        return PENDING
    if not st.ok:   # every finished trajectory diverged
        s = DIV_ALL.format(k=n_div, n=n_fin)
    else:
        mean, std = val
        num = _fmt_number(mean * 10 ** TABLE_EXP, FMT)
        if bold:
            num = f"**{num}**" if md else rf"\textbf{{{num}}}"
        elif underline:
            num = f"_{num}_" if md else rf"\underline{{{num}}}"
        sd = None if std is None else std * 10 ** TABLE_EXP
        if md:
            s = num + (f" ± {FMT.format(sd)}" if sd is not None and len(st.ok) >= 2
                       else "")
        else:
            s = num + (r"\\" if STACK_STD else "") + std_suffix(sd, FMT, n=len(st.ok))
        if n_div:
            s += f" (div {n_div}/{n_fin})" if md else DIV_SOME.format(k=n_div, n=n_fin)
    if n_fin < n_exp:
        s += f" [{n_fin}/{n_exp} done]" if md else rf"\textsuperscript{{$\dagger${n_fin}/{n_exp}}}"
    if not md and STACK_STD and r"\\" in s:
        s = rf"\begin{{tabular}}[c]{{@{{}}c@{{}}}}{s}\end{{tabular}}"
    return s


def build_grid(states: list[CellState], scalar):
    by_key = {(s.cell.method, s.cell.variant, s.cell.scenario): s for s in states}
    rows = table_rows()
    cols = [(m, s) for m, _ in TABLE_METRICS for s, _, _ in SCENARIOS]
    grid: list[list[tuple[CellState | None, float | None, bool]]] = []
    for r in rows:
        line = []
        for m, s in cols:
            if r.method is None:
                line.append((None, None, False))
                continue
            same = r.same_as is not None and s not in SUPERRES
            key = (*r.same_as, s) if same else (r.method, r.variant, s)
            st = by_key.get(key)
            v = scalar.get(key, {}).get(m)
            line.append((st, None if v is None else (v[0], v[1]), same))
        grid.append(line)
    # Bold: per column, among COMPLETED cells only (5/5 finished, none diverged);
    # a "= EnKF" duplicate does not compete twice. Same tie rule as tab:ns_accuracy.
    best, second = [], []
    for j in range(len(cols)):
        vals = []
        for i, r in enumerate(rows):
            st, v, same = grid[i][j]
            ok = (st is not None and not same and st.status == "done"
                  and not st.diverged and v is not None)
            vals.append(v[0] if ok else None)
        b = _best_in_column(vals, FMT, lower_is_better=True)
        best.append(b)
        second.append(_second_in_column(vals, FMT, b, lower_is_better=True))
    return rows, cols, grid, best, second


def _counts_phrase(states: list[CellState]) -> str:
    n_exp = sum(s.n_expected for s in states)
    n_fin = sum(len(s.finished) for s in states)
    return f"{n_fin} of {n_exp} trajectory-cells finished"


def caption(states) -> str:
    return (
        r"Navier--Stokes from a poor initial condition, all methods and variants: "
        r"vorticity RMSE, CRPS and spread--skill $|1-\mathrm{spread}/\mathrm{skill}|$ "
        r"(lower is better; every metric in units of $10^{-1}$). Initial condition "
        r"taken from a sixth held-out trajectory, 40 physical steps (5 history + 35 "
        r"assimilation steps), $E=64$, $M=100$ sampler steps for every generative "
        r"method. Each entry is the mean over the non-diverged trajectories of $n=5$ "
        r"held-out trajectories. ``div $k/n$'' marks a cell in which $k$ of $n$ "
        r"trajectories diverged (tripped the divergence guard); they are excluded "
        r"from every metric, and a number carrying a div superscript is the mean over "
        r"the $n-k$ survivors. ``pending'' marks a run that has not finished any "
        r"trajectory yet; a superscript $\dagger j/5$ marks a cell of which only $j$ "
        r"trajectories have finished, the number being the mean over those. "
        + BOLD_UNDERLINE_SENTENCE[:-1]
        + r", among cells with all five trajectories finished and none diverged. "
        + STD_SENTENCE + " "
        + 
        r"+ FC applies the forecast-residual correction with an online estimate "
        r"$\hat{Q}$ of the forecast-error covariance, 25 forecast steps and gain "
        r"scale 1, at localisation radius $r=8$ or $r=20$. + SURGE resamples our "
        r"posterior ensemble with SURGE (ESS threshold 0.5). "
        r"Generative prior + EnKF: the unguided forecast of our trained prior "
        r"($M=100$) followed by the same stochastic EnKF analysis as the EnKF row, "
        r"i.e.\ the EnKF with only its forecast model replaced. "
        r"EnKF: Gaspari--Cohn localisation with radius $r=20$ on the sparse "
        r"scenarios and none (global) on super-resolution. "
        r"\textsuperscript{$\ast$}EnKF with $r=20$ also on super-resolution; its sparse "
        r"cells are the EnKF row's run (identical configuration), shown as "
        r"``$=$\,EnKF''. Cost is omitted: these runs were timed under GPU contention."
    )


def render_tex(rows, cols, grid, best, second, states) -> str:
    n_s = len(SCENARIOS)
    span = 1 + len(cols)
    groups = [rf"{lbl} ($\times 10^{{-{TABLE_EXP}}}$)" for _, lbl in TABLE_METRICS]
    head = [rf"& \multicolumn{{{n_s}}}{{c|}}{{\textbf{{{g}}}}}" for g in groups[:-1]]
    head.append(rf"& \multicolumn{{{n_s}}}{{c}}{{\textbf{{{groups[-1]}}}}} \\")
    sub = " & ".join(h for _, _, h in SCENARIOS)
    colfmt = "l|" + "|".join("c" * n_s for _ in TABLE_METRICS)
    out = [
        "% Auto-generated by paper_experiments/aggregate_ns_all.py -- DO NOT EDIT BY HAND.",
        f"% Status at generation: {_counts_phrase(states)}; rerun "
        "paper_experiments/refresh_ns_appendix.sh as pending runs finish.",
        "% Requires: booktabs, graphicx, review_colors (\\revC).",
        "",
        r"\begin{table*}[p]",
        r"    \centering",
        r"    \scriptsize",
        r"    \captionsetup{font=small}",
        r"    \setlength{\tabcolsep}{3.5pt}",
        rf"    \caption{{\revC{{{caption(states)}}}}}",
        r"    \label{tab:ns_all_wrong_ic}",
        r"    \resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
        rf"    \begin{{tabular}}{{{colfmt}}}",
        r"        \toprule",
        "        " + "\n        ".join(head),
        "        & " + " & ".join([sub] * len(TABLE_METRICS)) + r" \\",
    ]
    for i, r in enumerate(rows):
        if r.method is None:
            out.append(r"        \midrule")
            out.append(rf"        \multicolumn{{{span}}}{{l}}{{\textit{{{r.label}}}}} \\")
            continue
        cells = []
        for j in range(len(cols)):
            st, v, same = grid[i][j]
            cells.append(SAME_AS_ENKF if same else
                         _cell_text(st, v, bold=i in best[j], md=False,
                                    underline=i in second[j]))
        out.append(f"        {r.label} & " + " & ".join(cells) + r" \\")
    out += [r"        \bottomrule", r"    \end{tabular}}", r"\end{table*}", ""]
    return "\n".join(out)


def _md_label(s: str) -> str:
    for a, b in ((r"\quad ", "&nbsp;&nbsp;"), (r"($r=8$)", "(r=8)"), (r"($r=20$)", "(r=20)"),
                 (r"\textsuperscript{$\ast$}", "*"), (r"$k=10$", "k=10"), ("--", "–"),
                 (r"($r=20$ everywhere)", "(r=20 everywhere)")):
        s = s.replace(a, b)
    return s


def render_md(rows, cols, grid, best, second, states) -> str:
    hdr = ["Method"] + [f"{lbl} {h}" for _, lbl in TABLE_METRICS
                        for _, _, h in SCENARIOS]
    hdr = [h.replace("Vorticity RMSE", "RMSE").replace("Spread--skill", "SS")
           .replace(r"$16^2$", "16²").replace(r"$32^2$", "32²").replace(r"$5\%$", "5%")
           .replace(r"$\tfrac{1}{64}$", "1/64") for h in hdr]
    out = [
        "## tab:ns_all_wrong_ic -- every NS wrong-IC result (x10^-1)",
        "",
        f"Status: {_counts_phrase(states)}. `pending` = no trajectory finished; "
        "`[j/5 done]` = mean over the j finished; `(div k/n)` = k of n diverged, "
        "excluded; **bold** = best, _underline_ = second best, among complete, "
        "non-diverged cells (2-decimal ties, unscaled); ± = std over non-diverged "
        "trajectories (n>=2).",
        "",
        "| " + " | ".join(hdr) + " |",
        "|" + "---|" * len(hdr),
    ]
    for i, r in enumerate(rows):
        if r.method is None:
            out.append(f"| ***{_md_label(r.label)}*** |" + " |" * len(cols))
            continue
        cells = []
        for j in range(len(cols)):
            st, v, same = grid[i][j]
            cells.append("= EnKF" if same else _cell_text(st, v, bold=i in best[j], md=True,
                                                          underline=i in second[j]))
        out.append(f"| {_md_label(r.label)} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=SCRATCH_PE,
                    help="directory holding the results_wrong_ic_np40* trees")
    ap.add_argument("--enkf-loc20-root", type=Path,
                    default=Path(os.environ.get("ENKF_LOC20_ROOT", ENKF_LOC20_ROOT)),
                    help="production root of the r=20 super-res EnKF (default: %(default)s)")
    args = ap.parse_args()

    groups = build_groups(args.base, args.enkf_loc20_root)
    cells = expected_cells(groups)
    states = scan(cells)
    unexpected = _unexpected_files(groups, cells)
    for d, names in unexpected.items():
        print(f"[ns_all] WARNING {len(names)} file(s) in {d} match no expected cell, "
              f"e.g. {names[:3]}")
    scalar, ps_rows = aggregate(states)

    out_root = args.base / "results_wrong_ic_np40"
    rows, cols, grid, best, second = build_grid(states, scalar)
    tables = out_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "ns_all_wrong_ic_scaled.tex").write_text(
        render_tex(rows, cols, grid, best, second, states), encoding="utf-8")
    (tables / "ns_all_wrong_ic_scaled.md").write_text(
        render_md(rows, cols, grid, best, second, states), encoding="utf-8")
    print(f"[ns_all] table -> {tables / 'ns_all_wrong_ic_scaled.tex'} (+ .md)")
    # READY last, after every aggregate file is written.
    write_outputs(out_root / CASE / "aggregated_all", groups, states, scalar,
                  ps_rows, unexpected)
    print(f"[ns_all] aggregate -> {out_root / CASE / 'aggregated_all'}")


if __name__ == "__main__":
    main()
