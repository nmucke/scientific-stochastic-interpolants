"""Aggregate the Navier--Stokes (Case 2) per-cell results over trajectories.

The NS grid (``run_ns_grid.sh``) writes, per ``(scenario, M, trajectory, group)``
cell, two tidy files:

* ``results/navier_stokes/metrics/<...>__traj<N>__<group>.csv`` -- the scalar
  metrics, each already averaged over assimilation steps IN-RUN (one value per
  trajectory).
* ``results/navier_stokes/per_step/<...>__traj<N>__<group>.csv`` -- the
  per-assimilation-step metric curves for that trajectory.

This produces the two aggregates the user asked for, both reduced ACROSS
trajectories (trajectory identity comes from the ``traj<N>`` filename token):

* ``aggregated/all.csv``      -- SCALAR metrics, mean +/- std over trajectories
                                 (metric already time-averaged). Canonical schema
                                 + ``n_traj``; consumed by ``make_ns_figures``
                                 (metric-vs-M) and ``make_tables``.
* ``aggregated/per_step.csv`` -- PER-STEP metrics, mean +/- std over trajectories
                                 at each assimilation step; consumed by
                                 ``make_ns_figures`` (metric-vs-step).
* ``tables.tex``              -- LaTeX results tables, ONE PER SAMPLER-STEP COUNT
                                 ``M`` (RMSE / CRPS / spread--skill x the four
                                 scenarios + per-step cost), built from the same
                                 aggregated rows so they can never drift from the
                                 CSVs. ``\\input`` straight into the manuscript.
* ``tables_scaled.tex``       -- the SAME tables in per-metric units of $10^{-n}$
                                 (the factor is named once in each metric header);
                                 pick whichever notation reads better.

Every NS metric present flows through both files (``rmse``, ``energy_spec_rmse``,
``kl_points``, ``crps``, ``crps_observed``, ``crps_unobserved``,
``spread_skill``, plus ``nfe`` / ``seconds``).

One further input, produced separately because it is computed from saved states
rather than from the grid's per-cell CSVs: ``reference/metrics.csv``, the
large-ensemble (E=1000) EnKF reference, written by
``compute_ns_reference_metrics.py``. It is loaded as-is, reduced over trajectories
like everything else, and rendered as an out-of-competition table row.

    .venv/bin/python paper_experiments/compute_ns_reference_metrics.py   # once
    .venv/bin/python paper_experiments/aggregate_ns.py
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import sys
from collections import defaultdict
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from common.aggregate_lib import (  # noqa: E402
    SCALAR_FIELDNAMES,
    aggregate_per_step,
    aggregate_scalar,
    aggregate_scalar_records,
    discover_by_traj,
    load_records_by_traj,
    print_scalar_table,
    write_per_step_csv,
    write_scalar_csv,
)
from common.latex_tables import TableSpec, write_latex_tables  # noqa: E402
from common.timing_lib import apply_timing, describe, load_timing  # noqa: E402

RESULTS = _here / "results"
CASE = "navier_stokes"

# Clean per-DA-step wall-clock, measured on an idle GPU by measure_timing.py. The
# ``seconds`` the grid runs recorded is contaminated: many cells were timed while
# another CUDA job shared the card (measured 2026-07-30: one D-Flow cell went
# 160 s -> 302 s, 1.9x, from contention alone) and some fell back to CPU. When this
# file exists its numbers REPLACE the ``seconds`` column and the ``metric=seconds``
# rows; every other metric is untouched. Absent -> the grid's own seconds are kept
# and a warning is printed, so the tables never silently mix the two sources.
KEY_METRICS = ("rmse", "energy_spec_rmse", "crps", "spread_skill", "kl_points")

# The large-ensemble reference EnKF (E=1000 vs the graded E=64 -- the same run that
# provides the KL reference distribution), as computed from its saved states by
# ``compute_ns_reference_metrics.py``. That script owns the metric computation and
# the method name; this one only LOADS its CSV and reduces it over trajectories
# like any other cell. Absent -> no reference row, with a pointer to the script.
REF_METHOD = "EnKF (reference)"

# The LaTeX tables: metric groups x the four observation scenarios, + cost.
TABLE_SPEC = TableSpec(
    case=CASE,
    case_label="Navier--Stokes",
    label_stem="tab:ns_accuracy",
    source="paper_experiments/aggregate_ns.py",
    scenarios=(
        ("16^2->128^2", r"$16^2$"),
        ("32^2->128^2", r"$32^2$"),
        ("sparse 5%", r"$5\%$"),
        ("sparse 1.5625%", r"$\tfrac{1}{64}$"),
    ),
    # The energy-spectrum RMSE is still aggregated into all.csv (and printed by
    # print_scalar_table below), but is deliberately NOT a table column: the
    # enstrophy-spectrum figures in make_ns_figures.py carry that comparison.
    metric_groups=(
        ("rmse", "Vorticity RMSE"),
        ("crps", "CRPS"),
        ("spread_skill", "Spread--skill"),
    ),
    ours=(
        ("Ours (SI-SDE)", "Ours (SI-SDE)"),
        ("Ours (DM-SDE)", "Ours (DM-SDE)"),
        ("Ours (FM-ODE)", "Ours (FM-ODE)"),
    ),
    # The bare FlowDAS / SDA rows are dropped from the table: their SURGE-augmented
    # counterparts are what the paper compares against. Both still run and are
    # aggregated into all.csv.
    baselines=(
        ("SURGE (FlowDAS)", "FlowDAS + SURGE"),
        ("SURGE (SDA)", "SDA + SURGE"),
        ("D-Flow SGLD", "D-Flow SGLD"),
        ("Guided FM (FIG)", "Guided FM (FIG)"),
    ),
    classical=(
        ("EnKF", "EnKF"),
        ("Particle filter", "Particle filter"),
    ),
    reference=((REF_METHOD, "EnKF reference"),),
    reference_nominal_E=((REF_METHOD, 1000),),
    metrics_phrase=(
        r"vorticity RMSE, CRPS, and spread--skill "
        r"($|1-\mathrm{spread}/\mathrm{skill}|$, $0=$ calibrated)"
    ),
    notes=(
        "The solver-free baselines share the prior; the conventional filters "
        "(EnKF, particle filter) use the true solver and have no sampler-step "
        "count, so their rows are the same run repeated in every $M$ table."
    ),
)


# A cell whose run tripped the divergence guard writes ``rmse = nan`` (and NaN for
# most other metrics), but some columns survive -- ``energy_spec_rmse`` of the
# states before the guard fired, ``seconds`` / ``NFE`` of the truncated run. Left
# in, those would be averaged with the healthy trajectories (and a fully diverged
# cell would still produce an energy-spectrum row). So a trajectory-cell is
# DIVERGED when its ``rmse`` is NaN or absent, and it is dropped WHOLE, from both
# the scalar and the per-step aggregates. The drops are counted per cell
# (``n_div`` column of all.csv) and listed in ``aggregated/diverged.csv``, so the
# tables can print "div k/n" instead of silently reporting fewer trajectories.
DIVERGENCE_METRIC = "rmse"


def _is_diverged(path: Path) -> tuple[bool, dict[str, str]]:
    """(diverged?, identifying fields) for one per-cell scalar CSV."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ident = dict(rows[0]) if rows else {}
    vals = [r.get("value", "") for r in rows if r.get("metric") == DIVERGENCE_METRIC]
    finite = []
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isnan(f):
            finite.append(f)
    return (not finite), ident


def _split_diverged(by_traj: dict[int, list[Path]]):
    """Remove diverged trajectory-cells; return (kept, diverged_records)."""
    kept: dict[int, list[Path]] = {}
    diverged: list[dict[str, object]] = []
    for traj, paths in by_traj.items():
        keep = []
        for p in paths:
            div, ident = _is_diverged(p)
            if div:
                diverged.append({
                    "file": p.name, "traj": traj,
                    "method": ident.get("method", ""),
                    "variant": ident.get("variant", "") or "",
                    "scenario": ident.get("scenario", ""),
                    "E": ident.get("E", ""), "M": ident.get("M", ""),
                })
            else:
                keep.append(p)
        if keep:
            kept[traj] = keep
    return kept, diverged


def _cell_key(method, variant, scenario, M) -> tuple:
    v = "" if variant in (None, "None") else str(variant)
    m = "" if M in (None, "") else str(int(float(M)))
    return (str(method), v, str(scenario), m)


def _write_scalar_csv_with_div(out: Path, rows, n_div: dict[tuple, int]) -> None:
    """all.csv = canonical scalar schema + n_traj + n_div (diverged, excluded)."""
    fields = [*SCALAR_FIELDNAMES, "n_div"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            rec = {k: ("" if row.get(k) is None else row.get(k)) for k in fields}
            rec["n_div"] = n_div.get(
                _cell_key(row["method"], row.get("variant"), row["scenario"], row.get("M")), 0
            )
            w.writerow(rec)


# The manuscript's appendix tab:ablation: our samplers only, RMSE / CRPS /
# spread--skill + cost, one sub-block per Jacobian refresh cadence k in {1, 5, 10}
# and the Jacobian-free covariance. The k=1 and k=5 cadences come from the
# separate ablation grid (run_ns_ablation.sh); when that was not run they are
# rendered as placeholder blocks of "--" so the table keeps its shape. The
# k=10 / Jacobian-free rows are the main grid's.
ABLATION_SPEC = dataclasses.replace(
    TABLE_SPEC,
    label_stem="tab:ablation",
    metric_groups=(
        ("rmse", "Vorticity RMSE"),
        ("crps", "CRPS"),
        ("spread_skill", "Spread--skill"),
    ),
    include_others=False,
    placeholder_variants=("shared_jac1", "shared_jac5"),
    variant_labels=(
        ("shared_jac1", r"Ensemble-shared inflated covariance -- Jacobian refreshed "
                        r"every step ($k=1$)"),
        ("shared_jac5", r"Ensemble-shared inflated covariance -- Jacobian refreshed "
                        r"every $k=5$ steps"),
        ("shared_jac10", r"Ensemble-shared inflated covariance -- Jacobian refreshed "
                         r"every $k=10$ steps"),
        ("jacfree", "Jacobian-free covariance"),
    ),
    metrics_phrase=(
        r"ablation of the covariance approximation (Jacobian refresh cadence $k$ "
        r"vs.\ Jacobian-free): vorticity RMSE, CRPS and spread--skill "
        r"($|1-\mathrm{spread}/\mathrm{skill}|$)"
    ),
    notes="Blocks whose every cell is ``--'' were not run in this setting.",
)

# Review tables written to <results-root>/tables/ (plus PREVIEW.md), each in the
# plain and the scaled (manuscript) notation:
#   (file stem, spec, metric groups or None = spec's own, preview title)
REVIEW_TABLES = (
    ("ns_accuracy", TABLE_SPEC, ("rmse", "crps"),
     "tab:ns_accuracy (results.tex) -- RMSE, CRPS, cost"),
    ("ns_accuracy_with_spread_skill", TABLE_SPEC, ("rmse", "crps", "spread_skill"),
     "tab:ns_accuracy, commented-out variant with spread--skill (results.tex)"),
    ("ns_ablation", ABLATION_SPEC, None,
     "tab:ablation (appendix_ns_case.tex) -- PARTIAL: k=1/k=5 not run"),
)


_PHRASES = {
    "rmse": "vorticity RMSE",
    "crps": "CRPS",
    "spread_skill": r"spread--skill ($|1-\mathrm{spread}/\mathrm{skill}|$, $0=$ calibrated)",
}


def _metrics_phrase(metrics) -> str:
    """Caption phrase naming exactly the metric groups the table carries."""
    words = [_PHRASES.get(m, m) for m in metrics]
    if len(words) <= 2:
        return " and ".join(words)
    return ", ".join(words[:-1]) + ", and " + words[-1]


def _write_review_tables(out_dir: Path, rows, n_div, *, only_M, with_std) -> list[str]:
    """Write every REVIEW_TABLES entry (plain + scaled .tex, scaled .md)."""
    md_parts: list[str] = []
    for stem, base, metrics, title in REVIEW_TABLES:
        spec = base
        if metrics is not None:
            groups = {**dict(TABLE_SPEC.metric_groups), **dict(ABLATION_SPEC.metric_groups)}
            spec = dataclasses.replace(base, metric_groups=tuple(
                (m, groups[m]) for m in metrics), metrics_phrase=_metrics_phrase(metrics))
        for scaled in (False, True):
            tex = out_dir / f"{stem}{'_scaled' if scaled else ''}.tex"
            md = out_dir / f"{stem}{'_scaled' if scaled else ''}.md"
            write_latex_tables(spec, rows, tex, with_std=with_std, scaled=scaled,
                               only_M=only_M, n_div=n_div, markdown_out=md,
                               markdown_title=title)
        # Preview: the manuscript (scaled) notation, with the across-traj std.
        md_std = out_dir / f"{stem}_scaled_std.md"
        write_latex_tables(spec, rows, out_dir / f".{stem}_std.tex", with_std=True,
                           scaled=True, only_M=only_M, n_div=n_div,
                           markdown_out=md_std, markdown_title=title + " (mean ± std)")
        (out_dir / f".{stem}_std.tex").unlink(missing_ok=True)
        md_parts.append((out_dir / f"{stem}_scaled.md").read_text(encoding="utf-8"))
        md_parts.append(md_std.read_text(encoding="utf-8"))
        print(f"[{CASE}] review table {stem} -> {out_dir}")
    return md_parts


# --------------------------------------------------------------------------- #
# Forecast correction (FC): a separate grid (``--fc-root``) of our samplers with
# the forecast-residual correction, tagged ``__<fc_tag>__`` in the filename and
# ``<variant>_<fc_tag>`` in the variant column. Aggregated on its own into
# ``<results-root>/navier_stokes/aggregated_fc/`` (the main aggregated/ is never
# touched), with the same whole-cell divergence handling, and rendered as
# tab:ns_accuracy_fc: each plain row of ours (read back from the main all.csv)
# directly followed by its + FC row.
# --------------------------------------------------------------------------- #
FC_TAG_DEFAULT = "fc25_70a3134d"


def _fc_spec(tag: str):
    return dataclasses.replace(
        TABLE_SPEC,
        label_stem="tab:ns_accuracy_fc",
        metric_groups=(("rmse", "Vorticity RMSE"), ("crps", "CRPS")),
        metrics_phrase=(
            "effect of the forecast correction (FC) on vorticity RMSE and CRPS"
        ),
        # Baselines and classical filters as in tab:ns_accuracy, from the same
        # poor-IC grid; the E=1000 EnKF reference was not run for this setting.
        include_others=True,
        reference=(),
        reference_nominal_E=(),
        include_cost=False,
        paired_variants=(
            ("shared_jac10", f"shared_jac10_{tag}", " + FC"),
            ("jacfree", f"jacfree_{tag}", " + FC"),
        ),
        include_missing_cell_note=False,
        notes=(
            "+ FC applies the forecast-residual correction with an online estimate "
            r"$\hat{Q}$ of the forecast-error covariance (25 forecast steps, "
            "localisation radius 8, gain scale 1; untuned defaults); each plain row "
            "is the same sampler without the correction, and the baseline rows are "
            "the baselines of Table~\\ref{tab:ns_accuracy}, all run under the same "
            "poor-initial-condition setting. "
            "Cost is omitted: the FC grid was timed under GPU contention and has no "
            "clean measurement."
        ),
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _ndiv_from_rows(rows) -> dict[tuple, int]:
    out: dict[tuple, int] = {}
    for r in rows:
        k = int(float(r.get("n_div") or 0))
        if k:
            out[_cell_key(r["method"], r.get("variant"), r["scenario"], r.get("M"))] = k
    return out


def aggregate_fc(results_root: Path, fc_root: Path, tag: str, tables_dir: Path,
                 only_M: int | None) -> Path:
    """Aggregate the FC cells and write tab:ns_accuracy_fc; return the out dir."""
    src = fc_root / CASE
    out_dir = results_root / CASE / "aggregated_fc"
    token = f"__{tag}__"
    pick = lambda by: {t: [p for p in ps if token in p.name]  # noqa: E731
                       for t, ps in by.items() if any(token in p.name for p in ps)}

    scalar_by_traj = pick(discover_by_traj(src / "metrics"))
    n_files = sum(len(v) for v in scalar_by_traj.values())
    print(f"[{CASE}/fc] {n_files} FC cells ({tag}) in {src / 'metrics'}")
    scalar_by_traj, diverged = _split_diverged(scalar_by_traj)
    n_div: dict[tuple, int] = defaultdict(int)
    for d in diverged:
        n_div[_cell_key(d["method"], d["variant"], d["scenario"], d["M"])] += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "READY_FC").unlink(missing_ok=True)
    with open(out_dir / "diverged.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "traj", "method", "variant",
                                           "scenario", "E", "M"])
        w.writeheader()
        w.writerows(diverged)
    print(f"[{CASE}/fc] {len(diverged)} diverged FC trajectory-cells EXCLUDED")
    for d in diverged:
        print(f"[{CASE}/fc]   diverged: {d['file']}")
    rows = aggregate_scalar(scalar_by_traj)
    # No timing splice: the FC grid has no clean measurement; seconds stay the
    # grid's own (contended) values and the FC table carries no cost column.
    _write_scalar_csv_with_div(out_dir / "all.csv", rows, n_div)

    div_files = {d["file"] for d in diverged}
    ps_by_traj = {t: [p for p in ps if p.name not in div_files]
                  for t, ps in pick(discover_by_traj(src / "per_step")).items()}
    ps_rows = aggregate_per_step(ps_by_traj)
    write_per_step_csv(out_dir / "per_step.csv", ps_rows)
    print(f"[{CASE}/fc] wrote {len(rows)} scalar / {len(ps_rows)} per-step rows -> {out_dir}")

    # The table: our plain rows from the (untouched) main all.csv + the FC rows.
    main_csv = results_root / CASE / "aggregated" / "all.csv"
    main_rows = _read_csv_rows(main_csv)  # Ours plain + baselines + classical
    fc_rows = _read_csv_rows(out_dir / "all.csv")
    all_rows = main_rows + fc_rows
    # A cell whose trajectories ALL diverged has no all.csv row, so its count can
    # only come from the main grid's diverged.csv (as in tab:ns_accuracy).
    fc_n_div = _ndiv_from_rows(all_rows)
    main_div_csv = results_root / CASE / "aggregated" / "diverged.csv"
    if main_div_csv.exists():
        main_div: dict[tuple, int] = defaultdict(int)
        for d in _read_csv_rows(main_div_csv):
            main_div[_cell_key(d["method"], d.get("variant"), d["scenario"], d.get("M"))] += 1
        for k, v in main_div.items():
            fc_n_div.setdefault(k, v)
    spec = _fc_spec(tag)
    for scaled in (False, True):
        stem = f"ns_accuracy_fc{'_scaled' if scaled else ''}"
        write_latex_tables(
            spec, all_rows, tables_dir / f"{stem}.tex", scaled=scaled, only_M=only_M,
            n_div=fc_n_div, markdown_out=tables_dir / f"{stem}.md",
            markdown_title="tab:ns_accuracy_fc -- forecast correction, RMSE and CRPS",
        )
    write_latex_tables(
        spec, all_rows, tables_dir / ".fc_std.tex", with_std=True, scaled=True,
        only_M=only_M, n_div=fc_n_div,
        markdown_out=tables_dir / "ns_accuracy_fc_scaled_std.md",
        markdown_title="tab:ns_accuracy_fc (mean ± std)",
    )
    (tables_dir / ".fc_std.tex").unlink(missing_ok=True)
    print(f"[{CASE}/fc] tab:ns_accuracy_fc -> {tables_dir}")
    (out_dir / "READY_FC").write_text(
        f"FC aggregates complete ({n_files} cells, tag {tag}, source {src}).\n"
        f"{len(diverged)} diverged trajectory-cells excluded whole (diverged.csv); "
        "all.csv has n_traj (survivors) and n_div.\n"
        "seconds are the FC grid's own CONTENDED timings (no clean measurement).\n",
        encoding="utf-8",
    )
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--std", action="store_true",
        help="(default; kept for old command lines) print each metric cell as "
             "mean {\\tiny +/- std} over trajectories",
    )
    ap.add_argument(
        "--no-std", action="store_true",
        help="print the LaTeX tables WITHOUT the across-trajectory std",
    )
    ap.add_argument(
        "--results-root", type=Path, default=RESULTS,
        help="directory holding navier_stokes/ (default: results/ beside this "
             "script). Everything is read from and written under it.",
    )
    ap.add_argument(
        "--timing-csv", type=Path, default=None,
        help="clean per-DA-step timings from measure_timing.py (default: "
             "<results-root>/timing/seconds_per_step.csv)",
    )
    ap.add_argument(
        "--only-M", type=int, default=None,
        help="render the LaTeX tables at this single M. Use on a one-rung grid, or "
             "the classical rows (labelled M=50) get a table of their own.",
    )
    ap.add_argument(
        "--table-metrics", default=None,
        help="comma-separated metric groups for the LaTeX tables (default: every "
             "group in TABLE_SPEC). The manuscript's tab:ns_accuracy is rmse,crps.",
    )
    ap.add_argument(
        "--no-timing-override", action="store_true",
        help="keep the grid's own (contended) seconds instead of splicing in the "
             "clean measurements",
    )
    ap.add_argument(
        "--tables-dir", type=Path, default=None,
        help="where the review tables (tab:ns_accuracy, its spread--skill variant, "
             "tab:ablation) and PREVIEW-ready Markdown go (default: "
             "<results-root>/tables)",
    )
    ap.add_argument(
        "--fc-root", type=Path, default=None,
        help="results root of the forecast-correction grid (holding navier_stokes/); "
             "its FC cells are aggregated into <results-root>/navier_stokes/"
             "aggregated_fc/ and tab:ns_accuracy_fc is written to --tables-dir",
    )
    ap.add_argument("--fc-tag", default=FC_TAG_DEFAULT,
                    help="FC filename/variant tag (default: %(default)s)")
    ap.add_argument(
        "--fc-only", action="store_true",
        help="with --fc-root: do ONLY the FC step (reads the existing main "
             "aggregated/all.csv, rewrites nothing of the main aggregation)",
    )
    args = ap.parse_args()
    if args.fc_only:
        if args.fc_root is None:
            ap.error("--fc-only needs --fc-root")
        aggregate_fc(args.results_root, args.fc_root, args.fc_tag,
                     args.tables_dir or args.results_root / "tables", args.only_M)
        return
    root = args.results_root / CASE
    timing_csv = args.timing_csv or args.results_root / "timing" / "seconds_per_step.csv"
    ref_metrics_csv = root / "reference" / "metrics.csv"
    spec = TABLE_SPEC
    if args.table_metrics:
        groups = dict(spec.metric_groups)
        spec = dataclasses.replace(spec, metric_groups=tuple(
            (m, groups[m]) for m in args.table_metrics.split(",")
        ), metrics_phrase=_metrics_phrase(args.table_metrics.split(",")))

    # (1) SCALAR metrics, aggregated over trajectories (time already averaged).
    scalar_by_traj = discover_by_traj(root / "metrics")
    diverged_files: set[str] = set()
    if not scalar_by_traj:
        print(f"[{CASE}] no per-cell metric files in {root / 'metrics'}")
    else:
        print(f"[{CASE}] scalar trajectories: {sorted(scalar_by_traj)} "
              f"({sum(len(v) for v in scalar_by_traj.values())} files)")
        scalar_by_traj, diverged = _split_diverged(scalar_by_traj)
        n_div: dict[tuple, int] = defaultdict(int)
        for d in diverged:
            n_div[_cell_key(d["method"], d["variant"], d["scenario"], d["M"])] += 1
        div_out = root / "aggregated" / "diverged.csv"
        div_out.parent.mkdir(parents=True, exist_ok=True)
        with open(div_out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["file", "traj", "method", "variant",
                                               "scenario", "E", "M"])
            w.writeheader()
            w.writerows(diverged)
        print(f"[{CASE}] {len(diverged)} diverged trajectory-cells (rmse NaN) "
              f"EXCLUDED from every metric -> {div_out}")
        for d in diverged:
            print(f"[{CASE}]   diverged: {d['file']}")
        diverged_files = {d["file"] for d in diverged}
        rows = aggregate_scalar(scalar_by_traj)

        # Replace the contended grid timings with the clean measurements BEFORE
        # anything is written, so all.csv and tables.tex can never disagree about
        # cost. Only `seconds` moves; every accuracy metric is left alone.
        if args.no_timing_override:
            print(f"[{CASE}] timing override DISABLED -- keeping the grid's own "
                  f"seconds (contended; see measure_timing.py)")
        else:
            timing = load_timing(timing_csv)
            if timing is None:
                print(f"[{CASE}] WARNING: no timing file at {timing_csv} -- "
                      f"keeping the grid's own seconds, which are CONTENDED and "
                      f"should not be quoted. Run measure_timing.py.")
            else:
                stats = apply_timing(rows, timing)
                print(f"[{CASE}] timings <- {timing_csv.name} "
                      f"({timing.n_rows} measured rows): {describe(stats)}")
                for cell in stats.get("_unmatched_preview", []) or []:
                    print(f"[{CASE}]   unmatched: {cell}")

        # (1a) The large-ensemble reference EnKF, appended AFTER the timing splice:
        # measure_timing.py never re-measured it, and the E-agnostic fallback in
        # apply_timing would otherwise hand it the 64-member EnKF's cost (~7 s
        # against its own ~100 s). It therefore keeps the seconds its own run
        # recorded -- contended like the rest of the grid's raw timings, so it is
        # an order-of-magnitude figure, not a clean measurement.
        if ref_metrics_csv.exists():
            ref_rows = aggregate_scalar_records(load_records_by_traj(ref_metrics_csv))
            rows.extend(ref_rows)
            Es = sorted({r["E"] for r in ref_rows if r.get("E") is not None})
            n_traj = max((int(r["n_traj"]) for r in ref_rows), default=0)
            print(f"[{CASE}] reference <- {ref_metrics_csv.name}: {len(ref_rows)} rows "
                  f"(E={Es}, {n_traj} trajectories); seconds are the run's own "
                  f"(not re-measured)")
        else:
            print(f"[{CASE}] no reference metrics at {ref_metrics_csv} -- no "
                  f"reference row. Build it with compute_ns_reference_metrics.py")

        out = root / "aggregated" / "all.csv"
        _write_scalar_csv_with_div(out, rows, n_div)
        print(f"[{CASE}] wrote {len(rows)} scalar rows -> {out}")
        print_scalar_table(rows, KEY_METRICS)

        # (1b) LaTeX tables, one per M, from those same aggregated rows -- in two
        # interchangeable notations, so the manuscript can \input whichever reads
        # better without either drifting from the CSVs:
        #   tables.tex         plain fixed-point (0.034)
        #   tables_scaled.tex  per-column units of 10^-n named in the header (3.400)
        # Same numbers, same bolding, distinct \label suffix (_scaled).
        for tex, scaled in ((root / "tables.tex", False),
                            (root / "tables_scaled.tex", True)):
            Ms = write_latex_tables(
                spec, rows, tex, with_std=not args.no_std, scaled=scaled,
                only_M=args.only_M,
                n_div=dict(n_div),
            )
            if Ms:
                print(f"[{CASE}] wrote {len(Ms)} LaTeX tables (M={Ms}) -> {tex}")
            else:
                print(f"[{CASE}] no rows carry an M; no LaTeX tables written")

        # (1c) The review tables, in the manuscript's layouts.
        tables_dir = args.tables_dir or args.results_root / "tables"
        _write_review_tables(tables_dir, rows, dict(n_div),
                             only_M=args.only_M, with_std=not args.no_std)

    # (2) PER-STEP curves, aggregated over trajectories at each step.
    ps_by_traj = discover_by_traj(root / "per_step")
    if scalar_by_traj and diverged_files:
        # Same exclusion as the scalars: a diverged trajectory's per-step curve is
        # finite only up to the guard, which would make n_traj jump mid-curve.
        ps_by_traj = {t: [p for p in ps if p.name not in diverged_files]
                      for t, ps in ps_by_traj.items()}
    if not ps_by_traj:
        print(f"[{CASE}] no per-step files in {root / 'per_step'} (skipping per_step.csv)")
    else:
        print(f"[{CASE}] per-step trajectories: {sorted(ps_by_traj)} "
              f"({sum(len(v) for v in ps_by_traj.values())} files)")
        ps_rows = aggregate_per_step(ps_by_traj)
        ps_out = root / "aggregated" / "per_step.csv"
        write_per_step_csv(ps_out, ps_rows)
        print(f"[{CASE}] wrote {len(ps_rows)} per-step rows -> {ps_out}")

    # (3) Forecast-correction grid, after the main all.csv it reads exists.
    if args.fc_root is not None:
        aggregate_fc(args.results_root, args.fc_root, args.fc_tag,
                     args.tables_dir or args.results_root / "tables", args.only_M)


if __name__ == "__main__":
    main()
