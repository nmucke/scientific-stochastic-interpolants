"""Emit LaTeX table snippets from a tidy results file.

This is the E3 deliverable: a working converter that reads the canonical tidy
results file (see :mod:`results_schema`) and produces, for every labelled table
in ``manuscript/sections/results.tex``, a standalone ``.tex`` snippet that the
paper can ``\\input``.

Run::

    python paper_experiments/make_tables.py --results <tidy.csv> --out paper_experiments/generated

or, with no real numbers yet, prove the pipeline end-to-end on a synthetic file::

    python paper_experiments/make_tables.py --demo

Each emitted snippet is the ``tabular`` body (the rows between ``\\midrule``s)
for one table, so it slots straight into the matching ``\\begin{tabular}`` in
``results.tex``. The mapping from a tidy (method, scenario, metric) triple to a
column is defined declaratively in ``TABLE_SPECS`` below and documented in the
module docstring of each spec.

--------------------------------------------------------------------------- ##
Mapping overview (tidy row  ->  results.tex label)
--------------------------------------------------------------------------- ##

  tab:analytical_results   metric in {kl_points, sliced_w2}, scenario=analytical
  tab:ns_accuracy          case=navier_stokes; cols = (metric x scenario) of
                           {rmse, energy_spec_rmse, kl_points} x {32^2->128^2, 5%}
  tab:ns_calibration_cost  case=navier_stokes; cols = {crps, spread_skill} x
                           {32^2->128^2, 5%} plus {nfe, seconds}
  tab:urban_accuracy       case=urban; {rmse_velocity, rmse_temperature,
                           kl_points} x {32^2->128^2, 5%}
  tab:urban_calibration_cost  case=urban; mirror of ns_calibration_cost
  tab:ablation             special: ablation rows (handled separately)

The (case, method, scenario, metric) keys are the canonical enum *values* from
``results_schema``; nothing here hard-codes a magic string that is not also a
schema constant.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from results_schema import (
    Case,
    Method,
    Metric,
    ResultRecord,
    Scenario,
    load_records,
    write_records,
)

# Placeholder printed when a (method, scenario, metric) cell has no data.
MISSING_CELL = "--"


# --------------------------------------------------------------------------- #
# Row ordering and LaTeX labels for methods
# --------------------------------------------------------------------------- #

# Order methods appear as rows, with a rule (None) separating our samplers from
# the baselines -- matching the \midrule structure already in results.tex.
_OURS: tuple[Method, ...] = (
    Method.OURS_SI_SDE,
    Method.OURS_FM_ODE,
    Method.OURS_DM_SDE,
)
# Generative baselines, grouped: SI+SDE, then FM+ODE, then DM+SDE.
_GENERATIVE_BASELINES: tuple[Method, ...] = (
    Method.FLOWDAS,
    Method.GUIDED_FM_OTODE,
    Method.D_FLOW_SGLD,
    Method.SDA,
    Method.SURGE,
)
_ANALYTICAL_BASELINES: tuple[Method, ...] = _GENERATIVE_BASELINES + (
    Method.ENKF,
    Method.PARTICLE_FILTER,
)
_FIELD_BASELINES: tuple[Method, ...] = _GENERATIVE_BASELINES + (
    # Classical / true-solver baselines (EnSF is now a true-solver method too).
    Method.ENKF,
    Method.LETKF,
    Method.PARTICLE_FILTER,
    Method.ENSEMBLE_SCORE_FILTER,
)

# Map a method to the exact LaTeX row label (with \cite) used in results.tex.
METHOD_LATEX_LABEL: dict[Method, str] = {
    Method.OURS_SI_SDE: r"Ours (SI-SDE)",
    Method.OURS_FM_ODE: r"Ours (FM-ODE)",
    Method.OURS_DM_SDE: r"Ours (DM-SDE)",
    Method.FLOWDAS: r"FlowDAS \cite{chen_flowdas_2025}",
    Method.GUIDED_FM_FIG: r"Guided FM (FIG) \cite{yan_fig_2025}",
    Method.GUIDED_FM_OTODE: r"Guided FM (OT-ODE) \cite{pokle_training-free_2024}",
    Method.D_FLOW_SGLD: r"D-Flow SGLD \cite{ben-hamu_d-flow_2024}",
    Method.SDA: r"SDA \cite{rozet_score-based_2023}",
    Method.SURGE: r"SURGE",
    Method.SURGE_SDA: r"SURGE + SDA",
    Method.SURGE_FLOWDAS: r"SURGE + FlowDAS",
    Method.ENSEMBLE_SCORE_FILTER: r"Ensemble score filter \cite{bao_ensemble_2024}",
    Method.ENKF: r"EnKF \cite{evensen_data_2022}",
    Method.LETKF: r"LETKF \cite{hunt_efficient_2007}",
    Method.PARTICLE_FILTER: r"Particle filter \cite{carrassi_data_2018}",
}


# --------------------------------------------------------------------------- #
# Cell formatting
# --------------------------------------------------------------------------- #


def format_cell(value: float | None, std: float | None, *, fmt: str = "{:.3f}") -> str:
    """Render one cell as ``mean`` or ``mean $\\pm$ std`` (or the missing dash)."""
    if value is None:
        return MISSING_CELL
    if std is None:
        return fmt.format(value)
    return f"{fmt.format(value)} $\\pm$ {fmt.format(std)}"


# --------------------------------------------------------------------------- #
# Lookup index over tidy records
# --------------------------------------------------------------------------- #


class RecordIndex:
    """Index aggregated tidy rows by (case, method, scenario, metric).

    Only *aggregated* rows (one row per key, carrying mean ``value`` and across-
    seed ``std``) are expected here. If several rows share a key (e.g. raw
    per-seed rows leaked in), they are reduced with
    :func:`common.aggregation.aggregate_over_seeds`-compatible mean/std at lookup
    so the emitter never silently drops data -- but the intended input is the
    already-aggregated file.
    """

    def __init__(self, records: list[ResultRecord]) -> None:
        self._by_key: dict[tuple[str, str, str, str], list[ResultRecord]] = (
            defaultdict(list)
        )
        for r in records:
            self._by_key[(r.case, r.method, r.scenario, r.metric)].append(r)

    def get(
        self, case: str, method: str, scenario: str, metric: str
    ) -> tuple[float | None, float | None]:
        rows = self._by_key.get((case, method, scenario, metric))
        if not rows:
            return None, None
        if len(rows) == 1:
            return rows[0].value, rows[0].std
        # Reduce stray duplicates defensively (mean of values, pooled std).
        vals = [r.value for r in rows]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
        return mean, var**0.5


# --------------------------------------------------------------------------- #
# Declarative table specifications
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Column:
    """One data column: a (metric, scenario) pair selecting tidy rows."""

    metric: Metric
    scenario: Scenario


@dataclass(frozen=True)
class TableSpec:
    """A labelled results.tex table.

    Attributes
    ----------
    label: the ``\\label`` in results.tex (e.g. ``tab:ns_accuracy``).
    case: which case's rows feed it.
    methods: ordered method rows (a leading group then baselines).
    n_ours: how many leading rows are "ours" (a \\midrule is emitted after them).
    columns: ordered data columns.
    fmt: per-cell number format.
    """

    label: str
    case: Case
    methods: tuple[Method, ...]
    n_ours: int
    columns: tuple[Column, ...]
    fmt: str = "{:.3f}"


# tab:analytical_results -- KL and sliced-W2 to the exact posterior.
SPEC_ANALYTICAL = TableSpec(
    label="tab:analytical_results",
    case=Case.ANALYTICAL,
    methods=_OURS + _ANALYTICAL_BASELINES,
    n_ours=len(_OURS),
    columns=(
        Column(Metric.KL_POINTS, Scenario.ANALYTICAL),
        Column(Metric.SLICED_W2, Scenario.ANALYTICAL),
    ),
)

# tab:ns_accuracy -- (RMSE | energy-spec RMSE | KL) x (32^2->128^2 | 5%).
SPEC_NS_ACCURACY = TableSpec(
    label="tab:ns_accuracy",
    case=Case.NAVIER_STOKES,
    methods=_OURS + _FIELD_BASELINES,
    n_ours=len(_OURS),
    columns=(
        Column(Metric.RMSE, Scenario.SUPERRES_32),
        Column(Metric.RMSE, Scenario.SPARSE_5),
        Column(Metric.ENERGY_SPEC_RMSE, Scenario.SUPERRES_32),
        Column(Metric.ENERGY_SPEC_RMSE, Scenario.SPARSE_5),
        Column(Metric.KL_POINTS, Scenario.SUPERRES_32),
        Column(Metric.KL_POINTS, Scenario.SPARSE_5),
    ),
)

# tab:ns_calibration_cost -- (CRPS | spread-skill) x scenarios + (NFE | s/step).
# Cost columns are scenario-independent; we read them from the 32^2->128^2 rows
# (cost is the same across scenarios at matched E by construction).
SPEC_NS_CALIBRATION_COST = TableSpec(
    label="tab:ns_calibration_cost",
    case=Case.NAVIER_STOKES,
    methods=_OURS + _FIELD_BASELINES,
    n_ours=len(_OURS),
    columns=(
        Column(Metric.CRPS, Scenario.SUPERRES_32),
        Column(Metric.CRPS, Scenario.SPARSE_5),
        Column(Metric.SPREAD_SKILL, Scenario.SUPERRES_32),
        Column(Metric.SPREAD_SKILL, Scenario.SPARSE_5),
        Column(Metric.NFE, Scenario.SUPERRES_32),
        Column(Metric.SECONDS, Scenario.SUPERRES_32),
    ),
    fmt="{:.3f}",
)

# Urban methods: generative-only (no true solver => no EnKF/LETKF/PF/EnSF; the
# Ensemble Score Filter is now a true-solver method too. See urban driver.)
_URBAN_METHODS = _OURS + _GENERATIVE_BASELINES

# tab:urban_accuracy -- (velocity RMSE | temperature RMSE) x scenarios. No KL: urban
# has no ground-truth posterior to reference (only a ground-truth state).
SPEC_URBAN_ACCURACY = TableSpec(
    label="tab:urban_accuracy",
    case=Case.URBAN,
    methods=_URBAN_METHODS,
    n_ours=len(_OURS),
    columns=(
        Column(Metric.RMSE_VELOCITY, Scenario.SUPERRES_32),
        Column(Metric.RMSE_VELOCITY, Scenario.SPARSE_5),
        Column(Metric.RMSE_TEMPERATURE, Scenario.SUPERRES_32),
        Column(Metric.RMSE_TEMPERATURE, Scenario.SPARSE_5),
    ),
)

# tab:urban_calibration_cost -- spread--skill + CRPS (vs the ground-truth state) + cost.
SPEC_URBAN_CALIBRATION_COST = TableSpec(
    label="tab:urban_calibration_cost",
    case=Case.URBAN,
    methods=_URBAN_METHODS,
    n_ours=len(_OURS),
    columns=(
        Column(Metric.CRPS, Scenario.SUPERRES_32),
        Column(Metric.CRPS, Scenario.SPARSE_5),
        Column(Metric.SPREAD_SKILL, Scenario.SUPERRES_32),
        Column(Metric.SPREAD_SKILL, Scenario.SPARSE_5),
        Column(Metric.NFE, Scenario.SUPERRES_32),
        Column(Metric.SECONDS, Scenario.SUPERRES_32),
    ),
)

# All method-row tables (tab:ablation is handled by a dedicated emitter).
TABLE_SPECS: tuple[TableSpec, ...] = (
    SPEC_ANALYTICAL,
    SPEC_NS_ACCURACY,
    SPEC_NS_CALIBRATION_COST,
    SPEC_URBAN_ACCURACY,
    SPEC_URBAN_CALIBRATION_COST,
)


# --------------------------------------------------------------------------- #
# Emitters
# --------------------------------------------------------------------------- #


def render_table_body(spec: TableSpec, index: RecordIndex) -> str:
    """Render the tabular body (rows + the ours/baselines \\midrule) for a spec."""
    lines: list[str] = []
    for i, method in enumerate(spec.methods):
        if i == spec.n_ours:
            lines.append(r"\midrule")
        label = METHOD_LATEX_LABEL[method]
        cells = []
        for col in spec.columns:
            value, std = index.get(
                spec.case.value, method.value, col.scenario.value, col.metric.value
            )
            cells.append(format_cell(value, std, fmt=spec.fmt))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


# Ablation table is structurally different: fixed configuration row labels rather
# than the method list. We key ablation rows by an explicit ``scenario`` tag set
# by the ablation driver, with metric in {rmse, crps, spread_skill}.
ABLATION_ROWS: tuple[tuple[str, str], ...] = (
    # (LaTeX row label, scenario tag emitted by the ablation driver)
    (r"Inflated covariance $\bar\Sigma_\tau$", "ablation:cov_inflated"),
    (r"Jacobian-free covariance", "ablation:cov_jacfree"),
    (r"$\gdiff_\tau$ sweep (low/med/high)", "ablation:gdiff_sweep"),
    (r"Steps $M$ (e.g.\ $10/50/100$)", "ablation:steps_sweep"),
    (r"Ensemble $E$ (e.g.\ $16/64/256$)", "ablation:ensemble_sweep"),
)
ABLATION_MIDRULE_AFTER = 2  # \midrule after the two covariance rows.
ABLATION_METRICS: tuple[Metric, ...] = (
    Metric.RMSE,
    Metric.CRPS,
    Metric.SPREAD_SKILL,
)


def render_ablation_body(index: RecordIndex) -> str:
    """Render the tab:ablation body.

    Ablation tidy rows use ``case=navier_stokes``, ``method=Ours (DM-SDE)`` (the
    sampler the ablations vary), and the ablation tag in the ``scenario`` column.
    """
    lines: list[str] = []
    for i, (label, tag) in enumerate(ABLATION_ROWS):
        if i == ABLATION_MIDRULE_AFTER:
            lines.append(r"\midrule")
        cells = []
        for metric in ABLATION_METRICS:
            # Any "ours" method may carry the ablation; DM-SDE is the canonical one.
            value, std = index.get(
                Case.NAVIER_STOKES.value,
                Method.OURS_DM_SDE.value,
                tag,
                metric.value,
            )
            cells.append(format_cell(value, std))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def _snippet_filename(label: str) -> str:
    """``tab:ns_accuracy`` -> ``tab_ns_accuracy.tex``."""
    return label.replace(":", "_") + ".tex"


def emit_all(records: list[ResultRecord], out_dir: str | Path) -> list[Path]:
    """Write every table snippet to ``out_dir``; return the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = RecordIndex(records)
    written: list[Path] = []

    for spec in TABLE_SPECS:
        body = render_table_body(spec, index)
        path = out_dir / _snippet_filename(spec.label)
        path.write_text(_wrap_snippet(spec.label, body), encoding="utf-8")
        written.append(path)

    abl_path = out_dir / _snippet_filename("tab:ablation")
    abl_path.write_text(
        _wrap_snippet("tab:ablation", render_ablation_body(index)),
        encoding="utf-8",
    )
    written.append(abl_path)
    return written


def _wrap_snippet(label: str, body: str) -> str:
    return (
        f"% Auto-generated by paper_experiments/make_tables.py for {label}.\n"
        f"% Do not edit by hand; regenerate from the tidy results file.\n"
        f"% \\input this between the \\toprule/header and \\bottomrule of the\n"
        f"% matching \\begin{{tabular}} in sections/results.tex.\n"
        f"{body}\n"
    )


# --------------------------------------------------------------------------- #
# Synthetic demo data (proves the pipeline before real numbers exist)
# --------------------------------------------------------------------------- #


def make_demo_records(seed: int = 0) -> list[ResultRecord]:
    """Build a small synthetic tidy dataset covering every table/cell."""
    rng = random.Random(seed)
    records: list[ResultRecord] = []

    def add(
        case: Case,
        method: Method,
        scenario: str,
        metric: Metric,
        base: float,
    ) -> None:
        value = base * (1.0 + 0.1 * rng.random())
        std = 0.05 * value
        records.append(
            ResultRecord(
                case=case.value,
                method=method.value,
                scenario=scenario,
                metric=metric.value,
                value=value,
                std=std,
                E=64,
                M=50,
                seed=-1,
                nfe=50.0 if method in _OURS else 100.0,
                seconds=0.4 if method in _OURS else 0.9,
            )
        )

    # Case 1 -- analytical.
    for m in SPEC_ANALYTICAL.methods:
        add(Case.ANALYTICAL, m, Scenario.ANALYTICAL.value, Metric.KL_POINTS, 0.02)
        add(Case.ANALYTICAL, m, Scenario.ANALYTICAL.value, Metric.SLICED_W2, 0.05)

    # Case 2 -- Navier-Stokes (accuracy + calibration/cost).
    ns_scen = [Scenario.SUPERRES_32, Scenario.SPARSE_5]
    for m in SPEC_NS_ACCURACY.methods:
        for s in ns_scen:
            add(Case.NAVIER_STOKES, m, s.value, Metric.RMSE, 0.15)
            add(Case.NAVIER_STOKES, m, s.value, Metric.ENERGY_SPEC_RMSE, 0.30)
            add(Case.NAVIER_STOKES, m, s.value, Metric.KL_POINTS, 0.08)
            add(Case.NAVIER_STOKES, m, s.value, Metric.CRPS, 0.07)
            add(Case.NAVIER_STOKES, m, s.value, Metric.SPREAD_SKILL, 0.12)
        add(Case.NAVIER_STOKES, m, Scenario.SUPERRES_32.value, Metric.NFE, 50.0)
        add(Case.NAVIER_STOKES, m, Scenario.SUPERRES_32.value, Metric.SECONDS, 0.5)

    # Case 3 -- urban (per-variable RMSE + calibration/cost).
    for m in SPEC_URBAN_ACCURACY.methods:
        for s in ns_scen:
            add(Case.URBAN, m, s.value, Metric.RMSE_VELOCITY, 0.20)
            add(Case.URBAN, m, s.value, Metric.RMSE_TEMPERATURE, 0.18)
            add(Case.URBAN, m, s.value, Metric.KL_POINTS, 0.10)
            add(Case.URBAN, m, s.value, Metric.CRPS, 0.09)
            add(Case.URBAN, m, s.value, Metric.SPREAD_SKILL, 0.14)
        add(Case.URBAN, m, Scenario.SUPERRES_32.value, Metric.NFE, 50.0)
        add(Case.URBAN, m, Scenario.SUPERRES_32.value, Metric.SECONDS, 0.6)

    # Ablations (DM-SDE on NS, ablation tags in the scenario column).
    for _, tag in ABLATION_ROWS:
        add(Case.NAVIER_STOKES, Method.OURS_DM_SDE, tag, Metric.RMSE, 0.15)
        add(Case.NAVIER_STOKES, Method.OURS_DM_SDE, tag, Metric.CRPS, 0.07)
        add(Case.NAVIER_STOKES, Method.OURS_DM_SDE, tag, Metric.SPREAD_SKILL, 0.10)

    return records


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Tidy results file (.csv or .jsonl). Omit with --demo.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "generated",
        help="Directory for the emitted .tex snippets.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate a synthetic tidy file and emit tables from it.",
    )
    args = parser.parse_args()

    if args.demo:
        demo_path = args.out / "demo_results.csv"
        records = make_demo_records()
        write_records(demo_path, records)
        print(f"[make_tables] wrote synthetic tidy results -> {demo_path}")
    else:
        if args.results is None:
            parser.error("provide --results <file> or use --demo")
        records = load_records(args.results)
        print(f"[make_tables] loaded {len(records)} rows from {args.results}")

    written = emit_all(records, args.out)
    for p in written:
        print(f"[make_tables] emitted {p}")


if __name__ == "__main__":
    main()
