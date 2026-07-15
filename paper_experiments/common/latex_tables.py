"""LaTeX result tables, ONE TABLE PER SAMPLER-STEP COUNT $M$.

``aggregate_ns.py`` / ``aggregate_urban.py`` call :func:`write_latex_tables` with
the scalar rows they just reduced (mean over trajectories, from
:func:`common.aggregate_lib.aggregate_scalar`) and a per-case :class:`TableSpec`.
The result is a single ``results/<case>/tables.tex`` holding one ``table*`` per
distinct ``M`` in the data -- ready to ``\\input`` into the manuscript.

Layout (columns are METRIC GROUP x SCENARIO, plus one cost column):

    method | <metric 1> x scenarios | <metric 2> x scenarios | ... | s/step

Rows are the method lineup: our samplers rendered once per likelihood-covariance
mode present in the data (the tidy ``variant`` column -- ``jacfree`` /
``shared`` / ``shared_jac<k>``), each as a labelled sub-block, then the
solver-free baselines, then (NS) the classical true-solver filters.

Cells are the mean ACROSS TRAJECTORIES (with ``\\pm`` the across-trajectory std
when ``with_std``). A cell with no aggregated row -- a method/scenario/M that has
not been run, or whose every trajectory diverged to NaN and was dropped by the
aggregation -- prints ``--`` rather than a number, so a partially-run grid is
visibly partial instead of silently sparse. The caption is generated from the
data actually present (trajectory count, $E$, $M$), so it can never drift out of
sync with the numbers underneath it.

Requires ``booktabs`` (\\toprule/\\midrule/\\bottomrule) and ``amsmath``
(\\tfrac), both already used by the manuscript.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

# Printed for a (method, variant, scenario, metric, M) cell with no aggregated row.
MISSING_CELL = "--"


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TableSpec:
    """Everything case-specific about the table: columns, rows, caption wording.

    ``scenarios`` / ``metric_groups`` / ``methods`` hold the CANONICAL keys as they
    appear in the tidy rows (``results_schema`` enum values) paired with the LaTeX
    text to print, so the emitter never has to guess a display name.
    """

    case: str                                    # tidy ``case`` value
    case_label: str                              # caption lead-in, e.g. "Navier--Stokes"
    label_stem: str                              # \label{<stem>_M<M>}
    source: str                                  # generating script, named in the file header
    scenarios: tuple[tuple[str, str], ...]       # (tidy key, column header)
    metric_groups: tuple[tuple[str, str], ...]   # (tidy key, group header)
    ours: tuple[tuple[str, str], ...]            # (tidy key, row label)
    baselines: tuple[tuple[str, str], ...]       # solver-free baselines
    classical: tuple[tuple[str, str], ...] = ()  # true-solver filters (NS only)
    metrics_phrase: str = ""                     # caption: "vorticity RMSE, CRPS, ..."
    cost_metric: str = "seconds"
    cost_header: str = "s/step"
    fmt: str = "{:.3f}"
    cost_fmt: str = "{:.1f}"
    # Metrics where a LARGER number is better. Everything the three cases report
    # (RMSE, CRPS, |1 - spread/skill|, cost) is lower-is-better, so this is empty;
    # naming a metric here flips which cell gets bolded in its columns.
    higher_is_better: tuple[str, ...] = ()
    # Caption sentence(s) appended after the auto-generated data description.
    notes: str = ""


# Our samplers' likelihood-covariance modes (the tidy ``variant`` column) ->
# sub-block header. ``shared_jac<k>`` is the lagged-Jacobian shared mode: k is
# parsed out of the variant so any cadence renders without a new entry here.
_VARIANT_ORDER: tuple[str, ...] = ("shared", "jacfree")


def _variant_label(variant: str) -> str:
    """Sub-block header for one likelihood-covariance mode."""
    if variant == "jacfree":
        return "Ours -- Jacobian-free covariance"
    if variant == "shared":
        return "Ours -- ensemble-shared inflated covariance"
    if variant.startswith("shared_jac"):
        k = variant[len("shared_jac"):]
        return (
            "Ours -- ensemble-shared inflated covariance "
            rf"(Jacobian refreshed every $k={k}$ steps)"
        )
    if not variant:
        return "Ours"
    return f"Ours -- {variant}"


def _variant_rank(variant: str) -> tuple[int, str]:
    """Sort key: the shared modes first (the paper's headline), jacfree after."""
    if variant.startswith("shared"):
        return (0, variant)
    if variant == "jacfree":
        return (1, variant)
    return (2, variant)


# --------------------------------------------------------------------------- #
# Cell formatting
# --------------------------------------------------------------------------- #


def _fmt_number(x: float, fmt: str) -> str:
    """Fixed-point, falling back to scientific notation for blown-up magnitudes."""
    if abs(x) >= 1e4 or (x != 0.0 and abs(x) < 1e-3):
        return f"{x:.1e}".replace("e+0", r"e{+}").replace("e-0", r"e{-}")
    return fmt.format(x)


def _cell(
    value: float | None, std: float | None, *, fmt: str, with_std: bool,
    bold: bool = False,
) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return MISSING_CELL
    out = _fmt_number(value, fmt)
    if bold:
        out = rf"\textbf{{{out}}}"
    # The std is context, not the compared quantity -- it stays unbolded.
    if with_std and std is not None and not math.isnan(std):
        out += rf" {{\tiny $\pm$ {_fmt_number(std, fmt)}}}"
    return out


# --------------------------------------------------------------------------- #
# Index over the aggregated rows
# --------------------------------------------------------------------------- #


def _as_opt_float(v: object) -> float | None:
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return None
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_opt_int(v: object) -> int | None:
    f = _as_opt_float(v)
    return None if f is None else int(f)


def _norm_variant(v: object) -> str:
    s = "" if v is None else str(v).strip()
    return "" if s in ("", "None") else s


@dataclass
class _Index:
    """(method, variant, scenario, metric, M) -> (value, std), plus the M/E/n_traj sets."""

    by_key: dict[tuple[str, str, str, str, int | None], tuple[float | None, float | None]] = (
        field(default_factory=dict)
    )
    Ms: set[int] = field(default_factory=set)
    E_by_M: dict[int, set[int]] = field(default_factory=dict)
    ntraj_by_M: dict[int, set[int]] = field(default_factory=dict)
    variants_by_M: dict[int, set[str]] = field(default_factory=dict)

    def get(
        self, method: str, variant: str, scenario: str, metric: str, M: int | None
    ) -> tuple[float | None, float | None]:
        return self.by_key.get((method, variant, scenario, metric, M), (None, None))


def _build_index(rows: list[dict[str, object]], spec: TableSpec) -> _Index:
    idx = _Index()
    ours_keys = {m for m, _ in spec.ours}
    for r in rows:
        if str(r.get("case")) != spec.case:
            continue
        M = _as_opt_int(r.get("M"))
        method = str(r.get("method"))
        variant = _norm_variant(r.get("variant"))
        key = (method, variant, str(r.get("scenario")), str(r.get("metric")), M)
        idx.by_key[key] = (_as_opt_float(r.get("value")), _as_opt_float(r.get("std")))
        if M is None:
            continue
        idx.Ms.add(M)
        E = _as_opt_int(r.get("E"))
        if E is not None:
            idx.E_by_M.setdefault(M, set()).add(E)
        n = _as_opt_int(r.get("n_traj"))
        if n is not None:
            idx.ntraj_by_M.setdefault(M, set()).add(n)
        # Only OUR rows define the covariance-mode sub-blocks; a baseline has no variant.
        if method in ours_keys and variant:
            idx.variants_by_M.setdefault(M, set()).add(variant)
    return idx


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _column_format(spec: TableSpec) -> str:
    """``l|cccc|cccc|cccc|c`` -- one group of scenario columns per metric, then cost."""
    groups = "|".join("c" * len(spec.scenarios) for _ in spec.metric_groups)
    return f"l|{groups}|c"


def _header(spec: TableSpec) -> list[str]:
    n_scen = len(spec.scenarios)
    top = [
        rf"& \multicolumn{{{n_scen}}}{{c|}}{{\textbf{{{lbl}}}}}"
        for _, lbl in spec.metric_groups
    ]
    top.append(r"& \textbf{Cost} \\")
    scen_cells = " & ".join(h for _, h in spec.scenarios)
    sub = "        & " + " & ".join([scen_cells] * len(spec.metric_groups))
    sub += rf" & {spec.cost_header} \\"
    return ["        " + "\n        ".join(top), sub]


def _row_values(
    spec: TableSpec, idx: _Index, method: str, variant: str, M: int
) -> list[tuple[float | None, float | None]]:
    """One row's (value, std) per column: metric x scenario, then the cost column."""
    out: list[tuple[float | None, float | None]] = []
    for metric, _ in spec.metric_groups:
        for scenario, _ in spec.scenarios:
            out.append(idx.get(method, variant, scenario, metric, M))
    # Cost is scenario-independent (same sampler, same E): average the per-scenario
    # cost rows present rather than arbitrarily picking one scenario's.
    costs = [
        v for scenario, _ in spec.scenarios
        if (v := idx.get(method, variant, scenario, spec.cost_metric, M)[0]) is not None
        and not math.isnan(v)
    ]
    out.append((sum(costs) / len(costs) if costs else None, None))
    return out


def _best_in_column(
    values: list[float | None], fmt: str, *, lower_is_better: bool
) -> set[int]:
    """Row indices holding the best value in one column, ties included.

    "Best" is decided on the value as PRINTED (``fmt``, i.e. to the third decimal
    for the metric columns): every row whose rendered number equals the rendered
    winner is returned, so cells that are indistinguishable in the table are all
    bolded rather than one being silently favoured by invisible digits. A column
    with fewer than two numbers has no winner -- bolding the only entry present
    would claim a comparison that was never made.
    """
    present = [
        (i, v) for i, v in enumerate(values)
        if v is not None and not math.isnan(v)
    ]
    if len(present) < 2:
        return set()
    pick = max if lower_is_better is False else min
    best = pick(v for _, v in present)
    best_str = _fmt_number(best, fmt)
    return {i for i, v in present if _fmt_number(v, fmt) == best_str}


def _caption(spec: TableSpec, idx: _Index, M: int, *, with_std: bool) -> str:
    ntraj = sorted(idx.ntraj_by_M.get(M, set()))
    Es = sorted(idx.E_by_M.get(M, set()))
    if not ntraj:
        traj_txt = "held-out trajectories"
    elif len(ntraj) == 1:
        n = ntraj[0]
        traj_txt = f"{n} held-out trajector" + ("y" if n == 1 else "ies")
    else:
        traj_txt = f"{ntraj[0]}--{ntraj[-1]} held-out trajectories (cell-dependent)"
    E_txt = f"$E={Es[0]}$" if len(Es) == 1 else "$E \\in \\{" + ", ".join(map(str, Es)) + "\\}$"

    stat = "Mean $\\pm$ std" if with_std else "Mean"
    metrics_phrase = spec.metrics_phrase or "metrics"
    n_scen = {2: "two", 3: "three", 4: "four"}.get(len(spec.scenarios), str(len(spec.scenarios)))
    parts = [
        f"{spec.case_label}: {metrics_phrase} across the {n_scen} observation "
        f"scenarios, with per-step cost (lower is better).",
        f"{stat} over {traj_txt} at {E_txt}, $M={M}$ sampler steps.",
    ]
    parts.append(
        r"\textbf{Bold} marks the best value in each column (ties to the printed "
        "precision are all bolded)."
    )
    if spec.notes:
        parts.append(spec.notes)
    parts.append(
        rf"``{MISSING_CELL}'' marks a cell with no result at this $M$ "
        "(not run, or every trajectory diverged)."
    )
    return " ".join(parts)


def render_table(
    spec: TableSpec, idx: _Index, M: int, *, with_std: bool = False
) -> str:
    """One ``table*`` for a single sampler-step count ``M``.

    Rendered in two passes: the whole table's values are collected first so the
    best entry in each column (over EVERY row -- both covariance modes, the
    baselines and the classical filters) can be bolded, then the rows are emitted.
    """
    span = 1 + len(spec.metric_groups) * len(spec.scenarios) + 1  # + method + cost

    # Pass 1: the row plan + its value grid. Each entry carries the decoration to
    # emit before it (a \midrule, and for our samplers a covariance-mode header),
    # so the emit loop below is a straight walk with no group bookkeeping.
    @dataclass
    class _Row:
        method: str
        label: str
        variant: str
        rule: bool = False        # emit \midrule before this row
        head: str | None = None   # emit a \multicolumn sub-block header before it

    plan: list[_Row] = []
    variants = sorted(idx.variants_by_M.get(M, set()), key=_variant_rank)
    if not variants:
        variants = [""]  # no variant column in the data -> a single unlabelled block
    for b, variant in enumerate(variants):
        for i, (method, label) in enumerate(spec.ours):
            plan.append(_Row(
                method, label, variant,
                rule=(i == 0 and b > 0),
                head=_variant_label(variant) if i == 0 else None,
            ))
    for group in (spec.baselines, spec.classical):
        for i, (method, label) in enumerate(group):
            plan.append(_Row(method, label, "", rule=(i == 0)))

    grid = [_row_values(spec, idx, r.method, r.variant, M) for r in plan]
    n_cols = len(spec.metric_groups) * len(spec.scenarios) + 1
    # Every metric here (RMSE, CRPS, |1 - spread/skill|) and the cost are
    # lower-is-better; ``higher_is_better`` names any exception.
    col_metric = [m for m, _ in spec.metric_groups for _ in spec.scenarios]
    col_metric.append(spec.cost_metric)
    best: list[set[int]] = [
        _best_in_column(
            [row[c][0] for row in grid],
            spec.cost_fmt if c == n_cols - 1 else spec.fmt,
            lower_is_better=col_metric[c] not in spec.higher_is_better,
        )
        for c in range(n_cols)
    ]

    # Pass 2: emit.
    lines: list[str] = [
        r"\begin{table*}[ht]",
        r"    \centering",
        r"    \scriptsize",
        r"    \setlength{\tabcolsep}{3.5pt}",
        rf"    \caption{{{_caption(spec, idx, M, with_std=with_std)}}}",
        rf"    \label{{{spec.label_stem}_M{M}}}",
        rf"    \begin{{tabular}}{{{_column_format(spec)}}}",
        r"        \toprule",
        *_header(spec),
        r"        \midrule",
    ]
    for r, row in enumerate(plan):
        if row.rule:
            lines.append(r"        \midrule")
        if row.head is not None:
            lines.append(
                rf"        \multicolumn{{{span}}}{{l}}{{\textit{{{row.head}}}}} \\"
            )
        cells = [
            _cell(
                v, s,
                fmt=spec.cost_fmt if c == n_cols - 1 else spec.fmt,
                with_std=with_std and c != n_cols - 1,
                bold=r in best[c],
            )
            for c, (v, s) in enumerate(grid[r])
        ]
        lines.append(f"        {row.label} & " + " & ".join(cells) + r" \\")

    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def write_latex_tables(
    spec: TableSpec,
    rows: list[dict[str, object]],
    out: Path,
    *,
    with_std: bool = False,
) -> list[int]:
    """Write one table per M in ``rows`` to ``out``; return the Ms rendered.

    ``rows`` are the aggregated scalar rows (:func:`aggregate_scalar` output).
    Returns ``[]`` (and writes nothing) when the case has no rows with an ``M``.
    """
    idx = _build_index(rows, spec)
    Ms = sorted(idx.Ms)
    if not Ms:
        return []
    body = "\n\n".join(render_table(spec, idx, M, with_std=with_std) for M in Ms)
    header = (
        f"% Auto-generated by {spec.source} -- DO NOT EDIT BY HAND.\n"
        "% One table per sampler-step count M; regenerate after every aggregation.\n"
        f"% M values present: {', '.join(map(str, Ms))}.\n"
        "% Requires: booktabs, amsmath.\n\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + body + "\n", encoding="utf-8")
    return Ms


__all__ = ["TableSpec", "write_latex_tables", "render_table", "MISSING_CELL"]
