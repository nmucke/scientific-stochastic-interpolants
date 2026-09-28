"""LaTeX result tables, ONE TABLE PER SAMPLER-STEP COUNT $M$.

``aggregate_ns.py`` / ``aggregate_urban.py`` call :func:`write_latex_tables` with
the scalar rows they just reduced (mean over trajectories, from
:func:`common.aggregate_lib.aggregate_scalar`) and a per-case :class:`TableSpec`.
The result is a single ``results/<case>/tables.tex`` holding one ``table*`` per
distinct ``M`` in the data -- ready to ``\\input`` into the manuscript.

Layout (columns are METRIC GROUP x SCENARIO, with cost as a final such group):

    method | <metric 1> x scenarios | <metric 2> x scenarios | ... | cost x scenarios

Cost is reported PER SCENARIO, not as one averaged column. It used to be averaged
on the assumption that cost is scenario-independent (same sampler, same $E$), but
that is false for the guided methods: their work scales with the OBSERVATION COUNT
-- the shared Jacobian costs one JVP per observation -- so e.g. urban sparse 5%
($N_y\\approx2081$) is ~3.2x dearer than sparse 1.5625% ($N_y\\approx650$) at
identical $M$. A single averaged number hid a spread that large.

Rows are the method lineup: our samplers rendered once per likelihood-covariance
mode present in the data (the tidy ``variant`` column -- ``jacfree`` /
``shared`` / ``shared_jac<k>``), each as a labelled sub-block, then the
solver-free baselines, then (NS) the classical true-solver filters, then any
out-of-competition ``reference`` rows. The filters have no sampler-step count --
they are run once, under whatever M the cell was labelled with -- so their row is
repeated verbatim in every $M$ table instead of printing ``--`` in all but one
(see :meth:`_Index.get_any_M`); the same holds for the reference rows.

Cells are the mean ACROSS TRAJECTORIES (with ``\\pm`` the across-trajectory std
when ``with_std``). A cell with no aggregated row -- a method/scenario/M that has
not been run, or whose every trajectory diverged to NaN and was dropped by the
aggregation -- prints ``--`` rather than a number, so a partially-run grid is
visibly partial instead of silently sparse. The caption is generated from the
data actually present (trajectory count, $E$, $M$), so it can never drift out of
sync with the numbers underneath it.

Requires ``booktabs`` (\\toprule/\\midrule/\\bottomrule), ``amsmath`` (\\tfrac)
and ``graphicx`` (the shrink-to-fit \\resizebox around each tabular), all already
used by the manuscript.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

# Printed for a (method, variant, scenario, metric, M) cell with no aggregated row.
MISSING_CELL = "--"

# A column's winner is bolded together with every value that MATCHES IT TO THIS
# MANY DECIMALS. The metrics are printed to three decimals, but a third-decimal
# gap is not a difference anyone should read as a ranking -- it is well inside the
# across-trajectory spread -- so methods that tie at two decimals are all bolded.
BOLD_DECIMALS = 2


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
    # Rows run OUT OF COMPETITION -- the NS reference EnKF, whose ensemble is far
    # larger than everyone else's. They print in a final block, annotated with the
    # $E$ they were run at, and are excluded from the bolding and from the
    # caption's $E$: they are context for how far the graded methods are from a
    # well-resourced filter, not an entry in the same comparison.
    reference: tuple[tuple[str, str], ...] = ()
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
    # Whether to explain the ``--`` placeholder in the caption.
    include_missing_cell_note: bool = True


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
    Ms_by_method: dict[str, set[int]] = field(default_factory=dict)
    E_by_method: dict[str, set[int]] = field(default_factory=dict)

    def get(
        self, method: str, variant: str, scenario: str, metric: str, M: int | None
    ) -> tuple[float | None, float | None]:
        return self.by_key.get((method, variant, scenario, metric, M), (None, None))

    def get_any_M(
        self, method: str, variant: str, scenario: str, metric: str, M: int | None
    ) -> tuple[float | None, float | None]:
        """:meth:`get`, but falling back to the method's own $M$ when it has only one.

        For an M-INDEPENDENT method -- the classical filters have no sampler-step
        count, so the grid runs each exactly once and the tidy rows carry whatever
        M the cell was labelled with -- the same numbers belong in every table.
        Without this they would print ``--`` in all but one table, reading as "not
        run" rather than "does not depend on $M$". If such a method somehow has
        several Ms, no fallback is applied (there would be no unambiguous choice).
        """
        hit = self.get(method, variant, scenario, metric, M)
        if hit != (None, None):
            return hit
        own = self.Ms_by_method.get(method, set())
        if len(own) != 1:
            return hit
        return self.get(method, variant, scenario, metric, next(iter(own)))


def _build_index(rows: list[dict[str, object]], spec: TableSpec) -> _Index:
    idx = _Index()
    ours_keys = {m for m, _ in spec.ours}
    ref_keys = {m for m, _ in spec.reference}
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
        E = _as_opt_int(r.get("E"))
        if E is not None:
            idx.E_by_method.setdefault(method, set()).add(E)
        idx.Ms_by_method.setdefault(method, set()).add(M)
        # An out-of-competition reference contributes NOTHING to the table-level
        # sets: its ensemble is not the graded $E$, and its M is only the label its
        # single run happened to carry -- letting it into idx.Ms would mint a table
        # for an M nobody ran the lineup at.
        if method in ref_keys:
            continue
        idx.Ms.add(M)
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
    """``l|cccc|cccc|cccc|cccc`` -- one group of scenario columns per metric, then
    the SAME scenario columns again for cost (cost is per-scenario, not averaged)."""
    groups = "|".join("c" * len(spec.scenarios)
                      for _ in range(len(spec.metric_groups) + 1))
    return f"l|{groups}"


def _rescale(
    value: float | None, std: float | None, exp: int
) -> tuple[float | None, float | None]:
    """Express ``(value, std)`` in units of $10^{-exp}$ (``exp=0`` -> unchanged)."""
    if exp == 0:
        return value, std
    f = 10.0 ** exp
    return (
        None if value is None else value * f,
        None if std is None else std * f,
    )


def _group_exponent(values: list[float | None]) -> int:
    """``n`` such that a metric group reads best in units of $10^{-n}$ (0 = none).

    One exponent per METRIC GROUP -- over every scenario column it spans -- so the
    factor is written once, next to the metric name, rather than repeated on each
    scenario sub-header.

    Chosen from the group's MEDIAN magnitude, so the bulk of it lands in $[1, 10)$:
    a group sitting around 0.034 gets ``n=2`` and prints 3.400 instead of 0.034,
    recovering two significant digits that the fixed three-decimal format was
    throwing away. The median rather than the maximum, because one blown-up
    baseline (the particle filter is an order of magnitude off in most NS columns)
    would otherwise pin the group at ``n=0`` and waste the notation on every other
    row. The cap keeps the largest entry under 100 so that outlier stays readable,
    and a group already at order 1 or above (cost) gets ``n=0`` -- scaling that
    would only add a factor to read past.
    """
    present = sorted(
        abs(v) for v in values
        if v is not None and not math.isnan(v) and v != 0.0
    )
    if not present:
        return 0
    median = present[len(present) // 2]
    n_bulk = -math.floor(math.log10(median))
    n_cap = -math.floor(math.log10(present[-1])) + 1   # keeps max < 100
    return max(0, min(n_bulk, n_cap))


def _header(spec: TableSpec, exps: list[int] | None = None) -> list[str]:
    """Two header rows: metric groups over the scenario columns they span.

    ``exps`` (one per GROUP -- the metric groups then cost, from
    :func:`_group_exponent`) puts the units on the group header, beside the metric
    name, where it is written once and covers that group's whole block of scenario
    columns.
    """
    n_scen = len(spec.scenarios)
    group_labels = [lbl for _, lbl in spec.metric_groups]
    group_labels.append(f"Cost ({spec.cost_header})")
    if exps is not None:
        group_labels = [
            lbl if e == 0 else rf"{lbl} ($\times 10^{{-{e}}}$)"
            for lbl, e in zip(group_labels, exps)
        ]
    top = [
        rf"& \multicolumn{{{n_scen}}}{{c|}}{{\textbf{{{lbl}}}}}"
        for lbl in group_labels[:-1]
    ]
    # Cost gets its own scenario-wide block, like any other metric group. It is the
    # LAST block, so it closes with `c` rather than `c|`.
    top.append(rf"& \multicolumn{{{n_scen}}}{{c}}{{\textbf{{{group_labels[-1]}}}}} \\")
    scen_cells = " & ".join(h for _, h in spec.scenarios)
    sub = "        & " + " & ".join([scen_cells] * (len(spec.metric_groups) + 1))
    sub += r" \\"
    return ["        " + "\n        ".join(top), sub]


def _row_values(
    spec: TableSpec, idx: _Index, method: str, variant: str, M: int,
    *, m_independent: bool = False,
) -> list[tuple[float | None, float | None]]:
    """One row's (value, std) per column: metric x scenario, then cost x scenario.

    ``m_independent`` rows (the classical filters, which have no sampler-step
    count) fall back to the single M they were run at, so they appear in EVERY
    table rather than in one.
    """
    get = idx.get_any_M if m_independent else idx.get
    out: list[tuple[float | None, float | None]] = []
    for metric, _ in spec.metric_groups:
        for scenario, _ in spec.scenarios:
            out.append(get(method, variant, scenario, metric, M))
    # Cost PER SCENARIO. It used to be one averaged column on the assumption that
    # cost is scenario-independent (same sampler, same E) -- that is false for the
    # guided methods, whose work scales with the OBSERVATION COUNT: the shared
    # Jacobian costs one JVP per observation, so e.g. urban sparse 5% (N_y~2081)
    # is ~3.2x dearer than sparse 1.5625% (N_y~650) at identical M. Averaging hid a
    # spread that large behind a single number.
    for scenario, _ in spec.scenarios:
        out.append((get(method, variant, scenario, spec.cost_metric, M)[0], None))
    return out


def _best_in_column(
    values: list[float | None], fmt: str, *, lower_is_better: bool
) -> set[int]:
    """Row indices holding the best value in one column, ties included.

    A row is bolded when it is the winner, when it TIES THE WINNER TO
    ``BOLD_DECIMALS`` decimal places (the headline rule: a two-decimal tie is
    "just as good"), or when it merely renders identically to the winner under
    ``fmt`` -- the last case catches the cost columns, printed to one decimal, and
    any column pushed into scientific notation, where two-decimal agreement would
    be a STRICTER test than the eye can apply to the printed table.

    A column with fewer than two numbers has no winner -- bolding the only entry
    present would claim a comparison that was never made.
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
    best_rounded = round(best, BOLD_DECIMALS)
    return {
        i for i, v in present
        if round(v, BOLD_DECIMALS) == best_rounded or _fmt_number(v, fmt) == best_str
    }


def _caption(
    spec: TableSpec, idx: _Index, M: int, *, with_std: bool, scaled: bool = False
) -> str:
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
    # The bold rule is applied to the UNSCALED values in both variants, so the two
    # notations always bold the same cells; the caption says so, because in the
    # scaled table two-decimal agreement is not what the printed digits show.
    tie_units = " (on the unscaled values)" if scaled else ""
    parts.append(
        r"\textbf{Bold} marks the best value in each column, together with every "
        f"value equal to it to {BOLD_DECIMALS} decimal places{tie_units}."
    )
    if scaled:
        parts.append(
            "A metric carrying a factor in its header is reported in those units "
            "throughout its columns (e.g. $\\times 10^{-2}$ means the printed "
            "entry is a multiple of $10^{-2}$)."
        )
    if spec.reference:
        names = ", ".join(lbl for _, lbl in spec.reference)
        Es = sorted({
            e for m, _ in spec.reference for e in idx.E_by_method.get(m, set())
        })
        at_E = f" at $E={Es[0]}$" if len(Es) == 1 else ""
        parts.append(
            f"{names}{at_E} is a REFERENCE run out of competition -- shown for "
            "context and excluded from the bolding."
        )
    if spec.notes:
        parts.append(spec.notes)
    if spec.include_missing_cell_note:
        parts.append(
            rf"``{MISSING_CELL}'' marks a cell with no result at this $M$ "
            "(not run, or every trajectory diverged)."
        )
    return " ".join(parts)


def render_table(
    spec: TableSpec, idx: _Index, M: int, *, with_std: bool = False,
    scaled: bool = False,
) -> str:
    """One ``table*`` for a single sampler-step count ``M``.

    Rendered in two passes: the whole table's values are collected first so the
    best entry in each column (over EVERY row -- both covariance modes, the
    baselines and the classical filters) can be bolded, then the rows are emitted.

    ``scaled`` emits the power-of-ten variant: each metric group is expressed in
    units of its own $10^{-n}$ (:func:`_group_exponent`), named once beside the
    metric in the group header and covering that group's scenario columns. The
    numbers, the bolding and the caption are otherwise IDENTICAL to the plain
    variant -- in particular the bold set is decided on the true values, so the two
    variants of a table always bold the same cells and differ only in notation.
    """
    # + method column + (metric groups + the cost group) x scenarios
    span = 1 + (len(spec.metric_groups) + 1) * len(spec.scenarios)

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
        m_independent: bool = False  # no sampler steps -> same row in every table
        competes: bool = True     # False -> printed, but never bolded (reference)

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
    for group, m_indep in ((spec.baselines, False), (spec.classical, True)):
        for i, (method, label) in enumerate(group):
            plan.append(_Row(method, label, "", rule=(i == 0), m_independent=m_indep))
    # The reference block closes the table: run once, at its own ensemble size
    # (named in the label, taken from the data so it cannot drift), out of
    # competition for the bolding.
    for i, (method, label) in enumerate(spec.reference):
        Es = sorted(idx.E_by_method.get(method, set()))
        if len(Es) == 1:
            label = rf"{label} ($E={Es[0]}$)"
        plan.append(_Row(
            method, label, "", rule=(i == 0), m_independent=True, competes=False,
        ))

    grid = [
        _row_values(spec, idx, r.method, r.variant, M, m_independent=r.m_independent)
        for r in plan
    ]
    n_cols = (len(spec.metric_groups) + 1) * len(spec.scenarios)
    n_metric_cols = len(spec.metric_groups) * len(spec.scenarios)
    # Every metric here (RMSE, CRPS, |1 - spread/skill|) and the cost are
    # lower-is-better; ``higher_is_better`` names any exception.
    col_metric = [m for m, _ in spec.metric_groups for _ in spec.scenarios]
    col_metric.extend([spec.cost_metric] * len(spec.scenarios))
    # Reference rows are masked out of the comparison (not merely left unbolded):
    # a 1000-member filter winning a column would otherwise silently redefine what
    # the bold means for every other row.
    best: list[set[int]] = [
        _best_in_column(
            [row[c][0] if plan[r].competes else None for r, row in enumerate(grid)],
            spec.cost_fmt if c >= n_metric_cols else spec.fmt,
            lower_is_better=col_metric[c] not in spec.higher_is_better,
        )
        for c in range(n_cols)
    ]
    # Per-GROUP units (metric groups, then cost), chosen AFTER the bolding so the
    # two variants agree. ``col_exps`` fans the group exponent back out over the
    # scenario columns it spans, which is what the cells are scaled by.
    n_scen = len(spec.scenarios)
    exps = (
        [
            _group_exponent([
                row[g * n_scen + j][0] for row in grid for j in range(n_scen)
            ])
            for g in range(len(spec.metric_groups) + 1)
        ]
        if scaled else None
    )
    col_exps = [e for e in exps for _ in range(n_scen)] if exps else None

    # Pass 2: emit.
    lines: list[str] = [
        r"\begin{table*}[ht]",
        r"    \centering",
        r"    \scriptsize",
        r"    \setlength{\tabcolsep}{3.5pt}",
        rf"    \caption{{{_caption(spec, idx, M, with_std=with_std, scaled=scaled)}}}",
        # A DISTINCT label for the scaled variant: both files may be \input by the
        # same manuscript while one of the two is being chosen, and duplicate
        # \label keys would silently cross-wire every \ref to these tables.
        rf"    \label{{{spec.label_stem}_M{M}{'_scaled' if scaled else ''}}}",
        # Shrink-to-fit: metric groups x scenarios grows the table past
        # \textwidth (the NS spec is four groups x four scenarios + cost = 17
        # columns, ~15% too wide at \scriptsize), and an overfull table silently
        # runs into the margin. The \ifdim guard scales ONLY when the natural
        # width exceeds the text block, so a narrow table (urban) is left at its
        # nominal font size rather than being blown up to fill the line.
        r"    \resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
        rf"    \begin{{tabular}}{{{_column_format(spec)}}}",
        r"        \toprule",
        *_header(spec, exps),
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
                *_rescale(v, s, col_exps[c] if col_exps else 0),
                fmt=spec.cost_fmt if c >= n_metric_cols else spec.fmt,
                with_std=with_std and c < n_metric_cols,
                bold=r in best[c],
            )
            for c, (v, s) in enumerate(grid[r])
        ]
        lines.append(f"        {row.label} & " + " & ".join(cells) + r" \\")

    lines += [
        r"        \bottomrule", r"    \end{tabular}}", r"\end{table*}",
    ]
    return "\n".join(lines)


def write_latex_tables(
    spec: TableSpec,
    rows: list[dict[str, object]],
    out: Path,
    *,
    with_std: bool = False,
    scaled: bool = False,
) -> list[int]:
    """Write one table per M in ``rows`` to ``out``; return the Ms rendered.

    ``rows`` are the aggregated scalar rows (:func:`aggregate_scalar` output).
    Returns ``[]`` (and writes nothing) when the case has no rows with an ``M``.
    ``scaled`` selects the power-of-ten variant (see :func:`render_table`); call
    twice, with two output paths, to keep both on disk.
    """
    idx = _build_index(rows, spec)
    Ms = sorted(idx.Ms)
    if not Ms:
        return []
    body = "\n\n".join(
        render_table(spec, idx, M, with_std=with_std, scaled=scaled) for M in Ms
    )
    header = (
        f"% Auto-generated by {spec.source} -- DO NOT EDIT BY HAND.\n"
        "% One table per sampler-step count M; regenerate after every aggregation.\n"
        f"% M values present: {', '.join(map(str, Ms))}.\n"
        + ("% Per-column units of 10^-n, named in the column headers; the plain\n"
           "% variant of these same numbers is in tables.tex (labels: ..._scaled).\n"
           if scaled else "")
        + "% Requires: booktabs, amsmath.\n\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + body + "\n", encoding="utf-8")
    return Ms


__all__ = ["TableSpec", "write_latex_tables", "render_table", "MISSING_CELL"]
