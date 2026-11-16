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

# Divergence. A trajectory-cell whose run tripped the divergence guard is dropped
# whole by the aggregation (aggregate_ns.py) and counted per cell as ``n_div``.
# A cell where EVERY trajectory diverged prints DIV_ALL ("div 5/5"); a cell where
# SOME did prints the mean over the survivors with a DIV_SOME superscript
# ("0.873^{div 1/5}"). Partially diverged cells are EXCLUDED from the bolding --
# their mean is over a survivor-biased subset, so it is not comparable.
DIV_ALL = "div {k}/{n}"
DIV_SOME = r"\textsuperscript{{div\,{k}/{n}}}"


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
    # Optional ``(variant, sub-block header)`` overrides of :func:`_variant_label`
    # (the cadence ablation words its sub-blocks differently from the main table).
    variant_labels: tuple[tuple[str, str], ...] = ()
    # Nominal ensemble size of each reference row, used in its label when the
    # reference has NO data (not run): the row is still printed, as "--" cells,
    # so a missing reference stays visible instead of silently vanishing.
    reference_nominal_E: tuple[tuple[str, int], ...] = ()
    # Covariance modes that were NOT run but belong in the table's layout (e.g.
    # the k=1 / k=5 cadences of the ablation): each is rendered as its sub-block
    # header with every cell ``--``, so the table keeps the manuscript's shape.
    placeholder_variants: tuple[str, ...] = ()
    # Show baselines / classical / reference blocks (False -> ours only: ablation).
    include_others: bool = True
    # False -> no cost block (e.g. when no honest clean timing exists for a row).
    include_cost: bool = True
    # ``(base_variant, paired_variant, row-label suffix)``: our samplers are laid
    # out in one sub-block per BASE variant, each row followed directly by the
    # same sampler under the paired variant (e.g. plain vs + forecast correction).
    paired_variants: tuple[tuple[str, str, str], ...] = ()


def _n_groups(spec: "TableSpec") -> int:
    """Column groups: the metric groups, plus cost unless ``include_cost`` is off."""
    return len(spec.metric_groups) + (1 if spec.include_cost else 0)


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


def _variant_rank(variant: str) -> tuple[int, int, str]:
    """Sort key: the shared modes first (the paper's headline), jacfree after.

    Shared cadences sort NUMERICALLY (k=1, 5, 10), not as strings, which would put
    ``shared_jac10`` before ``shared_jac5``.
    """
    if variant.startswith("shared"):
        k = variant[len("shared_jac"):]
        return (0, int(k) if k.isdigit() else 1, variant)
    if variant == "jacfree":
        return (1, 0, variant)
    return (2, 0, variant)


# --------------------------------------------------------------------------- #
# Cell formatting
# --------------------------------------------------------------------------- #


def _fmt_number(x: float, fmt: str) -> str:
    """Fixed-point, falling back to scientific notation for blown-up magnitudes."""
    if abs(x) >= 1e4 or (x != 0.0 and abs(x) < 1e-3):
        return f"{x:.1e}".replace("e+0", r"e{+}").replace("e-0", r"e{-}")
    return fmt.format(x)


# The across-trajectory std printed after a metric value, in a smaller font than
# the (already \scriptsize) table body, at the value's scale and decimals.
STD_FONT = r"\tiny"


def std_suffix(std: float | None, fmt: str, *, n: int | None = None) -> str:
    r"""``{\tiny\,$\pm$<std>}``, or "" without a usable std (absent, NaN, n<2)."""
    if std is None or math.isnan(std) or (n is not None and n < 2):
        return ""
    # Same decimals as the value: a std below 10^-3 prints 0.000, not 4.3e-4.
    txt = fmt.format(std) if abs(std) < 1e4 else _fmt_number(std, fmt)
    return rf"{{{STD_FONT}\,$\pm${txt}}}"


def _cell(
    value: float | None, std: float | None, *, fmt: str, with_std: bool,
    bold: bool = False, underline: bool = False, n: int | None = None,
) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return MISSING_CELL
    out = _fmt_number(value, fmt)
    if bold:
        out = rf"\textbf{{{out}}}"
    elif underline:
        out = rf"\underline{{{out}}}"
    # The std is context, not the compared quantity -- it stays unmarked.
    if with_std:
        out += std_suffix(std, fmt, n=n)
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
    # (method, variant, scenario, M) -> trajectories that SURVIVED (n_traj of the
    # rmse row) and -> trajectories that diverged (from aggregate_ns n_div).
    nok_by_cell: dict[tuple[str, str, str, int | None], int] = field(default_factory=dict)
    ndiv_by_cell: dict[tuple[str, str, str, int | None], int] = field(default_factory=dict)

    def div_counts(
        self, method: str, variant: str, scenario: str, M: int | None,
        m_independent: bool = False,
    ) -> tuple[int, int]:
        """(n_ok, n_div) for one cell; m_independent rows match any M label."""
        def pick(d: dict) -> int:
            if (method, variant, scenario, M) in d:
                return d[(method, variant, scenario, M)]
            if m_independent:
                hits = [v for (mm, vv, ss, _), v in d.items()
                        if (mm, vv, ss) == (method, variant, scenario)]
                if len(hits) == 1:
                    return hits[0]
            return 0
        return pick(self.nok_by_cell), pick(self.ndiv_by_cell)

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


def _build_index(
    rows: list[dict[str, object]], spec: TableSpec,
    n_div: dict[tuple, int] | None = None,
) -> _Index:
    idx = _Index()
    for (method, variant, scenario, M), k in (n_div or {}).items():
        if k:
            idx.ndiv_by_cell[(str(method), _norm_variant(variant), str(scenario),
                              _as_opt_int(M))] = int(k)
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
        if str(r.get("metric")) == "rmse":
            n_ok = _as_opt_int(r.get("n_traj"))
            if n_ok is not None:
                idx.nok_by_cell[(method, variant, str(r.get("scenario")), M)] = n_ok
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
                      for _ in range(_n_groups(spec)))
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
    if spec.include_cost:
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
    sub = "        & " + " & ".join([scen_cells] * _n_groups(spec))
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
    if spec.include_cost:
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


def _second_in_column(
    values: list[float | None], fmt: str, best: set[int], *, lower_is_better: bool
) -> set[int]:
    """Row indices holding the SECOND-best value of a column, ties included.

    Same competing set (``values``, with ``None`` for masked cells) and the same
    tie rule as :func:`_best_in_column`: the second best is the best of the values
    left once every bolded (best-tied) entry is removed, and every remaining value
    tying it is underlined. No bold -> no underline (fewer than two competitors).
    """
    if not best:
        return set()
    rest = [None if i in best else v for i, v in enumerate(values)]
    present = [(i, v) for i, v in enumerate(rest) if v is not None and not math.isnan(v)]
    if not present:
        return set()
    pick = max if lower_is_better is False else min
    second = pick(v for _, v in present)
    s_str, s_round = _fmt_number(second, fmt), round(second, BOLD_DECIMALS)
    return {i for i, v in present
            if round(v, BOLD_DECIMALS) == s_round or _fmt_number(v, fmt) == s_str}


BOLD_UNDERLINE_SENTENCE = (
    r"\textbf{Bold} marks the best value in each column and \underline{underline} "
    f"the second best, each together with every value equal to it to {BOLD_DECIMALS} "
    "decimal places in unscaled units."
)
STD_SENTENCE = r"$\pm$ gives the standard deviation over the (non-diverged) trajectories."


def _caption(
    spec: TableSpec, idx: _Index, M: int, *, with_std: bool, scaled: bool = False
) -> str:
    ntraj = sorted(idx.ntraj_by_M.get(M, set()))
    # With divergence bookkeeping the per-cell count is survivors + diverged; quote
    # THAT (the number of trajectories run), and explain the div marks separately.
    if idx.ndiv_by_cell:
        cells = set(idx.nok_by_cell) | set(idx.ndiv_by_cell)
        ntraj = sorted({
            idx.nok_by_cell.get(k, 0) + idx.ndiv_by_cell.get(k, 0) for k in cells
            if k[0] not in {m for m, _ in spec.reference}
        })
    Es = sorted(idx.E_by_M.get(M, set()))
    if not ntraj:
        traj_txt = "held-out trajectories"
    elif len(ntraj) == 1:
        n = ntraj[0]
        traj_txt = f"{n} held-out trajector" + ("y" if n == 1 else "ies")
    else:
        traj_txt = f"{ntraj[0]}--{ntraj[-1]} held-out trajectories (cell-dependent)"
    E_txt = f"$E={Es[0]}$" if len(Es) == 1 else "$E \\in \\{" + ", ".join(map(str, Es)) + "\\}$"

    stat = "Mean"
    metrics_phrase = spec.metrics_phrase or "metrics"
    n_scen = {2: "two", 3: "three", 4: "four"}.get(len(spec.scenarios), str(len(spec.scenarios)))
    parts = [
        f"{spec.case_label}: {metrics_phrase} across the {n_scen} observation "
        f"scenarios"
        + (", with per-step cost (lower is better)." if spec.include_cost
           else " (lower is better)."),
        f"{stat} over {traj_txt} at {E_txt}, $M={M}$ sampler steps.",
    ]
    # The bold rule is applied to the UNSCALED values in both variants, so the two
    # notations always bold the same cells; the caption says so, because in the
    # scaled table two-decimal agreement is not what the printed digits show.
    parts.append(BOLD_UNDERLINE_SENTENCE)
    if with_std:
        parts.append(STD_SENTENCE)
    if scaled:
        parts.append(
            "A metric carrying a factor in its header is reported in those units "
            "throughout its columns (e.g. $\\times 10^{-2}$ means the printed "
            "entry is a multiple of $10^{-2}$)."
        )
    if spec.reference and spec.include_others:
        ran = [(m, lbl) for m, lbl in spec.reference if idx.E_by_method.get(m)]
        not_run = [lbl for m, lbl in spec.reference if not idx.E_by_method.get(m)]
        if ran:
            names = ", ".join(lbl for _, lbl in ran)
            Es = sorted({e for m, _ in ran for e in idx.E_by_method.get(m, set())})
            at_E = f" at $E={Es[0]}$" if len(Es) == 1 else ""
            parts.append(
                f"{names}{at_E} is a REFERENCE run out of competition -- shown for "
                "context and excluded from the bolding."
            )
        if not_run:
            parts.append(
                f"{', '.join(not_run)}: not run for this setting (``{MISSING_CELL}'')."
            )
    if idx.ndiv_by_cell:
        parts.append(
            "``div $k/n$'' marks a cell in which $k$ of the $n$ trajectories "
            "diverged (tripped the divergence guard); those trajectories are "
            "excluded from every metric. A number carrying a div superscript is "
            "the mean over the surviving $n-k$ trajectories and is neither bolded nor "
            "underlined."
        )
    if spec.notes:
        parts.append(spec.notes)
    if spec.include_missing_cell_note:
        parts.append(
            rf"``{MISSING_CELL}'' marks a cell with no result at this $M$ "
            "(not run, or every trajectory diverged)."
        )
    return " ".join(parts)


@dataclass
class _Row:
    method: str
    label: str
    variant: str
    rule: bool = False        # emit \midrule before this row
    head: str | None = None   # emit a \multicolumn sub-block header before it
    m_independent: bool = False  # no sampler steps -> same row in every table
    competes: bool = True     # False -> printed, but never bolded (reference)


@dataclass
class _Built:
    """Everything an emitter needs: row plan, values, bold sets, units, div marks."""

    plan: list[_Row]
    grid: list[list[tuple[float | None, float | None]]]
    best: list[set[int]]
    second: list[set[int]]   # underlined (second best), same competing set
    exps: list[int] | None
    col_exps: list[int] | None
    # Per row, per column: "" (normal), or the divergence decoration.
    div: list[list[tuple[int, int]]]   # (n_div, n_total) per cell; (0, n) = clean
    n_metric_cols: int
    span: int


def _build_table(
    spec: TableSpec, idx: _Index, M: int, *, scaled: bool = False,
) -> _Built:
    """Pass 1 of rendering: the row plan, the value grid and the bolding.

    The best entry in each column is chosen over EVERY competing row (both
    covariance modes, the baselines and the classical filters); reference rows and
    PARTIALLY DIVERGED cells are masked out of that comparison.
    """
    span = 1 + _n_groups(spec) * len(spec.scenarios)
    plan: list[_Row] = []
    variants = sorted(
        set(idx.variants_by_M.get(M, set())) | set(spec.placeholder_variants),
        key=_variant_rank,
    )
    pairs = {base: (paired, suffix) for base, paired, suffix in spec.paired_variants}
    if pairs:
        variants = sorted(pairs, key=_variant_rank)
    if not variants:
        variants = [""]  # no variant column in the data -> a single unlabelled block
    for b, variant in enumerate(variants):
        for i, (method, label) in enumerate(spec.ours):
            plan.append(_Row(
                method, label, variant,
                rule=(i == 0 and b > 0),
                head=(
                    dict(spec.variant_labels).get(variant, _variant_label(variant))
                    if i == 0 else None
                ),
            ))
            if variant in pairs:
                paired, suffix = pairs[variant]
                plan.append(_Row(method, f"{label}{suffix}", paired))
    if spec.include_others:
        for group, m_indep in ((spec.baselines, False), (spec.classical, True)):
            for i, (method, label) in enumerate(group):
                plan.append(_Row(method, label, "", rule=(i == 0), m_independent=m_indep))
        # The reference block closes the table: run once, at its own ensemble size
        # (named in the label, taken from the data so it cannot drift -- or from
        # ``reference_nominal_E`` when it was not run), out of competition.
        nominal = dict(spec.reference_nominal_E)
        for i, (method, label) in enumerate(spec.reference):
            Es = sorted(idx.E_by_method.get(method, set()))
            if len(Es) == 1:
                label = rf"{label} ($E={Es[0]}$)"
            elif not Es and method in nominal:
                label = rf"{label} ($E={nominal[method]}$)"
            plan.append(_Row(
                method, label, "", rule=(i == 0), m_independent=True, competes=False,
            ))

    grid = [
        _row_values(spec, idx, r.method, r.variant, M, m_independent=r.m_independent)
        for r in plan
    ]
    n_cols = _n_groups(spec) * len(spec.scenarios)
    n_metric_cols = len(spec.metric_groups) * len(spec.scenarios)
    n_scen = len(spec.scenarios)

    # Divergence counts per cell (same for every metric column of one scenario).
    div: list[list[tuple[int, int]]] = []
    for r in plan:
        row_div = []
        for c in range(n_cols):
            scenario = spec.scenarios[c % n_scen][0]
            n_ok, n_dv = idx.div_counts(r.method, r.variant, scenario, M, r.m_independent)
            row_div.append((n_dv, n_ok + n_dv))
        div.append(row_div)

    col_metric = [m for m, _ in spec.metric_groups for _ in spec.scenarios]
    if spec.include_cost:
        col_metric.extend([spec.cost_metric] * len(spec.scenarios))
    # Reference rows are masked out of the comparison (not merely left unbolded):
    # a 1000-member filter winning a column would otherwise silently redefine what
    # the bold means for every other row. Partially diverged cells likewise: their
    # mean is over the survivors only. (Cost columns are not masked -- the cost of
    # a step does not depend on whether a later step blew up.)
    competing = [
        [
            row[c][0]
            if plan[r].competes and (c >= n_metric_cols or div[r][c][0] == 0)
            else None
            for r, row in enumerate(grid)
        ]
        for c in range(n_cols)
    ]
    col_fmt = [spec.cost_fmt if c >= n_metric_cols else spec.fmt for c in range(n_cols)]
    lower = [col_metric[c] not in spec.higher_is_better for c in range(n_cols)]
    best: list[set[int]] = [
        _best_in_column(competing[c], col_fmt[c], lower_is_better=lower[c])
        for c in range(n_cols)
    ]
    second: list[set[int]] = [
        _second_in_column(competing[c], col_fmt[c], best[c], lower_is_better=lower[c])
        for c in range(n_cols)
    ]
    exps = (
        [
            _group_exponent([
                row[g * n_scen + j][0] for row in grid for j in range(n_scen)
            ])
            for g in range(_n_groups(spec))
        ]
        if scaled else None
    )
    col_exps = [e for e in exps for _ in range(n_scen)] if exps else None
    return _Built(plan, grid, best, second, exps, col_exps, div, n_metric_cols, span)


def _cells_of_row(
    spec: TableSpec, t: _Built, r: int, *, with_std: bool, markdown: bool = False,
) -> list[str]:
    out = []
    for c, (v, s) in enumerate(t.grid[r]):
        is_metric = c < t.n_metric_cols
        n_dv, n_tot = t.div[r][c]
        if n_dv and n_dv == n_tot:
            # Every trajectory diverged: the metric columns say so; cost has no
            # finished run to report.
            out.append(DIV_ALL.format(k=n_dv, n=n_tot) if is_metric else MISSING_CELL)
            continue
        fmt = spec.cost_fmt if not is_metric else spec.fmt
        val, sd = _rescale(v, s, t.col_exps[c] if t.col_exps else 0)
        if markdown:
            if val is None or math.isnan(val):
                cell = MISSING_CELL
            else:
                cell = _fmt_number(val, fmt)
                if r in t.best[c]:
                    cell = f"**{cell}**"
                elif r in t.second[c]:
                    cell = f"_{cell}_"
                if (with_std and is_metric and sd is not None and not math.isnan(sd)
                        and n_tot - n_dv >= 2):
                    cell += f" ± {fmt.format(sd)}"
        else:
            cell = _cell(val, sd, fmt=fmt, with_std=with_std and is_metric,
                         bold=r in t.best[c], underline=r in t.second[c],
                         n=n_tot - n_dv if n_tot else None)
        if is_metric and n_dv and cell != MISSING_CELL:
            cell += (f" (div {n_dv}/{n_tot})" if markdown
                     else DIV_SOME.format(k=n_dv, n=n_tot))
        out.append(cell)
    return out


def render_table(
    spec: TableSpec, idx: _Index, M: int, *, with_std: bool = False,
    scaled: bool = False,
) -> str:
    """One ``table*`` for a single sampler-step count ``M``.

    Rendered in two passes (:func:`_build_table`, then emit). ``scaled`` emits the
    power-of-ten variant: each metric group is expressed in units of its own
    $10^{-n}$ (:func:`_group_exponent`), named once beside the metric in the group
    header. The bold set is decided on the true values, so the two variants of a
    table always bold the same cells and differ only in notation.
    """
    t = _build_table(spec, idx, M, scaled=scaled)
    lines: list[str] = [
        r"\begin{table*}[ht]",
        r"    \centering",
        r"    \scriptsize",
        r"    \setlength{\tabcolsep}{3.5pt}",
        rf"    \caption{{{_caption(spec, idx, M, with_std=with_std, scaled=scaled)}}}",
        # A DISTINCT label for the scaled variant: both files may be \input by the
        # same manuscript while one of the two is being chosen.
        rf"    \label{{{spec.label_stem}_M{M}{'_scaled' if scaled else ''}}}",
        # Shrink-to-fit, only when the natural width exceeds the text block.
        r"    \resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
        rf"    \begin{{tabular}}{{{_column_format(spec)}}}",
        r"        \toprule",
        *_header(spec, t.exps),
        r"        \midrule",
    ]
    for r, row in enumerate(t.plan):
        if row.rule:
            lines.append(r"        \midrule")
        if row.head is not None:
            lines.append(
                rf"        \multicolumn{{{t.span}}}{{l}}{{\textit{{{row.head}}}}} \\"
            )
        cells = _cells_of_row(spec, t, r, with_std=with_std)
        lines.append(f"        {row.label} & " + " & ".join(cells) + r" \\")
    lines += [
        r"        \bottomrule", r"    \end{tabular}}", r"\end{table*}",
    ]
    return "\n".join(lines)


def _md_text(s: str) -> str:
    """Rough LaTeX -> Markdown for labels and headers in the preview."""
    for a, b in ((r"$\tfrac{1}{64}$", "1/64"), (r"$16^2$", "16²"), (r"$32^2$", "32²"),
                 (r"$5\%$", "5%"), (r"\textit{", ""), (r"\textbf{", ""), ("--", "–"),
                 ("$", ""), (r"\times 10^{-", "×10^-"), ("}", ""), ("{", "")):
        s = s.replace(a, b)
    return s


def render_markdown(
    spec: TableSpec, idx: _Index, M: int, *, with_std: bool = False,
    scaled: bool = False, title: str = "",
) -> str:
    """The same table as :func:`render_table`, as a Markdown preview.

    Bold uses ``**``, underline ``_x_``; a fully diverged cell prints ``div k/n``; a partially
    diverged one prints its survivor mean followed by ``(div k/n)``.
    """
    t = _build_table(spec, idx, M, scaled=scaled)
    n_scen = len(spec.scenarios)
    groups = [lbl for _, lbl in spec.metric_groups]
    if spec.include_cost:
        groups.append(f"Cost ({spec.cost_header})")
    if t.exps is not None:
        groups = [g if e == 0 else rf"{g} (×10^-{e})" for g, e in zip(groups, t.exps)]
    head = ["Method"] + [
        f"{_md_text(g)} {_md_text(h)}" for g in groups for _, h in spec.scenarios
    ]
    out = [f"### {title}" if title else "", "",
           "| " + " | ".join(head) + " |",
           "|" + "---|" * len(head)]
    for r, row in enumerate(t.plan):
        if row.head is not None:
            out.append(f"| *{_md_text(row.head)}* |" + " |" * (len(head) - 1))
        cells = _cells_of_row(spec, t, r, with_std=with_std, markdown=True)
        out.append(f"| {_md_text(row.label)} | " + " | ".join(cells) + " |")
    del n_scen
    return "\n".join(out)


def write_latex_tables(
    spec: TableSpec,
    rows: list[dict[str, object]],
    out: Path,
    *,
    with_std: bool = True,
    scaled: bool = False,
    only_M: int | None = None,
    n_div: dict[tuple, int] | None = None,
    markdown_out: Path | None = None,
    markdown_title: str = "",
) -> list[int]:
    """Write one table per M in ``rows`` to ``out``; return the Ms rendered.

    ``only_M`` renders that single M. It is needed for a one-rung grid: the
    classical filters carry whatever M their cell was labelled with, which would
    otherwise mint a second table holding nothing but those rows.

    ``n_div`` maps ``(method, variant, scenario, M)`` to the number of diverged
    trajectory-cells the aggregation dropped there (see ``DIV_ALL`` /
    ``DIV_SOME``). ``markdown_out`` additionally writes a Markdown preview of the
    same tables.

    ``rows`` are the aggregated scalar rows (:func:`aggregate_scalar` output).
    Returns ``[]`` (and writes nothing) when the case has no rows with an ``M``.
    ``scaled`` selects the power-of-ten variant (see :func:`render_table`); call
    twice, with two output paths, to keep both on disk.
    """
    idx = _build_index(rows, spec, n_div)
    Ms = sorted(idx.Ms)
    if only_M is not None:
        Ms = [M for M in Ms if M == only_M]
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
    if markdown_out is not None:
        md = "\n\n".join(
            render_markdown(spec, idx, M, with_std=with_std, scaled=scaled,
                            title=f"{markdown_title} (M={M})")
            for M in Ms
        )
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(md + "\n", encoding="utf-8")
    return Ms


__all__ = [
    "TableSpec", "write_latex_tables", "render_table", "render_markdown",
    "MISSING_CELL", "DIV_ALL", "DIV_SOME", "std_suffix", "BOLD_UNDERLINE_SENTENCE",
    "STD_SENTENCE",
]
