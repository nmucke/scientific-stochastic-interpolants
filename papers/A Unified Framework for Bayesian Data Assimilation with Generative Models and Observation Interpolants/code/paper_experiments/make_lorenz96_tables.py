"""LaTeX and markdown tables of the Lorenz 96 case, one per observation sweep.

Reads ``results/lorenz96/<obs>/aggregated/all.csv`` (``aggregate_lorenz96.py``) and
writes

* ``generated/tab_lorenz96_<obs>.tex`` -- a ``tabular`` (booktabs) to ``\\input``;
* ``results/lorenz96/<obs>/aggregated/table_lorenz96_<obs>.md`` -- the same table
  for the results document;
* ``generated/tab_lorenz96_<obs>_vs_M.tex`` (and ``table_lorenz96_<obs>_vs_M.md``) --
  the generative cells against M: ``VS_M_METRICS`` at each M of the sweep.

The main table is at M=50 (``lorenz96_lineup.HEADLINE_M``); the M-independent rows
appear once. Both sweeps get the same columns: vs PF (mean error, variance error,
sliced W2, k-NN KL, energy distance), vs
truth (RMSE, spread, CRPS), lag-1 innovation autocorrelation, NFE and seconds per
assimilation step, and for the SMC cells the mean ESS fraction and resampling
events per step (blank elsewhere). Entries are mean +/- std over seeds; ``(n=k)``
marks a row with fewer seeds than the others, and a single-seed row (the PF
reference) has no std. Rows follow ``lorenz96_lineup.ROWS`` and its groups; the
best non-reference mean of each accuracy column is bold (spread: closest to the PF
reference; lag-1: closest to zero). Rows of the aggregated file that are not in
``ROWS`` (tagged sweep cells, renamed methods) are left out with a warning.

    python paper_experiments/make_lorenz96_tables.py [--obs linear|arctan]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from lorenz96_lineup import (  # noqa: E402
    CASE,
    ENERGY_DISTANCE,
    GROUPS,
    HEADLINE_M,
    KL_KNN,
    M_VALUES,
    OBS_CHOICES,
    ROW_OF,
    ROWS,
    SMC_ESS_FRACTION,
    SMC_RESAMPLES,
    Entry,
    pick_m,
    read_aggregated,
    sweep_root,
)

RESULTS = _here / "results"
GENERATED = _here / "generated"
OBS_TITLE = {"linear": "linear observations", "arctan": "arctan observations"}

# (metric, latex header, markdown header, best rule): "min" lowest mean, "abs"
# closest to zero, "pf" closest to the PF reference, None no bold.
COLUMNS: tuple[tuple[str, str, str, str | None], ...] = (
    ("mean_error_vs_pf", "mean err.", "mean err. vs PF", "min"),
    ("var_error_vs_pf", "var. err.", "var. err. vs PF", "min"),
    ("sliced_w2", r"SW$_2$", "sliced W2 vs PF", "min"),
    (KL_KNN, "KL", "k-NN KL vs PF", "min"),
    (ENERGY_DISTANCE, "energy", "energy dist. vs PF", "min"),
    ("rmse", "RMSE", "RMSE", "min"),
    ("spread", "spread", "spread", "pf"),
    ("crps", "CRPS", "CRPS", "min"),
    ("innovation_lag1", "lag-1", "innov. lag-1", "abs"),
    ("nfe", "NFE", "NFE / step", None),
    ("seconds", "s", "s / step", None),
    (SMC_ESS_FRACTION, "ESS", "ESS fraction", None),
    (SMC_RESAMPLES, "resamp.", "resamples / step", None),
)
# Top header: (title, number of columns), spanning COLUMNS in order.
SPANS: tuple[tuple[str, int], ...] = (
    ("vs PF", 5),
    ("vs truth", 3),
    ("innov.", 1),
    ("cost / step", 2),
    ("SMC", 2),
)
assert sum(n for _, n in SPANS) == len(COLUMNS)
# Columns of the vs-M table (each at every M of lorenz96_lineup.M_VALUES).
VS_M_METRICS: tuple[str, ...] = ("sliced_w2", KL_KNN)
# Rows never bold: the reference posterior itself and its subsample.
NOT_BOLD_GROUPS = ("reference",)


def load_by_m(path: Path) -> tuple[dict, list[tuple[str, str]]]:
    """``{(method, variant): {metric: {M: (mean, std, n)}}}`` of the rows in
    ``ROWS`` (``lorenz96_lineup.read_aggregated``), and the unlisted rows."""
    data = read_aggregated(path)
    unlisted = sorted(key for key in data if key not in ROW_OF)
    for key in unlisted:
        print(f"[{CASE}] warning: {key} is not in lorenz96_lineup.ROWS; left out")
    return {key: value for key, value in data.items() if key in ROW_OF}, unlisted


def at_m(by_m: dict, m: int = HEADLINE_M) -> dict:
    """``{(method, variant): {metric: entry}}`` at step count ``m``.

    Generative rows need exactly ``m``; M-independent rows keep their one entry.
    """
    data: dict[tuple[str, str], dict[str, Entry]] = {}
    for key, metrics in by_m.items():
        row = ROW_OF[key]
        picked = {
            metric: entry
            for metric, entries in metrics.items()
            if (entry := pick_m(entries, row, m)) is not None
        }
        if picked:
            data[key] = picked
    return data


def load_table(path: Path, m: int = HEADLINE_M) -> tuple[dict, list[tuple[str, str]]]:
    """``{(method, variant): {metric: (mean, std, n)}}`` at ``m``, and the unlisted rows."""
    by_m, unlisted = load_by_m(path)
    return at_m(by_m, m), unlisted


def _decimals(value: float) -> int:
    size = abs(value)
    if size >= 100:
        return 0
    if size >= 10:
        return 1
    if size >= 1:
        return 2
    return 3


def format_entry(entry: Entry | None, n_seeds: int, tex: bool, bold: bool = False) -> str:
    if entry is None or not math.isfinite(entry[0]):
        return "" if entry is None else "--"
    mean, std, n = entry
    if abs(mean) and (abs(mean) < 1e-3 or abs(mean) >= 1e5):
        text = f"{mean:.2e}"
        spread = f"{std:.1e}"
    else:
        digits = _decimals(mean)
        text = f"{mean:.{digits}f}"
        spread = f"{std:.{digits}f}"
    if n > 1 and math.isfinite(std) and std > 0:
        text += (r" $\pm$ " if tex else " ± ") + spread
    if bold:
        text = rf"\textbf{{{text}}}" if tex else f"**{text}**"
    if n < n_seeds and n > 1:
        text += f" (n={n})"
    return text


def best_rows(data, rows) -> dict[str, tuple[str, str] | None]:
    """The (method, variant) with the best mean per column, or None."""
    reference = data.get(("Particle filter", "reference"), {})
    best: dict[str, tuple[str, str] | None] = {}
    for metric, _, _, rule in COLUMNS:
        if rule is None or (rule == "pf" and metric not in reference):
            best[metric] = None
            continue
        target = reference[metric][0] if rule == "pf" else 0.0
        candidates = [
            ((row.method, row.variant), data[(row.method, row.variant)][metric][0])
            for row in rows
            if row.group not in NOT_BOLD_GROUPS
            and metric in data.get((row.method, row.variant), {})
            and math.isfinite(data[(row.method, row.variant)][metric][0])
        ]
        if not candidates:
            best[metric] = None
            continue
        score = (lambda v: v) if rule == "min" else (lambda v: abs(v - target))
        best[metric] = min(candidates, key=lambda c: score(c[1]))[0]
    return best


def table_rows(data) -> list[tuple[str, list]]:
    """``[(group title, [Row, ...]), ...]`` for the groups with data, in order."""
    out = []
    for key, title in GROUPS:
        rows = [r for r in ROWS if r.group == key and (r.method, r.variant) in data]
        if rows:
            out.append((title, rows))
    return out


def _tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def write_tex(data, obs: str, path: Path) -> None:
    groups = table_rows(data)
    flat = [row for _, rows in groups for row in rows]
    best = best_rows(data, flat)
    n_seeds = max((e[2] for d in data.values() for e in d.values()), default=1)
    ncols = 2 + len(COLUMNS)
    spans = " & ".join(rf"\multicolumn{{{n}}}{{c}}{{{t}}}" for t, n in SPANS)
    rules, start = [], 3
    for _, n in SPANS:
        rules.append(rf"\cmidrule(lr){{{start}-{start + n - 1}}}")
        start += n
    lines = [
        f"% Auto-generated by paper_experiments/make_lorenz96_tables.py: {CASE}, "
        f"{OBS_TITLE[obs]}. Mean +/- std over seeds; do not edit by hand.",
        rf"\begin{{tabular}}{{ll{'c' * len(COLUMNS)}}}",
        r"\toprule",
        rf" & & {spans} \\",
        " ".join(rules),
        "Method & Variant & " + " & ".join(h for _, h, _, _ in COLUMNS) + r" \\",
        r"\midrule",
    ]
    for title, rows in groups:
        lines.append(rf"\multicolumn{{{ncols}}}{{l}}{{\emph{{{title}}}}} \\")
        for row in rows:
            entries = data[(row.method, row.variant)]
            cells = [
                format_entry(
                    entries.get(metric), n_seeds, tex=True,
                    bold=best[metric] == (row.method, row.variant),
                )
                for metric, *_ in COLUMNS
            ]
            lines.append(
                f"{_tex_escape(row.method)} & {_tex_escape(row.label)} & "
                + " & ".join(cells) + r" \\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(data, obs: str, path: Path, unlisted) -> None:
    groups = table_rows(data)
    flat = [row for _, rows in groups for row in rows]
    best = best_rows(data, flat)
    n_seeds = max((e[2] for d in data.values() for e in d.values()), default=1)
    head = ["group", "method", "variant", *(h for _, _, h, _ in COLUMNS)]
    lines = [
        f"### Lorenz 96, {OBS_TITLE[obs]} (mean ± std over {n_seeds} seeds)",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for title, rows in groups:
        for row in rows:
            entries = data[(row.method, row.variant)]
            cells = [
                format_entry(
                    entries.get(metric), n_seeds, tex=False,
                    bold=best[metric] == (row.method, row.variant),
                )
                for metric, *_ in COLUMNS
            ]
            lines.append(f"| {title} | {row.method} | {row.label} | " + " | ".join(cells) + " |")
    if unlisted:
        lines += ["", "Left out (not in `lorenz96_lineup.ROWS`): " + ", ".join(
            f"{m} [{v}]" if v else m for m, v in unlisted
        )]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def vs_m_rows(by_m) -> list:
    """Generative rows with a value of a ``VS_M_METRICS`` metric at some M."""
    return [
        row for row in ROWS
        if row.generative
        and any(
            m in M_VALUES
            for metric in VS_M_METRICS
            for m in by_m.get((row.method, row.variant), {}).get(metric, {})
        )
    ]


def write_vs_m(by_m, obs: str, tex_path: Path, md_path: Path) -> bool:
    """Generative rows x (metric, M): the convergence of each cell in M."""
    rows = vs_m_rows(by_m)
    if not rows:
        return False
    n_seeds = max(
        (e[2] for d in by_m.values() for entries in d.values() for e in entries.values()),
        default=1,
    )
    titles = {metric: (tex, md) for metric, tex, md, _ in COLUMNS}
    ncols = len(VS_M_METRICS) * len(M_VALUES)
    tex = [
        f"% Auto-generated by paper_experiments/make_lorenz96_tables.py: {CASE}, "
        f"{OBS_TITLE[obs]}, generative cells against M. Mean +/- std over seeds.",
        rf"\begin{{tabular}}{{ll{'c' * ncols}}}",
        r"\toprule",
        " & & " + " & ".join(
            rf"\multicolumn{{{len(M_VALUES)}}}{{c}}{{{titles[m][0]}}}" for m in VS_M_METRICS
        ) + r" \\",
        "Method & Variant & " + " & ".join(
            f"$M={m}$" for _ in VS_M_METRICS for m in M_VALUES
        ) + r" \\",
        r"\midrule",
    ]
    head = ["method", "variant", *(f"{titles[k][1]} M={m}" for k in VS_M_METRICS for m in M_VALUES)]
    md = [
        f"### Lorenz 96, {OBS_TITLE[obs]}: generative cells against M "
        f"(mean ± std over {n_seeds} seeds)",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for row in rows:
        metrics = by_m.get((row.method, row.variant), {})
        cells = [(metrics.get(k, {}).get(m)) for k in VS_M_METRICS for m in M_VALUES]
        tex.append(
            f"{_tex_escape(row.method)} & {_tex_escape(row.label)} & "
            + " & ".join(format_entry(c, n_seeds, tex=True) for c in cells) + r" \\"
        )
        md.append(
            f"| {row.method} | {row.label} | "
            + " | ".join(format_entry(c, n_seeds, tex=False) for c in cells) + " |"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return True


def make_sweep(obs: str) -> list[Path]:
    source = sweep_root(RESULTS, obs) / "aggregated" / "all.csv"
    if not source.exists():
        print(f"[{CASE}/{obs}] no {source}; run aggregate_lorenz96.py first. Skipped.")
        return []
    by_m, unlisted = load_by_m(source)
    data = at_m(by_m)
    if not data:
        print(f"[{CASE}/{obs}] no row of {source} is in lorenz96_lineup.ROWS at M={HEADLINE_M}. Skipped.")
        return []
    tex = GENERATED / f"tab_lorenz96_{obs}.tex"
    md = source.parent / f"table_lorenz96_{obs}.md"
    write_tex(data, obs, tex)
    write_markdown(data, obs, md, unlisted)
    written = [tex, md]
    tex_m = GENERATED / f"tab_lorenz96_{obs}_vs_M.tex"
    md_m = source.parent / f"table_lorenz96_{obs}_vs_M.md"
    if write_vs_m(by_m, obs, tex_m, md_m):
        written += [tex_m, md_m]
    print(f"[{CASE}/{obs}] wrote " + ", ".join(str(p) for p in written))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--obs",
        choices=OBS_CHOICES,
        action="append",
        help="observation sweep (repeatable); default: every sweep",
    )
    args = parser.parse_args()
    for obs in args.obs or OBS_CHOICES:
        make_sweep(obs)


if __name__ == "__main__":
    main()
