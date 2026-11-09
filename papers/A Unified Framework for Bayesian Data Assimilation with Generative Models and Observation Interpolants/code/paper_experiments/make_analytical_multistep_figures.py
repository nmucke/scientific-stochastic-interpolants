"""Table and figure of the sequential linear--Gaussian case.

Reads ``<root>/aggregated/per_step.csv`` and writes

* ``analytical_multistep_error_vs_step.{pdf,png}`` -- KL and sliced W2 to the Kalman
  posterior against the assimilation cycle, one row per system. Lines are the mean
  over seeds, bands +/- one std, and a cross marks a cycle where some seeds have no
  finite value (the mean covers the survivors only). KL above the axis ceiling is
  a collapsed ensemble.
* ``analytical_multistep_surge_vs_step.{pdf,png}`` -- the same axes for the rows
  that bear on the dropped ``p(y^{k+1} | x^k_i)`` weights: Ours with and without
  SURGE, the resampling diagnostic, the SURGE baselines and the two references.
* ``tab_analytical_multistep.tex`` and ``<root>/aggregated/table_<system>.md`` --
  mean +/- std at selected cycles. Every cycle is in ``per_step.csv``. Corrected
  rows (``<variant>_fc<S>``, see ``cases/analytical/multistep.py``) enter at the
  forecast steps in ``FC_TABLE_STEPS``.
* With corrected rows present (forecast-residual correction, FORECAST_CORRECTION.md):
  ``analytical_multistep_correction_vs_step.{pdf,png}`` -- KL and the variance ratio
  per cycle for plain Ours (shared), corrected Ours at ``FC_TABLE_STEPS``, the exact
  chain, ``adapted_shared`` and EnKF; ``analytical_multistep_error_vs_forecast_steps.{pdf,png}``
  -- cycle-mean KL to Kalman against the forecast steps ``S``, one line per sampler
  and Q_hat source / group, with the plain, chain, adapted and EnKF levels; and
  ``tab_analytical_multistep_correction.tex`` / ``<root>/aggregated/correction_<system>.md``
  -- cycle means of every corrected row with cost (NFE, seconds per cycle).

For the default root the outputs go to ``figures/analytical_multistep/`` and
``generated/``, otherwise under ``<root>``. Unlike the other metric figures this one
keeps FlowDAS and SDA: the sequential case is about each method's own error, not
the SURGE-only lineup.

    "$PY" "$EXPERIMENT_DIR/make_analytical_multistep_figures.py"
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

from cases.analytical.multistep import parse_fc_variant  # noqa: E402
from figure_common import FIGURES_DIR, SERIES, _save_fig, apply_style  # noqa: E402

ROOT = _here / "results" / "analytical_multistep"
DEFAULT_OUT = FIGURES_DIR / "analytical_multistep"
TEX_OUT = _here / "generated" / "tab_analytical_multistep.tex"

SYSTEM_LABELS: dict[str, str] = {
    "multistep:paper": r"$A=Q=H=R=I$",
    "multistep:partial": "Damped rotation, partial observation",
}
METRICS: tuple[tuple[str, str], ...] = (
    ("kl_points", "KL to Kalman posterior"),
    ("sliced_w2", r"Sliced $W_2$ to Kalman posterior"),
)
TABLE_CYCLES: tuple[int, ...] = (1, 2, 5, 10, 20)
KL_CEILING = 30.0  # collapsed ensembles sit far above; clip the axis, not the data

ADAPTED_SERIES: tuple[tuple, ...] = tuple(
    (method, "adapted_shared", f"{label} + resampling", colour, "-.", marker, True)
    for method, label, colour, marker in (
        ("Ours (SI-SDE)", "Ours SI-SDE (shared)", "#0072B2", "o"),
        ("Ours (DM-SDE)", "Ours DM-SDE (shared)", "#009E73", "s"),
        ("Ours (FM-ODE)", "Ours FM-ODE (shared)", "#CC79A7", "^"),
    )
)
SURGE_SERIES: tuple[tuple, ...] = tuple(
    (method, f"surge_{variant}", f"SURGE, {label} {name}", colour, ls, marker, filled)
    for method, label, colour, marker in (
        ("Ours (SI-SDE)", "SI-SDE", "#56B4E9", "o"),
        ("Ours (DM-SDE)", "DM-SDE", "#2CA25F", "s"),
    )
    for variant, name, ls, filled in (
        ("shared", "(shared)", "-", True),
        ("jacfree", "(Jac-free)", "--", False),
        ("unguided", "prior, no guidance", ":", False),
    )
)
REFERENCE_SERIES: tuple[tuple, ...] = (
    ("Exact per-member chain", None, "Exact per-member chain", "#000000", "-", "", False),
    ("Kalman filter (sampled)", None, "Sampling floor", "#999999", "-", "", False),
)
_OURS_STYLE: dict[str, tuple[str, str, str]] = {
    "Ours (SI-SDE)": ("SI-SDE", "#0072B2", "o"),
    "Ours (DM-SDE)": ("DM-SDE", "#009E73", "s"),
    "Ours (FM-ODE)": ("FM-ODE", "#CC79A7", "^"),
}
_BASE_NAME = {"shared": "shared", "jacfree": "Jac-free"}
FC_TABLE_STEPS: tuple[int, ...] = (5, 50)  # corrected rows in the per-cycle tables
FC_FIGURE_STEPS: tuple[int, ...] = (2, 5, 50)


def fc_label(method: str, variant: str) -> str:
    spec = parse_fc_variant(variant)
    name = _OURS_STYLE[method][0]
    extra = "".join((
        ", bank" if spec["q_source"] == "bank" else "",
        f", E={spec['group']}" if spec["group"] else "",
    ))
    return f"Ours {name} ({_BASE_NAME.get(spec['base'], spec['base'])}) + FC S={spec['steps']}{extra}"


def fc_series(keys, steps: tuple[int, ...] | None = None) -> tuple[tuple, ...]:
    """Series tuples of the corrected rows present in ``keys`` (scenario, method, variant, ...)."""
    found = sorted(
        {(k[1], k[2]) for k in keys if k[1] in _OURS_STYLE and parse_fc_variant(k[2])},
        key=lambda mv: (
            list(_OURS_STYLE).index(mv[0]), parse_fc_variant(mv[1])["base"] != "shared",
            parse_fc_variant(mv[1])["q_source"], parse_fc_variant(mv[1])["group"] or 0,
            parse_fc_variant(mv[1])["steps"],
        ),
    )
    out = []
    for method, variant in found:
        spec = parse_fc_variant(variant)
        if steps is not None and spec["steps"] not in steps:
            continue
        _, colour, marker = _OURS_STYLE[method]
        out.append((method, variant, fc_label(method, variant), colour,
                    "-" if spec["base"] == "shared" else "--", marker, False))
    return tuple(out)


ALL_SERIES = (*SERIES, *ADAPTED_SERIES, *SURGE_SERIES, *REFERENCE_SERIES)
FOCUS_KEYS: tuple[tuple[str, str | None], ...] = (
    ("Ours (SI-SDE)", "shared"),
    ("Ours (DM-SDE)", "shared"),
    ("Ours (SI-SDE)", "adapted_shared"),
    ("Ours (DM-SDE)", "adapted_shared"),
    ("SURGE (FlowDAS)", None),
    ("SURGE (SDA)", None),
)
FOCUS_SERIES = tuple(
    series for series in ALL_SERIES
    if series[:2] in FOCUS_KEYS or series in SURGE_SERIES or series in REFERENCE_SERIES
)

Key = tuple[str, str, str | None, str]  # scenario, method, variant, metric
Curve = dict[int, tuple[float, float, int]]  # cycle (1-based) -> mean, std, n_seeds


def load(
    path: Path, M: int | None, E: int | None
) -> tuple[dict[Key, Curve], int, int]:
    """Curves for a single ``(M, E)``, which is returned alongside them."""
    with open(path, encoding="utf-8") as fh:
        rows = [
            row for row in csv.DictReader(fh)
            if (M is None or int(row["M"]) == M) and (E is None or int(row["E"]) == E)
        ]
    settings = {(int(row["M"]), int(row["E"])) for row in rows}
    if len(settings) != 1:
        raise SystemExit(
            f"[multistep] need exactly one (M, E) in {path}, found {sorted(settings)}; "
            "select one with --M and --E"
        )
    out: dict[Key, Curve] = defaultdict(dict)
    for row in rows:
        key = (row["scenario"], row["method"], row["variant"] or None, row["metric"])
        out[key][int(row["step"]) + 1] = (
            float(row["value"]), float(row["std"]), int(row["n_traj"]),
        )
    return out, *settings.pop()


def full_seed_count(data) -> int:
    """The seed count of a complete cell: the largest ``n_traj`` in the data."""
    return max(entry[2] for curve in data.values() for entry in curve.values())


def make_figure(
    data, systems: list[str], out_stem: Path, series: tuple[tuple, ...] = ALL_SERIES
) -> list[Path]:
    apply_style()
    n_seeds = full_seed_count(data)
    fig, axes = plt.subplots(
        len(systems), len(METRICS), figsize=(11.0, 4.1 * len(systems)),
        squeeze=False, sharex=True,
    )
    handles: dict[str, object] = {}
    for row, scenario in enumerate(systems):
        for col, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row][col]
            for method, variant, label, colour, ls, marker, filled in series:
                curve = data.get((scenario, method, variant, metric))
                if not curve:
                    continue
                steps = sorted(curve)
                mean = np.array([curve[s][0] for s in steps])
                std = np.array([curve[s][1] for s in steps])
                reference = not marker
                (line,) = ax.plot(
                    steps, mean, color=colour, linestyle=ls, marker=marker or None,
                    markersize=3.5, markevery=3, linewidth=2.2 if reference else 1.3,
                    markerfacecolor=colour if filled else "white",
                    zorder=4 if reference else 3,
                )
                # mean - std can be <= 0, which a log axis cannot show.
                ax.fill_between(
                    steps, np.maximum(mean - std, mean * 0.05), mean + std,
                    color=colour, alpha=0.10, linewidth=0, zorder=2,
                )
                partial = [s for s in steps if curve[s][2] < n_seeds]
                ax.plot(
                    partial, [curve[s][0] for s in partial], linestyle="none",
                    marker="x", markersize=6, color=colour, zorder=5,
                )
                handles.setdefault(label, line)
            ax.set_yscale("log")
            if metric == "kl_points":
                ax.set_ylim(top=KL_CEILING)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_ylabel(ylabel)
            ax.set_title(SYSTEM_LABELS.get(scenario, scenario), fontsize=11)
            if row == len(systems) - 1:
                ax.set_xlabel("Assimilation step $k$")
    labels = [s[2] for s in series if s[2] in handles]
    fig.legend(
        [handles[label] for label in labels], labels, loc="lower center", ncol=4,
        frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.13 * 2 / max(len(systems), 2), 1, 1))
    return _save_fig(fig, out_stem)


def _cell(entry: tuple[float, float, int] | None, tex: bool, n_seeds: int) -> str:
    if entry is None:
        return "--"
    mean, std, n = entry
    fmt = "{:.2e}" if abs(mean) >= 1e3 or abs(mean) < 1e-3 else "{:.4f}"
    if n == 1:  # a std over one seed is meaningless
        return f"{fmt.format(mean)} (n=1)"
    pm = r"$\pm$" if tex else "+/-"
    text = f"{fmt.format(mean)} {pm} {fmt.format(std)}"
    return text if n == n_seeds else f"{text} (n={n})"


def table_rows(data, scenario: str, metric: str) -> list[tuple[str, Curve]]:
    """``(label, curve)`` for every series with data, in legend order.

    Corrected rows come last, at ``FC_TABLE_STEPS`` with online Q_hat and no groups.
    """
    corrected = tuple(
        s for s in fc_series(data, FC_TABLE_STEPS)
        if parse_fc_variant(s[1])["q_source"] == "online"
        and parse_fc_variant(s[1])["group"] is None
    )
    return [
        (label, curve)
        for method, variant, label, *_ in (*ALL_SERIES, *corrected)
        if (curve := data.get((scenario, method, variant, metric)))
    ]


def table_cycles(data) -> list[int]:
    """``TABLE_CYCLES`` that exist, always ending with the final cycle."""
    last = max(cycle for curve in data.values() for cycle in curve)
    return sorted({c for c in TABLE_CYCLES if c <= last} | {last})


def write_markdown(data, scenario: str, path: Path, M: int, E: int) -> None:
    cycles = table_cycles(data)
    n_seeds = full_seed_count(data)
    lines: list[str] = []
    for metric, title in METRICS:
        lines += [f"### {title} ({scenario}, M={M}, E={E})", ""]
        lines.append("| method | " + " | ".join(f"k={c}" for c in cycles) + " |")
        lines.append("|---|" + "---|" * len(cycles))
        for label, curve in table_rows(data, scenario, metric):
            cells = [_cell(curve.get(c), tex=False, n_seeds=n_seeds) for c in cycles]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tex(data, systems: list[str], path: Path) -> None:
    cycles = table_cycles(data)
    n_seeds = full_seed_count(data)
    body: list[str] = [rf"\begin{{tabular}}{{l{'c' * len(cycles)}}}", r"\toprule"]
    for scenario in systems:
        title = SYSTEM_LABELS.get(scenario, scenario)
        for metric, metric_title in METRICS:
            head = " & ".join(f"$k={c}$" for c in cycles)
            body += [
                rf"\multicolumn{{{len(cycles) + 1}}}{{l}}{{{title}: {metric_title}}} \\",
                rf"Method & {head} \\",
                r"\midrule",
            ]
            for label, curve in table_rows(data, scenario, metric):
                cells = " & ".join(
                    _cell(curve.get(c), tex=True, n_seeds=n_seeds) for c in cycles
                )
                body.append(rf"{label} & {cells} \\")
            body.append(r"\midrule")
    body[-1] = r"\bottomrule"
    body.append(r"\end{tabular}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


# -- forecast-residual correction --------------------------------------------- #

FC_METRICS: tuple[tuple[str, str], ...] = (
    ("kl_points", "KL to Kalman posterior"),
    ("var_ratio", "tr(ensemble cov) / tr(Kalman cov)"),
)
FC_COMPARATORS: tuple[tuple, ...] = (
    ("Ours (SI-SDE)", "shared", "Ours SI-SDE (shared), plain", "#0072B2", ":", "o", True),
    ("Ours (DM-SDE)", "shared", "Ours DM-SDE (shared), plain", "#009E73", ":", "s", True),
    ("Ours (FM-ODE)", "shared", "Ours FM-ODE (shared), plain", "#CC79A7", ":", "^", True),
    ("Ours (SI-SDE)", "adapted_shared", "SI-SDE adapted_shared", "#56B4E9", "-.", "", False),
    ("EnKF", None, "EnKF", "#E69F00", "-", "", False),
    *REFERENCE_SERIES,
)


def make_correction_figure(data, systems: list[str], out_stem: Path) -> list[Path]:
    """Per cycle: plain vs corrected Ours (shared, online, all members) and references."""
    corrected = tuple(
        s for s in fc_series(data, FC_FIGURE_STEPS)
        if parse_fc_variant(s[1])["base"] == "shared"
        and parse_fc_variant(s[1])["q_source"] == "online"
        and parse_fc_variant(s[1])["group"] is None
    )
    styles = ("--", "-", (0, (1, 1)))
    restyled = tuple(
        (m, v, lab, c, styles[FC_FIGURE_STEPS.index(parse_fc_variant(v)["steps"]) % 3], mk, f)
        for m, v, lab, c, _, mk, f in corrected
    )
    apply_style()
    fig, axes = plt.subplots(
        len(systems), len(FC_METRICS), figsize=(11.0, 4.1 * len(systems)),
        squeeze=False, sharex=True,
    )
    handles: dict[str, object] = {}
    series = (*FC_COMPARATORS, *restyled)
    for row, scenario in enumerate(systems):
        for col, (metric, ylabel) in enumerate(FC_METRICS):
            ax = axes[row][col]
            for method, variant, label, colour, ls, marker, filled in series:
                curve = data.get((scenario, method, variant, metric))
                if not curve:
                    continue
                steps = sorted(curve)
                mean = np.array([curve[s][0] for s in steps])
                (line,) = ax.plot(
                    steps, mean, color=colour, linestyle=ls, marker=marker or None,
                    markersize=3.5, markevery=3, linewidth=1.3 if marker else 2.0,
                    markerfacecolor=colour if filled else "white",
                )
                handles.setdefault(label, line)
            if metric == "kl_points":
                ax.set_yscale("log")
            else:
                ax.axhline(1.0, color="#999999", linewidth=0.8)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_ylabel(ylabel)
            ax.set_title(SYSTEM_LABELS.get(scenario, scenario), fontsize=11)
            if row == len(systems) - 1:
                ax.set_xlabel("Assimilation step $k$")
    labels = [s[2] for s in series if s[2] in handles]
    fig.legend(
        [handles[label] for label in labels], labels, loc="lower center", ncol=3,
        frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.22 * 2 / max(len(systems), 2), 1, 1))
    return _save_fig(fig, out_stem)


ScalarKey = tuple[str, str, str | None, str]
Scalar = dict[str, float]


def load_scalar(path: Path, M: int, E: int) -> dict[ScalarKey, Scalar]:
    """Cycle-mean rows of ``all.csv`` at one ``(M, E)``: value, std, NFE, seconds, n."""
    out: dict[ScalarKey, Scalar] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if int(row["M"]) != M or int(row["E"]) != E:
                continue
            out[(row["scenario"], row["method"], row["variant"] or None, row["metric"])] = {
                "value": float(row["value"]), "std": float(row["std"] or "nan"),
                "nfe": float(row["NFE"]) if row["NFE"] not in ("", None) else float("nan"),
                "seconds": float(row["seconds"]) if row["seconds"] not in ("", None) else float("nan"),
                "n": int(row["n_traj"]),
            }
    return out


def make_forecast_steps_figure(scalar, systems: list[str], out_stem: Path) -> list[Path]:
    """Cycle-mean KL to Kalman against the forecast steps S of the correction."""
    apply_style()
    fig, axes = plt.subplots(
        len(systems), 2, figsize=(11.0, 4.1 * len(systems)), squeeze=False, sharex=True,
    )
    handles: dict[str, object] = {}
    for row, scenario in enumerate(systems):
        for col, base in enumerate(("shared", "jacfree")):
            ax = axes[row][col]
            lines: dict[tuple, list[tuple[int, float, float]]] = defaultdict(list)
            for (scen, method, variant, metric), entry in scalar.items():
                spec = parse_fc_variant(variant)
                if scen != scenario or metric != "kl_points" or not spec or spec["base"] != base:
                    continue
                lines[(method, spec["q_source"], spec["group"])].append(
                    (spec["steps"], entry["value"], entry["std"])
                )
            for (method, q, group), points in sorted(
                lines.items(), key=lambda kv: (list(_OURS_STYLE).index(kv[0][0]), kv[0][1], kv[0][2] or 0)
            ):
                name, colour, marker = _OURS_STYLE[method]
                points.sort()
                xs, ys, sd = (np.array(v) for v in zip(*points))
                ls = "-" if group is None else "--"
                label = f"{name} + FC" + (", bank" if q == "bank" else "") + (
                    f", E={group}" if group else ""
                )
                (line,) = ax.plot(
                    xs, ys, color=colour, linestyle=ls, marker=marker, markersize=5,
                    markerfacecolor="white" if q == "bank" else colour, linewidth=1.3,
                )
                if group is None and len(points) > 1:
                    ax.fill_between(xs, np.maximum(ys - sd, ys * 0.05), ys + sd,
                                    color=colour, alpha=0.10, linewidth=0)
                handles.setdefault(label, line)
                plain = scalar.get((scenario, method, base, "kl_points"))
                if plain:
                    h = ax.axhline(plain["value"], color=colour, linestyle=":", linewidth=1.0)
                    handles.setdefault(f"{name} ({_BASE_NAME[base]}), plain", h)
            for method, variant, label, colour, ls in (
                (REFERENCE_MEMBER_CHAIN_NAME, None, "Exact per-member chain", "#000000", "-"),
                ("Ours (SI-SDE)", "adapted_shared", "SI-SDE adapted_shared", "#56B4E9", "-."),
                ("EnKF", None, "EnKF", "#E69F00", "-"),
                ("EnKF", "g64", "EnKF, E=64", "#E69F00", "--"),
                ("Kalman filter (sampled)", None, "Sampling floor", "#999999", "-"),
            ):
                entry = scalar.get((scenario, method, variant, "kl_points"))
                if entry:
                    h = ax.axhline(entry["value"], color=colour, linestyle=ls, linewidth=1.2)
                    handles.setdefault(label, h)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([1, 2, 5, 10, 25, 50])
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_ylabel("Cycle-mean KL to Kalman posterior")
            ax.set_title(
                f"{SYSTEM_LABELS.get(scenario, scenario)}: {_BASE_NAME[base]}", fontsize=11
            )
            if row == len(systems) - 1:
                ax.set_xlabel("Forecast steps $S$ of the correction")
    labels = list(handles)
    fig.legend(
        [handles[label] for label in labels], labels, loc="lower center", ncol=4,
        frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.2 * 2 / max(len(systems), 2), 1, 1))
    return _save_fig(fig, out_stem)


REFERENCE_MEMBER_CHAIN_NAME = "Exact per-member chain"
CORRECTION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("kl_points", "KL"), ("kl_exact", "KL (exact moments)"),
    ("var_ratio", "var ratio"), ("mean_err", "mean err"),
)


def correction_rows(scalar, scenario: str) -> list[tuple[str, str | None, str]]:
    """``(method, variant, label)``: comparators first, then every corrected row."""
    keys = {(k[1], k[2]) for k in scalar if k[0] == scenario and k[3] == "kl_points"}
    rows = [
        (m, v, lab) for m, v, lab in (
            (REFERENCE_KALMAN_NAME, None, "Sampling floor (Kalman, sampled)"),
            (REFERENCE_MEMBER_CHAIN_NAME, None, "Exact per-member chain"),
            ("EnKF", None, "EnKF"),
            ("EnKF", "g64", "EnKF, E=64"),
            *(
                (m, v, f"Ours {_OURS_STYLE[m][0]} ({_BASE_NAME.get(v, v)}), plain")
                for m in _OURS_STYLE for v in ("shared", "jacfree")
            ),
            *((m, "adapted_shared", f"Ours {_OURS_STYLE[m][0]} adapted_shared") for m in _OURS_STYLE),
            *((m, "adapted_shared_g64", f"Ours {_OURS_STYLE[m][0]} adapted_shared, E=64")
              for m in _OURS_STYLE),
        ) if (m, v) in keys
    ]
    rows += [(m, v, lab) for m, v, lab, *_ in fc_series({(scenario, *mv) for mv in keys})]
    return rows


REFERENCE_KALMAN_NAME = "Kalman filter (sampled)"


def _scalar_cell(entry: Scalar | None, tex: bool, n_seeds: int) -> str:
    if entry is None:
        return "--"
    return _cell((entry["value"], entry["std"], entry["n"]), tex, n_seeds)


def write_correction_tables(scalar, systems: list[str], md_dir: Path, tex_path: Path) -> list[Path]:
    written: list[Path] = []
    n_seeds = int(max(entry["n"] for entry in scalar.values()))
    tex: list[str] = [
        rf"\begin{{tabular}}{{l{'c' * len(CORRECTION_COLUMNS)}cc}}", r"\toprule",
    ]
    for scenario in systems:
        rows = correction_rows(scalar, scenario)
        if not any(parse_fc_variant(v) for _, v, _ in rows):
            continue
        head = [title for _, title in CORRECTION_COLUMNS] + ["NFE", "s / cycle"]
        lines = [
            f"### Cycle means, forecast-residual correction ({scenario})", "",
            "| method | " + " | ".join(head) + " |", "|---|" + "---|" * len(head),
        ]
        tex += [
            rf"\multicolumn{{{len(head) + 1}}}{{l}}{{{SYSTEM_LABELS.get(scenario, scenario)}}} \\",
            "Method & " + " & ".join(head) + r" \\", r"\midrule",
        ]
        for method, variant, label in rows:
            entries = [scalar.get((scenario, method, variant, m)) for m, _ in CORRECTION_COLUMNS]
            cost = next((e for e in entries if e), None)
            nfe = "--" if cost is None or not np.isfinite(cost["nfe"]) else f"{cost['nfe']:.0f}"
            sec = "--" if cost is None or not np.isfinite(cost["seconds"]) else f"{cost['seconds']:.2f}"
            lines.append(
                f"| {label} | " + " | ".join(_scalar_cell(e, False, n_seeds) for e in entries)
                + f" | {nfe} | {sec} |"
            )
            tex.append(
                label.replace("_", r"\_") + " & " + " & ".join(_scalar_cell(e, True, n_seeds) for e in entries)
                + f" & {nfe} & {sec} \\\\"
            )
        tex.append(r"\midrule")
        target = md_dir / f"correction_{scenario.split(':')[-1]}.md"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(target)
    if written:
        tex[-1] = r"\bottomrule"
        tex.append(r"\end{tabular}")
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
        written.append(tex_path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tex", type=Path, default=None)
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--E", type=int, default=None)
    args = ap.parse_args()
    default_root = args.root.resolve() == ROOT.resolve()
    out = args.out or (DEFAULT_OUT if default_root else args.root / "figures")
    tex = args.tex or (TEX_OUT if default_root else args.root / TEX_OUT.name)

    path = args.root / "aggregated" / "per_step.csv"
    if not path.exists():
        print(f"[multistep] {path} is missing; run the grid and the aggregation first")
        return
    data, M, E = load(path, args.M, args.E)
    systems = sorted({key[0] for key in data})
    print(f"[multistep] M={M}, E={E}, systems={systems}")
    for written in make_figure(data, systems, out / "analytical_multistep_error_vs_step"):
        print(f"[fig] wrote {written}")
    if any(key[2] in {s[1] for s in SURGE_SERIES} for key in data):
        for written in make_figure(
            data, systems, out / "analytical_multistep_surge_vs_step", FOCUS_SERIES
        ):
            print(f"[fig] wrote {written}")
    for scenario in systems:
        target = args.root / "aggregated" / f"table_{scenario.split(':')[-1]}.md"
        write_markdown(data, scenario, target, M, E)
        print(f"[table] wrote {target}")
    write_tex(data, systems, tex)
    print(f"[table] wrote {tex}")
    if any(parse_fc_variant(key[2]) for key in data):
        for written in make_correction_figure(
            data, systems, out / "analytical_multistep_correction_vs_step"
        ):
            print(f"[fig] wrote {written}")
        scalar = load_scalar(args.root / "aggregated" / "all.csv", M, E)
        for written in make_forecast_steps_figure(
            scalar, systems, out / "analytical_multistep_error_vs_forecast_steps"
        ):
            print(f"[fig] wrote {written}")
        for written in write_correction_tables(
            scalar, systems, args.root / "aggregated",
            tex.with_name("tab_analytical_multistep_correction.tex"),
        ):
            print(f"[table] wrote {written}")


if __name__ == "__main__":
    main()
