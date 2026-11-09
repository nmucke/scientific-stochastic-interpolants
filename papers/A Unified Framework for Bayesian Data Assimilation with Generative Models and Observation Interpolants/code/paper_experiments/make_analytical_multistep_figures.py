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
  mean +/- std at selected cycles. Every cycle is in ``per_step.csv``.

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

from common.seeding import SEED_LIST  # noqa: E402
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


def make_figure(
    data, systems: list[str], out_stem: Path, series: tuple[tuple, ...] = ALL_SERIES
) -> list[Path]:
    apply_style()
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
                partial = [s for s in steps if curve[s][2] < len(SEED_LIST)]
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


def _cell(entry: tuple[float, float, int] | None, tex: bool) -> str:
    if entry is None:
        return "--"
    mean, std, n = entry
    fmt = "{:.2e}" if abs(mean) >= 1e3 or abs(mean) < 1e-3 else "{:.4f}"
    if n == 1:  # a std over one seed is meaningless
        return f"{fmt.format(mean)} (n=1)"
    pm = r"$\pm$" if tex else "+/-"
    text = f"{fmt.format(mean)} {pm} {fmt.format(std)}"
    return text if n == len(SEED_LIST) else f"{text} (n={n})"


def table_rows(data, scenario: str, metric: str) -> list[tuple[str, Curve]]:
    """``(label, curve)`` for every series with data, in legend order."""
    return [
        (label, curve)
        for method, variant, label, *_ in ALL_SERIES
        if (curve := data.get((scenario, method, variant, metric)))
    ]


def table_cycles(data) -> list[int]:
    """``TABLE_CYCLES`` that exist, always ending with the final cycle."""
    last = max(cycle for curve in data.values() for cycle in curve)
    return sorted({c for c in TABLE_CYCLES if c <= last} | {last})


def write_markdown(data, scenario: str, path: Path, M: int, E: int) -> None:
    cycles = table_cycles(data)
    lines: list[str] = []
    for metric, title in METRICS:
        lines += [f"### {title} ({scenario}, M={M}, E={E})", ""]
        lines.append("| method | " + " | ".join(f"k={c}" for c in cycles) + " |")
        lines.append("|---|" + "---|" * len(cycles))
        for label, curve in table_rows(data, scenario, metric):
            cells = [_cell(curve.get(c), tex=False) for c in cycles]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tex(data, systems: list[str], path: Path) -> None:
    cycles = table_cycles(data)
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
                cells = " & ".join(_cell(curve.get(c), tex=True) for c in cycles)
                body.append(rf"{label} & {cells} \\")
            body.append(r"\midrule")
    body[-1] = r"\bottomrule"
    body.append(r"\end{tabular}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tex", type=Path, default=None)
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--E", type=int, default=None)
    args = ap.parse_args()
    default_root = args.root == ROOT
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


if __name__ == "__main__":
    main()
