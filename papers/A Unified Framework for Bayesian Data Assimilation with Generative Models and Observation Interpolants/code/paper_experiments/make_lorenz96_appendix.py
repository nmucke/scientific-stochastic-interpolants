"""Appendix tables and figures of the Lorenz 96 case: every run, both sweeps.

Reads the per-cell files directly (``results/lorenz96/<obs>/{metrics,per_step}/``,
seed 0), so a run that has not finished yet shows up as "pending" in the tables and
is left out of the figures (its legend entry says so). Includes the tagged
shared-Jacobian cells (``--tag shared``), which the aggregated CSVs and the main
figures do not draw.

Writes
  generated/tab_lorenz96_appendix_m50.tex   every cell at M=50 (and the M-free cells),
                                            all metrics, linear and arctan
  generated/tab_lorenz96_appendix_vs_M.tex  every generative cell at every M:
                                            sliced W2, k-NN KL and mean error vs PF
  results/lorenz96/appendix_tables.md       the same tables in Markdown
  figures/lorenz96/appendix/lorenz96_vs_M_{ours,baselines}.{pdf,png}
  figures/lorenz96/appendix/lorenz96_vs_step_M50_{ours,baselines}.{pdf,png}

Bold: best, underlined: second best value of a column within a sweep among the
E=64 methods (the PF
reference rows and EnKF with E=2000 are not ranked); lowest for errors, spread
closest to the PF reference's (its E-draw row, which both sweeps have), innovation
lag-1 closest to 0. Each entry is the mean +- the std over the 12 test sequences
(seed 0). A dagger marks a run with diverged members (share given).

    python paper_experiments/make_lorenz96_appendix.py
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter  # noqa: E402

from figure_common import FIGURES_DIR, RESULTS, _save_fig, apply_style  # noqa: E402
from lorenz96_lineup import M_VALUES, ROW_BY_CELL  # noqa: E402

OBS = ("linear", "arctan")
OBS_TITLE = {"linear": "linear observations", "arctan": r"arctan observations"}
HEADLINE_M = 50
SEED = 0
GENERATED = _here / "generated"


@dataclass(frozen=True)
class Series:
    cell: str
    tag: str  # "" for the default cell
    label: str  # table / legend label
    group: str  # ours | baselines | fixed
    colour: str
    style: str
    marker: str = ""

    @property
    def generative(self) -> bool:
        return self.group != "fixed"

    def stem(self, m: int | None) -> str:
        label = f"{self.cell}-{self.tag}" if self.tag else self.cell
        return label if m is None else f"{label}__M{m}"


def _row(cell: str, label: str, group: str, marker: str = "") -> Series:
    r = ROW_BY_CELL[cell]
    return Series(cell, "", label, group, r.colour, r.style, marker)


def _shared(cell: str, label: str, style: str, marker: str) -> Series:
    return Series(cell, "shared", label, "ours", ROW_BY_CELL[cell].colour, style, marker)


SERIES: tuple[Series, ...] = (
    _row("pf_reference", r"PF reference ($N=4\cdot10^6$)", "fixed"),
    _row("pf_subsample", "PF reference, $E$ draws", "fixed"),
    _row("enkf", "EnKF", "fixed"),
    _row("enkf_large", "EnKF, $E=2000$", "fixed"),
    _row("pf_matched", "Particle filter", "fixed"),
    _row("oracle", "One-step oracle", "fixed"),
    _row("oracle_fc", "One-step oracle + correction", "fixed"),
    _row("prior_enkf", "EnKF, SI prior", "baselines"),
    _row("prior_enkf_fm", "EnKF, FM prior", "baselines"),
    _row("flowdas", "FlowDAS", "baselines"),
    _row("surge_flowdas", "SURGE (FlowDAS)", "baselines"),
    _row("sda", "SDA", "baselines"),
    _row("surge_sda", "SURGE (SDA)", "baselines"),
    _row("dflow", "D-Flow SGLD", "baselines"),
    _row("fig", "Guided FM (FIG)", "baselines"),
    _row("si_sde", "Ours (SI-SDE)", "ours"),
    _row("fm_ode", "Ours (FM-ODE)", "ours"),
    _row("dm_sde", "Ours (DM-SDE)", "ours"),
    _row("si_sde_fc", "Ours (SI-SDE) + correction", "ours"),
    _row("fm_ode_fc", "Ours (FM-ODE) + correction", "ours"),
    _row("dm_sde_fc", "Ours (DM-SDE) + correction", "ours"),
    _shared("si_sde", "Ours (SI-SDE), shared Jacobian", "-", "o"),
    _shared("fm_ode", "Ours (FM-ODE), shared Jacobian", "-", "o"),
    _shared("dm_sde", "Ours (DM-SDE), shared Jacobian", "-", "o"),
    _shared("si_sde_fc", "Ours (SI-SDE), shared Jacobian + correction", "--", "s"),
    _shared("fm_ode_fc", "Ours (FM-ODE), shared Jacobian + correction", "--", "s"),
    _shared("dm_sde_fc", "Ours (DM-SDE), shared Jacobian + correction", "--", "s"),
    _row("surge_si_sde", "Ours (SI-SDE) + SURGE", "ours"),
    _row("surge_dm_sde", "Ours (DM-SDE) + SURGE", "ours"),
    _row("surge_unguided_si", "SURGE, unguided (SI prior)", "ours"),
    _row("surge_unguided_dm", "SURGE, unguided (DM prior)", "ours"),
)
#: Not ranked for bold: the reference itself and the 30x larger EnKF ensemble.
UNRANKED = {"pf_reference", "pf_subsample", "enkf_large"}
#: Horizontal reference lines of the figures (M-free cells).
REFERENCE_LINES = {
    "ours": ("pf_subsample", "enkf"),
    "baselines": ("pf_subsample", "enkf", "enkf_large", "pf_matched", "oracle", "oracle_fc"),
}

# (metric key, column header, rule) -- rule: low | spread | zero | none
COLUMNS = (
    ("mean_error_vs_pf", "mean err.", "low"),
    ("var_error_vs_pf", "var. err.", "low"),
    ("sliced_w2", "sliced $W_2$", "low"),
    ("kl_knn", "$k$-NN KL", "low"),
    ("energy_distance", "energy", "low"),
    ("rmse", "RMSE", "low"),
    ("spread", "spread", "spread"),
    ("crps", "CRPS", "low"),
    ("innovation_lag1", "innov. lag-1", "zero"),
    ("nfe", "NFE/step", "none"),
    ("seconds", "s/step", "none"),
)
VS_M_METRICS = (("sliced_w2", "sliced $W_2$"), ("kl_knn", "$k$-NN KL"),
                ("mean_error_vs_pf", "mean err."))
STEP_METRICS = (("kl_knn", "$k$-NN KL vs PF"), ("sliced_w2", "sliced $W_2$ vs PF"))
FIG_METRICS = (("sliced_w2", "sliced $W_2$ vs PF"), ("kl_knn", "$k$-NN KL vs PF"),
               ("mean_error_vs_pf", "mean error vs PF"))


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def _path(obs: str, kind: str, s: Series, m: int | None) -> Path:
    return RESULTS / "lorenz96" / obs / kind / f"lorenz96__{s.stem(m)}__traj{SEED}__.csv"


def metrics(obs: str, s: Series, m: int | None) -> dict[str, float] | None:
    p = _path(obs, "metrics", s, m if s.generative else None)
    if not p.exists():
        return None
    return {r["metric"]: float(r["value"]) for r in csv.DictReader(p.open())}


def per_step(obs: str, s: Series, m: int, metric: str) -> list[tuple[int, float]] | None:
    p = _path(obs, "per_step", s, m if s.generative else None)
    if not p.exists():
        return None
    rows = [
        (int(r["step"]), float(r["value"]))
        for r in csv.DictReader(p.open())
        if r["metric"] == metric and int(r["test_index"]) == -1
    ]
    return sorted(rows) or None


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def spread_over_sequences(obs: str, s: Series, m: int | None) -> dict[str, float]:
    """Sample std over the test sequences of each time-averaged metric (seed 0)."""
    p = (RESULTS / "lorenz96" / obs / "per_sequence"
         / f"lorenz96__{s.stem(m if s.generative else None)}__traj{SEED}__.json")
    if not p.exists():
        return {}
    out = {}
    for key, values in json.loads(p.read_text()).items():
        finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        if len(finite) > 1:
            out[key] = statistics.stdev(finite)
    return out


def _decimals(v: float, key: str) -> int:
    if key == "nfe":
        return 0
    a = abs(v)
    return 0 if a >= 100 else 1 if a >= 10 else 3


def _num(v: float | None, key: str, decimals: int | None = None) -> str:
    """``v`` rounded to ``decimals`` (default: by its size; a std takes its mean's)."""
    if v is None or not math.isfinite(v):
        return "--" if v is None else ("inf" if v > 0 else "nan")
    return f"{v:.{_decimals(v, key) if decimals is None else decimals}f}"


def _score(v: float, rule: str, ref_spread: float | None) -> float:
    if rule == "spread":
        return abs(v - ref_spread) if ref_spread is not None else v
    return abs(v) if rule == "zero" else v


def _targets(values, rule: str, ref_spread: float | None) -> tuple[float | None, float | None]:
    """(best, second best) scores; ties share a rank, the next distinct value is second."""
    if rule == "none":
        return None, None
    scores = sorted({round(_score(v, rule, ref_spread), 12)
                     for v in values if v is not None and math.isfinite(v)})
    return (scores[0] if scores else None), (scores[1] if len(scores) > 1 else None)


def _rank(v, rule, targets, ref_spread) -> int:
    """1 for the best value, 2 for the second best, 0 otherwise."""
    if v is None or rule == "none" or not math.isfinite(v):
        return 0
    score = round(_score(v, rule, ref_spread), 12)
    return 1 if score == targets[0] else 2 if score == targets[1] else 0


def _cell(v, sd, key, rank, fmt: str) -> str:
    """One table entry: mean (bold if best, underlined if second) +- std over sequences."""
    text = _num(v, key)
    if fmt == "tex":
        text = rf"\textbf{{{text}}}" if rank == 1 else rf"\underline{{{text}}}" if rank == 2 else text
        if sd is not None and v is not None and math.isfinite(v):
            text += rf"{{\scriptsize$\,\pm\,${_num(sd, key, _decimals(v, key))}}}"
        return text
    text = f"**{text}**" if rank == 1 else f"<u>{text}</u>" if rank == 2 else text
    if sd is not None and v is not None and math.isfinite(v):
        text += f" ± {_num(sd, key, _decimals(v, key))}"
    return text


def _diverged(d: dict | None) -> float:
    return 0.0 if not d else d.get("diverged_fraction", 0.0)


def _label(s: Series, d: dict | None, fmt: str) -> str:
    label = s.label if fmt == "tex" else s.label.replace("$", "").replace(r"\cdot", "·")
    frac = _diverged(d)
    if frac > 0:
        label += (rf"$^\dagger$ ({100 * frac:.1f}\%)" if fmt == "tex" else f" † ({100 * frac:.1f}%)")
    return label


def m50_data(obs: str):
    """[(series, metrics or None, std dict)] at M=50 and the rank targets per column."""
    data = [(s, metrics(obs, s, HEADLINE_M), spread_over_sequences(obs, s, HEADLINE_M))
            for s in SERIES]
    ref = next((d for s, d, _ in data if s.cell == "pf_subsample"), None)
    ref_spread = ref["spread"] if ref else None
    targets = {
        key: _targets([d.get(key) for s, d, _ in data if d and s.cell not in UNRANKED],
                      rule, ref_spread)
        for key, _, rule in COLUMNS
    }
    return data, targets, ref_spread


def m50_rows(obs: str, fmt: str) -> list[tuple[Series, str, list[str] | None]]:
    data, targets, ref_spread = m50_data(obs)
    rows = []
    for s, d, sd in data:
        if d is None:
            rows.append((s, _label(s, d, fmt), None))
            continue
        cells = [
            _cell(d.get(key), sd.get(key), key,
                  0 if s.cell in UNRANKED else _rank(d.get(key), rule, targets[key], ref_spread),
                  fmt)
            for key, _, rule in COLUMNS
        ]
        rows.append((s, _label(s, d, fmt), cells))
    return rows


def vs_m_rows(obs: str, key: str, fmt: str) -> list[tuple[Series, str, list[str] | None]]:
    gen = [s for s in SERIES if s.generative]
    data = {(s, m): metrics(obs, s, m) for s in gen for m in M_VALUES}
    targets = {m: _targets([(data[s, m] or {}).get(key) for s in gen], "low", None)
               for m in M_VALUES}
    rows = []
    for s in gen:
        if all(data[s, m] is None for m in M_VALUES):
            rows.append((s, _label(s, None, fmt), None))
            continue
        cells = []
        for m in M_VALUES:
            d = data[s, m]
            if d is None:
                cells.append(r"\emph{pending}" if fmt == "tex" else "*pending*")
                continue
            text = _cell(d.get(key), spread_over_sequences(obs, s, m).get(key), key,
                         _rank(d.get(key), "low", targets[m], None), fmt)
            frac = _diverged(d)
            if frac > 0:
                text += (rf"$^\dagger$" if fmt == "tex" else f" † ({100 * frac:.1f}%)")
            cells.append(text)
        rows.append((s, _label(s, None, fmt), cells))
    return rows


def _tex_body(rows, ncols: int) -> list[str]:
    out, group = [], None
    for s, label, cells in rows:
        if group is not None and s.group != group:
            out.append(r"\addlinespace")
        group = s.group
        if cells is None:
            out.append(f"{label} & " + rf"\multicolumn{{{ncols}}}{{c}}{{\emph{{pending}}}} \\")
        else:
            out.append(f"{label} & " + " & ".join(cells) + r" \\")
    return out


RANK_NOTE = (r"Bold: best, underlined: second best among the $E=64$ methods (spread: "
             r"closest to the PF reference; innovation lag-1: closest to 0). "
             r"$\pm$: standard deviation over the 12 test sequences.")


def latex_m50() -> str:
    out = ["% Generated by make_lorenz96_appendix.py -- do not edit by hand."]
    for obs in OBS:
        out += [
            r"\begin{table}[p]",
            r"\centering",
            rf"\caption{{Lorenz 96, {OBS_TITLE[obs]}: every run at $M=50$ SDE/ODE steps "
            r"(seed 0, 12 test sequences, $E=64$ unless stated; errors against the "
            r"particle-filter reference, averaged over the 50 assimilation steps). "
            + RANK_NOTE + r" $^\dagger$: some ensemble members diverged (share in brackets).}",
            rf"\label{{tab:lorenz96_appendix_m50_{obs}}}",
            r"\resizebox{\linewidth}{!}{%",
            rf"\begin{{tabular}}{{l{'r' * len(COLUMNS)}}}",
            r"\toprule",
            "method & " + " & ".join(h for _, h, _ in COLUMNS) + r" \\",
            r"\midrule",
            *_tex_body(m50_rows(obs, "tex"), len(COLUMNS)),
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
            "",
        ]
    return "\n".join(out)


def latex_vs_m() -> str:
    n = len(M_VALUES)
    out = ["% Generated by make_lorenz96_appendix.py -- do not edit by hand."]
    for key, head in VS_M_METRICS:
        out += [
            r"\begin{table}[p]",
            r"\centering",
            rf"\caption{{Lorenz 96: time-averaged {head} against the PF reference for every "
            r"generative run against the number of SDE/ODE steps $M$ (seed 0, 12 test "
            r"sequences, $E=64$). Hyperparameters are tuned at $M=50$ and reused at every "
            r"$M$. Bold: best, underlined: second best per column. $\pm$: standard "
            r"deviation over the 12 test sequences. $^\dagger$: some ensemble members "
            r"diverged.}",
            rf"\label{{tab:lorenz96_appendix_vs_M_{key}}}",
            r"\resizebox{\linewidth}{!}{%",
            rf"\begin{{tabular}}{{l{'r' * (2 * n)}}}",
            r"\toprule",
            " & " + " & ".join(rf"\multicolumn{{{n}}}{{c}}{{{OBS_TITLE[o].capitalize()}}}" for o in OBS) + r" \\",
            rf"\cmidrule(lr){{2-{1 + n}}}\cmidrule(lr){{{2 + n}-{1 + 2 * n}}}",
            "method & " + " & ".join(f"$M={m}$" for _ in OBS for m in M_VALUES) + r" \\",
            r"\midrule",
        ]
        per_obs = [vs_m_rows(obs, key, "tex") for obs in OBS]
        group = None
        for rows in zip(*per_obs):
            s, label = rows[0][0], rows[0][1]
            if group is not None and s.group != group:
                out.append(r"\addlinespace")
            group = s.group
            cells = []
            for _, _, c in rows:
                cells += c if c is not None else [r"\emph{pending}"] * n
            out.append(f"{label} & " + " & ".join(cells) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return "\n".join(out)


def markdown() -> str:
    md = ["# Lorenz 96 appendix tables (generated by make_lorenz96_appendix.py)\n",
          "Bold: best, <u>underlined</u>: second best among the E=64 methods (spread: closest "
          "to the PF reference; innovation lag-1: closest to 0). ± is the std over the 12 test "
          "sequences. † marks diverged members.\n"]
    for obs in OBS:
        md += [f"\n## {OBS_TITLE[obs].capitalize()}, M = 50\n",
               "| method | " + " | ".join(h.replace("$", "") for _, h, _ in COLUMNS) + " |",
               "|---" * (len(COLUMNS) + 1) + "|"]
        for _, label, cells in m50_rows(obs, "md"):
            md.append(f"| {label} | " + (" | ".join(cells) if cells else
                                         "*pending*" + " |" * (len(COLUMNS) - 1)) + " |")
    for key, head in VS_M_METRICS:
        md += [f"\n## {head.replace('$', '')} against M\n",
               "| method | " + " | ".join(f"{o} M={m}" for o in OBS for m in M_VALUES) + " |",
               "|---" * (2 * len(M_VALUES) + 1) + "|"]
        for rows in zip(*(vs_m_rows(obs, key, "md") for obs in OBS)):
            cells = []
            for _, _, c in rows:
                cells += c if c is not None else ["*pending*"] * len(M_VALUES)
            md.append(f"| {rows[0][1]} | " + " | ".join(cells) + " |")
    return "\n".join(md) + "\n"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _handle(s: Series, pending: bool) -> Line2D:
    label = s.label.replace("$", "") + (" (pending)" if pending else "")
    return Line2D([], [], color=s.colour, ls=s.style, marker=s.marker or None,
                  markersize=4, lw=1.4, label=label)


def _ref_handle(s: Series) -> Line2D:
    return Line2D([], [], color=s.colour, ls=s.style, lw=1.0, alpha=0.8,
                  label=s.label.replace("$", "") + " (no $M$)")


def _log_y(ax) -> None:
    """Log y axis labelled at 1, 2, 5 x 10^k in plain numbers."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def figure_vs_m(group: str) -> list[Path]:
    series = [s for s in SERIES if s.group == group]
    refs = [s for s in SERIES if s.cell in REFERENCE_LINES[group]]
    fig, axes = plt.subplots(len(OBS), len(FIG_METRICS), figsize=(7.2, 4.6), sharex=True)
    pending, any_diverged = set(), False
    for i, obs in enumerate(OBS):
        for j, (key, title) in enumerate(FIG_METRICS):
            ax = axes[i, j]
            for s in series:
                pts = [(m, (metrics(obs, s, m) or {}).get(key)) for m in M_VALUES]
                pts = [(m, v) for m, v in pts if v is not None]
                if not pts:
                    pending.add(s)
                    continue
                ax.plot(*zip(*pts), color=s.colour, ls=s.style, marker=s.marker or "o",
                        markersize=3 if not s.marker else 4, lw=1.4)
                for m, v in pts:  # diverged members: hollow ring on the point
                    if _diverged(metrics(obs, s, m)) > 0:
                        any_diverged = True
                        ax.plot([m], [v], marker="o", mfc="none", mec="#d62728", ms=8, mew=1.2)
            for s in refs:
                d = metrics(obs, s, None)
                if d and key in d:
                    ax.axhline(d[key], color=s.colour, ls=s.style, lw=1.0, alpha=0.8)
            ax.set_xscale("log")
            _log_y(ax)
            ax.set_xticks(M_VALUES)
            ax.set_xticklabels([str(m) for m in M_VALUES])
            ax.minorticks_off()
            if i == 0:
                ax.set_title(title)
            if i == len(OBS) - 1:
                ax.set_xlabel("SDE/ODE steps $M$")
            if j == 0:
                ax.set_ylabel(OBS_TITLE[obs])
    handles = [_handle(s, s in pending) for s in series] + [_ref_handle(s) for s in refs]
    if any_diverged:
        handles.append(Line2D([], [], ls="none", marker="o", mfc="none", mec="#d62728", ms=8,
                              label="diverged members"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=3, fontsize=7, frameon=False)
    fig.tight_layout()
    return _save_fig(fig, FIGURES_DIR / "lorenz96" / "appendix" / f"lorenz96_vs_M_{group}")


def figure_vs_step(group: str) -> list[Path]:
    series = [s for s in SERIES if s.group == group]
    refs = [s for s in SERIES if s.cell in REFERENCE_LINES[group]]
    fig, axes = plt.subplots(len(OBS), len(STEP_METRICS), figsize=(7.2, 4.6), sharex=True)
    pending = set()
    for i, obs in enumerate(OBS):
        for j, (key, title) in enumerate(STEP_METRICS):
            ax = axes[i, j]
            for s in (*series, *refs):
                pts = per_step(obs, s, HEADLINE_M, key)
                if pts is None:
                    pending.add(s)
                    continue
                steps, values = zip(*pts)
                steps = [k + 1 for k in steps]  # step 1 = first analysis
                is_ref = s in refs
                ax.plot(steps, values, color=s.colour, ls=s.style, lw=1.0 if is_ref else 1.3,
                        alpha=0.8 if is_ref else 1.0, marker=s.marker or None,
                        markevery=5, markersize=3)
            _log_y(ax)
            if i == 0:
                ax.set_title(title)
            if i == len(OBS) - 1:
                ax.set_xlabel("assimilation step")
            if j == 0:
                ax.set_ylabel(OBS_TITLE[obs])
    handles = [_handle(s, s in pending) for s in series] + [
        Line2D([], [], color=s.colour, ls=s.style, lw=1.0, alpha=0.8, label=s.label.replace("$", ""))
        for s in refs
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=3, fontsize=7, frameon=False)
    fig.tight_layout()
    return _save_fig(fig, FIGURES_DIR / "lorenz96" / "appendix" / f"lorenz96_vs_step_M50_{group}")


def main() -> None:
    apply_style()
    GENERATED.mkdir(exist_ok=True)
    (GENERATED / "tab_lorenz96_appendix_m50.tex").write_text(latex_m50())
    (GENERATED / "tab_lorenz96_appendix_vs_M.tex").write_text(latex_vs_m())
    (RESULTS / "lorenz96" / "appendix_tables.md").write_text(markdown())
    written = []
    for group in ("ours", "baselines"):
        written += figure_vs_m(group)
        written += figure_vs_step(group)
    for p in [GENERATED / "tab_lorenz96_appendix_m50.tex", GENERATED / "tab_lorenz96_appendix_vs_M.tex",
              RESULTS / "lorenz96" / "appendix_tables.md", *written]:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
