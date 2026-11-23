#!/usr/bin/env python3
"""Per-case result tables (one LaTeX table per test case) for the manuscript revision.

Reads the per-trajectory / per-sequence / per-seed result files in
/export/scratch2/ntm/delftblue_results/results (read only) and writes

    results_tables_body.tex        every method           (python3 make_results_tables.py)
    results_tables_ours_body.tex   our samplers only      (python3 make_results_tables.py --ours-only)

next to this script; results_tables.tex / results_tables_ours.tex \\input them.

Every entry is a time-averaged metric (averaged over the assimilation steps / cycles
inside each run), reported as mean +/- sample std (ddof=1) over independent
trajectories (NS, uDALES: held-out test trajectories; Lorenz 96: test sequences;
multistep linear-Gaussian: seeds). Diverged trajectories are excluded and counted.

All five tables share ONE layout, the one of the manuscript's results tables:
method rows grouped under italic headers (our variants nested as "\\quad + ..."
rows under their sampler), one column group per metric with one sub-column per
observation scenario, bold = best and underline = second best per column.
Method names follow the manuscript (sections_review/*).
"""

from __future__ import annotations

import json
import re
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("/export/scratch2/ntm/delftblue_results/results")
HERE = Path(__file__).resolve().parent
NS_GUARD = 10.0          # divergence_rmse_threshold of the NS / uDALES runs
MS_M = 250               # multistep: largest sampler-step count run
L96_M = 250              # Lorenz 96: largest sampler-step count run
FLOAT = False            # --manuscript: emit the combined table as a table float
OURS_ONLY = False        # --ours-only: keep only the groups of our samplers

SAMPLERS = (("SI-SDE", "si_sde"), ("DM-SDE", "dm_sde"), ("FM-ODE", "fm_ode"))
SHARED_K10 = r"Ours -- ensemble-shared inflated covariance (Jacobian refreshed every $k=10$ steps)"
JACFREE = "Ours -- Jacobian-free covariance"


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #


@dataclass
class Metric:
    key: str
    header: str
    fmt: str = "d2"        # "d<N>": N decimals | "sig": 3 significant digits
    scale: float = 1.0     # unit conversion applied before printing
    exp: int = 0           # column factor: entries are printed in units of 10^exp (header)
    std: bool = True       # print +/- std
    rank: bool = True      # bold / underline this column (lower is better)


@dataclass
class Cell:
    mean: float = math.nan
    std: float = math.nan
    n: int = 0             # trajectories averaged
    n_div: int = 0         # diverged (excluded)
    n_exp: int = 0         # expected trajectories (0 = not applicable)
    text: str = ""         # fixed text instead of a number (e.g. "= EnKF")
    note: str = ""         # superscript note


@dataclass
class Row:
    label: str
    cells: dict = field(default_factory=dict)   # (metric key, scenario key) -> Cell
    ranked: bool = True    # take part in bold / underline
    rid: object = None     # source-cell id (used to merge rows across tables)


@dataclass
class Table:
    caption: str
    label: str
    metrics: list
    scenarios: list        # [(key, header)]; one scenario -> no sub-header row
    groups: list = field(default_factory=list)  # [(title, [Row])]
    # optional explicit column groups [(header, [(sub-header, Metric, scenario key)])];
    # default: one group per metric with one sub-column per scenario
    colgroups: list | None = None
    short: str = ""        # list-of-tables entry
    inline_pm: bool = False  # +/- on the same line as the value (compact tables)


def mean_std(xs) -> tuple[float, float, int]:
    xs = [float(x) for x in xs if x is not None and np.isfinite(float(x))]
    if not xs:
        return (math.nan, math.nan, 0)
    if len(xs) == 1:
        return (xs[0], math.nan, 1)
    return (float(np.mean(xs)), float(np.std(xs, ddof=1)), len(xs))


# --------------------------------------------------------------------------- #
# formatting + rendering (shared by all tables)
# --------------------------------------------------------------------------- #


def _sci(a: float) -> bool:
    """Only extreme outliers keep a per-number power of ten."""
    return a < 1e-3 or a >= 1e5


def _dec(m: Metric, a: float) -> int:
    """Decimals of the mean: fixed for "d<N>", 3 significant digits for "sig"."""
    if m.fmt.startswith("d"):
        return int(m.fmt[1:])
    return max(0, 2 - int(math.floor(math.log10(a)))) if a > 0 else 0


def _num(v: float, m: Metric) -> str:
    """Printed mean (also the key for ties: equal printed values tie). The column
    factor 10^exp (shown in the header) is divided out."""
    v = v * m.scale / 10.0 ** m.exp
    a = abs(v)
    if m.fmt == "sig" and a and _sci(a):
        e = int(math.floor(math.log10(a)))
        return f"{v / 10**e:.2f}" + rf"{{$\cdot$}}$10^{{{e}}}$"
    out = f"{v:.{_dec(m, a)}f}"
    return out.lstrip("-") if out.startswith("-") and not out.strip("-0.") else out


def _pm(c: Cell, m: Metric, factor: bool = True) -> str:
    """The +/- of a cell, with the decimals of the mean (more only if it would round to 0)."""
    if not (m.std and c.n >= 2 and np.isfinite(c.std)):
        return ""
    v = c.mean * m.scale / 10.0 ** m.exp
    sd = c.std * m.scale / 10.0 ** m.exp
    a, fac = abs(v), ""
    if m.fmt == "sig" and a and _sci(a):
        e = int(math.floor(math.log10(a)))
        t, fac = f"{sd / 10**e:.2f}", rf"{{$\cdot$}}$10^{{{e}}}$"
    else:
        d = _dec(m, a)
        top = d + 3
        while d < top and sd > 0 and round(sd, d) == 0:
            d += 1
        t = f"{sd:.{d}f}"
    return rf"{{\tiny$\pm${t}{fac if factor else ''}}}"


def _stack(top: str, bottom: str) -> str:
    """Manuscript-appendix cell: value on top, +/- std below."""
    if not bottom:
        return top
    return rf"\begin{{tabular}}[c]{{@{{}}c@{{}}}}{top}\\[{"1pt" if FLOAT else "-2.5pt"}]{bottom}\end{{tabular}}"


def _rankable(r: Row, c: Cell | None) -> bool:
    return (r.ranked and c is not None and not c.text and not c.n_div
            and np.isfinite(c.mean) and (not c.n_exp or c.n == c.n_exp))


def render(t: Table) -> str:
    groups = t.groups
    caption, label = t.caption, t.label
    if OURS_ONLY:
        groups = [(g, rs) for g, rs in groups if g.startswith("Ours")]
        caption = r"\textbf{Our samplers only.} " + re.sub("\x00.*?\x01", "", caption, flags=re.S)
        label += "_ours"
    caption = caption.replace("\x00", "").replace("\x01", "")
    rows = [r for _, rs in groups for r in rs]
    metric_in_sub = t.colgroups is not None   # combined table: metric names are sub-headers
    cg = t.colgroups or [(m.header, [(h if len(t.scenarios) > 1 else "", m, s)
                                     for s, h in t.scenarios]) for m in t.metrics]

    # per-column power-of-ten factor (auto-formatted metrics only): chosen so the
    # column's median entry prints as an ordinary number; shown in the header
    def col_exp(m, s):
        return m.exp

    fac = lambda e: rf" ($\times 10^{{{e}}}$)" if e else ""  # noqa: E731
    ARROW = r"\,$\downarrow$"   # every metric here is lower-is-better
    new_cg = []
    for hd, subs in cg:
        exps = [col_exp(m, s) for _, m, s in subs]
        subs2 = list(subs)
        if metric_in_sub:
            subs2 = [(h + fac(e) + ARROW, m, s) for (h, m, s), e in zip(subs2, exps)]
        elif len(set(exps)) == 1:
            hd = hd + fac(exps[0]) + ARROW
        else:
            hd = hd + ARROW
            subs2 = [(h + fac(e), m, s) for (h, m, s), e in zip(subs2, exps)]
        new_cg.append((hd, subs2))
    cg = new_cg
    if not getattr(t, "no_factor_note", False) and \
            any(r"\times 10^" in h for hd, subs in cg for h in [hd, *(x for x, _, _ in subs)]):
        caption += (r" $\times 10^{k}$ in a header: entries are multiples of $10^{k}$." if FLOAT else
                    r" A factor $\times 10^{k}$ in a header means that the entries of that "
                    r"column are multiples of $10^{k}$.")
    cols = [(m, s) for _, subs in cg for _, m, s in subs]
    ncol = 1 + len(cols)

    # best / second best per column, ties at the printed precision
    rank = {}
    for m, s in cols:
        if not m.rank:
            continue
        vals = sorted({(r.cells[(m.key, s)].mean * m.scale, _num(r.cells[(m.key, s)].mean, m))
                       for r in rows if _rankable(r, r.cells.get((m.key, s)))})
        distinct = []
        for _, txt in vals:
            if txt not in distinct:
                distinct.append(txt)
        rank[(m.key, s)] = (distinct[:1], distinct[1:2])

    spec = "@{}l|" + "|".join("c" * len(subs) for _, subs in cg) + "@{}"
    h1 = []
    for i, (hd, subs) in enumerate(cg):
        bar = "|" if i < len(cg) - 1 else ""
        h1.append(rf"\multicolumn{{{len(subs)}}}{{c{bar}}}{{\textbf{{{hd}}}}}")
    head = [r"\toprule", " & ".join([""] + h1) + r" \\"]
    if any(h for _, subs in cg for h, _, _ in subs):
        head.append(" & ".join([""] + [h for _, subs in cg for h, _, _ in subs]) + r" \\")
    if FLOAT:   # manuscript float: table + tabular (no page breaks), scaled to the text width
        caption = "\\" + getattr(t, "rev", "revB") + "{" + caption + "}"   # reviewer colouring
        out = [rf"\begin{{table}}[{getattr(t, 'placement', 'ht')}]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
               rf"\caption{{{caption}}}", rf"\label{{{label}}}",
               r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
               rf"\begin{{tabular}}{{{spec}}}", *head]
    else:
        out = None
    out = out or [r"\begin{center}", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
           rf"\begin{{longtable}}{{{spec}}}",
           rf"\caption[{t.short or label}]{{{caption}}}\label{{{label}}}\\", *head, r"\endfirsthead",
           rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\emph{{Table~\ref{{{label}}} (continued)}}}}\\",
           *head, r"\endhead", r"\bottomrule", r"\endfoot"]
    for gi, (gt, rs) in enumerate(groups):
        out.append(r"\midrule")
        out.append(rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\rule{{0pt}}{{2.1ex}}\textit{{{gt}}}}} \\*")
        for r in rs:
            cells = [r.label]
            for m, s in cols:
                c = r.cells.get((m.key, s))
                if c is None:
                    cells.append("--")
                    continue
                if c.text:
                    cells.append(c.text)
                    continue
                if c.n_exp and c.n_div == c.n_exp:
                    cells.append(rf"\textcolor{{gray}}{{div\,{c.n_div}/{c.n_exp}}}")  # every run diverged
                    continue
                if not np.isfinite(c.mean):
                    cells.append("--")
                    continue
                num = _num(c.mean, m)
                best, second = rank.get((m.key, s), ([], []))
                if _rankable(r, c) and num in best:
                    num = rf"\textbf{{{num}}}"
                elif _rankable(r, c) and num in second:
                    num = rf"\underline{{{num}}}"
                sup = ""
                if c.n_div:
                    sup += r"$^{\ast}$"   # some runs diverged
                if c.note:
                    sup += rf"\,\textsuperscript{{\tiny {c.note}}}"
                if t.inline_pm:
                    # value{+/-std}[x10^k][superscripts]
                    pm = _pm(c, m, factor=False)
                    k = num.find(r"{$\cdot$}")
                    pm = (r"\," + pm) if pm else ""
                    cells.append((num + pm if k < 0 else num[:k] + pm + num[k:]) + sup)
                else:
                    cells.append(_stack(num + sup, _pm(c, m)))
            # page breaks only between method groups
            out.append(" & ".join(cells) + (r" \\" if r is rs[-1] else r" \\*"))
    if FLOAT:
        out += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    else:
        out += [r"\end{longtable}", r"\end{center}", ""]
    return "\n".join(out)


BOLD = (r"\textbf{Bold} marks the best value in each column and \underline{underline} the "
        r"second best, each together with every value equal to it at the printed precision. ")
STD = r"$\pm$ gives the standard deviation over "


def NO(text: str) -> str:
    """Caption text about rows that the --ours-only file leaves out."""
    return "\x00" + text + "\x01"


# --------------------------------------------------------------------------- #
# Case 1: multistep linear-Gaussian
# --------------------------------------------------------------------------- #


def multistep(M: int = MS_M) -> Table:
    files = sorted((RES / "analytical_multistep/metrics").glob(f"paper*__M{M}__K20__E100000__traj*.csv"))
    assert len(files) == 20, files   # paper, paper_surge, paper_fc<M>, paper_tmpd: 4 cells x 5 seeds
    d = pd.concat([pd.read_csv(f).assign(traj=f.name.split("__traj")[1].split(".")[0]) for f in files])
    d["variant"] = d["variant"].fillna("")
    assert set(d.M) == {M} and set(d.scenario) == {"multistep:paper"}
    metrics = [Metric("kl_points", "Gaussian KL", "sig", exp=-3),
               Metric("sliced_w1", "SW$_1$", "sig", exp=-2),
               Metric("sliced_w2", "SW$_2$", "sig", exp=-2)]
    S0 = "paper"

    def row(label, method, variant, ranked=True):
        s = d[(d.method == method) & (d.variant == variant)]
        assert not s.empty, (method, variant)
        assert s.groupby("metric").traj.nunique().eq(5).all()
        # a seed with any non-finite metric (collapsed ensemble) counts as diverged and is
        # excluded from EVERY metric, so all columns average the same seeds
        wide = s.pivot_table(index="traj", columns="metric", values="value", aggfunc="first")
        ok = wide[np.isfinite(wide.astype(float)).all(axis=1)]
        r = Row(label, ranked=ranked, rid=(method, variant))
        for m in metrics:
            mu, sd, n = mean_std(ok[m.key])
            r.cells[(m.key, S0)] = Cell(mu, sd, n, n_div=5 - len(ok), n_exp=5)
        return r

    t = Table("", "tab:res_multistep", metrics, [(S0, "")])
    meth = {"si_sde": "Ours (SI-SDE)", "dm_sde": "Ours (DM-SDE)", "fm_ode": "Ours (FM-ODE)"}
    for cov, title in (("shared", "Ours -- ensemble-shared inflated covariance (exact in this case)"),
                       ("jacfree", JACFREE)):
        rs = []
        for lab, a in SAMPLERS:
            rs.append(row(f"Ours ({lab})", meth[a], cov))
            rs.append(row(rf"\quad + corr.\ ($S={M}$)", meth[a], f"{cov}_fc{M}"))
            if a != "fm_ode":
                rs.append(row(r"\quad + SURGE", meth[a], f"surge_{cov}"))
        t.groups.append((title, rs))
    t.groups.append(("Generative baselines", [
        row("FlowDAS", "FlowDAS", ""), row("FlowDAS + SURGE", "SURGE (FlowDAS)", ""),
        row("SDA", "SDA", ""), row("SDA + SURGE", "SURGE (SDA)", ""),
        row("D-Flow SGLD", "D-Flow SGLD", ""), row("Guided FM (FIG)", "Guided FM (FIG)", ""),
        row("TMPD", "TMPD", "")]))
    t.groups.append(("Classical filters (exact model)", [
        row("EnKF", "EnKF", "", False), row("Particle filter", "Particle filter", "", False)]))
    t.groups.append(("References", [
        row("Kalman filter (sampled)", "Kalman filter (sampled)", "", False),
        row("Exact per-member chain", "Exact per-member chain", "", False)]))
    t.caption = (
        rf"Multistep linear--Gaussian test, paper system ($A=Q=\obsmatrix=\obscov=\bI$, $d=2$): "
        rf"Gaussian KL and sliced $W_1$/$W_2$ (SW$_1$, SW$_2$) to the Kalman filtering posterior, "
        rf"averaged over the $K=20$ cycles (lower is better). $N=100{{,}}000$ members, $M={M}$ "
        rf"sampler steps (the largest run); ``+\,corr.'' applies the forecast-residual correction "
        rf"with online $\hat Q$ and $S=M$ forecast steps. " + STD +
        r"five seeds. " + BOLD +
        NO(r"The classical filters and references are not ranked. ") + r"$^{\ast}$: the ensemble collapsed "
        r"(non-finite metric) on some seeds; the entry is the mean over the remaining seeds and is "
        r"not ranked.")
    return t


# --------------------------------------------------------------------------- #
# Cases 2, 3: Lorenz 96
# --------------------------------------------------------------------------- #


def lorenz96(obs: str, M: int = L96_M) -> Table:
    root = RES / "lorenz96" / obs
    metrics = [Metric("sliced_w2", "SW$_2$", "d2"), Metric("kl_knn", "$k$NN-KL", "d2"),
               Metric("energy_distance", "Energy distance", "d2")]
    S0 = obs

    def row(label, cell, has_m=True, ranked=True):
        stem = f"lorenz96__{cell}__M{M}__traj0__" if has_m else f"lorenz96__{cell}__traj0__"
        ps, mt = root / "per_sequence" / f"{stem}.json", root / "metrics" / f"{stem}.csv"
        if not ps.exists():
            return None
        seq = json.load(open(ps))
        if mt.exists():  # sanity: per-sequence mean == stored cell value
            m = pd.read_csv(mt)
            for k in ("rmse", "sliced_w2"):
                mv = float(m[m.metric == k].value.iloc[0])
                assert abs(mv - float(np.mean(seq[k]))) < 1e-6 * max(1, abs(mv)), (cell, k)
        n = len(seq["sliced_w2"])
        # a sequence is diverged if any reported metric is non-finite there
        # (diverged_fraction is the fraction of members dropped per step, not a
        # per-sequence flag)
        good = [all(np.isfinite(float(seq[mm.key][i])) for mm in metrics) for i in range(len(seq["sliced_w2"]))]
        n_div = len(good) - sum(good)
        seq = {k: [x for x, g in zip(v, good) if g] if isinstance(v, list) and len(v) == len(good) else v
               for k, v in seq.items()}
        r = Row(label, ranked=ranked, rid=cell)
        for mm in metrics:
            mu, sd, k = mean_std(seq[mm.key])
            r.cells[(mm.key, S0)] = Cell(mu, sd, k, n_div=n_div, n_exp=n)
        return r

    def rows(spec, **kw):
        return [r for r in (row(*s, **kw) for s in spec) if r is not None]

    t = Table("", f"tab:res_l96_{obs}", metrics, [(S0, "")])
    rs = []
    for lab, c in SAMPLERS:
        rs += rows([(f"Ours ({lab})", c), (r"\quad + corr.", f"{c}_fc")])
        if c != "fm_ode":
            rs += rows([(r"\quad + SURGE", f"surge_{c}")])
    t.groups.append((JACFREE, rs))
    rs = []
    for lab, c in SAMPLERS:
        rs += rows([(f"Ours ({lab})", f"{c}-shared"), (r"\quad + corr.", f"{c}_fc-shared")])
    t.groups.append((r"Ours -- ensemble-shared inflated covariance (Jacobian refreshed every step, $k=1$)", rs))
    t.groups.append(("Generative baselines", rows([
        ("FlowDAS", "flowdas"), ("FlowDAS + SURGE", "surge_flowdas"), ("SDA", "sda"),
        ("SDA + SURGE", "surge_sda"), ("D-Flow SGLD", "dflow"), ("Guided FM (FIG)", "fig"),
        ("TMPD", "tmpd")])))
    t.groups.append(("Generative prior + EnKF analysis", rows([
        ("Trained prior (SI) + EnKF", "prior_enkf"), ("Trained prior (FM) + EnKF", "prior_enkf_fm")])))
    t.groups.append(("Classical filters (true model)", rows([
        ("EnKF ($E=64$)", "enkf"), ("EnKF ($E=2000$)", "enkf_large"),
        ("Particle filter ($E=64$)", "pf_matched")], has_m=False, ranked=False)))
    t.groups.append(("Controls and references", rows([
        ("Oracle (exact one-step kernel)", "oracle"), ("Oracle + corr.", "oracle_fc"),
        ("PF subsample ($64$ draws, floor)", "pf_subsample"),
        ("PF reference (2nd replicate)", "pf_reference")], has_m=False, ranked=False)))
    name = {"linear": r"linear observations $\obs^n=L\state^n+\obsnoise^n$",
            "arctan": r"arctan observations $\obsoperator(\state)=\arctan(cL\state)/c$"}[obs]
    tmpd = (NO(r" TMPD (untuned) was run for linear observations only.")
            if obs == "linear" else "")
    t.caption = (
        rf"Lorenz~96, {name}: sliced $W_2$ (SW$_2$), $k$-nearest-neighbour KL and energy distance "
        r"to the particle-filter reference ($4\cdot10^6$ particles), averaged over the 50 "
        rf"assimilation steps (lower is better). $E=64$, generative samplers $M={M}$ (the largest "
        r"run); ``+\,corr.'' applies the forecast-residual correction with online $\hat Q$ and "
        r"$S=M$ forecast steps. " + STD + r"the 12 test sequences (seed 0). " + BOLD +
        NO(r"The classical filters, controls and references are not ranked.") + tmpd)
    return t


# --------------------------------------------------------------------------- #
# Cases 4, 5: field cases (NS wrong IC, uDALES)
# --------------------------------------------------------------------------- #


def _traj_cell(files: list[Path], scen: str, method: str | None = None,
               variant: str | None = None) -> dict | None:
    """{metric: Cell} over trajectories; diverged = rmse NaN or any per-step rmse above
    the guard (catches a guard trip on the last step, which leaves rmse finite)."""
    per, n_div, n_fin = {}, 0, 0
    for f in files:
        if not f.exists():
            continue
        n_fin += 1
        d = pd.read_csv(f)
        d = d[d.scenario == scen]
        if method is not None:
            d = d[d.method == method]
        if variant is not None:
            d = d[d.variant.fillna("") == variant]
        assert not d.empty, (f, scen, method, variant)
        v = dict(zip(d.metric, d.value.astype(float)))
        div = not np.isfinite(v.get("rmse", math.nan))
        ps = f.parent.parent / "per_step" / f.name
        if not div and ps.exists():
            p = pd.read_csv(ps)
            p = p[(p.scenario == scen) & (p.metric == "rmse")]
            if method is not None:
                p = p[p.method == method]
            if variant is not None and "variant" in p:
                p = p[p.variant.fillna("") == variant]
            if len(p) and (p.value.max() > NS_GUARD or p.value.isna().any()):
                div = True
        if div:
            n_div += 1
            continue
        for k, x in v.items():
            per.setdefault(k, []).append(x)
    if n_fin == 0:
        return None
    out = {}
    keys = set(per) | {"rmse"}
    for k in keys:
        mu, sd, n = mean_std(per.get(k, []))
        out[k] = Cell(mu, sd, n, n_div=n_div, n_exp=n_fin,
                      note="")  # missing trajectory files are stated in the caption
    return out


def _field_row(t: Table, label: str, per_scen: dict, ranked=True) -> Row:
    """per_scen: scenario key -> {metric: Cell} | Cell (fixed text) | None."""
    r = Row(label, ranked=ranked)
    for m in t.metrics:
        for s, _ in t.scenarios:
            got = per_scen.get(s)
            if got is None:
                continue
            if isinstance(got, Cell):
                r.cells[(m.key, s)] = got
            elif m.key in got:
                r.cells[(m.key, s)] = got[m.key]
            elif "rmse" in got and got["rmse"].n_div == got["rmse"].n_exp:
                r.cells[(m.key, s)] = got["rmse"]   # all diverged
    return r


NS = RES / "navier_stokes_wrong_ic_np40"
NS_TRAJS = (10, 11, 12, 13, 14)
NS_SCEN = [("16^2->128^2", "16_2_128_2", r"$16^2$"), ("32^2->128^2", "32_2_128_2", r"$32^2$"),
           ("sparse 5%", "sparse_5", r"$5\%$"), ("sparse 1.5625%", "sparse_1_5625", r"$\tfrac{1}{64}$")]


def ns() -> Table:
    main = NS / "main/navier_stokes/metrics"
    ext = NS / "ext/navier_stokes/metrics"
    pe = NS / "prior_enkf/navier_stokes/metrics"
    tm = NS / "tmpd/navier_stokes/metrics"
    l20 = NS / "enkf_loc20_superres/production/metrics"
    FC = (("8", "fc25_70a3134d"), ("20", "fc25_8797d2ae"))
    metrics = [Metric("rmse", "Vorticity RMSE", "d2", exp=-1),
               Metric("crps", "CRPS", "d2", exp=-1),
               Metric("spread_skill", "Spread--skill", "d2", exp=-1),   # Table 7 only
               Metric("energy_spec_rmse", "KE-spectrum RMSE", "d2"),
               Metric("seconds", r"Cost (s/step)$^{\dagger}$", "d1", std=False, rank=False)]
    t = Table("", "tab:res_ns_wrong_ic", metrics, [(s, h) for s, _, h in NS_SCEN])

    def row(label, stem, ranked=True, scens=None, text=None):
        per = {}
        for s, slug, _ in NS_SCEN:
            if scens is not None and s not in scens:
                per[s] = Cell(text=text) if text else None
                continue
            got = _traj_cell([Path(stem.format(s=slug, t=tr)) for tr in NS_TRAJS], s)
            assert got is not None, (label, s)
            per[s] = got
        return _field_row(t, label, per, ranked)

    for cov, title in (("shared_jac10", SHARED_K10), ("jacfree", JACFREE)):
        rs = []
        for lab, a in SAMPLERS:
            rs.append(row(f"Ours ({lab})", f"{main}/{{s}}__M100__traj{{t}}__ours_{cov}__{a}.csv"))
            for r_, tag in FC:
                rs.append(row(rf"\quad + corr.\ ($r={r_}$)", f"{ext}/{{s}}__M100__traj{{t}}__ours_{cov}__{tag}__{a}.csv"))
            if a != "fm_ode":
                rs.append(row(r"\quad + SURGE", f"{ext}/{{s}}__M100__traj{{t}}__surge_ours_{cov}__{a}.csv"))
        t.groups.append((title, rs))
    rs = [row(lab, f"{main}/{{s}}__M100__traj{{t}}__baselines__{k}.csv") for lab, k in (
        ("FlowDAS", "flowdas"), ("FlowDAS + SURGE", "surge_flowdas"), ("SDA", "sda"),
        ("SDA + SURGE", "surge_sda"), ("D-Flow SGLD", "dflow_sgld"), ("Guided FM (FIG)", "fig"))]
    rs.append(row("TMPD", f"{tm}/{{s}}__M100__traj{{t}}__baselines__tmpd.csv"))
    t.groups.append(("Generative baselines", rs))
    t.groups.append(("Generative prior + EnKF analysis", [
        row(f"{lab} prior + EnKF", f"{pe}/{{s}}__M100__traj{{t}}__prior_enkf__{a}.csv") for lab, a in SAMPLERS]))
    superres = ("16^2->128^2", "32^2->128^2")
    t.groups.append(("Classical filters (true solver)", [
        row("EnKF", f"{main}/{{s}}__traj{{t}}__classical__enkf.csv", ranked=False),
        row(r"EnKF ($r=20$ everywhere)", f"{l20}/{{s}}__traj{{t}}__classical__enkf.csv", ranked=False,
            scens=superres, text=r"$=$\,EnKF"),
        row("Particle filter", f"{main}/{{s}}__traj{{t}}__classical__particle_filter.csv", ranked=False)]))
    t.caption = (
        r"Navier--Stokes from a poor initial condition: vorticity RMSE, CRPS, "
        r"RMSE of the log kinetic-energy spectrum of the posterior mean, "
        r"and cost across the four observation scenarios (lower is better). Every run starts from "
        r"the five-step history of a different held-out trajectory and assimilates 35 steps; "
        r"metrics are averaged over the 35 steps. $E=64$, $M=100$ sampler steps. " + STD +
        r"the five held-out trajectories. " + BOLD +
        NO(r"The classical filters are not ranked. ") + r"$^{\ast}$: some runs diverged (RMSE above 10 at "
        r"some step); they are excluded from every metric, and the entry is not ranked; "
        r"``div $n/n$'': every run diverged. ``+\,corr.'' applies the forecast-residual correction with an online "
        r"$\hat Q$, 25 forecast steps, localisation radius $r$ and gain 1 (untuned). " + NO(r"The EnKF is "
        r"localised with Gaspari--Cohn $r=20$ on the sparse scenarios and global on "
        r"super-resolution; ``EnKF ($r=20$ everywhere)'' also localises super-resolution. ") +
        r"$^{\dagger}$Cost as recorded in these runs (V100/A100 MIG, partly under GPU contention): "
        r"indicative only and not ranked.")
    # spread--skill is loaded for every cell but shown only in the super-resolution table
    t.all_metrics = t.metrics
    t.metrics = [m for m in t.metrics if m.key != "spread_skill"]
    return t


URB_SCEN = [("sparse 1.5625%", "sparse_1_5625", r"$\tfrac{1}{64}$"),
            ("sparse 0.78125%", "sparse_0_78125", r"$\tfrac{1}{128}$")]


def urban() -> Table:
    u = RES / "urban/metrics"
    fc = RES / "urban_fc/urban/metrics"
    tm = RES / "urban_tmpd/urban/metrics"
    trajs = (1, 2, 3, 4, 5)
    metrics = [Metric("rmse_velocity", "V-RMSE", "d2", exp=-1),
               Metric("rmse_temperature", "T-RMSE", "d2", exp=-1),
               Metric("crps_velocity", "V-CRPS", "d2", exp=-1),
               Metric("crps_temperature", "T-CRPS", "d2", exp=-1)]
    t = Table("", "tab:res_urban", metrics, [(s, h) for s, _, h in URB_SCEN])
    meth = {"si_sde": "Ours (SI-SDE)", "dm_sde": "Ours (DM-SDE)", "fm_ode": "Ours (FM-ODE)"}

    def row(label, pattern, method=None, variant=None, ranked=True):
        per = {}
        for s, slug, _ in URB_SCEN:
            per[s] = _traj_cell([Path(pattern.format(s=slug, t=tr)) for tr in trajs], s, method, variant)
        return _field_row(t, label, per, ranked)

    rs = []
    for lab, a in SAMPLERS:
        rs.append(row(f"Ours ({lab})", f"{u}/{{s}}__M50__traj{{t}}__ours_shared_jac10.csv", meth[a], "shared_jac10"))
        rs.append(row(r"\quad + corr.\ ($r=8$)", f"{fc}/{{s}}__M50__traj{{t}}__ours_shared_jac10__fc25_780902ac__{a}.csv", meth[a]))
    t.groups.append((SHARED_K10, rs))
    t.groups.append((JACFREE, [row(f"Ours ({lab})", f"{u}/{{s}}__M50__traj{{t}}__ours_jacfree.csv", meth[a], "jacfree")
                               for lab, a in SAMPLERS]))
    t.groups.append(("Generative baselines", [
        row("FlowDAS + SURGE", f"{u}/{{s}}__M50__traj{{t}}__baselines.csv", "SURGE (FlowDAS)"),
        row("SDA + SURGE", f"{u}/{{s}}__M50__traj{{t}}__baselines.csv", "SURGE (SDA)"),
        row("D-Flow SGLD", f"{u}/{{s}}__M50__traj{{t}}__dflow.csv", "D-Flow SGLD"),
        row("Guided FM (FIG)", f"{u}/{{s}}__M50__traj{{t}}__baselines.csv", "Guided FM (FIG)"),
        row("TMPD", f"{tm}/{{s}}__M50__traj{{t}}__tmpd.csv", "TMPD")]))
    t.caption = (
        r"Urban (uDALES): rollout-averaged velocity RMSE (V-RMSE), temperature RMSE (T-RMSE), "
        r"velocity CRPS (V-CRPS) and temperature CRPS (T-CRPS) at sensor densities $\tfrac{1}{64}$ "
        r"($1.5625\%$) and $\tfrac{1}{128}$ ($0.78125\%$) (lower is better). Foreign initial "
        r"condition, 145 generated steps, velocity observations every third step (49 updates), "
        r"$E=64$, $M=50$ sampler steps. " + STD + r"the five held-out trajectories"
        + NO(r" (four for FlowDAS + SURGE, SDA + SURGE and Guided FM (FIG) at $\tfrac{1}{64}$, "
             r"where one run has no result)") + r". " + BOLD +
        r"$^{\ast}$: some runs diverged (divergence guard); they are excluded, and the entry is "
        r"not ranked; ``div $n/n$'': every run diverged. ``+\,corr.'' applies the forecast-residual "
        r"correction (online $\hat Q$, 25 forecast steps, radius 8, gain 1; untuned).")
    return t


def ns_superres() -> Table:
    """The NS table restricted to super-resolution, without the KE-spectrum column and
    without the r=8 correction rows (cells are those of ns())."""
    full = ns()
    keep = ("16^2->128^2", "32^2->128^2")
    t = Table("", "tab:res_ns_wrong_ic_superres",
              [m for m in full.all_metrics if m.key != "energy_spec_rmse"],
              [(k, h) for k, h in full.scenarios if k in keep])
    t.groups = [(g, [r for r in rs if "r=8" not in r.label]) for g, rs in full.groups]
    for _, rs in t.groups:   # super-resolution only: "everywhere" is redundant
        for r in rs:
            r.label = r.label.replace(r"($r=20$ everywhere)", r"($r=20$)")
    t.caption = (
        r"Navier--Stokes from a poor initial condition, super-resolution scenarios: vorticity "
        r"RMSE, CRPS and cost (lower is better). Every run starts from the five-step history of "
        r"a different held-out trajectory and assimilates 35 steps; metrics are averaged over "
        r"the 35 steps. $E=64$, $M=100$ sampler steps. " + STD + r"the five held-out "
        r"trajectories. " + BOLD + NO(r"The classical filters are not ranked. ") +
        r"$^{\ast}$: some runs diverged (RMSE above 10 at some step); they are excluded from every "
        r"metric, and the entry is not ranked; ``div $n/n$'': every run diverged. ``+\,corr.'' "
        r"applies the forecast-residual correction with an online $\hat Q$, 25 forecast steps, "
        r"localisation radius $r=20$ and gain 1 (untuned). " + NO(r"``EnKF'' is global (no "
        r"localisation) on super-resolution; ``EnKF ($r=20$ everywhere)'' is localised with "
        r"Gaspari--Cohn $r=20$. ") + r"$^{\dagger}$Cost as recorded in these runs (V100/A100 "
        r"MIG, partly under GPU contention): indicative only and not ranked.")
    return t


# --------------------------------------------------------------------------- #
# Combined: multistep linear-Gaussian + Lorenz 96 (linear, arctan)
# --------------------------------------------------------------------------- #


def combined() -> Table:
    """One table, shared rows; columns: multistep (Gaussian KL, SW2), Lorenz 96 linear
    and arctan (SW2, kNN-KL). Cells are copied from the per-case tables."""
    src = {"ms": multistep(), "lin": lorenz96("linear"), "arc": lorenz96("arctan")}
    src_scen = {"ms": "paper", "lin": "linear", "arc": "arctan"}
    index = {k: {r.rid: r for _, rs in t.groups for r in rs} for k, t in src.items()}
    kl = Metric("kl_points", "Gaussian KL", "sig", exp=-3)
    sw2 = Metric("sliced_w2", "Sliced-$W_2$", "sig", exp=-2)
    sw2_l96 = Metric("sliced_w2", "Sliced-$W_2$", "d2")
    knn = Metric("kl_knn", "$k$NN-KL", "d2")
    cg = [("Multistep linear--Gaussian", [("Gaussian KL", kl, "ms"), ("Sliced-$W_2$", sw2, "ms")]),
          ("Lorenz 96, linear", [("$k$NN-KL", knn, "lin"), ("Sliced-$W_2$", sw2_l96, "lin")]),
          ("Lorenz 96, arctan", [("$k$NN-KL", knn, "arc"), ("Sliced-$W_2$", sw2_l96, "arc")])]
    t = Table("", "tab:res_ms_l96", [kl, sw2, knn], [], colgroups=cg, inline_pm=True)

    def row(label, ms=None, l96=None, ranked=True, l96_lin_only=False):
        r = Row(label, ranked=ranked)
        ids = {"ms": ms, "lin": l96, "arc": None if l96_lin_only else l96}
        for _, subs in cg:
            for _, m, case in subs:
                if ids[case] is None or ids[case] not in index[case]:
                    continue
                c = index[case][ids[case]].cells.get((m.key, src_scen[case]))
                if c is not None:
                    r.cells[(m.key, case)] = c
        return r

    meth = {"si_sde": "Ours (SI-SDE)", "dm_sde": "Ours (DM-SDE)", "fm_ode": "Ours (FM-ODE)"}
    for cov, l96sfx, title in (
            ("shared", "-shared", "Ours -- ensemble-shared inflated covariance"),
            ("jacfree", "", JACFREE)):
        rs = []
        for lab, a in SAMPLERS:
            rs.append(row(f"Ours ({lab})", (meth[a], cov), f"{a}{l96sfx}"))
            rs.append(row(r"\quad + corr.", (meth[a], f"{cov}_fc{MS_M}"), f"{a}_fc{l96sfx}"))
            if a != "fm_ode":
                rs.append(row(r"\quad + SURGE", (meth[a], f"surge_{cov}"),
                              f"surge_{a}" if cov == "jacfree" else None))
        t.groups.append((title, rs))
    t.groups.append(("Generative baselines", [
        row("FlowDAS", ("FlowDAS", ""), "flowdas"),
        row("FlowDAS + SURGE", ("SURGE (FlowDAS)", ""), "surge_flowdas"),
        row("SDA", ("SDA", ""), "sda"), row("SDA + SURGE", ("SURGE (SDA)", ""), "surge_sda"),
        row("D-Flow SGLD", ("D-Flow SGLD", ""), "dflow"),
        row("Guided FM (FIG)", ("Guided FM (FIG)", ""), "fig"),
        row("TMPD", ("TMPD", ""), "tmpd", l96_lin_only=True)]))
    t.groups.append(("Generative prior + EnKF analysis", [
        row("Trained prior (SI) + EnKF", None, "prior_enkf"),
        row("Trained prior (FM) + EnKF", None, "prior_enkf_fm")]))
    t.groups.append(("Classical filters (true model)", [
        row("EnKF", ("EnKF", ""), "enkf", ranked=False),
        row("EnKF ($E=2000$)", None, "enkf_large", ranked=False),
        row("Particle filter", ("Particle filter", ""), "pf_matched", ranked=False)]))
    t.groups.append(("Controls and references", [
        row("Oracle (exact one-step kernel)", ("Exact per-member chain", ""), "oracle", ranked=False),
        row("Oracle + corr.", None, "oracle_fc", ranked=False),
        row("Sampling floor", ("Kalman filter (sampled)", ""), "pf_subsample", ranked=False),
        row("PF reference (2nd replicate)", None, "pf_reference", ranked=False)]))
    t.caption = (
        r"Multistep linear--Gaussian test (paper system $A=Q=\obsmatrix=\obscov=\bI$) and "
        r"Lorenz~96 with linear and arctan observations: distances to the reference posterior, "
        r"averaged over the cycles / assimilation steps (lower is better). Reference: the Kalman "
        r"filtering posterior (multistep; Gaussian KL and sliced $W_2$) and the particle-filter "
        r"reference with $4\cdot10^6$ particles (Lorenz~96; sliced $W_2$ and $k$-nearest-neighbour "
        rf"KL). All generative samplers use $M={MS_M}$ steps, the largest run. Multistep: "
        r"$N=100{,}000$ members, the shared covariance is exact" + NO(r", and the EnKF and particle filter "
        r"use $N$ members") + r"; Lorenz~96: $E=64$, the shared Jacobian is refreshed every step ($k=1$), "
        r"and SURGE was run around the Jacobian-free samplers only. ``+\,corr.'' applies the "
        r"forecast-residual correction with online $\hat Q$ and $S=M$ forecast steps. "
        + NO(r"Oracle: every member sampled from the exact one-step posterior without correction "
        r"(the exact per-member chain of the multistep test). Sampling floor: the Kalman "
        r"posterior sampled with $N$ members (multistep) or 64 draws of the PF reference "
        r"(Lorenz~96). ") + STD + r"five seeds (multistep) or the 12 test sequences (Lorenz~96). "
        + BOLD + NO(r"The classical filters, controls and references are not ranked. ") +
        r"$^{\ast}$: some runs collapsed or diverged; the entry is the mean over the rest and is "
        r"not ranked. ``--'': not run.")
    return t


# --------------------------------------------------------------------------- #


def main() -> None:
    global OURS_ONLY, FLOAT
    OURS_ONLY = "--ours-only" in sys.argv[1:]
    if "--manuscript" in sys.argv[1:]:
        # the combined multistep + Lorenz 96 table as a manuscript float
        FLOAT = True
        t = combined()
        t.label = "tab:ms_l96"
        t.caption = (  # short manuscript caption; setups live in the two appendices
            r"Multistep linear--Gaussian test and Lorenz~96 with linear and \revC{arctan} "
            r"observations: distances to the reference posterior at $M=250$, averaged over cycles "
            r"or assimilation steps (lower is better). Reference: the Kalman filtering posterior "
            r"(multistep) or a particle filter with $4\cdot10^6$ particles (Lorenz~96). Setups, "
            r"oracle and sampling floor: Appendices~\ref{appendix:analytical_setup} "
            r"and~\ref{appendix:lorenz96_setup}. $\pm$: standard deviation over five seeds "
            r"(multistep) or 12 test sequences (Lorenz~96). \textbf{Bold}/\underline{underline}: "
            r"best/second best per column, ties at the printed precision included; classical "
            r"filters, controls and references are not ranked. $^{\ast}$: some runs collapsed or "
            r"diverged; mean over the rest, not ranked. ``--'': not run.")
        out = HERE / "manuscript_tab_ms_l96.tex"
        out.write_text("% generated by make_results_tables.py --manuscript -- do not edit\n" + render(t))
        print("wrote", out)
        # the Navier--Stokes poor-IC super-resolution table
        t = ns_superres()
        t.label, t.rev, t.placement = "tab:ns_wic_superres", "revC", "p"
        t.no_factor_note = True   # as tab:ns_accuracy
        t.inline_pm = True        # enough width: +/- next to the value
        t.caption = (
            r"Navier--Stokes super-resolution from a poor initial condition (35 steps; "
            r"Section~\ref{subsec:ns_wic}): vorticity RMSE, CRPS, spread--skill ($|1-\mathrm{ratio}|$) "
            r"and cost, averaged over the steps "
            r"(lower is better). $\pm$: standard deviation over the five trajectories. "
            r"\textbf{Bold}/\underline{underline}: best/second best per column, ties at the printed "
            r"precision included; classical filters are not ranked. $^{\ast}$: some runs diverged "
            r"(unscaled RMSE above 10); mean over the rest, not ranked. ``div $n/n$'': every run "
            r"diverged. $^{\dagger}$Partly timed under GPU contention (V100/A100): indicative only.")
        out = HERE / "manuscript_tab_ns_wic_superres.tex"
        out.write_text("% generated by make_results_tables.py --manuscript -- do not edit\n" + render(t))
        print("wrote", out)
        return
    out = HERE / ("results_tables_ours_body.tex" if OURS_ONLY else "results_tables_body.tex")
    parts = []
    for title, make in (("Multistep linear--Gaussian", multistep),
                        ("Lorenz 96, linear observations", lambda: lorenz96("linear")),
                        ("Lorenz 96, nonlinear (arctan) observations", lambda: lorenz96("arctan")),
                        ("Navier--Stokes, poor initial condition", ns),
                        ("Navier--Stokes, poor initial condition: super-resolution", ns_superres),
                        ("Urban airflow (uDALES)", urban),
                        ("Combined: multistep linear--Gaussian and Lorenz 96", combined)):
        tab = make()
        tab.short = title
        parts += [render(tab), r"\clearpage"]
    out.write_text("% generated by make_results_tables.py -- do not edit\n" + "\n".join(parts))
    print("wrote", out)


if __name__ == "__main__":
    main()
