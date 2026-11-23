#!/usr/bin/env python3
"""New Table 1 (tab:ns_accuracy) and the Navier--Stokes timing table (tab:ns_timing).

Table 1: vorticity RMSE and CRPS (x10^-1, two decimals, +/- next to the entry) for
- the four truth-initialised scenarios: results/navier_stokes/aggregated/all.csv (M=100),
  checked against the published three-decimal values of the old hand-typed table
  (ns_table1_source_results.tex, a copy of sections_review/results.tex before the swap);
- the two poor-initial-condition super-resolution scenarios: make_results_tables.ns()
  (same per-trajectory loader as the appendix table; divergence excluded and marked).
Rows: the old lineup, plus "+ corr. (r=20)" under each of our samplers (run only from the
poor IC), TMPD and the SI/FM prior + EnKF (run only from the poor IC). Missing cells: TBD.

Timing table: cost per assimilation step for the same rows (truth-initialised: the old
Table 1 costs, RTX 3090; poor IC: as recorded, partly under GPU contention).

    python3 make_ns_table1.py ns_table1_source_results.tex

Writes ns_table1.tex (Table 1: super-resolution, true and poor IC, with cost) and
ns_table_sparse.tex (appendix: sparse sensors, true IC, with cost) next to this script.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mrt", HERE / "make_results_tables.py")
mrt = importlib.util.module_from_spec(spec)
sys.modules["mrt"] = mrt
spec.loader.exec_module(mrt)

TRUE_AGG = Path("/export/scratch2/ntm/delftblue_results/results/navier_stokes/aggregated/all.csv")
TRUE_SCEN = ("16^2->128^2", "32^2->128^2", "sparse 5%", "sparse 1.5625%")
TRUE_HDR = (r"$16^2$", r"$32^2$", r"$5\%$", r"$\tfrac{1}{64}$")
POOR_SC = (("16^2->128^2", r"$16^2$"), ("32^2->128^2", r"$32^2$"))
SHARED, JACFREE = "shared", "jacfree"
SAMPLERS = ("SI-SDE", "DM-SDE", "FM-ODE")
CORR = r"\quad + corr.\ ($r=20$)"

# rows: (group title or None, [(label, key)]); key identifies the method in both data sources
GROUPS = [
    (r"Ours -- shared covariance (Jacobian refreshed every $k=10$ steps)",
     [x for s in SAMPLERS for x in ((f"Ours ({s})", (SHARED, s)), (CORR, (SHARED, s, "corr")))]),
    (r"Ours -- Jacobian-free covariance",
     [x for s in SAMPLERS for x in ((f"Ours ({s})", (JACFREE, s)), (CORR, (JACFREE, s, "corr")))]),
    (None, [(lab, (lab,)) for lab in ("FlowDAS + SURGE", "SDA + SURGE", "D-Flow SGLD",
                                      "Guided FM (FIG)", "TMPD")]),
    (None, [(lab, (lab,)) for lab in ("SI-SDE prior + EnKF", "FM-ODE prior + EnKF")]),
    (None, [("EnKF", ("EnKF",)), ("Particle filter", ("Particle filter",))]),
    (None, [("EnKF reference ($E=1000$)", ("ref",))]),
]
NEW = {"TMPD", "SI-SDE prior + EnKF", "FM-ODE prior + EnKF", CORR}
UNRANKED = {("ref",), ("EnKF",), ("Particle filter",)}   # classical filters not ranked
# truth-initialised aggregate: key -> (method, variant)
TRUE_SRC = {**{(SHARED, s): (f"Ours ({s})", "shared_jac10") for s in SAMPLERS},
            **{(JACFREE, s): (f"Ours ({s})", "jacfree") for s in SAMPLERS},
            ("FlowDAS + SURGE",): ("SURGE (FlowDAS)", ""), ("SDA + SURGE",): ("SURGE (SDA)", ""),
            ("D-Flow SGLD",): ("D-Flow SGLD", ""), ("Guided FM (FIG)",): ("Guided FM (FIG)", ""),
            ("EnKF",): ("EnKF", ""), ("Particle filter",): ("Particle filter", ""),
            ("ref",): ("EnKF (reference)", "")}
# ---- truth-initialised runs added for the rebuttal (cluster, V100/A100/MIG): per-trajectory files,
# loaded like the poor IC (mrt._traj_cell: divergence excluded and marked). Cost: shown with a dagger,
# never ranked. TRUE_CORR_TAG selects the "+ corr." runs (fc25_70a3134d: r=8, fc25_8797d2ae: r=20);
# None leaves the true-IC "+ corr." cells TBD. Jacobian-free + corr. (r=20 only) arrived 2026-11-18.
TIC_ROOT = Path("/export/scratch2/ntm/delftblue_results/results/navier_stokes_true_ic")
TRUE_CORR_TAG = "fc25_8797d2ae"   # r=20, as the poor IC
TRUE_SLUG = {"16^2->128^2": "16_2_128_2", "32^2->128^2": "32_2_128_2",
             "sparse 5%": "sparse_5", "sparse 1.5625%": "sparse_1_5625"}
TRUE_TRAJS = (10, 11, 12, 13, 14)
_A = {"SI-SDE": "si_sde", "DM-SDE": "dm_sde", "FM-ODE": "fm_ode"}


def true_new_stem(k):
    """metrics-file stem ({s}=scenario slug, {t}=trajectory) of a rebuttal true-IC run, or None."""
    if k == ("TMPD",):
        return f"{TIC_ROOT}/tmpd/navier_stokes/metrics/{{s}}__M100__traj{{t}}__tmpd.csv"
    if k in (("SI-SDE prior + EnKF",), ("FM-ODE prior + EnKF",)):
        a = _A[k[0].split()[0]]
        return f"{TIC_ROOT}/prior_enkf/navier_stokes/metrics/{{s}}__M100__traj{{t}}__prior_enkf_r20__{a}.csv"
    if len(k) == 3 and k[2] == "corr" and TRUE_CORR_TAG:
        cov = "shared_jac10" if k[0] == SHARED else "jacfree"
        return (f"{TIC_ROOT}/ext/navier_stokes/metrics/{{s}}__M100__traj{{t}}__ours_{cov}__"
                f"{TRUE_CORR_TAG}__{_A[k[1]]}.csv")
    return None


_TNEW = {}


def true_new_cells(k, scen):
    """{metric: Cell} over the true-IC trajectories, or None (not run)."""
    if (k, scen) not in _TNEW:
        stem = true_new_stem(k)
        _TNEW[(k, scen)] = None if stem is None else mrt._traj_cell(
            [Path(stem.format(s=TRUE_SLUG[scen], t=t)) for t in TRUE_TRAJS], scen)
    return _TNEW[(k, scen)]


# old table label -> key (for the cross-check and the costs)
OLD_LABEL = {**{(SHARED, f"Ours ({s})"): (SHARED, s) for s in SAMPLERS},
             **{(JACFREE, f"Ours ({s})"): (JACFREE, s) for s in SAMPLERS},
             **{(None, lab): k for k, (lab_, _) in [] for lab in []}}


def parse_old_table(results_tex: Path) -> dict:
    """{key: {"rmse": [(value, std)]*4, "crps": [...], "cost": [str]*4}} from the old tab:ns_accuracy."""
    s = results_tex.read_text()
    j = s.index(r"\label{tab:ns_accuracy}")
    i = s.rfind(r"\begin{table*}", 0, j)
    k = s.index(r"\end{table*}", j)
    out, block = {}, SHARED
    num = re.compile(r"(?:\\textbf\{|\\underline\{)?([0-9.]+)\}?(?:\{\\scriptsize\\,\$\\pm\$([0-9.]+)\})?")
    names = {"FlowDAS + SURGE": ("FlowDAS + SURGE",), "SDA + SURGE": ("SDA + SURGE",),
             "D-Flow SGLD": ("D-Flow SGLD",), "Guided FM (FIG)": ("Guided FM (FIG)",),
             "EnKF": ("EnKF",), "Particle filter": ("Particle filter",),
             "EnKF reference ($E=1000$)": ("ref",)}
    for line in s[i:k].splitlines():
        line = line.strip()
        if line.startswith("%"):
            continue
        if "Jacobian-free covariance" in line:
            block = JACFREE
        if "&" not in line or line.startswith("&") or line.startswith(r"\multicolumn"):
            continue
        cells = [c.strip().rstrip("\\").strip() for c in line.split("&")]
        label, vals = cells[0], cells[1:]
        if len(vals) != 12:
            continue
        parsed = []
        for v in vals:
            m = num.fullmatch(v.replace(" ", ""))
            assert m, (label, v)
            parsed.append((float(m.group(1)), m.group(1)))
        key = (block, label.split("(")[1].rstrip(")")) if label.startswith("Ours") else names[label]
        out[key] = {"rmse": parsed[0:4], "crps": parsed[4:8], "cost": [p[1] for p in parsed[8:12]]}
    return out


def true_values(old: dict) -> dict:
    """{(key, metric, scen): (mean, std)} from the aggregate, cross-checked with the old table."""
    d = pd.read_csv(TRUE_AGG)
    d["variant"] = d["variant"].fillna("")
    out = {}
    for key, (method, variant) in TRUE_SRC.items():
        for metric in ("rmse", "crps"):
            for c, scen in enumerate(TRUE_SCEN):
                q = d[(d.method == method) & (d.variant == variant) & (d.metric == metric) & (d.scenario == scen)]
                if method.startswith("Ours") or method in ("SURGE (FlowDAS)", "SURGE (SDA)", "D-Flow SGLD",
                                                          "Guided FM (FIG)"):
                    q = q[q.M == 100]
                assert len(q) == 1, (key, metric, scen, len(q))
                m, sd = float(q.value.iloc[0]), float(q["std"].iloc[0])
                printed = old[key][metric][c][0]
                assert abs(round(m * 10, 3) - printed) < 1.1e-3, (key, metric, scen, m, printed)
                out[(key, metric, scen)] = (m, sd)
    return out


def poor_cells() -> dict:
    """{key: {(metric, scen): Cell}} from make_results_tables.ns()."""
    t = mrt.ns()
    out, sampler = {}, None
    for gt, rs in t.groups:
        block = SHARED if "shared" in gt else JACFREE if "Jacobian-free" in gt else None
        for r in rs:
            lab = r.label
            if lab.startswith("Ours ("):
                sampler = lab[6:-1]
                out[(block, sampler)] = r.cells
            elif "r=20" in lab and block:
                out[(block, sampler, "corr")] = r.cells
            elif lab.startswith(r"\quad"):
                continue
            elif "prior + EnKF" in lab or lab in ("FlowDAS + SURGE", "SDA + SURGE", "D-Flow SGLD",
                                                  "Guided FM (FIG)", "TMPD", "EnKF", "Particle filter"):
                out[(lab,)] = r.cells
    return out


TBD = r"\tbd"
PM = r"{{\scriptsize\,$\pm${}}}"


def build(results_tex: Path):
    old = parse_old_table(results_tex)
    tv = true_values(old)
    poor = poor_cells()
    rows = [(g, rs) for g, rs in GROUPS]
    keys = [k for _, rs in rows for _, k in rs]

    def value(k, metric, kind, scen):
        """(rank value or None, text, sd text or None) or None (= TBD)."""
        if kind == "true" and true_new_stem(k) is not None:
            cells = {(m, scen): c for m, c in (true_new_cells(k, scen) or {}).items()}
        elif kind == "true":
            if metric == "seconds":
                if k not in old:
                    return None
                txt = old[k]["cost"][TRUE_SCEN.index(scen)]
                return (float(txt), txt, None)
            v = tv.get((k, metric, scen))
            return None if v is None else (round(v[0] * 10, 2), f"{v[0] * 10:.2f}", f"{v[1] * 10:.2f}")
        else:
            cells = poor.get(k) or {}
        r = cells.get(("rmse", scen))
        if r is not None and r.n_exp and r.n_div == r.n_exp:
            return (None, rf"\textcolor{{gray}}{{div\,{r.n_div}/{r.n_exp}}}", None)
        c = cells.get((metric, scen))
        if c is None or not math.isfinite(c.mean):
            return None
        if metric == "seconds":   # recorded on the cluster / under contention: shown, never ranked
            return (None, f"{c.mean:.1f}" + (r"$^{\dagger}$" if kind == "true" else ""), None)
        txt = f"{c.mean * 10:.2f}" + (r"$^{\ast}$" if c.n_div else "")
        sd = f"{c.std * 10:.2f}" if c.n >= 2 and math.isfinite(c.std) else None
        return (None if c.n_div else round(c.mean * 10, 2), txt, sd)

    def label(lab):
        return rf"\revA{{{lab}}}" if lab == CORR else rf"\revB{{{lab}}}" if lab in NEW else lab

    def table(cols, metric_hdrs, ic_row, caption, tlabel, env="table*"):
        """cols: [(metric, kind, scen, header)], grouped by metric in order."""
        grid = {(k, i): value(k, m, kind, sc) for k in keys for i, (m, kind, sc, _) in enumerate(cols)}
        marks = {}
        for i in range(len(cols)):
            vals = sorted({grid[(k, i)][0] for k in keys
                           if grid[(k, i)] and grid[(k, i)][0] is not None and k not in UNRANKED})
            for k in keys:
                g = grid[(k, i)]
                if not g or g[0] is None or k in UNRANKED:
                    continue
                if g[0] == vals[0]:
                    marks[(k, i)] = "textbf"
                elif len(vals) > 1 and g[0] == vals[1]:
                    marks[(k, i)] = "underline"

        def cell(k, i):
            g = grid[(k, i)]
            if g is None:
                return TBD
            _, txt, sd = g
            if (k, i) in marks:
                txt = rf"\{marks[(k, i)]}{{{txt}}}"
            return txt + (PM.format(sd) if sd else "")

        groups = []                    # (metric, n columns)
        for m, *_ in cols:
            if groups and groups[-1][0] == m:
                groups[-1][1] += 1
            else:
                groups.append([m, 1])
        spec = "l|" + "|".join("c" * n for _, n in groups)
        ncol = 1 + len(cols)
        h1 = []
        for gi, (m, n) in enumerate(groups):
            bar = "|" if gi < len(groups) - 1 else ""
            h1.append(rf"\multicolumn{{{n}}}{{c{bar}}}{{\textbf{{{metric_hdrs[m]}}}}}")
        head = [r"\toprule", "& " + " & ".join(h1) + r" \\"]
        if ic_row:
            head.append(ic_row)
        head.append("& " + " & ".join(h for *_, h in cols) + r" \\")
        body = []
        for gt, rs in rows:
            body.append(r"\midrule")
            if gt:
                body.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{{gt}}}}} \\")
            for lab, k in rs:
                body.append(label(lab) + " & " + " & ".join(cell(k, i) for i in range(len(cols))) + r" \\")
        return "\n".join([
            r"% generated by make_ns_table1.py -- do not edit",
            rf"\begin{{{env}}}[ht]", r"    \centering", r"    \setlength{\tabcolsep}{3.5pt}",
            rf"    \caption{{{caption}}}", rf"    \label{{{tlabel}}}", r"    \vspace{-10pt}",
            r"    \resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
            rf"    \begin{{tabular}}{{{spec}}}", *head, *body, r"\bottomrule",
            r"    \end{tabular}}", r"    \vspace{-4pt}" if env == "table*" else "", rf"\end{{{env}}}", ""])

    HDR = {"rmse": r"Vorticity RMSE ($\times 10^{-1}$)\,$\downarrow$",
           "crps": r"Vorticity CRPS ($\times 10^{-1}$)\,$\downarrow$", "seconds": r"Cost (s/step)\,$\downarrow$"}
    # ---- Table 1: super-resolution, true and poor IC
    sup = [("true", "16^2->128^2", r"$16^2$"), ("true", "32^2->128^2", r"$32^2$"),
           ("poor", "16^2->128^2", r"\revC{$16^2$}"), ("poor", "32^2->128^2", r"\revC{$32^2$}")]
    cols1 = [(m, kind, sc, h) for m in ("rmse", "crps", "seconds") for kind, sc, h in sup]
    ic1 = ("& " + " & ".join([r"\multicolumn{2}{c}{true IC} & \multicolumn{2}{c|}{\revC{poor IC}}"] * 2)
           + r" & \multicolumn{2}{c}{true IC} & \multicolumn{2}{c}{\revC{poor IC$^{\dagger}$}} \\")
    cap1 = "Navier--Stokes super-resolution from the true IC (15 steps) \\revC{and a poor IC (35 steps)}: vorticity RMSE, CRPS, and cost per assimilation step, with $E=64$ and $M=100$. Mean $\\pm$ standard deviation over the held-out trajectories. \\textbf{Bold}/\\underline{underline}: best/second best per column. Classical filters are not ranked. \\revC{$^{\\ast}$: some trajectories diverged and are excluded. ``div $n/n$'': all diverged. \\tbd: not run. $^{\\dagger}$: timed on V100/A100 GPUs, not ranked.}"   # = manuscript (Overleaf) wording
    t1 = table(cols1, HDR, ic1, cap1, "tab:ns_accuracy")
    # ---- appendix: sparse sensors, true IC
    sp = [("true", "sparse 5%", r"$5\%$"), ("true", "sparse 1.5625%", r"$\tfrac{1}{64}$")]
    cols2 = [(m, kind, sc, h) for m in ("rmse", "crps", "seconds") for kind, sc, h in sp]
    cap2 = 'Navier--Stokes, sparse sensors ($5\\%$ and $\\tfrac{1}{64}$ of the grid), true IC: as Table~\\ref{tab:ns_accuracy}.'   # = manuscript (Overleaf) wording
    t2 = table(cols2, HDR, None, cap2, "tab:ns_accuracy_sparse", env="table")
    (HERE / "ns_table1.tex").write_text(t1)
    (HERE / "ns_table_sparse.tex").write_text(t2)
    print("wrote", HERE / "ns_table1.tex", HERE / "ns_table_sparse.tex")


if __name__ == "__main__":
    build(Path(sys.argv[1]))
