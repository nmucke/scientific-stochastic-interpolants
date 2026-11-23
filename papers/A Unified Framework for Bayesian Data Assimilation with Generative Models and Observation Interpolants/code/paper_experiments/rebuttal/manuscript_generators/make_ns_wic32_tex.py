#!/usr/bin/env python3
"""LaTeX for the appendix field figures, one per scenario (panels from make_ns_wic_fields.py):
poor IC (stem ns_wic_<slug>, slugs SCENARIOS) and true IC (stem ns_tic_<slug>, slugs
TIC_SCENARIOS). Writes <stem>_figure.tex with the panel directory as \\nswicdir; the caption's
assimilation step and divergence note are computed from the data (make_ns_wic_fields;
pending true-IC methods get no note). Run after make_ns_wic_fields.py:
    python3 make_ns_wic_fields.py && python3 make_ns_wic32_tex.py
Usage: make_ns_wic32_tex.py [panel width] [slug | tic:<slug> ...]"""
from pathlib import Path

from make_ns_wic_fields import SCENARIOS, TIC_SCENARIOS, TIC_TRAJS, TRAJS, divergence, n_assim_steps, tic_divergence, tic_m_note

W = r"0.135\linewidth"
two = lambda a, b: rf"\begin{{tabular}}[b]{{@{{}}c@{{}}}}\textbf{{{a}}}\\[-1pt]\textbf{{{b}}}\end{{tabular}}"
one = lambda a: rf"\textbf{{{a}}}"
BLOCKS = [
    (r"Ours, shared covariance ($k=10$)",
     [(f"{a}_shared{c}", two(lab, "+ corr.") if c else two(lab, ""))
      for a, lab in (("si_sde", "SI-SDE"), ("dm_sde", "DM-SDE"), ("fm_ode", "FM-ODE")) for c in ("", "_corr")]),
    (r"Ours, Jacobian-free covariance",
     [(f"{a}_jac_free{c}", two(lab, "+ corr.") if c else two(lab, ""))
      for a, lab in (("si_sde", "SI-SDE"), ("dm_sde", "DM-SDE"), ("fm_ode", "FM-ODE")) for c in ("", "_corr")]),
    (r"Baselines",
     [("tmpd", two("TMPD", "")), ("flowdas_surge", two("FlowDAS", "+ SURGE")), ("sda_surge", two("SDA", "+ SURGE")),
      ("d_flow_sgld", two("D-Flow", "SGLD")), ("guided_fm_fig", two("Guided FM", "(FIG)")),
      ("fm_ode_prior_enkf", two("FM-ODE prior", "+ EnKF"))]),
]
NAMES = {f"{a}_{tag}{c}": f"{lab} ({t}){' + corr.' if c else ''}"
         for a, lab in (("si_sde", "SI-SDE"), ("dm_sde", "DM-SDE"), ("fm_ode", "FM-ODE"))
         for tag, t in (("shared", "shared"), ("jac_free", "Jacobian-free")) for c in ("", "_corr")}
NAMES.update(tmpd="TMPD", flowdas_surge="FlowDAS + SURGE", sda_surge="SDA + SURGE", d_flow_sgld="D-Flow SGLD",
             guided_fm_fig="Guided FM (FIG)", fm_ode_prior_enkf="FM-ODE prior + EnKF")
# slug -> (scenario text, observation text, label); poor-IC labels get the _wic infix (WIC_LABEL)
SCEN = {
    "32_2_128_2": (r"$32^2\!\to\!128^2$", r"one dot per observed $4\times4$ block average", "fig:ns_fields_32"),
    "16_2_128_2": (r"$16^2\!\to\!128^2$", r"one dot per observed $8\times8$ block average", "fig:ns_fields_16"),
    "sparse_5": (r"sparse $5\%$", "one dot per observed grid point", "fig:ns_fields_sparse5"),
    "sparse_1_5625": (r"sparse $1.5625\%$", "one dot per observed grid point", "fig:ns_fields_sparse1p5"),
}


def tex_name(slug):
    n = NAMES[slug]
    return n[:-1] + r".\ " if n.endswith(".") else n + " "


WIC_LABEL = {"32_2_128_2": "fig:ns_fields_wic_32", "16_2_128_2": "fig:ns_fields_wic_16"}
IC_TEXT = {"wic": "poor IC", "tic": "true IC"}  # author's wording (Overleaf, 2026-11-16)


def divergence_note(sc, ic="wic"):
    """'X diverged on this trajectory (k of n).' for every method diverged on the shown trajectory."""
    dv, n = (divergence(sc), len(TRAJS)) if ic == "wic" else (tic_divergence(sc), len(TIC_TRAJS))
    d = [(s, k) for s, (shown, k) in dv.items() if shown]
    seed = ""
    if ic == "wic":
        from make_ns_wic_fields import WIC_SEED_OVERRIDE
        for (osc, s), (sd, _, _) in WIC_SEED_OVERRIDE.items():
            if osc == sc:
                seed = (rf" {tex_name(s).strip()} is shown for a rerun with another seed, "
                        rf"as the original run diverged on this trajectory.")
    if not d:
        return seed.strip()
    if len(d) == 1:
        s, k = d[0]
        return rf"{tex_name(s)}diverged on this trajectory." + seed
    items = [tex_name(s).strip() for s, k in d]
    return (", ".join(items[:-1]) + " and " + items[-1]) + " diverged on this trajectory." + seed


# panel width per figure where the default 0.135 does not fit: the true-IC 16^2 caption has one
# extra line (the M note), so its panels are ~3% narrower (same layout otherwise)
W_OVERRIDE = {}

ROWS = (("mean", "Mean", "field"), ("std", "Std.", "std"), ("abserr", "$|$Error$|$", "abserr"))


def figure(sc="32_2_128_2", W=r"0.135\linewidth", WT=r"0.155\linewidth", gap="-3.8pt", bgap="1.5pt", sep="0.35pt",
           ic="wic"):
    """Truth and observations centred on top; three blocks (mean / std / |error| rows) with the
    block title as a vertical label on the left; fixed panel width, tight spacing.
    ic: 'wic' (poor IC, stem ns_wic_<sc>) or 'tic' (true IC, stem ns_tic_<sc>)."""
    stem = f"ns_{ic}_{sc}"
    scen, obs_txt, label = SCEN[sc]
    if ic == "wic":
        label = WIC_LABEL[sc]
    step = rf"the last of the {n_assim_steps(ic, sc)} assimilation steps"
    L = [r"\begin{figure*}[p]", r"    \centering", rf"    \setlength{{\tabcolsep}}{{{sep}}}",
         r"    \renewcommand{\arraystretch}{0.6}",
         r"    \begin{tabular}{@{}c@{\hspace{8pt}}c@{}}",
         rf"        \includegraphics[width={WT}]{{\nswicdir {stem}__truth.pdf}} &"
         rf" \includegraphics[width={WT}]{{\nswicdir {stem}__obs.pdf}} \\",
         r"        {\scriptsize\textbf{True state}} & {\scriptsize\textbf{Observations}}",
         r"    \end{tabular}\par\vspace{4pt}",
         r"    \begin{tabular}{@{}c@{\hspace{7pt}}cccccccc@{}}"]
    for bi, (title, ms) in enumerate(BLOCKS):
        L.append("        & & " + " & ".join(rf"{{\scriptsize {t}}}" for _, t in ms) + r" & \\[1pt]")
        for ri, (q, name, cb) in enumerate(ROWS):
            cb = cb if cb == "field" else f"{cb}_g{0 if bi < 2 else 1}"   # ours blocks share a scale
            cells = " & ".join(rf"\includegraphics[width={W}]{{\nswicdir {stem}__{q}__{s}.pdf}}" for s, _ in ms)
            last = ri == len(ROWS) - 1
            end = (rf" \\[{bgap}]" if bi < len(BLOCKS) - 1 else r" \\") if last else rf" \\[{gap}]"
            # block title: one vertical label centred on the middle row, spanning the block
            half = W.replace(r"\linewidth", "")
            btitle = (rf"\raisebox{{{float(half) / 2}\linewidth}}[0pt][0pt]{{\rotatebox{{90}}{{\makebox[0pt][c]{{\scriptsize\textit{{{title}}}}}}}}}"
                      if ri == 1 else "")
            L.append(rf"        {btitle} & \rotatebox{{90}}{{\makebox[{W}][c]{{\scriptsize\textbf{{{name}}}}}}} & {cells} & "
                     rf"\includegraphics[height={W}]{{\nswicdir {stem}__cbar_{cb}.pdf}}{end}")
    L += [r"    \end{tabular}", r"    \vspace{-4pt}",
          r"    \caption{\revC{Navier--Stokes, " + IC_TEXT[ic] + ", " + scen + ": posterior mean, "
          r"ensemble standard deviation and absolute error at the final assimilation step of one held-out trajectory." + (" " if divergence_note(sc, ic) else "") + divergence_note(sc, ic) + "}}",
          rf"    \label{{{label}}}", r"\end{figure*}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    W = args.pop(0) if args and "linewidth" in args[0] else r"0.135\linewidth"
    jobs = ([("tic", a[4:]) if a.startswith("tic:") else ("wic", a) for a in args]
            or [("wic", sc) for sc in SCENARIOS] + [("tic", sc) for sc in TIC_SCENARIOS])
    for ic, sc in jobs:
        out = Path(__file__).resolve().parent / f"ns_{ic}_{sc}_figure.tex"
        out.write_text("% generated by make_ns_wic32_tex.py -- do not edit\n" + figure(sc, W=W_OVERRIDE.get((ic, sc), W), ic=ic))
        print("wrote", out)
