"""|x| sweep, 5 test sequences, E=64: time-averaged metrics of every cell (mean ± sd over
sequences) and % vs the EnKF (+ = better). Writes summary.md / summary.csv next to this file."""
import json, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent
PS = HERE.parent / "lorenz96/abs/per_sequence"
MET = (("sliced_w2", "Sliced W2"), ("kl_knn", "kNN-KL"), ("energy_distance", "Energy dist."))
def load(label):
    p = PS / f"lorenz96__{label}__traj0__.json"
    if not p.exists(): return None
    d = json.loads(p.read_text())
    return {k: (st.mean(d[k]), st.stdev(d[k]) if len(d[k]) > 1 else 0.0, len(d[k])) for k, _ in MET}
ROWS = []
for a, an in (("si_sde", "SI-SDE"), ("dm_sde", "DM-SDE"), ("fm_ode", "FM-ODE")):
    for fc, tag, vn in (("_fc", "-shared", "shared + corr."), ("", "-shared", "shared"),
                        ("_fc", "", "Jac-free + corr."), ("", "", "Jac-free")):
        ROWS.append((f"Ours {an}", vn, f"{a}{fc}{tag}", True))
for c, n in (("surge_si_sde", "Ours SI-SDE + SURGE"), ("surge_dm_sde", "Ours DM-SDE + SURGE"),
             ("flowdas", "FlowDAS"), ("surge_flowdas", "FlowDAS + SURGE"), ("sda", "SDA"),
             ("surge_sda", "SDA + SURGE"), ("dflow", "D-Flow SGLD"), ("fig", "Guided FM (FIG)"),
             ("prior_enkf", "SI prior + EnKF"), ("prior_enkf_fm", "FM prior + EnKF")):
    ROWS.append((n, "", c, True))
for c, n in (("enkf", "EnKF (E=64)"), ("enkf_large", "EnKF (E=2000)"), ("pf_matched", "PF (E=64)"),
             ("pf_subsample", "Sampling floor (PF, 64 draws)"), ("pf_reference", "PF reference (2nd replicate)")):
    ROWS.append((n, "", c, False))
enkf = load("enkf")
lines = ["# |x| case, 5 test sequences, E=64: time-averaged metrics, mean ± sd over sequences (% vs EnKF, + = better)", "",
         "| Method | Variant | M | n | " + " | ".join(n for _, n in MET) + " |", "|" + "---|" * (4 + len(MET))]
csv = ["method,variant,M,n," + ",".join(f"{k},{k}_sd,{k}_pct" for k, _ in MET)]
for name, var, cell, gen in ROWS:
    for M in ((25, 50, 100, 250) if gen else (None,)):
        v = load(f"{cell}__M{M}" if gen else cell)
        if v is None:
            lines.append(f"| {name} | {var} | {M or '–'} | – | " + " | ".join("missing" for _ in MET) + " |"); continue
        pct = {k: (enkf[k][0] - v[k][0]) / enkf[k][0] * 100 if enkf else float("nan") for k, _ in MET}
        n = v[MET[0][0]][2]
        lines.append(f"| {name} | {var} | {M or '–'} | {n} | " + " | ".join(
            f"{v[k][0]:.3g} ± {v[k][1]:.2g} ({pct[k]:+.0f}%)" for k, _ in MET) + " |")
        csv.append(f"{name},{var},{M or ''},{n}," + ",".join(f"{v[k][0]:.6g},{v[k][1]:.6g},{pct[k]:.2f}" for k, _ in MET))
(HERE / "summary.md").write_text("\n".join(lines) + "\n")
(HERE / "summary.csv").write_text("\n".join(csv) + "\n")
print("\n".join(lines))
