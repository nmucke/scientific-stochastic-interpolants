"""Overnight |x| sweep (trajectory 0, E=64): time-averaged metrics of every cell vs the EnKF.

Reads lorenz96/abs/per_sequence/lorenz96__<label>__traj0__.json (one value per sequence = the
time average), writes summary.md / summary.csv next to this file. Positive % = better than EnKF.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PS = HERE.parent / "lorenz96/abs/per_sequence"
METRICS = (("sliced_w2", "Sliced W2"), ("kl_knn", "kNN-KL"), ("energy_distance", "Energy dist."))
SAMPLERS = (("si_sde", "SI-SDE"), ("dm_sde", "DM-SDE"), ("fm_ode", "FM-ODE"))
VARIANTS = (("", "", "Jac-free"), ("_fc", "", "Jac-free + corr."),
            ("", "-shared", "shared"), ("_fc", "-shared", "shared + corr."))


def load(label):
    p = PS / f"lorenz96__{label}__traj0__.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {k: d[k][0] for k, _ in METRICS}


enkf = load("enkf")
rows = []
for a, an in SAMPLERS:
    for fc, tag, vn in VARIANTS:
        for M in (25, 50, 100, 250):
            v = load(f"{a}{fc}{tag}__M{M}")
            rows.append((an, vn, M, v))

head = "| Sampler | Variant | M | " + " | ".join(n for _, n in METRICS) + " |"
lines = ["# |x| case, trajectory 0, E=64: time-averaged metrics (vs EnKF, + = better)", "",
         "| EnKF | – | – | " + " | ".join(f"{enkf[k]:.3g}" for k, _ in METRICS) + " |", "",
         head, "|" + "---|" * (3 + len(METRICS))]
csv = ["sampler,variant,M," + ",".join(k for k, _ in METRICS) + "," + ",".join(f"{k}_pct" for k, _ in METRICS)]
for an, vn, M, v in rows:
    if v is None:
        lines.append(f"| {an} | {vn} | {M} | " + " | ".join("missing" for _ in METRICS) + " |")
        continue
    pct = {k: (enkf[k] - v[k]) / enkf[k] * 100 for k, _ in METRICS}
    lines.append(f"| {an} | {vn} | {M} | " + " | ".join(f"{v[k]:.3g} ({pct[k]:+.0f}%)" for k, _ in METRICS) + " |")
    csv.append(f"{an},{vn},{M}," + ",".join(f"{v[k]:.6g}" for k, _ in METRICS) + "," + ",".join(f"{pct[k]:.2f}" for k, _ in METRICS))
(HERE / "summary.md").write_text("\n".join(lines) + "\n")
(HERE / "summary.csv").write_text("\n".join(csv) + "\n")
print("\n".join(lines))
