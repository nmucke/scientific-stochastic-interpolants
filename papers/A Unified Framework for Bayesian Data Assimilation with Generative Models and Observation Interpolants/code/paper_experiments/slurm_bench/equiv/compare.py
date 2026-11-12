"""Compare old_run1 vs old_run2 (noise floor) and old_run1 vs new_run1 (code change)."""
from pathlib import Path

import numpy as np
import pandas as pd

E = Path("/scratch/ntmucke/scientific-stochastic-interpolants/slurm_bench/equiv")
RUNS = {"old1": E / "old_run1", "old2": E / "old_run2", "new": E / "new_run1"}
KEY = ["method", "scenario", "variant", "metric", "step"]


# Sections and the hardware they ran on (old/new always on the same hardware per section).
SECTIONS = {"jf_mig": "MIG", "shared_k10": "MIG", "classical": "MIG", "classical_loc": "MIG",
            "jf_sda": "V100", "dflow": "V100"}  # jacfree_base outputs are partial: ignored


def per_step(run: Path) -> pd.DataFrame:
    frames = []
    for sec, hw in SECTIONS.items():
        p = run / f"ps_{sec}.csv"
        if p.exists():
            f = pd.read_csv(p); f["section"] = f"{sec} ({hw})"; frames.append(f)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=KEY + ["value", "section"])
    df["variant"] = df["variant"].fillna("")
    return df[df.metric.isin(["rmse", "crps", "spread_skill", "energy_spec_rmse"])]


def states(run: Path) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for sec in SECTIONS:
        for p in sorted((run / f"states_{sec}").glob("**/*.npz")):
            with np.load(p, allow_pickle=False) as d:
                out[f"{sec}/{p.name}"] = {k: d[k] for k in d.files if d[k].dtype.kind == "f" and d[k].size > 1}
    return out


ps = {k: per_step(v) for k, v in RUNS.items()}
st = {k: states(v) for k, v in RUNS.items()}


def cmp_ps(a: str, b: str) -> pd.DataFrame:
    m = ps[a].merge(ps[b], on=KEY + ["section"], suffixes=("_a", "_b"))
    m["absdiff"] = (m.value_a - m.value_b).abs()
    m["reldiff"] = m.absdiff / m.value_a.abs().clip(lower=1e-12)
    g = m.groupby(["section", "method", "variant", "scenario"]).agg(
        n=("absdiff", "size"), max_abs=("absdiff", "max"), max_rel=("reldiff", "max"),
        nan_a=("value_a", lambda s: int(s.isna().sum())), nan_b=("value_b", lambda s: int(s.isna().sum())))
    return g


def cmp_states(a: str, b: str) -> pd.DataFrame:
    rows = []
    for name in sorted(set(st[a]) & set(st[b])):
        for k in sorted(set(st[a][name]) & set(st[b][name])):
            x, y = st[a][name][k], st[b][name][k]
            if x.shape != y.shape:
                rows.append(dict(file=name, key=k, max_abs=np.inf, identical=False)); continue
            d = np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64))) if x.size else 0.0
            rows.append(dict(file=name, key=k, max_abs=float(d), identical=bool(np.array_equal(x, y, equal_nan=True))))
    return pd.DataFrame(rows)


for k, v in RUNS.items():
    print(f"{k}: per-step rows={len(ps[k])} methods={sorted(set(ps[k].method + '|' + ps[k].variant))} states={len(st[k])}")
pd.set_option("display.width", 220)
floor = cmp_ps("old1", "old2")
chg = cmp_ps("old1", "new")
t = floor.join(chg, lsuffix="_floor", rsuffix="_new", how="outer")
print("\n=== per-step metrics: max |diff| old1-vs-old2 (floor) and old1-vs-new ===")
print(t[["n_floor", "max_abs_floor", "max_rel_floor", "n_new", "max_abs_new", "max_rel_new", "nan_a_new", "nan_b_new"]].to_string())
sf, sn = cmp_states("old1", "old2"), cmp_states("old1", "new")
if len(sf) and len(sn):
    j = sf.merge(sn, on=["file", "key"], suffixes=("_floor", "_new"), how="outer")
    j = j[j.key.str.contains("ens|mean|sample|state|traj", case=False, regex=True) | True]
    print("\n=== saved states: max |diff| (floor vs new) ===")
    print(j.groupby("file")[["max_abs_floor", "max_abs_new", "identical_floor", "identical_new"]].agg(
        {"max_abs_floor": "max", "max_abs_new": "max", "identical_floor": "all", "identical_new": "all"}).to_string())
only = {k: sorted(set(st[k])) for k in RUNS}
print("\nstates only in some runs:", {k: sorted(set(v) - set(only['old1'])) for k, v in only.items() if set(v) != set(only['old1'])})
