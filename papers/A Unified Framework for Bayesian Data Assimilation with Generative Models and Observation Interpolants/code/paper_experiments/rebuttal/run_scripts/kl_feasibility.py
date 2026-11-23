"""Is a nonparametric KL usable for Lorenz 96 (E=64, d=16) against the PF reference?

Scores known-quality cells on the 4 validation sequences, seeds 0-2, with
  kl_ens_pf  = KL(ensemble || PF)   kNN (Perez-Cruz), x = 64 members, y = 4096 PF draws
  kl_pf_ens  = KL(PF || ensemble)   kNN, x = 4096 PF draws, y = 64 members
  energy     = energy distance (ensemble vs PF draws)
next to the existing mean error and sliced W2. The estimator is usable if the
pf_subsample floor sits clearly below EnKF, EnKF below the oracle (the known order
from mean error), and the seed spread is small against those gaps.
Also counts duplicate members (resampled ensembles), which break KL(ens || PF).
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

PE = Path(os.environ["PE"])
sys.path.insert(0, str(PE))
from cases.lorenz96 import driver  # noqa: E402


def knn_kl(x, y):
    """Perez-Cruz 1-NN estimator of KL(P||Q), x ~ P [n,d], y ~ Q [m,d]; exact NN."""
    n, d = x.shape
    m = y.shape[0]
    r = cKDTree(x).query(x, k=2)[0][:, 1]
    s = cKDTree(y).query(x, k=1)[0]
    with np.errstate(divide="ignore"):
        return float(d * np.mean(np.log(s / r)) + np.log(m / (n - 1)))


def energy_distance(x, y):
    """2 E|X-Y| - E|X-X'| - E|Y-Y'| (U-statistics within samples)."""
    x = torch.as_tensor(x)
    y = torch.as_tensor(y)
    xy = torch.cdist(x, y).mean()
    xx = torch.cdist(x, x).sum() / (x.shape[0] * (x.shape[0] - 1))
    yy = torch.cdist(y, y).sum() / (y.shape[0] * (y.shape[0] - 1))
    return float(2 * xy - xx - yy)


def main():
    torch.set_num_threads(4)
    cfg = driver.case_config()
    problem = driver.problem_of(cfg)
    seqs = driver.load_sequences(problem, "validation", 4)
    refs = [driver.load_reference("validation", i, 0, problem) for i in range(4)]
    dup_ref = [
        int(len(r["samples"][t]) - len(torch.unique(r["samples"][t], dim=0)))
        for r in refs for t in range(0, 50, 10)
    ]
    print("duplicate rows in PF reference samples (seq x 5 steps):", dup_ref, flush=True)
    rows = []
    for name in ("pf_subsample", "pf_matched", "enkf", "oracle"):
        for seed in (0, 1, 2):
            settings = driver.method_settings(cfg, name)
            out = driver.run_method(cfg, name, problem, seqs, "validation", seed, settings)
            vals = {k: [] for k in ("kl_ens_pf", "kl_pf_ens", "energy", "mean_err", "sw2", "dups")}
            for s, (ens, ref) in enumerate(zip(out.ensembles, refs)):
                sc = driver.score_ensemble(ens, ref, seqs[s].truth[1:])
                vals["mean_err"].append(np.nanmean(sc["mean_error_vs_pf"]))
                vals["sw2"].append(np.nanmean(sc["sliced_w2"]))
                for t in range(10, 50):  # skip the transient, as the gate does
                    x = ens[t].double().numpy()
                    y = ref["samples"][t].double().numpy()
                    vals["dups"].append(x.shape[0] - np.unique(x, axis=0).shape[0])
                    vals["kl_ens_pf"].append(knn_kl(x, y))
                    vals["kl_pf_ens"].append(knn_kl(y, x))
                    vals["energy"].append(energy_distance(x, y))
            row = {k: float(np.mean(v)) for k, v in vals.items()}
            row.update(name=name, seed=seed)
            rows.append(row)
            print(row, flush=True)
    print("\nsummary (mean over seeds | std over seeds)")
    for name in ("pf_subsample", "pf_matched", "enkf", "oracle"):
        sub = [r for r in rows if r["name"] == name]
        print(name, {k: (round(float(np.mean([r[k] for r in sub])), 4),
                         round(float(np.std([r[k] for r in sub])), 4))
                     for k in ("kl_ens_pf", "kl_pf_ens", "energy", "mean_err", "sw2", "dups")})


if __name__ == "__main__":
    main()
