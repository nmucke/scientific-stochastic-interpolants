"""Snapshot raw ensembles of existing linear Lorenz 96 cells, for a bitwise before/after check.

    python linear_identity_snapshot.py <out.pt>            # take a snapshot
    python linear_identity_snapshot.py <new.pt> --compare <old.pt>

CPU, 2 validation sequences, seed 0. Samplers at M=10 (both likelihood modes) to keep
it quick; everything else at its config settings. No references needed: compares
ensembles, not scores. Runs from the clone that contains this interpreter's PYTHONPATH,
so run it with REPO/PE of the code under test.
"""

import os
import sys
from pathlib import Path

import torch

PE = Path(os.environ["PE"])
sys.path.insert(0, str(PE))
from cases.lorenz96 import driver  # noqa: E402

CELLS = [
    ("si_sde", {"num_steps": 10, "likelihood_mode": "inflated"}),
    ("si_sde", {"num_steps": 10, "likelihood_mode": "dps_jacobian_free"}),
    ("fm_ode", {"num_steps": 10, "likelihood_mode": "dps_jacobian_free"}),
    ("dm_sde", {"num_steps": 10, "likelihood_mode": "dps_jacobian_free"}),
    ("si_sde_fc", {"num_steps": 10, "likelihood_mode": "dps_jacobian_free"}),
    ("prior_enkf", {"num_steps": 10}),
    ("enkf", {"localisation": 4}),
    ("enkf_large", {}),
    ("oracle", None),
    ("oracle_fc", None),
    ("pf_matched", None),
]


def snapshot():
    torch.set_num_threads(4)
    cfg = driver.case_config()
    cfg.evaluation.device = "cpu"
    problem = driver.problem_of(cfg)
    seqs = driver.load_sequences(problem, "validation", 2)
    out = {}
    for name, extra in CELLS:
        settings = driver.method_settings(cfg, name)
        if extra:
            settings = {**settings, **extra}
        res = driver.run_method(cfg, name, problem, seqs, "validation", 0, settings)
        key = f"{name}|{sorted((extra or {}).items())}"
        out[key] = res.ensembles.clone()
        print(key, tuple(res.ensembles.shape), float(res.ensembles.double().abs().sum()), flush=True)
    # The sequences themselves (truth + observations) must not move either.
    out["sequences"] = torch.stack([torch.cat([s.truth[1:], s.observations], 1) for s in seqs])
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1])
    snap = snapshot()
    torch.save(snap, target)
    if "--compare" in sys.argv:
        old = torch.load(sys.argv[sys.argv.index("--compare") + 1])
        bad = [k for k in old if k not in snap or not torch.equal(old[k], snap[k], )]
        for k in old:
            if k in snap and not torch.equal(old[k], snap[k]):
                diff = (old[k].double() - snap[k].double()).abs()
                print("DIFF", k, float(diff.nan_to_num().max()), flush=True)
        print("IDENTICAL" if not bad else f"CHANGED: {bad}")
        sys.exit(1 if bad else 0)
