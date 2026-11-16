#!/usr/bin/env python3
"""Generative prior + EnKF on the ORIGINAL (manuscript) NS case, EnKF localisation r=20 everywhere.

True IC (no ic_index), num_physical_steps=20, E=64, seed 0, traj 10..14, the 4 scenarios; priors
SI-SDE and FM-ODE (unguided forecast with M sampler steps), standard EnKF analysis (perturbed
observations, inflation 1.0) with Gaspari-Cohn r=20 on ALL scenarios (super-res uses the
|H|-weighted block centroids of 948fa40). No KL. One prior per cell.
Fields: tag  metrics_csv  per_step_csv  run.py overrides...
Usage: make_manifest.py <root> <manifest> <np> <M> <E> <save_traj> <radius> <traj...>
"""
import json
import re
import sys
from pathlib import Path

SCENARIOS = ["16^2->128^2", "32^2->128^2", "sparse 5%", "sparse 1.5625%"]
PRIORS = {"si_sde": "SI-SDE prior + EnKF", "fm_ode": "FM-ODE prior + EnKF"}


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def main():
    root, manifest = Path(sys.argv[1]), Path(sys.argv[2])
    npsteps, M, E, save_traj, radius = (int(x) for x in sys.argv[3:8])
    trajs = [int(x) for x in sys.argv[8:]]
    rows = []
    for N in trajs:
        save = N == save_traj
        states = root / "states" / (f"traj{N}" if save else "_unused")
        for scen in SCENARIOS:
            ss = slug(scen)
            for key, method in PRIORS.items():
                stem = f"{ss}__M{M}__traj{N}__prior_enkf_r{radius}__{key}"
                rows.append("\t".join([
                    f"prior_enkf_r{radius}/{key}/{ss}/traj{N}",
                    str(root / "metrics" / f"{stem}.csv"), str(root / "per_step" / f"{stem}.csv"),
                    "+ns_methods=" + json.dumps([method]), "+ns_scenarios=" + json.dumps([scen]),
                    f"case.num_physical_steps={npsteps}", "+skip_kl_reference=true",
                    f"ensemble_size={E}", f"num_steps={M}",
                    "+enkf_inflation=1.0", f"+enkf_localization_radius={radius}",
                    f"+save_states={str(save).lower()}", f"+states_root={states}",
                    f"+prior_enkf_diagnostics_dir={root / 'diagnostics'}"]))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(rows) + "\n")
    print(f"{len(rows)} cells")


if __name__ == "__main__":
    main()
