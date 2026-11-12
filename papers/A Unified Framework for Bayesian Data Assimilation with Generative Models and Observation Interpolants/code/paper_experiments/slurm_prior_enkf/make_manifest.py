#!/usr/bin/env python3
"""Write the generative-prior + EnKF manifest for the wrong-IC Navier--Stokes grid.

One task = ONE prior (SI-SDE / DM-SDE / FM-ODE) in ONE (trajectory, scenario) cell:
3 priors x 4 scenarios x len(traj) trajectories. Settings match the wrong-IC grid
(slurm_wrong_ic/make_manifest.py): foreign IC, NP, E, seed 0, no KL; the EnKF
analysis uses the EnKF cells' localisation (Gaspari-Cohn r=20 on the sparse
scenarios, none on super-resolution) and inflation 1.0; the prior forecasts with
M sampler steps (the Ours cells' M). Fields are tab-separated:
    tag  metrics_csv  per_step_csv  run.py overrides...
Usage: make_manifest.py <root> <manifest> <ic_index> <np> <M> <E> <save_traj> <traj...>
"""
import json
import re
import sys
from pathlib import Path

SCENARIOS = ["16^2->128^2", "32^2->128^2", "sparse 5%", "sparse 1.5625%"]
PRIORS = {"si_sde": "SI-SDE prior + EnKF", "dm_sde": "DM-SDE prior + EnKF",
          "fm_ode": "FM-ODE prior + EnKF"}


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def main():
    root, manifest = Path(sys.argv[1]), Path(sys.argv[2])
    ic, npsteps, M, E, save_traj = (int(x) for x in sys.argv[3:8])
    trajs = [int(x) for x in sys.argv[8:]]
    rows = []
    for N in trajs:
        save = N == save_traj
        states = root / "states" / (f"traj{N}" if save else "_unused")
        for scen in SCENARIOS:
            ss = slug(scen)
            loc = ["+enkf_localization_radius=20"] if scen.startswith("sparse") else []
            for key, method in PRIORS.items():
                stem = f"{ss}__M{M}__traj{N}__prior_enkf__{key}"
                rows.append("\t".join([
                    f"prior_enkf/{key}/{ss}/traj{N}",
                    str(root / "metrics" / f"{stem}.csv"),
                    str(root / "per_step" / f"{stem}.csv"),
                    "+ns_methods=" + json.dumps([method]),
                    "+ns_scenarios=" + json.dumps([scen]),
                    f"+ic_index={ic}", f"case.num_physical_steps={npsteps}",
                    "+skip_kl_reference=true", f"ensemble_size={E}", f"num_steps={M}",
                    "+enkf_inflation=1.0", *loc,
                    f"+save_states={str(save).lower()}", f"+states_root={states}",
                    f"+prior_enkf_diagnostics_dir={root / 'diagnostics'}",
                ]))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(rows) + "\n")
    print(f"{len(rows)} tasks -> {manifest}")


if __name__ == "__main__":
    main()
