#!/usr/bin/env python3
"""Write the wrong-IC Navier--Stokes task manifests (one file per trajectory/scenario).

Every task is ONE method in ONE (trajectory, scenario) cell, so a job's runtime is
one method's rollout. Fields are tab-separated:
    tag  metrics_csv  per_step_csv|-  run.py overrides...
Usage: make_manifest.py <root> <manifest_dir> <ic_index> <np> <M> <E> <save_traj> <traj...>
"""
import json
import os
import re
import sys
from pathlib import Path

SCENARIOS = ["16^2->128^2", "32^2->128^2", "sparse 5%", "sparse 1.5625%"]
OURS = {"si_sde": "Ours (SI-SDE)", "dm_sde": "Ours (DM-SDE)", "fm_ode": "Ours (FM-ODE)"}
BASELINES = {"flowdas": "FlowDAS", "surge_flowdas": "SURGE (FlowDAS)", "sda": "SDA",
             "surge_sda": "SURGE (SDA)", "dflow_sgld": "D-Flow SGLD", "fig": "Guided FM (FIG)"}
CLASSICAL = {"enkf": "EnKF", "particle_filter": "Particle filter"}


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def main():
    root, mdir = Path(sys.argv[1]), Path(sys.argv[2])
    ic, npsteps, M, E, save_traj = (int(x) for x in sys.argv[3:8])
    trajs = [int(x) for x in sys.argv[8:]]
    mdir.mkdir(parents=True, exist_ok=True)
    common = [f"+ic_index={ic}", f"case.num_physical_steps={npsteps}"]
    refs, n = [], 0
    for N in trajs:
        klref = root / "reference" / f"traj{N}" / "gt"
        save = N == save_traj
        states = root / "states" / (f"traj{N}" if save else "_unused")
        for scen in SCENARIOS:
            ss = slug(scen)
            scen_arg = "+ns_scenarios=" + json.dumps([scen])
            # KL is not reported for NS in the paper: by default skip both the KL
            # reference load and the driver's own large SI-SDE reference draw.
            kl = ([f"+kl_reference_states={klref}"] if os.environ.get("WITH_KL") == "1"
                  else ["+skip_kl_reference=true"])
            st = [f"+save_states={str(save).lower()}", f"+states_root={states}"]
            refs.append("\t".join([
                f"reference/{ss}/traj{N}", str(root / "metrics" / f"ref_{ss}__traj{N}.csv"), "-",
                "+ns_methods=[\"EnKF\"]", scen_arg, "num_steps=50", *common,
                "case.reference_ensemble_size=8", "+skip_kl_reference=true",
                "+save_states=true", f"+states_root={klref}", "ensemble_size=1000"]))
            rows = []

            def add(tree, stem, tag, method, extra):
                d = root / tree if tree else root
                rows.append("\t".join([
                    tag, str(d / "metrics" / f"{stem}.csv"), str(d / "per_step" / f"{stem}.csv"),
                    "+ns_methods=" + json.dumps([method]), scen_arg, *common, *kl,
                    f"ensemble_size={E}", *extra]))

            base = f"{ss}__M{M}__traj{N}"
            for key, method in OURS.items():
                add("", f"{base}__ours_jacfree__{key}", f"jacfree/{key}/{ss}/traj{N}", method,
                    [f"num_steps={M}", "likelihood_mode=dps_jacobian_free", *st])
                add("", f"{base}__ours_shared_jac10__{key}", f"k10/{key}/{ss}/traj{N}", method,
                    [f"num_steps={M}", "likelihood_mode=inflated_shared",
                     "+jacobian_refresh_every=10", "+variant_override=shared_jac10", *st])
                for k in ((1, 5) if os.environ.get("WITH_ABLATION") == "1" else ()):
                    # ablation (opt-in): separate tree, metrics only
                    add("ablation", f"{base}__ours_shared_jac{k}__{key}",
                        f"abl_k{k}/{key}/{ss}/traj{N}", method,
                        [f"num_steps={M}", "likelihood_mode=inflated_shared",
                         f"+jacobian_refresh_every={k}", f"+variant_override=shared_jac{k}"])
            for key, method in BASELINES.items():
                add("", f"{base}__baselines__{key}", f"baseline/{key}/{ss}/traj{N}", method,
                    [f"num_steps={M}", "likelihood_mode=dps_jacobian_free", *st])
            loc = ["+enkf_localization_radius=20"] if scen.startswith("sparse") else []
            for key, method in CLASSICAL.items():
                add("", f"{ss}__traj{N}__classical__{key}", f"classical/{key}/{ss}/traj{N}",
                    method, ["num_steps=50", *loc, *st])
            (mdir / f"cells_traj{N}_{ss}.tsv").write_text("\n".join(rows) + "\n")
            n += len(rows)
    (mdir / "reference.tsv").write_text("\n".join(refs) + "\n")
    print(f"{len(refs)} reference tasks + {n} cell tasks = {len(refs) + n}")


if __name__ == "__main__":
    main()
