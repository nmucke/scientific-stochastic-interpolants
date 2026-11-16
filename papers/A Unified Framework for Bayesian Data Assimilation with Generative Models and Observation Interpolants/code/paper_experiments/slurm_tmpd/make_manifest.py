#!/usr/bin/env python3
"""Task manifest for the TMPD grids on Navier--Stokes and uDALES (TMPD.md).

ONE line = ONE cell = one run.py call (one method, scenario, trajectory). Array task
i runs line i+1 (or lines i*PACK+1 .. (i+1)*PACK, see run_task.sh). Tab-separated:

    tag  case  metrics_csv  per_step_csv  run.py overrides...

Grids (defaults = the settings of the grid TMPD is compared against):

* ``ns_wrong_ic``: slurm_wrong_ic/ (REBUTTAL_EXPERIMENTS.md section 1): ic_index 15,
  num_physical_steps 40, M 100, E 64, traj 10-14, 4 scenarios, skip KL, states for
  traj 11. File names as that grid's baseline cells:
  ``<scen>__M100__traj<N>__baselines__tmpd.csv``, so aggregate_ns.py picks them up.
* ``ns_trueic``: the manuscript's original ``tab:ns_accuracy`` cells (run_ns_grid.sh
  as run on 2026-07-13..08-03 into results/navier_stokes): the truth's own IC,
  num_physical_steps 20, M 100, E 64, traj 10-14, 4 scenarios, the E=1000 EnKF KL
  reference of that tree (read only), states for traj 11.
  ``<scen>__M100__traj<N>__tmpd.csv``.
* ``udales``: tab:urban_accuracy_M50 (slurm_udales_fc/ without the FC flags):
  ic_index 8, num_physical_steps 150, DA every 3 from step 5, M 50, E 64, traj 1-5,
  sparse 1.5625% and 0.78125%, states for traj 1. ``<scen>__M50__traj<N>__tmpd.csv``.

Every grid adds the divergence guard 10 and per-step curves (run_task.sh). Only TMPD
by default; ``--methods`` adds Ours (DM-SDE) comparators (smoke tests), named as in
the grid they come from.

Usage: make_manifest.py <grid> <root> <manifest_dir> [--np N] [--traj ...]
       [--scenarios 'a|b'] [--methods tmpd,dm_sde_jacfree,dm_sde_shared_jac10]
       [--no-kl] [--save-traj N] [--extra 'k=v k=v']
"""
import argparse
import json
import re
from pathlib import Path

NS_SCENARIOS = ["16^2->128^2", "32^2->128^2", "sparse 5%", "sparse 1.5625%"]
URBAN_SCENARIOS = ["sparse 1.5625%", "sparse 0.78125%"]
# The manuscript's original true-IC NS tree: its E=1000 EnKF KL references are read,
# never written.
TRUEIC_KL_ROOT = ("/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/"
                  "results/navier_stokes/reference")

GRIDS = {
    "ns_wrong_ic": dict(case="navier_stokes", ic=15, np=40, M=100, E=64,
                        traj=[10, 11, 12, 13, 14], scenarios=NS_SCENARIOS, save=11, batch=16),
    "ns_trueic": dict(case="navier_stokes", ic=None, np=20, M=100, E=64,
                      traj=[10, 11, 12, 13, 14], scenarios=NS_SCENARIOS, save=11, batch=16),
    "udales": dict(case="urban", ic=8, np=150, M=50, E=64, every=3, start=5,
                   traj=[1, 2, 3, 4, 5], scenarios=URBAN_SCENARIOS, save=1, batch=8),
}
METHODS = {
    "tmpd": ("TMPD", []),
    "dm_sde_jacfree": ("Ours (DM-SDE)", ["likelihood_mode=dps_jacobian_free"]),
    "dm_sde_shared_jac10": ("Ours (DM-SDE)", [
        "likelihood_mode=inflated_shared", "+jacobian_refresh_every=10",
        "+variant_override=shared_jac10"]),
}


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def stem(grid, key, ss, M, N):
    """File stem in the naming of the grid the cell belongs to."""
    base = f"{ss}__M{M}__traj{N}"
    if key == "tmpd":
        return f"{base}__baselines__tmpd" if grid == "ns_wrong_ic" else f"{base}__tmpd"
    variant = key.split("_", 2)[2]  # jacfree | shared_jac10
    return f"{base}__ours_{variant}__dm_sde"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid", choices=sorted(GRIDS))
    ap.add_argument("root", type=Path)
    ap.add_argument("mdir", type=Path)
    ap.add_argument("--np", type=int)
    ap.add_argument("--traj", type=int, nargs="+")
    ap.add_argument("--scenarios")
    ap.add_argument("--methods", default="tmpd")
    ap.add_argument("--save-traj", type=int)
    ap.add_argument("--no-kl", action="store_true", help="ns_trueic: skip the KL reference")
    ap.add_argument("--tmpd-batch", type=int,
                    help="TMPD members per forward/backward pass (+posterior_batch_size); "
                         "0 = all E in one batch. E=64 in one batch does not fit a 9.5 GiB "
                         "MIG slice (smoke 2026-11-13).")
    ap.add_argument("--num-steps", type=int, help="M (smoke tests; default: the grid's)")
    ap.add_argument("--ensemble", type=int, help="E of the cell (smoke tests; default: the grid's)")
    ap.add_argument("--parts", type=int, default=1,
                    help="split each TMPD cell into P independent ensemble parts of E/P "
                         "members (udales only; merged by udales_split/merge.py)")
    ap.add_argument("--extra", default="", help="more run.py overrides, space-separated")
    a = ap.parse_args()
    g = dict(GRIDS[a.grid])
    np_ = a.np or g["np"]
    trajs = a.traj or g["traj"]
    scens = [s for s in a.scenarios.split("|") if s] if a.scenarios else g["scenarios"]
    tmpd_batch = g["batch"] if a.tmpd_batch is None else a.tmpd_batch
    save_traj = a.save_traj if a.save_traj is not None else g["save"]
    methods = [m for m in a.methods.split(",") if m]
    bad = set(methods) - set(METHODS)
    if bad:
        raise SystemExit(f"unknown methods {sorted(bad)}; choose from {sorted(METHODS)}")
    bad = set(scens) - set(g["scenarios"])
    if bad:
        raise SystemExit(f"scenarios {sorted(bad)} are not in the {a.grid} grid")
    if g["ic"] is not None and g["ic"] in trajs:
        # prepare_truth_and_obs uses the truth's own IC when ic_index == test_index.
        raise SystemExit(f"ic_index={g['ic']} is also a truth trajectory {trajs}")
    M, E = a.num_steps or g["M"], a.ensemble or g["E"]
    P = a.parts
    if P > 1 and (a.grid != "udales" or methods != ["tmpd"] or E % P):
        raise SystemExit("--parts > 1: udales grid, tmpd only, P must divide E")
    common = [f"case.num_physical_steps={np_}", f"ensemble_size={E // P}", f"num_steps={M}"]
    if g["ic"] is not None:
        common.insert(0, f"+ic_index={g['ic']}")
    if a.grid == "udales":
        common += [f"+assimilate_every={g['every']}", f"+assimilate_from={g['start']}"]
    common += a.extra.split()

    rows = []
    for key in methods:
        method, margs = METHODS[key]
        for N in trajs:
            save = N == save_traj
            states = a.root / "states" / (f"traj{N}" if save else "_unused")
            st = [f"+save_states={str(save).lower()}", f"+states_root={states}"]
            for scen in scens:
                ss = slug(scen)
                if g["case"] == "navier_stokes":
                    sel = ["+ns_methods=" + json.dumps([method]),
                           "+ns_scenarios=" + json.dumps([scen])]
                    if a.grid == "ns_trueic" and not a.no_kl:
                        kl = [f"+kl_reference_states={TRUEIC_KL_ROOT}/traj{N}/gt"]
                    else:
                        kl = ["+skip_kl_reference=true"]
                    # The NS baseline cells of both grids pass the jacfree mode too
                    # (inert for TMPD; kept so the argument lists match).
                    margs_ = ["likelihood_mode=dps_jacobian_free"] if key == "tmpd" else margs
                else:
                    sel = ["+urban_methods=" + json.dumps([method]),
                           "+urban_scenarios=" + json.dumps([scen])]
                    kl, margs_ = [], margs
                if key == "tmpd" and tmpd_batch:
                    # Chunks the ensemble only: TMPD treats every member on its own.
                    margs_ = [*margs_, f"+posterior_batch_size={tmpd_batch}"]
                s = stem(a.grid, key, ss, M, N)
                if P == 1:
                    rows.append("\t".join([
                        f"{key}/{ss}/traj{N}", g["case"],
                        str(a.root / "metrics" / f"{s}.csv"),
                        str(a.root / "per_step" / f"{s}.csv"),
                        *sel, f"+test_index={N}", *common, *margs_, *kl, *st]))
                    continue
                for p in range(P):
                    # Part p of P: E/P members, own sampler seed (urban driver
                    # _seed_ensemble_part), full states kept for the merge.
                    d = a.root / "parts" / s / f"part{p}of{P}"
                    rows.append("\t".join([
                        f"{key}/{ss}/traj{N}/part{p}of{P}", g["case"],
                        str(d / "metrics.csv"), str(d / "per_step.csv"),
                        *sel, f"+test_index={N}", *common, *margs_,
                        f"+ensemble_part={p}", f"+ensemble_parts={P}",
                        "+save_states=true", f"+states_root={d / 'states'}"]))
    a.mdir.mkdir(parents=True, exist_ok=True)
    (a.mdir / "cells.tsv").write_text("\n".join(rows) + "\n")
    kl_desc = ("E=1000 EnKF reference " + TRUEIC_KL_ROOT
               if a.grid == "ns_trueic" and not a.no_kl else
               "skipped" if g["case"] == "navier_stokes" else "n/a (urban)")
    sched = (f" | DA every {g['every']} from {g['start']}" if a.grid == "udales" else "")
    print(f"{a.grid}: case={g['case']} | ic_index={g['ic'] if g['ic'] is not None else 'own (truth)'}"
          f" | num_physical_steps={np_}{sched} | M={M} E={E} | traj={trajs}"
          f" | scenarios={scens} | save_states traj{save_traj} | KL: {kl_desc}"
          f" | divergence guard 10 (run_task.sh) | methods={methods}"
          f" | parts={P} x {E // P} members | tmpd batch={tmpd_batch or E // P} | extra={a.extra or '-'} | {len(rows)} cells")


if __name__ == "__main__":
    main()
