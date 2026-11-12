#!/usr/bin/env python3
"""Manifest for the uDALES (urban) forecast-correction runs: Ours x shared-Jacobian x FC.

One array TASK per (method, scenario, trajectory): 3 methods x 2 densities x 5
trajectories = 30 tasks by default (TRAJ=1 SCENARIOS="sparse 1.5625%" -> 3). A task
runs one row (the corrected cell), or two with include_plain=1 (plain first).
Settings follow run_urban_grid.sh (run_cell + the ours_shared_k<k> loop) with FC=on.

Fields of cells.tsv (tab-separated):
    task_key  tag  metrics_csv  per_step_csv  run.py overrides...
tasks.txt lists the task keys in array order (task i = line i+1).

Filenames carry the __traj<N>__ token that aggregate_urban.py / status.py key on:
    <scen>__M<M>__traj<N>__ours_shared_jac<k>[__<fc_tag>]__<method_key>.csv
The `variant` column in the rows is shared_jac<k>[_<fc_tag>] (driver-built).

Usage (all positional, submit.sh fills them from env vars):
  make_manifest.py <root> <manifest_dir> <ic|-1> <np> <M> <E> <k> <assim_every> <assim_from>
                   <save_traj> <methods,csv> <include_plain 0|1> <fc_tag> <fc_overrides|->
                   <extra_args|-> <scenarios '|'-sep> <traj...>
"""
import json
import re
import sys
from pathlib import Path

OURS = {"si_sde": "Ours (SI-SDE)", "dm_sde": "Ours (DM-SDE)", "fm_ode": "Ours (FM-ODE)"}


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def main():
    a = sys.argv[1:]
    root, mdir = Path(a[0]), Path(a[1])
    ic, npsteps, M, E, k, every, start, save_traj = (int(x) for x in a[2:10])
    methods = [m for m in a[10].split(",") if m]
    include_plain = a[11] == "1"
    fc_tag, fc_over, extra = a[12], a[13], a[14]
    scenarios = [s for s in a[15].split("|") if s]
    trajs = [int(x) for x in a[16:]]
    bad = set(methods) - set(OURS)
    if bad:
        sys.exit(f"unknown methods {sorted(bad)}; choose from {sorted(OURS)}")
    if not re.fullmatch(r"fc(M|\d+)_[0-9a-f]{8}", fc_tag):
        sys.exit(f"bad fc_tag {fc_tag!r}")
    if ic in trajs:
        sys.exit(f"ic_index={ic} is also a truth trajectory {trajs}")
    if save_traj not in trajs:
        sys.exit(f"save_traj={save_traj} not in {trajs}")

    var = f"shared_jac{k}"
    lik = ["likelihood_mode=inflated_shared", f"+jacobian_refresh_every={k}",
           f"+variant_override={var}"]
    fc_args = ["case.forecast_correction.enabled=true",
               *([] if fc_over == "-" else [f"case.forecast_correction.{kv}" for kv in fc_over.split()]),
               "case.forecast_correction.diagnostics=true",
               f"case.forecast_correction.diagnostics_root={root / 'forecast_correction'}"]
    common = [f"+ic_index={'null' if ic < 0 else ic}", f"case.num_physical_steps={npsteps}",
              f"ensemble_size={E}", f"num_steps={M}",
              f"+assimilate_every={every}", f"+assimilate_from={start}",
              *([] if extra == "-" else extra.split())]

    mdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in methods:
        method = OURS[key]
        for scen in scenarios:
            ss = slug(scen)
            for N in trajs:
                tkey = f"{key}__{ss}__traj{N}"
                save = N == save_traj
                states = root / "states" / (f"traj{N}" if save else "_unused")
                st = [f"+save_states={str(save).lower()}", f"+states_root={states}"]
                base = f"{ss}__M{M}__traj{N}__ours_{var}"
                kinds = ([("plain", base, [])] if include_plain else []) + \
                        [("fc", f"{base}__{fc_tag}", fc_args)]
                for kind, stem, fca in kinds:
                    stem = f"{stem}__{key}"
                    rows.append("\t".join([
                        tkey, f"{kind}_{var}/{key}/{ss}/traj{N}",
                        str(root / "metrics" / f"{stem}.csv"),
                        str(root / "per_step" / f"{stem}.csv"),
                        "+urban_methods=" + json.dumps([method]),
                        "+urban_scenarios=" + json.dumps([scen]),
                        f"+test_index={N}", *common, *lik, *fca, *st]))
    (mdir / "cells.tsv").write_text("\n".join(rows) + "\n")
    tasks = list(dict.fromkeys(r.split("\t", 1)[0] for r in rows))
    (mdir / "tasks.txt").write_text("\n".join(tasks) + "\n")
    print(f"{len(rows)} cells over {len(tasks)} array tasks")


if __name__ == "__main__":
    main()
