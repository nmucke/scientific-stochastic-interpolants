#!/usr/bin/env python3
"""Manifest: forecast correction on the ORIGINAL (manuscript, tab:ns_accuracy) Navier--Stokes case.

True IC (no ic_index), num_physical_steps=20 (5 history + 15 DA), M=100, E=64, seed 0,
traj 10..14, the 4 scenarios, Ours (SI-SDE / DM-SDE / FM-ODE) with the shared Jacobian
(inflated_shared, refresh every 10; lambda from configs/method/*.yaml, as run_ns_grid.sh's
ours_shared_k10 group), + the forecast correction. No KL (user, 2026-11-16).
One method per task. Fields: tag  metrics_csv  per_step_csv  run.py overrides...
File stems: <scen>__M100__traj<N>__ours_shared_jac10__<fc_tag>__<si_sde|dm_sde|fm_ode>.

Usage: make_manifest.py <root> <manifest_file> <np> <M> <E> <save_traj> <fc_tag> <fc_overrides|-> <variant> <traj...>
  variant: shared_jac10 (inflated_shared, refresh 10) | jacfree (dps_jacobian_free, as run_ns_grid.sh ours_jacfree)
"""
import json
import re
import sys
from pathlib import Path

SCENARIOS = ["16^2->128^2", "32^2->128^2", "sparse 5%", "sparse 1.5625%"]
OURS = {"si_sde": "Ours (SI-SDE)", "dm_sde": "Ours (DM-SDE)", "fm_ode": "Ours (FM-ODE)"}
VARIANTS = {
    "shared_jac10": ["likelihood_mode=inflated_shared", "+jacobian_refresh_every=10", "+variant_override=shared_jac10"],
    "jacfree": ["likelihood_mode=dps_jacobian_free"],
}


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def main():
    root, mfile = Path(sys.argv[1]), Path(sys.argv[2])
    npsteps, M, E, save_traj = (int(x) for x in sys.argv[3:7])
    fc_tag, fc_over = sys.argv[7], sys.argv[8]
    variant = sys.argv[9]
    if variant not in VARIANTS:
        sys.exit(f"unknown variant {variant!r}")
    trajs = [int(x) for x in sys.argv[10:]]
    if not re.fullmatch(r"fc(M|\d+)_[0-9a-f]{8}", fc_tag):
        sys.exit(f"bad fc_tag {fc_tag!r}")
    fc_args = ["case.forecast_correction.enabled=true",
               *([] if fc_over == "-" else [f"case.forecast_correction.{kv}" for kv in fc_over.split()]),
               "case.forecast_correction.diagnostics=true",
               f"case.forecast_correction.diagnostics_root={root / 'forecast_correction'}"]
    common = [f"case.num_physical_steps={npsteps}", "+skip_kl_reference=true",
              f"ensemble_size={E}", f"num_steps={M}"]
    rows = []
    for N in trajs:
        save = N == save_traj
        states = root / "states" / (f"traj{N}" if save else "_unused")
        st = [f"+save_states={str(save).lower()}", f"+states_root={states}"]
        for scen in SCENARIOS:
            ss = slug(scen)
            for key, method in OURS.items():
                stem = f"{ss}__M{M}__traj{N}__ours_{variant}__{fc_tag}__{key}"
                rows.append("\t".join([
                    f"fc_{variant}_{fc_tag}/{key}/{ss}/traj{N}",
                    str(root / "metrics" / f"{stem}.csv"), str(root / "per_step" / f"{stem}.csv"),
                    "+ns_methods=" + json.dumps([method]), "+ns_scenarios=" + json.dumps([scen]),
                    *common, *VARIANTS[variant], *fc_args, *st]))
    mfile.parent.mkdir(parents=True, exist_ok=True)
    mfile.write_text("\n".join(rows) + "\n")
    print(f"{len(rows)} cells")


if __name__ == "__main__":
    main()
