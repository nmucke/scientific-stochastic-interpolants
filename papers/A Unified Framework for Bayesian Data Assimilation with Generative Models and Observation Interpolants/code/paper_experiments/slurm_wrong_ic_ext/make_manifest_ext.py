#!/usr/bin/env python3
"""Manifest for the wrong-IC Navier--Stokes EXTENSION grid (one method per task).

Same wrong-IC setting as ../slurm_wrong_ic (ic_index, NP=40, M=100, E=64, traj 10..14,
4 scenarios, no KL), with two new groups:

  surge_ours  "SURGE (SI-SDE)" / "SURGE (DM-SDE)" (SURGE has no FM-ODE form)
              x variants jacfree / shared_jac10.
  fc          Ours (SI-SDE, DM-SDE, FM-ODE) x jacfree / shared_jac10 with the
              forecast-residual correction (case.forecast_correction.*).

Fields (tab-separated, as ../slurm_wrong_ic/run_task.sh expects):
    tag  metrics_csv  per_step_csv  run.py overrides...
The tag ends in traj<N> (run_task.sh reads test_index from it); filenames carry the
__traj<N>__ token that aggregate_ns.py / status.py key on.

Usage: make_manifest_ext.py <root> <manifest_dir> <ic> <np> <M> <E> <save_traj> <groups>
       <fc_tag> <fc_overrides|-> <traj...>
  groups: comma list from {surge_ours, fc}
  fc_tag: "fc<steps|M>_<hash8>" from `python -m common.forecast_correction tag`
          for the SAME overrides (submit_ext.sh computes it).
"""
import json
import re
import sys
from pathlib import Path

SCENARIOS = ["16^2->128^2", "32^2->128^2", "sparse 5%", "sparse 1.5625%"]
OURS = {"si_sde": "Ours (SI-SDE)", "dm_sde": "Ours (DM-SDE)", "fm_ode": "Ours (FM-ODE)"}
SURGE_OURS = {"si_sde": "SURGE (SI-SDE)", "dm_sde": "SURGE (DM-SDE)"}

# SURGE-on-Ours (FINAL, coordinator 2026-11-10): method configs surge_si_sde.yaml /
# surge_dm_sde.yaml inherit the per-scenario lambda tables. The methods are in
# WIRED_METHODS but not NS_METHODS, so +ns_methods names them explicitly (below).
# Per-variant arguments are exactly the Ours ones (VARIANTS); the driver tags their
# variants like the Ours rows. No further overrides are needed.
SURGE_OURS_EXTRA: list[str] = []

# Likelihood modes, as in the main grid (run_ns_grid.sh / slurm_wrong_ic).
VARIANTS = {
    "jacfree": ["likelihood_mode=dps_jacobian_free"],
    "shared_jac10": ["likelihood_mode=inflated_shared", "+jacobian_refresh_every=10",
                     "+variant_override=shared_jac10"],
}


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def main():
    root, mdir = Path(sys.argv[1]), Path(sys.argv[2])
    ic, npsteps, M, E, save_traj = (int(x) for x in sys.argv[3:8])
    groups = [g for g in sys.argv[8].split(",") if g]
    fc_tag, fc_over = sys.argv[9], sys.argv[10]
    trajs = [int(x) for x in sys.argv[11:]]
    bad = set(groups) - {"surge_ours", "fc"}
    if bad:
        sys.exit(f"unknown groups {sorted(bad)}")
    if "fc" in groups and not re.fullmatch(r"fc(M|\d+)_[0-9a-f]{8}", fc_tag):
        sys.exit(f"bad fc_tag {fc_tag!r}")
    fc_args = ["case.forecast_correction.enabled=true",
               *([] if fc_over == "-" else [f"case.forecast_correction.{kv}" for kv in fc_over.split()]),
               "case.forecast_correction.diagnostics=true",
               f"case.forecast_correction.diagnostics_root={root / 'forecast_correction'}"]
    common = [f"+ic_index={ic}", f"case.num_physical_steps={npsteps}", "+skip_kl_reference=true",
              f"ensemble_size={E}", f"num_steps={M}"]
    mdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for N in trajs:
        save = N == save_traj
        states = root / "states" / (f"traj{N}" if save else "_unused")
        st = [f"+save_states={str(save).lower()}", f"+states_root={states}"]
        for scen in SCENARIOS:
            ss = slug(scen)
            base = f"{ss}__M{M}__traj{N}"

            def add(stem, tag, method, extra):
                rows.append("\t".join([
                    f"{tag}/{ss}/traj{N}", str(root / "metrics" / f"{stem}.csv"),
                    str(root / "per_step" / f"{stem}.csv"),
                    "+ns_methods=" + json.dumps([method]), "+ns_scenarios=" + json.dumps([scen]),
                    *common, *extra, *st]))

            for var, vargs in VARIANTS.items():
                if "surge_ours" in groups:
                    for key, method in SURGE_OURS.items():
                        add(f"{base}__surge_ours_{var}__{key}", f"surge_ours_{var}/{key}", method,
                            [*vargs, *SURGE_OURS_EXTRA])
                if "fc" in groups:
                    for key, method in OURS.items():
                        add(f"{base}__ours_{var}__{fc_tag}__{key}", f"fc_{var}/{key}", method,
                            [*vargs, *fc_args])
    # Longest first: shared before jacfree (both groups), so the tail is short.
    rows.sort(key=lambda r: 0 if "shared_jac10" in r.split("\t")[0] else 1)
    (mdir / "ext_cells.tsv").write_text("\n".join(rows) + "\n")
    counts = {}
    for r in rows:
        g = r.split("\t")[0].split("/")[0]
        counts[g] = counts.get(g, 0) + 1
    print(f"{len(rows)} cell tasks: {counts}")


if __name__ == "__main__":
    main()
