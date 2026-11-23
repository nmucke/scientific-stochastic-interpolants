#!/usr/bin/env bash
# Poor-IC NS E=1000 global EnKF reference ("EnKF reference (E=1000)" row), as
# slurm_wrong_ic/make_manifest.py's reference tasks: ic_index=15, NP=40, traj 10-14,
# super-resolution 32^2 and 16^2. Resumable (skips cells whose metrics exist).
set -u
source "$(dirname "$0")/../env.sh"
cd "$REPO"
D="$PE/results/_runs/ns_wic_ref"
export JAX_COMPILATION_CACHE_DIR="$REPO/.jax_cache"
for N in ${TRAJ:-10 11 12 13 14}; do
  for SCEN in "32^2->128^2" "16^2->128^2"; do
    SS=$(echo "$SCEN" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//')
    OUT="$D/metrics/ref_${SS}__traj${N}.csv"
    [ -f "$OUT" ] && { echo "[ref] SKIP $SS traj$N"; continue; }
    echo "[ref] RUN $SS traj$N $(date -Is)"
    if "${PY_REF:-$REPO/.venv-jaxcfd/bin/python}" -u "$PE/run.py" case=navier_stokes seeds=[0] case.require_weights=true case.device=cuda \
        "+test_index=$N" +divergence_rmse_threshold=10.0 '+ns_methods=["EnKF"]' "+ns_scenarios=[\"$SCEN\"]" \
        num_steps=50 +ic_index=15 case.num_physical_steps=40 case.reference_ensemble_size=8 \
        +skip_kl_reference=true +save_states=true "+states_root=$D/reference/traj$N/gt" ensemble_size=1000 \
        "results_file=$D/.tmp_${SS}__traj${N}.csv" "hydra.run.dir=$D/hydra/${SS}__traj${N}" \
        > "$D/logs/ref_${SS}__traj${N}.log" 2>&1; then
      mv "$D/.tmp_${SS}__traj${N}.csv" "$OUT"; echo "[ref] OK  $SS traj$N $(date -Is)"
    else
      echo "[ref] FAIL $SS traj$N $(date -Is)"
    fi
  done
done
echo "[ref] ALL DONE $(date -Is)"
