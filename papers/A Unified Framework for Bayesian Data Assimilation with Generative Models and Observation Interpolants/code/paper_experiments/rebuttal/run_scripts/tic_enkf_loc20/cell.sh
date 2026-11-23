#!/usr/bin/env bash
# True-IC NS EnKF, E=64, Gaspari-Cohn localisation r=20, on CPU (as run_ns_grid.sh's classical
# cell, EnKF only, + enkf_localization_radius=20; KL reference skipped, not reported).
# usage: cell.sh <scenario slug> <traj> <NP> <outdir> <cpu list>
set -u
source "$(dirname "$0")/../env.sh"
cd "$REPO"
SS=$1; N=$2; NP=$3; OUT=$4; CPUS=$5
declare -A SCN=([32_2_128_2]="32^2->128^2" [16_2_128_2]="16^2->128^2")
NT=$(echo "$CPUS" | awk -F- '{print $2-$1+1}')
export ENKF_JAX_PLATFORM=cpu JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NT"
export JAX_COMPILATION_CACHE_DIR="$REPO/.jax_cache_cpu"
ID="${SS}__traj${N}__classical__enkf"
mkdir -p "$OUT"/{metrics,per_step,logs,hydra}
[ -f "$OUT/metrics/$ID.csv" ] && { echo "SKIP $ID"; exit 0; }
taskset -c "$CPUS" "$REPO/.venv-jaxcfd/bin/python" -u "$PE/run.py" case=navier_stokes seeds=[0] \
  ensemble_size=64 case.num_physical_steps=$NP case.require_weights=true case.device=cpu \
  "+test_index=$N" +save_per_step=true "+per_step_file=$OUT/per_step/$ID.csv" \
  +divergence_rmse_threshold=10.0 "results_file=$OUT/.tmp_$ID.csv" "hydra.run.dir=$OUT/hydra/$ID" \
  '+ns_methods=["EnKF"]' "+ns_scenarios=[\"${SCN[$SS]}\"]" num_steps=50 \
  +enkf_localization_radius=20 +skip_kl_reference=true +save_states=false \
  > "$OUT/logs/$ID.log" 2>&1 && mv "$OUT/.tmp_$ID.csv" "$OUT/metrics/$ID.csv" && echo "OK $ID $(date -Is)" || echo "FAIL $ID $(date -Is)"
