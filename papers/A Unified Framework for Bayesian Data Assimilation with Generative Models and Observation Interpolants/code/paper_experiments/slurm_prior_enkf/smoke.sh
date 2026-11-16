#!/bin/bash
# Smoke for generative prior + EnKF: 3 assimilation steps (NP=8), E=64, M=100,
# traj 11, foreign IC 15, the three priors + the true-solver EnKF on one
# super-resolution (32^2, no localisation) and one sparse (5%, r=20) cell.
# sbatch -p gpu-a100-small -n 1 -c 2 --gres=gpu:1 --mem-per-cpu=8000M -A research-ceg-gse \
#        -t 03:30:00 -o $SMOKE/slurm_%j.out smoke.sh
set -u
REPO=/scratch/ntmucke/scientific-stochastic-interpolants/code_ns_prior_enkf
PE="$REPO/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments"
PY=/scratch/ntmucke/scientific-stochastic-interpolants/.venv-cu126/bin/python
SMOKE=${SMOKE:-/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/smoke_prior_enkf}
export PYTHONPATH="$REPO/src:$REPO/external_libs/jax_cfd_lib/src"
export JAX_COMPILATION_CACHE_DIR="$SMOKE/.jax_cache" XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
cd "$REPO" || exit 1
"$PY" -c "import scisi; print('scisi:', scisi.__file__)"
nvidia-smi -L
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>&1 | tr '\n' ' ') | $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>&1 | tr '\n' ' ')"; sleep 5; done ) > "$SMOKE/gpu_mem.log" &
MON=$!
METHODS='["SI-SDE prior + EnKF","DM-SDE prior + EnKF","FM-ODE prior + EnKF","EnKF"]'
run() { # run <scenario> <extra...>
  local scen=$1; shift; local id; id=$(echo "$scen" | tr -c 'A-Za-z0-9' _)
  echo "=== $scen $(date +%T)"
  /usr/bin/time -v "$PY" -u "$PE/run.py" case=navier_stokes seeds=[0] case.require_weights=true \
    case.device=cuda +test_index=11 +ic_index=15 case.num_physical_steps=8 \
    +skip_kl_reference=true +divergence_rmse_threshold=10.0 ensemble_size=64 num_steps=100 \
    "+ns_methods=$METHODS" "+ns_scenarios=[\"$scen\"]" +enkf_inflation=1.0 \
    "results_file=$SMOKE/metrics_$id.csv" "hydra.run.dir=$SMOKE/hydra/$id" \
    +save_states=true "+states_root=$SMOKE/states" +save_per_step=true \
    "+per_step_file=$SMOKE/per_step_$id.csv" "+prior_enkf_diagnostics_dir=$SMOKE/diagnostics" \
    "$@" > "$SMOKE/log_$id.txt" 2>&1
  echo "rc=$? $(date +%T)"
}
run "32^2->128^2"
run "sparse 5%" +enkf_localization_radius=20
kill $MON
