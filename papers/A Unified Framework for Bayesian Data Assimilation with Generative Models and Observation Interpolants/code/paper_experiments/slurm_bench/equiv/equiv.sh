#!/bin/bash
# equiv.sh <old|new> <tag>: run short wrong-IC NS cells (grid settings, NP=7) with old (frozen) or new (live) code.
set -u
VER=$1; TAG=$2; ONLY=${3:-all}
F=/scratch/ntmucke/scientific-stochastic-interpolants/code_ns_wrong_ic
L=/home/ntmucke/scientific-stochastic-interpolants
PY=/scratch/ntmucke/scientific-stochastic-interpolants/.venv-cu126/bin/python
OUT=/scratch/ntmucke/scientific-stochastic-interpolants/slurm_bench/equiv/${VER}_${TAG}; mkdir -p $OUT
if [ "$VER" = old ]; then
  REPO=$F; export PYTHONPATH="$F/src:$F/external_libs/jax_cfd_lib/src"
else
  REPO=$L; unset PYTHONPATH
fi
PE=$(echo $REPO/papers/*/code/paper_experiments)
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 JAX_COMPILATION_CACHE_DIR=$OUT/.jax_cache
cd "$REPO"; nvidia-smi --query-gpu=name --format=csv,noheader; hostname
"$PY" -c "import scisi, importlib.util as u; print('scisi from', scisi.__file__)"
COMMON=(case=navier_stokes seeds=[0] case.require_weights=true case.device=cuda +test_index=10
  +divergence_rmse_threshold=10.0 +ic_index=15 case.num_physical_steps=7 +skip_kl_reference=true ensemble_size=64
  +save_per_step=true +save_states=true)
run() { # run <name> <args...>
  local n=$1; shift
  [ "$ONLY" != all ] && [[ ",$ONLY," != *",$n,"* ]] && return 0
  local t0=$(date +%s)
  "$PY" -u "$PE/run.py" "${COMMON[@]}" "+per_step_file=$OUT/ps_$n.csv" "+states_root=$OUT/states_$n" \
     "results_file=$OUT/m_$n.csv" "hydra.run.dir=$OUT/hydra_$n" "$@" > "$OUT/log_$n.log" 2>&1
  echo "RUN $n rc=$? wall=$(( $(date +%s)-t0 ))s"
}
SC='+ns_scenarios=["32^2->128^2","sparse 5%"]'
run jacfree_base '+ns_methods=["Ours (SI-SDE)","Ours (DM-SDE)","Ours (FM-ODE)","FlowDAS","SURGE (FlowDAS)","SDA","SURGE (SDA)","Guided FM (FIG)"]' "$SC" num_steps=100 likelihood_mode=dps_jacobian_free
run shared_k10 '+ns_methods=["Ours (SI-SDE)","Ours (DM-SDE)","Ours (FM-ODE)"]' "$SC" num_steps=100 likelihood_mode=inflated_shared +jacobian_refresh_every=10 +variant_override=shared_jac10
run classical '+ns_methods=["EnKF","Particle filter"]' '+ns_scenarios=["32^2->128^2","sparse 5%"]' num_steps=50
run classical_loc '+ns_methods=["EnKF"]' '+ns_scenarios=["sparse 5%"]' num_steps=50 +enkf_localization_radius=20
run dflow '+ns_methods=["D-Flow SGLD"]' '+ns_scenarios=["32^2->128^2"]' num_steps=100 likelihood_mode=dps_jacobian_free case.num_physical_steps=6
echo ALLDONE
