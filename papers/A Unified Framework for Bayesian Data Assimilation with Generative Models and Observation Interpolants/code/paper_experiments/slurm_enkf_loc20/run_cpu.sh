#!/bin/bash
# run_cpu.sh <scenario> <traj> <num_physical_steps> <out_root>
# True-solver EnKF, Gaspari-Cohn r=20, on CPU, from the FROZEN copy that produced the
# main wrong-IC EnKF cells (only the localisation and the device differ).
set -u
SCEN=$1; TRAJ=$2; NP=$3; ROOT=$4
F=/scratch/ntmucke/scientific-stochastic-interpolants/code_ns_wrong_ic
PE="$F/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments"
PY=/scratch/ntmucke/scientific-stochastic-interpolants/.venv-cu126/bin/python
export PYTHONPATH="$F/src:$F/external_libs/jax_cfd_lib/src"
export PYTHONUNBUFFERED=1
if [ "${DEVICE:-cpu}" = gpu ]; then
  DEV=cuda; export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest  # no TF32 on A100
else
  DEV=cpu; export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK:-8}"
SS=$(echo "$SCEN" | sed 's/\^/_/g; s/->/_/; s/[^A-Za-z0-9_]/_/g'); ID="${SS}__traj${TRAJ}__classical__enkf"
OUT="$ROOT/metrics/$ID.csv"; PS="$ROOT/per_step/$ID.csv"
[ -f "$OUT" ] && { echo "SKIP exists $ID"; exit 0; }
mkdir -p "$ROOT"/{metrics,per_step,logs,hydra,.tmp,states/traj11}
TMP="$ROOT/.tmp/$ID.${SLURM_JOB_ID:-x}.csv"; PSTMP="$ROOT/.tmp/$ID.${SLURM_JOB_ID:-x}.ps.csv"
SAVE=(+save_states=false "+states_root=$ROOT/states/_unused"); [ "$TRAJ" = 11 ] && SAVE=(+save_states=true "+states_root=$ROOT/states/traj11")
cd "$F"
"$PY" -c "import scisi,sys; assert scisi.__file__.startswith('$F/'), scisi.__file__; print('scisi', scisi.__file__)" || exit 4
"$PY" -u "$PE/run.py" case=navier_stokes seeds=[0] case.require_weights=true case.device=$DEV \
  "+test_index=$TRAJ" +divergence_rmse_threshold=10.0 "results_file=$TMP" "hydra.run.dir=$ROOT/hydra/$ID" \
  +save_per_step=true "+per_step_file=$PSTMP" '+ns_methods=["EnKF"]' "+ns_scenarios=[\"$SCEN\"]" \
  +ic_index=15 case.num_physical_steps=$NP +skip_kl_reference=true ensemble_size=64 num_steps=50 \
  +enkf_localization_radius=20 "${SAVE[@]}" > "$ROOT/logs/$ID.log" 2>&1
rc=$?
if [ $rc -eq 0 ] && [ -s "$TMP" ]; then mv -f "$PSTMP" "$PS"; mv -f "$TMP" "$OUT"; echo "OK $ID"; else echo "FAIL rc=$rc $ID (log $ROOT/logs/$ID.log)"; exit 1; fi
