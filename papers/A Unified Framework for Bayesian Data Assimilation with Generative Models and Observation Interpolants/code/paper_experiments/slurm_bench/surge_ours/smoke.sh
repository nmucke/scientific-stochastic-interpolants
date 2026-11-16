#!/bin/bash
set -u
REPO=/home/ntmucke/scientific-stochastic-interpolants
PY=/scratch/ntmucke/scientific-stochastic-interpolants/.venv-cu126/bin/python
PE="$REPO/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments"
OUT=/scratch/ntmucke/scientific-stochastic-interpolants/slurm_bench/surge_ours
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$OUT/mem_${SLURM_JOB_ID}_${SLURM_STEP_ID:-0}.txt"; sleep 5; done ) & MON=$!
METHODS='["SURGE (SI-SDE)","Ours (SI-SDE)","SURGE (DM-SDE)","Ours (DM-SDE)"]'
for V in ${1:-jacfree shared_jac10}; do
  if [ $V = jacfree ]; then EXTRA=(likelihood_mode=dps_jacobian_free); else EXTRA=(likelihood_mode=inflated_shared +jacobian_refresh_every=10 +variant_override=shared_jac10); fi
  t0=$(date +%s)
  "$PY" -u "$PE/run.py" case=navier_stokes seeds=[0] case.require_weights=true case.device=cuda +test_index=10 \
    +divergence_rmse_threshold=10.0 "+ns_methods=$METHODS" '+ns_scenarios=["32^2->128^2","sparse 5%"]' +ic_index=15 \
    case.num_physical_steps=7 +skip_kl_reference=true ensemble_size=64 num_steps=100 "${EXTRA[@]}" \
    +save_per_step=true "+per_step_file=$OUT/per_step_$V.csv" \
    "results_file=$OUT/metrics_$V.csv" "hydra.run.dir=$OUT/hydra_$V" > "$OUT/run_$V.log" 2>&1
  echo "RUN $V rc=$? wall=$(( $(date +%s)-t0 ))s"
  grep -hE '\[NS\]|ess|Error' "$OUT/run_$V.log" | grep -vE 'UserWarning' | cut -c1-260 | tail -12
done
kill $MON; echo "PEAK MiB: $(sort -n "$OUT/mem_${SLURM_JOB_ID}_${SLURM_STEP_ID:-0}.txt" | tail -1)"
