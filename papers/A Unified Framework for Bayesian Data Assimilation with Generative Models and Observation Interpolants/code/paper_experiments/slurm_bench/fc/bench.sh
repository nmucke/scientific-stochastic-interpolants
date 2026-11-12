#!/bin/bash
set -u
REPO=/home/ntmucke/scientific-stochastic-interpolants
PY=/scratch/ntmucke/scientific-stochastic-interpolants/.venv-cu126/bin/python
PE="$REPO/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments"
OUT=/scratch/ntmucke/scientific-stochastic-interpolants/slurm_bench/fc
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
cd "$REPO"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# cell: name|method|scenario|likelihood_mode|extra
CELLS=(
 "si_jf_32|Ours (SI-SDE)|32^2->128^2|dps_jacobian_free|"
 "fm_jf_32|Ours (FM-ODE)|32^2->128^2|dps_jacobian_free|"
 "dm_jf_s5|Ours (DM-SDE)|sparse 5%|dps_jacobian_free|"
 "dm_k10_s5|Ours (DM-SDE)|sparse 5%|inflated_shared|+jacobian_refresh_every=10 +variant_override=shared_jac10"
)
ONLY=${1:-all}
for c in "${CELLS[@]}"; do
  IFS='|' read -r n meth scen lm extra <<< "$c"
  [ "$ONLY" != all ] && [[ ",$ONLY," != *",$n,"* ]] && continue
  ( peak=0; while sleep 5; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -gt "$peak" ] && peak=$u && echo $peak > $OUT/peak_$n; done ) & MON=$!
  t0=$(date +%s)
  "$PY" -u "$PE/run.py" case=navier_stokes seeds=[0] case.require_weights=true case.device=cuda +test_index=10 \
    +divergence_rmse_threshold=10.0 "+ns_methods=[\"$meth\"]" "+ns_scenarios=[\"$scen\"]" +ic_index=15 \
    case.num_physical_steps=7 +skip_kl_reference=true ensemble_size=64 num_steps=100 likelihood_mode=$lm $extra \
    case.forecast_correction.enabled=true case.forecast_correction.diagnostics=true \
    "case.forecast_correction.diagnostics_root=$OUT/diag" \
    "+save_per_step=true" "+per_step_file=$OUT/ps_$n.csv" \
    "results_file=$OUT/cell_$n.csv" "hydra.run.dir=$OUT/hydra_$n" > "$OUT/cell_$n.log" 2>&1
  rc=$?; kill $MON 2>/dev/null; wait $MON 2>/dev/null
  echo "CELL $n rc=$rc wall=$(( $(date +%s)-t0 ))s peakMiB=$(cat $OUT/peak_$n 2>/dev/null)"
  grep -hE "\[NS\]" "$OUT/cell_$n.log" | tail -1 | cut -c1-260
  [ $rc -ne 0 ] && grep -hE 'Error|error:' "$OUT/cell_$n.log" | tail -3 | cut -c1-260
done
