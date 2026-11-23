#!/bin/bash
# Re-run of the diverged poor-IC cell: Ours (SI-SDE), shared_jac10, FC r=20 (tag fc25_8797d2ae),
# 32^2->128^2, traj 11, settings identical to ext_cells.tsv (job 1305160), different seeds, states saved.
# usage: run.sh <seed> <num_physical_steps> <outdir>
set -u
source "/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments/results/_runs/env.sh"
SEED=$1; NP=$2; OUT=$3
mkdir -p "$OUT"/{metrics,per_step,states/traj11,forecast_correction,hydra,logs}
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
cd "$REPO"
ID=32_2_128_2__M100__traj11__ours_shared_jac10__fc25_8797d2ae__si_sde__seed${SEED}
"$PY" -u "$PE/run.py" case=navier_stokes "seeds=[$SEED]" case.require_weights=true case.device=cuda \
  +test_index=11 +divergence_rmse_threshold=10.0 \
  "results_file=$OUT/metrics/$ID.csv" "hydra.run.dir=$OUT/hydra/$ID" \
  +save_per_step=true "+per_step_file=$OUT/per_step/$ID.csv" \
  '+ns_methods=["Ours (SI-SDE)"]' '+ns_scenarios=["32^2->128^2"]' +ic_index=15 \
  case.num_physical_steps=$NP +skip_kl_reference=true ensemble_size=64 num_steps=100 \
  likelihood_mode=inflated_shared +jacobian_refresh_every=10 +variant_override=shared_jac10 \
  case.forecast_correction.enabled=true case.forecast_correction.forecast_num_steps=25 \
  case.forecast_correction.localisation_radius=20 case.forecast_correction.diagnostics=true \
  "case.forecast_correction.diagnostics_root=$OUT/forecast_correction" \
  +save_states=true "+states_root=$OUT/states/traj11" > "$OUT/logs/$ID.log" 2>&1
echo "exit $? seed $SEED $(date)" >> "$OUT/logs/$ID.log"
