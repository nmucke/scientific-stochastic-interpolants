#!/bin/bash
# True IC (NP=20: 5 history + 15 DA), 16^2->128^2, traj 11, M=100, E=64, seed 0: re-run the cells whose
# traj-11 states were never saved, with states saved. Settings as run_ns_grid.sh (KL skipped: states only).
set -u
source "/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments/results/_runs/env.sh"
OUT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"/{metrics,per_step,states/traj11,hydra,logs}
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
cd "$REPO"
cell() {  # cell <id> <args...>
  local id=$1; shift
  [ -f "$OUT/metrics/$id.csv" ] && { echo "SKIP $id"; return; }
  echo "RUN $id $(date +%T)"
  "$PY" -u "$PE/run.py" case=navier_stokes 'seeds=[0]' ensemble_size=64 case.num_physical_steps=20 \
    case.require_weights=true case.device=cuda +test_index=11 +divergence_rmse_threshold=10.0 \
    +save_per_step=true "+per_step_file=$OUT/per_step/$id.csv" "results_file=$OUT/metrics/$id.csv" \
    "hydra.run.dir=$OUT/hydra/$id" '+ns_scenarios=["16^2->128^2"]' num_steps=100 \
    likelihood_mode=dps_jacobian_free +skip_kl_reference=true \
    +save_states=true "+states_root=$OUT/states/traj11" "$@" > "$OUT/logs/$id.log" 2>&1
  echo "END $id rc=$? $(date +%T)"
}
cell 16_2_128_2__M100__traj11__ours_jacfree '+ns_methods=["Ours (SI-SDE)","Ours (DM-SDE)","Ours (FM-ODE)"]'
cell 16_2_128_2__M100__traj11__baselines_surge_flowdas_fig '+ns_methods=["SURGE (FlowDAS)","Guided FM (FIG)"]'
echo ALLDONE
