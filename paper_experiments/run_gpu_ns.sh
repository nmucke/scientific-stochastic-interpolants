#!/usr/bin/env bash
# =============================================================================
# Full-scale Navier-Stokes experiment runner (GPU machine).
# Runs: sanity gate -> headline grid -> ablation -> table regeneration.
# Read paper_experiments/HANDOFF_GPU.md first. Edit the variables below, then:
#     bash paper_experiments/run_gpu_ns.sh
#
# PREREQUISITES (see HANDOFF_GPU.md sections 3 and 6.1):
#   * Real trained weights at checkpoints/stochastic_navier_stokes/<si_run>/model.pth
#     (and <fm_run>/model.pth once an FM prior is trained -- there is none yet).
#   * Set checkpoints.si_run / checkpoints.fm_run / require_weights in
#     paper_experiments/configs/case/navier_stokes.yaml (or override on the CLI below).
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

# ---- knobs ------------------------------------------------------------------
DEVICE="cuda"
MODE="inflated"                       # likelihood_mode: inflated | dps_full | dps_jacobian_free
E=64                                  # ensemble size
M=50                                  # pseudo-time steps
METHODS="si_sde,fm_sde,fm_ode"        # the three samplers (FM rows need a trained FM prior; see 6.1)
MAIN_SCENARIOS="superres_32,sparse_5" # main-table columns
APPX_SCENARIOS="superres_16,sparse_1p5"
ABLATION_SCENARIO="sparse 5%"
REQUIRE_WEIGHTS="true"                # set false to allow the random-weights fallback (smoke only)
PY="python"                           # or .venv/bin/python / uv run python
# -----------------------------------------------------------------------------

echo "==> [1/5] Sanity gate: one cheap cell (SI-SDE, sparse 1.5625%, seed 0)"
echo "    Confirm the log says 'Loaded trained weights ...' (NOT random-weights) and RMSE is physical."
$PY paper_experiments/run.py case=navier_stokes \
  method=si_sde scenario=sparse_1p5 seeds="[0]" \
  ensemble_size="$E" num_steps="$M" likelihood_mode="$MODE" \
  case.require_weights="$REQUIRE_WEIGHTS" case.device="$DEVICE"

echo "==> [2/5] Headline grid: ${METHODS} x {${MAIN_SCENARIOS}}"
$PY paper_experiments/run.py --multirun case=navier_stokes \
  method="$METHODS" scenario="$MAIN_SCENARIOS" \
  ensemble_size="$E" num_steps="$M" likelihood_mode="$MODE" \
  case.require_weights="$REQUIRE_WEIGHTS" case.device="$DEVICE"

echo "==> [3/5] Appendix scenarios: ${METHODS} x {${APPX_SCENARIOS}}"
$PY paper_experiments/run.py --multirun case=navier_stokes \
  method="$METHODS" scenario="$APPX_SCENARIOS" \
  ensemble_size="$E" num_steps="$M" likelihood_mode="$MODE" \
  case.require_weights="$REQUIRE_WEIGHTS" case.device="$DEVICE"

echo "==> [4/5] Ablation (fills tab:ablation): correction axis + g/M/E sweeps on '${ABLATION_SCENARIO}'"
$PY paper_experiments/run.py case=navier_stokes ablation=true \
  ablation_smoke=false ablation_scenario="$ABLATION_SCENARIO" \
  ensemble_size="$E" num_steps="$M" likelihood_mode="$MODE" \
  case.require_weights="$REQUIRE_WEIGHTS" case.device="$DEVICE"

echo "==> [5/5] Regenerate LaTeX tables from the union of all case CSVs"
RES=paper_experiments/results
if [ -f "$RES/analytical_results.csv" ]; then
  head -1 "$RES/analytical_results.csv" > "$RES/all_results.csv"
else
  # fall back to the NS header if analytical has not been run on this box
  head -1 "$RES/navier_stokes_results.csv" > "$RES/all_results.csv"
fi
tail -q -n +2 "$RES"/*_results.csv >> "$RES/all_results.csv"
$PY paper_experiments/make_tables.py --results "$RES/all_results.csv"

echo "==> Done. Tidy results: $RES/all_results.csv ; LaTeX: paper_experiments/generated/tab_*.tex"
echo "    NOTE: 'inflated' mode does ~N_y network-JVPs per pseudo-time step (N_y up to 1024) and the"
echo "    large-E reference ensemble inherits that cost -- budget GPU-hours. See HANDOFF_GPU.md section 7."
