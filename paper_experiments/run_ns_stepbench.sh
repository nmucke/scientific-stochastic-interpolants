#!/bin/bash
# =============================================================================
# Navier--Stokes STEP-COUNT BENCHMARK  (set up 2026-06-29)
#
# Sweeps the SDE/ODE integration step count M in {50,100,250,500} for the 8
# step-based generative samplers, over ALL 4 observation scenarios and ALL 5 test
# trajectories. Metrics are computed for every run (5 x 128 = 640 runs); the
# metrics-vs-M figure averages over the 5 trajectories. To bound storage, the
# posterior STATES (and the per-step metric curves inside the .npz) are saved for
# TRAJECTORY 1 ONLY.
#
#   Runs:        8 methods x 4 M x 4 scenarios x 5 traj = 640  (metrics, CSV).
#   Saved states: traj 1 only -> 8 x 4 x 4 = 128 posterior trajectories,
#                 ~63 MB each -> ~8.2 GB total.
# Metrics (RMSE, CRPS, spread-skill, KL-to-EnKF(E=1000)) are in the CSVs for all
# 640 runs AND, per time step, inside each traj-1 .npz (per_step_* arrays).
# Timing: every CSV has a `seconds` (s/step) row + NFE; each .npz stores
# seconds_per_step + seconds_total.
#
# D-Flow SGLD is EXCLUDED (optimiser; its "steps" are Langevin iterations K=200,
# cost K*M -> ~25 h/traj at M=500, infeasible for an M-sweep). It appears in the
# headline M=50 table instead.
# Classical filters are M-independent; the E=1000 EnKF is the KL reference
# (already saved under multitraj/states/traj1/gt).
#
# Launch detached (GPU):
#   setsid nohup bash paper_experiments/run_ns_stepbench.sh >/dev/null 2>&1 &
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
PY=.venv/bin/python
SC=/tmp/claude-12096798/-export-scratch1-ntm-postdoc-scientific-stochastic-interpolants/ee14f21c-3ae9-47e8-aae9-b2bd89a70d12/scratchpad
OUT=paper_experiments/results/stepbench
MT=paper_experiments/results/multitraj
LOG=$SC/run_ns_stepbench.log

# ---- knobs ------------------------------------------------------------------
STEPS="50 100 250 500"
TRAJ="1 2 3 4 5"                          # run all 5; metrics for all
SAVE_TRAJ=1                              # save STATES for this trajectory only
SCENARIOS=("32^2->128^2" "16^2->128^2" "sparse 5%" "sparse 1.5625%")
METHODS='["Ours (SI-SDE)","Ours (FM-SDE)","Ours (FM-ODE)","FlowDAS","Guided FM (FIG)","Guided FM (OT-ODE)","SDA","SURGE"]'
# -----------------------------------------------------------------------------

mkdir -p "$OUT/csv" "$OUT/states"
echo "[stepbench] start $(date +%T) | traj=[$TRAJ] save_states_traj=$SAVE_TRAJ | 8 methods x 4 M x ${#SCENARIOS[@]} scen x 5 traj = 640 runs" > "$LOG"

slug() { echo "$1" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//'; }

for M in $STEPS; do
  echo "[stepbench] ===== M=$M start $(date +%T) =====" >> "$LOG"
  for N in $TRAJ; do
    if [ "$N" = "$SAVE_TRAJ" ]; then SAVE=true; else SAVE=false; fi
    for SCEN in "${SCENARIOS[@]}"; do
      SS=$(slug "$SCEN")
      echo "[stepbench] M$M traj$N scen='$SCEN' (save_states=$SAVE) start $(date +%T)" >> "$LOG"
      $PY -u paper_experiments/run.py case=navier_stokes +test_index=$N seeds=[0] \
        likelihood_mode=dps_jacobian_free \
        "+ns_methods=$METHODS" \
        "+ns_scenarios=[\"$SCEN\"]" \
        ensemble_size=64 num_steps=$M case.num_physical_steps=15 \
        "+kl_reference_states=$MT/states/traj$N/gt" \
        +save_states=$SAVE "+states_root=$OUT/states/traj$SAVE_TRAJ/M$M" \
        case.require_weights=true case.device=cuda \
        results_file="$OUT/csv/stepbench_${SS}_M${M}_traj${N}.csv" >> "$LOG" 2>&1 \
        && echo "[stepbench] M$M traj$N scen='$SCEN' OK $(date +%T)" >> "$LOG" \
        || echo "[stepbench] M$M traj$N scen='$SCEN' CRASHED $(date +%T)" >> "$LOG"
    done
  done
  echo "[stepbench] ===== M=$M done $(date +%T) =====" >> "$LOG"
done
echo "[stepbench] ALL DONE $(date +%T)" >> "$LOG"
