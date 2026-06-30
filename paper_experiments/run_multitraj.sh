#!/bin/bash
# =============================================================================
# Multi-trajectory Navier-Stokes data-assimilation master run.
#
# Computes DA metrics over 5 held-out test trajectories (test indices 1..5),
# ONE seed per trajectory (seeds=[0]; the obs-noise seed obs_seed(case,scenario,
# test_index,seed) already differs per trajectory). Each trajectory is one run
# per phase via +test_index=N; results are written to per-trajectory folders so
# the EnKF ground-truth and conventional-EnKF state files do NOT collide.
#
# NO harness/driver change is needed: driver.py:288 reads +test_index into the
# RunContext, prepare_truth_and_obs(test_index=N) selects test_dataset[N], and
# the KL reference loader (_reference_trajectory, +kl_reference_states=<dir>)
# globs the EnKF refs in that dir per scenario+seed.
#
# Verified (CPU, scratchpad/verify_traj.py): test_dataset[1..5] are 5 distinct
# trajectories (max|diff| ~20-22, none identical; dataset has 20 test trajs).
#
# PHASES (most valuable results land first; CSVs are written at the END of each
# grid, states are written per-cell so partial progress is salvageable):
#   Phase 1  gt   -- EnKF NON-localized E=1000, all 4 scenarios (KL ground-truth
#                    posterior + a conventional method).        -> .../gt
#   Phase 2  gen  -- 7 generative posteriors E=64, all 4 scenarios, KL-vs-EnKF.
#                    Needs Phase-1 gt as the KL reference.      -> .../gen
#   Phase 3  conv -- conventional EnKF(localized)+PF+EnSF E=64, 2 sparse only,
#                    KL-vs-EnKF (expensive baselines, last).    -> .../conv
#
# Per-command exit codes are logged explicitly (... && echo OK || echo CRASHED),
# NOT a bare "echo DONE" (which would mask a crash). Each completed cell also
# emits an "[NS] ... {metrics}" log line, recoverable even if a run dies.
#
# Background runs are killed at session boundaries on this box -- launch with:
#   setsid nohup bash paper_experiments/run_multitraj.sh >/dev/null 2>&1 &
# then tail the log:  tail -f <SC>/run_multitraj.log
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
PY=.venv/bin/python
SC=/tmp/claude-12096798/-export-scratch1-ntm-postdoc-scientific-stochastic-interpolants/ee14f21c-3ae9-47e8-aae9-b2bd89a70d12/scratchpad
LOG=$SC/run_multitraj.log
MT=paper_experiments/results/multitraj

TRAJ="1 2 3 4 5"

echo "[multitraj] start $(date +%T)" > $LOG

# -----------------------------------------------------------------------------
# PHASE 1 -- gt: EnKF NON-localized, E=1000, all 4 scenarios (the KL refs).
#   NO +enkf_localization_radius  => global (non-localized) EnKF.
# -----------------------------------------------------------------------------
echo "[multitraj] ===== PHASE 1 (gt: EnKF non-loc E=1000) start $(date +%T) =====" >> $LOG
for N in $TRAJ; do
  echo "[multitraj] P1 gt traj$N start $(date +%T)" >> $LOG
  $PY -u paper_experiments/run.py case=navier_stokes +test_index=$N seeds=[0] \
    likelihood_mode=dps_jacobian_free \
    '+ns_methods=["EnKF"]' \
    ensemble_size=1000 case.num_physical_steps=15 case.reference_ensemble_size=8 \
    +save_states=true "+states_root=$MT/states/traj$N/gt" \
    case.require_weights=true case.device=cuda \
    results_file=$MT/csv/gt_traj$N.csv >> $LOG 2>&1 \
    && echo "[multitraj] P1 gt traj$N OK $(date +%T)" >> $LOG \
    || echo "[multitraj] P1 gt traj$N CRASHED $(date +%T)" >> $LOG
done
echo "[multitraj] ===== PHASE 1 done $(date +%T) =====" >> $LOG

# -----------------------------------------------------------------------------
# PHASE 2 -- gen: the 7 generative posteriors, E=64, all 4 scenarios.
#   +kl_reference_states points at THIS trajectory's gt dir (Phase 1).
# -----------------------------------------------------------------------------
echo "[multitraj] ===== PHASE 2 (gen: 7 methods E=64) start $(date +%T) =====" >> $LOG
for N in $TRAJ; do
  echo "[multitraj] P2 gen traj$N start $(date +%T)" >> $LOG
  $PY -u paper_experiments/run.py case=navier_stokes +test_index=$N seeds=[0] \
    likelihood_mode=dps_jacobian_free \
    '+ns_methods=["Ours (SI-SDE)","Ours (FM-SDE)","Ours (FM-ODE)","FlowDAS","Guided FM","Guided diffusion","SDA"]' \
    ensemble_size=64 num_steps=50 case.num_physical_steps=15 \
    "+kl_reference_states=$MT/states/traj$N/gt" \
    +save_states=true "+states_root=$MT/states/traj$N/gen" \
    case.require_weights=true case.device=cuda \
    results_file=$MT/csv/gen_traj$N.csv >> $LOG 2>&1 \
    && echo "[multitraj] P2 gen traj$N OK $(date +%T)" >> $LOG \
    || echo "[multitraj] P2 gen traj$N CRASHED $(date +%T)" >> $LOG
done
echo "[multitraj] ===== PHASE 2 done $(date +%T) =====" >> $LOG

# -----------------------------------------------------------------------------
# PHASE 3 -- conv: conventional EnKF(localized) + PF + EnSF, E=64,
#   2 sparse scenarios only. SEPARATE states dir from gt (both write an EnKF
#   file with the same name) -> /conv. KL-vs-EnKF against THIS traj's gt dir.
# -----------------------------------------------------------------------------
echo "[multitraj] ===== PHASE 3 (conv: EnKF-loc+PF+EnSF E=1000 sparse) start $(date +%T) =====" >> $LOG
for N in $TRAJ; do
  echo "[multitraj] P3 conv traj$N start $(date +%T)" >> $LOG
  $PY -u paper_experiments/run.py case=navier_stokes +test_index=$N seeds=[0] \
    likelihood_mode=dps_jacobian_free \
    '+ns_methods=["EnKF","Particle filter","Ensemble score filter"]' \
    '+ns_scenarios=["sparse 5%","sparse 1.5625%"]' \
    ensemble_size=64 +enkf_localization_radius=20 \
    case.num_physical_steps=15 case.reference_ensemble_size=8 \
    "+kl_reference_states=$MT/states/traj$N/gt" \
    +save_states=true "+states_root=$MT/states/traj$N/conv" \
    case.require_weights=true case.device=cuda \
    results_file=$MT/csv/conv_traj$N.csv >> $LOG 2>&1 \
    && echo "[multitraj] P3 conv traj$N OK $(date +%T)" >> $LOG \
    || echo "[multitraj] P3 conv traj$N CRASHED $(date +%T)" >> $LOG
done
echo "[multitraj] ===== PHASE 3 done $(date +%T) =====" >> $LOG

echo "[multitraj] ALL DONE $(date +%T)" >> $LOG
# Salvage summary: print every per-cell metric line at the end of the log.
grep -E "^\[NS\] " $LOG | tail -80 >> $LOG 2>/dev/null
