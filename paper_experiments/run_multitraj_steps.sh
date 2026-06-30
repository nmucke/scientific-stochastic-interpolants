#!/bin/bash
# =============================================================================
# Step-count benchmark for the generative models (multi-trajectory).
#
# Re-runs the 7 generative posteriors over all 5 test trajectories at
# num_steps in {100, 500} SDE/ODE integration steps (the current multitraj run
# is the num_steps=50 "few-steps" point). Goal: show Ours stays strong with few
# steps while baselines need many -> Ours @ few steps ~ baselines @ many steps.
#
# Generative only: num_steps does NOT apply to the true-solver EnKF/PF (their
# forecast is the jax-cfd solver). KL is vs each trajectory's E=1000 EnKF
# ground truth in multitraj/states/traj<N>/gt (produced by run_multitraj.sh).
#
# Metrics only (NO +save_states): the step benchmark needs the CSVs, and saving
# 7x5x4x2 ~ 280 ensembles would cost ~20 GB. Figures stay at the M=50 point.
#
# M=100 runs first (cheaper, lands first); M=500 second (the long pole).
# Per-cell exit-code logging (&& OK || CRASHED). Launch detached:
#   setsid nohup bash paper_experiments/run_multitraj_steps.sh >/dev/null 2>&1 &
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
PY=.venv/bin/python
SC=/tmp/claude-12096798/-export-scratch1-ntm-postdoc-scientific-stochastic-interpolants/ee14f21c-3ae9-47e8-aae9-b2bd89a70d12/scratchpad
LOG=$SC/run_multitraj_steps.log
MT=paper_experiments/results/multitraj
TRAJ="1 2 3 4 5"
STEPS="100 500"

echo "[steps] start $(date +%T)" > $LOG
for M in $STEPS; do
  echo "[steps] ===== num_steps=$M start $(date +%T) =====" >> $LOG
  for N in $TRAJ; do
    echo "[steps] M$M gen traj$N start $(date +%T)" >> $LOG
    $PY -u paper_experiments/run.py case=navier_stokes +test_index=$N seeds=[0] \
      likelihood_mode=dps_jacobian_free \
      '+ns_methods=["Ours (SI-SDE)","Ours (FM-SDE)","Ours (FM-ODE)","FlowDAS","Guided FM","Guided diffusion","SDA"]' \
      ensemble_size=64 num_steps=$M case.num_physical_steps=15 \
      "+kl_reference_states=$MT/states/traj$N/gt" \
      case.require_weights=true case.device=cuda \
      results_file=$MT/csv/gen_traj${N}_steps${M}.csv >> $LOG 2>&1 \
      && echo "[steps] M$M gen traj$N OK $(date +%T)" >> $LOG \
      || echo "[steps] M$M gen traj$N CRASHED $(date +%T)" >> $LOG
  done
  echo "[steps] ===== num_steps=$M done $(date +%T) =====" >> $LOG
done
echo "[steps] ALL DONE $(date +%T)" >> $LOG
grep -E "^\[steps\] (M|=)" $LOG | tail -40 >> $LOG 2>/dev/null
