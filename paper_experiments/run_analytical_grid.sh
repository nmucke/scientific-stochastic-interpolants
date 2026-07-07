#!/bin/bash
# =============================================================================
# Analytical (linear-Gaussian) REDUCED-GRID master run  (2026-07-01)
#
# Case 1 correctness probe, writing into results/analytical/ (see
# results/README.md). Closed-form: no trained prior, no trajectories, no raw
# states, no per-step curves (a single x^0 -> x^1 assimilation, not an
# autoregressive rollout). Instead of trajectories it averages over the 5-seed
# SEED_LIST in ONE invocation per step count.
#
# GRID
#   seeds     : 0..4 (SEED_LIST, averaged in-run)
#   scenario  : analytical (single joint)
#   steps M   : 50 100 250 500
#   methods   : the full reduced lineup, emitted in ONE run per M --
#               Ours (SI-SDE/DM-SDE/FM-ODE) x {jacfree, shared} (variant column),
#               FlowDAS, SURGE (FlowDAS), SDA, SURGE (SDA), D-Flow SGLD, EnKF, PF.
#   E (n_eval): the driver's fixed 4096-sample estimate (KL / sliced-W2).
#   hparams   : each baseline's per-M hyperparameters come from the `analytical:`
#               matrix in configs/method/*.yaml (single joint scenario "analytical",
#               columns 50/100/250/500). Cells ship all-null -> the config `default:`
#               is used (a "not tuned yet" warning is logged) until they are filled;
#               fill the YAML cells to tune per M (no code change, no edit here).
#
# Timings (seconds + NFE) are recorded per method/variant row.
#
# Runs on CPU in minutes (2-D closed form) -- no GPU needed.
#   bash paper_experiments/run_analytical_grid.sh
# Track: .venv/bin/python paper_experiments/status.py --case analytical
# Env overrides: STEPS, SEEDS.
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
PY=.venv/bin/python
ROOT="${ROOT:-paper_experiments/results/analytical}"   # override for smoke tests
MET=$ROOT/metrics
mkdir -p "$MET"

STEPS="${STEPS:-50 100 250 500}"
SEEDS="${SEEDS:-[0,1,2,3,4]}"

LOG="$ROOT/run_analytical_grid.log"
echo "[angrid] START $(date) | seeds=$SEEDS steps=[$STEPS]" | tee -a "$LOG"

for M in $STEPS; do
  OUTFILE="$MET/analytical__M${M}.csv"
  if [ -f "$OUTFILE" ]; then echo "[angrid] SKIP (exists) $OUTFILE" | tee -a "$LOG"; continue; fi
  echo "[angrid] RUN  M=$M $(date +%T)" | tee -a "$LOG"
  $PY -u paper_experiments/run.py case=analytical seeds=$SEEDS num_steps=$M \
      results_file="$OUTFILE" >> "$LOG" 2>&1 \
    && echo "[angrid] OK   M=$M $(date +%T)" | tee -a "$LOG" \
    || echo "[angrid] FAIL M=$M $(date +%T)" | tee -a "$LOG"
done
echo "[angrid] ALL DONE $(date)" | tee -a "$LOG"
