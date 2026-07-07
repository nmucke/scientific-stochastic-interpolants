#!/bin/bash
# =============================================================================
# Navier--Stokes KL-reference generation  (2026-07-01)
#
# The E=1000 NON-localized (global) EnKF ground-truth posterior, per trajectory
# and scenario -- the reference ensemble every generative method's KL-at-points
# is measured against (spec Section 9). Saved as raw states under
#   results/navier_stokes/reference/traj<N>/gt/
# so run_ns_grid.sh can point +kl_reference_states at that dir.
#
# EXPENSIVE (E=1000 true jax-cfd solver, 5000 sub-steps/forecast; ~30 min/cell).
# Run this FIRST (or in parallel on a spare GPU) before/with run_ns_grid.sh;
# without it the KL column is NaN but every other metric still lands.
#
# skip_kl_reference: this run PRODUCES the KL reference, so the driver's own
# KL-reference draw (an SI-SDE sample that defaults to the hours-per-cell
# ``inflated`` likelihood mode) is skipped; kl_points is NaN in ref_*.csv by
# design.
#
# Launch detached:
#   setsid nohup bash paper_experiments/run_ns_reference.sh >run_ns_reference.log 2>&1 & disown
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
PY=.venv/bin/python
# Persistent XLA compile cache: the 5000-sub-step forecast scan jits once ever
# instead of ~2-3 min in each of the 20 per-cell processes.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/.jax_cache}"
ROOT=paper_experiments/results/navier_stokes
REF=$ROOT/reference
mkdir -p "$REF"

TRAJ="${TRAJ:-1 2 3 4 5}"
NP="${NP:-20}"
E_REF="${E_REF:-1000}"
DEVICE="${DEVICE:-cuda}"
if [ -n "${SCENARIOS:-}" ]; then IFS='|' read -r -a SCEN_ARR <<< "$SCENARIOS";
else SCEN_ARR=("16^2->128^2" "32^2->128^2" "sparse 5%" "sparse 1.5625%"); fi

LOG="$ROOT/run_ns_reference.log"
echo "[nsref] START $(date) | E=$E_REF NP=$NP dev=$DEVICE | traj=[$TRAJ]" | tee -a "$LOG"

for N in $TRAJ; do
  echo "[nsref] ===== traj$N $(date +%T) =====" | tee -a "$LOG"
  for SCEN in "${SCEN_ARR[@]}"; do
    GTDIR="$REF/traj${N}/gt"
    # EnKF saves one state file per scenario in GTDIR; skip if this scenario's
    # reference already exists (glob for the scenario slug).
    SS=$(echo "$SCEN" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//')
    if ls "$GTDIR"/*"${SS}"*.npz >/dev/null 2>&1; then
      echo "[nsref] SKIP traj$N '$SCEN' (exists)" | tee -a "$LOG"; continue
    fi
    echo "[nsref] RUN  traj$N '$SCEN' $(date +%T)" | tee -a "$LOG"
    $PY -u paper_experiments/run.py case=navier_stokes +test_index=$N seeds=[0] \
        '+ns_methods=["EnKF"]' "+ns_scenarios=[\"$SCEN\"]" \
        ensemble_size=$E_REF num_steps=50 case.num_physical_steps=$NP \
        case.reference_ensemble_size=8 \
        +skip_kl_reference=true \
        +save_states=true "+states_root=$GTDIR" \
        case.require_weights=true case.device=$DEVICE \
        results_file=$ROOT/metrics/ref_${SS}__traj${N}.csv >> "$LOG" 2>&1 \
      && echo "[nsref] OK   traj$N '$SCEN' $(date +%T)" | tee -a "$LOG" \
      || echo "[nsref] FAIL traj$N '$SCEN' $(date +%T)" | tee -a "$LOG"
  done
done
echo "[nsref] ALL DONE $(date)" | tee -a "$LOG"
