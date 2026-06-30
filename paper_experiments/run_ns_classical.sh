#!/bin/bash
# =============================================================================
# Navier--Stokes CLASSICAL BASELINE HEADLINE GRID  (written 2026-06-30)
#
# Runs EnKF (E=64 localized, r=20), LETKF, Particle Filter, and Ensemble Score
# Filter over the sparse scenarios (classical methods are not run for super-res:
# localization is incoherent for block-average super-res -- run non-localized or
# omit those columns).
#
# The E=1000 non-localized EnKF (ground-truth reference) runs separately via
# run_enkf_groundtruth.sh (or was already run under multitraj/states/trajN/gt/).
#
# All classical methods use the jax-cfd 256^2 true solver (INNER_STEPS=5000 fixed).
# jax runs on GPU by default (XLA_PYTHON_CLIENT_PREALLOCATE=false).
# Force CPU with ENKF_JAX_PLATFORM=cpu (then reduce E).
#
# Launch detached:
#   setsid nohup bash paper_experiments/run_ns_classical.sh >run_ns_classical.log 2>&1 &
#   disown
# Progress: tail -f run_ns_classical.log
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
PY=.venv/bin/python
OUT=paper_experiments/results/headline

TRAJ="1 2 3 4 5"
# Classical: sparse scenarios only (no super-res)
SCENARIOS=("sparse 5%" "sparse 1.5625%")
CLASSICAL_METHODS='["EnKF","LETKF","Particle filter","Ensemble score filter"]'

mkdir -p "$OUT"
echo "[classical] start $(date) | E=64 r=20 | sparse scenarios | traj=[$TRAJ]" | tee -a "$OUT/run_classical.log"

slug() { echo "$1" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//'; }

for N in $TRAJ; do
    echo "[classical] ===== traj$N start $(date +%T) =====" | tee -a "$OUT/run_classical.log"
    for SCEN in "${SCENARIOS[@]}"; do
        SS=$(slug "$SCEN")
        OUTFILE="$OUT/classical_${SS}_traj${N}.csv"
        if [ -f "$OUTFILE" ]; then
            echo "[classical] SKIP traj$N scen='$SCEN' (exists)" | tee -a "$OUT/run_classical.log"
            continue
        fi
        echo "[classical] traj$N scen='$SCEN' start $(date +%T)" | tee -a "$OUT/run_classical.log"
        $PY -u paper_experiments/run.py case=navier_stokes \
            +test_index=$N seeds=[0] \
            likelihood_mode=dps_jacobian_free \
            "+ns_methods=$CLASSICAL_METHODS" \
            "+ns_scenarios=[\"$SCEN\"]" \
            ensemble_size=64 num_steps=20 \
            +enkf_localization_radius=20 \
            +save_states=false \
            case.require_weights=true case.device=cuda \
            results_file="$OUTFILE" \
            >> "$OUT/run_classical.log" 2>&1 \
            && echo "[classical] traj$N scen='$SCEN' OK $(date +%T)" | tee -a "$OUT/run_classical.log" \
            || echo "[classical] traj$N scen='$SCEN' CRASHED $(date +%T)" | tee -a "$OUT/run_classical.log"
    done
    echo "[classical] ===== traj$N done $(date +%T) =====" | tee -a "$OUT/run_classical.log"
done
echo "[classical] ALL DONE $(date)" | tee -a "$OUT/run_classical.log"
