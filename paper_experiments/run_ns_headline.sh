#!/bin/bash
# =============================================================================
# Navier--Stokes HEADLINE 5-TRAJECTORY GRID  (written 2026-06-30)
#
# Runs ALL 8 step-based generative methods × 4 scenarios × 5 test trajectories
# at the paper's standard settings (E=64, M=50, num_physical_steps=25, seed=0
# per trajectory) and writes per-(scenario × traj) CSVs for resilience.
#
# D-Flow SGLD is INCLUDED (dflow_sgld.yaml, K=20, ~34 min/cell at n_assim=20)
# but can be excluded with INCLUDE_DFLOW=0 to run it separately after a sweep.
#
# Classical filters (EnKF E=64, LETKF, PF, EnSF) are NOT in this script --
# run them with run_ns_classical.sh (separate, jax-cfd GPU dependency).
#
# Launch detached:
#   setsid nohup bash paper_experiments/run_ns_headline.sh >run_ns_headline.log 2>&1 &
#   disown
# Progress: tail -f run_ns_headline.log
# Salvage: adapt paper_experiments/cases/navier_stokes/reconstruct_csv.py if
#   the run dies before writing a CSV (per-cell [NS]...{metrics} lines survive).
#
# After all CSVs land, merge into all_results.csv and regenerate tables:
#   bash paper_experiments/merge_and_tables.sh
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
PY=.venv/bin/python
OUT=paper_experiments/results/headline
MT=paper_experiments/results/multitraj

# ---- knobs ------------------------------------------------------------------
TRAJ="1 2 3 4 5"
SCENARIOS=("32^2->128^2" "16^2->128^2" "sparse 5%" "sparse 1.5625%")
# 8 step-based methods (D-Flow can be excluded with INCLUDE_DFLOW=0).
INCLUDE_DFLOW="${INCLUDE_DFLOW:-1}"
METHODS_BASE='["Ours (SI-SDE)","Ours (FM-SDE)","Ours (FM-ODE)","FlowDAS","Guided FM (FIG)","Guided FM (OT-ODE)","SDA","SURGE"]'
METHODS_WITH_DFLOW='["Ours (SI-SDE)","Ours (FM-SDE)","Ours (FM-ODE)","FlowDAS","Guided FM (FIG)","Guided FM (OT-ODE)","D-Flow SGLD","SDA","SURGE"]'
# -----------------------------------------------------------------------------

if [ "$INCLUDE_DFLOW" = "1" ]; then
    METHODS="$METHODS_WITH_DFLOW"
    echo "[headline] Including D-Flow SGLD (K=20; ~34 min/cell)"
else
    METHODS="$METHODS_BASE"
    echo "[headline] Excluding D-Flow SGLD (run separately after sweep)"
fi

mkdir -p "$OUT"
echo "[headline] start $(date) | M=50 E=64 np=25 | traj=[$TRAJ] | ${#SCENARIOS[@]} scenarios | methods=$METHODS" | tee -a "$OUT/run.log"

slug() { echo "$1" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//'; }

for N in $TRAJ; do
    echo "[headline] ===== traj$N start $(date +%T) =====" | tee -a "$OUT/run.log"
    for SCEN in "${SCENARIOS[@]}"; do
        SS=$(slug "$SCEN")
        OUTFILE="$OUT/headline_${SS}_traj${N}.csv"
        if [ -f "$OUTFILE" ]; then
            echo "[headline] SKIP traj$N scen='$SCEN' (already exists: $OUTFILE)" | tee -a "$OUT/run.log"
            continue
        fi
        echo "[headline] traj$N scen='$SCEN' start $(date +%T)" | tee -a "$OUT/run.log"
        $PY -u paper_experiments/run.py case=navier_stokes \
            +test_index=$N seeds=[0] \
            likelihood_mode=dps_jacobian_free \
            "+ns_methods=$METHODS" \
            "+ns_scenarios=[\"$SCEN\"]" \
            ensemble_size=64 num_steps=50 \
            "+kl_reference_states=$MT/states/traj${N}/gt" \
            +save_states=false \
            case.require_weights=true case.device=cuda \
            results_file="$OUTFILE" \
            >> "$OUT/run.log" 2>&1 \
            && echo "[headline] traj$N scen='$SCEN' OK $(date +%T)" | tee -a "$OUT/run.log" \
            || echo "[headline] traj$N scen='$SCEN' CRASHED $(date +%T)" | tee -a "$OUT/run.log"
    done
    echo "[headline] ===== traj$N done $(date +%T) =====" | tee -a "$OUT/run.log"
done
echo "[headline] ALL DONE $(date)" | tee -a "$OUT/run.log"
