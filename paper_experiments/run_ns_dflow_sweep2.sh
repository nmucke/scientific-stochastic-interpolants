#!/bin/bash
# =============================================================================
# D-Flow SGLD follow-up sweep #2 -- Navier-Stokes, ALL 4 scenarios.
#
# Sweep #1 finding: the step size eta is the BINDING knob on this NS setup --
# eta=1e-2 beat eta=5e-2 by 2-4x RMSE and ~10-25x KL in every scenario, and the
# lambda ranking there was confounded (every lambda cell ran at the bad eta=5e-2).
#
# This follow-up re-centres on eta and re-sweeps lambda at each eta to DE-CONFOUND:
#   eta    in {2e-2, 1e-2, 5e-3, 2e-3}   (bracket the eta=1e-2 winner, go lower)
#   lambda in {0, 5e-6, 1e-4}            (unconfounded lambda ranking at good eta)
#   s      = 1e-3 (fixed);  K = 50 (matched budget for ranking)
# Full grid: 4 eta x 3 lambda x 4 scenarios = 48 cells.
#
# Settings match sweep #1 (compute-efficient): E=8, ONE trajectory, ONE seed, 2
# assimilation steps (num_physical_steps=7 = 5 history + 2), trained flow_matching
# weights (require_weights=true), midpoint-6 transport rollout (ode_steps).
#
# After this lands, re-validate the single best (eta, lambda) at higher K (e.g.
# K=300) to check how much of the eta=1e-2 win is a K=50 cold-start-transient
# effect -- done separately once the winner is known.
#
# Filenames reuse sweep #1's convention so aggregate_dflow_sweep.py works directly:
#   .venv/bin/python paper_experiments/aggregate_dflow_sweep.py --dir paper_experiments/results/dflow_sweep2
#
# Launch:  setsid nohup bash paper_experiments/run_ns_dflow_sweep2.sh >run_ns_dflow_sweep2.log 2>&1 & disown
# Preview: DRY_RUN=1 bash paper_experiments/run_ns_dflow_sweep2.sh
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
PY=.venv/bin/python
OUT=paper_experiments/results/dflow_sweep2
mkdir -p "$OUT"

# ---- fixed run settings (identical to sweep #1) -----------------------------
ENSEMBLE=8            # ensemble size (E)
NP_STEPS=7           # 5 history + 2 assimilation steps => n_assim=2
NUM_STEPS=50         # global M; ignored by D-Flow's rollout (uses ode_steps=6)
TRAJ=1               # ONE test trajectory
SEED=0               # ONE seed
REF_E=8              # cheap on-the-fly SI-SDE reference for KL-at-points
DEVICE="${DFLOW_SWEEP_DEVICE:-cuda}"
REQUIRE_WEIGHTS=true                        # trained flow_matching weights
NSTEPS="${DFLOW_NSTEPS_SWEEP:-50}"   # K = num_optim_steps
DRY_RUN="${DRY_RUN:-0}"

# ---- scenarios: ALL four ----------------------------------------------------
SCENARIOS=("32^2->128^2" "16^2->128^2" "sparse 5%" "sparse 1.5625%")

# ---- grid: eta (primary) x lambda, s fixed ----------------------------------
ETAS=(2e-2 1e-2 5e-3 2e-3)
LAMBDAS=(0 5e-6 1e-4)
S=1e-3

slug() { echo "$1" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//'; }

run_cell() {
    local scen="$1" eta="$2" lam="$3"
    local ss; ss=$(slug "$scen")
    # Keep sweep #1's filename order (lam,s,eta,K) so the aggregator regex matches.
    local outfile="$OUT/dflow_${ss}_lam${lam}_s${S}_eta${eta}_K${NSTEPS}.csv"
    if [ -f "$outfile" ]; then
        echo "[dflow-sweep2] SKIP (exists) $outfile" | tee -a "$OUT/run.log"
        return
    fi
    echo "[dflow-sweep2] scen='$scen' eta=$eta lambda=$lam s=$S K=$NSTEPS start $(date +%T)" | tee -a "$OUT/run.log"
    if [ "$DRY_RUN" = "1" ]; then
        echo "  DRY_RUN: DFLOW_STEP_SIZE=$eta DFLOW_LAMBDA=$lam DFLOW_NOISE_SCALE=$S DFLOW_NSTEPS=$NSTEPS -> $outfile"
        return
    fi
    DFLOW_LAMBDA="$lam" DFLOW_NOISE_SCALE="$S" DFLOW_STEP_SIZE="$eta" DFLOW_NSTEPS="$NSTEPS" \
    $PY -u paper_experiments/run.py case=navier_stokes \
        +test_index=$TRAJ seeds=[$SEED] \
        likelihood_mode=dps_jacobian_free \
        '+ns_methods=["D-Flow SGLD"]' \
        "+ns_scenarios=[\"$scen\"]" \
        ensemble_size=$ENSEMBLE num_steps=$NUM_STEPS case.num_physical_steps=$NP_STEPS \
        case.reference_ensemble_size=$REF_E \
        +save_states=false \
        case.require_weights=$REQUIRE_WEIGHTS case.device=$DEVICE \
        results_file="$outfile" \
        >> "$OUT/run.log" 2>&1 \
        && echo "[dflow-sweep2] OK $(date +%T)  -> $outfile" | tee -a "$OUT/run.log" \
        || echo "[dflow-sweep2] CRASHED $(date +%T)  ($ss eta=$eta lam=$lam)" | tee -a "$OUT/run.log"
}

echo "[dflow-sweep2] start $(date) | E=$ENSEMBLE np=$NP_STEPS (n_assim=2) K=$NSTEPS traj=$TRAJ device=$DEVICE" | tee -a "$OUT/run.log"
echo "[dflow-sweep2] 48 runs total (4 eta x 3 lambda x 4 scenarios). DRY_RUN=$DRY_RUN" | tee -a "$OUT/run.log"

for SCEN in "${SCENARIOS[@]}"; do
    echo "[dflow-sweep2] ===== scenario '$SCEN' =====" | tee -a "$OUT/run.log"
    for ETA in "${ETAS[@]}"; do
        for LAM in "${LAMBDAS[@]}"; do
            run_cell "$SCEN" "$ETA" "$LAM"
        done
    done
done
echo "[dflow-sweep2] ALL DONE $(date)" | tee -a "$OUT/run.log"
