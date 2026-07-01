#!/bin/bash
# =============================================================================
# D-Flow SGLD hyperparameter sweep -- Navier-Stokes, ALL 4 scenarios.
#
# Prepared (not launched). Informed by the paper's own ranking of which
# hyperparameters matter (Parikh, Chen & Wang, arXiv:2602.21469): the
# source-prior weight lambda is BY FAR the dominant knob -- it is the single
# balance parameter in the objective (Eq. 31, min ||y-F(x1)||^2 + lambda||x0||^2),
# the whole Discussion (Sec 4.1, Figs 9-10) is about its effect, and its tuned
# value spans ~5 orders of magnitude across the paper's cases (0.1 -> 5e-6). The
# noise scale s (posterior-sampling vs MAP) is second; the step size eta is third;
# the preconditioner (omega, delta), ODE steps, and burn are robust FIXED defaults
# (constant across all paper cases) and are NOT swept.
#
# Sweep structure (coordinate, centred on the paper's "Turb" column, to spend the
# budget where it matters):
#   Phase 1  lambda in {0, 1e-6, 5e-6, 1e-5, 1e-4, 1e-3, 1e-2}   (primary)
#   Phase 2  s      in {1e-3(base), 1e-2}                         (secondary)
#   Phase 3  eta    in {5e-2(base), 1e-2}                         (tertiary)
# => 7 + 1 + 1 = 9 runs/scenario x 4 scenarios = 36 runs.
#
# Compute-efficient settings (per request): ensemble_size=8, ONE trajectory, ONE
# seed, 2 assimilation steps. The FM prior needs len_field_history=5 seed steps,
# and the scored steps are n_assim = num_physical_steps - len_field_history, so
# num_physical_steps=7 gives EXACTLY 2 assimilation steps with the 5-step history
# included.
#
# Weights: D-Flow SGLD samples the ALREADY-TRAINED flow-matching prior
# (case.checkpoints.fm_run = flow_matching -> checkpoints/stochastic_navier_stokes/
# flow_matching/model.pth), i.e. the same weights every other FM experiment uses.
# require_weights=true makes the loader hard-fail rather than silently fall back to
# random weights. (The KL-at-points reference is a small SI-SDE self-draw and reuses
# the trained stochastic_interpolant_small prior, also already on disk.)
#
# K (num_optim_steps) is the dominant cost lever. It is REDUCED to 50 here (the
# paper uses 300 KS / 600 turb) so the sweep ranks configs cheaply; re-validate the
# winning (lambda, s, eta) at full K afterwards. D-Flow's transport rollout is
# always the 6-step MIDPOINT integrator (ode_steps; Table D.3, D-Flow / D-Flow SGLD
# column), independent of both K and the global num_steps=M.
#
# Hyperparameters are injected via the env-var overrides the pipeline reads at
# build time (top precedence over the YAML tables): DFLOW_LAMBDA, DFLOW_NOISE_SCALE,
# DFLOW_STEP_SIZE, DFLOW_NSTEPS. The tracked dflow_sgld.yaml is left untouched.
#
# Launch (when ready):
#   setsid nohup bash paper_experiments/run_ns_dflow_sweep.sh >run_ns_dflow_sweep.log 2>&1 &
#   disown
# Preview without running:   DRY_RUN=1 bash paper_experiments/run_ns_dflow_sweep.sh
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
PY=.venv/bin/python
OUT=paper_experiments/results/dflow_sweep
mkdir -p "$OUT"

# ---- fixed run settings (compute-efficient) ---------------------------------
ENSEMBLE=8            # ensemble size (E)
NP_STEPS=7           # 5 history (len_field_history) + 2 assimilation steps => n_assim=2
NUM_STEPS=50         # global M; ignored by D-Flow's rollout (uses ode_steps=6)
TRAJ=1               # ONE test trajectory
SEED=0               # ONE seed
REF_E=8              # cheap on-the-fly SI-SDE reference for KL-at-points
DEVICE="${DFLOW_SWEEP_DEVICE:-cuda}"        # cuda is available on this box
REQUIRE_WEIGHTS=true                        # use the trained flow_matching weights
NSTEPS="${DFLOW_NSTEPS_SWEEP:-50}"    # K = num_optim_steps (reduced for sweep speed)
DRY_RUN="${DRY_RUN:-0}"

# ---- scenarios: ALL four ----------------------------------------------------
SCENARIOS=("32^2->128^2" "16^2->128^2" "sparse 5%" "sparse 1.5625%")

# ---- swept axes, ordered by importance (lambda >> s > eta) -------------------
# Baseline = paper "Turb" column (best prior guess for this fluid case).
BASE_LAMBDA=5e-6
BASE_S=1e-3
BASE_ETA=5e-2
LAMBDAS=(0 1e-6 5e-6 1e-5 1e-4 1e-3 1e-2)   # THE key knob (Eq. 31) -> widest sweep
NOISE_SCALES=(1e-3 1e-2)                     # s: posterior-exploration strength
STEP_SIZES=(5e-2 1e-2)                       # eta: pSGLD step size

slug() { echo "$1" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//'; }

run_cell() {
    local scen="$1" lam="$2" s="$3" eta="$4"
    local ss; ss=$(slug "$scen")
    local tag="lam${lam}_s${s}_eta${eta}_K${NSTEPS}"
    local outfile="$OUT/dflow_${ss}_${tag}.csv"
    if [ -f "$outfile" ]; then
        echo "[dflow-sweep] SKIP (exists) $outfile" | tee -a "$OUT/run.log"
        return
    fi
    echo "[dflow-sweep] scen='$scen' lambda=$lam s=$s eta=$eta K=$NSTEPS start $(date +%T)" | tee -a "$OUT/run.log"
    if [ "$DRY_RUN" = "1" ]; then
        echo "  DRY_RUN: DFLOW_LAMBDA=$lam DFLOW_NOISE_SCALE=$s DFLOW_STEP_SIZE=$eta DFLOW_NSTEPS=$NSTEPS -> $outfile"
        return
    fi
    DFLOW_LAMBDA="$lam" DFLOW_NOISE_SCALE="$s" DFLOW_STEP_SIZE="$eta" DFLOW_NSTEPS="$NSTEPS" \
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
        && echo "[dflow-sweep] OK $(date +%T)  -> $outfile" | tee -a "$OUT/run.log" \
        || echo "[dflow-sweep] CRASHED $(date +%T)  ($ss $tag)" | tee -a "$OUT/run.log"
}

echo "[dflow-sweep] start $(date) | E=$ENSEMBLE np=$NP_STEPS (n_assim=2, hist=5) K=$NSTEPS traj=$TRAJ device=$DEVICE" | tee -a "$OUT/run.log"
echo "[dflow-sweep] 36 runs total (9/scenario x 4). DRY_RUN=$DRY_RUN" | tee -a "$OUT/run.log"

for SCEN in "${SCENARIOS[@]}"; do
    echo "[dflow-sweep] ===== scenario '$SCEN' =====" | tee -a "$OUT/run.log"
    # Phase 1: lambda sweep (primary), s & eta at baseline.
    for LAM in "${LAMBDAS[@]}"; do
        run_cell "$SCEN" "$LAM" "$BASE_S" "$BASE_ETA"
    done
    # Phase 2: noise-scale refinement at baseline lambda/eta (skip baseline dup).
    for S in "${NOISE_SCALES[@]}"; do
        [ "$S" = "$BASE_S" ] && continue
        run_cell "$SCEN" "$BASE_LAMBDA" "$S" "$BASE_ETA"
    done
    # Phase 3: step-size refinement at baseline lambda/s (skip baseline dup).
    for ETA in "${STEP_SIZES[@]}"; do
        [ "$ETA" = "$BASE_ETA" ] && continue
        run_cell "$SCEN" "$BASE_LAMBDA" "$BASE_S" "$ETA"
    done
done
echo "[dflow-sweep] ALL DONE $(date)" | tee -a "$OUT/run.log"
