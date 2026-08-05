#!/bin/bash
# =============================================================================
# NS lambda TRANSFER SANITY CHECK -- DM-SDE / FM-ODE  (2026-07-30)
#
# PURPOSE. The `jacobian_damping` (lambda) tables in configs/method/{si_sde,dm_sde,
# fm_ode}.yaml were tuned on **SI-SDE only** and copied to DM-SDE and FM-ODE. That is
# probably fine -- all three share the same shared-Jacobian machinery, and lambda tames
# the ensemble-mean Jacobian term rather than anything sampler-specific -- but it has
# never been checked. This is the cheap check, NOT a re-tune: it asks only whether
# DM-SDE and FM-ODE peak at the SAME lambda as SI-SDE, not what their exact optimum is.
#
# SI-SDE IS RUN TOO, AND THAT IS THE POINT. The user asked for DM-SDE + FM-ODE, and
# SI-SDE is added as an IN-RUN CONTROL. The si_sde.yaml table was measured at
# E=8 / NP=12 / M=50 / traj1; this check runs E=8 / NP=6 / M=25 / traj15. If DM/FM
# came out preferring a different lambda than the TABLE, there would be no way to tell
# a genuine method difference from the change of conditions. Running all three in the
# SAME cells makes the comparison internally controlled: the only question that matters
# is whether the three curves peak together, and that is answered regardless of how the
# conditions shift the common optimum. All three share the knob, so they go in ONE
# run.py call per cell and cost one startup between them. Set METHODS to override.
#
# WHAT IS SWEPT
#   lambda -> +jacobian_damping   {0.9, 0.95, 1.0}
#   at k=10 -> +jacobian_refresh_every  (per request; also the grid's default group)
# The three lambdas bracket BOTH SI incumbents -- 0.9 for the super-res rows, 0.95 for
# the sparse rows -- and 1.0 probes the divergence cliff (si_sde.yaml records lambda=1.0
# diverging at k=1 on every NS scenario, but a lagged Jacobian suppresses that, so at
# k=10 it is a real candidate rather than a wasted cell).
#
# DELIBERATELY SMALL (this is a sanity check, and urban just showed why cost matters:
# inflated_shared spends one JVP per OBSERVATION per Jacobian refresh, measured there at
# 465 s per DA step against jacfree's 4.74 s, ~98x):
#   * ONE DA step   -- NP=6 = 5 history + 1 (n_assim = NP - len_field_history(5))
#   * ONE M         -- M=25, the cheapest rung of the ladder
#   * TWO scenarios -- one per lambda group: `16^2->128^2` (SI table 0.9, and the
#                      CHEAPEST scenario at 16x16=256 observations) and `sparse 5%`
#                      (SI table 0.95, 819 observations). Skipping `32^2->128^2`
#                      (1024 obs) and `sparse 1.5625%` -- each duplicates the lambda
#                      group of one already covered, at equal or higher cost.
#   * E=8, one seed, ONE trajectory
#   -> 2 scenarios x 3 lambda = 6 shared cells + 2 jacfree reference cells.
#
# +skip_kl_reference=true IS NOT OPTIONAL. Without it the driver draws an SI-SDE
# KL reference per cell in the config-default `inflated` mode -- O(E_ref * N_y) JVPs per
# pseudo-step, diagnosed on run_ns_reference.sh as ~6 h per cell and 93% of its runtime.
# KL is meaningless here anyway; it comes out NaN, which is correct.
#
# TRAJECTORY. test_index indexes the test slice `trajectories_ids (180, 200)` -> 20
# rows, so 0..19 are valid. The paper grid evaluates 10..14, so 15 keeps the check off
# the evaluation trajectories.
#
# READING THE RESULT. Compare the three methods' lambda curves WITHIN a scenario:
#   * same argmin  -> the SI-tuned tables transfer; nothing to do.
#   * different argmin, or DM/FM diverging where SI is stable -> the copied tables are
#     wrong for that method and it needs its own row.
# The ABSOLUTE rmse values are not comparable to the si_sde.yaml table (different NP, M
# and trajectory) -- only the SHAPE and the argmin are.
#
# USAGE
#   bash paper_experiments/run_ns_lambda_check.sh          # the check
#   DRY_RUN=1 bash paper_experiments/run_ns_lambda_check.sh    # preview the matrix
#   DEVICE=cpu SMOKE=1 bash paper_experiments/run_ns_lambda_check.sh   # wiring smoke
#
# Launch detached on the GPU box when a slot is free (do NOT run alongside two other
# GPU jobs):
#   setsid nohup bash paper_experiments/run_ns_lambda_check.sh \
#     >run_ns_lambda_check.log 2>&1 & disown
#
# ENV OVERRIDES: DEVICE, STEPS, SCENARIOS(|-sep), TRAJ, E, NP, KS, LAMBDAS_STR,
#   METHODS, REQUIRE_W, DRY_RUN, SMOKE, OUT, JACFREE_REF, DIV_GUARD.
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
PY=.venv/bin/python

# ---- fixed run settings ------------------------------------------------------
E="${E:-8}"                       # ensemble size
NP="${NP:-6}"                     # 5 history + 1 DA step
TRAJ="${TRAJ:-15}"                # test_index; 0..19 valid, grid uses 10..14
SEED=0
DEVICE="${DEVICE:-cuda}"
REQUIRE_W="${REQUIRE_W:-true}"    # NS case yaml defaults to false -> force true here,
                                  # otherwise this silently runs on RANDOM weights and
                                  # the whole comparison is noise.
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
OUT="${OUT:-paper_experiments/results/navier_stokes/tuning_lambda_check}"
DIV_GUARD="${DIV_GUARD:-10.0}"    # lambda=1.0 may diverge; abort early + NaN-pad

# ---- sweep matrix ------------------------------------------------------------
STEPS="${STEPS:-25}"                                   # M
KS="${KS:-10}"                                         # jacobian_refresh_every
read -r -a LAMBDAS <<< "${LAMBDAS_STR:-0.9 0.95 1.0}"  # jacobian_damping
if [ -n "${SCENARIOS:-}" ]; then IFS='|' read -r -a SCEN_ARR <<< "$SCENARIOS";
else SCEN_ARR=("16^2->128^2" "sparse 5%"); fi
# All three in one call (they share the knob); SI-SDE is the in-run control.
METHODS="${METHODS:-[\"Ours (SI-SDE)\",\"Ours (DM-SDE)\",\"Ours (FM-ODE)\"]}"
JACFREE_REF="${JACFREE_REF:-1}"   # 1 untuned dps_jacobian_free cell per scenario

mkdir -p "$OUT"
LOG="$OUT/run_ns_lambda_check.log"

slug() { echo "$1" | sed -E 's/[^A-Za-z0-9.+-]+/_/g; s/^_+//; s/_+$//'; }
smoke_head() { if [ "$SMOKE" = "1" ]; then echo "$1"; else echo "$@"; fi; }

# run_cell <outfile> <tag> <extra run.py args...>
run_cell() {
  local outfile="$1"; shift
  local tag="$1"; shift
  if [ -f "$outfile" ]; then echo "[nslam] SKIP (exists) $outfile" | tee -a "$LOG"; return; fi
  echo "[nslam] RUN  $tag -> $(basename "$outfile") $(date +%T)" | tee -a "$LOG"
  if [ "$DRY_RUN" = "1" ]; then echo "  DRY_RUN: $tag -> $outfile"; return; fi
  $PY -u paper_experiments/run.py case=navier_stokes seeds=[$SEED] \
      +test_index=$TRAJ \
      ensemble_size=$E case.num_physical_steps=$NP num_steps=$M \
      case.require_weights=$REQUIRE_W case.device=$DEVICE \
      +skip_kl_reference=true \
      +divergence_rmse_threshold=$DIV_GUARD \
      "+ns_methods=$METHODS" "+ns_scenarios=[\"$SCEN\"]" \
      +save_states=false \
      results_file="$outfile" "$@" >> "$LOG" 2>&1 \
    && echo "[nslam] OK   $tag $(date +%T)" | tee -a "$LOG" \
    || echo "[nslam] FAIL $tag $(date +%T)" | tee -a "$LOG"
}

echo "[nslam] START $(date) | E=$E NP=$NP M=[$STEPS] k=[$KS] traj=$TRAJ dev=$DEVICE DRY_RUN=$DRY_RUN" | tee -a "$LOG"
echo "[nslam] methods=$METHODS" | tee -a "$LOG"

for SCEN in "${SCEN_ARR[@]}"; do
  SS=$(slug "$SCEN")
  for M in $STEPS; do
    # Untuned reference: dps_jacobian_free has no Jacobian knob -> one cell. This is
    # what every lambda has to beat for shared mode to be worth its cost.
    if [ "$JACFREE_REF" = "1" ]; then
      run_cell "$OUT/${SS}__M${M}__jacfree.csv" "$SS/M$M/jacfree" \
        likelihood_mode=dps_jacobian_free
    fi
    for K in $(smoke_head $KS); do
      for LAM in $(smoke_head "${LAMBDAS[@]}"); do
        run_cell "$OUT/${SS}__M${M}__k${K}_lam$(slug "$LAM").csv" \
                 "$SS/M$M/k=$K lambda=$LAM" \
          likelihood_mode=inflated_shared \
          +jacobian_refresh_every=$K +jacobian_damping=$LAM
      done
    done
  done
done
echo "[nslam] ALL DONE $(date)" | tee -a "$LOG"
