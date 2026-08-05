#!/bin/bash
# =============================================================================
# Urban (uDALES) HYPERPARAMETER TUNING sweeps -- OURS  (2026-07-30)
#
# Sibling of run_urban_tuning.sh, structurally IDENTICAL (same posterior-run
# settings, same per-(scenario, M) cell granularity, same skip-if-exists /
# DRY_RUN / SMOKE behaviour, same results tree). The only difference is WHICH
# methods and WHICH knob are swept: our three samplers instead of the baselines.
#
# WHAT IS TUNED
#   ours  =  Ours (SI-SDE) + Ours (DM-SDE) + Ours (FM-ODE), ALL THREE TOGETHER
#            knob: jacobian_damping (lambda)  -> +jacobian_damping
#            at cadence k                     -> +jacobian_refresh_every
#
# ALL THREE IN ONE CALL. Unlike the baseline script (where each method has its own
# knob, so each needs its own run.py call), SI/DM/FM share the SAME knob, so every
# cell passes all three in one `+urban_methods=[...]` exactly as run_urban_grid.sh
# does. That amortises the ~17 s startup/JIT/data-load across three methods instead
# of paying it three times, and it guarantees the three are compared under bitwise
# identical conditions. One CSV per cell holds all three methods' rows.
# The per-method keys (si_sde / dm_sde / fm_ode) remain available to re-run ONE
# method alone -- they write into their own subdir so they never collide.
#
# WHY LAMBDA IS THE KNOB. Our samplers have no likelihood hyperparameter of the
# kind the baselines have (no zeta / gamma / eta). The one per-cell number they do
# carry is `jacobian_damping` (lambda) in configs/method/{si_sde,dm_sde,fm_ode}.yaml,
# and it exists ONLY in `inflated_shared`: in `dps_jacobian_free` Sigma_s = rho I, so
# every Jacobian knob is inert (si_sde.yaml lines 19-21). Hence this script sweeps
# lambda in shared mode and runs jacfree once per cell as the untuned REFERENCE --
# the analogue of the zeta=0 control in the baseline sweep, and the number every
# lambda has to beat to justify shared mode on urban at all.
#
# WHY IT NEEDS SWEEPING. The urban lambda cells are PROVISIONAL: they carry the NS
# *sparse* optimum 0.95 verbatim and were never swept (si_sde.yaml lines 74-79).
# That is a riskier transfer than it looks, for two independent reasons:
#   1. On NS the optimum always sits JUST BELOW that scenario's divergence cliff,
#      and the cliff MOVES between scenarios (superres_32 diverges at 0.95 while the
#      sparse cells are still improving there). Urban's cliff has never been located,
#      so 0.95 could be either one grid point below it or already past it.
#   2. lambda was tuned at k=1, but the urban grid's default group is
#      ours_shared_k10. A staler Jacobian is itself a crude damping, so it SHIFTS
#      the optimum and suppresses the lambda=1 divergence -- si_sde.yaml lines 38-42
#      say explicitly to re-tune lambda rather than assume the k=1 values carry over.
#      That is why k is a sweep axis here and defaults to the grid's cadence.
#
# SWEEP MATRIX: a SEPARATE sweep per (scenario, M) cell, as in the baseline script.
#   scenarios : sparse 5%, sparse 1.5625%     (urban is sparse-only)
#   steps M   : 25 50 100 250                 (MUST match run_urban_grid.sh STEPS)
#   lambda    : see LAMBDAS below             (inflated_shared only)
#   cadence k : see KS below, default 10      (the urban grid's default group)
#
# POSTERIOR RUN: as the baseline sweep -- ONE urban test trajectory
# (+test_index=$TRAJ), ONE seed, ensemble_size=8 -- but num_physical_steps=8 = 5-step
# seeded history + 3 assimilation steps. Cheap enough to rank configs; re-validate
# winners at E=64 / full history before the final table. test_index must be in 0..8
# (urban test set is `files (170, 178)` -> 9 trajectories); the paper grid evaluates
# 1..5, so 6 keeps tuning off the evaluation trajectories.
#
# USAGE
#   bash paper_experiments/run_urban_tuning_ours.sh            # all three (default)
#   bash paper_experiments/run_urban_tuning_ours.sh ours       # same thing
#   bash paper_experiments/run_urban_tuning_ours.sh si_sde     # one method alone
#   <method> in: ours si_sde dm_sde fm_ode
#
#   # preview the run matrix without launching anything:
#   DRY_RUN=1 bash paper_experiments/run_urban_tuning_ours.sh
#
#   # CPU smoke test (small, fast -- wiring check):
#   DEVICE=cpu STEPS=5 SCENARIOS="sparse 5%" SMOKE=1 \
#     bash paper_experiments/run_urban_tuning_ours.sh
#
#   # re-tune at the exact-Jacobian cadence as well:
#   KS="1 10" bash paper_experiments/run_urban_tuning_ours.sh
#
# Full sweeps are for the GPU box (DEVICE=cuda, the default) -- do NOT launch them
# on CPU, and do NOT launch while run_urban_tuning.sh is still running (one
# GPU-heavy job at a time). Launch detached when ready:
#   setsid nohup bash paper_experiments/run_urban_tuning_ours.sh \
#     >run_urban_tuning_ours.log 2>&1 & disown
#
# ENV OVERRIDES (subset / smoke the sweep): DEVICE, STEPS, SCENARIOS(|-sep), TRAJ,
#   E, NP, REQUIRE_W, DRY_RUN, SMOKE(=1 -> only the first grid value per axis), OUT,
#   KS, JACFREE_REF(=0 to skip the reference cells), DIV_GUARD.
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
PY=.venv/bin/python

# ---- fixed posterior-run settings (identical to run_urban_tuning.sh) --------
E="${E:-8}"                       # ensemble size
NP="${NP:-8}"                     # num_physical_steps = 5 history + 3 DA steps
                                  # (n_assim = NP - len_field_history(5)). NOT 6:
                                  # at ONE DA step the urban metric is dominated by
                                  # prior forecast error -- the zeta=0 control sat
                                  # within 3% of the best FlowDAS cell and every
                                  # SDA/FIG cell was worse than it -- so the ranking
                                  # rested on almost no signal. 3 steps let error
                                  # accumulate. (The SDA/FIG re-tune used 4; 3 is
                                  # this sweep's setting, hence its own OUT root.)
TRAJ="${TRAJ:-6}"                 # ONE urban test trajectory; must be 0..8 (9 test
                                  # sims) and ideally OUTSIDE the paper grid's 1..5
SEED=0                            # one seed
DEVICE="${DEVICE:-cuda}"          # cuda for the real sweep; DEVICE=cpu to smoke
REQUIRE_W="${REQUIRE_W:-true}"    # udales checkpoints carry model.pth -> true
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"               # 1 -> only the first value of each swept axis
# Root is per-DA-step-count. Cell filenames encode only (scenario, M, hyperparameter),
# NOT the DA-step count, so runs at different NP MUST NOT share a root: skip-if-exists
# would silently skip every cell whose namesake already existed and the sweep would
# appear to run while producing nothing. tuning/=1 step, tuning_da4/=4, this=3.
OUT="${OUT:-paper_experiments/results/urban/tuning_da3}"

# Divergence safety net. NOT in the baseline script, and deliberately added here:
# divergence IS the failure mode this sweep is mapping (lambda too close to the
# cliff), so without the guard a bad cell burns its full budget and can poison the
# run, while with it the cell aborts early and NaN-pads. Same default as
# run_urban_grid.sh, well above any healthy value (~0.5).
DIV_GUARD="${DIV_GUARD:-10.0}"

# ---- sweep matrix (env-overridable) -----------------------------------------
# M AXIS CUT TO {25,50} 2026-07-30 ON MEASURED COST. inflated_shared spends one JVP
# per OBSERVATION per Jacobian refresh, and urban sparse_5 observes 5% of 4x128x128
# = 3277 points vs NS's 5% of 1x128x128 = 819 -- 4x, on top of a bigger UNet. Measured
# here: 465 s per DA step at M=25 against jacfree's 4.74 s, i.e. ~98x. Cost is ~linear
# in M, so one shared cell is ~70 min at M=25 and ~700 min at M=250; the full ladder
# would have been ~8 days. {25,50} keeps a second point so the M-dependence of lambda
# can be CHECKED rather than assumed -- every existing tuned table (NS x4 scenarios,
# analytical) has lambda constant across all five M columns, but that has never been
# verified on urban, whose observation count is what makes this case different.
STEPS="${STEPS:-25 50}"                                # M (SDE/ODE step axis)
if [ -n "${SCENARIOS:-}" ]; then IFS='|' read -r -a SCEN_ARR <<< "$SCENARIOS";
else SCEN_ARR=("sparse 5%" "sparse 1.5625%"); fi       # urban: sparse only

mkdir -p "$OUT"
LOG="$OUT/run_urban_tuning_ours.log"

slug() { echo "$1" | sed -E 's/[^A-Za-z0-9.+-]+/_/g; s/^_+//; s/_+$//'; }
# SMOKE mode keeps only the first element of a grid array (fast wiring check);
# otherwise returns the whole grid unchanged.
smoke_head() { if [ "$SMOKE" = "1" ]; then echo "$1"; else echo "$@"; fi; }

# ---- hyperparameter grid ----------------------------------------------------
# lambda = jacobian_damping. Incumbent (copied from NS sparse) is 0.95; the grid
# brackets it and walks up to the ceiling to LOCATE URBAN'S CLIFF, which is the
# actual deliverable here -- on NS the optimum was always the last stable point.
#   0.8  / 0.9   safe side; 0.9 is the documented fallback if 0.95 NaNs
#   0.95         the incumbent
#   0.98 / 1.0   above the incumbent. 1.0 is the hard theoretical ceiling (NS found
#                lambda>1 diverges everywhere, so there is no headroom past it) and
#                it DIVERGES at k=1 on NS sparse -- but lagging suppresses that, so
#                at k=10 it is a genuine candidate rather than a wasted cell.
# Cells that do diverge are cheap: the guard aborts them early.
# CUT TO 3 VALUES 2026-07-30 on measured cost (see STEPS below): the full 5-value
# grid over the 4-M ladder was ~8 DAYS. These three still do the job -- 0.95 is the
# incumbent, 1.0 is the ceiling/cliff probe, 0.9 is the documented safe fallback --
# giving a bracket around the incumbent and a divergence answer. Dropped: 0.8 (well
# below the incumbent and the NS curve is monotone up to the cliff, so it only
# matters if 0.9 already diverges) and 0.98 (resolution between 0.95 and 1.0, a
# refinement worth running only once the cliff location is known).
LAMBDAS=(0.9 0.95 1.0)

# k = jacobian_refresh_every. Defaults to the single cadence the urban grid actually
# runs (GRPS default ours_shared_k10 in run_urban_grid.sh), so the sweep tunes what
# will be used. Set KS="1 10" to re-tune the exact-Jacobian cadence too -- lambda and
# k are NOT independent, so a k you did not sweep is a lambda you have not tuned.
KS="${KS:-10}"

# One dps_jacobian_free cell per (scenario, M) as the untuned reference.
# No knob applies in that mode, so it is a single cell, not a grid -- and it is what
# tells you whether shared mode is worth its cost on urban at all. Set 0 to skip.
JACFREE_REF="${JACFREE_REF:-1}"

# All three of ours in ONE call (the default); the per-method keys below run one.
OURS_ALL='["Ours (SI-SDE)","Ours (DM-SDE)","Ours (FM-ODE)"]'

# run_cell <outfile> <tag> <extra run.py args...>
# Emits one run.py invocation. Unlike the baseline script, likelihood_mode is NOT
# fixed here -- it is passed by the caller, since the whole point is comparing
# inflated_shared (tuned) against dps_jacobian_free (reference).
run_cell() {
  local outfile="$1"; shift
  local tag="$1"; shift
  if [ -f "$outfile" ]; then echo "[urban-tune-ours] SKIP (exists) $outfile" | tee -a "$LOG"; return; fi
  echo "[urban-tune-ours] RUN  $tag -> $(basename "$outfile") $(date +%T)" | tee -a "$LOG"
  if [ "$DRY_RUN" = "1" ]; then
    echo "  DRY_RUN: $tag -> $outfile"; return
  fi
  $PY -u paper_experiments/run.py case=urban seeds=[$SEED] \
      +test_index=$TRAJ \
      ensemble_size=$E case.num_physical_steps=$NP num_steps=$M \
      case.require_weights=$REQUIRE_W case.device=$DEVICE \
      +divergence_rmse_threshold=$DIV_GUARD \
      "+urban_methods=$METHODS_JSON" "+urban_scenarios=[\"$SCEN\"]" \
      +save_states=false \
      results_file="$outfile" "$@" >> "$LOG" 2>&1 \
    && echo "[urban-tune-ours] OK   $tag $(date +%T)" | tee -a "$LOG" \
    || echo "[urban-tune-ours] FAIL $tag $(date +%T)" | tee -a "$LOG"
}

# sweep_method <key> : loops (scenario x M x k x lambda) for one key.
# key `ours` runs SI+DM+FM together in every cell; the single-method keys run one.
sweep_method() {
  local method="$1"
  case "$method" in
    ours)   METHODS_JSON="$OURS_ALL" ;;
    si_sde) METHODS_JSON='["Ours (SI-SDE)"]' ;;
    dm_sde) METHODS_JSON='["Ours (DM-SDE)"]' ;;
    fm_ode) METHODS_JSON='["Ours (FM-ODE)"]' ;;
    *) echo "[urban-tune-ours] unknown method '$method'"; exit 2 ;;
  esac
  local mdir="$OUT/$method"; mkdir -p "$mdir"
  echo "[urban-tune-ours] ===== $method = $METHODS_JSON (E=$E NP=$NP dev=$DEVICE traj=$TRAJ) $(date +%T) =====" | tee -a "$LOG"

  for SCEN in "${SCEN_ARR[@]}"; do
    local SS; SS=$(slug "$SCEN")
    for M in $STEPS; do
      # Untuned reference: dps_jacobian_free has no Jacobian knob, so one cell.
      if [ "$JACFREE_REF" = "1" ]; then
        run_cell "$mdir/${SS}__M${M}__jacfree.csv" "$method/$SS/M$M/jacfree" \
          likelihood_mode=dps_jacobian_free
      fi
      # Tuned: inflated_shared, lambda x cadence.
      for K in $(smoke_head $KS); do
        for LAM in $(smoke_head "${LAMBDAS[@]}"); do
          run_cell "$mdir/${SS}__M${M}__k${K}_lam$(slug "$LAM").csv" \
                   "$method/$SS/M$M/k=$K lambda=$LAM" \
            likelihood_mode=inflated_shared \
            +jacobian_refresh_every=$K +jacobian_damping=$LAM
        done
      done
    done
  done
}

# ---- dispatch ---------------------------------------------------------------
# Default (no argument) is `ours` = all three together, which is the intended use.
ALL_METHODS=(si_sde dm_sde fm_ode)
WHICH="${1:-ours}"
echo "[urban-tune-ours] START $(date) | which=$WHICH DRY_RUN=$DRY_RUN SMOKE=$SMOKE dev=$DEVICE KS=[$KS]" | tee -a "$LOG"
if [ "$WHICH" = all ]; then
  # `all` = each method separately (3x the startup cost); prefer the default `ours`.
  for m in "${ALL_METHODS[@]}"; do sweep_method "$m"; done
else
  sweep_method "$WHICH"
fi
echo "[urban-tune-ours] ALL DONE $(date)" | tee -a "$LOG"
