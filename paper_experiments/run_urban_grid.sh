#!/bin/bash
# =============================================================================
# Urban (uDALES) REDUCED-GRID master run  (2026-07-01; restructured 2026-07-13)
#
# Case 3 reduced grid, writing into results/urban/ (see results/README.md).
# Same structure and saving policy as run_ns_grid.sh; the differences are
# intrinsic to the case, not stylistic:
#   * GENERATIVE-ONLY -- urban has no in-repo CFD solver, so there is NO
#     `classical` group (no EnKF/PF) and NO KL reference (no ground-truth
#     posterior), hence no kl_reference_states argument.
#   * SPARSE-ONLY scenarios (no super-resolution observation operator).
#
# GRID
#   trajectories : test_index 1..5   (one seed each; seeds=[0])
#   scenarios    : sparse 5%, sparse 1.5625%
#   steps M      : 50 100 250 500
#   Ours modes   : jacfree (dps_jacobian_free) + shared (inflated_shared)
#   E=64, num_physical_steps=20 (5 history + 15 DA steps)
#
# METHOD GROUPS (each is one run.py call per (traj, scenario, M)):
#   ours_jacfree     : Ours (SI-SDE/DM-SDE/FM-ODE), likelihood_mode=dps_jacobian_free
#   ours_shared_k<K> : Ours (SI-SDE/DM-SDE/FM-ODE), likelihood_mode=inflated_shared
#                      with Jacobian refresh cadence k=K. One group per cadence
#                      (ours_shared_k1 / _k5 / _k10) -- pick the cost/accuracy point
#                      by picking the group; each writes its own files so cadences
#                      can coexist in one results tree. lambda (jacobian_damping)
#                      stays PER-SCENARIO in the method YAMLs.
#   baselines        : FlowDAS, SURGE (FlowDAS), SDA, SURGE (SDA), D-Flow SGLD,
#                      Guided FM (FIG)
#
# SAVING
#   save_states  : $SAVE_TRAJ ONLY (default traj1), ALL groups incl. BOTH Ours
#                  modes (variant is in the filename so jacfree/shared never collide).
#   save_per_step: ALWAYS -> per-step metric curves for every trajectory.
#   timings      : seconds + NFE are on every metric row and every per-step row.
#
# Launch: setsid nohup bash paper_experiments/run_urban_grid.sh >run_urban_grid.log 2>&1 & disown
# Track : .venv/bin/python paper_experiments/status.py --case urban
# Env overrides: TRAJ, SAVE_TRAJ, SCENARIOS(|-sep), STEPS, GRPS, E, NP, DEVICE, REQUIRE_W.
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
PY=.venv/bin/python
ROOT="${ROOT:-paper_experiments/results/urban}"   # override for smoke tests
MET=$ROOT/metrics
PS=$ROOT/per_step
mkdir -p "$MET" "$PS" "$ROOT/states"

# ---- knobs (env-overridable) ------------------------------------------------
TRAJ="${TRAJ:-1 2 3 4 5}"
# Which trajectory gets its raw ensembles written out (states are big, so only one).
# MUST be a member of TRAJ or nothing is saved (guarded below).
SAVE_TRAJ="${SAVE_TRAJ:-1}"
STEPS="${STEPS:-50 100 250 500}"
E="${E:-64}"
NP="${NP:-20}"                       # num_physical_steps (5 history + 15 DA)
DEVICE="${DEVICE:-cuda}"
REQUIRE_W="${REQUIRE_W:-true}"       # hard-fail if no trained weights
# Divergence safety net: abort a cell whose ensemble RMSE exceeds this (well above
# any healthy value) and NaN-pad the rest, rather than crashing the whole run.
DIV_GUARD="${DIV_GUARD:-10.0}"
#
# NOTE lambda (jacobian_damping) is NOT set here. It is PER-SCENARIO and lives in
# the method YAMLs (configs/method/{si_sde,dm_sde,fm_ode}.yaml) as [case][scenario][M]
# tables. Urban's cells were NEVER SWEPT: they currently carry the NS *sparse*
# optimum (0.95), on the argument that urban's scenarios are sparse too -- but the
# NS divergence cliff sits at 1.0 on sparse, so 0.95 is one grid point away from it
# and urban's own cliff has never been located. If urban shared cells NaN out, drop
# the urban rows of those tables to 0.9. To sweep a value on purpose, override on
# the CLI for one run:  +jacobian_damping=0.9

# Scenarios as a bash array (canonical labels). Urban is sparse-only.
if [ -n "${SCENARIOS:-}" ]; then IFS='|' read -r -a SCEN_ARR <<< "$SCENARIOS";
else SCEN_ARR=("sparse 5%" "sparse 1.5625%"); fi

# SHARED-MODE GROUPS -- one per Jacobian refresh cadence k. Pick the cadence by
# picking the group; each writes its own files (variant is stamped in), so several
# cadences can coexist in one results tree and be compared directly.
#   ours_shared_k1   k=1  exact per-step Jacobian. BEST rmse on every NS scenario.
#   ours_shared_k5   k=5  ~3-4x cheaper.
#   ours_shared_k10  k=10 ~5-7x cheaper.
# Measured on NS (SI-SDE), the lag is nearly free on the super-res cells but costs
# +15% (sparse 5%) to +49% (sparse 1.5625%) rmse at k=5 -- and urban is SPARSE-ONLY,
# i.e. it sits on the expensive side of that trade. lambda was tuned at k=1, so a
# lagged group is not re-tuned.
#
# k is read straight OUT OF the group name (ours_shared_k<k>), so any cadence works
# with no second list to keep in sync -- e.g. GRPS="... ours_shared_k20 ..." just runs.
# GRPS="${GRPS:-ours_jacfree ours_shared_k1 ours_shared_k5 baselines}"
GRPS="${GRPS:-ours_jacfree ours_shared_k10 baselines}"

# Cadences requested this run: every ours_shared_k<k> in GRPS, k parsed from the name.
SHARED_KS=$(echo "$GRPS" | tr ' ' '\n' | sed -nE 's/^ours_shared_k([0-9]+)$/\1/p')
# Catch a typo'd shared group (e.g. ours_shared_k5x, ours_shared) instead of silently
# skipping it -- a group that matches nothing would otherwise just never run.
for g in $GRPS; do
  case "$g" in
    ours_shared_k[0-9]*) echo "$g" | grep -qE '^ours_shared_k[0-9]+$' || {
        echo "[urbangrid] FATAL: malformed shared group '$g' (expected ours_shared_k<int>)" >&2; exit 1; };;
    ours_shared) echo "[urbangrid] FATAL: group 'ours_shared' is gone -- use ours_shared_k1 (or _k5/_k10)." >&2; exit 1;;
  esac
done

OURS='["Ours (SI-SDE)","Ours (DM-SDE)","Ours (FM-ODE)"]'
BASELINES='["FlowDAS","SURGE (FlowDAS)","SDA","SURGE (SDA)","D-Flow SGLD","Guided FM (FIG)"]'
# -----------------------------------------------------------------------------

slug() { echo "$1" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//'; }
has_group() { echo " $GRPS " | grep -q " $1 "; }

LOG="$ROOT/run_urban_grid.log"
# Fail loudly rather than silently saving no states at all.
if ! echo " $TRAJ " | grep -q " $SAVE_TRAJ "; then
  echo "[urbangrid] FATAL: SAVE_TRAJ=$SAVE_TRAJ is not in TRAJ=[$TRAJ] -- no states would be saved." >&2
  exit 1
fi
echo "[urbangrid] START $(date) | E=$E NP=$NP dev=$DEVICE | traj=[$TRAJ] steps=[$STEPS] groups=[$GRPS] save_states=traj$SAVE_TRAJ" | tee -a "$LOG"

# run_cell <outfile> <group-tag> <extra run.py args...>
run_cell() {
  local outfile="$1"; shift
  local tag="$1"; shift
  if [ -f "$outfile" ]; then echo "[urbangrid] SKIP (exists) $outfile" | tee -a "$LOG"; return; fi
  local psfile="$PS/$(basename "${outfile%.csv}").csv"
  echo "[urbangrid] RUN  $tag -> $(basename "$outfile") $(date +%T)" | tee -a "$LOG"
  # Trajectory N -> test sample N (test_sample_indices=[1..5]); WITHOUT this every
  # traj would rerun the default sample and the trajectory aggregation is a no-op.
  $PY -u paper_experiments/run.py case=urban seeds=[0] \
      ensemble_size=$E case.num_physical_steps=$NP \
      case.require_weights=$REQUIRE_W case.device=$DEVICE \
      +test_index=$N \
      +save_per_step=true "+per_step_file=$psfile" \
      +divergence_rmse_threshold=$DIV_GUARD \
      results_file="$outfile" "$@" >> "$LOG" 2>&1 \
    && echo "[urbangrid] OK   $tag $(basename "$outfile") $(date +%T)" | tee -a "$LOG" \
    || echo "[urbangrid] FAIL $tag $(basename "$outfile") $(date +%T)" | tee -a "$LOG"
}

for N in $TRAJ; do
  # states for SAVE_TRAJ only; ALL groups + both Ours modes.
  if [ "$N" = "$SAVE_TRAJ" ]; then SAVE_STATES=true; STATES_ROOT="$ROOT/states/traj${SAVE_TRAJ}"; else SAVE_STATES=false; STATES_ROOT="$ROOT/states/_unused"; fi
  echo "[urbangrid] ===== traj$N (save_states=$SAVE_STATES) $(date +%T) =====" | tee -a "$LOG"

  for SCEN in "${SCEN_ARR[@]}"; do
    SS=$(slug "$SCEN")

    for M in $STEPS; do
      # ours jacfree
      if has_group ours_jacfree; then
        run_cell "$MET/${SS}__M${M}__traj${N}__ours_jacfree.csv" "ours_jacfree/$SS/M$M/traj$N" \
          "+urban_methods=$OURS" "+urban_scenarios=[\"$SCEN\"]" num_steps=$M \
          likelihood_mode=dps_jacobian_free \
          +save_states=$SAVE_STATES "+states_root=$STATES_ROOT"
      fi
      # ours shared -- one cell per requested refresh cadence k. lambda still comes
      # from the per-scenario table in configs/method/*.yaml; only k is set here.
      # k=1 keeps the historical `ours_shared` / variant=shared naming; k>1 is
      # stamped as ours_shared_jac<k> / variant=shared_jac<k> so they never collide.
      for K in $SHARED_KS; do
        if [ "$K" = "1" ]; then
          KOUT="ours_shared"; KVAR="shared"
        else
          KOUT="ours_shared_jac${K}"; KVAR="shared_jac${K}"
        fi
        run_cell "$MET/${SS}__M${M}__traj${N}__${KOUT}.csv" "${KOUT}/$SS/M$M/traj$N" \
          "+urban_methods=$OURS" "+urban_scenarios=[\"$SCEN\"]" num_steps=$M \
          likelihood_mode=inflated_shared \
          +jacobian_refresh_every=$K "+variant_override=$KVAR" \
          +save_states=$SAVE_STATES "+states_root=$STATES_ROOT"
      done
      # baselines (ignore likelihood_mode; run once per M)
      if has_group baselines; then
        run_cell "$MET/${SS}__M${M}__traj${N}__baselines.csv" "baselines/$SS/M$M/traj$N" \
          "+urban_methods=$BASELINES" "+urban_scenarios=[\"$SCEN\"]" num_steps=$M \
          likelihood_mode=dps_jacobian_free \
          +save_states=$SAVE_STATES "+states_root=$STATES_ROOT"
      fi
    done
  done
  echo "[urbangrid] ===== traj$N done $(date +%T) =====" | tee -a "$LOG"
done
echo "[urbangrid] ALL DONE $(date)" | tee -a "$LOG"
