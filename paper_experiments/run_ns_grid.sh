#!/bin/bash
# =============================================================================
# Navier--Stokes REDUCED-GRID master run  (2026-07-01)
#
# The full reduced paper grid for Case 2, writing into the restructured
# results/navier_stokes/ tree (see results/README.md). Every cell is a separate
# run.py invocation with its own per-cell metrics + per-step CSV, so partial
# progress is salvageable and status.py can track coverage.
#
# GRID
#   trajectories : test_index 1..5   (one seed each; seeds=[0])
#   scenarios    : 16^2->128^2, 32^2->128^2, sparse 5%, sparse 1.5625%
#   steps M      : 50 100 250 500    (generative rows only; classical omit M)
#   Ours modes   : jacfree (dps_jacobian_free) + shared (inflated_shared)
#   E=64, num_physical_steps=20 (5 history + 15 DA steps)
#
# METHOD GROUPS (each is one run.py call per (traj, scenario, M)):
#   ours_jacfree : Ours (SI-SDE/DM-SDE/FM-ODE), likelihood_mode=dps_jacobian_free
#   ours_shared_k<K> : Ours (SI-SDE/DM-SDE/FM-ODE), likelihood_mode=inflated_shared
#                  with Jacobian refresh cadence k=K. One group per cadence
#                  (ours_shared_k1 / _k5 / _k10) -- pick the cost/accuracy point by
#                  picking the group; each writes its own files so they can coexist.
#                  lambda (jacobian_damping) stays PER-SCENARIO in the method YAMLs.
#   baselines    : FlowDAS, SURGE (FlowDAS), SDA, SURGE (SDA), D-Flow SGLD
#   classical    : EnKF, Particle filter   -- ONCE per (traj, scenario), no M sweep
#
# SAVING
#   save_states  : $SAVE_TRAJ ONLY (default traj11), ALL groups incl. BOTH Ours
#                  modes (variant is in the filename so jacfree/shared never collide).
#   save_per_step: ALWAYS -> per-step metric curves for every trajectory, so
#                  figures/tables can be rebuilt without the raw ensembles.
#   timings      : seconds + NFE are on every metric row and every per-step row.
#
# KL reference : results/navier_stokes/reference/traj<N>/gt (E=1000 non-loc EnKF,
#                produced by run_ns_reference.sh). Missing -> KL is NaN, rest runs.
#
# Launch detached:
#   setsid nohup bash paper_experiments/run_ns_grid.sh >run_ns_grid.log 2>&1 & disown
# Track:  .venv/bin/python paper_experiments/status.py --case navier_stokes
#
# Env overrides (subset the grid): TRAJ, SCENARIOS(|-sep), STEPS, GRPS, E, NP, DEVICE.
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
PY=.venv/bin/python
ROOT="${ROOT:-paper_experiments/results/navier_stokes}"   # override for smoke tests
MET=$ROOT/metrics
PS=$ROOT/per_step
REF=$ROOT/reference
mkdir -p "$MET" "$PS" "$ROOT/states"

# ---- knobs (env-overridable) ------------------------------------------------
# Slice rows into test_data (data[180:200]); 10..14 == global trajectories 190-194.
TRAJ="${TRAJ:-10 11 12 13 14}"
# Which trajectory gets its raw ensembles written out (states are big, so only one).
# MUST be a member of TRAJ or nothing is saved -- that was the case until 2026-07-13,
# when this was hard-coded to "1" while TRAJ defaulted to 10..14, so save_states was
# never true for any cell.
SAVE_TRAJ="${SAVE_TRAJ:-11}"
STEPS="${STEPS:-50 100 250}"
E="${E:-64}"
NP="${NP:-20}"                       # num_physical_steps (5 history + 15 DA)
DEVICE="${DEVICE:-cuda}"
REQUIRE_W="${REQUIRE_W:-true}"       # hard-fail if no trained weights
# Divergence safety net: abort a cell whose ensemble RMSE exceeds this (well above
# any healthy value ~0.5, so healthy cells are byte-unchanged) and NaN-pad the rest.
DIV_GUARD="${DIV_GUARD:-10.0}"
#
# NOTE the shared-mode knobs (jacobian_damping lambda, jacobian_refresh_every) are
# NOT set here. They are PER-SCENARIO and live in the method YAMLs
# (configs/method/{si_sde,dm_sde,fm_ode}.yaml) as [case][scenario][M] tables --
# sparse -> lambda 0.95, superres -> 0.9, refresh 1 everywhere. Pinning one value
# for the whole grid from this script is precisely the bug that shipped the old
# global lambda=0.7 (3.1x worse than optimal on sparse_1p5), so don't reintroduce
# it. To sweep a single value on purpose, override on the CLI for one run:
#   +jacobian_damping=0.9 +jacobian_refresh_every=5


# TRAJ="${TRAJ:-1 2}"
# STEPS="${STEPS:-100}"
# E="${E:-8}"
# NP="${NP:-7}"                       # num_physical_steps (5 history + 15 DA)
# DEVICE="${DEVICE:-cuda}"
# REQUIRE_W="${REQUIRE_W:-true}"       # hard-fail if no trained weights

# Scenarios as a bash array (canonical labels).
if [ -n "${SCENARIOS:-}" ]; then IFS='|' read -r -a SCEN_ARR <<< "$SCENARIOS";
else SCEN_ARR=("16^2->128^2" "32^2->128^2" "sparse 5%" "sparse 1.5625%"); fi

# if [ -n "${SCENARIOS:-}" ]; then IFS='|' read -r -a SCEN_ARR <<< "$SCENARIOS";
# else SCEN_ARR=("16^2->128^2"); fi

# SHARED-MODE GROUPS -- one per Jacobian refresh cadence k. Pick the cadence by
# picking the group; each writes its own files (variant is stamped in), so several
# cadences can coexist in one results tree and be compared directly.
#   ours_shared_k1   k=1  exact per-step Jacobian. BEST rmse on every scenario.
#   ours_shared_k5   k=5  ~3-4x cheaper. Costs +1% rmse on 32^2, +3% on 16^2,
#                         but +15% on sparse 5% and +49% on sparse 1.5625%.
#   ours_shared_k10  k=10 ~5-7x cheaper. +2% on 32^2, +16% on 16^2, but +59-69%
#                         on the sparse cells.
# The lag is nearly free where shared mode barely helps (superres) and expensive
# where it helps most (sparse) -- it costs you in proportion to what the Jacobian
# was contributing. lambda stays per-scenario in configs/method/*.yaml; note it was
# tuned AT k=1, so a lagged group is not re-tuned.
#
# k is read straight OUT OF the group name (ours_shared_k<k>), so any cadence works
# with no second list to keep in sync -- e.g. GRPS="... ours_shared_k20 ..." just runs.
# GRPS="${GRPS:-ours_jacfree ours_shared_k1 ours_shared_k5 baselines classical}"
GRPS="${GRPS:-ours_jacfree ours_shared_k10 baselines}"

# Cadences requested this run: every ours_shared_k<k> in GRPS, k parsed from the name.
SHARED_KS=$(echo "$GRPS" | tr ' ' '\n' | sed -nE 's/^ours_shared_k([0-9]+)$/\1/p')
# Catch a typo'd shared group (e.g. ours_shared_k5x, ours_shared) instead of silently
# skipping it -- a group that matches nothing would otherwise just never run.
for g in $GRPS; do
  case "$g" in
    ours_shared_k[0-9]*) echo "$g" | grep -qE '^ours_shared_k[0-9]+$' || {
        echo "[nsgrid] FATAL: malformed shared group '$g' (expected ours_shared_k<int>)" >&2; exit 1; };;
    ours_shared) echo "[nsgrid] FATAL: group 'ours_shared' is gone -- use ours_shared_k1 (or _k5/_k10)." >&2; exit 1;;
  esac
done

OURS='["Ours (SI-SDE)","Ours (DM-SDE)","Ours (FM-ODE)"]'
BASELINES='["FlowDAS","SURGE (FlowDAS)","SDA","SURGE (SDA)","D-Flow SGLD","Guided FM (FIG)"]'
CLASSICAL='["EnKF","Particle filter"]'
# -----------------------------------------------------------------------------

slug() { echo "$1" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//'; }
has_group() { echo " $GRPS " | grep -q " $1 "; }

LOG="$ROOT/run_ns_grid.log"
# Fail loudly rather than silently saving no states at all (the pre-2026-07-13 bug).
if ! echo " $TRAJ " | grep -q " $SAVE_TRAJ "; then
  echo "[nsgrid] FATAL: SAVE_TRAJ=$SAVE_TRAJ is not in TRAJ=[$TRAJ] -- no states would be saved." >&2
  exit 1
fi
echo "[nsgrid] START $(date) | E=$E NP=$NP dev=$DEVICE | traj=[$TRAJ] steps=[$STEPS] groups=[$GRPS] save_states=traj$SAVE_TRAJ" | tee -a "$LOG"

# run_cell <outfile> <group-tag> <extra run.py args...>
run_cell() {
  local outfile="$1"; shift
  local tag="$1"; shift
  if [ -f "$outfile" ]; then echo "[nsgrid] SKIP (exists) $outfile" | tee -a "$LOG"; return; fi
  local psfile="$PS/$(basename "${outfile%.csv}").csv"
  echo "[nsgrid] RUN  $tag -> $(basename "$outfile") $(date +%T)" | tee -a "$LOG"
  # Trajectory N -> test sample N (test_sample_indices=[1..5]); WITHOUT this every
  # traj would rerun the default sample and the trajectory aggregation is a no-op.
  $PY -u paper_experiments/run.py case=navier_stokes seeds=[0] \
      ensemble_size=$E case.num_physical_steps=$NP \
      case.require_weights=$REQUIRE_W case.device=$DEVICE \
      +test_index=$N \
      +save_per_step=true "+per_step_file=$psfile" \
      +divergence_rmse_threshold=$DIV_GUARD \
      results_file="$outfile" "$@" >> "$LOG" 2>&1 \
    && echo "[nsgrid] OK   $tag $(basename "$outfile") $(date +%T)" | tee -a "$LOG" \
    || echo "[nsgrid] FAIL $tag $(basename "$outfile") $(date +%T)" | tee -a "$LOG"
}

for N in $TRAJ; do
  # states for SAVE_TRAJ only; ALL groups + both Ours modes.
  if [ "$N" = "$SAVE_TRAJ" ]; then SAVE_STATES=true; STATES_ROOT="$ROOT/states/traj${SAVE_TRAJ}"; else SAVE_STATES=false; STATES_ROOT="$ROOT/states/_unused"; fi
  KLREF="$REF/traj${N}/gt"
  echo "[nsgrid] ===== traj$N (save_states=$SAVE_STATES) $(date +%T) =====" | tee -a "$LOG"

  for SCEN in "${SCEN_ARR[@]}"; do
    SS=$(slug "$SCEN")

    # ---- generative groups: swept over M --------------------------------
    for M in $STEPS; do
      # ours jacfree
      if has_group ours_jacfree; then
        run_cell "$MET/${SS}__M${M}__traj${N}__ours_jacfree.csv" "ours_jacfree/$SS/M$M/traj$N" \
          "+ns_methods=$OURS" "+ns_scenarios=[\"$SCEN\"]" num_steps=$M \
          likelihood_mode=dps_jacobian_free "+kl_reference_states=$KLREF" \
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
          "+ns_methods=$OURS" "+ns_scenarios=[\"$SCEN\"]" num_steps=$M \
          likelihood_mode=inflated_shared \
          +jacobian_refresh_every=$K "+variant_override=$KVAR" \
          "+kl_reference_states=$KLREF" \
          +save_states=$SAVE_STATES "+states_root=$STATES_ROOT"
      done
      # baselines (ignore likelihood_mode; run once per M)
      if has_group baselines; then
        run_cell "$MET/${SS}__M${M}__traj${N}__baselines.csv" "baselines/$SS/M$M/traj$N" \
          "+ns_methods=$BASELINES" "+ns_scenarios=[\"$SCEN\"]" num_steps=$M \
          likelihood_mode=dps_jacobian_free "+kl_reference_states=$KLREF" \
          +save_states=$SAVE_STATES "+states_root=$STATES_ROOT"
      fi
    done

    # ---- classical group: once per (traj, scenario), no M sweep ---------
    if has_group classical; then
      # localization: sparse -> Gaspari-Cohn r=20; super-res -> non-localized.
      LOC=""; case "$SCEN" in sparse*) LOC="+enkf_localization_radius=20";; esac
      run_cell "$MET/${SS}__traj${N}__classical.csv" "classical/$SS/traj$N" \
        "+ns_methods=$CLASSICAL" "+ns_scenarios=[\"$SCEN\"]" num_steps=50 \
        $LOC "+kl_reference_states=$KLREF" \
        +save_states=$SAVE_STATES "+states_root=$STATES_ROOT"
    fi
  done
  echo "[nsgrid] ===== traj$N done $(date +%T) =====" | tee -a "$LOG"
done
echo "[nsgrid] ALL DONE $(date)" | tee -a "$LOG"
