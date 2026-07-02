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
#   ours_shared  : Ours (SI-SDE/DM-SDE/FM-ODE), likelihood_mode=inflated_shared
#   baselines    : FlowDAS, SURGE (FlowDAS), SDA, SURGE (SDA), D-Flow SGLD
#   classical    : EnKF, Particle filter   -- ONCE per (traj, scenario), no M sweep
#
# SAVING
#   save_states  : traj1 ONLY, ALL groups incl. BOTH Ours modes (variant is in
#                  the filename so jacfree/shared never collide).
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
# TRAJ="${TRAJ:-1 2 3 4 5}"
# STEPS="${STEPS:-50 100 250 500}"
# E="${E:-64}"
# NP="${NP:-20}"                       # num_physical_steps (5 history + 15 DA)
# DEVICE="${DEVICE:-cuda}"
# REQUIRE_W="${REQUIRE_W:-true}"       # hard-fail if no trained weights


TRAJ="${TRAJ:-1 2}"
STEPS="${STEPS:-100}"
E="${E:-8}"
NP="${NP:-7}"                       # num_physical_steps (5 history + 15 DA)
DEVICE="${DEVICE:-cuda}"
REQUIRE_W="${REQUIRE_W:-true}"       # hard-fail if no trained weights

# Scenarios as a bash array (canonical labels).
# if [ -n "${SCENARIOS:-}" ]; then IFS='|' read -r -a SCEN_ARR <<< "$SCENARIOS";
# else SCEN_ARR=("16^2->128^2" "32^2->128^2" "sparse 5%" "sparse 1.5625%"); fi

if [ -n "${SCENARIOS:-}" ]; then IFS='|' read -r -a SCEN_ARR <<< "$SCENARIOS";
else SCEN_ARR=("16^2->128^2"); fi

# GRPS="${GRPS:-ours_jacfree ours_shared baselines classical}"
GRPS="${GRPS:-ours_jacfree baselines}"

OURS='["Ours (SI-SDE)","Ours (DM-SDE)","Ours (FM-ODE)"]'
BASELINES='["FlowDAS","SURGE (FlowDAS)","SDA","SURGE (SDA)","D-Flow SGLD","Guided FM (FIG)"]'
CLASSICAL='["EnKF","Particle filter"]'
# -----------------------------------------------------------------------------

slug() { echo "$1" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//'; }
has_group() { echo " $GRPS " | grep -q " $1 "; }

LOG="$ROOT/run_ns_grid.log"
echo "[nsgrid] START $(date) | E=$E NP=$NP dev=$DEVICE | traj=[$TRAJ] steps=[$STEPS] groups=[$GRPS]" | tee -a "$LOG"

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
      results_file="$outfile" "$@" >> "$LOG" 2>&1 \
    && echo "[nsgrid] OK   $tag $(basename "$outfile") $(date +%T)" | tee -a "$LOG" \
    || echo "[nsgrid] FAIL $tag $(basename "$outfile") $(date +%T)" | tee -a "$LOG"
}

for N in $TRAJ; do
  # states only for the first trajectory; ALL groups + both Ours modes.
  if [ "$N" = "1" ]; then SAVE_STATES=true; STATES_ROOT="$ROOT/states/traj1"; else SAVE_STATES=false; STATES_ROOT="$ROOT/states/_unused"; fi
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
      # ours shared
      if has_group ours_shared; then
        run_cell "$MET/${SS}__M${M}__traj${N}__ours_shared.csv" "ours_shared/$SS/M$M/traj$N" \
          "+ns_methods=$OURS" "+ns_scenarios=[\"$SCEN\"]" num_steps=$M \
          likelihood_mode=inflated_shared "+kl_reference_states=$KLREF" \
          +save_states=$SAVE_STATES "+states_root=$STATES_ROOT"
      fi
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
