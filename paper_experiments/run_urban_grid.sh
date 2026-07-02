#!/bin/bash
# =============================================================================
# Urban (uDALES) REDUCED-GRID master run  (2026-07-01)
#
# Case 3 reduced grid, writing into results/urban/ (see results/README.md).
# GENERATIVE-ONLY: urban has no in-repo CFD solver, so NO EnKF/PF/classical group
# and NO KL reference (no ground-truth posterior). Otherwise the same structure
# and saving policy as run_ns_grid.sh.
#
# GRID
#   trajectories : test_index 1..5   (one seed each; seeds=[0])
#   scenarios    : sparse 5%, sparse 1.5625%   (sparse only for urban)
#   steps M      : 50 100 250 500
#   Ours modes   : jacfree (dps_jacobian_free) + shared (inflated_shared)
#   E=64, num_physical_steps=20 (5 history + 15 DA steps)
#
# METHOD GROUPS (per (traj, scenario, M)):
#   ours_jacfree : Ours (SI-SDE/DM-SDE/FM-ODE), likelihood_mode=dps_jacobian_free
#   ours_shared  : Ours (SI-SDE/DM-SDE/FM-ODE), likelihood_mode=inflated_shared
#   baselines    : FlowDAS, SURGE (FlowDAS), SDA, SURGE (SDA), D-Flow SGLD
#
# SAVING: states for traj1 ONLY (all groups, both Ours modes; variant in filename);
#         per-step metric curves + timings for EVERY trajectory.
#
# Launch: setsid nohup bash paper_experiments/run_urban_grid.sh >run_urban_grid.log 2>&1 & disown
# Track : .venv/bin/python paper_experiments/status.py --case urban
# Env overrides: TRAJ, SCENARIOS(|-sep), STEPS, GRPS, E, NP, DEVICE, REQUIRE_W.
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
PY=.venv/bin/python
ROOT="${ROOT:-paper_experiments/results/urban}"   # override for smoke tests
MET=$ROOT/metrics
PS=$ROOT/per_step
mkdir -p "$MET" "$PS" "$ROOT/states"

TRAJ="${TRAJ:-1 2 3 4 5}"
STEPS="${STEPS:-50 100 250 500}"
E="${E:-64}"
NP="${NP:-20}"
DEVICE="${DEVICE:-cuda}"
REQUIRE_W="${REQUIRE_W:-true}"
if [ -n "${SCENARIOS:-}" ]; then IFS='|' read -r -a SCEN_ARR <<< "$SCENARIOS";
else SCEN_ARR=("sparse 5%" "sparse 1.5625%"); fi
GRPS="${GRPS:-ours_jacfree ours_shared baselines}"

OURS='["Ours (SI-SDE)","Ours (DM-SDE)","Ours (FM-ODE)"]'
BASELINES='["FlowDAS","SURGE (FlowDAS)","SDA","SURGE (SDA)","D-Flow SGLD","Guided FM (FIG)"]'

slug() { echo "$1" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//'; }
has_group() { echo " $GRPS " | grep -q " $1 "; }

LOG="$ROOT/run_urban_grid.log"
echo "[urbangrid] START $(date) | E=$E NP=$NP dev=$DEVICE | traj=[$TRAJ] steps=[$STEPS] groups=[$GRPS]" | tee -a "$LOG"

run_cell() {
  local outfile="$1"; shift
  local tag="$1"; shift
  if [ -f "$outfile" ]; then echo "[urbangrid] SKIP (exists) $outfile" | tee -a "$LOG"; return; fi
  local psfile="$PS/$(basename "${outfile%.csv}").csv"
  echo "[urbangrid] RUN  $tag -> $(basename "$outfile") $(date +%T)" | tee -a "$LOG"
  $PY -u paper_experiments/run.py case=urban seeds=[0] \
      ensemble_size=$E case.num_physical_steps=$NP \
      case.require_weights=$REQUIRE_W case.device=$DEVICE \
      +save_per_step=true "+per_step_file=$psfile" \
      results_file="$outfile" "$@" >> "$LOG" 2>&1 \
    && echo "[urbangrid] OK   $tag $(basename "$outfile") $(date +%T)" | tee -a "$LOG" \
    || echo "[urbangrid] FAIL $tag $(basename "$outfile") $(date +%T)" | tee -a "$LOG"
}

for N in $TRAJ; do
  if [ "$N" = "1" ]; then SAVE_STATES=true; STATES_ROOT="$ROOT/states/traj1"; else SAVE_STATES=false; STATES_ROOT="$ROOT/states/_unused"; fi
  echo "[urbangrid] ===== traj$N (save_states=$SAVE_STATES) $(date +%T) =====" | tee -a "$LOG"

  for SCEN in "${SCEN_ARR[@]}"; do
    SS=$(slug "$SCEN")
    for M in $STEPS; do
      if has_group ours_jacfree; then
        run_cell "$MET/${SS}__M${M}__traj${N}__ours_jacfree.csv" "ours_jacfree/$SS/M$M/traj$N" \
          "+urban_methods=$OURS" "+urban_scenarios=[\"$SCEN\"]" num_steps=$M \
          likelihood_mode=dps_jacobian_free \
          +save_states=$SAVE_STATES "+states_root=$STATES_ROOT"
      fi
      if has_group ours_shared; then
        run_cell "$MET/${SS}__M${M}__traj${N}__ours_shared.csv" "ours_shared/$SS/M$M/traj$N" \
          "+urban_methods=$OURS" "+urban_scenarios=[\"$SCEN\"]" num_steps=$M \
          likelihood_mode=inflated_shared \
          +save_states=$SAVE_STATES "+states_root=$STATES_ROOT"
      fi
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
