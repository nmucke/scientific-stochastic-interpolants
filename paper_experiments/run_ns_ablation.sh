#!/bin/bash
# =============================================================================
# Navier--Stokes ABLATION run: shared-covariance Jacobian refresh cadence
# (2026-07-28)
#
# Runs ONLY the Ours samplers (SI-SDE / DM-SDE / FM-ODE) in ensemble-shared
# (inflated_shared) mode at the cadences in KS (default k=1 and k=5), at a
# SINGLE sampler-step count (default M=100). Together with the MAIN grid's
# existing rows at the same M -- ours_shared_k10 (k=10) and ours_jacfree (the
# Jacobian-free reference) -- this yields the cadence axis k = 1 / 5 / 10 for
# the ablation table and figures (make_ablation_figures.py).
#
# RESULTS TREE -- deliberately SEPARATE from the paper grid:
#   results/navier_stokes/ablation/{metrics,per_step}
# NOT results/navier_stokes/metrics/. The paper pipeline canonicalises every
# shared_jac<k> variant onto ONE "shared" series (figure_common._canon_variant),
# so cadence rows dropped into the main tree would silently mix into the
# manuscript's metric-vs-M figures. Only make_ablation_figures.py reads this
# tree; aggregate_ns.py / make_ns_figures.py never see it.
#
# Variants are stamped shared_jac1 / shared_jac5 EXPLICITLY (unlike the main
# grid, where k=1 keeps the historical bare "shared" tag) so the cadence is
# readable straight off every tidy row.
#
# GRID (env-overridable, same conventions as run_ns_grid.sh)
#   trajectories : TRAJ      (default 10..14, the paper's five test slices)
#   scenarios    : SCENARIOS (default all four; '|'-separated to subset, e.g.
#                             SCENARIOS="sparse 5%" for the one-scenario table)
#   steps M      : STEPS     (default 100 -- single M, per the ablation design)
#   cadences k   : KS        (default "1 5"; k=10 + jacfree come from the main grid)
#   E=64, NP=20, per-step curves always, NO state saving (metrics-only run).
# lambda (jacobian_damping) stays PER-SCENARIO in configs/method/*.yaml, exactly
# as in the main grid (it was tuned at k=1); override once via +jacobian_damping=.
#
# Launch: setsid nohup bash paper_experiments/run_ns_ablation.sh >run_ns_ablation.log 2>&1 & disown
# Then  : .venv/bin/python paper_experiments/make_ablation_figures.py
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
PY=.venv/bin/python
GRID_ROOT="paper_experiments/results/navier_stokes"
ROOT="${ROOT:-$GRID_ROOT/ablation}"   # override for smoke tests
MET=$ROOT/metrics
PS=$ROOT/per_step
REF=$GRID_ROOT/reference              # KL reference lives with the main grid
mkdir -p "$MET" "$PS"

# ---- knobs (env-overridable) ------------------------------------------------
TRAJ="${TRAJ:-10 11 12 13 14}"
STEPS="${STEPS:-100}"
KS="${KS:-1 5}"
E="${E:-64}"
NP="${NP:-20}"                       # num_physical_steps (5 history + 15 DA)
DEVICE="${DEVICE:-cuda}"
REQUIRE_W="${REQUIRE_W:-true}"       # hard-fail if no trained weights
DIV_GUARD="${DIV_GUARD:-10.0}"       # abort+NaN-pad a diverged cell, don't crash

# Scenarios as a bash array (canonical labels).
if [ -n "${SCENARIOS:-}" ]; then IFS='|' read -r -a SCEN_ARR <<< "$SCENARIOS";
else SCEN_ARR=("16^2->128^2" "32^2->128^2" "sparse 5%" "sparse 1.5625%"); fi

OURS='["Ours (SI-SDE)","Ours (DM-SDE)","Ours (FM-ODE)"]'
# -----------------------------------------------------------------------------

slug() { echo "$1" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//'; }

LOG="$ROOT/run_ns_ablation.log"
echo "[nsabl] START $(date) | E=$E NP=$NP dev=$DEVICE | traj=[$TRAJ] steps=[$STEPS] k=[$KS]" | tee -a "$LOG"

# run_cell <outfile> <group-tag> <extra run.py args...>  (same skip-if-exists
# resume behaviour as the main grid).
run_cell() {
  local outfile="$1"; shift
  local tag="$1"; shift
  if [ -f "$outfile" ]; then echo "[nsabl] SKIP (exists) $outfile" | tee -a "$LOG"; return; fi
  local psfile="$PS/$(basename "${outfile%.csv}").csv"
  echo "[nsabl] RUN  $tag -> $(basename "$outfile") $(date +%T)" | tee -a "$LOG"
  $PY -u paper_experiments/run.py case=navier_stokes seeds=[0] \
      ensemble_size=$E case.num_physical_steps=$NP \
      case.require_weights=$REQUIRE_W case.device=$DEVICE \
      +test_index=$N \
      +save_per_step=true "+per_step_file=$psfile" \
      +divergence_rmse_threshold=$DIV_GUARD \
      results_file="$outfile" "$@" >> "$LOG" 2>&1 \
    && echo "[nsabl] OK   $tag $(basename "$outfile") $(date +%T)" | tee -a "$LOG" \
    || echo "[nsabl] FAIL $tag $(basename "$outfile") $(date +%T)" | tee -a "$LOG"
}

for N in $TRAJ; do
  KLREF="$REF/traj${N}/gt"
  echo "[nsabl] ===== traj$N $(date +%T) =====" | tee -a "$LOG"
  for SCEN in "${SCEN_ARR[@]}"; do
    SS=$(slug "$SCEN")
    for M in $STEPS; do
      for K in $KS; do
        run_cell "$MET/${SS}__M${M}__traj${N}__ours_shared_jac${K}.csv" \
          "ours_shared_jac${K}/$SS/M$M/traj$N" \
          "+ns_methods=$OURS" "+ns_scenarios=[\"$SCEN\"]" num_steps=$M \
          likelihood_mode=inflated_shared \
          +jacobian_refresh_every=$K "+variant_override=shared_jac${K}" \
          "+kl_reference_states=$KLREF"
      done
    done
  done
  echo "[nsabl] ===== traj$N done $(date +%T) =====" | tee -a "$LOG"
done
echo "[nsabl] ALL DONE $(date)" | tee -a "$LOG"
