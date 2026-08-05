#!/bin/bash
# =============================================================================
# Navier--Stokes REDUCED-GRID master run -- M=25 FILL, D-FLOW ONLY (2026-07-30)
#
# The other half of run_ns_grid_M25_nodflow.sh. That script runs
#   baselines_nodflow : FlowDAS, SURGE (FlowDAS), SDA, SURGE (SDA), Guided FM (FIG)
#                       (+ the two Ours groups)
# and deliberately OMITS D-Flow SGLD, because on CPU it costs ~4.7 h per
# assimilation step => ~71 h for ONE cell, against ~20 h for the other five
# baselines COMBINED. This script is that omitted method, and nothing else:
#   dflow : D-Flow SGLD ONLY
#
# Same grid, same per-cell configs, same skip-existing resume, same output
# filenames (``__dflow.csv``) as `GRPS=dflow bash run_ns_grid_M25_nodflow.sh`
# would have produced -- this is just a standalone file so it can be launched and
# logged on its own without touching the live nodflow job.
#
# The two halves write SEPARATE per-cell CSVs (``__baselines_nodflow.csv`` /
# ``__dflow.csv``). Aggregation globs every CSV and keys on (method, variant,
# scenario, metric, M), so the split is invisible downstream and the halves rejoin
# on their own -- no merge step.
#
# DUPLICATE GUARD. The older run_ns_grid_M25.sh writes ONE ``__baselines.csv`` per
# cell holding all six methods, D-Flow included. If such a file already exists for
# a cell, D-Flow is already recorded there, so this script SKIPS that cell --
# otherwise the method would be counted twice in the same trajectory. See the
# LEGACY check in run_cell.
#
# DEVICE=cuda is the DEFAULT here (the nodflow script defaults to cpu). GPU is
# ~6x faster and D-Flow's CPU cost is the entire reason it was split out. This job
# runs D-Flow alone, so it does not have to share the card with the five other
# baselines -- which is what caused the 2026-07-15 CUDA OOM losses (34/60 cells;
# that job peaked at 18.2 GiB on a 23.6 GiB card while a concurrent M=250 job held
# 4.9 GiB). If the card is busy or it OOMs anyway, fall back with DEVICE=cpu and
# expect ~71 h/cell. Resumable either way: run_cell skips cells whose CSV exists.
#
# GRID (identical to run_ns_grid_M25_nodflow.sh)
#   trajectories : test_index 10..14  (one seed each; seeds=[0])
#   scenarios    : 16^2->128^2, 32^2->128^2, sparse 5%, sparse 1.5625%
#   steps M      : 25 ONLY
#   E=64, num_physical_steps=20 (5 history + 15 DA steps)
#
# SAVING
#   save_states  : $SAVE_TRAJ ONLY (default traj11)
#   save_per_step: ALWAYS -> per-step metric curves for every trajectory
#   timings      : seconds + NFE are on every metric row and every per-step row.
#
# KL reference : results/navier_stokes/reference/traj<N>/gt (E=1000 non-loc EnKF,
#                produced by run_ns_reference.sh). Missing -> KL is NaN, rest runs.
#
# Launch detached (safe to run while the nodflow job is going):
#   setsid nohup bash paper_experiments/run_ns_grid_M25_dflow.sh \
#     >run_ns_grid_M25_dflow.log 2>&1 & disown
# CPU fallback:
#   DEVICE=cpu setsid nohup bash paper_experiments/run_ns_grid_M25_dflow.sh \
#     >run_ns_grid_M25_dflow.log 2>&1 & disown
# Track:  .venv/bin/python paper_experiments/status.py --case navier_stokes
#
# Env overrides (subset the grid): TRAJ, SCENARIOS(|-sep), STEPS, E, NP, DEVICE.
# There is no GRPS knob -- this script is the `dflow` group by construction.
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
PY="${PY:-.venv/bin/python}"   # overridable so the cell selection can be dry-run
ROOT="${ROOT:-paper_experiments/results/navier_stokes}"   # override for smoke tests
MET=$ROOT/metrics
PS=$ROOT/per_step
REF=$ROOT/reference
mkdir -p "$MET" "$PS" "$ROOT/states"

# ---- knobs (env-overridable) ------------------------------------------------
# Slice rows into test_data (data[180:200]); 10..14 == global trajectories 190-194.
TRAJ="${TRAJ:-10 11 12 13 14}"
# Which trajectory gets its raw ensembles written out (states are big, so only one).
# MUST be a member of TRAJ or nothing is saved.
SAVE_TRAJ="${SAVE_TRAJ:-11}"
STEPS="${STEPS:-25}"                 # M=25 fill variant -- pinned to 25
E="${E:-64}"
NP="${NP:-20}"                       # num_physical_steps (5 history + 15 DA)
DEVICE="${DEVICE:-cuda}"             # D-Flow is the slow method: GPU by default (see header)
REQUIRE_W="${REQUIRE_W:-true}"       # hard-fail if no trained weights
# CPU thread pool. Torch grabs all 32 cores by default; cap it if you need to leave
# headroom for another job's dataloader. Override with THREADS=<n>.
export OMP_NUM_THREADS="${THREADS:-32}"
export MKL_NUM_THREADS="${THREADS:-32}"
# Divergence safety net: abort a cell whose ensemble RMSE exceeds this (well above
# any healthy value ~0.5, so healthy cells are byte-unchanged) and NaN-pad the rest.
DIV_GUARD="${DIV_GUARD:-10.0}"
#
# NOTE D-Flow ignores likelihood_mode and the shared-mode knobs entirely; the
# `likelihood_mode=dps_jacobian_free` below is passed only to keep the invocation
# byte-identical to the nodflow script's dflow branch. D-Flow's own knobs
# (SGLD steps/step size) are M-driven and live in its method config.

# Scenarios as a bash array (canonical labels).
if [ -n "${SCENARIOS:-}" ]; then IFS='|' read -r -a SCEN_ARR <<< "$SCENARIOS";
else SCEN_ARR=("16^2->128^2" "32^2->128^2" "sparse 5%" "sparse 1.5625%"); fi

DFLOW='["D-Flow SGLD"]'
# -----------------------------------------------------------------------------

slug() { echo "$1" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//'; }

# Own log, env-overridable so parallel shards each get their own file instead of
# interleaving run.py output into one (run_cell appends run.py stdout here).
LOG="${LOG:-$ROOT/run_ns_grid_M25_dflow.log}"
# Fail loudly rather than silently saving no states at all.
# SAVE_TRAJ=none is the ONE legitimate way to save nothing: parallel shards that do
# not own traj$SAVE_TRAJ pass it so only the shard holding traj11 writes states.
if [ "$SAVE_TRAJ" != "none" ] && ! echo " $TRAJ " | grep -q " $SAVE_TRAJ "; then
  echo "[nsdflow] FATAL: SAVE_TRAJ=$SAVE_TRAJ is not in TRAJ=[$TRAJ] -- no states would be saved." >&2
  echo "[nsdflow]        (pass SAVE_TRAJ=none if this shard is meant to save no states.)" >&2
  exit 1
fi
echo "[nsdflow] START $(date) | E=$E NP=$NP dev=$DEVICE | traj=[$TRAJ] steps=[$STEPS] groups=[dflow] save_states=traj$SAVE_TRAJ" | tee -a "$LOG"

# run_cell <outfile> <group-tag> <extra run.py args...>
#
# LEGACY is an optional pre-set path: a file whose presence means this cell's
# method was ALREADY recorded by the old combined `baselines` group. Running anyway
# would enter D-Flow twice for one trajectory, which the aggregation would silently
# average. Cleared after each call so it can never leak into the next cell.
run_cell() {
  local outfile="$1"; shift
  local tag="$1"; shift
  local legacy="${LEGACY:-}"; LEGACY=""
  if [ -f "$outfile" ]; then echo "[nsdflow] SKIP (exists) $outfile" | tee -a "$LOG"; return; fi
  if [ -n "$legacy" ] && [ -f "$legacy" ]; then
    echo "[nsdflow] SKIP (covered by $(basename "$legacy")) $tag" | tee -a "$LOG"; return
  fi
  local psfile="$PS/$(basename "${outfile%.csv}").csv"
  echo "[nsdflow] RUN  $tag -> $(basename "$outfile") $(date +%T)" | tee -a "$LOG"
  # Trajectory N -> test sample N (test_sample_indices=[1..5]); WITHOUT this every
  # traj would rerun the default sample and the trajectory aggregation is a no-op.
  $PY -u paper_experiments/run.py case=navier_stokes seeds=[0] \
      ensemble_size=$E case.num_physical_steps=$NP \
      case.require_weights=$REQUIRE_W case.device=$DEVICE \
      +test_index=$N \
      +save_per_step=true "+per_step_file=$psfile" \
      +divergence_rmse_threshold=$DIV_GUARD \
      results_file="$outfile" "$@" >> "$LOG" 2>&1 \
    && echo "[nsdflow] OK   $tag $(basename "$outfile") $(date +%T)" | tee -a "$LOG" \
    || echo "[nsdflow] FAIL $tag $(basename "$outfile") $(date +%T)" | tee -a "$LOG"
}

for N in $TRAJ; do
  # states for SAVE_TRAJ only.
  if [ "$N" = "$SAVE_TRAJ" ]; then SAVE_STATES=true; STATES_ROOT="$ROOT/states/traj${SAVE_TRAJ}"; else SAVE_STATES=false; STATES_ROOT="$ROOT/states/_unused"; fi
  KLREF="$REF/traj${N}/gt"
  echo "[nsdflow] ===== traj$N (save_states=$SAVE_STATES) $(date +%T) =====" | tee -a "$LOG"

  for SCEN in "${SCEN_ARR[@]}"; do
    SS=$(slug "$SCEN")

    # ---- D-Flow SGLD alone: swept over M -------------------------------
    for M in $STEPS; do
      LEGACY="$MET/${SS}__M${M}__traj${N}__baselines.csv" \
      run_cell "$MET/${SS}__M${M}__traj${N}__dflow.csv" "dflow/$SS/M$M/traj$N" \
        "+ns_methods=$DFLOW" "+ns_scenarios=[\"$SCEN\"]" num_steps=$M \
        likelihood_mode=dps_jacobian_free "+kl_reference_states=$KLREF" \
        +save_states=$SAVE_STATES "+states_root=$STATES_ROOT"
    done
  done
  echo "[nsdflow] ===== traj$N done $(date +%T) =====" | tee -a "$LOG"
done
echo "[nsdflow] ALL DONE $(date)" | tee -a "$LOG"
