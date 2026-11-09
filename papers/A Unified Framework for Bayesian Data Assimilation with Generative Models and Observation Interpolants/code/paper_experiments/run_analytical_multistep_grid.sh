#!/bin/bash
# =============================================================================
# Sequential linear-Gaussian grid: K assimilation cycles, Kalman-filter reference.
#
# One invocation per (setting, seed), written as
# {metrics,per_step}/<setting>__M<M>__K<K>__E<E>__traj<seed>.csv, so
# aggregate_analytical_multistep.py reduces over seeds with the shared
# common.aggregate_lib, as the NS grid does over trajectories. A cell runs into
# tmp/ and is moved into place only on success, so it resumes cleanly; a different
# M, K or E is a different cell, never a false skip.
#
#   bash "$EXPERIMENT_DIR/run_analytical_multistep_grid.sh"
#   "$PY" "$EXPERIMENT_DIR/aggregate_analytical_multistep.py"
#   "$PY" "$EXPERIMENT_DIR/make_analytical_multistep_figures.py"
#
# Each setting has two cells per seed: the lineup, and <setting>_surge with only the
# SURGE-wrapped Ours rows. The second is separate so those rows can join a finished
# grid: the lineup files are skipped, not rewritten. It holds no reference rows.
#
# Forecast-residual correction (FORECAST_CORRECTION.md), not in the default LINEUPS:
#   LINEUPS=fc     one cell per forecast step count S in FC_STEPS, holding only the
#                  corrected Ours rows, <setting>_fc<S>[_bank[<B>]][_g<E>]. The rows
#                  are variants <jacfree|shared>_fc<S>[_bank[<B>]][_g<E>].
#   LINEUPS=group  <setting>_g<E>: EnKF and adapted_shared on the same groups of
#                  FC_GROUP members, the comparators of a grouped fc cell.
# FC_STEPS (default "1 2 5 10 25 50"), FC_Q_SOURCE (online | bank, default online),
# FC_GROUP (members per independent ensemble; empty = all N_EVAL members),
# FC_VARIANTS (plain Ours variants to correct, e.g. "shared"; empty = jacfree and
# shared; a subset is part of the cell name), FC_BANK_SIZE (residual pairs with
# FC_Q_SOURCE=bank; empty = the case's 4096; any other size is part of the cell and
# variant name, _bank<B>). An empty FC_STEPS gives no fc cells: logged and counted
# as a failure.
#
# Env overrides: SETTINGS, LINEUPS, SEEDS, M, N_EVAL, CYCLES, ROOT, PY, FC_*. N_EVAL defaults to
# the manuscript's 100000. Use another ROOT for smoke runs: the aggregation refuses
# to mix horizons, and the figure script to mix M or E.
# =============================================================================
set -u
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/common/paths.sh" || exit 1
cd "$PE_REPO_ROOT"
PY="${PY:-$PE_PYTHON}"
ROOT="${ROOT:-$PE_SCRIPT_DIR/results/analytical_multistep}"
SETTINGS="${SETTINGS:-paper partial}"
LINEUPS="${LINEUPS:-base surge}"
SEEDS="${SEEDS:-0 1 2 3 4}"
M="${M:-50}"
N_EVAL="${N_EVAL:-100000}"
CYCLES="${CYCLES:-20}"
FC_STEPS="${FC_STEPS:-1 2 5 10 25 50}"
FC_Q_SOURCE="${FC_Q_SOURCE:-online}"
FC_GROUP="${FC_GROUP:-}"
FC_VARIANTS="${FC_VARIANTS:-}"
FC_BANK_SIZE="${FC_BANK_SIZE:-}"
FC_SUFFIX=""; FC_ARGS=("+fc_q_source=$FC_Q_SOURCE")
if [ -n "$FC_VARIANTS" ]; then
  FC_SUFFIX="_${FC_VARIANTS// /-}"; FC_ARGS+=("+fc_variants=[${FC_VARIANTS// /,}]")
fi
[ "$FC_Q_SOURCE" = online ] || FC_SUFFIX="${FC_SUFFIX}_$FC_Q_SOURCE"
if [ -n "$FC_BANK_SIZE" ]; then
  [ "$FC_Q_SOURCE" = bank ] || { echo "[msgrid] FC_BANK_SIZE needs FC_Q_SOURCE=bank" >&2; exit 1; }
  FC_ARGS+=("+fc_bank_size=$FC_BANK_SIZE")
  [ "$FC_BANK_SIZE" = 4096 ] || FC_SUFFIX="${FC_SUFFIX}$FC_BANK_SIZE"
fi
if [ -n "$FC_GROUP" ]; then FC_SUFFIX="${FC_SUFFIX}_g$FC_GROUP"; FC_ARGS+=("+fc_group_size=$FC_GROUP"); fi
mkdir -p "$ROOT/metrics" "$ROOT/per_step" "$ROOT/tmp"
LOG="$ROOT/run_analytical_multistep_grid.log"
FAILS=0

echo "[msgrid] START $(date) | settings=[$SETTINGS] lineups=[$LINEUPS] seeds=[$SEEDS] M=$M K=$CYCLES E=$N_EVAL" | tee -a "$LOG"
for SETTING in $SETTINGS; do
  CASE=analytical_multistep
  [ "$SETTING" = paper ] || CASE="analytical_multistep_$SETTING"
  for LINEUP in $LINEUPS; do
  # One "cell spec" per cell name: name|extra hydra args (space separated).
  SPECS=()
  case "$LINEUP" in
    base)  SPECS=("$SETTING|") ;;
    surge) SPECS=("${SETTING}_surge|+surge_only=true") ;;
    fc)    for S in $FC_STEPS; do
             SPECS+=("${SETTING}_fc${S}${FC_SUFFIX}|+fc_only=true +fc_steps=[$S] ${FC_ARGS[*]}")
           done ;;
    group) [ -n "$FC_GROUP" ] || { echo "[msgrid] LINEUPS=group needs FC_GROUP" | tee -a "$LOG"; exit 1; }
           SPECS=("${SETTING}_g${FC_GROUP}|+group_only=true +fc_group_size=$FC_GROUP") ;;
    *)     echo "[msgrid] unknown lineup $LINEUP" | tee -a "$LOG"; exit 1 ;;
  esac
  if [ "${#SPECS[@]}" -eq 0 ]; then
    echo "[msgrid] FAIL no cells for lineup $LINEUP (empty FC_STEPS?)" | tee -a "$LOG"
    FAILS=$((FAILS + 1)); continue
  fi
  # ${a[@]+"${a[@]}"}: bash 3.2 (macOS /bin/bash) rejects "${a[@]}" of an empty
  # array under set -u.
  for SPEC in ${SPECS[@]+"${SPECS[@]}"}; do
  CELL="${SPEC%%|*}"; read -r -a EXTRA <<< "${SPEC#*|}"
  for SEED in $SEEDS; do
    NAME="${CELL}__M${M}__K${CYCLES}__E${N_EVAL}__traj${SEED}.csv"
    if [ -f "$ROOT/metrics/$NAME" ] && [ -f "$ROOT/per_step/$NAME" ]; then
      echo "[msgrid] SKIP (exists) $NAME" | tee -a "$LOG"; continue
    fi
    TMP_METRICS="$ROOT/tmp/metrics__$NAME"
    TMP_PER_STEP="$ROOT/tmp/per_step__$NAME"
    rm -f "$TMP_METRICS" "$TMP_PER_STEP"
    echo "[msgrid] RUN  $NAME $(date +%T)" | tee -a "$LOG"
    if "$PY" -u "$PE_SCRIPT_DIR/run.py" "case=$CASE" "seeds=[$SEED]" "num_steps=$M" \
        "+n_eval_samples=$N_EVAL" "+num_cycles=$CYCLES" ${EXTRA[@]+"${EXTRA[@]}"} \
        "results_file=$TMP_METRICS" "+per_step_file=$TMP_PER_STEP" \
        "hydra.run.dir=$ROOT/hydra/${NAME%.csv}" >> "$LOG" 2>&1; then
      mv -f "$TMP_METRICS" "$ROOT/metrics/$NAME"
      mv -f "$TMP_PER_STEP" "$ROOT/per_step/$NAME"
      echo "[msgrid] OK   $NAME $(date +%T)" | tee -a "$LOG"
    else
      echo "[msgrid] FAIL $NAME $(date +%T)" | tee -a "$LOG"
      FAILS=$((FAILS + 1))
    fi
  done
  done
  done
done
echo "[msgrid] ALL DONE $(date) | failed cells: $FAILS" | tee -a "$LOG"
exit $((FAILS > 0))
