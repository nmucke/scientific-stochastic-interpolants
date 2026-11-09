#!/usr/bin/env bash
# One Lorenz 96 observation sweep, end to end, in two human-gated halves.
#
#   OBS=linear|arctan STAGE=tune  ./run_lorenz96_sweep.sh
#       validation references -> stage-1 tuning -> selection -> stage-2 tuning
#       (stage-1 picks passed via --set) -> selection, then STOP. A human writes
#       results/lorenz96/$OBS/tuning/selected_stage{1,2}.yaml into `methods:`
#       (linear) or `methods_arctan:` (arctan) of configs/case/lorenz96.yaml and
#       approves the compute for the final grid.
#   OBS=linear|arctan STAGE=final ./run_lorenz96_sweep.sh
#       run_lorenz96_grid.sh: test references, every cell x SEEDS, aggregation,
#       figures, table.
#
# Everything is resumable (finished jobs and cells are skipped). The log goes to
# results/lorenz96/$OBS/logs/<stage>_<time>.log and its exit status to <log>.exit.
# PY, SEEDS, JOBS, FORCE, THREADS, ONLY are passed through to the stage scripts.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
OBS=${OBS:-linear}
STAGE=${STAGE:-tune}
case "$OBS" in linear|arctan) ;; *) echo "OBS must be linear or arctan, not '$OBS'" >&2; exit 2 ;; esac
case "$STAGE" in tune|final) ;; *) echo "STAGE must be tune or final, not '$STAGE'" >&2; exit 2 ;; esac
export PY OBS

LOG_DIR=results/lorenz96/$OBS/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${STAGE}_$(date +%Y%m%d-%H%M%S).log"
trap 'status=$?; echo "$status" > "$LOG.exit"; echo "exit $status (log: $LOG)"' EXIT
exec > >(tee -a "$LOG") 2>&1
echo "lorenz96 sweep OBS=$OBS STAGE=$STAGE start $(date -Is) commit $(git rev-parse --short HEAD 2>/dev/null || echo '?')"

if [[ "$STAGE" == tune ]]; then
  "$PY" cases/lorenz96/driver.py --obs "$OBS" reference --split validation
  GRID=1 ./tune_lorenz96.sh
  "$PY" select_lorenz96_tuning.py --obs "$OBS" --stage 1
  GRID=2 ./tune_lorenz96.sh
  "$PY" select_lorenz96_tuning.py --obs "$OBS" --stage 2
  echo
  echo "STOP: tuning of the $OBS sweep done. Review"
  echo "  results/lorenz96/$OBS/tuning/selected_stage1.md and selected_stage2.md,"
  echo "write the picks into configs/case/lorenz96.yaml, then run"
  echo "  OBS=$OBS STAGE=final ./run_lorenz96_sweep.sh"
else
  ./run_lorenz96_grid.sh
fi
echo "lorenz96 sweep OBS=$OBS STAGE=$STAGE done $(date -Is)"
