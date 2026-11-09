#!/usr/bin/env bash
# Full Lorenz 96 comparison for one observation sweep, after training and tuning
# (see cases/lorenz96/README.md). Settings come from configs/case/lorenz96.yaml
# (`methods:` for OBS=linear, `methods_arctan:` for OBS=arctan); tuning is a
# separate, validation-only step (tune_lorenz96.sh).
#
#   OBS=linear|arctan (default linear)  -> results/lorenz96/$OBS/, figures/lorenz96/$OBS/,
#                                          generated/tab_lorenz96_$OBS.tex
#   SEEDS="0 1 2"  M_VALUES="25 50 100 250"  JOBS=1  FORCE=0  PY=python
#
# Generative cells run at every M of M_VALUES (`cell --num-steps M`), all cells at
# M=50 first, then the other M in order; the M-independent cells run once per seed.
# Tuning is at M=50 only, and its picks are reused at every M.
#
# A (cell, M, seed) whose metrics file already exists is skipped, so after a failure a rerun
# only does what is missing; FORCE=1 reruns every cell. References are skipped per
# file by the driver itself. If any cell fails, the script stops after the cell
# loop, before pf_reference, aggregation, figures and the table.
#
# JOBS=1 by default: cells running at the same time share the machine, which
# inflates their `seconds` cost metric. Raise it only if those timings are not
# reported, or say that they ran concurrently.
set -euo pipefail

# --------------------------------------------------------------------------- #
# The lineup, in run order. Edit these lists (and lorenz96_lineup.ROWS) only.
# Generative cells run at every M of M_VALUES (files lorenz96__<cell>__M<M>__traj<s>__.csv);
# M-independent cells run once per seed (lorenz96__<cell>__traj<s>__.csv).
# pf_reference is also seed-independent and runs once after these.
GENERATIVE_CELLS=(
  # ours, plain; forecast-residual correction; trained-prior EnKF
  si_sde fm_ode dm_sde
  si_sde_fc fm_ode_fc dm_sde_fc prior_enkf prior_enkf_fm
  # SURGE (SMC) on our samplers, and the unguided SURGE control
  surge_si_sde surge_dm_sde surge_unguided_si surge_unguided_dm
  # generative baselines
  flowdas surge_flowdas sda surge_sda dflow fig
)
M_INDEPENDENT_CELLS=(
  enkf enkf_large pf_matched pf_subsample oracle oracle_fc
)
# --------------------------------------------------------------------------- #

cd "$(dirname "$0")"
PY=${PY:-python}
OBS=${OBS:-linear}
SEEDS=${SEEDS:-"0 1 2"}
M_VALUES=${M_VALUES:-"25 50 100 250"}
JOBS=${JOBS:-1}
FORCE=${FORCE:-0}
case "$OBS" in
  linear|arctan) ;;
  *) echo "OBS must be linear or arctan, not '$OBS'" >&2; exit 2 ;;
esac
ROOT=results/lorenz96/$OBS
METRICS=$ROOT/metrics  # where `driver.py --obs $OBS cell` writes; see driver.cell_path

for m in $M_VALUES; do
  case "$m" in 25|50|100|250) ;; *) echo "M_VALUES may hold 25 50 100 250 only, not '$m'" >&2; exit 2 ;; esac
done

# run_cell <cell> <seed> <M or ->: "-" for an M-independent cell.
run_cell() {
  local method=$1 seed=$2 m=$3
  local out="$METRICS/lorenz96__${method}__traj${seed}__.csv"
  local extra=()
  if [[ "$m" != - ]]; then
    out="$METRICS/lorenz96__${method}__M${m}__traj${seed}__.csv"
    extra=(--num-steps "$m")
  fi
  if [[ "$FORCE" != 1 && -s "$out" ]]; then
    echo "skip $OBS $method M=$m seed $seed: $out exists (FORCE=1 reruns it)"
    return 0
  fi
  echo "run $OBS $method M=$m seed $seed"
  "$PY" cases/lorenz96/driver.py --obs "$OBS" cell --method "$method" --seed "$seed" "${extra[@]}"
}
export -f run_cell
export PY FORCE METRICS OBS

# M=50 first (with the M-independent cells), so the headline results land first.
ordered=()
[[ " $M_VALUES " == *" 50 "* ]] && ordered=(50)
for m in $M_VALUES; do [[ "$m" == 50 ]] || ordered+=("$m"); done

"$PY" cases/lorenz96/driver.py --obs "$OBS" reference --split test
"$PY" cases/lorenz96/driver.py --obs "$OBS" prior-quality
{
  first=1
  for m in "${ordered[@]}" -; do
    for seed in $SEEDS; do
      if [[ "$m" != - ]]; then
        for method in "${GENERATIVE_CELLS[@]}"; do echo "$method $seed $m"; done
      fi
      if [[ "$first" == 1 ]]; then
        for method in "${M_INDEPENDENT_CELLS[@]}"; do echo "$method $seed -"; done
      fi
    done
    first=0
  done
} | xargs -P "$JOBS" -L 1 bash -c 'run_cell "$0" "$1" "$2"'
# The reference does not depend on a seed.
run_cell pf_reference 0 -
"$PY" aggregate_lorenz96.py --obs "$OBS"
"$PY" make_lorenz96_figures.py --obs "$OBS"
"$PY" make_lorenz96_tables.py --obs "$OBS"
