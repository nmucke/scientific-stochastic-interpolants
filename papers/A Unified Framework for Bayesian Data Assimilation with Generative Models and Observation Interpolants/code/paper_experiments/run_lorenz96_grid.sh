#!/usr/bin/env bash
# Full Lorenz 96 comparison, after training (see cases/lorenz96/README.md). Settings
# come from configs/case/lorenz96.yaml; tuning is a separate, validation-only step.
#
# A cell whose metrics file already exists is skipped, so after a failure a rerun
# only does what is missing; FORCE=1 reruns every cell. References are skipped per
# file by the driver itself.
#
# JOBS=1 by default: cells running at the same time share the machine, which
# inflates their `seconds` cost metric. Raise it only if those timings are not
# reported, or say that they ran concurrently.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
SEEDS=${SEEDS:-"0 1 2"}
JOBS=${JOBS:-1}
FORCE=${FORCE:-0}
METRICS=results/lorenz96/metrics  # where driver.py cell writes; see cell_path

run_cell() {
  local method=$1 seed=$2
  local out="$METRICS/lorenz96__${method}__traj${seed}__.csv"
  if [[ "$FORCE" != 1 && -s "$out" ]]; then
    echo "skip $method seed $seed: $out exists (FORCE=1 reruns it)"
    return 0
  fi
  "$PY" cases/lorenz96/driver.py cell --method "$method" --seed "$seed"
}
export -f run_cell
export PY FORCE METRICS

"$PY" cases/lorenz96/driver.py reference --split test
"$PY" cases/lorenz96/driver.py prior-quality
for seed in $SEEDS; do
  for method in si_sde fm_ode dm_sde enkf enkf_large oracle pf_matched pf_subsample; do
    echo "$method $seed"
  done
done | xargs -P "$JOBS" -L 1 bash -c 'run_cell "$0" "$1"'
# The reference does not depend on a seed.
run_cell pf_reference 0
"$PY" aggregate_lorenz96.py
"$PY" make_lorenz96_figures.py
