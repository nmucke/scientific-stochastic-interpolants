#!/usr/bin/env bash
# Full Lorenz 96 comparison, after training (see cases/lorenz96/README.md). Settings
# come from configs/case/lorenz96.yaml; tuning is a separate, validation-only step.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
SEEDS=${SEEDS:-"0 1 2"}
JOBS=${JOBS:-3}

"$PY" cases/lorenz96/driver.py reference --split test
"$PY" cases/lorenz96/driver.py prior-quality
for seed in $SEEDS; do
  for method in si_sde fm_ode dm_sde enkf enkf_large oracle pf_matched pf_subsample; do
    echo "$method $seed"
  done
done | xargs -P "$JOBS" -L 1 bash -c '"$0" cases/lorenz96/driver.py cell --method "$1" --seed "$2"' "$PY"
# The reference does not depend on a seed.
"$PY" cases/lorenz96/driver.py cell --method pf_reference --seed 0
"$PY" aggregate_lorenz96.py
"$PY" make_lorenz96_figures.py
