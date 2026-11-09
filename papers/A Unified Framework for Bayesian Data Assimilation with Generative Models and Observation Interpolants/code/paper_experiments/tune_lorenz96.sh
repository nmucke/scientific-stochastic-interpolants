#!/usr/bin/env bash
# Validation-only tuning of the Lorenz 96 cells for one observation sweep.
#
#   OBS=linear|arctan (default linear)   GRID=1|2 (default 1)
#   ONLY="enkf fm_ode"  run only these cells' jobs of the stage
#   PY=python  THREADS=2
#
# Each job is `driver.py --obs $OBS tune --method <cell> --grid <grid> --tag s<GRID>`
# and appends to results/lorenz96/$OBS/tuning/<cell>__<keys>__s<GRID>.csv.
# A job whose CSV already has every grid point is skipped; a partial CSV (an
# interrupted job, or one with other grid keys) is moved aside to
# <csv>.partial.<time> and the job reruns in full. Jobs run one at a time: the
# `seconds` column is a cost measure.
#
# Stage 2 needs results/lorenz96/$OBS/tuning/selected_stage1.yaml
# (select_lorenz96_tuning.py --obs $OBS --stage 1): each stage-2 job gets its
# parent's stage-1 picks as --set, and {"times":[...]} grid values are factors on
# the parent's pick of that key.
set -euo pipefail

# --------------------------------------------------------------------------- #
# THE GRIDS. One row per tuning job: "stage|cell|parent|grid JSON".
#   parent  stage 2 only: the stage-1 cell whose picks are passed as --set
#           (keys not in this grid) and that {"times":[..]} values refer to.
# M is fixed at 50 and NOT tuned: no grid may contain num_steps (refused below).
GRIDS=(
  # ---- stage 1 ---------------------------------------------------------- #
  '1|enkf||{"inflation":[1.0,1.02,1.05,1.1,1.2],"localisation":[null,4,8]}'
  '1|enkf_large||{"inflation":[1.0,1.02,1.05],"localisation":[null,4,8]}'
  '1|si_sde||{"likelihood_mode":["dps_jacobian_free","inflated"],"jacobian_damping":[1.0,0.95,0.9]}'
  '1|fm_ode||{"likelihood_mode":["dps_jacobian_free","inflated"],"jacobian_damping":[1.0,0.95,0.9]}'
  '1|dm_sde||{"likelihood_mode":["dps_jacobian_free","inflated"],"jacobian_damping":[1.0,0.95,0.9]}'
  '1|prior_enkf||{"inflation":[1.0,1.02,1.05],"localisation":[4,8]}'
  '1|prior_enkf_fm||{"inflation":[1.0,1.02,1.05],"localisation":[4,8]}'
  '1|flowdas||{"guidance_scale":[0.004,0.016,0.064,0.25,1.0,4.0]}'
  '1|sda||{"gamma_sda":[1e-4,1e-3,1e-2,1e-1,1.0]}'
  '1|dflow||{"step_size":[0.001,0.005,0.02],"lambda_reg":[1e-4,1e-3]}'
  '1|fig||{"guidance_steps":[1,3],"guidance_scale":[1,4,16,64]}'
  # ---- stage 2 (after stage-1 selection) -------------------------------- #
  '2|si_sde_fc|si_sde|{"correction.gain_scale":[0.25,0.5,0.75,1.0],"correction.q_source":["online","exact"]}'
  '2|fm_ode_fc|fm_ode|{"correction.gain_scale":[0.25,0.5,0.75,1.0],"correction.q_source":["online","exact"]}'
  '2|dm_sde_fc|dm_sde|{"correction.gain_scale":[0.25,0.5,0.75,1.0],"correction.q_source":["online","exact"]}'
  # SURGE on our samplers: inherits the sampler's stage-1 picks (drop the parent
  # if the SURGE cells do not read likelihood_mode / jacobian_damping).
  '2|surge_si_sde|si_sde|{"ess_threshold":[0.25,0.5,1.0]}'
  '2|surge_dm_sde|dm_sde|{"ess_threshold":[0.25,0.5,1.0]}'
  '2|surge_unguided_si||{"ess_threshold":[0.25,0.5,1.0]}'
  '2|surge_unguided_dm||{"ess_threshold":[0.25,0.5,1.0]}'
  '2|surge_flowdas|flowdas|{"guidance_scale":{"times":[0.25,1,4]},"ess_threshold":[0.5,1.0]}'
  '2|surge_sda|sda|{"gamma_sda":{"times":[0.1,1,10]},"ess_threshold":[0.5,1.0]}'
)
# --------------------------------------------------------------------------- #

cd "$(dirname "$0")"
PY=${PY:-python}
OBS=${OBS:-linear}
GRID=${GRID:-1}
ONLY=${ONLY:-}
THREADS=${THREADS:-2}
case "$OBS" in linear|arctan) ;; *) echo "OBS must be linear or arctan, not '$OBS'" >&2; exit 2 ;; esac
case "$GRID" in 1|2) ;; *) echo "GRID must be 1 or 2, not '$GRID'" >&2; exit 2 ;; esac

# Refuse a tuned step count before anything runs (also correction.forecast_num_steps).
for row in "${GRIDS[@]}"; do
  if [[ "$row" == *num_steps* ]]; then
    echo "refused: grid row tunes a step count, but M is fixed at 50: $row" >&2
    exit 2
  fi
done

selected=0
for row in "${GRIDS[@]}"; do
  IFS='|' read -r stage cell parent grid <<<"$row"
  [[ "$stage" == "$GRID" ]] || continue
  if [[ -n "$ONLY" && " $ONLY " != *" $cell "* ]]; then
    continue
  fi
  selected=$((selected + 1))
  sets=()
  if [[ "$stage" == 2 ]]; then
    # Line 1: the resolved grid; then one --set per line (the parent's picks).
    out=$("$PY" select_lorenz96_tuning.py --obs "$OBS" resolve \
      --parent "$parent" --grid "$grid")
    mapfile -t resolved <<<"$out"
    if [[ -z "${resolved[0]:-}" ]]; then
      echo "could not resolve the stage-2 grid of $cell from $parent" >&2
      exit 1
    fi
    grid=${resolved[0]}
    for entry in "${resolved[@]:1}"; do
      sets+=(--set "$entry")
    done
  fi
  out=$("$PY" select_lorenz96_tuning.py --obs "$OBS" status \
    --cell "$cell" --grid "$grid" --stage "$stage")
  read -r state csv <<<"$out"
  case "$state" in
    complete)
      echo "skip $OBS stage $stage $cell: $csv has every grid point"
      continue ;;
    partial)
      aside="$csv.partial.$(date +%Y%m%d-%H%M%S)"
      echo "partial $csv: moved to $aside, rerunning $cell in full"
      mv "$csv" "$aside" ;;
    missing) ;;
    *) echo "unexpected status '$state' for $cell" >&2; exit 1 ;;
  esac
  echo "tune $OBS stage $stage $cell grid=$grid ${sets[*]:-}"
  "$PY" cases/lorenz96/driver.py --obs "$OBS" tune --method "$cell" --grid "$grid" \
    --tag "s$stage" --threads "$THREADS" "${sets[@]}"
done
if [[ "$selected" -eq 0 ]]; then
  echo "no stage-$GRID job matched ONLY='$ONLY'" >&2
  exit 1
fi
echo "stage $GRID tuning of the $OBS sweep done"
