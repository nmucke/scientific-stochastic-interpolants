#!/usr/bin/env bash
# One Lorenz 96 observation sweep, in three human-gated stages.
#
#   ./run_lorenz96.sh tune  linear|arctan
#       validation references -> stage-1 tuning -> selection -> stage-2 tuning (each
#       job gets its parent's stage-1 picks as --set) -> selection, then STOP.
#       Review results/lorenz96/<obs>/tuning/selected_stage{1,2}.md.
#   ./run_lorenz96.sh apply linear|arctan
#       select_lorenz96_tuning.py apply: merges the picks into `methods:` (linear) /
#       `methods_arctan:` (arctan) of configs/case/lorenz96.yaml, nothing else.
#   ./run_lorenz96.sh final linear|arctan
#       test references, prior quality (skipped when its JSON exists), every cell x
#       SEEDS (generative cells at every M: 50 first, then the others), pf_reference
#       once, then aggregation, figures and tables.
#
# Environment: PY=python  THREADS=2 (tune)  SEEDS="0 1 2"  M_VALUES (default
# lorenz96_lineup.M_VALUES; M=50 always runs first)  FORCE=0  ONLY="cell ..." (tune:
# only these cells' jobs of each stage).
#
# Everything is resumable. tune: a job whose CSV has every grid point is skipped, a
# partial CSV (interrupted, other grid keys) is moved aside to <csv>.partial.<time>
# and the job reruns in full. final: a (cell, M, seed) whose metrics file exists is
# skipped (FORCE=1 reruns it); if any cell fails the others still run, and the script
# stops before pf_reference and the aggregation. Cells run one at a time (JOBS=1):
# concurrent cells would inflate each other's `seconds`. M is checked by the driver
# (tune: M=50 only; cell: M_VALUES).
#
# The log goes to results/lorenz96/<obs>/logs/<stage>_<time>.log, its exit status
# to <log>.exit.
set -euo pipefail
cd "$(dirname "$0")"
STAGE=${1:-}
OBS=${2:-}
case "$STAGE" in tune|apply|final) ;; *) echo "usage: $0 tune|apply|final linear|arctan" >&2; exit 2 ;; esac
case "$OBS" in linear|arctan) ;; *) echo "usage: $0 tune|apply|final linear|arctan (not '$OBS')" >&2; exit 2 ;; esac
PY=${PY:-python}
THREADS=${THREADS:-2}
SEEDS=${SEEDS:-"0 1 2"}
FORCE=${FORCE:-0}
ONLY=${ONLY:-}
ROOT=results/lorenz96/$OBS
DRIVER=(cases/lorenz96/driver.py --obs "$OBS")

# --------------------------------------------------------------------------- #
# THE TUNING GRIDS. One row per job: "stage|cell|parent|grid JSON".
#   parent  stage 2 only: the stage-1 cell whose picks are passed as --set (keys not
#           in this grid) and that {"times":[..]} values refer to. The _fc and SURGE
#           cells around Ours inherit their parent's block, so this gives them the
#           stage-1 sampler picks before `apply` writes those into the config.
# M is fixed at 50 and not tuned (the driver refuses a num_steps grid).
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
  # SURGE on our samplers: the guidance is the plain cell's likelihood, so it reads
  # likelihood_mode / jacobian_damping (and diffusion_scale) of the parent.
  '2|surge_si_sde|si_sde|{"ess_threshold":[0.25,0.5,1.0]}'
  '2|surge_dm_sde|dm_sde|{"ess_threshold":[0.25,0.5,1.0]}'
  '2|surge_unguided_si||{"ess_threshold":[0.25,0.5,1.0]}'
  '2|surge_unguided_dm||{"ess_threshold":[0.25,0.5,1.0]}'
  '2|surge_flowdas|flowdas|{"guidance_scale":{"times":[0.25,1,4]},"ess_threshold":[0.5,1.0]}'
  '2|surge_sda|sda|{"gamma_sda":{"times":[0.1,1,10]},"ess_threshold":[0.5,1.0]}'
)
# --------------------------------------------------------------------------- #

# tune_stage <k>: every GRIDS job of stage k (and of ONLY, if set).
tune_stage() {
  local want=$1 selected=0 stage cell parent grid out state csv aside
  for row in "${GRIDS[@]}"; do
    IFS='|' read -r stage cell parent grid <<<"$row"
    [[ "$stage" == "$want" ]] || continue
    if [[ -n "$ONLY" && " $ONLY " != *" $cell "* ]]; then continue; fi
    selected=$((selected + 1))
    local sets=()
    if [[ -n "$parent" ]]; then
      # Line 1: the resolved grid; then one --set per line (the parent's picks).
      out=$("$PY" select_lorenz96_tuning.py --obs "$OBS" resolve --parent "$parent" --grid "$grid")
      mapfile -t resolved <<<"$out"
      if [[ -z "${resolved[0]:-}" ]]; then
        echo "could not resolve the stage-2 grid of $cell from $parent" >&2
        return 1
      fi
      grid=${resolved[0]}
      for entry in "${resolved[@]:1}"; do sets+=(--set "$entry"); done
    fi
    out=$("$PY" select_lorenz96_tuning.py --obs "$OBS" status --cell "$cell" --grid "$grid" \
      --stage "$stage" "${sets[@]}")
    read -r state csv <<<"$out"
    case "$state" in
      complete) echo "skip $OBS stage $stage $cell: $csv has every grid point"; continue ;;
      partial)
        aside="$csv.partial.$(date +%Y%m%d-%H%M%S)"
        echo "partial $csv: moved to $aside, rerunning $cell in full"
        mv "$csv" "$aside" ;;
      missing) ;;
      *) echo "unexpected status '$state' for $cell" >&2; return 1 ;;
    esac
    echo "tune $OBS stage $stage $cell grid=$grid ${sets[*]:-}"
    "$PY" "${DRIVER[@]}" tune --method "$cell" --grid "$grid" --tag "s$stage" \
      --threads "$THREADS" "${sets[@]}"
  done
  if [[ "$selected" -eq 0 ]]; then
    echo "no stage-$want job matched ONLY='$ONLY'" >&2
    return 1
  fi
}

# run_cell <cell> <seed> <M or ->: "-" for a cell without M. Returns 1 on failure.
run_cell() {
  local cell=$1 seed=$2 m=$3 token= extra=()
  if [[ "$m" != - ]]; then token=__M$m; extra=(--num-steps "$m"); fi
  local out="$ROOT/metrics/lorenz96__${cell}${token}__traj${seed}__.csv"
  if [[ "$FORCE" != 1 && -s "$out" ]]; then
    echo "skip $OBS $cell M=$m seed $seed: $out exists (FORCE=1 reruns it)"
    return 0
  fi
  echo "run $OBS $cell M=$m seed $seed"
  "$PY" "${DRIVER[@]}" cell --method "$cell" --seed "$seed" "${extra[@]}"
}

stage_tune() {
  "$PY" "${DRIVER[@]}" reference --split validation
  tune_stage 1
  "$PY" select_lorenz96_tuning.py --obs "$OBS" --stage 1
  tune_stage 2
  "$PY" select_lorenz96_tuning.py --obs "$OBS" --stage 2
  echo
  echo "STOP: tuning of the $OBS sweep done. Review"
  echo "  $ROOT/tuning/selected_stage1.md and selected_stage2.md, then run"
  echo "  ./run_lorenz96.sh apply $OBS      # writes the picks into configs/case/lorenz96.yaml"
  echo "  ./run_lorenz96.sh final $OBS      # after approving the compute"
}

stage_final() {
  local generative fixed m_values ordered=() failed=0
  read -ra generative <<<"$("$PY" lorenz96_lineup.py cells generative)"
  read -ra fixed <<<"$("$PY" lorenz96_lineup.py cells fixed)"
  m_values=${M_VALUES:-$("$PY" lorenz96_lineup.py m-values)}
  [[ " $m_values " == *" 50 "* ]] && ordered=(50)
  for m in $m_values; do [[ "$m" == 50 ]] || ordered+=("$m"); done
  "$PY" "${DRIVER[@]}" reference --split test
  if [[ -s "$ROOT/prior_quality.json" ]]; then
    echo "skip prior-quality: $ROOT/prior_quality.json exists"
  else
    "$PY" "${DRIVER[@]}" prior-quality
  fi
  local first=1
  for m in "${ordered[@]}"; do
    for seed in $SEEDS; do
      for cell in "${generative[@]}"; do run_cell "$cell" "$seed" "$m" || failed=1; done
      if [[ "$first" == 1 ]]; then
        for cell in "${fixed[@]}"; do run_cell "$cell" "$seed" - || failed=1; done
      fi
    done
    first=0
  done
  if [[ "$failed" != 0 ]]; then
    echo "a cell failed: stopping before pf_reference and the aggregation (rerun to resume)" >&2
    return 1
  fi
  # The reference does not depend on a seed.
  run_cell pf_reference 0 -
  "$PY" aggregate_lorenz96.py --obs "$OBS"
  "$PY" make_lorenz96_figures.py --obs "$OBS"
  "$PY" make_lorenz96_tables.py --obs "$OBS"
}

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/${STAGE}_$(date +%Y%m%d-%H%M%S).log"
trap 'status=$?; echo "$status" > "$LOG.exit"; echo "exit $status (log: $LOG)"' EXIT
exec > >(tee -a "$LOG") 2>&1
echo "lorenz96 $STAGE OBS=$OBS start $(date -Is) commit $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
case "$STAGE" in
  tune) stage_tune ;;
  apply) "$PY" select_lorenz96_tuning.py --obs "$OBS" apply ;;
  final) stage_final ;;
esac
echo "lorenz96 $STAGE OBS=$OBS done $(date -Is)"
