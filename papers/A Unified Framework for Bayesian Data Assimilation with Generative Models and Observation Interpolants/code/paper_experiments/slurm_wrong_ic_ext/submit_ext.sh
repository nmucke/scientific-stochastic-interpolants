#!/bin/bash
# Wrong-IC Navier--Stokes EXTENSION grid: SURGE on Ours, and Ours + forecast correction.
# Same setting as ../slurm_wrong_ic (foreign IC, NP=40, M=100, E=64, traj 10..14, 4
# scenarios, no KL) but a separate results tree, and the LIVE repo code (not the
# frozen copy that the main grid runs from).
#
#   DRY_RUN=1 bash submit_ext.sh          # write manifest + print the sbatch call only
#   GROUPS_EXT=fc bash submit_ext.sh      # one group only (default: surge_ours,fc)
#   FC_OVERRIDES="forecast_num_steps=25"  # space-separated keys of the FC block
#
# One array, one method per task, %THROTTLE concurrent. The account allows 64 running
# jobs in total; the main grid's frozen requeue array uses up to 40 + 6 live ones, so
# the default THROTTLE=16 keeps the sum under 64.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$HERE/../common/paths.sh"
PE=$PE_SCRIPT_DIR; REPO=$PE_REPO_ROOT
ROOT=${ROOT:-/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/results_wrong_ic_np40_ext/navier_stokes}
TRAJ=${TRAJ:-10 11 12 13 14}; IC_INDEX=${IC_INDEX:-15}; NP=${NP:-40}; M=${M:-100}; E=${E:-64}; SAVE_TRAJ=${SAVE_TRAJ:-11}
GROUPS_EXT=${GROUPS_EXT:-surge_ours,fc}; FC_OVERRIDES=${FC_OVERRIDES:-}
PARTITION=${PARTITION:-gpu-v100}; ACCOUNT=${ACCOUNT:-research-ceg-gse}; GRES=${GRES:-gpu:1}
CPUS=${CPUS:-8}; MEM_PER_CPU=${MEM_PER_CPU:-5000M}; TIME_CELL=${TIME_CELL:-24:00:00}
THROTTLE=${THROTTLE:-16}; DRY_RUN=${DRY_RUN:-0}
PY=${PY:-/scratch/ntmucke/scientific-stochastic-interpolants/.venv-cu126/bin/python}

if echo " $TRAJ " | grep -q " $IC_INDEX "; then echo "FATAL: IC_INDEX=$IC_INDEX is also in TRAJ=[$TRAJ]" >&2; exit 1; fi
echo " $TRAJ " | grep -q " $SAVE_TRAJ " || { echo "FATAL: SAVE_TRAJ=$SAVE_TRAJ not in TRAJ" >&2; exit 1; }
case "$ROOT" in */results/*|*/results|*/results_wrong_ic_np40/*|*/results_wrong_ic_np40)
  echo "FATAL: ROOT=$ROOT is an existing tree; the extension needs its own" >&2; exit 1;; esac
case "$REPO" in */code_ns_wrong_ic*) echo "FATAL: run from the live repo, not the frozen copy" >&2; exit 1;; esac

# SURGE-on-Ours must exist in the method registry before submitting that group.
if [[ ",$GROUPS_EXT," == *",surge_ours,"* ]]; then
  for m in "SURGE (SI-SDE)" "SURGE (DM-SDE)"; do
    grep -qF "\"$m\"" "$PE/results_schema.py" || {
      echo "FATAL: method '$m' is not in results_schema.py yet (GROUPS_EXT=fc to skip)" >&2; exit 1; }
  done
fi

# Forecast-correction tag from the SAME composed config the runs use.
FC_TAG=-
if [[ ",$GROUPS_EXT," == *",fc,"* ]]; then
  FC_ARGS=(case.forecast_correction.enabled=true)
  for kv in $FC_OVERRIDES; do FC_ARGS+=("case.forecast_correction.$kv"); done
  FC_TAG=$(cd "$PE" && "$PY" -m common.forecast_correction tag navier_stokes "${FC_ARGS[@]}") \
    || { echo "FATAL: bad FC_OVERRIDES='$FC_OVERRIDES'" >&2; exit 1; }
fi

MDIR="$ROOT/manifests"; mkdir -p "$ROOT"/{metrics,per_step,states,logs,slurm,forecast_correction}
python3 "$HERE/make_manifest_ext.py" "$ROOT" "$MDIR" "$IC_INDEX" "$NP" "$M" "$E" "$SAVE_TRAJ" \
  "$GROUPS_EXT" "$FC_TAG" "${FC_OVERRIDES:--}" $TRAJ
N=$(wc -l < "$MDIR/ext_cells.tsv")

STAMP=$(date +%Y%m%d_%H%M%S)
git -C "$REPO" diff HEAD > "$ROOT/code_${STAMP}.diff"
cat >> "$ROOT/run_config.txt" <<C
=== $(date -Is)  submit_ext.sh
commit: $(git -C "$REPO" rev-parse HEAD)  dirty_files: $(git -C "$REPO" status --porcelain | wc -l)  diff: code_${STAMP}.diff
repo: $REPO (live tree)  python: $PY
traj: $TRAJ | ic_index: $IC_INDEX | num_physical_steps: $NP | M: $M | E: $E | save_traj: $SAVE_TRAJ | no KL (+skip_kl_reference=true)
groups: $GROUPS_EXT | fc_tag: $FC_TAG | fc_overrides: ${FC_OVERRIDES:-<defaults>}
slurm: $PARTITION $ACCOUNT $GRES cpus=$CPUS mem=$MEM_PER_CPU time=$TIME_CELL throttle=$THROTTLE tasks=$N
C

export REPO ROOT PE PY PACK=1
CMD=(sbatch --parsable --job-name=nsic_ext --partition="$PARTITION" --account="$ACCOUNT" --ntasks=1
  --cpus-per-task="$CPUS" --mem-per-cpu="$MEM_PER_CPU" --gres="$GRES" --time="$TIME_CELL"
  --array="0-$((N-1))%$THROTTLE" --output="$ROOT/slurm/%x_%A_%a.out" --export=ALL
  "$HERE/run_task.sh" "$MDIR/ext_cells.tsv")
if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN: ${CMD[*]}"; else J=$("${CMD[@]}"); echo "extension array: $J ($N tasks)"; echo "job: $J" >> "$ROOT/run_config.txt"; fi
