#!/bin/bash
# Submit cells of the E=1000 r=20 EnKF grid (poor-IC NS) as a Slurm array on a GPU partition.
#   CELLS=0-9 ACCOUNT=innovation bash submit.sh          # V100, account innovation (1 job at a time)
#   CELLS=10-19 bash submit.sh                           # V100, research-ceg-gse
#   CELLS=... PARTITION=gpu-a100-small bash submit.sh    # MIG slice (memory for E=1000 unverified)
#   DRY_RUN=1 ...
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CELLS=${CELLS:-0-19}; ACCOUNT=${ACCOUNT:-research-ceg-gse}; PARTITION=${PARTITION:-gpu-v100}
TIME=${TIME:-04:00:00}; THROTTLE=${THROTTLE:-8}; DRY_RUN=${DRY_RUN:-0}
ROOT=${ROOT:-/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/results_wrong_ic_np40_enkf_r20_e1000/navier_stokes}
if [ "$PARTITION" = gpu-a100-small ]; then RES=(-c 2 --gres=gpu:1 --mem-per-cpu=7900M); else RES=(-c 8 --gres=gpu:1 --mem-per-cpu=5000M); fi
[ "$ACCOUNT" = innovation ] && THROTTLE=1
CMD=(sbatch --parsable --job-name=enkf_r20_e1000 --partition="$PARTITION" --account="$ACCOUNT" -n 1 "${RES[@]}"
  --time="$TIME" --array="${CELLS}%${THROTTLE}" --output="$ROOT/slurm/%x_%A_%a.out" --export=ALL,ROOT="$ROOT"
  --wrap "bash '$ROOT/launcher/run_cell.sh' \$SLURM_ARRAY_TASK_ID")
echo "cells [$CELLS] -> $ROOT"; echo "${CMD[*]}"
[ "$DRY_RUN" = 1 ] && exit 0
mkdir -p "$ROOT/slurm" "$ROOT/launcher"
# the jobs run a copy of run_cell.sh kept with the results (later repo edits cannot change queued cells)
if [ -f "$ROOT/launcher/run_cell.sh" ]; then cmp -s "$HERE/run_cell.sh" "$ROOT/launcher/run_cell.sh" || { echo "FATAL: run_cell.sh differs from the copy in $ROOT/launcher" >&2; exit 1; }
else cp -p "$HERE/run_cell.sh" "$ROOT/launcher/run_cell.sh"; fi
J=$("${CMD[@]}"); echo "array $J"
printf '=== %s  slurm_enkf_e1000/submit.sh cells=[%s] partition=%s account=%s array=%s | EnKF E=1000, GC r=20 all scenarios, inflation default (1.0), ic_index=15 NP=40 traj 10-14, no KL | code: frozen copy code_ns_wrong_ic (README_FROZEN.txt) | launcher repo commit %s\n' \
  "$(date -Is)" "$CELLS" "$PARTITION" "$ACCOUNT" "$J" "$(git -C "$HERE" rev-parse --short HEAD)" >> "$ROOT/run_config.txt"
