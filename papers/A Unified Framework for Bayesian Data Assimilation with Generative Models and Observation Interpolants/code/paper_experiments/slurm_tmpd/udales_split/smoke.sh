#!/bin/bash
# End-to-end smoke of the split uDALES TMPD launcher: traj 1, sparse 1.5625%, NP=11
# (DA at physical steps 5 and 8), 2 parts, then merge.sh (verify + merge). Uses
# make_manifest.py, run_task.sh and merge.sh unchanged.
#   bash smoke.sh mig   # 2 parts x 16 members (the production part size), M=50, on
#                       # 2 MIG slices in one job (queues behind other MIG jobs)
#   bash smoke.sh cpu   # 2 parts x 2 members, M=5, here on the CPU (logic only)
#   DRY_RUN=1 bash smoke.sh <mode>
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"; SPLIT_DIR=$HERE
SLURM_DIR="$(cd -- "$HERE/.." && pwd -P)"
source "$SLURM_DIR/../common/paths.sh"
PE=$PE_SCRIPT_DIR; REPO=$PE_REPO_ROOT; MODE=${1:-mig}; DRY_RUN=${DRY_RUN:-0}
case "$REPO" in */wt_tmpd_ns_urban|*/wt_tmpd_ns_urban/*) echo "FATAL: REPO=$REPO" >&2; exit 1;; esac
PY=${PY:-/scratch/ntmucke/scientific-stochastic-interpolants/.venv-cu126/bin/python}
ROOT=${ROOT:-/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/results_tmpd_smoke/udales_split_$MODE/urban}
case "$MODE" in
  mig) SIZE=(--ensemble 32);; cpu) SIZE=(--ensemble 4 --num-steps 5);;
  *) echo "usage: smoke.sh mig|cpu" >&2; exit 1;; esac
STAMP=$(date +%Y%m%d_%H%M%S); MD="$ROOT/manifests/smoke_$STAMP"
[ "$DRY_RUN" = 1 ] && MD=$(mktemp -d)
python3 "$SLURM_DIR/make_manifest.py" udales "$ROOT" "$MD" --traj 1 --scenarios "sparse 1.5625%" \
  --np 11 --parts 2 --tmpd-batch 8 "${SIZE[@]}"
if [ "$DRY_RUN" = 1 ]; then cut -f1,3 "$MD/cells.tsv"; rm -rf "$MD"; exit 0; fi
mkdir -p "$ROOT"/{slurm,logs}
printf '=== %s udales_split/smoke.sh %s commit %s (%s) manifest %s\n' "$(date -Is)" "$MODE" \
  "$(git -C "$REPO" rev-parse HEAD)" "$(git -C "$REPO" rev-parse --abbrev-ref HEAD)" "$MD" >> "$ROOT/run_config.txt"
git -C "$REPO" diff HEAD > "$ROOT/code_${STAMP}.diff"
export REPO ROOT PE PY HERE="$SLURM_DIR"
MERGE="ROOT='$ROOT' PARTS=2 bash '$SPLIT_DIR/merge.sh'"
if [ "$MODE" = cpu ]; then
  DEVICE=cpu ALL=1 bash "$SLURM_DIR/run_task.sh" "$MD/cells.tsv" && eval "$MERGE"
else
  sbatch --parsable -J tmpd_udsplit_smoke -p gpu-a100-small -A research-ceg-gse -n 2 -c 2 --gres=gpu:2 \
    --mem-per-cpu=7900M -t 04:00:00 -o "$ROOT/slurm/%x_%j.out" --export=ALL,PACK=2 --wrap "
      nvidia-smi -L
      SLURM_ARRAY_TASK_ID=0 bash '$SLURM_DIR/run_task.sh' '$MD/cells.tsv' && $MERGE
      echo SMOKEDONE"
fi
