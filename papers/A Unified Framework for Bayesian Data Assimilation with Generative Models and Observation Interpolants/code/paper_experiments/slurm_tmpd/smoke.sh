#!/bin/bash
# Smoke + cost test of TMPD on NS and uDALES, on two gpu-a100-small MIG slices (one job).
#   lane A (TMPD):          NS wrong-IC 32^2 and sparse 5% (traj 11, NP=7: 2 DA steps);
#                           uDALES 1.5625% (traj 1, NP=11: DA at steps 5 and 8)
#   lane B (Ours (DM-SDE)): the same NS cells (jacfree) and the uDALES cell
#                           (shared_jac10, the manuscript row, then jacfree)
# Every other setting is the production grid's. Uses make_manifest.py + run_task.sh
# unchanged, so it also tests the launcher. Peak memory: logs/timing.tsv.
#   DRY_RUN=1 bash smoke.sh   |   bash smoke.sh
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$HERE/../common/paths.sh"
PE=$PE_SCRIPT_DIR; REPO=$PE_REPO_ROOT; DRY_RUN=${DRY_RUN:-0}
ROOT=${ROOT:-/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/results_tmpd_smoke}
PY=${PY:-/scratch/ntmucke/scientific-stochastic-interpolants/.venv-cu126/bin/python}
STAMP=$(date +%Y%m%d_%H%M%S); MD="$ROOT/manifests/smoke_$STAMP"
[ "$DRY_RUN" = 1 ] && MD=$(mktemp -d)
mk() {  # <dir> <grid> <methods> <args...>
  local d=$1 g=$2 m=$3; shift 3
  local r="$ROOT/navier_stokes"; [ "$g" = udales ] && r="$ROOT/urban"
  python3 "$HERE/make_manifest.py" "$g" "$r" "$d" --methods "$m" "$@" >/dev/null; }
NS=(--np 7 --traj 11 --scenarios "32^2->128^2|sparse 5%")
UD=(--np 11 --traj 1 --scenarios "sparse 1.5625%")
mk "$MD/a_ns" ns_wrong_ic tmpd "${NS[@]}"; mk "$MD/a_ud" udales tmpd "${UD[@]}"
mk "$MD/b_ns" ns_wrong_ic dm_sde_jacfree "${NS[@]}"
mk "$MD/b_ud" udales dm_sde_shared_jac10,dm_sde_jacfree "${UD[@]}"
cat "$MD/a_ns/cells.tsv" "$MD/a_ud/cells.tsv" > "$MD/lane_a.tsv"
cat "$MD/b_ns/cells.tsv" "$MD/b_ud/cells.tsv" > "$MD/lane_b.tsv"
if [ "$DRY_RUN" = 1 ]; then
  for l in a b; do echo "lane $l:"; cut -f1,3 "$MD/lane_$l.tsv"; done; rm -rf "$MD"; exit 0; fi
mkdir -p "$ROOT"/{slurm,logs}
git -C "$REPO" diff HEAD > "$ROOT/code_${STAMP}.diff"
printf '=== %s smoke.sh commit %s (%s) diff code_%s.diff manifests %s\n' "$(date -Is)" \
  "$(git -C "$REPO" rev-parse HEAD)" "$(git -C "$REPO" rev-parse --abbrev-ref HEAD)" "$STAMP" "$MD" \
  >> "$ROOT/run_config.txt"
export REPO ROOT PE PY HERE ALL=1
LANE="bash '$HERE/run_task.sh'"
sbatch --parsable -J tmpd_nsud_smoke -p gpu-a100-small -A research-ceg-gse -n 2 -c 2 --gres=gpu:2 \
  --mem-per-cpu=7900M -t 04:00:00 -o "$ROOT/slurm/%x_%j.out" --export=ALL --wrap "
    nvidia-smi -L
    srun --exact -n1 -c2 --gres=gpu:1 --mem=15000M bash -c \"$LANE '$MD/lane_a.tsv'\" > '$ROOT/slurm/laneA_\$SLURM_JOB_ID.out' 2>&1 &
    srun --exact -n1 -c2 --gres=gpu:1 --mem=15000M bash -c \"$LANE '$MD/lane_b.tsv'\" > '$ROOT/slurm/laneB_\$SLURM_JOB_ID.out' 2>&1 &
    wait; echo SMOKEDONE"
