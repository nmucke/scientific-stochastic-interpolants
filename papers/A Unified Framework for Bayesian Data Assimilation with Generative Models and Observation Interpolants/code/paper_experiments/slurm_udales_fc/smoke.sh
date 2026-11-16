#!/bin/bash
# Smoke + cost test for the uDALES FC launcher, manuscript settings with a short
# horizon: NP=11 (5 history + 6 generated; DA at physical steps 5 and 8), traj 1,
# sparse 1.5625%, E=64, M=50, shared_jac10. Two lanes (one GPU each):
#   lane A: SI-SDE plain -> SI-SDE FC (forecast_num_steps=M) -> SI-SDE FC (25)
#   lane B: DM-SDE FC (M) -> FM-ODE FC (M)
# Uses make_manifest.py + run_task.sh unchanged, so it also tests the launcher.
#
#   bash smoke.sh mig     # gpu-a100-small, 2 MIG slices, 4 h (user's default for tests)
#   bash smoke.sh v100    # two 1-GPU gpu-v100 jobs, 2 h each (if it does not fit 9.5 GiB)
#   DRY_RUN=1 bash smoke.sh mig
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$HERE/../common/paths.sh"
PE=$PE_SCRIPT_DIR; REPO=$PE_REPO_ROOT
WHERE=${1:-mig}; DRY_RUN=${DRY_RUN:-0}
ROOT=${ROOT:-/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/results_udales_fc/smoke}
PY=${PY:-/scratch/ntmucke/scientific-stochastic-interpolants/.venv-cu126/bin/python}
NP=${NP:-11}; SCEN=${SCEN:-sparse 1.5625%}; N=${N:-1}

tag() { (cd "$PE" && "$PY" -m common.forecast_correction tag urban case.forecast_correction.enabled=true "$@" 2>/dev/null | tail -1); }
TAG_M=$(tag); TAG_25=$(tag case.forecast_correction.forecast_num_steps=25)
STAMP=$(date +%Y%m%d_%H%M%S); MD="$ROOT/manifests/smoke_$STAMP"
mk() {  # <dir> <methods> <include_plain> <fc_tag> <fc_overrides|->
  python3 "$HERE/make_manifest.py" "$ROOT" "$1" 8 "$NP" 50 64 10 3 5 "$N" "$2" "$3" "$4" "$5" - "$SCEN" "$N"; }
mk "$MD/si_M" si_sde 1 "$TAG_M" -
mk "$MD/si_25" si_sde 0 "$TAG_25" "forecast_num_steps=25"
mk "$MD/dmfm_M" dm_sde,fm_ode 0 "$TAG_M" -
# task ids: si_M:0 (plain+fc), si_25:0, dmfm_M:0 (dm), dmfm_M:1 (fm)
LANE_A="SLURM_ARRAY_TASK_ID=0 bash '$HERE/run_task.sh' '$MD/si_M'; SLURM_ARRAY_TASK_ID=0 bash '$HERE/run_task.sh' '$MD/si_25'"
LANE_B="SLURM_ARRAY_TASK_ID=0 bash '$HERE/run_task.sh' '$MD/dmfm_M'; SLURM_ARRAY_TASK_ID=1 bash '$HERE/run_task.sh' '$MD/dmfm_M'"
mkdir -p "$ROOT"/{slurm,logs}
export REPO ROOT PE PY HERE
echo "fc tags: M=$TAG_M 25=$TAG_25 | manifests $MD"
if [ "$DRY_RUN" = 1 ]; then cat "$MD"/*/cells.tsv | cut -f2,3; echo "lane A: $LANE_A"; echo "lane B: $LANE_B"; exit 0; fi
git -C "$REPO" diff HEAD > "$ROOT/code_${STAMP}.diff"
printf '=== %s smoke.sh %s commit %s diff code_%s.diff NP=%s traj=%s scen=%s\n' "$(date -Is)" "$WHERE" \
  "$(git -C "$REPO" rev-parse HEAD)" "$STAMP" "$NP" "$N" "$SCEN" >> "$ROOT/run_config.txt"
case "$WHERE" in
  mig)
    sbatch --parsable -J udfc_smoke -p gpu-a100-small -A research-ceg-gse -n 2 -c 2 --gres=gpu:2 \
      --mem-per-cpu=7900M -t 04:00:00 -o "$ROOT/slurm/%x_%j.out" --export=ALL --wrap "
        nvidia-smi -L
        srun --exact -n1 -c2 --gres=gpu:1 --mem=15000M bash -c \"export SLURM_ARRAY_TASK_ID=0; $LANE_A\" > '$ROOT/slurm/laneA_\$SLURM_JOB_ID.out' 2>&1 &
        srun --exact -n1 -c2 --gres=gpu:1 --mem=15000M bash -c \"export SLURM_ARRAY_TASK_ID=0; $LANE_B\" > '$ROOT/slurm/laneB_\$SLURM_JOB_ID.out' 2>&1 &
        wait; echo SMOKEDONE";;
  v100)
    for L in A B; do
      eval "C=\$LANE_$L"
      sbatch --parsable -J "udfc_smoke$L" -p gpu-v100 -A research-ceg-gse --ntasks=1 --cpus-per-task=8 \
        --mem-per-cpu=5000M --gres=gpu:1 -t 02:00:00 -o "$ROOT/slurm/%x_%j.out" --export=ALL \
        --wrap "nvidia-smi -L; $C"
    done;;
  *) echo "usage: smoke.sh mig|v100" >&2; exit 1;;
esac
