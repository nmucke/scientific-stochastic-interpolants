#!/bin/bash
# Phase 0 diagnostics on the existing SI checkpoint, as resumable stages.
#   STAGES="tests_before tests_after budget d1_smoke d1" sbatch research/ito_maps/experiments/slurm_phase0/phase0.sbatch
# Stages skip when their output exists, so a resubmission continues. Smoke
# stages run first in the same job; with set -e a failing smoke stops the job.
# The shared .venv (torch 2.15+cu132) has no sm_70 kernels: no V100.
#SBATCH --job-name=p0_diag
#SBATCH --account=research-ceg-gse
#SBATCH --partition=gpu-a100-small
#SBATCH --time=03:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=7000M
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/ntmucke/scientific-stochastic-interpolants/slurm_itomap/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=$PWD/.venv/bin/python
EXP=research/ito_maps/experiments
RES=research/ito_maps/results/phase0
ARR=/scratch/ntmucke/scientific-stochastic-interpolants/results_itomap/phase0
CACHE=$ARR/cache
SI=checkpoints/stochastic_navier_stokes/stochastic_interpolant_small
PRISTINE=/scratch/ntmucke/scientific-stochastic-interpolants/wt_itomap_phase0_pristine
mkdir -p $RES $ARR $CACHE
echo "$(date '+%F %T') job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short HEAD), dirty files $(git status --porcelain | wc -l)"
nvidia-smi -L || true
$PY -c "import scisi, torch; print('scisi from', scisi.__file__, torch.__version__, torch.cuda.get_device_name(0))"
run() { echo "$(date '+%F %T') == stage $1"; }

for stage in ${STAGES:-cache d1_smoke short_smoke long_smoke budget d1}; do
  case $stage in
  cache)
    run $stage
    $PY $EXP/phase0_common.py --checkpoint-dir $SI --cache-dir $CACHE ;;
  tests_before)
    run $stage
    (cd $PRISTINE && PYTHONPATH=$PRISTINE/src $PY -m pytest tests research/ito_maps/tests -q -p no:cacheprovider 2>&1 | tail -25) || true ;;
  tests_after)
    run $stage
    ($PY -m pytest tests research/ito_maps/tests -q -p no:cacheprovider 2>&1 | tail -25) || true ;;
  budget)
    run $stage
    [ -f $RES/budget_archive.json ] || $PY $EXP/ns_budget.py --checkpoint-dir $SI --data-cache $CACHE --output $RES/budget_archive.json ;;
  d1_smoke)
    run $stage
    $PY $EXP/uncertainty_spectrum.py --checkpoint-dir $SI --data-cache $CACHE \
      --max-val-trajectories 2 --max-train-trajectories 2 --origins 5 45 --ensemble 4 --steps 8 --bootstrap 50 \
      --arrays-dir $ARR/smoke/d1_$SLURM_JOB_ID --output $ARR/smoke/d1_$SLURM_JOB_ID/d1.json ;;
  d1)
    run $stage
    [ -f $RES/d1_uncertainty_spectrum.json ] || $PY $EXP/uncertainty_spectrum.py --checkpoint-dir $SI --data-cache $CACHE \
      --ensemble 16 --steps 128 --si-noise em exact \
      --arrays-dir $ARR/d1 --output $RES/d1_uncertainty_spectrum.json ;;
  short_smoke)
    run $stage
    $PY $EXP/rollout_diagnostics.py short --checkpoint-dir $SI --data-cache $CACHE --trajectories 1 --origins 5 \
      --ensemble 3 --leads 2 --steps 4 --arrays-dir $ARR/smoke/short_$SLURM_JOB_ID --output $ARR/smoke/short_$SLURM_JOB_ID/short.json ;;
  short)
    run $stage
    [ -f $RES/d5_d6_short_si128_exact.json ] || $PY $EXP/rollout_diagnostics.py short --checkpoint-dir $SI --data-cache $CACHE \
      --trajectories 4 --origins 5 45 85 --ensemble 8 --leads 10 --steps 128 --si-noise exact \
      --arrays-dir $ARR/short_si128_exact --output $RES/d5_d6_short_si128_exact.json ;;
  long_smoke)
    run $stage
    out=$ARR/smoke/long_$SLURM_JOB_ID
    $PY $EXP/long_horizon_ns.py --checkpoint-dir $SI --output $out --steps 4 --horizon 3 --ensemble 2 \
      --trajectories 160 --origin 5 --raw-every 1 --raw-dense 3 --checkpoint-every 1 --si-noise exact --num-threads 2
    $PY $EXP/long_horizon_ns.py --checkpoint-dir $SI --output $out --steps 4 --horizon 5 --ensemble 2 \
      --trajectories 160 --origin 5 --raw-every 1 --raw-dense 5 --checkpoint-every 1 --si-noise exact --num-threads 2 --resume
    $PY $EXP/rollout_diagnostics.py long --checkpoint-dir $SI --run-dir $out --data-cache $CACHE --burn-in 1 --output $out/long.json ;;
  long)
    run $stage
    out=$ARR/long_si128_exact
    resume=(); [ -f $out/restart.pt ] && resume=(--resume)
    $PY $EXP/long_horizon_ns.py --checkpoint-dir $SI --output $out --steps 128 --horizon ${HORIZON:-500} --ensemble 8 \
      --trajectories 160 161 162 163 --origin 5 --raw-every 1 --raw-dense 500 --checkpoint-every 10 --si-noise exact \
      --num-threads 2 "${resume[@]}"
    $PY $EXP/rollout_diagnostics.py long --checkpoint-dir $SI --run-dir $out --data-cache $CACHE --burn-in 100 \
      --output $RES/d5_d6_d7_long_si128_exact.json ;;
  *) echo "unknown stage $stage"; exit 2 ;;
  esac
done
echo "$(date '+%F %T') done"
