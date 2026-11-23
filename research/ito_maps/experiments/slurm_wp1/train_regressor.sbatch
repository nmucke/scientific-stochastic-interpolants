#!/bin/bash
# Deterministic conditional-mean regressor m(h) = c + UNet(c, h) with MSE, for
# the preferred spectrum estimate (validation residuals). Resumable; DONE marker.
#   sbatch research/ito_maps/experiments/slurm_wp1/train_regressor.sbatch
# Not submitted yet (prepared only).
#SBATCH --job-name=wp1_regressor
#SBATCH --account=research-ceg-gse
#SBATCH --partition=gpu-a100
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=12000M
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/ntmucke/scientific-stochastic-interpolants/slurm_itomap/logs/%x_%j.out
source research/ito_maps/experiments/slurm_wp1/common.sh
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-150}
EXTRA=${EXTRA:-}
RUN=${RUN:-wp1_conditional_mean_seed${SEED}}
ckpt="checkpoints/stochastic_navier_stokes/$RUN"
if [ -f "$ckpt/DONE" ]; then echo "$RUN already done"; exit 0; fi
resume=()
if [ -f "$ckpt/last.pth" ]; then resume=(trainer.resume_from="$ckpt/last.pth"); fi
echo "$(date '+%F %T') start $RUN on $(hostname), commit $(git rev-parse --short HEAD)"
$PY scripts/train.py --config-name stochastic_navier_stokes \
  models=conditional_mean \
  experiment_tracking.name="$RUN" \
  seed="$SEED" \
  trainer.num_epochs="$EPOCHS" \
  scheduler.T_max="$EPOCHS" \
  hydra.run.dir="$WP1_ROOT/hydra/$RUN" \
  "${resume[@]}" $EXTRA
date '+%F %T' > "$ckpt/DONE"
echo "$(date '+%F %T') done $RUN"
