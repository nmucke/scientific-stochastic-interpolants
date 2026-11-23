#!/bin/bash
# Smoke test of every WP1 step-1 launcher on a MIG slice, tiny budgets:
# increment spectrum (4 batches) -> 1-epoch SI drift on each process (+ resume
# to 2 epochs) -> path compression and exact-noise Euler on the matched one ->
# 1-epoch regressor -> residual spectrum. Outputs under results_itomap/wp1_smoke_<job>.
#SBATCH --job-name=wp1_smoke
#SBATCH --account=research-ceg-gse
#SBATCH --partition=gpu-a100-small
#SBATCH --time=03:55:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=7000M
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/ntmucke/scientific-stochastic-interpolants/slurm_itomap/logs/%x_%j.out
set -euo pipefail
D=research/ito_maps/experiments/slurm_wp1
export WP1_ROOT=/scratch/ntmucke/scientific-stochastic-interpolants/results_itomap/wp1_smoke_${SLURM_JOB_ID}
TAG=wp1_smoke_${SLURM_JOB_ID}
TINY="+trainer.max_train_batches=10 +trainer.max_val_batches=2"
stamp() { echo "=== $(date '+%F %T') $*"; }

stamp spectrum
MODE=increment NAME=increment_smoke MAX_BATCHES=4 bash $D/estimate_spectrum.sbatch
export SPECTRUM=$WP1_ROOT/spectra/increment_smoke.npz

for variant in white_t2 coloured_t coloured_matched; do
  stamp train $variant
  VARIANT=$variant RUN=${TAG}_$variant EPOCHS=1 EXTRA="$TINY" bash $D/train_si.sbatch
done
stamp resume coloured_matched to 2 epochs
rm checkpoints/stochastic_navier_stokes/${TAG}_coloured_matched/DONE
VARIANT=coloured_matched RUN=${TAG}_coloured_matched EPOCHS=2 EXTRA="$TINY" bash $D/train_si.sbatch

stamp eval coloured_matched
RUN=${TAG}_coloured_matched HISTORIES="160:45" GROUPS_=2 REPLICAS=2 TEACHER_STEPS=8 \
  EULER_ARGS="--trajectories 160 --origins 45 --ensemble 4 --steps 1 2 --reference-steps 8" \
  bash $D/eval_si.sbatch

stamp regressor
RUN=${TAG}_regressor EPOCHS=1 EXTRA="$TINY" bash $D/train_regressor.sbatch
stamp residual spectrum
MODE=residuals REGRESSOR=checkpoints/stochastic_navier_stokes/${TAG}_regressor NAME=residuals_smoke \
  MAX_BATCHES=2 bash $D/estimate_spectrum.sbatch
stamp done
ls -la $WP1_ROOT $WP1_ROOT/spectra checkpoints/stochastic_navier_stokes/${TAG}_*
