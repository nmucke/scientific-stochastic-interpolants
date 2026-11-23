#!/bin/bash
# Smoke test for train_baseline_map.sbatch: both baseline models, 3 tiny epochs
# (1 diagonal, 1 ramp, 1 full), then a resume to 4 epochs from last.pth.
#SBATCH --job-name=itomap_smoke
#SBATCH --account=research-ceg-gse
#SBATCH --partition=gpu-a100-small
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=7000M
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/ntmucke/scientific-stochastic-interpolants/slurm_itomap/logs/%x_%j.out
set -euo pipefail
S=research/ito_maps/experiments/slurm/train_baseline_map.sbatch
export DIAG=1 WARM=1 MAX_BATCHES=5 SEED=0
EPOCHS=3 RUN=itomap_smoke_ito_${SLURM_JOB_ID} MODEL=ito_map EXTRA="model.brownian_features=local" bash $S
rm checkpoints/stochastic_navier_stokes/itomap_smoke_ito_${SLURM_JOB_ID}/DONE
EPOCHS=4 RUN=itomap_smoke_ito_${SLURM_JOB_ID} MODEL=ito_map EXTRA="model.brownian_features=local" bash $S
EPOCHS=3 RUN=itomap_smoke_flow_${SLURM_JOB_ID} MODEL=flow_map bash $S
ls -la checkpoints/stochastic_navier_stokes/itomap_smoke_*_${SLURM_JOB_ID}
