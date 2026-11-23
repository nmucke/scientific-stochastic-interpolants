#!/bin/bash
# Retrain one baseline map (Itô or flow) from the committed NS training-grid
# settings, with dropout 0 (research/ito_maps/PROTOCOL.md). Resumable: if the
# checkpoint directory has last.pth the run continues from it; on success a
# DONE marker is written and a resubmission does nothing.
#
#   RUN=itomap_base_ito_psd_local_seed0 MODEL=ito_map SEED=0 \
#     EXTRA="model.brownian_features=local" \
#     sbatch -p gpu-v100 research/ito_maps/experiments/slurm/train_baseline_map.sbatch
#
# Smoke test: EPOCHS=3 DIAG=1 WARM=1 MAX_BATCHES=5 on gpu-a100-small (smoke_baselines.sbatch).
#SBATCH --job-name=itomap_base
#SBATCH --account=research-ceg-gse
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5000M
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/ntmucke/scientific-stochastic-interpolants/slurm_itomap/logs/%x_%j.out
set -euo pipefail
: "${RUN:?set RUN}" "${MODEL:?set MODEL}"
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-70}
MAX_BATCHES=${MAX_BATCHES:-1000}
DIAG=${DIAG:-10}
WARM=${WARM:-10}
EXTRA=${EXTRA:-}

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
ckpt="checkpoints/stochastic_navier_stokes/$RUN"
if [ -f "$ckpt/DONE" ]; then echo "$RUN already done"; exit 0; fi
resume=()
if [ -f "$ckpt/last.pth" ]; then resume=(trainer.resume_from="$ckpt/last.pth"); fi
echo "$(date '+%F %T') start $RUN on $(hostname), commit $(git rev-parse --short HEAD), resume=${#resume[@]}"
.venv/bin/python -c "import scisi; print('scisi from', scisi.__file__)"

# Grid settings (train_ns_grid.sh): 10 diagonal-only epochs, then a 10-epoch
# ramp of the off-diagonal weight, cosine schedule over the full budget.
.venv/bin/python scripts/train.py --config-name stochastic_navier_stokes \
  models="$MODEL" \
  trainer.off_diagonal=progressive_self_distillation \
  experiment_tracking.name="$RUN" \
  seed="$SEED" \
  trainer.num_epochs="$EPOCHS" \
  scheduler.T_max="$EPOCHS" \
  trainer.diagonal_epochs="$DIAG" \
  trainer.warmup_epochs="$WARM" \
  trainer.max_train_batches="$MAX_BATCHES" \
  backbone.dropout_rate=0.0 \
  backbone.attention.dropout_rate=0.0 \
  hydra.run.dir="/scratch/ntmucke/scientific-stochastic-interpolants/results_itomap/baselines/hydra/$RUN" \
  "${resume[@]}" $EXTRA
date '+%F %T' > "$ckpt/DONE"
echo "$(date '+%F %T') done $RUN"
