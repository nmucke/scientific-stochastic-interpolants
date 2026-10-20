#!/usr/bin/env bash
set -euo pipefail
# Optional controlled second stage: evaluate PSD on validation before using it.
# All runs keep the original 0:160/160:180/180:200 trajectory split.
for kind in ito flow; do
  if [ "$kind" = ito ]; then config=follmer_ito_map_stochastic_navier_stokes; else config=flow_map_stochastic_navier_stokes; fi
  .venv/bin/python -m scisi.bin.main_map_model_train --config-name "$config" \
    initial_student_checkpoint="checkpoints/stochastic_navier_stokes/${kind}_psd_seed0/model.pth" \
    trainer.objective=teacher_rollout trainer.num_epochs=10 \
    trainer.max_train_batches=50 trainer.max_val_batches=4 \
    trainer.teacher_steps=32 trainer.consistency_ramp_steps=0 \
    trainer.interval_curriculum_steps=0 trainer.endpoint_probability=0.5 \
    optimizer.lr=0.00005 scheduler.T_max=10 \
    trainer.checkpoint_dir="checkpoints/stochastic_navier_stokes/${kind}_rollout_seed0"
  .venv/bin/python research/ito_maps/experiments/compare.py --split validation \
    --checkpoints "${kind}_rollout_seed0" --steps 1 2 4 8 \
    --origins 5 45 85 --trajectories 4 --ensemble 8 \
    --output research/ito_maps/results/validation_rollout
done
