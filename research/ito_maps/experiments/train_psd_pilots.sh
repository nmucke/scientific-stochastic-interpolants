#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root; serialize GPU jobs.
for kind in ito flow; do
  if [ "$kind" = ito ]; then config=follmer_ito_map_stochastic_navier_stokes; else config=flow_map_stochastic_navier_stokes; fi
  .venv/bin/python -m scisi.bin.main_map_model_train --config-name "$config" \
    trainer.num_epochs=20 trainer.max_train_batches=100 trainer.max_val_batches=4 \
    trainer.teacher_steps=32 trainer.consistency_ramp_steps=100 \
    trainer.interval_curriculum_steps=500 trainer.ema_decay=0.95 \
    trainer.checkpoint_dir="checkpoints/stochastic_navier_stokes/${kind}_psd_seed0" \
    scheduler.T_max=20
 done
