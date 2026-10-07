#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root on an otherwise idle GPU, after freezing K.
# Every replication retrains both stages; teacher weights and data split stay fixed.
path_features=${1:-4}
shift || true
seeds=("$@")
if [ ${#seeds[@]} -eq 0 ]; then seeds=(1 2); fi
for seed in "${seeds[@]}"; do
  for kind in ito flow; do
    if [ "$kind" = ito ]; then config=follmer_ito_map_stochastic_navier_stokes; else config=flow_map_stochastic_navier_stokes; fi
    psd_dir="checkpoints/stochastic_navier_stokes/${kind}_psd_seed${seed}"
    .venv/bin/python -m scisi.bin.main_map_model_train --config-name "$config" \
      seed="$seed" trainer.num_epochs=20 trainer.max_train_batches=100 \
      trainer.max_val_batches=4 trainer.teacher_steps=32 \
      trainer.consistency_ramp_steps=100 trainer.interval_curriculum_steps=500 \
      trainer.ema_decay=0.95 trainer.checkpoint_dir="$psd_dir" scheduler.T_max=20
    initial="$psd_dir/model.pth"
    final_dir="checkpoints/stochastic_navier_stokes/${kind}_rollout_seed${seed}"
    extra=()
    if [ "$kind" = ito ] && [ "$path_features" -ne 4 ]; then
      refined="checkpoints/stochastic_navier_stokes/ito_psd_k${path_features}_init_seed${seed}"
      .venv/bin/python research/ito_maps/experiments/refine_path_features.py \
        --checkpoint-dir "$psd_dir" --output-dir "$refined" --num-path-features "$path_features"
      initial="$refined/model.pth"
      final_dir="checkpoints/stochastic_navier_stokes/ito_k${path_features}_rollout_seed${seed}"
      extra=("model.num_path_features=$path_features" "model.map_model.path_channels=$path_features")
    fi
    .venv/bin/python -m scisi.bin.main_map_model_train --config-name "$config" \
      seed="$seed" initial_student_checkpoint="$initial" "${extra[@]}" \
      trainer.objective=teacher_rollout trainer.num_epochs=10 \
      trainer.max_train_batches=50 trainer.max_val_batches=4 \
      trainer.teacher_steps=32 trainer.consistency_ramp_steps=0 \
      trainer.interval_curriculum_steps=0 trainer.endpoint_probability=0.5 \
      optimizer.lr=0.00005 scheduler.T_max=10 trainer.checkpoint_dir="$final_dir"
  done
done
