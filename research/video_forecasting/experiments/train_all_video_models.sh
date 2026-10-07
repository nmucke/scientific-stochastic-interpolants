#!/usr/bin/env bash
set -euo pipefail
# Run from repository root on an otherwise idle GPU. Existing completed stages
# are checked rather than overwritten. This is the initial pixel-space protocol.
python=.venv/bin/python
result_root=research/video_forecasting/results
mkdir -p "$result_root"
stage_status() {
  "$python" research/video_forecasting/experiments/training_stage_status.py "$1" "$2"
}

datasets=("$@")
if [ ${#datasets[@]} -eq 0 ]; then datasets=(kth clevrer); fi
for dataset in "${datasets[@]}"; do
  if [[ "$dataset" != kth && "$dataset" != clevrer ]]; then
    echo "Unknown video dataset: $dataset" >&2
    exit 2
  fi
  manifest="data/video_forecasting/${dataset}_64/manifest.json"
  if [ ! -f "$manifest" ]; then
    echo "Waiting for full prepared dataset: $manifest"
    while [ ! -f "$manifest" ]; do sleep 10; done
  fi
  for kind in follmer_stochastic_interpolant flow_matching; do
    directory="checkpoints/video/$dataset/$kind"
    status="$(stage_status "$directory" 20000)"
    if [[ "$status" == missing ]]; then
      "$python" -m scisi.bin.main_video_model_train --config-name "${kind}_${dataset}" \
        trainer.num_epochs=200 trainer.max_train_batches=100 trainer.max_val_batches=8 \
        trainer.early_stopping.patience=250 \
        > "$result_root/${dataset}_${kind}_train.log" 2>&1
    else
      echo "Preserving complete stage $directory: 20000 updates"
    fi
  done
  for kind in follmer_ito_map flow_map; do
    psd="checkpoints/video/$dataset/${kind}_psd"
    final="checkpoints/video/$dataset/$kind"
    status="$(stage_status "$psd" 2000)"
    if [[ "$status" == missing ]]; then
      "$python" -m scisi.bin.main_video_model_train --config-name "${kind}_${dataset}" \
        trainer.num_epochs=20 trainer.max_train_batches=100 trainer.max_val_batches=8 \
        trainer.checkpoint_dir="$psd" trainer.teacher_steps=32 \
        > "$result_root/${dataset}_${kind}_psd.log" 2>&1
    fi
    status="$(stage_status "$final" 500)"
    if [[ "$status" == missing ]]; then
      "$python" -m scisi.bin.main_video_model_train --config-name "${kind}_${dataset}" \
        initial_student_checkpoint="$psd/model.pth" trainer.checkpoint_dir="$final" \
        trainer.objective=teacher_rollout trainer.num_epochs=10 trainer.max_train_batches=50 \
        trainer.max_val_batches=8 trainer.teacher_steps=32 trainer.diagonal_weight=1 \
        trainer.interval_curriculum_steps=0 trainer.initial_max_interval=1 \
        trainer.consistency_ramp_steps=0 trainer.endpoint_probability=0.5 \
        optimizer.lr=0.00005 scheduler.eta_min=0.00001 \
        > "$result_root/${dataset}_${kind}_refinement.log" 2>&1
    fi
  done
done
