#!/usr/bin/env bash
set -euo pipefail
# Run from repository root on an otherwise idle GPU after the training script.
python=.venv/bin/python
result_root=research/video_forecasting/results
names=(follmer_stochastic_interpolant flow_matching follmer_ito_map flow_map persistence)
datasets=("$@")
if [ ${#datasets[@]} -eq 0 ]; then datasets=(kth clevrer); fi
for dataset in "${datasets[@]}"; do
  if [[ "$dataset" != kth && "$dataset" != clevrer ]]; then
    echo "Unknown video dataset: $dataset" >&2
    exit 2
  fi
  protocol="$result_root/${dataset}_protocol.json"
  if [ -f "$protocol" ]; then
    "$python" research/video_forecasting/experiments/freeze_video_protocol.py --dataset "$dataset" --verify-only
  else
    "$python" research/video_forecasting/experiments/freeze_video_protocol.py --dataset "$dataset"
  fi
  for name in "${names[@]}"; do
    directory="checkpoints/video/$dataset/$name"
    destination="$result_root/test/$dataset/$name"
    extra=()
    if [[ "$name" == persistence ]]; then
      directory="checkpoints/video/$dataset/follmer_stochastic_interpolant"
      steps=(1)
      extra=(--persistence)
    elif [[ "$name" == *_map ]]; then
      steps=(1 8)
    else
      steps=(16 64)
    fi
    if [ -f "$destination/evaluation.json" ]; then
      echo "Preserving saved evaluation: $destination"
    else
      "$python" -m scisi.bin.main_video_model_test \
        --checkpoint-dir "$directory" --output-dir "$destination" \
        --num-clips 128 --ensemble-size 4 --seed 20261007 \
        --steps "${steps[@]}" "${extra[@]}" \
        > "$result_root/${dataset}_${name}_test.log" 2>&1
    fi
  done
  "$python" research/video_forecasting/experiments/verify_video_results.py --dataset "$dataset"
done
