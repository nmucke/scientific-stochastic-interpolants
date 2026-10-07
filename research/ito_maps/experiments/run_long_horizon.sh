#!/usr/bin/env bash
set -euo pipefail
# Stability diagnostics may share the GPU with video TRAINING. Finish this
# script before timed video evaluation. Run from the repository root.
root=research/ito_maps/results/long_horizon
names=(flow_map follmer_ito_map follmer_stochastic_interpolant flow_matching)
checkpoints=(flow_rollout_seed0 ito_rollout_seed0 stochastic_interpolant flow_matching)
steps=(1 8 64 64)
mkdir -p "$root"
for index in "${!names[@]}"; do
  output="$root/${names[$index]}"
  extra=()
  if [ -f "$output/protocol.json" ]; then extra=(--resume); fi
  .venv/bin/python research/ito_maps/experiments/long_horizon_ns.py \
    --checkpoint-dir "checkpoints/stochastic_navier_stokes/${checkpoints[$index]}" \
    --output "$output" --steps "${steps[$index]}" --horizon 4096 --ensemble 8 \
    --device cuda --num-threads 2 \
    --time-audit research/ito_maps/results/long_rollout/data_reference_audit.json \
    --concurrent-workload 'CLEVRER training shares RTX3090 during initial rollout; diagnostic runtime only' \
    "${extra[@]}" >> "$root/${names[$index]}.log" 2>&1
done
.venv/bin/python research/ito_maps/experiments/report_long_horizon.py \
  --inputs "$root/flow_map" "$root/follmer_ito_map" \
    "$root/follmer_stochastic_interpolant" "$root/flow_matching" \
  --output "$root/report" --reference "$root/reference_enstrophy.json"
