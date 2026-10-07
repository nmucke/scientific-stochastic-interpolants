#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root after run_confirmation.sh has completed.
# This reads saved evaluations; it never trains or samples a model.
result_root=research/ito_maps/results
one=()
rollout=()
extra_one=()
extra_rollout=()
for model in ito_rollout_seed0 flow_rollout_seed0 stochastic_interpolant flow_matching; do
  one+=("$result_root/test_one_step/$model/evaluation.json")
  rollout+=("$result_root/test_rollout/$model/evaluation.json")
done
for seed in 1 2; do
  for kind in ito flow; do
    model="${kind}_rollout_seed${seed}"
    extra_one+=("$result_root/test_one_step/$model/evaluation.json")
    extra_rollout+=("$result_root/test_rollout/$model/evaluation.json")
  done
done
.venv/bin/python research/ito_maps/experiments/final_report.py \
  --one-step "${one[@]}" --rollout "${rollout[@]}" \
  --seed-one-step "${extra_one[@]}" --seed-rollout "${extra_rollout[@]}" \
  --persistence-one-step "$result_root/test_persistence/one_step/evaluation.json" \
  --persistence-rollout "$result_root/test_persistence/rollout/evaluation.json" \
  --output "$result_root/final_report"
