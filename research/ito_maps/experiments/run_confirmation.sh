#!/usr/bin/env bash
set -euo pipefail
# Exact frozen confirmation protocol. Run once, from repository root, on idle GPU.
test_root=research/ito_maps/results/test_one_step
rollout_root=research/ito_maps/results/test_rollout
common=(--split test --trajectories 20 --ensemble 16)
compare=(.venv/bin/python research/ito_maps/experiments/compare.py)
"${compare[@]}" "${common[@]}" --checkpoints ito_rollout_seed0 flow_rollout_seed0 \
  --steps 1 2 4 8 --origins 5 45 85 --output "$test_root"
"${compare[@]}" "${common[@]}" --checkpoints stochastic_interpolant flow_matching \
  --steps 1 4 16 64 128 --origins 5 45 85 --output "$test_root"
"${compare[@]}" "${common[@]}" --checkpoints ito_rollout_seed0 flow_rollout_seed0 \
  --steps 1 8 --origins 45 --horizon 10 --output "$rollout_root"
"${compare[@]}" "${common[@]}" --checkpoints stochastic_interpolant flow_matching \
  --steps 64 --origins 45 --horizon 10 --output "$rollout_root"
.venv/bin/python research/ito_maps/experiments/plot_forecasts.py \
  --checkpoints checkpoints/stochastic_navier_stokes/ito_rollout_seed0 \
    checkpoints/stochastic_navier_stokes/flow_rollout_seed0 \
    checkpoints/stochastic_navier_stokes/stochastic_interpolant \
    checkpoints/stochastic_navier_stokes/flow_matching \
  --steps 8 1 64 64 --labels 'Ito map (8)' 'Flow map (1)' 'SI (64)' 'FM (64)' \
  --output research/ito_maps/results/test_qualitative --save-ensembles

# Independent full student training replications, with no test-driven selection.
bash research/ito_maps/experiments/train_replications.sh 4 1 2
for seed in 1 2; do
  checkpoints=("ito_rollout_seed${seed}" "flow_rollout_seed${seed}")
  "${compare[@]}" "${common[@]}" --checkpoints "${checkpoints[@]}" \
    --steps 1 8 --origins 5 45 85 --output "$test_root"
  "${compare[@]}" "${common[@]}" --checkpoints "${checkpoints[@]}" \
    --steps 1 8 --origins 45 --horizon 10 --output "$rollout_root"
  .venv/bin/python research/ito_maps/experiments/archive_training.py \
    --checkpoint-dirs "checkpoints/stochastic_navier_stokes/ito_psd_seed${seed}" \
      "checkpoints/stochastic_navier_stokes/flow_psd_seed${seed}" \
      "checkpoints/stochastic_navier_stokes/ito_rollout_seed${seed}" \
      "checkpoints/stochastic_navier_stokes/flow_rollout_seed${seed}"
done
