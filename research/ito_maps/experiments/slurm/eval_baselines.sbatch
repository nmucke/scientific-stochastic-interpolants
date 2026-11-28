#!/bin/bash
# Validation scores of the 64x64 baselines under PROTOCOL.md: the six maps at
# 1/2/4/8 map steps and the white SI at 16/64/128 solver steps (Euler-Maruyama),
# E = 16, 20 trajectories x origins 5/25/45/65/85, fair CRPS + calibration.
#SBATCH --job-name=eval_base64
#SBATCH --account=research-ceg-gse
#SBATCH --partition=gpu-v100
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5000M
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/ntmucke/scientific-stochastic-interpolants/slurm_itomap/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/scratch/ntmucke/scientific-stochastic-interpolants/.venv_cu126/bin/python
OUT=/scratch/ntmucke/scientific-stochastic-interpolants/results_itomap/baselines/eval64
echo "$(date '+%F %T') start, commit $(git rev-parse --short HEAD)"
MAPS=(itomap_base_ito_psd_local_seed{0,1,2} itomap_base_flow_psd_seed{0,1,2})
todo=(); for m in "${MAPS[@]}"; do [ -f "$OUT/maps/$m/evaluation.json" ] || todo+=("$m"); done
if [ ${#todo[@]} -gt 0 ]; then
  $PY research/ito_maps/experiments/compare.py --split validation --checkpoints "${todo[@]}" --steps 1 2 4 8 --output "$OUT/maps"
fi
if [ ! -f "$OUT/si/wp1_si_white_t2_e150_seed0/evaluation.json" ]; then
  $PY research/ito_maps/experiments/compare.py --split validation --checkpoints wp1_si_white_t2_e150_seed0 --steps 16 64 128 --output "$OUT/si"
fi
echo "$(date '+%F %T') done"
