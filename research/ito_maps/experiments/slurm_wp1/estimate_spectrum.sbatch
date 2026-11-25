#!/bin/bash
# Coloured-driver spectrum from the dataset config (WP1).
#   MODE=increment KAPPA=1 sbatch .../estimate_spectrum.sbatch            (first pass, upper bound)
#   MODE=residuals REGRESSOR=checkpoints/stochastic_navier_stokes_64/wp1_conditional_mean_seed0 sbatch ...
#   MODE=power INPUT=$PHASE0/arrays/d1_spectra_fft2.npz KEY=anomaly_si128_exact sbatch ...
# NAME sets the output file $WP1_ROOT/$CONFIG/spectra/<NAME>.npz.
#SBATCH --job-name=wp1_spectrum
#SBATCH --account=research-ceg-gse
#SBATCH --partition=gpu-a100-small
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8000M
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/ntmucke/scientific-stochastic-interpolants/slurm_itomap/logs/%x_%j.out
source research/ito_maps/experiments/slurm_wp1/common.sh
MODE=${MODE:-increment}
KAPPA=${KAPPA:-1}
NAME=${NAME:-${MODE}_kappa${KAPPA}}
args=(--config-name "$CONFIG" --mode "$MODE" --kappa "$KAPPA"
      --output "$SPECTRA/$NAME.npz" ${ZERO_MEAN:+--zero-mean} ${MAX_BATCHES:+--max-batches $MAX_BATCHES})
case "$MODE" in
  residuals) args+=(--regressor-dir "${REGRESSOR:?set REGRESSOR}") ;;
  anomalies|power) args+=(--input "${INPUT:?set INPUT}" --input-key "${KEY:-increment_train}") ;;
esac
if [ -f "$SPECTRA/$NAME.npz" ]; then echo "exists: $SPECTRA/$NAME.npz"; exit 0; fi
$PY research/ito_maps/experiments/estimate_noise_spectrum.py "${args[@]}"
