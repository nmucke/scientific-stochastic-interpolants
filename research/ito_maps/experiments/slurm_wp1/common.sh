# Shared settings for the WP1 step-1 launchers (sourced, not run).
# Run from the worktree root; PYTHONPATH pins scisi to this worktree.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
PY=${PY:-/scratch/ntmucke/scientific-stochastic-interpolants/.venv_cu126/bin/python}  # torch 2.9.1+cu126: V100 and A100
# Dataset config (hydra --config-name). Its experiment_tracking.project equals
# its name, so train.py writes checkpoints to checkpoints/$CONFIG/<run>.
CONFIG=${CONFIG:-stochastic_navier_stokes_64}
CKPT_ROOT=checkpoints/$CONFIG
WP1_ROOT=${WP1_ROOT:-/scratch/ntmucke/scientific-stochastic-interpolants/results_itomap/wp1}
OUT=$WP1_ROOT/$CONFIG  # per-config spectra/, hydra/, trackio/, eval/
SPECTRA=$OUT/spectra
mkdir -p "$OUT" "$SPECTRA"
echo "CONFIG=$CONFIG OUT=$OUT"
$PY -c "import scisi, torch; print('scisi from', scisi.__file__, 'torch', torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# Hydra overrides for the three processes of WP1 step 1 (VARIANT):
#   white_t2         QuadraticStochasticInterpolation, C = Id (today's process)
#   coloured_t       LinearStochasticInterpolation,    C = FourierDiagonal(SPECTRUM)
#   coloured_matched MatchedStochasticInterpolation,   C = FourierDiagonal(SPECTRUM)
variant_overrides() {
  local cov
  case "$1" in
    white_t2) return 0 ;;
    coloured_t) echo -n "model.interpolation._target_=scisi.models.interpolations.LinearStochasticInterpolation " ;;
    coloured_matched) echo -n "model.interpolation._target_=scisi.models.interpolations.MatchedStochasticInterpolation " ;;
    *) echo "unknown VARIANT $1" >&2; return 1 ;;
  esac
  cov="${SPECTRUM:?set SPECTRUM for a coloured variant}"
  echo "+model.interpolation.noise_covariance._target_=scisi.sampling.noise_covariance.FourierDiagonalCovariance" \
       "+model.interpolation.noise_covariance.spectrum_path=$cov"
}
