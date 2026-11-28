#!/bin/bash
# WP1 step-1 report from finished eval_si / spectrum / training outputs (CPU):
# runs the report tests, then wp1_step1_report.py, and copies the small result
# JSONs to research/ito_maps/results/wp1_<CONFIG>/ for committing.
#   LOGS="regressor=<log> white_t2=<log> ..." sbatch research/ito_maps/experiments/slurm_wp1/report.sbatch
#SBATCH --job-name=wp1_report
#SBATCH --account=research-ceg-gse
#SBATCH --partition=compute-p1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/ntmucke/scientific-stochastic-interpolants/slurm_itomap/logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1
PY=${PY:-/scratch/ntmucke/scientific-stochastic-interpolants/.venv_cu126/bin/python}
CONFIG=${CONFIG:-stochastic_navier_stokes_64}
WP1_ROOT=${WP1_ROOT:-/scratch/ntmucke/scientific-stochastic-interpolants/results_itomap/wp1}
OUT=$WP1_ROOT/$CONFIG
SEED=${SEED:-0}
REFERENCE_SPECTRUM=${REFERENCE_SPECTRUM:-$WP1_ROOT/spectra/residuals_kappa1.npz}
DEST=research/ito_maps/results/wp1_$CONFIG
$PY -m pytest -q -p no:cacheprovider research/ito_maps/tests/test_wp1_step1_report.py
runs=(); logs=()
for v in white_t2 coloured_t coloured_matched; do runs+=(--run "$v=wp1_si_${v}_e150_seed${SEED}"); done
for item in ${LOGS:-}; do logs+=(--log "$item"); done
$PY research/ito_maps/experiments/wp1_step1_report.py --eval-root "$OUT/eval" "${runs[@]}" \
  --spectrum "$OUT/spectra/residuals_kappa1.npz" --reference-spectrum "$REFERENCE_SPECTRUM" \
  "${logs[@]}" --output "$OUT/report/wp1_step1_report_seed$SEED.json"
mkdir -p "$DEST"
cp "$OUT/report/wp1_step1_report_seed$SEED.json" "$OUT/spectra/residuals_kappa1.json" "$DEST/"
for v in white_t2 coloured_t coloured_matched; do
  run=wp1_si_${v}_e150_seed${SEED}
  mkdir -p "$DEST/$run/path_compression"
  cp "$OUT/eval/$run/euler_exact_noise/results.json" "$DEST/$run/euler_exact_noise_results.json"
  for d in "$OUT/eval/$run/path_compression"/*/; do cp "$d/results.json" "$DEST/$run/path_compression/$(basename "$d").json"; done
done
echo "$(date '+%F %T') done"
