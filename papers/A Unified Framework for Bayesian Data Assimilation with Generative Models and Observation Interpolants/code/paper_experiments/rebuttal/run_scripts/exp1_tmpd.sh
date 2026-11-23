#!/usr/bin/env bash
# Exp 1 TMPD (opt-in baseline) on the multistep linear-Gaussian paper system, K=20, N=1e5,
# seeds 0-4, M given as arguments. Cells paper_tmpd__M<M>__..., no reference rows. Resumable.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
for M in "$@"; do
  echo "== M=$M tmpd $(date -Is)"; M=$M SETTINGS=paper LINEUPS=tmpd bash "$PE/run_analytical_multistep_grid.sh"
done
echo "TMPD DONE $* $(date -Is)"
