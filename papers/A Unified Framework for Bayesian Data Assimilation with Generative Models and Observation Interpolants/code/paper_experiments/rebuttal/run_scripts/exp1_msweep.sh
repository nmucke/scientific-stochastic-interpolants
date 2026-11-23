#!/usr/bin/env bash
# Exp 1 M sweep (paper system, K=20, N=1e5, seeds 0-4): base + surge + fc (S = M, shared and
# Jac-free) at each M given as arguments. M=50 exists. Stops on the first failure; resumable.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
G="$PE/run_analytical_multistep_grid.sh"
for M in "$@"; do
  echo "== M=$M base+surge $(date -Is)"; M=$M SETTINGS=paper LINEUPS="base surge" bash "$G"
  echo "== M=$M fc S=$M $(date -Is)"; M=$M SETTINGS=paper LINEUPS=fc FC_STEPS="$M" bash "$G"
done
echo "MSWEEP DONE $* $(date -Is)"
