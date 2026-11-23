#!/usr/bin/env bash
# PF references (N=4e6, 2 replicates): validation then test. Resumable (existing files skipped).
set -e
source "$(dirname "$0")/env.sh"
cd "$PE"; export OMP_NUM_THREADS=1
python cases/lorenz96/driver.py reference --split validation --workers 14
python cases/lorenz96/driver.py reference --split test --workers 14
ls results/lorenz96/reference/validation | wc -l; ls results/lorenz96/reference/test | wc -l
