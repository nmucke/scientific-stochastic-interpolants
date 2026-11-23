#!/usr/bin/env bash
# Arctan PF gate (same procedure and thresholds as linear), then arctan references.
# References start only if the convergence gate passes. Resumable: references skip
# existing files; rerunning the gate recomputes it.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PE"; export OMP_NUM_THREADS=1
W=12
python cases/lorenz96/pf_gate.py --obs arctan converge --ladder 1e5,4e6 --replicates 8 --sequences 10 --workers $W --threads 1 --out convergence_4e6.json | tee /dev/stderr | grep -q "GATE PASSED" || { echo "GATE FAILED or no verdict: stopping before references"; exit 3; }
python cases/lorenz96/pf_gate.py --obs arctan calibrate --n 100000 --sequences 32 --workers $W --threads 1
python cases/lorenz96/pf_gate.py --obs arctan cross-proposal --problem '{"dim_state":8,"obs_stride":2,"num_steps":20}' --n 1000000 --replicates 8 --sequences 2 --workers $W --threads 1
python cases/lorenz96/driver.py --obs arctan reference --split validation --workers $W
python cases/lorenz96/driver.py --obs arctan reference --split test --workers $W
ls results/lorenz96/arctan/reference/validation | wc -l; ls results/lorenz96/arctan/reference/test | wc -l
echo "ARCTAN GATE+REFS DONE"
