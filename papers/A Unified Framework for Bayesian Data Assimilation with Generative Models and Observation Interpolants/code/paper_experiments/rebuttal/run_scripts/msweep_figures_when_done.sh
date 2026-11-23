#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"; cd "$PE"
until ls results/_runs/exp1_msweep_M250_*.log.exit results/_runs/exp1_msweep_M10_25_100_*.log.exit >/dev/null 2>&1; do sleep 60; done
cat results/_runs/exp1_msweep_*.log.exit
python make_analytical_multistep_msweep_figures.py
