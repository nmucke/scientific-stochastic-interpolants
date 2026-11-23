#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
A="$PE/results/analytical_multistep"
S="$A/_stale_surge_velocity_denoiser"
mkdir -p "$S/metrics" "$S/per_step"
for d in metrics per_step; do
  for f in "$A/$d"/paper_surge__* "$A/$d"/partial_surge__*; do [ -e "$f" ] && mv "$f" "$S/$d/"; done
done
ls "$S/metrics" | wc -l
cd "$REPO"
SETTINGS="paper partial" LINEUPS=surge bash "$PE/run_analytical_multistep_grid.sh"
python "$PE/aggregate_analytical_multistep.py"
python "$PE/make_analytical_multistep_figures.py"
python "$PE/diagnose_analytical_multistep_surge.py"
echo "EXP1 SURGE RERUN DONE"
