#!/usr/bin/env bash
# Experiment 1: multistep linear-Gaussian (K=20, M=50, N=1e5, seeds 0-4). Resumable:
# the grid skips finished cells, so rerunning this script continues where it stopped.
set -e
source "$(dirname "$0")/env.sh"
cd "$REPO"
G="$PE/run_analytical_multistep_grid.sh"
echo "== a"; SETTINGS="paper partial" LINEUPS="base surge" bash "$G"
echo "== b"; SETTINGS=paper LINEUPS=fc FC_VARIANTS=shared FC_STEPS="1 2 5 10 25 50" bash "$G"
echo "== c"; SETTINGS=paper LINEUPS=fc FC_VARIANTS=jacfree FC_STEPS="5 50" bash "$G"
echo "== d"; SETTINGS=paper LINEUPS="fc group" FC_GROUP=64 FC_STEPS="5 50" bash "$G"
echo "== e"; SETTINGS=paper LINEUPS=fc FC_VARIANTS=shared FC_STEPS=5 FC_Q_SOURCE=bank bash "$G"
echo "== aggregate"; python "$PE/aggregate_analytical_multistep.py"
echo "== figures"; python "$PE/make_analytical_multistep_figures.py"
echo "== diagnose"; python "$PE/diagnose_analytical_multistep_surge.py"
echo "== EXP1 DONE"
