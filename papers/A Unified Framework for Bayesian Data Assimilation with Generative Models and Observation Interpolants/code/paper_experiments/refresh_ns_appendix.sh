#!/bin/bash
# Regenerate the NS wrong-IC tables and the unified aggregate as pending runs finish.
#   bash paper_experiments/refresh_ns_appendix.sh
# Post-processing only (reads the result trees, writes aggregates + tables).
#   ENKF_LOC20_ROOT=<dir>  overrides the r=20 super-res EnKF production root
#                          (default: aggregate_ns_all.py's ENKF_LOC20_ROOT).
set -euo pipefail
PE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/ntmucke/scientific-stochastic-interpolants/.venv/bin/python
BASE=/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments
R=$BASE/results_wrong_ic_np40

# 1. tab:ns_accuracy family (main grid; clean timings from the original grid).
"$PY" "$PE/aggregate_ns.py" --results-root "$R" --only-M 100 --table-metrics rmse,crps \
  --timing-csv "$BASE/results/timing/seconds_per_step.csv"
# 2. tab:ns_accuracy_fc (radius-8 FC grid; reads the main aggregated/all.csv).
"$PY" "$PE/aggregate_ns.py" --results-root "$R" --only-M 100 \
  --fc-root "$BASE/results_wrong_ic_np40_ext" --fc-only
# 3. Unified aggregate (aggregated_all/, READY last) + tab:ns_all_wrong_ic.
"$PY" "$PE/aggregate_ns_all.py" --base "$BASE"
# 4. Figures (figures agent: add its step below).
#    Appendix all-runs per-step curves (sets ours + baselines; legend/footnote
#    status re-derived from aggregated_all/all.csv on every run). NEW names only:
#    singles/ns_<metric>_vs_step_all_{ours,baselines}_<scen>, singles/ns_legend_all_*.
"$PY" "$PE/make_ns_figures.py" --results-root "$R" --out "$R/figures/navier_stokes" --appendix-all
