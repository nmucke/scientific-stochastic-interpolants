#!/bin/bash
# TMPD (plain, row_sum) on the manuscript's original true-IC NS setting (tab:ns_accuracy;
# run_ns_grid.sh as run into results/navier_stokes): own IC, num_physical_steps 20
# (15 DA steps), M 100, E 64, traj 10-14, 4 scenarios, KL against that tree's E=1000
# EnKF references (read only; NO_KL=1 skips it). 20 cells.
#   DRY_RUN=1 bash submit.sh     # print everything, submit nothing
#   bash submit.sh               # submit (env knobs: ../submit_common.sh)
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GRID=ns_trueic
ROOT=${ROOT:-/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/results_ns_trueic_tmpd/navier_stokes}
source "$HERE/../submit_common.sh"
