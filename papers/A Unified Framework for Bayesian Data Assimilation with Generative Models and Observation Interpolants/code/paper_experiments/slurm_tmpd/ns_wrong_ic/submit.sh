#!/bin/bash
# TMPD (plain, row_sum) on the NS wrong-IC / long-horizon grid: the settings of
# slurm_wrong_ic/ (REBUTTAL_EXPERIMENTS.md section 1). 20 cells = 4 scenarios x traj 10-14.
#   DRY_RUN=1 bash submit.sh     # print everything, submit nothing
#   bash submit.sh               # submit (env knobs: ../submit_common.sh)
# Aggregate together with the main grid: aggregate_ns.py over both metrics/ dirs
# (file names follow that grid's baseline cells, <scen>__M100__traj<N>__baselines__tmpd.csv).
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GRID=ns_wrong_ic
ROOT=${ROOT:-/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/results_wrong_ic_np40_tmpd/navier_stokes}
source "$HERE/../submit_common.sh"
