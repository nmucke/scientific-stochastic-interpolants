#!/bin/bash
# TMPD (plain, row_sum) on the uDALES manuscript setting (tab:urban_accuracy_M50; the
# argument set of slurm_udales_fc/ without the FC flags): foreign IC 8, 150 physical
# steps, DA every 3 from step 5, E 64, M 50, sparse 1.5625% and 0.78125%, traj 1-5. 10 cells.
#   DRY_RUN=1 bash submit.sh     # print everything, submit nothing
#   bash submit.sh               # submit (env knobs: ../submit_common.sh)
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GRID=udales
ROOT=${ROOT:-/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/results_udales_tmpd/urban}
source "$HERE/../submit_common.sh"
