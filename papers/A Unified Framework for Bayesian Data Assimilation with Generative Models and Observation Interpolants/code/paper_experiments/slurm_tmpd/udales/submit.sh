#!/bin/bash
# TMPD (plain, row_sum) on the uDALES manuscript setting (tab:urban_accuracy_M50; the
# argument set of slurm_udales_fc/ without the FC flags): foreign IC 8, 150 physical
# steps, DA every 3 from step 5, E 64, M 50, sparse 1.5625% and 0.78125%, traj 1-5. 10 cells.
#   DRY_RUN=1 bash submit.sh     # print everything, submit nothing
#   bash submit.sh               # submit (env knobs: ../submit_common.sh)
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GRID=udales
# Does NOT fit gpu-a100-small: MIG smoke 2026-11-13 measured 147 s per generated step
# (batch 8, 8.0 GiB peak) -> ~6 h per 145-step cell > the 4 h limit. Default: gpu-v100.
PARTITION=${PARTITION:-gpu-v100}; CPUS=${CPUS:-8}; MEM_PER_CPU=${MEM_PER_CPU:-5000M}
TIME=${TIME:-12:00:00}; PACK=${PACK:-1}
ROOT=${ROOT:-/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/results_udales_tmpd/urban}
source "$HERE/../submit_common.sh"
