#!/bin/bash
# uDALES TMPD (plain, row_sum) on gpu-a100-small: each cell (tab:urban_accuracy_M50
# setting: foreign IC 8, 150 physical steps, DA every 3 from step 5, M 50, E 64,
# sparse 1.5625% / 0.78125%, traj 1-5) split into PARTS independent ensemble parts of
# 64/PARTS members (TMPD treats members independently; +ensemble_part seeds each
# part's sampler, urban driver _seed_ensemble_part). 10 cells x 4 parts = 40 parts,
# PACK=2 parts per job (one MIG slice each), throttle 1 (QoS: 1 running job/user).
# Each part keeps its full states; merge.sh then verifies and merges the parts into
# metrics/<scen>__M50__traj<N>__tmpd.csv + per_step/ (aggregate_urban.py format).
#   DRY_RUN=1 bash submit.sh     # print everything, submit nothing
#   bash submit.sh               # submit; then, when all parts are done: bash merge.sh
# Timing (MIG smoke 2026-11-13: 147 s per generated step for E=64 at batch 8, i.e.
# ~2.3 s per member-step): 16 members x 145 steps ~ 1.5 h + ~5 min startup per part.
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GRID=udales
ROOT=${ROOT:-/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/results_udales_tmpd/urban}
PARTS=${PARTS:-4}; TMPD_BATCH=${TMPD_BATCH:-8}
PARTITION=${PARTITION:-gpu-a100-small}; PACK=${PACK:-2}; THROTTLE=${THROTTLE:-1}
TIME=${TIME:-04:00:00}; CPUS=${CPUS:-2}; MEM_PER_CPU=${MEM_PER_CPU:-7900M}
JOB_NAME=${JOB_NAME:-tmpd_udales_split}
source "$HERE/../submit_common.sh"
