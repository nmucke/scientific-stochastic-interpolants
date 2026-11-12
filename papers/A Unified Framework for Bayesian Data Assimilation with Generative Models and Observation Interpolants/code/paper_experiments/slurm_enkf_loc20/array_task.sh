#!/bin/bash
# One production cell per array task: index -> (scenario, trajectory). 0-4: 16^2, 5-9: 32^2; traj 10-14.
D=/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/enkf_loc20_superres
SC=("16^2->128^2" "32^2->128^2"); i=${SLURM_ARRAY_TASK_ID:?}
nvidia-smi -L 2>/dev/null; bash $D/run_cpu.sh "${SC[$((i/5))]}" $((10 + i%5)) 40 $D/production
