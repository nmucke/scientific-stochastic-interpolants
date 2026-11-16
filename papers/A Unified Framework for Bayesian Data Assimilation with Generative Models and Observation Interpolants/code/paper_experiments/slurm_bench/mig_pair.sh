#!/bin/bash
# mig_pair.sh "<cmd A>" "<cmd B>": run two commands concurrently, one MIG slice each (job asks for gpu:2).
set -u
echo "host $(hostname) job $SLURM_JOB_ID"; nvidia-smi -L
i=0
for c in "$@"; do
  srun --exact -n1 -c2 --gres=gpu:1 --mem=15000M bash -c "nvidia-smi -L; $c" > "$LOGDIR/${SLURM_JOB_NAME}_step$i.out" 2>&1 &
  i=$((i+1))
done
wait; echo PAIRDONE
