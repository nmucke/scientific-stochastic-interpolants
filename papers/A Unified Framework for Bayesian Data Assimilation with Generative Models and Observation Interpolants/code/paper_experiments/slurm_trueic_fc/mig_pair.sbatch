#!/bin/bash
# MIG helper for submit.sh: array task k runs manifest cells MIG_LO+2k and MIG_LO+2k+1 (<= MIG_HI)
# concurrently, one MIG 1g.10gb slice each, via run_task.sh. Env from submit.sh: MF MIG_LO MIG_HI REPO ROOT PE PY.
set -u
k=${SLURM_ARRAY_TASK_ID:?}; a=$(( MIG_LO + 2 * k )); b=$(( a + 1 )); pids=(); fail=0
for i in $a $b; do
  [ "$i" -le "$MIG_HI" ] || continue
  srun --exact -n1 -c 2 --gres=gpu:1 --mem=15000M env SLURM_ARRAY_TASK_ID=$i bash "$PE/slurm_trueic_fc/run_task.sh" "$MF" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fail=1; done
exit $fail
