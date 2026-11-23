#!/usr/bin/env bash
# lane.sh <lane index> <n lanes> <first core>: runs every n-th job of jobs.txt on 4 cores, CPU only.
# Each job: run (--defer-scoring), then rescore against the PF reference. Resumable: a job whose
# per_sequence JSON exists is skipped. Log: overnight/logs/<name>.log, status line in overnight/status.txt.
set -u
L=$1; N=$2; C0=$3
source "$(dirname "$0")/../../env.sh"
cd "$PE"
O="$PE/results/_runs/abs1_quick/overnight"; R="$PE/results/_runs/abs1_quick"; mkdir -p "$O/logs"
cores="$C0-$((C0+3))"
i=0
while IFS='|' read -r name args; do
  [ -z "$name" ] && continue
  if [ $((i % N)) -eq "$L" ]; then
    echo "$(date -Is) START $name" >> "$O/status.txt"
    if env CUDA_VISIBLE_DEVICES= SCISI_REQUIRE_CUDA=0 SCISI_SCORING_DEVICE=cpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
         taskset -c "$cores" "$PY" cases/lorenz96/quick_cell.py --root "$R" --sequences 1 -- --obs abs cell $args --seed 0 --threads 4 --defer-scoring > "$O/logs/$name.log" 2>&1 \
       && env CUDA_VISIBLE_DEVICES= SCISI_REQUIRE_CUDA=0 SCISI_SCORING_DEVICE=cpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
         taskset -c "$cores" "$PY" cases/lorenz96/quick_cell.py --root "$R" --sequences 1 -- --obs abs cell $args --seed 0 --rescore >> "$O/logs/$name.log" 2>&1; then
      echo "$(date -Is) OK    $name" >> "$O/status.txt"
    else
      echo "$(date -Is) FAIL  $name" >> "$O/status.txt"
    fi
  fi
  i=$((i + 1))
done < "$O/jobs.txt"
echo "$(date -Is) LANE $L DONE" >> "$O/status.txt"
