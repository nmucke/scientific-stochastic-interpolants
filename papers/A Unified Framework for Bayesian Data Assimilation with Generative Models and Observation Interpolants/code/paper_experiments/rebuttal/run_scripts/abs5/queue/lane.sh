#!/usr/bin/env bash
# lane.sh <first core>: pulls the next job of jobs.txt (shared counter, flock), runs it on 4 cores,
# CPU only: --defer-scoring, then (once the PF reference is complete: REF_DONE) --rescore.
# pf_reference / pf_subsample need the reference to run at all: they wait for REF_DONE and run scored.
set -u
C0=$1
source "$(dirname "$0")/../../env.sh"
cd "$PE"
Q="$PE/results/_runs/abs5/queue"; R="$PE/results/_runs/abs5"
cores="$C0-$((C0+3))"
CPU=(env CUDA_VISIBLE_DEVICES= SCISI_REQUIRE_CUDA=0 SCISI_SCORING_DEVICE=cpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 taskset -c "$cores" "$PY" cases/lorenz96/quick_cell.py --root "$R" --sequences 5 -- --obs abs cell)
wait_ref() { while [ ! -f "$Q/REF_DONE" ]; do sleep 30; done; }
while true; do
  n=$(flock "$Q/.lock" bash -c 'n=$(cat "'"$Q"'/next" 2>/dev/null || echo 0); echo $((n+1)) > "'"$Q"'/next"; echo $n')
  line=$(sed -n "$((n+1))p" "$Q/jobs.txt")
  [ -z "$line" ] && break
  name=${line%%|*}; args=${line#*|}
  echo "$(date -Is) START $name (cores $cores)" >> "$Q/status.txt"
  log="$Q/logs/$name.log"
  case $name in
    pf_reference|pf_subsample)
      wait_ref; "${CPU[@]}" $args --seed 0 --threads 4 > "$log" 2>&1; rc=$? ;;
    *)
      "${CPU[@]}" $args --seed 0 --threads 4 --defer-scoring > "$log" 2>&1; rc=$?
      if [ $rc -eq 0 ]; then echo "$(date -Is) RAN   $name" >> "$Q/status.txt"; wait_ref
        "${CPU[@]}" $args --seed 0 --rescore >> "$log" 2>&1; rc=$?; fi ;;
  esac
  if [ $rc -eq 0 ]; then echo "$(date -Is) OK    $name" >> "$Q/status.txt"; else echo "$(date -Is) FAIL  $name" >> "$Q/status.txt"; fi
done
echo "$(date -Is) LANE $C0 DONE" >> "$Q/status.txt"
