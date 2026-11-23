#!/usr/bin/env bash
# |x| sweep on 5 test sequences: PF reference for seqs 3-4 (cores 28-31), 7 lanes x 4 cores, then summary.
Q="$(cd "$(dirname "$0")" && pwd)"
source "$Q/../../env.sh"; cd "$PE"
echo "$(date -Is) QUEUE START ($(wc -l < "$Q/jobs.txt") jobs, 7 lanes)" >> "$Q/status.txt"
( env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 28-31 "$PY" cases/lorenz96/driver.py --obs abs reference --split test --workers 4 > "$Q/logs/reference.log" 2>&1 \
    && touch "$Q/REF_DONE" && echo "$(date -Is) REFERENCE DONE" >> "$Q/status.txt" \
    || echo "$(date -Is) REFERENCE FAIL" >> "$Q/status.txt" ) &
for c in 0 4 8 12 16 20 24; do bash "$Q/lane.sh" $c & done
wait
echo "$(date -Is) ALL LANES DONE" >> "$Q/status.txt"
"$PY" "$Q/summary.py" > "$Q/summary.log" 2>&1
echo "$(date -Is) SUMMARY WRITTEN" >> "$Q/status.txt"
