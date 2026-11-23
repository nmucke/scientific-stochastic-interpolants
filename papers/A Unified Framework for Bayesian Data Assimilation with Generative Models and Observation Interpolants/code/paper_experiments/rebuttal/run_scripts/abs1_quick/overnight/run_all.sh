#!/usr/bin/env bash
# Overnight |x| sweep: waits for the running M=25/50 SI-SDE cells, then 6 lanes x 4 cores over
# jobs.txt (36 cells), then summary.py. Status: overnight/status.txt.
O="$(cd "$(dirname "$0")" && pwd)"
while tmux ls 2>/dev/null | grep -q "rb_abs1_.*_M\(25\|50\)"; do sleep 30; done
sleep 60   # let the M=25/50 rescoring watcher finish
echo "$(date -Is) QUEUE START (36 jobs, 6 lanes)" >> "$O/status.txt"
for l in 0 1 2 3 4 5; do bash "$O/lane.sh" $l 6 $((l * 4)) & done
wait
echo "$(date -Is) ALL LANES DONE" >> "$O/status.txt"
python3 "$O/summary.py" > "$O/summary.log" 2>&1
echo "$(date -Is) SUMMARY WRITTEN" >> "$O/status.txt"
