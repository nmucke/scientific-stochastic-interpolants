#!/bin/bash
# One tuning row "stage|cell|parent|grid", same logic as tune_stage() in run_lorenz96.sh.
set -euo pipefail
row=$1; OBS=${OBS:?}; PY=python; THREADS=${THREADS:-2}
IFS='|' read -r stage cell parent grid <<<"$row"
sets=()
if [[ -n "$parent" ]]; then
  out=$($PY select_lorenz96_tuning.py --obs "$OBS" resolve --parent "$parent" --grid "$grid")
  mapfile -t resolved <<<"$out"
  [[ -n "${resolved[0]:-}" ]] || { echo "[$cell] could not resolve grid from $parent" >&2; exit 1; }
  grid=${resolved[0]}
  for e in "${resolved[@]:1}"; do sets+=(--set "$e"); done
fi
out=$($PY select_lorenz96_tuning.py --obs "$OBS" status --cell "$cell" --grid "$grid" --stage "$stage" "${sets[@]}")
read -r state csv <<<"$out"
case "$state" in
  complete) echo "[$cell] skip: complete"; exit 0 ;;
  partial) mv "$csv" "$csv.partial.$(date +%Y%m%d-%H%M%S)"; echo "[$cell] partial moved aside" ;;
  missing) ;;
  *) echo "[$cell] unexpected status $state" >&2; exit 1 ;;
esac
echo "[$cell] start $(date -Is) stage $stage grid=$grid ${sets[*]:-}"
$PY cases/lorenz96/driver.py --obs "$OBS" tune --method "$cell" --grid "$grid" --tag "s$stage" --threads "$THREADS" "${sets[@]}" \
  > "results/lorenz96/$OBS/logs/lane_s${stage}_${cell}.log" 2>&1
echo "[$cell] done $(date -Is)"
