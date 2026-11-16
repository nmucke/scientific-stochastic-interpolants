#!/bin/bash
# Shared-Jacobian tuning of one sweep: stage 1 (si/fm/dm) -> pick -> stage 2 (*_fc).
set -euo pipefail
OBS=$1
W=/home/ntmucke/ssi-lorenz96
PE="$W/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments"
cd "$W"; source .venv/bin/activate
export PYTHONPATH=$W/src MPLBACKEND=Agg SCISI_REQUIRE_CUDA=1
cd "$PE"; T=results/lorenz96/$OBS/tuning; L=results/lorenz96/$OBS/logs; mkdir -p "$L"
SEL=/scratch/ntmucke/lorenz96_delftblue/slurm/shared_select.py
echo "[$OBS] host $(hostname) $(nvidia-smi -L | head -1) commit $(git rev-parse --short HEAD) start $(date -Is)"
# run <cell> <stage> <grid json> <expected rows> [--set ...]
run() {
  local cell=$1 st=$2 grid=$3 n=$4; shift 4
  local keys; keys=$(python -c "import json,sys;print('_'.join(json.loads(sys.argv[1])))" "$grid")
  local csv="$T/${cell}__${keys}__shared-s${st}.csv"
  if [[ -s "$csv" ]]; then
    local have; have=$(($(wc -l < "$csv") - 1))
    if [[ $have -eq $n ]]; then echo "[$OBS] skip $cell s$st: $have rows"; return 0; fi
    mv "$csv" "$csv.partial.$(date +%Y%m%d-%H%M%S)"; echo "[$OBS] partial $csv moved aside"
  fi
  echo "[$OBS] tune $cell s$st grid=$grid $* start $(date -Is)"
  python cases/lorenz96/driver.py --obs "$OBS" tune --method "$cell" --grid "$grid" \
    --tag "shared-s$st" --threads 2 "$@" >> "$L/shared_tune_${cell}_s${st}.log" 2>&1
  echo "[$OBS] done $cell s$st $(date -Is)"
}
G1='{"likelihood_mode":["inflated_shared"],"jacobian_damping":[1.0,0.95,0.9]}'
for c in si_sde fm_ode dm_sde; do run $c 1 "$G1" 3; done
python -I "$SEL" "$T" 1 > "$T/selected_shared_stage1.tsv"
cat "$T/selected_shared_stage1.tsv"
G2='{"correction.gain_scale":[0.25,0.5,0.75,1.0],"correction.q_source":["online","exact"]}'
for c in si_sde fm_ode dm_sde; do
  pick=$(awk -F'\t' -v c=$c '$1==c{print $2}' "$T/selected_shared_stage1.tsv")
  [[ -n $pick ]] || { echo "[$OBS] no eligible stage-1 pick for $c" >&2; exit 1; }
  sets=(); while IFS= read -r kv; do sets+=(--set "$kv"); done < <(python -c "import json,sys;[print(f'{k}={json.dumps(v)}') for k,v in json.loads(sys.argv[1]).items()]" "$pick")
  run ${c}_fc 2 "$G2" 8 "${sets[@]}"
done
python -I "$SEL" "$T" 2
echo "[$OBS] finished $(date -Is)"
