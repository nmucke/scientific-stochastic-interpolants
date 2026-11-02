#!/bin/bash
# Weekend NS training grid: Itô maps over {PSD, LSD} x {local, KL} Brownian
# features, plus the deterministic flow map with PSD and LSD. All from scratch
# (no teacher), seed 0, identical epoch budget. Runs one after another on one
# GPU; a failed run is logged and the next one starts.
#
#   nohup research/ito_maps/experiments/train_ns_grid.sh > research/ito_maps/results/training_grid/grid.log 2>&1 &
#
# Checkpoints: checkpoints/stochastic_navier_stokes/ns_grid_<name>/. A run is
# skipped when its directory already has a DONE marker; an unfinished run
# resumes from its last.pth. A crashed run is resumed up to MAX_RETRIES
# (default 5) times. EPOCHS (default 70) also sets the cosine T_max; RUNS
# selects a subset by name.
set -u
cd "$(dirname "$0")/../../.."

EPOCHS=${EPOCHS:-70}
MAX_RETRIES=${MAX_RETRIES:-5}
OUT=research/ito_maps/results/training_grid
mkdir -p "$OUT"

# name | model | off-diagonal loss | extra overrides
GRID=(
  "ito_psd_local|ito_map|progressive_self_distillation|model.brownian_features=local"
  "ito_psd_kl|ito_map|progressive_self_distillation|model.brownian_features=karhunen_loeve"
  "ito_lsd_local|ito_map|lagrangian_self_distillation|model.brownian_features=local"
  "flow_psd|flow_map|progressive_self_distillation|"
  "flow_lsd|flow_map|lagrangian_self_distillation|"
  # Last: crashed with a CUDA illegal memory access in a bf16-from-step-0 benchmark.
  "ito_lsd_kl|ito_map|lagrangian_self_distillation|model.brownian_features=karhunen_loeve"
)
RUNS=${RUNS:-}

for entry in "${GRID[@]}"; do
  IFS="|" read -r name model loss extra <<< "$entry"
  if [ -n "$RUNS" ] && [[ " $RUNS " != *" $name "* ]]; then continue; fi
  run="ns_grid_$name"
  ckpt="checkpoints/stochastic_navier_stokes/$run"
  if [ -f "$ckpt/DONE" ]; then echo "$(date '+%F %T') skip $run (done)"; continue; fi
  attempt=0
  while true; do
    # Continue from the last completed epoch when one exists (after a crash
    # or a relaunch); intermittent CUDA faults otherwise cost the whole run.
    resume=()
    if [ -f "$ckpt/last.pth" ]; then resume=(trainer.resume_from="$ckpt/last.pth"); fi
    echo "$(date '+%F %T') start $run (attempt $attempt${resume:+, resuming})"
    .venv/bin/python scripts/train.py --config-name stochastic_navier_stokes \
      models="$model" \
      trainer.off_diagonal="$loss" \
      experiment_tracking.name="$run" \
      seed=0 \
      trainer.num_epochs="$EPOCHS" \
      scheduler.T_max="$EPOCHS" \
      trainer.diagonal_epochs=10 \
      trainer.warmup_epochs=10 \
      hydra.run.dir="$OUT/hydra/$run" \
      "${resume[@]}" \
      $extra >> "$OUT/$run.log" 2>&1
    status=$?
    if [ $status -eq 0 ]; then
      date '+%F %T' > "$ckpt/DONE"
      echo "$(date '+%F %T') done $run"
      break
    fi
    echo "$(date '+%F %T') FAILED $run (exit $status), see $OUT/$run.log"
    attempt=$((attempt + 1))
    if [ $attempt -gt "$MAX_RETRIES" ]; then
      echo "$(date '+%F %T') giving up on $run after $MAX_RETRIES retries"
      break
    fi
    sleep 30
  done
done
echo "$(date '+%F %T') ALL DONE"
