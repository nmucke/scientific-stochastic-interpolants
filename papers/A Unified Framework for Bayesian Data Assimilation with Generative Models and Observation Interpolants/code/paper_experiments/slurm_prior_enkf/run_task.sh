#!/bin/bash
# run_task.sh <manifest.tsv>: run line SLURM_ARRAY_TASK_ID+1 of the manifest.
# Copied from slurm_wrong_ic/run_task.sh (skip if the output exists, publish on
# success only); the KL guard is dropped (these cells never read a KL reference).
set -u
MANIFEST=$1; IDX=${SLURM_ARRAY_TASK_ID:?}
REPO=${REPO:?}; ROOT=${ROOT:?}; PE=${PE:?}; PY=${PY:?}
# ISOLATED COPY: import scisi / jax_cfd_lib from this copy, not the editable
# install that points at the live repo.
export PYTHONPATH="$REPO/src:$REPO/external_libs/jax_cfd_lib/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_COMPILATION_CACHE_DIR="$ROOT/.jax_cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # jax shares the GPU with torch
export PYTHONUNBUFFERED=1
cd "$REPO" || exit 1
"$PY" -c "import scisi, sys; p = scisi.__file__; sys.exit(0 if p.startswith('$REPO/') else f'scisi from {p}, not $REPO')" || exit 4

line=$(sed -n "$((IDX + 1))p" "$MANIFEST"); [ -n "$line" ] || { echo "[task] no line $IDX"; exit 2; }
IFS=$'\t' read -r -a F <<< "$line"
tag=${F[0]} out=${F[1]} ps=${F[2]}; args=("${F[@]:3}")
[ -f "$out" ] && { echo "[task] SKIP (exists) $tag"; exit 0; }
mkdir -p "$(dirname "$out")" "$(dirname "$ps")" "$ROOT/logs" "$ROOT/hydra"
id=$(basename "${out%.csv}"); log="$ROOT/logs/$id.log"
# In-progress files live outside metrics/ and per_step/ (the aggregators glob
# *.csv there) and keep a .csv suffix (ResultsWriter infers the format).
tdir="$ROOT/.tmp/${SLURM_JOB_ID:-local}"; mkdir -p "$tdir"
tmp="$tdir/$id.metrics.csv" pstmp="$tdir/$id.per_step.csv"
echo "[task] RUN $tag $(date +%T) on $(hostname)"
"$PY" -u "$PE/run.py" case=navier_stokes seeds=[0] case.require_weights=true \
    case.device=cuda "+test_index=$(echo "$tag" | sed -E 's/.*traj//')" \
    +divergence_rmse_threshold=10.0 "results_file=$tmp" "hydra.run.dir=$ROOT/hydra/$id" \
    +save_per_step=true "+per_step_file=$pstmp" "${args[@]}" >> "$log" 2>&1
rc=$?
if [ $rc -eq 0 ] && [ -s "$tmp" ]; then
  # publish only finished cells, so a timed-out job is rerun, not skipped
  [ -f "$pstmp" ] && mv -f "$pstmp" "$ps"
  mv -f "$tmp" "$out"; echo "[task] OK $tag $(date +%T)"; exit 0
fi
echo "[task] FAIL rc=$rc $tag (see $log)"; exit 1
