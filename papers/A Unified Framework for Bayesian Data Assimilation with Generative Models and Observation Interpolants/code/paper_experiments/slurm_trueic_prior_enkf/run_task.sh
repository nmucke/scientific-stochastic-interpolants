#!/bin/bash
# run_task.sh <manifest.tsv>: array task i runs manifest lines i*PACK+1 .. (i+1)*PACK, one
# MIG slice each (srun --exact). Code pinned to $REPO (PYTHONPATH, checked). Skip existing
# outputs; publish on success only. Per-cell time -> $ROOT/logs/timing.tsv.
set -u
MANIFEST=$1; IDX=${SLURM_ARRAY_TASK_ID:?}; PACK=${PACK:-1}
REPO=${REPO:?}; ROOT=${ROOT:?}; PE=${PE:?}; PY=${PY:?}
export PYTHONPATH="$REPO/src:$REPO/external_libs/jax_cfd_lib/src"
export JAX_COMPILATION_CACHE_DIR="$ROOT/.jax_cache" XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
cd "$REPO" || exit 1
SCISI=$("$PY" -c 'import scisi; print(scisi.__file__)' 2>/dev/null)
case "$SCISI" in "$REPO"/src/*) ;; *) echo "[task] FATAL scisi from '$SCISI', not $REPO/src"; exit 2;; esac
run_line() {
  local line=$1 F; IFS=$'\t' read -r -a F <<< "$line"
  local tag=${F[0]} out=${F[1]} ps=${F[2]}; local args=("${F[@]:3}")
  [ -f "$out" ] && { echo "[task] SKIP (exists) $tag"; return 0; }
  mkdir -p "$(dirname "$out")" "$(dirname "$ps")" "$ROOT/logs" "$ROOT/hydra"
  local id; id=$(basename "${out%.csv}"); local log="$ROOT/logs/$id.log"
  local tdir="$ROOT/.tmp/${SLURM_JOB_ID}"; mkdir -p "$tdir"
  local tmp="$tdir/$id.metrics.csv" pstmp="$tdir/$id.per_step.csv"
  echo "[task] RUN $tag on $(hostname) $(date +%T)"; local t0=$SECONDS
  "$PY" -u "$PE/run.py" case=navier_stokes seeds=[0] case.require_weights=true case.device=cuda \
      "+test_index=$(echo "$tag" | sed -E 's/.*traj//')" +divergence_rmse_threshold=10.0 \
      "results_file=$tmp" "hydra.run.dir=$ROOT/hydra/$id" +save_per_step=true "+per_step_file=$pstmp" \
      "${args[@]}" >> "$log" 2>&1
  local rc=$?
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$SLURM_JOB_ID" "$IDX" "$(hostname)" "$id" "$rc" "$((SECONDS - t0))" >> "$ROOT/logs/timing.tsv"
  if [ $rc -eq 0 ] && [ -s "$tmp" ]; then
    [ -f "$pstmp" ] && mv -f "$pstmp" "$ps"
    mv -f "$tmp" "$out"; echo "[task] OK $tag $(date +%T)"; return 0
  fi
  echo "[task] FAIL rc=$rc $tag (see $log)"; return 1
}
export -f run_line; export REPO ROOT PE PY IDX
first=$(( IDX * PACK + 1 )); last=$(( first + PACK - 1 )); fail=0; pids=()
mapfile -t LINES < <(sed -n "${first},${last}p" "$MANIFEST")
[ ${#LINES[@]} -gt 0 ] || { echo "[task] FATAL no line for task $IDX"; exit 2; }
if [ "$PACK" -gt 1 ]; then
  for line in "${LINES[@]}"; do
    srun --exact -n1 -c "${SLURM_CPUS_PER_TASK:-2}" --gres=gpu:1 --mem=15000M bash -c 'run_line "$1"' _ "$line" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
else
  run_line "${LINES[0]}" || fail=1
fi
exit $fail
