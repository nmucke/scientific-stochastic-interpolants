#!/bin/bash
# run_task.sh <manifest_dir>: run the cell(s) of ONE task (array task i -> key on line
# i+1 of tasks.txt; one (method, scenario, traj), plus its plain twin with
# INCLUDE_PLAIN=1), sequentially, from the LIVE tree. Same policy as
# ../slurm_wrong_ic_ext/run_task.sh: skip a cell whose metrics CSV exists, write to a
# job-local tmp file and publish (mv) only on success, so a timed-out task is rerun,
# not skipped. Adds: per-cell wall time and peak GPU memory (nvidia-smi, 1 s) in
# $ROOT/logs/timing.tsv.
set -u
HERE=${HERE:?}; MDIR=$1; IDX=${SLURM_ARRAY_TASK_ID:?}
REPO=${REPO:?}; ROOT=${ROOT:?}; PE=${PE:?}; PY=${PY:?}
export PYTHONUNBUFFERED=1
cd "$REPO" || exit 1
KEY=$(sed -n "$((IDX + 1))p" "$MDIR/tasks.txt")
[ -n "$KEY" ] || { echo "[task] FATAL no method for task $IDX"; exit 2; }
mkdir -p "$ROOT/logs" "$ROOT/hydra"
TIMING="$ROOT/logs/timing.tsv"
[ -f "$TIMING" ] || printf 'job\ttask\tnode\tgpu\tcell\trc\tseconds\tpeak_gpu_mib\n' > "$TIMING"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_')
echo "[task] $KEY on $(hostname) ${GPU_NAME:-?} job ${SLURM_JOB_ID:-?} $(date -Is)"

run_line() {
  local line=$1 F; IFS=$'\t' read -r -a F <<< "$line"
  local tag=${F[1]} out=${F[2]} ps=${F[3]}; local args=("${F[@]:4}")
  [ -f "$out" ] && { echo "[task] SKIP (exists) $tag"; return 0; }
  mkdir -p "$(dirname "$out")" "$(dirname "$ps")"
  local id; id=$(basename "${out%.csv}"); local log="$ROOT/logs/$id.log"
  # In-progress files live outside metrics/ and per_step/ (aggregators glob *.csv
  # there) and keep a .csv suffix (ResultsWriter infers the format from it).
  local tdir="$ROOT/.tmp/${SLURM_JOB_ID:-local}"; mkdir -p "$tdir"
  local tmp="$tdir/$id.metrics.csv" pstmp="$tdir/$id.per_step.csv"
  rm -f "$tmp" "$pstmp"
  local mon="$tdir/$id.gpumem"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$mon" 2>/dev/null < /dev/null &
  local monpid=$!
  echo "[task] RUN $tag $(date +%T)"
  local t0=$SECONDS
  "$PY" -u "$HERE/peakmem_run.py" "$PE/run.py" case=urban seeds=[0] case.require_weights=true case.device=cuda \
      +divergence_rmse_threshold=${DIV_GUARD:-10.0} +save_per_step=true "+per_step_file=$pstmp" \
      "results_file=$tmp" "hydra.run.dir=$ROOT/hydra/$id" "${args[@]}" >> "$log" 2>&1 < /dev/null
  local rc=$? dt=$((SECONDS - t0))
  kill "$monpid" 2>/dev/null; wait "$monpid" 2>/dev/null
  # Peak: torch allocator (reserved) from the wrapper; nvidia-smi (whole GPU) as backup
  # (MIG slices report N/A there).
  local peak; peak=$(grep -o 'reserved=[0-9]*' "$log" 2>/dev/null | tail -1 | cut -d= -f2)
  [ -n "$peak" ] || peak=$(grep -E '^[0-9]+$' "$mon" 2>/dev/null | sort -n | tail -1)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${SLURM_JOB_ID:-local}" "$IDX" "$(hostname)" \
      "${GPU_NAME:-?}" "$id" "$rc" "$dt" "${peak:-?}" >> "$TIMING"
  if [ $rc -eq 0 ] && [ -s "$tmp" ]; then
    [ -f "$pstmp" ] && mv -f "$pstmp" "$ps"
    mv -f "$tmp" "$out"; echo "[task] OK $tag ${dt}s peak ${peak:-?} MiB $(date +%T)"; return 0
  fi
  echo "[task] FAIL rc=$rc $tag (see $log)"; return 1
}

fail=0
while IFS= read -r line; do
  [ "${line%%$'\t'*}" = "$KEY" ] || continue
  run_line "$line" || fail=1
done < "$MDIR/cells.tsv"
exit $fail
