#!/bin/bash
# run_task.sh <cells.tsv>: run the cell(s) of ONE array task of a TMPD grid.
#   task i runs lines i*PACK+1 .. (i+1)*PACK; PACK>1 runs them concurrently, one
#   `srun --exact` step (one GPU / MIG slice) each, in a job with -n PACK --gres=gpu:PACK.
#   ALL=1 runs every line of the file sequentially (smoke lanes; IDX ignored).
# Policy as slurm_wrong_ic/ and slurm_udales_fc/: skip a cell whose metrics CSV
# exists; write to a job-local tmp file and publish (mv) only on success, so a
# timed-out task is rerun, not skipped. Per-cell wall time and torch peak memory
# (peakmem_run.py; nvidia-smi reports N/A on MIG) go to $ROOT/logs/timing.tsv.
set -u
HERE=${HERE:?}; CELLS=$1; REPO=${REPO:?}; ROOT=${ROOT:?}; PE=${PE:?}; PY=${PY:?}
PACK=${PACK:-1}; ALL=${ALL:-0}; DIV_GUARD=${DIV_GUARD:-10.0}
# The worktree's scisi, not the editable install of the main checkout.
export PYTHONPATH="$REPO/src:$REPO/external_libs/jax_cfd_lib/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$ROOT/.jax_cache"
cd "$REPO" || exit 1
mkdir -p "$ROOT/logs" "$ROOT/hydra"
TIMING="$ROOT/logs/timing.tsv"
[ -f "$TIMING" ] || printf 'job\ttask\tnode\tgpu\tcell\trc\tseconds\tpeak_reserved_mib\tpeak_allocated_mib\n' > "$TIMING"
SCISI=$("$PY" -c 'import scisi; print(scisi.__file__)' 2>/dev/null)
case "$SCISI" in "$REPO"/src/*) ;; *) echo "[task] FATAL scisi from '$SCISI', not $REPO/src"; exit 2;; esac

run_line() {
  local line=$1 F; IFS=$'\t' read -r -a F <<< "$line"
  local tag=${F[0]} case=${F[1]} out=${F[2]} ps=${F[3]}; local args=("${F[@]:4}")
  [ -f "$out" ] && { echo "[task] SKIP (exists) $tag"; return 0; }
  mkdir -p "$(dirname "$out")" "$(dirname "$ps")"
  local id; id=$(basename "${out%.csv}"); local log="$ROOT/logs/$id.log"
  # KL guard (true-IC grid): the reference of THIS scenario must exist.
  local a ref=""
  for a in "${args[@]}"; do case "$a" in +kl_reference_states=*) ref=${a#*=};; esac; done
  if [ -n "$ref" ]; then
    local sc; sc=$(basename "$out" | sed -E 's/__.*$//')
    ls "$ref"/navier_stokes__EnKF__"${sc}"__seed0__*.npz >/dev/null 2>&1 \
      || { echo "[task] FATAL no KL reference for $sc in $ref"; return 3; }
  fi
  # In-progress files live outside metrics/ and per_step/ (aggregators glob *.csv
  # there) and keep a .csv suffix (ResultsWriter infers the format from it).
  local tdir="$ROOT/.tmp/${SLURM_JOB_ID:-local}"; mkdir -p "$tdir"
  local tmp="$tdir/$id.metrics.csv" pstmp="$tdir/$id.per_step.csv"
  rm -f "$tmp" "$pstmp"
  local gpu; gpu=$(nvidia-smi -L 2>/dev/null | grep -m1 -oE 'MIG [^ ]+|GPU [0-9]+: [^(]+' | tr ' ' '_')
  echo "[task] RUN $tag on $(hostname) ${gpu:-?} $(date +%T)"
  local t0=$SECONDS
  "$PY" -u "$PE/slurm_udales_fc/peakmem_run.py" "$PE/run.py" "case=$case" seeds=[0] \
      case.require_weights=true case.device=cuda \
      "+divergence_rmse_threshold=$DIV_GUARD" +save_per_step=true "+per_step_file=$pstmp" \
      "results_file=$tmp" "hydra.run.dir=$ROOT/hydra/$id" "${args[@]}" >> "$log" 2>&1 < /dev/null
  local rc=$? dt=$((SECONDS - t0))
  local res alc
  res=$(grep -o 'reserved=[0-9]*' "$log" 2>/dev/null | tail -1 | cut -d= -f2)
  alc=$(grep -o 'allocated=[0-9]*' "$log" 2>/dev/null | tail -1 | cut -d= -f2)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${SLURM_JOB_ID:-local}" "${SLURM_ARRAY_TASK_ID:-}" \
      "$(hostname)" "${gpu:-?}" "$id" "$rc" "$dt" "${res:-?}" "${alc:-?}" >> "$TIMING"
  if [ $rc -eq 0 ] && [ -s "$tmp" ]; then
    [ -f "$pstmp" ] && mv -f "$pstmp" "$ps"
    mv -f "$tmp" "$out"; echo "[task] OK $tag ${dt}s peak ${res:-?} MiB $(date +%T)"; return 0
  fi
  echo "[task] FAIL rc=$rc $tag (see $log)"; return 1
}

fail=0
if [ "$ALL" = 1 ]; then
  while IFS= read -r line; do [ -n "$line" ] && { run_line "$line" || fail=1; }; done < "$CELLS"
  exit $fail
fi
IDX=${SLURM_ARRAY_TASK_ID:?}
first=$(( IDX * PACK + 1 )); last=$(( first + PACK - 1 )); pids=()
mapfile -t LINES < <(sed -n "${first},${last}p" "$CELLS")
[ ${#LINES[@]} -gt 0 ] || { echo "[task] FATAL no line for task $IDX"; exit 2; }
if [ "$PACK" -gt 1 ]; then
  export -f run_line; export HERE ROOT REPO PE PY DIV_GUARD TIMING
  for line in "${LINES[@]}"; do
    srun --exact -n1 -c "${SLURM_CPUS_PER_TASK:-2}" --gres=gpu:1 --mem="${STEP_MEM:-15000M}" \
      bash -c 'run_line "$1"' _ "$line" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
else
  for line in "${LINES[@]}"; do run_line "$line" || fail=1; done
fi
exit $fail
