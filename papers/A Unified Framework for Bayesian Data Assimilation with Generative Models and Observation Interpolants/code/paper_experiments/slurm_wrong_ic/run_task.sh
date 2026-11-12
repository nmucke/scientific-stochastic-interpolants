#!/bin/bash
# run_task.sh <manifest.tsv>: run line(s) of a manifest for this array task.
# PACK cells run concurrently on the allocated GPU (PACK=1 on a MIG slice).
set -u
MANIFEST=$1; PACK=${PACK:-1}; IDX=${SLURM_ARRAY_TASK_ID:?}
REPO=${REPO:?}; ROOT=${ROOT:?}; PE=${PE:?}; PY=${PY:-"$REPO/.venv/bin/python"}
export JAX_COMPILATION_CACHE_DIR="$ROOT/.jax_cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # jax must not grab the slice
export PYTHONUNBUFFERED=1
cd "$REPO" || exit 1

run_line() {
  local line=$1 F; IFS=$'\t' read -r -a F <<< "$line"
  local tag=${F[0]} out=${F[1]} ps=${F[2]}; local args=("${F[@]:3}")
  [ -f "$out" ] && { echo "[task] SKIP (exists) $tag"; return 0; }
  mkdir -p "$(dirname "$out")" "$ROOT/logs" "$ROOT/hydra"
  local id; id=$(basename "${out%.csv}"); local log="$ROOT/logs/$id.log"
  # KL guard: a generative/classical cell must have its OWN new-tree reference.
  # Never fall back: the driver matches on filename only, so any other tree
  # would silently supply true-IC, NP=20 references.
  local a ref=""
  for a in "${args[@]}"; do case "$a" in +kl_reference_states=*) ref=${a#*=};; esac; done
  if [ -n "$ref" ]; then
    case "$ref" in "$ROOT"/reference/*) ;; *) echo "[task] FATAL foreign KL tree $ref"; return 2;; esac
    local sc; sc=$(basename "$out" | sed -E 's/^(ref_)?//; s/__.*$//')
    ls "$ref"/navier_stokes__EnKF__"${sc}"__seed0__*.npz >/dev/null 2>&1 \
      || { echo "[task] FATAL no reference for $sc in $ref"; return 3; }
  fi
  # In-progress files live outside metrics/ and per_step/ (the aggregators glob
  # *.csv there) and must keep a .csv suffix (ResultsWriter infers the format).
  local tdir="$ROOT/.tmp/${SLURM_JOB_ID}"; mkdir -p "$tdir"
  local tmp="$tdir/$id.metrics.csv" pstmp=""; local psargs=()
  if [ "$ps" != "-" ]; then
    mkdir -p "$(dirname "$ps")"; pstmp="$tdir/$id.per_step.csv"
    psargs=(+save_per_step=true "+per_step_file=$pstmp")
  fi
  echo "[task] RUN $tag $(date +%T)"
  "$PY" -u "$PE/run.py" case=navier_stokes seeds=[0] case.require_weights=true \
      case.device=cuda "+test_index=$(echo "$tag" | sed -E 's/.*traj//')" \
      +divergence_rmse_threshold=10.0 "results_file=$tmp" "hydra.run.dir=$ROOT/hydra/$id" \
      "${psargs[@]}" "${args[@]}" >> "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -s "$tmp" ]; then
    # publish only finished cells, so a timed-out job is rerun, not skipped
    [ -n "$pstmp" ] && mv -f "$pstmp" "$ps"
    mv -f "$tmp" "$out"; echo "[task] OK $tag $(date +%T)"; return 0
  fi
  echo "[task] FAIL rc=$rc $tag (see $log)"; return 1
}

first=$(( IDX * PACK + 1 )); last=$(( first + PACK - 1 )); fail=0; pids=()
while IFS= read -r line; do run_line "$line" & pids+=($!); done \
  < <(sed -n "${first},${last}p" "$MANIFEST")
for p in "${pids[@]}"; do wait "$p" || fail=1; done
exit $fail
