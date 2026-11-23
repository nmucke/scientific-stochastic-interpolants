#!/bin/bash
# run_task.sh <manifest.tsv>: one cell (line IDX+1) of the true-IC FC grid.
# Same as ../slurm_wrong_ic_ext/run_task.sh, but the code is pinned: PYTHONPATH -> $REPO/src
# (scisi is an editable install of the main checkout) and checked before running.
set -u
MANIFEST=$1; IDX=${SLURM_ARRAY_TASK_ID:?}
REPO=${REPO:?}; ROOT=${ROOT:?}; PE=${PE:?}; PY=${PY:?}
export PYTHONPATH="$REPO/src:$REPO/external_libs/jax_cfd_lib/src"
export JAX_COMPILATION_CACHE_DIR="$ROOT/.jax_cache" XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
cd "$REPO" || exit 1
SCISI=$("$PY" -c 'import scisi; print(scisi.__file__)' 2>/dev/null)
case "$SCISI" in "$REPO"/src/*) ;; *) echo "[task] FATAL scisi from '$SCISI', not $REPO/src"; exit 2;; esac
line=$(sed -n "$((IDX + 1))p" "$MANIFEST"); [ -n "$line" ] || { echo "[task] FATAL no line $IDX"; exit 2; }
IFS=$'\t' read -r -a F <<< "$line"
tag=${F[0]} out=${F[1]} ps=${F[2]}; args=("${F[@]:3}")
[ -f "$out" ] && { echo "[task] SKIP (exists) $tag"; exit 0; }
mkdir -p "$(dirname "$out")" "$(dirname "$ps")" "$ROOT/logs" "$ROOT/hydra"
id=$(basename "${out%.csv}"); log="$ROOT/logs/$id.log"
tdir="$ROOT/.tmp/${SLURM_JOB_ID}"; mkdir -p "$tdir"
tmp="$tdir/$id.metrics.csv"; pstmp="$tdir/$id.per_step.csv"
echo "[task] RUN $tag on $(hostname) $(date +%T)"; t0=$SECONDS
"$PY" -u "$PE/run.py" case=navier_stokes seeds=[0] case.require_weights=true case.device=cuda \
    "+test_index=$(echo "$tag" | sed -E 's/.*traj//')" +divergence_rmse_threshold=10.0 \
    "results_file=$tmp" "hydra.run.dir=$ROOT/hydra/$id" +save_per_step=true "+per_step_file=$pstmp" \
    "${args[@]}" >> "$log" 2>&1
rc=$?
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${SLURM_JOB_ID}" "$IDX" "$(hostname)" "$id" "$rc" "$((SECONDS - t0))" >> "$ROOT/logs/timing.tsv"
if [ $rc -eq 0 ] && [ -s "$tmp" ]; then
  [ -f "$pstmp" ] && mv -f "$pstmp" "$ps"
  mv -f "$tmp" "$out"; echo "[task] OK $tag $(date +%T)"; exit 0
fi
echo "[task] FAIL rc=$rc $tag (see $log)"; exit 1
