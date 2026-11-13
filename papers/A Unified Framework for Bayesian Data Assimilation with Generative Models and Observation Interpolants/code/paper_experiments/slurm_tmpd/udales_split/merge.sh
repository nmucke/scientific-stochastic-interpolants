#!/bin/bash
# Verify and merge every finished uDALES TMPD cell (all PARTS parts done) of ROOT:
# writes metrics/ + per_step/ (+ merged states for SAVE_TRAJ). Cells with missing
# parts are listed as "incomplete" and left alone. CPU only (no data or model load).
#   bash merge.sh                       # merge, keep the part states
#   DELETE_PART_STATES=1 bash merge.sh  # delete each part's states after ITS cell merged
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$HERE/../../common/paths.sh"
REPO=$PE_REPO_ROOT
case "$REPO" in */wt_tmpd_ns_urban|*/wt_tmpd_ns_urban/*) echo "FATAL: REPO=$REPO" >&2; exit 1;; esac
ROOT=${ROOT:-/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/results_udales_tmpd/urban}
PY=${PY:-/scratch/ntmucke/scientific-stochastic-interpolants/.venv-cu126/bin/python}
PARTS=${PARTS:-4}; SAVE_TRAJ=${SAVE_TRAJ:-1}
ARGS=("$ROOT" --parts "$PARTS" --save-states-traj "$SAVE_TRAJ")
[ "${DELETE_PART_STATES:-0}" = 1 ] && ARGS+=(--delete-part-states)
export PYTHONPATH="$REPO/src:$REPO/external_libs/jax_cfd_lib/src"
"$PY" "$HERE/merge.py" "${ARGS[@]}" "$@" | tee -a "$ROOT/logs/merge.log"
