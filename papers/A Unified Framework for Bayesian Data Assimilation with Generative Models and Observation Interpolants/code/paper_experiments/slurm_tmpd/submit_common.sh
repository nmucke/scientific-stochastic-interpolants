#!/bin/bash
# Sourced by slurm_tmpd/<grid>/submit.sh (which sets GRID and the ROOT default).
# Builds the manifest (make_manifest.py), writes run_config.txt and submits ONE array,
# one cell per array task (PACK cells per task with PACK>1, one MIG slice each).
#   DRY_RUN=1   print the settings, every task's run.py arguments and the sbatch call;
#               nothing is submitted and nothing is written under ROOT.
# Env (defaults in brackets):
#   ROOT (per grid)  PY [.venv-cu126]  METHODS [tmpd]  TRAJ  SCENARIOS ('|'-sep)  NP
#   EXTRA (more run.py overrides, space-sep)  NO_KL=1 (ns_trueic: skip the KL reference)
#   PARTITION [gpu-a100-small]  ACCOUNT [research-ceg-gse]  TIME [04:00:00]
#   CPUS [2 per cell]  MEM_PER_CPU [7900M]  PACK [2 on gpu-a100-small, else 1]
#   THROTTLE [1 on gpu-a100-small (QoS: 1 running job/user), else none]
# gpu-v100 instead (32 GB, 2-day limit; queue saturated for days):
#   PARTITION=gpu-v100 CPUS=8 MEM_PER_CPU=5000M TIME=12:00:00 PACK=1
set -euo pipefail
: "${GRID:?}" "${HERE:?}"
SLURM_DIR="$(cd -- "$HERE/.." && pwd -P)"
source "$SLURM_DIR/../common/paths.sh"
PE=$PE_SCRIPT_DIR; REPO=$PE_REPO_ROOT
PY=${PY:-/scratch/ntmucke/scientific-stochastic-interpolants/.venv-cu126/bin/python}
METHODS=${METHODS:-tmpd}; TRAJ=${TRAJ:-}; SCENARIOS=${SCENARIOS:-}; NP=${NP:-}; EXTRA=${EXTRA:-}
NO_KL=${NO_KL:-0}
PARTITION=${PARTITION:-gpu-a100-small}; ACCOUNT=${ACCOUNT:-research-ceg-gse}
TIME=${TIME:-04:00:00}; CPUS=${CPUS:-2}; MEM_PER_CPU=${MEM_PER_CPU:-7900M}
if [ "$PARTITION" = gpu-a100-small ]; then PACK=${PACK:-2}; THROTTLE=${THROTTLE:-1}
else PACK=${PACK:-1}; THROTTLE=${THROTTLE:-}; fi
DRY_RUN=${DRY_RUN:-0}; JOB_NAME=${JOB_NAME:-tmpd_$GRID}

case "$ROOT" in
  */paper_experiments/results|*/paper_experiments/results/*|*/results_wrong_ic_np40|*/results_wrong_ic_np40/*|\
  */results_wrong_ic_np40_ext*|*/results_wrong_ic_np40_prior_enkf*|*/results_udales_fc*|*/enkf_loc20_superres*)
    echo "FATAL: ROOT=$ROOT is an existing results tree; use a new one" >&2; exit 1;; esac
case "$REPO" in */code_ns_wrong_ic*|*/code_rebuttal_*|*/code_ns_prior_enkf*)
  echo "FATAL: frozen copy $REPO" >&2; exit 1;; esac
[ -x "$PY" ] || { echo "FATAL: python $PY not found" >&2; exit 1; }
SCISI=$(PYTHONPATH="$REPO/src:$REPO/external_libs/jax_cfd_lib/src" "$PY" -c 'import scisi; print(scisi.__file__)')
case "$SCISI" in "$REPO"/src/*) ;; *) echo "FATAL: scisi resolves to $SCISI, not $REPO/src" >&2; exit 1;; esac

ARGS=("$GRID" "$ROOT")
[ -n "$TRAJ" ] && ARGS+=(--traj $TRAJ)
[ -n "$SCENARIOS" ] && ARGS+=(--scenarios "$SCENARIOS")
[ -n "$NP" ] && ARGS+=(--np "$NP")
[ -n "$EXTRA" ] && ARGS+=(--extra "$EXTRA")
[ "$NO_KL" = 1 ] && ARGS+=(--no-kl)
ARGS+=(--methods "$METHODS")
STAMP=$(date +%Y%m%d_%H%M%S)
if [ "$DRY_RUN" = 1 ]; then MDIR=$(mktemp -d); else MDIR="$ROOT/manifests/$STAMP"; fi
SETTINGS=$(python3 "$SLURM_DIR/make_manifest.py" "${ARGS[0]}" "${ARGS[1]}" "$MDIR" "${ARGS[@]:2}")
NC=$(grep -c . "$MDIR/cells.tsv"); NT=$(( (NC + PACK - 1) / PACK ))

export REPO ROOT PE PY HERE="$SLURM_DIR" PACK
CMD=(sbatch --parsable --job-name="$JOB_NAME" --partition="$PARTITION" --account="$ACCOUNT"
  -n "$PACK" -c "$CPUS" --mem-per-cpu="$MEM_PER_CPU" --gres="gpu:$PACK" --time="$TIME"
  --array="0-$((NT-1))${THROTTLE:+%$THROTTLE}" --output="$ROOT/slurm/%x_%A_%a.out" --export=ALL
  "$SLURM_DIR/run_task.sh" "$MDIR/cells.tsv")
SUMMARY="settings: $SETTINGS
make_manifest args: ${ARGS[*]:2}
cells: $NC | array tasks: $NT (pack $PACK)
python: $PY | scisi: $SCISI | PYTHONPATH: \$REPO/src:\$REPO/external_libs/jax_cfd_lib/src
slurm: $PARTITION $ACCOUNT -n $PACK -c $CPUS mem-per-cpu=$MEM_PER_CPU gres=gpu:$PACK time=$TIME throttle=${THROTTLE:-none}"

if [ "$DRY_RUN" = 1 ]; then
  echo "=== DRY RUN (nothing submitted, nothing written under ROOT)"; echo "$SUMMARY"; echo "ROOT: $ROOT"
  i=0
  while IFS=$'\t' read -r -a F; do
    [ ${#F[@]} -gt 0 ] || continue
    [ $((i % PACK)) = 0 ] && echo && echo "--- array task $((i / PACK))"
    echo "  cell ${F[0]} -> metrics/$(basename "${F[2]}")"
    printf '    run.py case=%s seeds=[0] case.require_weights=true case.device=cuda +divergence_rmse_threshold=10.0 +save_per_step=true' "${F[1]}"
    printf " '%s'" "${F[@]:4}"; echo
    i=$((i+1))
  done < "$MDIR/cells.tsv"
  echo; echo "DRY_RUN sbatch: ${CMD[*]}"; rm -rf "$MDIR"; exit 0
fi

mkdir -p "$ROOT"/{metrics,per_step,states,logs,slurm,hydra}
git -C "$REPO" diff HEAD > "$ROOT/code_${STAMP}.diff"
cat >> "$ROOT/run_config.txt" <<C
=== $(date -Is)  slurm_tmpd/$GRID/submit.sh
commit: $(git -C "$REPO" rev-parse HEAD) ($(git -C "$REPO" rev-parse --abbrev-ref HEAD))  dirty_files: $(git -C "$REPO" status --porcelain --untracked-files=no | wc -l)  diff: code_${STAMP}.diff
repo: $REPO  manifest: $MDIR/cells.tsv
$SUMMARY
C
J=$("${CMD[@]}"); echo "array: $J ($NT tasks, $NC cells)"; echo "job: $J" >> "$ROOT/run_config.txt"
