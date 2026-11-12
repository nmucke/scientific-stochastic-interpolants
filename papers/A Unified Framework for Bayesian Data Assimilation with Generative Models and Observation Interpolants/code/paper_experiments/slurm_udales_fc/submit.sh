#!/bin/bash
# uDALES (urban): Ours (SI-SDE / DM-SDE / FM-ODE) x shared Jacobian (inflated_shared,
# refresh every K sampler steps) x forecast-residual correction (FORECAST_CORRECTION.md).
# One array task per (method, density, trajectory): 30 tasks by default; one cell
# (3 tasks) is TRAJ=1 SCENARIOS="sparse 1.5625%". Runs the LIVE tree. Separate
# results tree (ROOT). Arguments = run_urban_grid.sh run_cell + ours_shared_k10 + FC=on.
#
#   DRY_RUN=1 bash submit.sh      # build the manifest in a temp dir, print each task's
#                                 # run.py arguments, the FC tag and the sbatch call
#   bash submit.sh                # submit
#
# Every setting is an env var. Defaults = the manuscript cells (tab:urban_accuracy_M50,
# results/urban/run_urban_grid.log); urban.yaml's own NP/IC/cadence defaults do NOT
# match it, so everything is passed explicitly:
#   TRAJ="1 2 3 4 5"  SCENARIOS="sparse 1.5625%|sparse 0.78125%"  IC_INDEX=8 (-1: own IC)
#   NP=150  ASSIM_EVERY=3  ASSIM_FROM=5  E=64  M=50  K=10  SAVE_TRAJ=1
#   FC_OVERRIDES=""                        keys of case.forecast_correction (space-sep);
#                                          empty = urban.yaml block (forecast_num_steps
#                                          null = M), e.g. "forecast_num_steps=25"
#   METHODS=si_sde,dm_sde,fm_ode
#   INCLUDE_PLAIN=0                        1: also run the uncorrected cell first
#   EXTRA_ARGS=""                          more run.py overrides (space-sep, no spaces
#                                          inside a value), e.g. "+jacobian_damping=0.9"
#   THROTTLE=""  (max concurrent tasks; account cap is 64 running jobs incl. the NS grid)
#   TIME_TASK=48:00:00 PARTITION=gpu-v100 CPUS=8 MEM_PER_CPU=5000M GRES=gpu:1
# lambda comes from the per-scenario urban rows of configs/method/{si,dm,fm}_*.yaml
# (M=50 -> 0.95 for all three); override it via EXTRA_ARGS if the manuscript differs.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$HERE/../common/paths.sh"
PE=$PE_SCRIPT_DIR; REPO=$PE_REPO_ROOT
ROOT=${ROOT:-/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/results_udales_fc/urban}
TRAJ=${TRAJ:-1 2 3 4 5}; SCENARIOS=${SCENARIOS:-sparse 1.5625%|sparse 0.78125%}
IC_INDEX=${IC_INDEX:-8}; NP=${NP:-150}; ASSIM_EVERY=${ASSIM_EVERY:-3}; ASSIM_FROM=${ASSIM_FROM:-5}
E=${E:-64}; M=${M:-50}; K=${K:-10}; SAVE_TRAJ=${SAVE_TRAJ:-1}
FC_OVERRIDES=${FC_OVERRIDES:-}
METHODS=${METHODS:-si_sde,dm_sde,fm_ode}; INCLUDE_PLAIN=${INCLUDE_PLAIN:-0}; EXTRA_ARGS=${EXTRA_ARGS:-}
PARTITION=${PARTITION:-gpu-v100}; ACCOUNT=${ACCOUNT:-research-ceg-gse}; GRES=${GRES:-gpu:1}
CPUS=${CPUS:-8}; MEM_PER_CPU=${MEM_PER_CPU:-5000M}; TIME_TASK=${TIME_TASK:-48:00:00}
JOB_NAME=${JOB_NAME:-udales_fc}; DRY_RUN=${DRY_RUN:-0}; THROTTLE=${THROTTLE:-}
PY=${PY:-/scratch/ntmucke/scientific-stochastic-interpolants/.venv-cu126/bin/python}

case "$ROOT" in
  */paper_experiments/results|*/paper_experiments/results/*|*/results_wrong_ic_np40*|*/results_full*)
    echo "FATAL: ROOT=$ROOT is an existing results tree; use a new one" >&2; exit 1;; esac
case "$REPO" in */code_ns_wrong_ic*) echo "FATAL: run from the live repo, not a frozen copy" >&2; exit 1;; esac
[ -x "$PY" ] || { echo "FATAL: python $PY not found" >&2; exit 1; }

# FC tag from the SAME composed config the runs use (also validates FC_OVERRIDES).
FC_ARGS=(case.forecast_correction.enabled=true)
for kv in $FC_OVERRIDES; do FC_ARGS+=("case.forecast_correction.$kv"); done
FC_TAG=$(cd "$PE" && "$PY" -m common.forecast_correction tag urban "${FC_ARGS[@]}" 2>/dev/null | tail -1)
[[ "$FC_TAG" =~ ^fc(M|[0-9]+)_[0-9a-f]{8}$ ]] || { echo "FATAL: bad FC_OVERRIDES='$FC_OVERRIDES' (tag '$FC_TAG')" >&2; exit 1; }

STAMP=$(date +%Y%m%d_%H%M%S)
if [ "$DRY_RUN" = 1 ]; then MDIR=$(mktemp -d); else MDIR="$ROOT/manifests/$STAMP"; fi
python3 "$HERE/make_manifest.py" "$ROOT" "$MDIR" "$IC_INDEX" "$NP" "$M" "$E" "$K" \
  "$ASSIM_EVERY" "$ASSIM_FROM" "$SAVE_TRAJ" "$METHODS" "$INCLUDE_PLAIN" "$FC_TAG" \
  "${FC_OVERRIDES:--}" "${EXTRA_ARGS:--}" "$SCENARIOS" $TRAJ
NT=$(wc -l < "$MDIR/tasks.txt")

export REPO ROOT PE PY HERE
CMD=(sbatch --parsable --job-name="$JOB_NAME" --partition="$PARTITION" --account="$ACCOUNT"
  --ntasks=1 --cpus-per-task="$CPUS" --mem-per-cpu="$MEM_PER_CPU" --gres="$GRES"
  --time="$TIME_TASK" --array="0-$((NT-1))${THROTTLE:+%$THROTTLE}" --output="$ROOT/slurm/%x_%A_%a.out" --export=ALL
  "$HERE/run_task.sh" "$MDIR")

SUMMARY="traj: $TRAJ | scenarios: $SCENARIOS | ic_index: $IC_INDEX | num_physical_steps: $NP | DA every $ASSIM_EVERY from $ASSIM_FROM
E: $E | M: $M | likelihood: inflated_shared, jacobian_refresh_every=$K (variant shared_jac$K) | save_traj: $SAVE_TRAJ
fc_tag: $FC_TAG | fc_overrides: ${FC_OVERRIDES:-<urban.yaml defaults>} | include_plain: $INCLUDE_PLAIN | extra: ${EXTRA_ARGS:-<none>}
methods: $METHODS (array task = method x density x traj)
slurm: $PARTITION $ACCOUNT $GRES cpus=$CPUS mem=$MEM_PER_CPU time=$TIME_TASK tasks=$NT"

if [ "$DRY_RUN" = 1 ]; then
  echo "=== DRY RUN (nothing submitted, nothing written under ROOT)"; echo "$SUMMARY"
  echo "ROOT: $ROOT"
  i=0
  while IFS= read -r key; do
    echo; echo "--- array task $i: $key"
    while IFS=$'\t' read -r -a F; do
      [ "${F[0]}" = "$key" ] || continue
      echo "  cell ${F[1]} -> $(basename "${F[2]}")"
      printf '    %s peakmem_run.py run.py case=urban seeds=[0] case.require_weights=true case.device=cuda +divergence_rmse_threshold=10.0 +save_per_step=true +per_step_file=<tmp> results_file=<tmp> hydra.run.dir=<ROOT/hydra/id>' "$PY"
      printf " '%s'" "${F[@]:4}"; echo
    done < "$MDIR/cells.tsv"
    i=$((i+1))
  done < "$MDIR/tasks.txt"
  echo; echo "DRY_RUN sbatch: ${CMD[*]}"; rm -rf "$MDIR"; exit 0
fi

mkdir -p "$ROOT"/{metrics,per_step,states,logs,slurm,forecast_correction,hydra}
git -C "$REPO" diff HEAD > "$ROOT/code_${STAMP}.diff"
cat >> "$ROOT/run_config.txt" <<C
=== $(date -Is)  slurm_udales_fc/submit.sh
commit: $(git -C "$REPO" rev-parse HEAD)  dirty_files: $(git -C "$REPO" status --porcelain | wc -l)  diff: code_${STAMP}.diff
repo: $REPO (live tree)  python: $PY  manifest: $MDIR
$SUMMARY
C
J=$("${CMD[@]}"); echo "array: $J ($NT tasks)"; echo "job: $J" >> "$ROOT/run_config.txt"
