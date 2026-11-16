#!/bin/bash
# Forecast correction on the ORIGINAL manuscript NS case (true IC, NP=20, M=100, E=64,
# traj 10..14, 4 scenarios, Ours x shared_jac10, no KL). One array of 60 cells per FC setting.
#   RADIUS=8  bash submit.sh     # FC defaults (tag fc25_70a3134d, as the wrong-IC FC table)
#   RADIUS=20 bash submit.sh     # localisation radius 20 (tag fc25_8797d2ae)
#   DRY_RUN=1 ...                # manifest to a temp dir + print; nothing submitted or written
#   SMOKE=1 ...                  # 1 cell, NP=7, MIG, separate smoke ROOT
# REPO = the frozen worktree this script lives in (run it from there, not the main checkout).
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$HERE/../common/paths.sh"
PE=$PE_SCRIPT_DIR; REPO=$PE_REPO_ROOT
PY=${PY:-/scratch/ntmucke/scientific-stochastic-interpolants/.venv-cu126/bin/python}
RADIUS=${RADIUS:?set RADIUS=8 or RADIUS=20}; SMOKE=${SMOKE:-0}; DRY_RUN=${DRY_RUN:-0}
TRAJ=${TRAJ:-10 11 12 13 14}; NP=${NP:-20}; M=${M:-100}; E=${E:-64}; SAVE_TRAJ=${SAVE_TRAJ:-11}
FC_OVERRIDES="forecast_num_steps=25"; [ "$RADIUS" = 8 ] || FC_OVERRIDES+=" localisation_radius=$RADIUS"
ROOT=${ROOT:-/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/results_ns_trueic_fc/navier_stokes}
PARTITION=${PARTITION:-gpu-v100}; CPUS=${CPUS:-8}; MEM_PER_CPU=${MEM_PER_CPU:-5000M}; GRES=gpu:1
TIME=${TIME:-04:00:00}; THROTTLE=${THROTTLE:-16}
if [ "$SMOKE" = 1 ]; then
  ROOT=${SMOKE_ROOT:-/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/results_ns_trueic_fc_smoke/navier_stokes}
  TRAJ=11; NP=7; PARTITION=gpu-a100-small; CPUS=2; MEM_PER_CPU=7900M; TIME=02:00:00; THROTTLE=1
fi
case "$REPO" in /home/ntmucke/scientific-stochastic-interpolants) [ "$SMOKE" = 1 ] || {
  echo "FATAL: production must run from the frozen worktree, not the main checkout" >&2; exit 1; };; esac
case "$ROOT" in */paper_experiments/results/*|*/results_wrong_ic_np40*) echo "FATAL: ROOT=$ROOT is an existing tree" >&2; exit 1;; esac
SCISI=$(PYTHONPATH="$REPO/src:$REPO/external_libs/jax_cfd_lib/src" "$PY" -c 'import scisi; print(scisi.__file__)')
case "$SCISI" in "$REPO"/src/*) ;; *) echo "FATAL: scisi resolves to $SCISI" >&2; exit 1;; esac
FC_ARGS=(case.forecast_correction.enabled=true); for kv in $FC_OVERRIDES; do FC_ARGS+=("case.forecast_correction.$kv"); done
FC_TAG=$(cd "$PE" && PYTHONPATH="$REPO/src:$REPO/external_libs/jax_cfd_lib/src" "$PY" -m common.forecast_correction tag navier_stokes "${FC_ARGS[@]}")
STAMP=$(date +%Y%m%d_%H%M%S)
if [ "$DRY_RUN" = 1 ]; then MF=$(mktemp -d)/cells.tsv; else MF="$ROOT/manifests/${FC_TAG}_${STAMP}.tsv"; mkdir -p "$ROOT"/{metrics,per_step,states,logs,slurm,forecast_correction}; fi
python3 "$HERE/make_manifest.py" "$ROOT" "$MF" "$NP" "$M" "$E" "$SAVE_TRAJ" "$FC_TAG" "$FC_OVERRIDES" $TRAJ
[ "$SMOKE" = 1 ] && { head -1 "$MF" > "$MF.1" && mv "$MF.1" "$MF"; }
N=$(grep -c . "$MF")
SLURM_N=(-n 1 -c "$CPUS" --gres="$GRES")
export REPO ROOT PE PY
CMD=(sbatch --parsable --job-name="nstrue_${FC_TAG}" --partition="$PARTITION" --account=research-ceg-gse
  "${SLURM_N[@]}" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" --array="0-$((N-1))%$THROTTLE"
  --output="$ROOT/slurm/%x_%A_%a.out" --export=ALL "$HERE/run_task.sh" "$MF")
SUMMARY="radius=$RADIUS fc_tag=$FC_TAG fc_overrides='$FC_OVERRIDES' | true IC, NP=$NP M=$M E=$E traj=[$TRAJ] save_traj=$SAVE_TRAJ | Ours x shared_jac10 | no KL | cells=$N
slurm: ${CMD[*]:2:9}
repo: $REPO commit $(git -C "$REPO" rev-parse HEAD) dirty $(git -C "$REPO" status --porcelain --untracked-files=no | wc -l) | scisi: $SCISI"
if [ "$DRY_RUN" = 1 ]; then echo "=== DRY RUN"; echo "$SUMMARY"; echo "ROOT: $ROOT"; echo "first cell:"; head -1 "$MF" | tr '\t' '\n' | sed 's/^/  /'; exit 0; fi
git -C "$REPO" diff HEAD > "$ROOT/code_${STAMP}.diff"
printf '=== %s  slurm_trueic_fc/submit.sh\n%s\nmanifest: %s\n' "$(date -Is)" "$SUMMARY" "$MF" >> "$ROOT/run_config.txt"
J=$("${CMD[@]}"); echo "array $J ($N cells) $FC_TAG"; echo "job: $J" >> "$ROOT/run_config.txt"
