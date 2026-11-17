#!/bin/bash
# Forecast correction on the ORIGINAL manuscript NS case (true IC, NP=20, M=100, E=64,
# traj 10..14, 4 scenarios, Ours x shared_jac10, no KL). One array of 60 cells per FC setting.
#   RADIUS=8  bash submit.sh     # FC defaults (tag fc25_70a3134d, as the wrong-IC FC table)
#   RADIUS=20 bash submit.sh     # localisation radius 20 (tag fc25_8797d2ae)
#   DRY_RUN=1 ...                # manifest to a temp dir + print; nothing submitted or written
#   SMOKE=1 ...                  # 1 cell, NP=7, MIG, separate smoke ROOT
#   VARIANT=jacfree ...          # likelihood: shared_jac10 (default) | jacfree
#   MF=<existing manifest> ARRAY=36-45 ACCOUNT=innovation THROTTLE=1 ...   # submit a slice of an existing manifest
#   PARTITION=gpu-a100-small ... # MIG: 2 cells per job (mig_pair.sbatch), ARRAY = cell range, THROTTLE ignored
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
VARIANT=${VARIANT:-shared_jac10}; ACCOUNT=${ACCOUNT:-research-ceg-gse}; ARRAY=${ARRAY:-}; MF=${MF:-}
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
VTAG=$([ "$VARIANT" = shared_jac10 ] && echo "" || echo "${VARIANT}_")
if [ -n "$MF" ]; then
  [ -f "$MF" ] || { echo "FATAL: MF=$MF not found" >&2; exit 1; }
  grep -q "__ours_${VARIANT}__${FC_TAG}__" "$MF" || { echo "FATAL: $MF is not a ${VARIANT}/${FC_TAG} manifest" >&2; exit 1; }
else
  if [ "$DRY_RUN" = 1 ]; then MF=$(mktemp -d)/cells.tsv; else MF="$ROOT/manifests/${VTAG}${FC_TAG}_${STAMP}.tsv"; mkdir -p "$ROOT"/{metrics,per_step,states,logs,slurm,forecast_correction}; fi
  python3 "$HERE/make_manifest.py" "$ROOT" "$MF" "$NP" "$M" "$E" "$SAVE_TRAJ" "$FC_TAG" "$FC_OVERRIDES" "$VARIANT" $TRAJ
fi
[ "$SMOKE" = 1 ] && { head -1 "$MF" > "$MF.1" && mv "$MF.1" "$MF"; }
N=$(grep -c . "$MF"); ARRAY=${ARRAY:-0-$((N-1))}
export REPO ROOT PE PY
JN="nstrue_${VTAG}${FC_TAG}"
if [ "$PARTITION" = gpu-a100-small ]; then
  # MIG: one job = 2 cells on 2 slices; array task k runs cells LO+2k, LO+2k+1 (QoS: 1 job at a time)
  LO=${ARRAY%-*}; HI=${ARRAY#*-}; NP2=$(( (HI - LO + 2) / 2 ))
  export MIG_LO=$LO MIG_HI=$HI MF
  CMD=(sbatch --parsable --job-name="${JN}_mig" --partition=gpu-a100-small --account="$ACCOUNT"
    -n 2 -c 2 --gres=gpu:2 --mem-per-cpu=7900M --time="$TIME" --array="0-$((NP2-1))%1"
    --output="$ROOT/slurm/%x_%A_%a.out" --export=ALL "$HERE/mig_pair.sbatch")
else
  CMD=(sbatch --parsable --job-name="$JN" --partition="$PARTITION" --account="$ACCOUNT"
    -n 1 -c "$CPUS" --gres="$GRES" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" --array="${ARRAY}%$THROTTLE"
    --output="$ROOT/slurm/%x_%A_%a.out" --export=ALL "$HERE/run_task.sh" "$MF")
fi
SUMMARY="variant=$VARIANT cells=[$ARRAY] account=$ACCOUNT | radius=$RADIUS fc_tag=$FC_TAG fc_overrides='$FC_OVERRIDES' | true IC, NP=$NP M=$M E=$E traj=[$TRAJ] save_traj=$SAVE_TRAJ | Ours x shared_jac10 | no KL | cells=$N
slurm: ${CMD[*]:2:9}
repo: $REPO commit $(git -C "$REPO" rev-parse HEAD) dirty $(git -C "$REPO" status --porcelain --untracked-files=no | wc -l) | scisi: $SCISI"
if [ "$DRY_RUN" = 1 ]; then echo "=== DRY RUN"; echo "$SUMMARY"; echo "ROOT: $ROOT"; echo "first cell:"; sed -n "$(( ${ARRAY%-*} + 1 ))p" "$MF" | tr '\t' '\n' | sed 's/^/  /'; exit 0; fi
mkdir -p "$ROOT/slurm"; git -C "$REPO" diff HEAD > "$ROOT/code_${STAMP}.diff"
printf '=== %s  slurm_trueic_fc/submit.sh\n%s\nmanifest: %s\n' "$(date -Is)" "$SUMMARY" "$MF" >> "$ROOT/run_config.txt"
J=$("${CMD[@]}"); echo "array $J ($N cells) $FC_TAG"; echo "job: $J" >> "$ROOT/run_config.txt"
