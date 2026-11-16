#!/bin/bash
# SI-SDE / FM-ODE prior + EnKF (r=20 on all scenarios) on the ORIGINAL manuscript NS case
# (true IC, NP=20, E=64, traj 10..14, 4 scenarios, no KL): 40 cells, MIG gpu-a100-small, 2 per job.
#   DRY_RUN=1 bash submit.sh   |   bash submit.sh
# Run from the frozen worktree this script lives in.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$HERE/../common/paths.sh"
PE=$PE_SCRIPT_DIR; REPO=$PE_REPO_ROOT
PY=${PY:-/scratch/ntmucke/scientific-stochastic-interpolants/.venv-cu126/bin/python}
TRAJ=${TRAJ:-10 11 12 13 14}; NP=${NP:-20}; M=${M:-100}; E=${E:-64}; SAVE_TRAJ=${SAVE_TRAJ:-11}; RADIUS=${RADIUS:-20}
ROOT=${ROOT:-/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/results_ns_trueic_prior_enkf_r20/navier_stokes}
PARTITION=${PARTITION:-gpu-a100-small}; PACK=${PACK:-2}; CPUS=${CPUS:-2}; MEM_PER_CPU=${MEM_PER_CPU:-7900M}
TIME=${TIME:-04:00:00}; THROTTLE=${THROTTLE:-1}; DRY_RUN=${DRY_RUN:-0}
case "$REPO" in /home/ntmucke/scientific-stochastic-interpolants) echo "FATAL: run from the frozen worktree" >&2; exit 1;; esac
case "$ROOT" in */paper_experiments/results/*|*/results_wrong_ic_np40*) echo "FATAL: existing tree $ROOT" >&2; exit 1;; esac
SCISI=$(PYTHONPATH="$REPO/src:$REPO/external_libs/jax_cfd_lib/src" "$PY" -c 'import scisi; print(scisi.__file__)')
case "$SCISI" in "$REPO"/src/*) ;; *) echo "FATAL: scisi resolves to $SCISI" >&2; exit 1;; esac
STAMP=$(date +%Y%m%d_%H%M%S)
if [ "$DRY_RUN" = 1 ]; then MF=$(mktemp -d)/cells.tsv; else MF="$ROOT/manifests/prior_enkf_r${RADIUS}_${STAMP}.tsv"; mkdir -p "$ROOT"/{metrics,per_step,states,diagnostics,logs,slurm}; fi
python3 "$HERE/make_manifest.py" "$ROOT" "$MF" "$NP" "$M" "$E" "$SAVE_TRAJ" "$RADIUS" $TRAJ
NC=$(grep -c . "$MF"); NT=$(( (NC + PACK - 1) / PACK ))
export REPO ROOT PE PY PACK
CMD=(sbatch --parsable --job-name=nstrue_prior_enkf --partition="$PARTITION" --account=research-ceg-gse
  -n "$PACK" -c "$CPUS" --gres="gpu:$PACK" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME"
  --array="0-$((NT-1))%$THROTTLE" --output="$ROOT/slurm/%x_%A_%a.out" --export=ALL "$HERE/run_task.sh" "$MF")
SUMMARY="priors SI-SDE, FM-ODE + EnKF (perturbed obs, inflation 1.0, GC r=$RADIUS all scenarios) | true IC NP=$NP M=$M E=$E traj=[$TRAJ] save_traj=$SAVE_TRAJ | no KL | cells=$NC tasks=$NT pack=$PACK
slurm: ${CMD[*]:2:10}
repo: $REPO commit $(git -C "$REPO" rev-parse HEAD) dirty $(git -C "$REPO" status --porcelain --untracked-files=no | wc -l) | scisi: $SCISI"
if [ "$DRY_RUN" = 1 ]; then echo "=== DRY RUN"; echo "$SUMMARY"; echo "ROOT: $ROOT"; head -1 "$MF" | tr '\t' '\n' | sed 's/^/  /'; exit 0; fi
printf '=== %s  slurm_trueic_prior_enkf/submit.sh\n%s\nmanifest: %s\n' "$(date -Is)" "$SUMMARY" "$MF" >> "$ROOT/run_config.txt"
J=$("${CMD[@]}"); echo "array $J ($NT tasks, $NC cells)"; echo "job: $J" >> "$ROOT/run_config.txt"
