#!/bin/bash
# Generative prior + EnKF on the wrong-IC Navier--Stokes grid: 3 priors x 4
# scenarios x 5 trajectories = 60 single-method tasks, run from the ISOLATED copy
# /scratch/ntmucke/scientific-stochastic-interpolants/code_ns_prior_enkf.
# DRY_RUN=1 writes the manifest and prints the sbatch call without submitting.
# Knobs: ROOT TRAJ IC_INDEX NP M E SAVE_TRAJ PARTITION ACCOUNT GRES CPUS MEM_PER_CPU
# TIME THROTTLE PY.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$HERE/../common/paths.sh"
PE=$PE_SCRIPT_DIR; REPO=$PE_REPO_ROOT
ROOT=${ROOT:-/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/results_wrong_ic_np40_prior_enkf/navier_stokes}
TRAJ=${TRAJ:-10 11 12 13 14}; IC_INDEX=${IC_INDEX:-15}; NP=${NP:-40}; M=${M:-100}; E=${E:-64}; SAVE_TRAJ=${SAVE_TRAJ:-11}
PARTITION=${PARTITION:-gpu-v100}; ACCOUNT=${ACCOUNT:-research-ceg-gse}; GRES=${GRES:-gpu:1}
CPUS=${CPUS:-8}; MEM_PER_CPU=${MEM_PER_CPU:-5000M}; TIME=${TIME:-24:00:00}; THROTTLE=${THROTTLE:-16}
DRY_RUN=${DRY_RUN:-0}
PY=${PY:-/scratch/ntmucke/scientific-stochastic-interpolants/.venv-cu126/bin/python}

case "$REPO" in */code_ns_prior_enkf) ;; *) echo "FATAL: REPO=$REPO is not the isolated copy" >&2; exit 1;; esac
if echo " $TRAJ " | grep -q " $IC_INDEX "; then
  echo "FATAL: IC_INDEX=$IC_INDEX is also in TRAJ=[$TRAJ]" >&2; exit 1; fi
echo " $TRAJ " | grep -q " $SAVE_TRAJ " || { echo "FATAL: SAVE_TRAJ=$SAVE_TRAJ not in TRAJ" >&2; exit 1; }
# Never write into an existing results tree.
case "$ROOT" in
  */results/*|*/results|*/results_wrong_ic_np40/*|*/results_wrong_ic_np40|*/results_wrong_ic_np40_ext*|*/results_udales_fc*)
    echo "FATAL: ROOT=$ROOT is an existing results tree" >&2; exit 1;;
esac

mkdir -p "$ROOT"/{metrics,per_step,states,diagnostics,logs,slurm,manifests,provenance}
MAN="$ROOT/manifests/prior_enkf.tsv"
"$PY" "$HERE/make_manifest.py" "$ROOT" "$MAN" "$IC_INDEX" "$NP" "$M" "$E" "$SAVE_TRAJ" $TRAJ
N=$(wc -l < "$MAN")

# Provenance: the copy is not a git checkout; record its origin and the hashes of
# every code file it runs, so a later edit of the copy is detectable.
STAMP=$(date +%Y%m%d_%H%M%S); PROV="$ROOT/provenance/$STAMP"; mkdir -p "$PROV"
cp -p "$REPO"/COPIED_FROM_COMMIT "$REPO"/COPIED_AT "$REPO"/COPIED_STATUS.txt "$REPO"/COPIED_UNCOMMITTED.diff "$PROV"/
[ -f "$REPO/PRIOR_ENKF.diff" ] && cp -p "$REPO/PRIOR_ENKF.diff" "$PROV/"
(cd "$REPO" && find src external_libs "${PE#$REPO/}" -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) \
   -not -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum) > "$PROV/code_sha256.txt"
IPS=$(grep -hoE 'init_perturb_std: float = [0-9.e-]+' "$PE/cases/navier_stokes/enkf_baseline.py" | awk '{print $NF}' | sort -u)
cat > "$ROOT/run_config.txt" <<C
date: $(date -Is)
code: $REPO (isolated copy of live commit $(cat "$REPO/COPIED_FROM_COMMIT") + uncommitted, copied $(cat "$REPO/COPIED_AT")); provenance in provenance/$STAMP
methods: SI-SDE prior + EnKF | DM-SDE prior + EnKF | FM-ODE prior + EnKF  (unguided Ours prior forecast, EnKF-baseline analysis)
traj: $TRAJ | ic_index: $IC_INDEX | num_physical_steps: $NP | M (prior sampler steps): $M | E: $E | seed: 0 | save_traj: $SAVE_TRAJ
enkf: stochastic perturbed-obs, inflation 1.0, Gaspari-Cohn r=20 on sparse, none on super-res (as the EnKF cells)
init_perturb_std: $IPS (the EnKF baseline's)  | skip_kl_reference=true | divergence_rmse_threshold=10.0
slurm: $PARTITION $ACCOUNT $GRES cpus=$CPUS mem=$MEM_PER_CPU time=$TIME throttle=$THROTTLE tasks=$N
python: $PY
C

export REPO ROOT PE PY
cmd=(sbatch --parsable --job-name=ns_prior_enkf --partition="$PARTITION" --account="$ACCOUNT"
  --ntasks=1 --cpus-per-task="$CPUS" --mem-per-cpu="$MEM_PER_CPU" --gres="$GRES" --time="$TIME"
  --array="0-$((N-1))%$THROTTLE" --output="$ROOT/slurm/%x_%A_%a.out"
  --export=ALL "$HERE/run_task.sh" "$MAN")
if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN: ${cmd[*]}"; exit 0; fi
J=$("${cmd[@]}"); echo "submitted array $J ($N tasks)"; echo "array: $J" >> "$ROOT/run_config.txt"
