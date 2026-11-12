#!/bin/bash
# Wrong-IC Navier--Stokes grid: 20 E=1000 reference tasks + 400 single-method cells.
# DRY_RUN=1 prints the sbatch calls without submitting.
# Slurm knobs: PARTITION ACCOUNT GRES CPUS MEM_PER_CPU TIME_CELL TIME_REF THROTTLE
# THROTTLE_REF PACK. Running jobs = THROTTLE_REF + 20*THROTTLE (account MaxJobs 64).
#
# Benchmarked 2026-11-07 on a gpu-a100-small 1g.10gb slice (32^2, E=64, M=100,
# s per assimilation step): SI jacfree 84 | DM shared k10 636 | SI k1 1762 |
# DM k1 3447 | EnKF64 28 | PF64 25 | EnKF E=1000 369 | D-Flow SGLD OOM (>9.3 GiB).
# Plus ~16 min startup per job. Many cells exceed that partition's 4 h limit and its
# QoS allows 1 running job (2 slices) per user, so use a whole-GPU partition, e.g.
#   PARTITION=gpu-v100 CPUS=8 MEM_PER_CPU=5000M TIME_CELL=24:00:00 TIME_REF=08:00:00
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$HERE/../common/paths.sh"
PE=$PE_SCRIPT_DIR; REPO=$PE_REPO_ROOT
ROOT=${ROOT:-/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/results_wrong_ic_np40/navier_stokes}
TRAJ=${TRAJ:-10 11 12 13 14}; IC_INDEX=${IC_INDEX:-15}; NP=${NP:-40}; M=${M:-100}; E=${E:-64}; SAVE_TRAJ=${SAVE_TRAJ:-11}
PARTITION=${PARTITION:-gpu-a100-small}; ACCOUNT=${ACCOUNT:-research-ceg-gse}; GRES=${GRES:-gpu:1}
CPUS=${CPUS:-2}; MEM_PER_CPU=${MEM_PER_CPU:-7900M}; TIME_CELL=${TIME_CELL:-04:00:00}; TIME_REF=${TIME_REF:-04:00:00}
THROTTLE=${THROTTLE:-2}; THROTTLE_REF=${THROTTLE_REF:-20}; PACK=${PACK:-1}; DRY_RUN=${DRY_RUN:-0}
# WITH_REF=1 submits the E=1000 EnKF reference array (the "EnKF reference" table row).
# WITH_KL=1 makes cells wait for it and compute KL; WITH_ABLATION=1 adds k=1/k=5.
# SUBMIT=cells|ref|all picks what to submit. Defaults: cells only, no KL, no ablation.
WITH_KL=${WITH_KL:-0}; WITH_ABLATION=${WITH_ABLATION:-0}; SUBMIT=${SUBMIT:-cells}
export WITH_KL WITH_ABLATION

# The foreign IC must not be a scored trajectory: prepare_truth_and_obs silently
# uses the truth's own IC when ic_index == test_index, for that cell alone.
if echo " $TRAJ " | grep -q " $IC_INDEX "; then
  echo "FATAL: IC_INDEX=$IC_INDEX is also in TRAJ=[$TRAJ]" >&2; exit 1; fi
echo " $TRAJ " | grep -q " $SAVE_TRAJ " || { echo "FATAL: SAVE_TRAJ=$SAVE_TRAJ not in TRAJ" >&2; exit 1; }
# Never write into, or read a KL reference from, the existing true-IC tree.
case "$ROOT" in */results/*|*/results) echo "FATAL: ROOT=$ROOT is inside results/" >&2; exit 1;; esac

MDIR="$ROOT/manifests"; mkdir -p "$ROOT"/{metrics,per_step,reference,states,logs,slurm} "$ROOT"/ablation/{metrics,per_step}
python3 "$HERE/make_manifest.py" "$ROOT" "$MDIR" "$IC_INDEX" "$NP" "$M" "$E" "$SAVE_TRAJ" $TRAJ

# Provenance. init_perturb_std is not passed by the driver, so the enkf_baseline
# default is what the filters (incl. the E=1000 reference) actually use.
IPS=$(grep -hoE 'init_perturb_std: float = [0-9.e-]+' "$PE/cases/navier_stokes/enkf_baseline.py" | awk '{print $NF}' | sort -u)
[ "$(echo "$IPS" | wc -l)" = 1 ] || { echo "FATAL: inconsistent init_perturb_std: $IPS" >&2; exit 1; }
grep -q init_perturb_std "$PE/cases/navier_stokes/driver.py" && { echo "FATAL: driver now sets init_perturb_std; update provenance" >&2; exit 1; }
cat > "$ROOT/run_config.txt" <<C
date: $(date -Is)
commit: $(git -C "$REPO" rev-parse HEAD)  dirty_files: $(git -C "$REPO" status --porcelain -- "$PE" | wc -l)
traj: $TRAJ | ic_index: $IC_INDEX | num_physical_steps: $NP | M: $M | E: $E | save_traj: $SAVE_TRAJ
reference: E=1000 EnKF, non-localised, num_steps label 50, skip_kl_reference=true
init_perturb_std: $IPS  (physical vorticity units, 256^2 white noise, added before downsampling)
slurm: $PARTITION $ACCOUNT $GRES cpus=$CPUS mem=$MEM_PER_CPU cell=$TIME_CELL ref=$TIME_REF throttle=$THROTTLE throttle_ref=$THROTTLE_REF pack=$PACK
options: submit=$SUBMIT with_kl=$WITH_KL with_ablation=$WITH_ABLATION
C
# Code is uncommitted: snapshot the exact diff the jobs run with.
git -C "$REPO" diff HEAD -- "$PE" > "$ROOT/code_$(date +%Y%m%d_%H%M%S).diff"
[ "$WITH_KL" = 1 ] && [ "$SUBMIT" = cells ] && { echo "FATAL: WITH_KL=1 needs the reference array (SUBMIT=all)" >&2; exit 1; }

PY=${PY:-"$REPO/.venv/bin/python"}; export REPO ROOT PE PACK PY
echo "python: $PY" >> "$ROOT/run_config.txt"
sub() { # sub <name> <time> <n_tasks> <throttle> <dependency|-> <manifest>
  local n=$(( ($3 + PACK - 1) / PACK )) dep=()
  [ "$5" != "-" ] && dep=(--dependency="$5" --kill-on-invalid-dep=no)
  local cmd=(sbatch --parsable --job-name="$1" --partition="$PARTITION" --account="$ACCOUNT"
    --ntasks=1 --cpus-per-task="$CPUS" --mem-per-cpu="$MEM_PER_CPU" --gres="$GRES" --time="$2"
    --array="0-$((n-1))%$4" --output="$ROOT/slurm/%x_%A_%a.out" --export=ALL "${dep[@]}"
    "$HERE/run_task.sh" "$6")
  if [ "$DRY_RUN" = 1 ]; then echo "${cmd[*]}" >&2; echo 900000; else "${cmd[@]}"; fi
}

REF_JOB=""
if [ "$SUBMIT" != cells ]; then
  SAVED_PACK=$PACK; PACK=1; export PACK   # E=1000 reference always runs alone
  RN=$(wc -l < "$MDIR/reference.tsv")
  REF_JOB=$(sub nsic_ref "$TIME_REF" "$RN" "$THROTTLE_REF" - "$MDIR/reference.tsv")
  echo "reference array: $REF_JOB ($RN tasks)"
  PACK=$SAVED_PACK; export PACK
fi
[ "$SUBMIT" = ref ] && exit 0
i=0
for N in $TRAJ; do for SS in 16_2_128_2 32_2_128_2 sparse_5 sparse_1_5625; do
  f="$MDIR/cells_traj${N}_${SS}.tsv"
  # Each cell array waits for exactly its own (traj, scenario) reference task.
  DEP=-; [ "$WITH_KL" = 1 ] && DEP="afterok:${REF_JOB}_${i}"
  J=$(sub "nsic_${N}_${SS}" "$TIME_CELL" "$(wc -l < "$f")" "$THROTTLE" "$DEP" "$f")
  echo "cells traj$N $SS: $J"; i=$((i+1))
done; done
