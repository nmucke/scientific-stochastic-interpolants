#!/bin/bash
# One Lorenz 96 cell: line $SLURM_ARRAY_TASK_ID (1-based) of $JOBLIST = "<cell> <M or ->"; OBS, SEED from env.
#SBATCH --job-name=l96_cell
#SBATCH --account=research-ceg-gse
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem-per-cpu=2000M
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ntmucke/lorenz96_delftblue/slurm/logs/%x_%A_%a.out
set -euo pipefail
: "${OBS:?}" "${JOBLIST:?}"; SEED=${SEED:-0}
read -r CELL M < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBLIST")
W=/home/ntmucke/ssi-lorenz96
PE="$W/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments"
cd "$W"; source .venv/bin/activate
export PYTHONPATH=$W/src MPLBACKEND=Agg SCISI_REQUIRE_CUDA=1
cd "$PE"
TOKEN=; EXTRA=()
if [[ "$M" != - ]]; then TOKEN=__M$M; EXTRA=(--num-steps "$M"); fi
OUT="results/lorenz96/$OBS/metrics/lorenz96__${CELL}${TOKEN}__traj${SEED}__.csv"
echo "job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID host $(hostname) obs=$OBS cell=$CELL M=$M seed=$SEED commit $(git rev-parse --short HEAD) start $(date -Is)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
if [[ -s "$OUT" ]]; then echo "skip: $OUT exists"; exit 0; fi
python cases/lorenz96/driver.py --obs "$OBS" cell --method "$CELL" --seed "$SEED" "${EXTRA[@]}" --threads 4
test -s "$OUT" || { echo "ERROR: $OUT not written"; exit 1; }
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
echo "done $(date -Is)"
