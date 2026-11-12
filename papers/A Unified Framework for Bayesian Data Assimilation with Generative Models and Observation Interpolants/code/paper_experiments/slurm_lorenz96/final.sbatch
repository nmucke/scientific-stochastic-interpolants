#!/bin/bash
#SBATCH --job-name=l96_final
#SBATCH --account=research-ceg-gse
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2000M
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ntmucke/lorenz96_delftblue/slurm/logs/%x_%j.out
set -euo pipefail
: "${OBS:?}"
W=/home/ntmucke/ssi-lorenz96
PE="$W/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments"
cd "$W"; source .venv/bin/activate
export PYTHONPATH=$W/src MPLBACKEND=Agg SCISI_REQUIRE_CUDA=1
cd "$PE"
echo "job $SLURM_JOB_ID host $(hostname) obs=$OBS commit $(git rev-parse --short HEAD) start $(date -Is)"
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
SEEDS=0 ./run_lorenz96.sh final "$OBS"
