#!/bin/bash
#SBATCH --job-name=l96_tests_pe
#SBATCH --partition=gpu-a100-small
#SBATCH --account=research-ceg-gse
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem-per-cpu=8000M
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ntmucke/lorenz96_delftblue/slurm/logs/%x_%j.out
set -u
W=/home/ntmucke/ssi-lorenz96
PE="$W/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments"
cd "$W"
source .venv/bin/activate
export PYTHONPATH=$PWD/src MPLBACKEND=Agg
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
git rev-parse HEAD
cd "$PE"
echo "=== PE lorenz96 suite ==="; date
python -m pytest -q tests -k lorenz96 2>&1 | tail -30
echo "PE exit: ${PIPESTATUS[0]}"
date
