#!/bin/bash
#SBATCH --job-name=b_jaxcheck
#SBATCH --partition=gpu-a100-small
#SBATCH --account=research-ceg-gse
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=7900M
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/ntmucke/scientific-stochastic-interpolants/slurm_bench/jaxcheck_%j.out
cd /home/ntmucke/scientific-stochastic-interpolants/papers/*/code/paper_experiments
/home/ntmucke/scientific-stochastic-interpolants/.venv/bin/python /scratch/ntmucke/scientific-stochastic-interpolants/slurm_bench/jaxcheck.py
echo PY_RC=$?
