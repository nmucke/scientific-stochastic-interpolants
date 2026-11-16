#!/bin/bash
#SBATCH --job-name=l96_shtune
#SBATCH --account=research-ceg-gse
#SBATCH --partition=gpu-a100-small
#SBATCH -n 2
#SBATCH -c 2
#SBATCH --mem-per-cpu=4000M
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ntmucke/lorenz96_delftblue/slurm/logs/%x_%j.out
S=/scratch/ntmucke/lorenz96_delftblue/slurm
echo "job $SLURM_JOB_ID $(date -Is)"; nvidia-smi -L
for obs in linear arctan; do
  srun --exact -n1 -c2 --gres=gpu:1 --mem=8000M bash "$S/shared_tune_lane.sh" $obs > "$S/logs/l96_shtune_${SLURM_JOB_ID}_$obs.out" 2>&1 &
done
wait; echo "all lanes done $(date -Is)"
