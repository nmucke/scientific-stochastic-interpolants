#!/bin/bash
#SBATCH --job-name=b_gpucheck
#SBATCH --partition=gpu-a100-small
#SBATCH --account=research-ceg-gse
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=7900M
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/ntmucke/scientific-stochastic-interpolants/slurm_bench/gpucheck_%j.out
cd /home/ntmucke/scientific-stochastic-interpolants
nvidia-smi; nvidia-smi -L
.venv/bin/python - <<'P'
import torch
print('torch',torch.__version__,torch.version.cuda,'avail',torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.mem_get_info())
    x=torch.randn(2048,2048,device='cuda'); print((x@x).sum().item())
P
echo PY_RC=$?
