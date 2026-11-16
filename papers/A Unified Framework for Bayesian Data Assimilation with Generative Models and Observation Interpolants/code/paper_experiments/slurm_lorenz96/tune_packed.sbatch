#!/bin/bash
#SBATCH --job-name=l96_tune
#SBATCH --account=research-ceg-gse
#SBATCH --partition=gpu-v100
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem-per-cpu=3000M
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ntmucke/lorenz96_delftblue/slurm/logs/%x_%j.out
set -euo pipefail
OBS=${OBS:-arctan}; LANES=${LANES:-6}
S=/scratch/ntmucke/lorenz96_delftblue/slurm
W=/home/ntmucke/ssi-lorenz96
PE="$W/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments"
cd "$W"; source .venv/bin/activate
export PYTHONPATH=$W/src MPLBACKEND=Agg SCISI_REQUIRE_CUDA=1 OBS THREADS=2
cd "$PE"; mkdir -p "results/lorenz96/$OBS/logs"
echo "job $SLURM_JOB_ID host $(hostname) obs=$OBS lanes=$LANES commit $(git rev-parse --short HEAD) start $(date -Is)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
( while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; sleep 60; done ) > "$S/logs/l96_tune_${SLURM_JOB_ID}_gpu.csv" &
MON=$!
python cases/lorenz96/driver.py --obs "$OBS" reference --split validation
rows() { grep -oE "^  '$1\|[^']*'" run_lorenz96.sh | sed "s/^  '//; s/'\$//"; }
# longest first (dflow, fig), the rest in script order
order() { grep -E '\|(dflow|fig|surge_sda|surge_flowdas)\|' || true; }
for st in 1 2; do
  echo "=== stage $st lanes $(date -Is)"
  { rows $st | order; rows $st | grep -vE '\|(dflow|fig|surge_sda|surge_flowdas)\|'; } \
    | xargs -d '\n' -P "$LANES" -I{} "$S/tune_lane.sh" {}
  if [[ $st == 1 ]]; then python select_lorenz96_tuning.py --obs "$OBS" --stage 1; fi
done
echo "=== canonical pass $(date -Is)"
kill $MON || true
SEEDS=0 ./run_lorenz96.sh tune "$OBS"
echo "done $(date -Is)"
