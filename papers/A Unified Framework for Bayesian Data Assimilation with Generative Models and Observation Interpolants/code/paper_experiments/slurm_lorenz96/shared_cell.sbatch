#!/bin/bash
# One shared-Jacobian cell: line $SLURM_ARRAY_TASK_ID of $JOBLIST = "<cell> <M>"; picks from
# results/lorenz96/$OBS/tuning/selected_shared_stage{1,2}.json; outputs tagged "shared".
#SBATCH --job-name=l96_shcell
#SBATCH --account=research-ceg-gse
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2000M
#SBATCH --output=/scratch/ntmucke/lorenz96_delftblue/slurm/logs/%x_%A_%a.out
set -euo pipefail
: "${OBS:?}" "${JOBLIST:?}"; SEED=0
read -r CELL M < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBLIST")
W=/home/ntmucke/ssi-lorenz96
PE="$W/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments"
cd "$W"; source .venv/bin/activate
export PYTHONPATH=$W/src MPLBACKEND=Agg SCISI_REQUIRE_CUDA=1
cd "$PE"; T=results/lorenz96/$OBS/tuning
PARENT=${CELL%_fc}
mapfile -t SETS < <(python - "$T" "$PARENT" "$CELL" <<'PY'
import json, sys
t, parent, cell = sys.argv[1:]
picks = dict(json.load(open(f"{t}/selected_shared_stage1.json"))[parent])
if cell != parent:
    picks.update(json.load(open(f"{t}/selected_shared_stage2.json"))[cell])
for k, v in picks.items():
    print("--set"); print(f"{k}={json.dumps(v)}")
PY
)
OUT="results/lorenz96/$OBS/metrics/lorenz96__${CELL}-shared__M${M}__traj${SEED}__.csv"
echo "job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID host $(hostname) obs=$OBS cell=$CELL-shared M=$M ${SETS[*]} commit $(git rev-parse --short HEAD) start $(date -Is)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
if [[ -s "$OUT" ]]; then echo "skip: $OUT exists"; exit 0; fi
python cases/lorenz96/driver.py --obs "$OBS" cell --method "$CELL" --seed "$SEED" --num-steps "$M" \
  --tag shared "${SETS[@]}" --threads 4
test -s "$OUT" || { echo "ERROR: $OUT not written"; exit 1; }
echo "done $(date -Is)"
