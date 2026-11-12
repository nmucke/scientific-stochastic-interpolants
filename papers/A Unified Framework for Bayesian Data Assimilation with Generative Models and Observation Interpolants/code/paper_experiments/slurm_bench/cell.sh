#!/bin/bash
# cell.sh <name> <timeout_s> <run.py args...>
B=/scratch/ntmucke/scientific-stochastic-interpolants/slurm_bench
name=$1; to=$2; shift 2
R=/home/ntmucke/scientific-stochastic-interpolants
cd $R; PE=$(echo $R/papers/*/code/paper_experiments)
export PYTHONPATH=$B/hook JAX_COMPILATION_CACHE_DIR=$B/jax_cache
t0=$(date +%s)
timeout $to $R/.venv/bin/python -u "$PE/run.py" case=navier_stokes seeds=[0] ensemble_size=${E:-64} \
  case.require_weights=true case.device=cuda +test_index=10 +skip_kl_reference=true \
  +divergence_rmse_threshold=10.0 results_file=$B/cells/$name.csv hydra.run.dir=$B/cells/hydra_$name \
  "$@" > $B/cells/$name.log 2>&1
rc=$?; echo "[$(date +%T)] CELL $name rc=$rc wall=$(( $(date +%s)-t0 ))s" | tee -a $B/stream.log
