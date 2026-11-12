#!/bin/bash
set -u
REPO=/home/ntmucke/scientific-stochastic-interpolants
PY=/scratch/ntmucke/scientific-stochastic-interpolants/.venv-cu126/bin/python
PE="$REPO/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments"
OUT=/scratch/ntmucke/scientific-stochastic-interpolants/slurm_bench/v100
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 JAX_COMPILATION_CACHE_DIR=$OUT/.jax_cache
cd "$REPO"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$PY" - <<'P'
import torch, jax, jax.numpy as jnp
print("torch", torch.__version__, torch.version.cuda, torch.cuda.get_arch_list(), torch.cuda.get_device_name(0))
a = torch.randn(512, 512, device="cuda"); print("torch matmul ok", float((a @ a).sum()) == float((a @ a).sum()))
c = torch.nn.Conv2d(4, 8, 3).cuda()(torch.randn(2, 4, 32, 32, device="cuda")); print("torch conv ok", tuple(c.shape))
print("jax", jax.__version__, jax.devices()); x = jnp.ones((256, 256)); print("jax matmul ok", float((x @ x).sum()))
import jax_cfd_lib, jax_cfd.base.grids; print("jax_cfd ok")
P
echo "PYCHECK rc=$?"
for M in "Ours (SI-SDE)|dps_jacobian_free" "EnKF|dps_jacobian_free" "D-Flow SGLD|dps_jacobian_free"; do
  meth=${M%|*}; s=$(echo "$meth" | tr -c 'A-Za-z0-9' _); t0=$(date +%s)
  "$PY" -u "$PE/run.py" case=navier_stokes seeds=[0] case.require_weights=true case.device=cuda +test_index=10 \
    +divergence_rmse_threshold=10.0 "+ns_methods=[\"$meth\"]" '+ns_scenarios=["32^2->128^2"]' +ic_index=15 \
    case.num_physical_steps=8 +skip_kl_reference=true ensemble_size=64 num_steps=100 likelihood_mode=${M#*|} \
    "results_file=$OUT/cell_$s.csv" "hydra.run.dir=$OUT/hydra_$s" > "$OUT/cell_$s.log" 2>&1
  rc=$?; echo "CELL $meth rc=$rc wall=$(( $(date +%s)-t0 ))s"; grep -hE "\[NS\]" "$OUT/cell_$s.log" | tail -1 | cut -c1-220
  [ $rc -ne 0 ] && grep -hE 'Error|error:' "$OUT/cell_$s.log" | tail -3 | cut -c1-220
done
nvidia-smi --query-gpu=memory.used --format=csv
