#!/bin/bash
# run_cell.sh <cell_index 0..19>: true-solver EnKF, E=1000, Gaspari-Cohn r=20 on ALL scenarios, on the
# wrong-IC ("poor IC") NS grid (ic_index=15, NP=40, traj 10..14), GPU, from the FROZEN copy that ran the
# main wrong-IC EnKF cells and the E=64 r=20 super-res cells (slurm_enkf_loc20/; only E differs from those).
# Cell i: scenario SC[i/5], trajectory 10 + i%5. Skips finished cells; publishes on success only.
set -u
i=${1:?cell index}; SC=("16^2->128^2" "32^2->128^2" "sparse 5%" "sparse 1.5625%")
SCEN=${SC[$((i / 5))]}; TRAJ=$((10 + i % 5)); E=${E:-1000}; NP=${NP:-40}
ROOT=${ROOT:-/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/results_wrong_ic_np40_enkf_r20_e1000/navier_stokes}
F=/scratch/ntmucke/scientific-stochastic-interpolants/code_ns_wrong_ic
PE="$F/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments"
PY=/scratch/ntmucke/scientific-stochastic-interpolants/.venv-cu126/bin/python
export PYTHONPATH="$F/src:$F/external_libs/jax_cfd_lib/src" PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest JAX_COMPILATION_CACHE_DIR="$ROOT/.jax_cache"
SS=$(echo "$SCEN" | sed 's/\^/_/g; s/->/_/; s/[^A-Za-z0-9_]/_/g; s/_\+/_/g; s/_$//'); ID="${SS}__traj${TRAJ}__classical__enkf_r20_E${E}"
OUT="$ROOT/metrics/$ID.csv"; PS="$ROOT/per_step/$ID.csv"
[ -f "$OUT" ] && { echo "SKIP exists $ID"; exit 0; }
mkdir -p "$ROOT"/{metrics,per_step,logs,hydra,.tmp,states/traj11}
TMP="$ROOT/.tmp/$ID.${SLURM_JOB_ID:-x}.csv"; PSTMP="$ROOT/.tmp/$ID.${SLURM_JOB_ID:-x}.ps.csv"
SAVE=(+save_states=false "+states_root=$ROOT/states/_unused"); [ "$TRAJ" = 11 ] && SAVE=(+save_states=true "+states_root=$ROOT/states/traj11")
cd "$F"
"$PY" -c "import scisi,sys; assert scisi.__file__.startswith('$F/'), scisi.__file__" || exit 4
echo "RUN $ID on $(hostname) $(nvidia-smi -L 2>/dev/null | head -1) $(date +%T)"; t0=$SECONDS
"$PY" -u "$PE/run.py" case=navier_stokes seeds=[0] case.require_weights=true case.device=cuda \
  "+test_index=$TRAJ" +divergence_rmse_threshold=10.0 "results_file=$TMP" "hydra.run.dir=$ROOT/hydra/$ID" \
  +save_per_step=true "+per_step_file=$PSTMP" '+ns_methods=["EnKF"]' "+ns_scenarios=[\"$SCEN\"]" \
  +ic_index=15 case.num_physical_steps=$NP +skip_kl_reference=true ensemble_size=$E num_steps=50 \
  +enkf_localization_radius=20 "${SAVE[@]}" > "$ROOT/logs/$ID.log" 2>&1
rc=$?
printf '%s\t%s\t%s\t%s\t%s\n' "${SLURM_JOB_ID:-x}" "$(hostname)" "$ID" "$rc" "$((SECONDS - t0))" >> "$ROOT/logs/timing.tsv"
if [ $rc -eq 0 ] && [ -s "$TMP" ]; then mv -f "$PSTMP" "$PS"; mv -f "$TMP" "$OUT"; echo "OK $ID $(date +%T)"; else echo "FAIL rc=$rc $ID (log $ROOT/logs/$ID.log)"; exit 1; fi
