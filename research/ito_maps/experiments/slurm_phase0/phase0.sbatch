#!/bin/bash
# Phase 0 diagnostics as resumable stages, for one dataset case.
#   CASE=ns64 STAGES="tests_before cache budget budget_trunc128 d1_data tests_after" \
#     sbatch research/ito_maps/experiments/slurm_phase0/phase0.sbatch
# CASE=ns128 (default) keeps the original 128x128 paths and checkpoint; CASE=ns64
# writes to research/ito_maps/results/phase0_64 and results_itomap/phase0_64.
# Everything dataset-specific (data file, normalisation, grid, history length,
# default gamma_multiplier) comes from the Hydra case config (data-only stages)
# or the checkpoint's config.yaml, never from this file.
# Stages skip when their output exists, so a resubmission after a timeout
# continues: d1 reuses its per-trajectory ensembles, short rollouts their
# per-case rows, long runs restart.pt; map d2/d3/d4 (minutes each) restart
# only the unfinished mode. Smoke
# stages run first in the same job; with set -e a failing smoke stops the job.
# Python: torch 2.9.1+cu126 venv (V100 and A100); the shared .venv has no sm_70 kernels.
# Map stages (map_*) need MAP=<checkpoint dir> and MAP_TAG=<name>; BUDGETS (default "8 4 2 1").
# Long stages belong on gpu-v100 (sbatch --partition=gpu-v100 --cpus-per-task=4 --mem-per-cpu=5000M --time=...).
#SBATCH --job-name=p0_diag
#SBATCH --account=research-ceg-gse
#SBATCH --partition=gpu-a100-small
#SBATCH --time=03:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=7000M
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/ntmucke/scientific-stochastic-interpolants/slurm_itomap/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=${PY:-/scratch/ntmucke/scientific-stochastic-interpolants/.venv_cu126/bin/python}
EXP=research/ito_maps/experiments
SCRATCH=/scratch/ntmucke/scientific-stochastic-interpolants
PRISTINE=${PRISTINE:-$SCRATCH/wt_itomap_phase0_pre64}
TESTS=${TESTS:-tests/test_quadratic_budget.py research/ito_maps/tests/test_phase0_diagnostics.py}
BUDGETS=${BUDGETS:-8 4 2 1}
case ${CASE:-ns128} in
  ns128)
    CONFIG_NAME=stochastic_navier_stokes
    SI=${SI:-checkpoints/stochastic_navier_stokes/stochastic_interpolant_small}
    RES=research/ito_maps/results/phase0
    ARR=$SCRATCH/results_itomap/phase0
    LONG_TRAJ=${LONG_TRAJ:-160 161 162 163}
    E_SPEC=${E_SPEC:-16}; E_SCORE=${E_SCORE:-8}            # as run at 128 (2026-11-23)
    SHORT_TRAJ=${SHORT_TRAJ:-4}; SHORT_ORIGINS=${SHORT_ORIGINS:-5 45 85} ;;
  ns64)
    CONFIG_NAME=stochastic_navier_stokes_64
    SI=${SI:-checkpoints/stochastic_navier_stokes_64/wp1_si_white_t2_e150_seed0}
    RES=research/ito_maps/results/phase0_64
    ARR=$SCRATCH/results_itomap/phase0_64
    # PROTOCOL.md evaluation cases: E=32 for spectra and spread ratios, E=16
    # for scores/calibration, short rollouts on all 20 validation trajectories
    # x origins 5/25/45/65/85, long runs on 160/165/170/175 with E=8.
    LONG_TRAJ=${LONG_TRAJ:-160 165 170 175}
    E_SPEC=${E_SPEC:-32}; E_SCORE=${E_SCORE:-16}
    SHORT_TRAJ=${SHORT_TRAJ:-20}; SHORT_ORIGINS=${SHORT_ORIGINS:-5 25 45 65 85} ;;
  *) echo "unknown CASE $CASE"; exit 2 ;;
esac
CACHE=$ARR/cache
CACHE128=$SCRATCH/results_itomap/phase0/cache
CACHE64=$SCRATCH/results_itomap/phase0_64/cache
mkdir -p $RES $ARR $CACHE
echo "$(date '+%F %T') job $SLURM_JOB_ID on $(hostname), case ${CASE:-ns128}, commit $(git rev-parse --short HEAD), dirty files $(git status --porcelain | wc -l)"
nvidia-smi -L 2>/dev/null || true
$PY -c "import scisi, torch; print('scisi from', scisi.__file__, torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
run() { echo "$(date '+%F %T') == stage $1"; }
need_map() { [ -n "${MAP:-}" ] && [ -n "${MAP_TAG:-}" ] || { echo "map stages need MAP and MAP_TAG"; exit 2; }; }
mres() { echo $RES/maps/$MAP_TAG; }
marr() { echo $ARR/maps/$MAP_TAG; }

for stage in ${STAGES:-cache d1_smoke short_smoke long_smoke budget d1}; do
  case $stage in
  cache)
    run $stage
    $PY $EXP/phase0_common.py --config-name $CONFIG_NAME --cache-dir $CACHE ;;
  tests_before)
    run $stage
    (cd $PRISTINE && echo "pristine commit $(git rev-parse --short HEAD)" && PYTHONPATH=$PRISTINE/src \
      PHASE0_DATA_CACHE=$CACHE128 $PY -m pytest $TESTS -q -p no:cacheprovider 2>&1 | tail -25) || true ;;
  tests_after)
    run $stage
    (PHASE0_DATA_CACHE=$CACHE128 PHASE0_DATA_CACHE_64=$CACHE64 \
      $PY -m pytest $TESTS -q -p no:cacheprovider 2>&1 | tail -25) || true ;;
  budget)
    # Archive budget closure of this case's data (train, validation).
    run $stage
    [ -f $RES/budget_archive.json ] || $PY $EXP/ns_budget.py --config-name $CONFIG_NAME --data-cache $CACHE \
      --output $RES/budget_archive.json ;;
  budget_trunc128)
    # 128x128 archive, full and low-passed to |k_i| < 32: the expected 64x64 shortfall.
    run $stage
    [ -f $RES/budget_truncation_128.json ] || $PY $EXP/ns_budget.py --config-name stochastic_navier_stokes \
      --data-cache $CACHE128 --spectral-cutoff 32 --output $RES/budget_truncation_128.json ;;
  d1_data)
    # D1 increment/climatology spectra of the data only (no model; white level from the case config).
    run $stage
    [ -f $RES/d1_increment_spectrum.json ] || $PY $EXP/uncertainty_spectrum.py --config-name $CONFIG_NAME \
      --data-cache $CACHE --no-ensemble --arrays-dir $ARR/d1_data --output $RES/d1_increment_spectrum.json ;;
  d1_smoke)
    run $stage
    $PY $EXP/uncertainty_spectrum.py --checkpoint-dir $SI --data-cache $CACHE \
      --max-val-trajectories 2 --max-train-trajectories 2 --origins 5 45 --ensemble 4 --steps 8 --bootstrap 50 \
      --arrays-dir $ARR/smoke/d1_$SLURM_JOB_ID --output $ARR/smoke/d1_$SLURM_JOB_ID/d1.json ;;
  d1)
    run $stage
    [ -f $RES/d1_uncertainty_spectrum.json ] || $PY $EXP/uncertainty_spectrum.py --checkpoint-dir $SI --data-cache $CACHE \
      --ensemble $E_SPEC --steps 128 --si-noise em exact \
      --arrays-dir $ARR/d1 --output $RES/d1_uncertainty_spectrum.json ;;
  short_smoke)
    run $stage
    $PY $EXP/rollout_diagnostics.py short --checkpoint-dir $SI --data-cache $CACHE --trajectories 1 --origins 5 \
      --ensemble 3 --leads 2 --steps 4 --arrays-dir $ARR/smoke/short_$SLURM_JOB_ID --output $ARR/smoke/short_$SLURM_JOB_ID/short.json ;;
  short)
    run $stage
    [ -f $RES/d5_d6_short_si128_exact.json ] || $PY $EXP/rollout_diagnostics.py short --checkpoint-dir $SI --data-cache $CACHE \
      --trajectories $SHORT_TRAJ --origins $SHORT_ORIGINS --ensemble $E_SCORE --leads 10 --steps 128 --si-noise exact \
      --arrays-dir $ARR/short_si128_exact --output $RES/d5_d6_short_si128_exact.json ;;
  long_smoke)
    run $stage
    out=$ARR/smoke/long_$SLURM_JOB_ID
    first=${LONG_TRAJ%% *}
    $PY $EXP/long_horizon_ns.py --checkpoint-dir $SI --output $out --steps 4 --horizon 3 --ensemble 2 \
      --trajectories $first --origin 5 --raw-every 1 --raw-dense 5 --checkpoint-every 1 --si-noise exact --num-threads 2
    $PY $EXP/long_horizon_ns.py --checkpoint-dir $SI --output $out --steps 4 --horizon 5 --ensemble 2 \
      --trajectories $first --origin 5 --raw-every 1 --raw-dense 5 --checkpoint-every 1 --si-noise exact --num-threads 2 --resume
    $PY $EXP/rollout_diagnostics.py long --checkpoint-dir $SI --run-dir $out --data-cache $CACHE --burn-in 1 --output $out/long.json ;;
  long)
    run $stage
    out=$ARR/long_si128_exact
    resume=(); [ -f $out/restart.pt ] && resume=(--resume)
    $PY $EXP/long_horizon_ns.py --checkpoint-dir $SI --output $out --steps 128 --horizon ${HORIZON:-500} --ensemble 8 \
      --trajectories $LONG_TRAJ --origin 5 --raw-every 1 --raw-dense 500 --checkpoint-every 10 --si-noise exact \
      --num-threads 2 "${resume[@]}"
    $PY $EXP/rollout_diagnostics.py long --checkpoint-dir $SI --run-dir $out --data-cache $CACHE --burn-in 100 \
      --output $RES/d5_d6_d7_long_si128_exact.json ;;
  plots)
    run $stage
    $PY $EXP/plot_phase0.py --results $RES ;;
  map_smoke)
    # Code test of the map paths (not a result). MAP_SMOKE: a smoke map checkpoint of this case.
    run $stage
    M=${MAP_SMOKE:-$SCRATCH/results_itomap/phase0/smoke_ckpt/itomap_smoke_ito_1445467}
    out=$ARR/smoke/map_$SLURM_JOB_ID
    for m in d3 d4; do
      $PY $EXP/map_spread_diagnostics.py $m --checkpoint-dir $M --data-cache $CACHE --val-ids 160 162 --origins 5 \
        --ensemble 4 --budgets 4 2 1 --arrays-dir $out/$m --output $out/$m.json
    done
    if [ -f $ARR/d1/d1_spectra_fft2.npz ]; then
      $PY $EXP/map_spread_diagnostics.py d2 --checkpoint-dir $M --data-cache $CACHE --ensemble 16 --budgets 8 1 \
        --reference $ARR/d1/d1_spectra_fft2.npz --arrays-dir $out/d2 --output $out/d2.json
    fi
    $PY $EXP/rollout_diagnostics.py short --checkpoint-dir $M --data-cache $CACHE --trajectories 1 --origins 5 \
      --ensemble 3 --leads 2 --steps 2 --arrays-dir $out/short --output $out/short.json
    $PY $EXP/long_horizon_ns.py --checkpoint-dir $M --output $out/long --steps 2 --horizon 3 --ensemble 2 \
      --trajectories 160 --origin 5 --raw-every 1 --raw-dense 5 --checkpoint-every 1 --num-threads 2
    $PY $EXP/rollout_diagnostics.py long --checkpoint-dir $M --run-dir $out/long --data-cache $CACHE --burn-in 1 --output $out/long.json ;;
  map_d2)
    # D2: map spread spectrum vs the SI-128 exact-noise reference of stage d1 (same cases, E=E_SPEC).
    run $stage; need_map; mkdir -p $(mres)
    [ -f $ARR/d1/d1_spectra_fft2.npz ] || { echo "D2 needs the stage-2 D1 reference"; exit 2; }
    [ -f $(mres)/d2.json ] || $PY $EXP/map_spread_diagnostics.py d2 --checkpoint-dir $MAP --data-cache $CACHE --ensemble $E_SPEC \
      --budgets $BUDGETS --reference $ARR/d1/d1_spectra_fft2.npz --reference-name anomaly_si128_exact \
      --arrays-dir $(marr)/d2 --output $(mres)/d2.json ;;
  map_d3|map_d4)
    # D3 zeroed path features / D4 noise-leak regression (Ito maps only).
    run $stage; need_map; mkdir -p $(mres); m=${stage#map_}
    [ -f $(mres)/$m.json ] || $PY $EXP/map_spread_diagnostics.py $m --checkpoint-dir $MAP --data-cache $CACHE --ensemble $E_SPEC \
      --budgets $BUDGETS --arrays-dir $(marr)/$m --output $(mres)/$m.json ;;
  map_short)
    run $stage; need_map; mkdir -p $(mres)
    for b in $BUDGETS; do
      [ -f $(mres)/d5_d6_short_b$b.json ] || $PY $EXP/rollout_diagnostics.py short --checkpoint-dir $MAP --data-cache $CACHE \
        --trajectories $SHORT_TRAJ --origins $SHORT_ORIGINS --ensemble $E_SCORE --leads 10 --steps $b \
        --arrays-dir $(marr)/short_b$b --output $(mres)/d5_d6_short_b$b.json
    done ;;
  map_long)
    run $stage; need_map; mkdir -p $(mres)
    for b in $BUDGETS; do
      out=$(marr)/long_b$b
      [ -f $(mres)/d5_d6_d7_long_b$b.json ] && continue
      resume=(); [ -f $out/restart.pt ] && resume=(--resume)
      $PY $EXP/long_horizon_ns.py --checkpoint-dir $MAP --output $out --steps $b --horizon ${HORIZON:-500} --ensemble 8 \
        --trajectories $LONG_TRAJ --origin 5 --raw-every 1 --raw-dense 500 --checkpoint-every 10 --num-threads 2 "${resume[@]}"
      $PY $EXP/rollout_diagnostics.py long --checkpoint-dir $MAP --run-dir $out --data-cache $CACHE --burn-in 100 \
        --output $(mres)/d5_d6_d7_long_b$b.json
    done ;;
  *) echo "unknown stage $stage"; exit 2 ;;
  esac
done
echo "$(date '+%F %T') done"
