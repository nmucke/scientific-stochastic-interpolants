#!/bin/bash
# Submit the 64x64 Phase 0 stages once their checkpoints exist (DONE marker).
# Run from the repository root (login node: only sbatch is called).
#   research/ito_maps/experiments/slurm_phase0/launch_64.sh stage2
#   research/ito_maps/experiments/slurm_phase0/launch_64.sh stage3 itomap_base_ito_psd_local_seed0 [after_job_id]
#   research/ito_maps/experiments/slurm_phase0/launch_64.sh stage3 itomap_base_flow_psd_seed0 [after_job_id]
# stage2: SI reference. Job A (MIG): smokes, D1 SI-128 ensembles (EM + exact)
#   with the gate, D5/D6 short. Job B (V100): long run (500 leads, validation
#   trajectories 160/165/170/175, E=8) and D7.
# stage3: one map, one V100 job: D2 (needs stage-2 job A's d1 reference;
#   pass its job id to add an afterok dependency), D3 + D4 (Ito maps only),
#   D5/D6 short and long at budgets 8/4/2/1.
set -euo pipefail
SB=research/ito_maps/experiments/slurm_phase0/phase0.sbatch
CK=checkpoints/stochastic_navier_stokes_64
V100=(--partition=gpu-v100 --cpus-per-task=4 --mem-per-cpu=5000M --gres=gpu:1)
done_or_die() { [ -f "$CK/$1/DONE" ] || { echo "$CK/$1 has no DONE marker yet"; exit 1; }; }

case ${1:-} in
  stage2)
    done_or_die wp1_si_white_t2_e150_seed0
    a=$(CASE=ns64 STAGES="d1_smoke short_smoke long_smoke d1 short" \
      sbatch --parsable --job-name=p0_64_si --time=03:30:00 $SB)
    b=$(CASE=ns64 STAGES="long" sbatch --parsable --job-name=p0_64_silong "${V100[@]}" --time=16:00:00 $SB)
    echo "stage2: SI d1+short job $a (MIG), SI long job $b (V100)" ;;
  stage3)
    tag=${2:?map run name}; done_or_die "$tag"
    dep=(); [ -n "${3:-}" ] && dep=(--dependency=afterok:$3)
    case $tag in
      *_ito_*) stages="map_smoke map_d2 map_d3 map_d4 map_short map_long"; smoke=$CK/ns64_smoke_1469970_ito3 ;;
      *) stages="map_smoke map_d2 map_short map_long"; smoke=$CK/ns64_smoke_1469970_flow3 ;;
    esac
    j=$(CASE=ns64 MAP=$CK/$tag MAP_TAG=$tag MAP_SMOKE=$smoke STAGES="$stages" \
      sbatch --parsable --job-name=p0_64_${tag#itomap_base_} "${V100[@]}" --time=12:00:00 "${dep[@]}" $SB)
    echo "stage3 $tag: job $j" ;;
  *) echo "usage: $0 stage2 | stage3 <map run> [after_job_id]"; exit 2 ;;
esac
