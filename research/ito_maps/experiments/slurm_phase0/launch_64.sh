#!/bin/bash
# Submit the 64x64 Phase 0 stages once their checkpoints exist (DONE marker).
# Run from the repository root (login node: only sbatch is called).
#   research/ito_maps/experiments/slurm_phase0/launch_64.sh stage2
#   research/ito_maps/experiments/slurm_phase0/launch_64.sh stage3 <map run> [d1_job_id]
# Settings follow PROTOCOL.md at 64x64 (phase0.sbatch, CASE=ns64): E=32 for D1/D2
# spectra and spread ratios, E=16 for short-rollout scores, all 20 validation
# trajectories x origins 5/25/45/65/85, long runs 160/165/170/175, E=8, 500 leads.
#
# Time limits are expected run time plus margin, so that the jobs can backfill
# (our fair-share is low). Expectations are the 128x128 MIG timings of
# 2026-11-23 scaled by the work and by ~1/2 to 1/4 for the 64x64 grid; they
# are estimates, not measurements. Every stage is resumable: if a job times
# out, resubmit the same command (finished outputs are skipped). Override a
# limit with TL_<JOB>=HH:MM:SS.
#   stage2  si_d1    smokes + D1 SI-128 EM+exact, E=32, 100 cases     ~0.5-0.8 h -> 1:30
#           si_short D5/D6 short, 100 cases x 10 leads x 3 calls, E=16 ~2-3 h    -> 4:00
#           si_long  500 leads x 4 traj x E=8, then D7                 ~0.7-1.2 h -> 2:00
#   stage3  map_spec smoke, D2 (+ D3, D4 for Ito), short at 8/4/2/1   ~0.5-1 h   -> 1:30
#           map_long long at 8/4/2/1 (15 map calls per lead in all)    ~0.3-0.6 h -> 1:00
set -euo pipefail
SB=research/ito_maps/experiments/slurm_phase0/phase0.sbatch
CK=checkpoints/stochastic_navier_stokes_64
V100=(--partition=gpu-v100 --cpus-per-task=4 --mem-per-cpu=5000M --gres=gpu:1)
done_or_die() { [ -f "$CK/$1/DONE" ] || { echo "$CK/$1 has no DONE marker yet"; exit 1; }; }

case ${1:-} in
  stage2)
    done_or_die wp1_si_white_t2_e150_seed0
    a=$(CASE=ns64 STAGES="d1_smoke short_smoke long_smoke d1" \
      sbatch --parsable --job-name=p0_64_si_d1 "${V100[@]}" --time=${TL_SI_D1:-01:30:00} $SB)
    b=$(CASE=ns64 STAGES="short" \
      sbatch --parsable --job-name=p0_64_si_short "${V100[@]}" --time=${TL_SI_SHORT:-04:00:00} $SB)
    c=$(CASE=ns64 STAGES="long" \
      sbatch --parsable --job-name=p0_64_si_long "${V100[@]}" --time=${TL_SI_LONG:-02:00:00} $SB)
    echo "stage2: SI d1 job $a, SI short job $b, SI long job $c (V100)"
    echo "stage3 needs the d1 reference: launch_64.sh stage3 <map run> $a" ;;
  stage3)
    tag=${2:?map run name}; done_or_die "$tag"
    dep=(); [ -n "${3:-}" ] && dep=(--dependency=afterok:$3)
    if [ -z "${3:-}" ] && [ ! -f /scratch/ntmucke/scientific-stochastic-interpolants/results_itomap/phase0_64/d1/d1_spectra_fft2.npz ]; then
      echo "no D1 reference yet: pass the stage-2 d1 job id"; exit 1
    fi
    case $tag in
      *_ito_*) spec="map_smoke map_d2 map_d3 map_d4 map_short"; smoke=$CK/ns64_smoke_1469970_ito3 ;;
      *) spec="map_smoke map_d2 map_short"; smoke=$CK/ns64_smoke_1469970_flow3 ;;
    esac
    name=${tag#itomap_base_}
    j=$(CASE=ns64 MAP=$CK/$tag MAP_TAG=$tag MAP_SMOKE=$smoke STAGES="$spec" \
      sbatch --parsable --job-name=p0_64_${name}_spec "${V100[@]}" --time=${TL_MAP_SPEC:-01:30:00} "${dep[@]}" $SB)
    k=$(CASE=ns64 MAP=$CK/$tag MAP_TAG=$tag STAGES="map_long" \
      sbatch --parsable --job-name=p0_64_${name}_long "${V100[@]}" --time=${TL_MAP_LONG:-01:00:00} $SB)
    echo "stage3 $tag: spectra/short job $j, long job $k" ;;
  *) echo "usage: $0 stage2 | stage3 <map run> [d1_job_id]"; exit 2 ;;
esac
