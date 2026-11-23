#!/bin/bash
# WP1 step 1 diagnostics of one trained SI drift, no map training:
#  (1) nested-MC path compression, omitted endpoint-variance fraction at K=1,4
#      (white-process lab value ~9.7% at K=4), on validation histories;
#  (2) Euler + exact integrated noise at N=1,2,4,8 against its own N=128:
#      spread ratio, fair CRPS, shell spectra (validation, 20 traj x 5 origins).
#
#   RUN=wp1_si_coloured_matched_seed0 sbatch research/ito_maps/experiments/slurm_wp1/eval_si.sbatch
#SBATCH --job-name=wp1_eval
#SBATCH --account=research-ceg-gse
#SBATCH --partition=gpu-a100
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=12000M
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/ntmucke/scientific-stochastic-interpolants/slurm_itomap/logs/%x_%j.out
source research/ito_maps/experiments/slurm_wp1/common.sh
: "${RUN:?set RUN}"
ckpt="checkpoints/stochastic_navier_stokes/$RUN"
out="$WP1_ROOT/eval/$RUN"
HISTORIES=${HISTORIES:-"160:45 165:25 170:65 175:85"}
GROUPS_=${GROUPS_:-16}
REPLICAS=${REPLICAS:-8}
TEACHER_STEPS=${TEACHER_STEPS:-64}
EULER_ARGS=${EULER_ARGS:-}
for pair in $HISTORIES; do
  trajectory=${pair%%:*}; origin=${pair##*:}
  target="$out/path_compression/traj${trajectory}_origin${origin}"
  if [ -f "$target/results.json" ]; then continue; fi
  $PY research/ito_maps/experiments/ns_path_compression.py --checkpoint-dir "$ckpt" \
    --trajectory "$trajectory" --origin "$origin" --features 1 4 \
    --groups "$GROUPS_" --replicas "$REPLICAS" --teacher-steps "$TEACHER_STEPS" \
    --batch-size 8 --device cuda --output "$target"
done
if [ ! -f "$out/euler_exact_noise/results.json" ]; then
  # shellcheck disable=SC2086
  $PY research/ito_maps/experiments/wp1_euler_exact_noise.py --checkpoint-dir "$ckpt" \
    --output "$out/euler_exact_noise" $EULER_ARGS
fi
echo "$(date '+%F %T') done $RUN"
