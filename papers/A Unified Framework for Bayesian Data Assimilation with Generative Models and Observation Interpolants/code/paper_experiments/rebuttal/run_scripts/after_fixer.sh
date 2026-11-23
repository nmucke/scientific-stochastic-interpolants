#!/usr/bin/env bash
# Gate on the merged code (e88c057+): both test suites and the bitwise linear identity.
# Then, in parallel: (A) rerun Exp 1's SURGE cells with the corrected SI denoiser and
# re-render its tables/figures (CPU); (B) linear tuning, stage 1 + 2 + selection (GPU),
# which stops for review before `apply`.
set -uo pipefail
source "$(dirname "$0")/env.sh"
R="$PE/results/_runs"
cd "$REPO" && OMP_NUM_THREADS=8 python -m pytest tests/ -q -p no:cacheprovider 2>&1 | tail -3; rc1=${PIPESTATUS[0]}
cd "$PE" && OMP_NUM_THREADS=8 python -m pytest tests/ -q -p no:cacheprovider 2>&1 | tail -8; rc2=${PIPESTATUS[0]}
python "$R/linear_identity_snapshot.py" "$R/identity/after_fixer.pt" --compare "$R/identity/before_bb58153.pt" 2>&1 | grep -E "IDENTICAL|CHANGED|DIFF"; rc3=${PIPESTATUS[0]}
echo "GATE root=$rc1 pe=$rc2 identity=$rc3"
if [ "$rc1" != 0 ] || [ "$rc2" != 0 ] || [ "$rc3" != 0 ]; then echo "GATE FAILED: nothing started"; exit 1; fi

# (A) Exp 1 SURGE cells: move the velocity-denoiser files aside, rerun, re-render.
tmux new -d -s rb_exp1_surge "bash '$R/run_job.sh' exp1_surge_rerun 'bash \"$R/exp1_surge_rerun.sh\"'"
# (B) linear tuning (stops after stage-2 selection).
tmux new -d -s rb_tune_linear "bash '$R/run_job.sh' tune_linear 'cd \"\$PE\" && THREADS=8 ./run_lorenz96.sh tune linear'"
echo "STARTED rb_exp1_surge and rb_tune_linear"
