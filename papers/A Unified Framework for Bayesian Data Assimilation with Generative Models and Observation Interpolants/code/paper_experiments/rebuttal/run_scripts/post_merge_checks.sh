#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
set -x
cd "$REPO" && OMP_NUM_THREADS=4 python -m pytest tests/ -q -p no:cacheprovider 2>&1 | tail -15; echo "ROOT_RC=${PIPESTATUS[0]}"
cd "$PE" && OMP_NUM_THREADS=4 python -m pytest tests/ -q -p no:cacheprovider 2>&1 | tail -25; echo "PE_RC=${PIPESTATUS[0]}"
python "$PE/results/_runs/linear_identity_snapshot.py" "$PE/results/_runs/identity/after_merge.pt" --compare "$PE/results/_runs/identity/before_bb58153.pt" 2>&1 | grep -v "it/s" | tail -15
