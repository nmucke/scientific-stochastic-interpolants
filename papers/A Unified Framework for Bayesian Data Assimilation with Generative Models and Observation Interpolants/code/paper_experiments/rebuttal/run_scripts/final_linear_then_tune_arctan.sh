#!/usr/bin/env bash
# Seed 0 only (user 2026-11-10). Stops on the first failure.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PE"
until ls results/_runs/build_pf_terms_*.log.exit >/dev/null 2>&1; do sleep 10; done
[[ "$(cat results/_runs/build_pf_terms_*.log.exit)" == 0 ]] || { echo "pf-term cache build failed"; exit 1; }
echo "== final linear SEEDS=0 $(date -Is) commit $(git rev-parse --short HEAD)"
SEEDS=0 THREADS=8 ./run_lorenz96.sh final linear
echo "== tune arctan $(date -Is)"
THREADS=8 ./run_lorenz96.sh tune arctan
echo "CHAIN DONE $(date -Is) (arctan picks NOT applied: review selected_stage{1,2}.md first)"
