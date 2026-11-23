#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
A="$REPO/archive/paper/scripts/analytical_utils"
mv "$A" "$A.hidden_for_test"
trap 'mv "$A.hidden_for_test" "$A"' EXIT
cd "$PE" && find cases tests -name __pycache__ -prune -exec rm -rf {} + ; python -m pytest tests/test_analytical_metrics.py tests/test_analytical_multistep.py tests/test_analytical_multistep_correction.py -q
