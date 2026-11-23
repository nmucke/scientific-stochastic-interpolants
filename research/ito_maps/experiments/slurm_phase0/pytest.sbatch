#!/bin/bash
# Run the repository test suite on a CPU node (torch import is slow on the login node).
#   TESTS="tests research/ito_maps/tests" sbatch research/ito_maps/experiments/slurm_phase0/pytest.sbatch
#SBATCH --job-name=p0_pytest
#SBATCH --account=research-ceg-gse
#SBATCH --partition=compute-p1
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/ntmucke/scientific-stochastic-interpolants/slurm_itomap/logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "$(date '+%F %T') commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
.venv/bin/python -c "import scisi, torch; print('scisi from', scisi.__file__, torch.__version__)"
.venv/bin/python -m pytest ${TESTS:-tests research/ito_maps/tests} -q ${PYTEST_ARGS:-}
echo "exit $?"
