#!/bin/bash
#SBATCH --job-name=sync_verify_results
#SBATCH --partition=gpu-a100-small
#SBATCH --account=research-ceg-gse
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=7900M
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/sync_logs/sync_verify_%j.out
# Copy every result tree into paper_experiments/results/ (copy-only, overwritten files kept
# in results/_overwritten/), then md5-verify EVERY file and write results/MANIFEST.md5.
# Runs as a job: the login nodes are not for jobs > 15 min. On a MIG slice (2 CPUs) because the
# compute partitions had no free slot before 2026-11-20; the work is I/O (rsync, md5), no GPU.
set -uo pipefail
S=/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments
TOOLS=${TOOLS:-$S}
echo "host $(hostname) start $(date -Is)"
bash "$TOOLS/sync_all_results.sh"; rc_sync=$?
echo "sync exit $rc_sync"
python3 "$TOOLS/verify_results_export.py" --procs "$SLURM_CPUS_PER_TASK"; rc_ver=$?
echo "verify exit $rc_ver   done $(date -Is)"
exit $(( rc_ver ))
