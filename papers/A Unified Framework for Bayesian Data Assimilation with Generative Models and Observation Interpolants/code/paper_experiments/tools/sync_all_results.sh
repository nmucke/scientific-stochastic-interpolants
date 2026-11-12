#!/usr/bin/env bash
# sync_all_results.sh -- copy every rebuttal result tree into the ONE shared results tree.
#
#   COPY ONLY: rsync -a, never --delete, never mv/rm. Every source stays where it is
#   (running jobs and post-processing scripts write/read the source paths).
#   Idempotent: re-run any time; it only adds/updates files in the destination.
#
# Usage:   bash /scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/sync_all_results.sh
#          NSAMPLE=50 bash .../sync_all_results.sh      # more checksum samples per tree
# Log:     paper_experiments/sync_logs/sync_<timestamp>.log  (+ sync_logs/latest.log symlink)
# Index:   paper_experiments/results/INDEX.md (what every case folder is, status, job ids)
set -uo pipefail

S=/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments
R=$S/results
L96=/scratch/ntmucke/lorenz96_delftblue
L96_PE="/home/ntmucke/ssi-lorenz96/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments"
NSAMPLE=${NSAMPLE:-20}

LOGDIR=$S/sync_logs
mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$LOGDIR/sync_$TS.log
ln -sfn "$(basename "$LOG")" "$LOGDIR/latest.log"
exec > >(tee -a "$LOG") 2>&1

# one run at a time
exec 9>"$LOGDIR/.lock"
flock -n 9 || { echo "another sync_all_results.sh is running; abort"; exit 1; }

# name | source dir | destination dir     (all copied as  rsync -a SRC/ DST/)
TREES=(
  "lorenz96_results|$L96/results/lorenz96|$R/lorenz96"
  "lorenz96_figures|$L96/figures/lorenz96|$S/figures/lorenz96"
  "lorenz96_provenance_slurm|$L96/slurm|$R/lorenz96/_provenance/slurm"
  "ns_wrong_ic_np40_main|$S/results_wrong_ic_np40|$R/navier_stokes_wrong_ic_np40/main"
  "ns_wrong_ic_np40_ext|$S/results_wrong_ic_np40_ext|$R/navier_stokes_wrong_ic_np40/ext"
  "ns_wrong_ic_np40_prior_enkf|$S/results_wrong_ic_np40_prior_enkf|$R/navier_stokes_wrong_ic_np40/prior_enkf"
  "ns_enkf_loc20_superres|$S/enkf_loc20_superres|$R/navier_stokes_wrong_ic_np40/enkf_loc20_superres"
  "urban_fc|$S/results_udales_fc|$R/urban_fc"
  "smoke_prior_enkf|$S/smoke_prior_enkf|$R/_smoke_and_bench/smoke_prior_enkf"
  "slurm_bench|/scratch/ntmucke/scientific-stochastic-interpolants/slurm_bench|$R/_smoke_and_bench/slurm_bench"
)
# single files: source | destination
FILES=(
  "$L96/STATUS.md|$R/lorenz96/_provenance/STATUS.md"
  "$L96_PE/LORENZ96_HANDOVER_2026-11-12.md|$R/lorenz96/_provenance/LORENZ96_HANDOVER_2026-11-12.md"
  "$L96_PE/LORENZ96_DELFTBLUE_HANDOVER.md|$R/lorenz96/_provenance/LORENZ96_DELFTBLUE_HANDOVER.md"
)
# generated LaTeX tables from the Lorenz 96 worktree -> shared generated/ (re-globbed each run)
for f in "$L96_PE"/generated/tab_lorenz96*.tex; do
  [ -e "$f" ] && FILES+=("$f|$S/generated/$(basename "$f")")
done

nfiles() { # count non-directory entries; $2 = optional path to exclude (prune)
  if [ -n "${2:-}" ]; then find "$1" -path "$2" -prune -o ! -type d -print 2>/dev/null | wc -l
  else find "$1" ! -type d 2>/dev/null | wc -l; fi
}

echo "=== sync_all_results.sh  $(date -Is)  host=$(hostname)  NSAMPLE=$NSAMPLE"
FAIL=0
SUMMARY=()
for t in "${TREES[@]}"; do
  IFS='|' read -r name src dst <<<"$t"
  echo; echo "--- [$name] $src/  ->  $dst/"
  if [ ! -d "$src" ]; then echo "    SOURCE MISSING, skipped"; SUMMARY+=("$name|missing|-|-|-"); continue; fi
  mkdir -p "$dst"
  rsync -a "$src"/ "$dst"/ ; rc=$?
  # rc 24 = "some files vanished before they could be transferred" (live job tmp files): not fatal
  if [ $rc -ne 0 ] && [ $rc -ne 24 ]; then echo "    rsync exit $rc"; FAIL=1; fi
  # destination may contain extra provenance subtrees not in the source (lorenz96/_provenance)
  excl=""; [ "$name" = lorenz96_results ] && excl="$dst/_provenance"
  ns=$(nfiles "$src"); nd=$(nfiles "$dst" "$excl")
  # checksum sample: NSAMPLE random files + every aggregated*/ csv and tables/ file
  ok=0; bad=0
  while IFS= read -r rel; do
    [ -z "$rel" ] && continue
    a=$(md5sum <"$src/$rel" 2>/dev/null | cut -d' ' -f1); b=$(md5sum <"$dst/$rel" 2>/dev/null | cut -d' ' -f1)
    if [ -n "$a" ] && [ "$a" = "$b" ]; then ok=$((ok+1)); echo "    md5 OK   $a  $rel"
    else bad=$((bad+1)); echo "    md5 DIFF src=$a dst=$b  $rel  (if a job is writing it: re-run sync)"; fi
  done < <( { (cd "$src" && find . -type f ! -path '*/.jax_cache/*' ! -path '*/.tmp/*' | shuf -n "$NSAMPLE");
              (cd "$src" && find . -type f \( -path '*/aggregated*/*.csv' -o -path './tables/*' \)); } | sed 's|^\./||' | sort -u )
  st=$([ "$ns" = "$nd" ] && [ $bad -eq 0 ] && echo MATCH || echo CHECK)
  [ "$st" = CHECK ] && FAIL=1
  echo "    files: src=$ns dst=$nd   md5 sample: ok=$ok diff=$bad   => $st"
  SUMMARY+=("$name|$ns|$nd|$ok/$((ok+bad))|$st")
done

echo; echo "--- single files"
for f in "${FILES[@]}"; do
  IFS='|' read -r src dst <<<"$f"
  if [ ! -f "$src" ]; then echo "    missing source: $src"; continue; fi
  mkdir -p "$(dirname "$dst")"
  rsync -a "$src" "$dst"
  a=$(md5sum <"$src" | cut -d' ' -f1); b=$(md5sum <"$dst" | cut -d' ' -f1)
  [ "$a" = "$b" ] && echo "    md5 OK   $a  $dst" || { echo "    md5 DIFF $dst"; FAIL=1; }
done

echo; echo "=== SUMMARY (files: source vs destination; md5 sample ok/checked)"
printf "%-30s %8s %8s %10s %6s\n" tree src dst md5 status
for s in "${SUMMARY[@]}"; do IFS='|' read -r a b c d e <<<"$s"; printf "%-30s %8s %8s %10s %6s\n" "$a" "$b" "$c" "$d" "$e"; done
echo "running jobs at sync time:"; squeue -u "$USER" -h -o "%i %j %T" 2>/dev/null | awk '{print $2, $3}' | sort | uniq -c
echo "=== done $(date -Is)  overall: $([ $FAIL -eq 0 ] && echo ALL MATCH || echo 'SOME TREES NEED A LOOK (see CHECK/DIFF above; live jobs cause this)')"
echo "log: $LOG"
exit $FAIL
