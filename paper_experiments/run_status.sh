#!/bin/bash
# =============================================================================
# One-glance status for whatever paper_experiments job is currently running.
#
#   bash paper_experiments/run_status.sh          # summary
#   bash paper_experiments/run_status.sh -v       # + per-cell timings and ETA
#
# Reads only logs and the results tree -- it starts nothing, kills nothing, and is
# safe to run at any time, including while a job is mid-cell.
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
VERBOSE="${1:-}"

hr() { printf '%s\n' "-------------------------------------------------------------"; }

# ---- which of our jobs are alive --------------------------------------------
echo "=== $(date '+%a %d %b %H:%M') ==="
hr
echo "JOBS ALIVE"
FOUND=0
for j in run_urban_grid.sh run_ns_grid.sh run_ns_classical.sh rerun_ns_surge_sda.sh \
         run_urban_tuning.sh run_urban_tuning_ours.sh run_ns_lambda_check.sh \
         measure_timing.py; do
  p=$(pgrep -f "$j" | head -1)
  [ -n "$p" ] && { printf "  %-28s pid=%-8s up=%s\n" "$j" "$p" "$(ps -o etime= -p "$p" | tr -d ' ')"; FOUND=1; }
done
[ "$FOUND" = 0 ] && echo "  (none -- the GPU is idle)"
echo "  scheduler(s) waiting:"
pgrep -af "sched_.*\.sh" | grep -v "bash -c" | sed -E 's|.*/([a-z_0-9]+\.sh).*|    \1|' || echo "    (none)"

# ---- GPU ---------------------------------------------------------------------
hr
echo "GPU"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader | sed 's/^/  /'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  /'

# ---- per-case coverage -------------------------------------------------------
# expected cell counts come from the grid scripts' own defaults.
# Derived from run_urban_grid.sh's CURRENT defaults rather than hardcoded, so this
# cannot drift when the grid is re-scoped (it has been re-scoped four times).
_g=paper_experiments/run_urban_grid.sh
_ntraj=$(sed -n 's/^TRAJ="${TRAJ:-\(.*\)}".*/\1/p'   $_g | wc -w)
_nstep=$(sed -n 's/^STEPS="${STEPS:-\(.*\)}".*/\1/p'  $_g | wc -w)
_ngrp=$(sed -n 's/^GRPS="${GRPS:-\(.*\)}".*/\1/p'     $_g | wc -w)
_nscen=$(sed -n 's/^else SCEN_ARR=(\(.*\)); fi.*/\1/p' $_g | grep -o '"' | wc -l)
_nscen=$(( _nscen / 2 )); [ "$_nscen" -lt 1 ] && _nscen=1
# GRPS is usually overridden on the command line (e.g. dflow dropped) -- prefer what
# the log says was actually requested for THIS run.
_loggrp=$(sed -n 's/.*groups=\[\([^]]*\)\].*/\1/p' paper_experiments/results/urban/run_urban_grid.log 2>/dev/null | tail -1 | wc -w)
[ "${_loggrp:-0}" -gt 0 ] && _ngrp=$_loggrp
urban_expected=$(( _ngrp * _nscen * _nstep * _ntraj ))
urban_shape="$_ngrp groups x $_nscen scen x $_nstep M x $_ntraj traj"
for case in navier_stokes urban; do
  MET=paper_experiments/results/$case/metrics
  [ -d "$MET" ] || continue
  hr
  n=$(ls "$MET"/*.csv 2>/dev/null | grep -v '/ref_' | wc -l)
  if [ "$case" = urban ]; then
    echo "URBAN GRID   $n / $urban_expected cells   ($urban_shape)"
  else
    echo "NAVIER-STOKES   $n metric files"
  fi
  L=paper_experiments/results/$case/run_${case}_grid.log
  [ "$case" = urban ] && L=paper_experiments/results/urban/run_urban_grid.log
  [ -f "$L" ] || continue
  # NB: grep -c exits 1 on zero matches, so `|| echo 0` would append a SECOND
  # "0" and break the arithmetic below. Count with grep -c and normalise instead.
  ok=$(grep -c '\] OK' "$L" 2>/dev/null); ok=${ok:-0}
  fail=$(grep -c '\] FAIL' "$L" 2>/dev/null); fail=${fail:-0}
  echo "  log: OK=$ok FAIL=$fail"
  # A FAIL only matters if that same cell never later succeeded (the grids are
  # skip-if-exists and get re-run, so raw FAIL counts over-report).
  unresolved=$(comm -23 \
      <(grep '\] FAIL' "$L" 2>/dev/null | awk '{print $4}' | sort -u) \
      <(grep '\] OK'   "$L" 2>/dev/null | awk '{print $4}' | sort -u))
  if [ -n "$unresolved" ]; then
    echo "  !! UNRESOLVED FAILURES (never succeeded on a retry):"
    echo "$unresolved" | sed 's/^/     /'
  else
    [ "$fail" -gt 0 ] && echo "  (all $fail failure(s) were later re-run OK)"
  fi
  echo "  now: $(grep -E '\] RUN' "$L" 2>/dev/null | tail -1 | cut -c1-88)"
  # within-cell progress: the driver's tqdm bar over the assimilation steps
  prog=$(tr '\r' '\n' < "$L" 2>/dev/null | grep -oE '[0-9]+%\|[^|]*\| *[0-9]+/[0-9]+ \[[^]]*\]' | tail -1)
  [ -n "$prog" ] && echo "  step: $prog"
done

# ---- verbose: recent cell wall-times ----------------------------------------
if [ "$VERBOSE" = "-v" ]; then
  hr
  echo "RECENT CELL TIMES (urban grid)"
  awk '/\] RUN/{tag=$3;t=$NF;split(t,a,":");s=a[1]*3600+a[2]*60+a[3]}
       /\] OK/ {t=$NF;split(t,a,":");e=a[1]*3600+a[2]*60+a[3];d=e-s;if(d<0)d+=86400;
                printf "  %-46s %6.1f min\n", tag, d/60}' \
      paper_experiments/results/urban/run_urban_grid.log 2>/dev/null | tail -8
fi

hr
echo "Coverage detail:  .venv/bin/python paper_experiments/status.py --case urban"
echo "Live log:         tail -f paper_experiments/results/urban/run_urban_grid.log"
