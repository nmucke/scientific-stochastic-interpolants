#!/bin/bash
# =============================================================================
# CHAIN: wait for the running urban grid, then launch the NS cadence ablation.
# (2026-08-12)
#
# Waits for the run_urban_grid.sh master process to exit, then starts
# run_ns_ablation.sh on the same box. The two jobs must NOT overlap -- both are
# single-GPU and the urban D-Flow cells already sit near the card's memory
# ceiling (posterior_batch_size=16 was chosen for a 24 GiB card), so starting
# the ablation early would OOM one or both.
#
# GUARD: the ablation starts ONLY if the urban log ends with "ALL DONE", i.e.
# the grid ran to completion. If the master is killed (Ctrl-C, reboot, OOM
# killer, or a deliberate stop) there is no ALL DONE line and this exits
# without launching -- stopping the urban run should not silently commit the
# box to another multi-day job. Individual FAILed cells do NOT block the
# launch: run_cell logs a failure and continues, so ALL DONE means "the script
# finished", not "every cell succeeded". Check the log for FAIL lines.
# Override the guard with FORCE=1 (launches whenever the PID disappears).
#
# PID reuse is guarded by re-reading /proc/<pid>/cmdline every poll: the wait
# ends when the PID is gone OR the slot has been recycled by something else.
#
# Usage (already launched; re-run only if this watcher itself dies):
#   URBAN_PID=<pid> setsid nohup bash paper_experiments/chain_ns_ablation.sh \
#       >>chain_ns_ablation.log 2>&1 & disown
# =============================================================================
set -u
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/common/paths.sh" || exit 1
cd "$PE_REPO_ROOT"

URBAN_PID="${URBAN_PID:-2650471}"
URBAN_LOG="${URBAN_LOG:-$PE_SCRIPT_DIR/results/urban/run_urban_grid.log}"
ABLATION_LOG="${ABLATION_LOG:-$PE_SCRIPT_DIR/run_ns_ablation.log}"
POLL="${POLL:-120}"          # seconds between liveness checks
SETTLE="${SETTLE:-120}"      # pause after exit, so the GPU frees before we start
FORCE="${FORCE:-0}"          # 1 = launch even if urban did not reach ALL DONE

say() { echo "[chain] $(date '+%F %T') $*"; }

say "watching urban grid pid=$URBAN_PID (poll ${POLL}s); will launch run_ns_ablation.sh on completion"

# Alive == the PID exists AND its cmdline still names the urban grid script.
urban_alive() {
  [ -r "/proc/$URBAN_PID/cmdline" ] || return 1
  tr '\0' ' ' < "/proc/$URBAN_PID/cmdline" 2>/dev/null | grep -q "run_urban_grid.sh"
}

if ! urban_alive; then
  say "WARNING: pid=$URBAN_PID is not a live run_urban_grid.sh right now -- proceeding to the completion check immediately."
fi

while urban_alive; do sleep "$POLL"; done
say "urban grid pid=$URBAN_PID has exited"

if [ "$FORCE" != "1" ]; then
  if tail -n 20 "$URBAN_LOG" 2>/dev/null | grep -q "ALL DONE"; then
    say "urban log ends with ALL DONE -- proceeding"
  else
    say "ABORT: no ALL DONE in the last 20 lines of $URBAN_LOG -- the grid did not"
    say "       run to completion, so the ablation was NOT started. Launch it by hand"
    say "       with: setsid nohup bash '$PE_SCRIPT_DIR/run_ns_ablation.sh' >'$ABLATION_LOG' 2>&1 & disown"
    exit 1
  fi
fi

nfail=$(grep -c "^\[urbangrid\] FAIL" "$URBAN_LOG" 2>/dev/null || echo 0)
say "urban cells that FAILed: $nfail (does not block the ablation; review the log)"

say "settling ${SETTLE}s for the GPU to free"
sleep "$SETTLE"

say "launching run_ns_ablation.sh (M=50, k=1 and 5, 4 scenarios, traj 10-14) -> $ABLATION_LOG"
bash "$PE_SCRIPT_DIR/run_ns_ablation.sh" >>"$ABLATION_LOG" 2>&1
say "run_ns_ablation.sh exited with status $?"
