#!/bin/bash
# =============================================================================
# Navier--Stokes M=25 fill -- PARALLEL CPU driver  (2026-07-15)
#
# Fans run_ns_grid_M25.sh out across (traj x scenario) shards on the CPU. This is
# a pure scheduler: every shard is just run_ns_grid_M25.sh with TRAJ/SCENARIOS
# pinned, so the grid definition, per-cell configs and skip-existing resume logic
# all live in ONE place (that script) and are not duplicated here.
#
# WHY SHARD AT ALL -- measured torch CPU conv scaling (E=64 @128^2, this box):
#     threads    1      2      4      8     16     32
#     ms/pass  9649   5295   3066   2086   2070   1922
#     speedup  1.00x  1.82x  3.15x  4.63x  4.66x  5.02x
#     eff.      100%    91%    79%    58%    29%    16%
# Scaling dies after ~8 threads: 8->32 threads buys 8% for 4x the cores. So N cells
# at 32/N threads beats 1 cell at 32 threads. Aggregate throughput vs 1x32:
#     4 workers x  8 thr -> 3.7x     8 workers x 4 thr -> 5.0x
#    16 workers x  2 thr -> 5.8x
#
# ...BUT RAM IS THE REAL CAP, NOT CORES. The 34 cells left are exactly the memory
# hogs -- 20 baselines (D-Flow SGLD backprops through the sampler) + 14 shared (holds
# the full Sigma_s Jacobian); every cheap jacfree cell already passed. A heavy cell
# peaks ~18 GB, and ~69 GB is free => WORKERS=3 (~54 GB) is the safe point, NOT 4
# (~72 GB, overcommitted). Raising WORKERS past what RAM allows trades a clean CUDA
# OOM for the kernel OOM-killer, which is worse: it SIGKILLs with no traceback and
# the cell dies without writing. Bump to 4 only after watching RSS on a first pass.
#
# Expected: ~5-6 days for the 34 remaining cells at WORKERS=3. (Borrowing the GPU for
# ~20 h is still ~6x faster -- this route's only advantage is that it leaves the
# running M=250 GPU job completely untouched.)
#
# Ordering: heavy groups (baselines) are the long pole, so shards are emitted with
# the sparse/superres mix interleaved -- no shard hogs the tail of the run.
#
# Launch detached:
#   setsid nohup bash paper_experiments/run_ns_grid_M25_parallel.sh \
#     >run_ns_grid_M25_parallel.log 2>&1 & disown
#
# Track:
#   tail -f paper_experiments/results/navier_stokes/logs_M25/shard_*.log
#   .venv/bin/python paper_experiments/status.py
#   grep -c OK paper_experiments/results/navier_stokes/logs_M25/*.log
#
# Env: WORKERS (default 4), THREADS (default 32/WORKERS), TRAJ, SCENARIOS, DEVICE.
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants

ROOT="${ROOT:-paper_experiments/results/navier_stokes}"
LOGDIR="$ROOT/logs_M25"
mkdir -p "$LOGDIR"

WORKERS="${WORKERS:-3}"                     # concurrent shards -- RAM-bound (~18 GB/shard)
THREADS="${THREADS:-$((32 / WORKERS))}"     # torch threads PER shard (32/3 = 10)
PEAK_GB="${PEAK_GB:-18}"                    # measured peak RSS of a heavy cell
SAVE_TRAJ_OWNER="${SAVE_TRAJ_OWNER:-11}"    # only this traj's shards write states
TRAJ_LIST="${TRAJ:-10 11 12 13 14}"
DEVICE="${DEVICE:-cpu}"

if [ -n "${SCENARIOS:-}" ]; then IFS='|' read -r -a SCEN_ARR <<< "$SCENARIOS";
else SCEN_ARR=("16^2->128^2" "32^2->128^2" "sparse 5%" "sparse 1.5625%"); fi

slug() { echo "$1" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//'; }

# ---- pre-flight: refuse to walk into the OOM-killer ------------------------
# The kernel SIGKILLs on overcommit with no traceback, so check BEFORE launching
# rather than discovering it 3 days in. Override with FORCE=1 if you know better.
AVAIL_GB=$(free -g | awk '/^Mem:/{print $7}')
NEED_GB=$((WORKERS * PEAK_GB))
echo "[par] RAM: ${AVAIL_GB} GB available | ${WORKERS} workers x ~${PEAK_GB} GB = ~${NEED_GB} GB needed"
if [ "$NEED_GB" -gt "$AVAIL_GB" ]; then
  echo "[par] WARNING: ~${NEED_GB} GB needed > ${AVAIL_GB} GB available -- risks the OOM-killer." >&2
  if [ "${FORCE:-0}" != "1" ]; then
    echo "[par] FATAL: refusing to start. Lower WORKERS (try $((AVAIL_GB / PEAK_GB))), or pass FORCE=1." >&2
    exit 1
  fi
  echo "[par] FORCE=1 set -- proceeding anyway." >&2
fi

echo "[par] START $(date) | workers=$WORKERS threads/shard=$THREADS dev=$DEVICE"
echo "[par] traj=[$TRAJ_LIST] scenarios=${#SCEN_ARR[@]} | logs -> $LOGDIR"
echo "[par] states written ONLY by traj$SAVE_TRAJ_OWNER shards (SAVE_TRAJ=none elsewhere)"

# ---- build the shard list: one line per (traj, scenario) ---------------------
# NUL-delimited so the scenario labels ("16^2->128^2", "sparse 5%") survive the
# hand-off to xargs intact -- they contain spaces, >, ^ and %.
SHARDS=$(mktemp); trap 'rm -f "$SHARDS"' EXIT
for N in $TRAJ_LIST; do
  for SCEN in "${SCEN_ARR[@]}"; do
    printf '%s\t%s\0' "$N" "$SCEN" >> "$SHARDS"
  done
done
NSHARD=$(tr -cd '\0' < "$SHARDS" | wc -c)
echo "[par] $NSHARD shards ($WORKERS at a time); each runs all groups for its (traj,scenario)"

# ---- worker: one (traj, scenario) shard -------------------------------------
run_shard() {
  local N="$1" SCEN="$2"
  local SS; SS=$(slug "$SCEN")
  local log="$LOGDIR/shard_traj${N}_${SS}.log"
  # Only the traj that owns state-saving passes a real SAVE_TRAJ; every other shard
  # says `none`, which the guard in run_ns_grid_M25.sh accepts as "save nothing".
  local st="none"; [ "$N" = "$SAVE_TRAJ_OWNER" ] && st="$N"
  echo "[par] -> traj$N / $SCEN (threads=$THREADS, save_states=$([ "$st" = none ] && echo no || echo yes))"
  TRAJ="$N" SCENARIOS="$SCEN" SAVE_TRAJ="$st" \
  THREADS="$THREADS" DEVICE="$DEVICE" LOG="$log" \
  OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
    bash paper_experiments/run_ns_grid_M25.sh >>"$log" 2>&1
  echo "[par] <- traj$N / $SCEN done rc=$? $(date +%T)"
}
export -f run_shard slug
export THREADS DEVICE LOGDIR SAVE_TRAJ_OWNER

# xargs -P: keeps exactly $WORKERS shards in flight, starting the next as one exits.
xargs -0 -n1 -P "$WORKERS" bash -c '
  IFS=$'"'"'\t'"'"' read -r n scen <<< "$0"; run_shard "$n" "$scen"
' < "$SHARDS"

echo "[par] ALL SHARDS DONE $(date)"
echo "[par] cells now present: $(ls $ROOT/metrics/*M25__* 2>/dev/null | wc -l) / 60"
echo "[par] failures: $(grep -h 'FAIL' $LOGDIR/*.log 2>/dev/null | wc -l)"
