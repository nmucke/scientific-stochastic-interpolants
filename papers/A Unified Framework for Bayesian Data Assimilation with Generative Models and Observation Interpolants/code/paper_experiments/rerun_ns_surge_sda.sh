#!/bin/bash
# =============================================================================
# Navier--Stokes: RERUN "SURGE (SDA)" after the implementation fix  (2026-07-25)
#
# WHY. Every SURGE (SDA) number in results/navier_stokes/ was produced by the
# buggy implementation and has to be recomputed. Those numbers do NOT live in
# files of their own: the grid scripts write one CSV per GROUP, so SURGE (SDA)
# sits inside ``<scen>__M<M>__traj<N>__baselines.csv`` (and
# ``__baselines_nodflow.csv`` for the M=25 fill) next to FlowDAS, SURGE
# (FlowDAS), SDA, D-Flow SGLD and Guided FM (FIG) -- all of them unaffected and
# expensive to redo (D-Flow SGLD alone is ~71 h/cell). Deleting those files to
# defeat the skip-if-exists resume would therefore throw away ~5 good methods to
# fix 1 bad one.
#
# WHAT THIS DOES INSTEAD (3 phases):
#   1. CLEAN  -- strip_method_rows.py removes every ``SURGE (SDA)`` row from the
#                per-cell metrics/ and per_step/ CSVs (backing each file up once
#                under $BACKUP), and records which cells were affected. Stale
#                states/ ensembles for the method are moved to the backup too.
#   2. RERUN  -- one run.py call per affected cell with ns_methods=["SURGE (SDA)"]
#                ONLY, writing its own ``__surge_sda.csv`` cell files. Aggregation
#                globs every CSV and keys on (method, scenario, metric, E, M,
#                variant), so the method rejoins its group automatically -- no
#                merge step, and no double counting now that the old rows are gone.
#   3. RECLEAN -- the strip is run once more at the end. The main grids are still
#                running and may have written FRESH pre-fix SURGE (SDA) rows into
#                new baselines cells while phase 2 was working; this catches those.
#
# The cell list is DERIVED from the data, not hard-coded: exactly those cells that
# actually contain SURGE (SDA) rows get rerun. So the count is NOT the full grid
# (4 scenarios x M in {25,50,100,250} x traj 10..14 = 80) and is not meant to be --
# 2026-07-25 it was 59 (traj 10..13 only); 2026-07-30 it is 77, the 3 absent cells
# being {32^2->128^2, sparse 5%, sparse 1.5625%} x M25 x traj14, which the M25-nodflow
# grid had not reached yet. Cells the grid has not reached are not invented here; the
# grid will produce them itself, correctly, once the fixed code is in place.
#
# RESUMING AN INTERRUPTED RUN -- pass the cell list explicitly:
#     CELLS_FILE=paper_experiments/results/navier_stokes/.rerun_surge_sda_cells.txt \
#       bash paper_experiments/rerun_ns_surge_sda.sh
#   Do NOT just relaunch. Phase 1 of the first run already stripped the rows, so
#   re-deriving the list now finds nothing and hands phase 2 an EMPTY list -- the
#   method would stay missing from the results entirely. With CELLS_FILE set, phase 1
#   keeps its hands off that file (its own findings go to *_phase1.txt) and off the
#   cells already finished, so the run picks up where it stopped.
#
# RUN IT AGAIN AFTER THE OTHER GRIDS FINISH:
#     CLEAN_ONLY=1 bash paper_experiments/rerun_ns_surge_sda.sh
#   Phase 3 cannot see a cell that finishes after this script exits. The clean pass
#   is idempotent and cheap; ``CLEAN_ONLY=1`` reruns just it (and reports whether
#   any new stale rows appeared -- if it reports 0 rows, everything is consistent).
#   If it DOES report rows, rerun this script in full to refill those cells.
#
# Launch detached:
#   setsid nohup bash paper_experiments/rerun_ns_surge_sda.sh >rerun_surge_sda.log 2>&1 & disown
# Then aggregate as usual: .venv/bin/python paper_experiments/aggregate_ns.py
#
# Env: ROOT, DEVICE (cuda|cpu), E, NP, SAVE_TRAJ, DIV_GUARD, REQUIRE_W,
#      CLEAN_ONLY=1 (phase 1 only), DRY_RUN=1 (print, change nothing),
#      CELLS_FILE (use an explicit cell list: "<scen-slug> <M> <traj>" per line).
#
# NOTE on DEVICE: the main grid holds the GPU; the 2026-07-15 attempt to run two
# cuda jobs at once lost 34/60 cells to OOM. Default here is cuda anyway (SURGE
# (SDA) alone is small), but use DEVICE=cpu if you see OOM in the log.
# =============================================================================
set -u
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/common/paths.sh" || exit 1
cd "$PE_REPO_ROOT"
PY="${PY:-$PE_PYTHON}"

ROOT="${ROOT:-$PE_SCRIPT_DIR/results/navier_stokes}"
MET=$ROOT/metrics
PS=$ROOT/per_step
REF=$ROOT/reference
BACKUP="${BACKUP:-$ROOT/backup_pre_surge_sda_fix}"

# The method being redone, its group (= output-file suffix) and its states slug.
METHOD="${METHOD:-SURGE (SDA)}"
METHODS_JSON="${METHODS_JSON:-[\"SURGE (SDA)\"]}"
GROUP="${GROUP:-surge_sda}"
METHOD_SLUG=$(echo "$METHOD" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//')  # SURGE_SDA

# Per-cell settings -- IDENTICAL to the baselines cells in run_ns_grid.sh and
# run_ns_grid_M25*.sh, so the rerun rows are comparable to their neighbours.
E="${E:-64}"
NP="${NP:-20}"
DEVICE="${DEVICE:-cuda}"
SAVE_TRAJ="${SAVE_TRAJ:-11}"
DIV_GUARD="${DIV_GUARD:-10.0}"
REQUIRE_W="${REQUIRE_W:-true}"
CLEAN_ONLY="${CLEAN_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

LOG="$ROOT/rerun_surge_sda.log"
# Phase 2 reads $CELLS. By default that is the list phase 1 derives from the data.
# With CELLS_FILE set it is the caller's own list, and phase 1 must NOT overwrite it
# -- its own findings go to a side file instead. This is what makes a RESUME work:
# after a first run has stripped the rows, re-deriving finds nothing (the rows are
# already gone) and would hand phase 2 an empty list.
CELLS="${CELLS_FILE:-$ROOT/.rerun_surge_sda_cells.txt}"
if [ -n "${CELLS_FILE:-}" ]; then
  STRIP_CELLS_OUT="${CELLS%.txt}_phase1.txt"
  [ -s "$CELLS" ] || { echo "[surge] FATAL: CELLS_FILE='$CELLS' is missing or empty" >&2; exit 1; }
else
  STRIP_CELLS_OUT="$CELLS"
fi

# --- guards ------------------------------------------------------------------
# "results copy/" is the user's untracked safety net -- never touch it.
case "$(readlink -f "$ROOT")" in
  *"results copy"*) echo "[surge] FATAL: ROOT points inside 'results copy' -- refusing." >&2; exit 1;;
esac
[ -d "$MET" ] || { echo "[surge] FATAL: no metrics dir at $MET" >&2; exit 1; }

echo "[surge] START $(date) | method='$METHOD' group=$GROUP | E=$E NP=$NP dev=$DEVICE save_states=traj$SAVE_TRAJ" | tee -a "$LOG"

# Concurrent grid jobs are fine (phase 3 re-cleans), but say so out loud: any cell
# they finish AFTER this script exits keeps its pre-fix rows until CLEAN_ONLY=1 runs.
LIVE=$(pgrep -af "run\.py .*ns_methods=.*SURGE \(SDA\)" | wc -l)
if [ "$LIVE" -gt 0 ]; then
  echo "[surge] NOTE: $LIVE live run.py process(es) still include '$METHOD'." | tee -a "$LOG"
  echo "[surge]       Re-run with CLEAN_ONLY=1 once they finish (see header)." | tee -a "$LOG"
fi

# --- phase 1: clean ----------------------------------------------------------
STRIP_ARGS=(--root "$ROOT" --method "$METHOD" --exclude-group "$GROUP"
            --backup "$BACKUP" --cells-out "$STRIP_CELLS_OUT")
[ "$DRY_RUN" = "1" ] && STRIP_ARGS+=(--dry-run)
echo "[surge] --- phase 1: strip '$METHOD' rows from group CSVs ---" | tee -a "$LOG"
"$PY" "$PE_SCRIPT_DIR/strip_method_rows.py" "${STRIP_ARGS[@]}" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[surge] FATAL: strip failed" >&2; exit 1; }

# The method's OWN artefacts from a PRE-FIX rerun -- cell CSVs and raw ensembles.
# Clear them (move, don't delete: the pre-fix output stays inspectable), or the
# grid-style skip-if-exists below would keep the stale numbers alive.
#
# NOT on a resume (CELLS_FILE set). There everything bearing this method's name is
# post-fix work that an interrupted run already finished, and skip-if-exists is
# exactly how we avoid redoing it. Clearing it would restart from cell 1 on every
# interruption, and would orphan the traj$SAVE_TRAJ ensembles of cells that then get
# skipped (the CSV survives, so the .npz is never rewritten).
if [ -n "${CELLS_FILE:-}" ]; then
  KEPT=$(ls -1 "$MET"/*__${GROUP}.csv 2>/dev/null | wc -l)
  echo "[surge] RESUME (CELLS_FILE set): keeping $KEPT finished $GROUP cell file(s) + their states; those cells are skipped below." | tee -a "$LOG"
else
  for f in "$MET"/*__${GROUP}.csv "$PS"/*__${GROUP}.csv; do
    [ -e "$f" ] || continue
    if [ "$DRY_RUN" = "1" ]; then echo "[surge] would delete $f" | tee -a "$LOG"; else
      mkdir -p "$BACKUP/$(basename "$(dirname "$f")")"
      mv "$f" "$BACKUP/$(basename "$(dirname "$f")")/" && echo "[surge] cleared $(basename "$f")" | tee -a "$LOG"
    fi
  done
  for f in "$ROOT"/states/traj*/*__${METHOD_SLUG}__*.npz; do
    [ -e "$f" ] || continue
    if [ "$DRY_RUN" = "1" ]; then echo "[surge] would move states $(basename "$f")" | tee -a "$LOG"; else
      dest="$BACKUP/states/$(basename "$(dirname "$f")")"
      mkdir -p "$dest"; mv "$f" "$dest/" && echo "[surge] cleared states $(basename "$f")" | tee -a "$LOG"
    fi
  done
fi

if [ "$CLEAN_ONLY" = "1" ]; then
  echo "[surge] CLEAN_ONLY=1 -- stopping after the clean phase. $(date)" | tee -a "$LOG"
  exit 0
fi
[ "$DRY_RUN" = "1" ] && { echo "[surge] DRY_RUN=1 -- stopping before the rerun (cells listed above)." | tee -a "$LOG"; exit 0; }

# --- phase 2: rerun ----------------------------------------------------------
# scenario slug (as used in filenames) -> the canonical label run.py expects.
unslug() {
  case "$1" in
    16_2_128_2)    echo "16^2->128^2";;
    32_2_128_2)    echo "32^2->128^2";;
    sparse_5)      echo "sparse 5%";;
    sparse_1_5625) echo "sparse 1.5625%";;
    *)             echo "";;
  esac
}

NCELLS=$(wc -l < "$CELLS" 2>/dev/null || echo 0)
echo "[surge] --- phase 2: rerun $NCELLS cell(s) ---" | tee -a "$LOG"
i=0
while read -r SS M N; do
  [ -n "${SS:-}" ] || continue
  i=$((i + 1))
  SCEN=$(unslug "$SS")
  if [ -z "$SCEN" ]; then
    echo "[surge] SKIP unknown scenario slug '$SS' (cell $SS/M$M/traj$N)" | tee -a "$LOG"; continue
  fi
  if [ "$M" = "-" ]; then
    echo "[surge] SKIP non-M cell $SS/traj$N (classical group has no '$METHOD')" | tee -a "$LOG"; continue
  fi

  OUT="$MET/${SS}__M${M}__traj${N}__${GROUP}.csv"
  PSF="$PS/${SS}__M${M}__traj${N}__${GROUP}.csv"
  if [ -f "$OUT" ]; then echo "[surge] SKIP (exists) $(basename "$OUT")" | tee -a "$LOG"; continue; fi

  # states for SAVE_TRAJ only, exactly as the grid scripts do.
  if [ "$N" = "$SAVE_TRAJ" ]; then SAVE_STATES=true; STATES_ROOT="$ROOT/states/traj${SAVE_TRAJ}";
  else SAVE_STATES=false; STATES_ROOT="$ROOT/states/_unused"; fi
  KLREF="$REF/traj${N}/gt"

  echo "[surge] RUN  ($i/$NCELLS) $SS/M$M/traj$N $(date +%T)" | tee -a "$LOG"
  "$PY" -u "$PE_SCRIPT_DIR/run.py" case=navier_stokes seeds=[0] \
      ensemble_size=$E case.num_physical_steps=$NP \
      case.require_weights=$REQUIRE_W case.device=$DEVICE \
      +test_index=$N \
      +save_per_step=true "+per_step_file=$PSF" \
      +divergence_rmse_threshold=$DIV_GUARD \
      results_file="$OUT" \
      "+ns_methods=$METHODS_JSON" "+ns_scenarios=[\"$SCEN\"]" num_steps=$M \
      likelihood_mode=dps_jacobian_free "+kl_reference_states=$KLREF" \
      +save_states=$SAVE_STATES "+states_root=$STATES_ROOT" >> "$LOG" 2>&1 \
    && echo "[surge] OK   ($i/$NCELLS) $(basename "$OUT") $(date +%T)" | tee -a "$LOG" \
    || echo "[surge] FAIL ($i/$NCELLS) $(basename "$OUT") $(date +%T)" | tee -a "$LOG"
done < "$CELLS"

# --- phase 3: re-clean -------------------------------------------------------
# Concurrent grid jobs may have written fresh pre-fix rows during phase 2.
echo "[surge] --- phase 3: re-strip (catch rows written during the rerun) ---" | tee -a "$LOG"
"$PY" "$PE_SCRIPT_DIR/strip_method_rows.py" --root "$ROOT" --method "$METHOD" \
    --exclude-group "$GROUP" --backup "$BACKUP" \
    --cells-out "${CELLS%.txt}_phase3.txt" 2>&1 | tee -a "$LOG"
echo "[surge] NOTE: any cell listed above lost its '$METHOD' rows and was NOT rerun" | tee -a "$LOG"
echo "[surge]       -- rerun this script once the other grids finish to refill them." | tee -a "$LOG"

echo "[surge] ALL DONE $(date)" | tee -a "$LOG"
