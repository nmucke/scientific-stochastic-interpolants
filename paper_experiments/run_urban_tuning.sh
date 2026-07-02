#!/bin/bash
# =============================================================================
# Urban (uDALES) HYPERPARAMETER TUNING sweeps  (2026-07-02)
#
# One driver script for tuning EVERY baseline that has a hyperparameter, so the
# urban `<method>.yaml` per-cell tables (currently `default`-only for urban) can
# be filled the same way the navier_stokes columns were. Each method gets its own
# small grid; the search ranges are INFORMED BY THE NS TUNED TABLES (see each
# method's configs/method/*.yaml), narrowed to urban's sparse-only scenarios.
#
# WHAT IS TUNED (env-var overrides, top precedence over the YAML tables, so the
# tracked configs are left untouched):
#   flowdas        FlowDAS zeta            -> FLOWDAS_ZETA
#   sda            SDA gamma_sda           -> SDA_GAMMA
#   dflow          D-Flow eta / lambda     -> DFLOW_STEP_SIZE / DFLOW_LAMBDA
#   fig            FIG (k, c)              -> FIG_K / FIG_C
#   surge_flowdas  SURGE+FlowDAS zeta      -> FLOWDAS_ZETA  (SMC layer may shift
#                                             the optimum vs standalone FlowDAS,
#                                             so it is tuned separately)
#   surge_sda      SURGE+SDA gamma_sda     -> SDA_GAMMA     (tuned separately too)
# (Ours SI/DM/FM-SDE have no per-cell likelihood knob -- the covariance mode is a
#  discrete grid axis, not tuned here -- and EnKF/PF are not run for urban.)
#
# SWEEP MATRIX: a SEPARATE sweep per (scenario, M) cell -- exactly the granularity
# of the per-cell tables the results feed. For each cell every grid value is one
# run.py call, so partial progress is salvageable and a cell can be re-run alone.
#   scenarios : sparse 5%, sparse 1.5625%     (urban is sparse-only)
#   steps M   : 50 100 250 500                (the SDE/ODE step axis)
#
# POSTERIOR RUN (per request): urban TEST TRAJECTORY 1 ONLY (+test_index=1), ONE
# seed, ensemble_size=8, num_physical_steps=7 = 5-step seeded history + 2
# assimilation steps (n_assim = 7 - len_field_history(5) = 2). Cheap enough to
# rank configs; re-validate winners at E=64 / full history before the final table.
#
# USAGE
#   bash paper_experiments/run_urban_tuning.sh <method>     # one method
#   bash paper_experiments/run_urban_tuning.sh all          # every method above
#   <method> in: flowdas sda dflow fig surge_flowdas surge_sda
#
#   # preview the run matrix without launching anything:
#   DRY_RUN=1 bash paper_experiments/run_urban_tuning.sh all
#
#   # CPU smoke test (small, fast -- what to run NOW to check the wiring):
#   DEVICE=cpu STEPS=5 SCENARIOS="sparse 5%" SMOKE=1 \
#     bash paper_experiments/run_urban_tuning.sh flowdas
#
# Full sweeps are for the GPU box (DEVICE=cuda, the default) -- do NOT launch them
# on CPU. Launch detached when ready:
#   setsid nohup bash paper_experiments/run_urban_tuning.sh all \
#     >run_urban_tuning.log 2>&1 & disown
#
# ENV OVERRIDES (subset / smoke the sweep): DEVICE, STEPS, SCENARIOS(|-sep), TRAJ,
#   E, NP, REQUIRE_W, DRY_RUN, SMOKE(=1 -> only the first grid value per axis), OUT.
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
PY=.venv/bin/python

# ---- fixed posterior-run settings (per request) -----------------------------
E="${E:-8}"                       # ensemble size
NP="${NP:-7}"                     # num_physical_steps = 5 history + 2 DA steps
TRAJ="${TRAJ:-1}"                 # urban test trajectory 1 ONLY
SEED=0                            # one seed
DEVICE="${DEVICE:-cuda}"          # cuda for the real sweep; DEVICE=cpu to smoke
REQUIRE_W="${REQUIRE_W:-true}"    # udales checkpoints carry model.pth -> true
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"               # 1 -> only the first value of each swept axis
OUT="${OUT:-paper_experiments/results/urban/tuning}"

# ---- sweep matrix (env-overridable) -----------------------------------------
STEPS="${STEPS:-50 100 250 500}"                       # M (SDE/ODE step axis)
if [ -n "${SCENARIOS:-}" ]; then IFS='|' read -r -a SCEN_ARR <<< "$SCENARIOS";
else SCEN_ARR=("sparse 5%" "sparse 1.5625%"); fi       # urban: sparse only

mkdir -p "$OUT"
LOG="$OUT/run_urban_tuning.log"

slug() { echo "$1" | sed -E 's/[^A-Za-z0-9.+-]+/_/g; s/^_+//; s/_+$//'; }
# SMOKE mode keeps only the first element of a grid array (fast wiring check);
# otherwise returns the whole grid unchanged.
smoke_head() { if [ "$SMOKE" = "1" ]; then echo "$1"; else echo "$@"; fi; }

# ---- per-method hyperparameter grids (informed by the NS tuned tables) -------
# FlowDAS zeta: NS sparse tuned to ~0.002-0.004 (rising to ~1-1.75 only at M=500);
# small log grid spanning that range plus a large value for the high-M cells.
FLOWDAS_ZETAS=(0.001 0.005 0.02 0.1 1.0)
# SDA gamma_sda: NS sparse tuned to 1e-4 / 1e-3 (default 1e-2); short log grid.
SDA_GAMMAS=(0.0001 0.001 0.01 0.1)
# D-Flow: eta is the dominant knob on NS (tuned 5e-3; 5e-2 ~2x worse), lambda
# nearly inert. Coordinate sweep: eta grid at base lambda, then lambda at base eta.
DFLOW_ETAS=(0.001 0.005 0.01 0.05)
DFLOW_LAMBDAS=(0.0001 0.001)          # base lambda = first entry (1e-4, the NS tune)
DFLOW_S=1e-3                          # noise scale s: NS default, held fixed
# FIG: k structural (NS sparse best k=3), c the numeric knob (NS sparse c=10).
FIG_KS=(1 3)
FIG_CS=(5 10 20)

# run_cell <outfile> <tag> <extra run.py args / inline env...>
# Emits one run.py invocation; env-var hyperparameter overrides are prefixed by
# the caller (e.g.  FLOWDAS_ZETA=0.01 run_cell ...).
run_cell() {
  local outfile="$1"; shift
  local tag="$1"; shift
  if [ -f "$outfile" ]; then echo "[urban-tune] SKIP (exists) $outfile" | tee -a "$LOG"; return; fi
  echo "[urban-tune] RUN  $tag -> $(basename "$outfile") $(date +%T)" | tee -a "$LOG"
  if [ "$DRY_RUN" = "1" ]; then
    echo "  DRY_RUN: $tag -> $outfile"; return
  fi
  $PY -u paper_experiments/run.py case=urban seeds=[$SEED] \
      +test_index=$TRAJ \
      ensemble_size=$E case.num_physical_steps=$NP num_steps=$M \
      case.require_weights=$REQUIRE_W case.device=$DEVICE \
      likelihood_mode=dps_jacobian_free \
      "+urban_methods=[\"$METHOD_LABEL\"]" "+urban_scenarios=[\"$SCEN\"]" \
      +save_states=false \
      results_file="$outfile" "$@" >> "$LOG" 2>&1 \
    && echo "[urban-tune] OK   $tag $(date +%T)" | tee -a "$LOG" \
    || echo "[urban-tune] FAIL $tag $(date +%T)" | tee -a "$LOG"
}

# sweep_method <method-key> : loops (scenario x M x grid) for one method.
sweep_method() {
  local method="$1"
  local mdir="$OUT/$method"; mkdir -p "$mdir"
  echo "[urban-tune] ===== method=$method (E=$E NP=$NP dev=$DEVICE traj=$TRAJ) $(date +%T) =====" | tee -a "$LOG"

  for SCEN in "${SCEN_ARR[@]}"; do
    local SS; SS=$(slug "$SCEN")
    for M in $STEPS; do
      case "$method" in
        flowdas|surge_flowdas)
          [ "$method" = flowdas ] && METHOD_LABEL="FlowDAS" || METHOD_LABEL="SURGE (FlowDAS)"
          for Z in $(smoke_head "${FLOWDAS_ZETAS[@]}"); do
            FLOWDAS_ZETA="$Z" \
              run_cell "$mdir/${SS}__M${M}__zeta$(slug "$Z").csv" "$method/$SS/M$M/zeta=$Z"
          done ;;
        sda|surge_sda)
          [ "$method" = sda ] && METHOD_LABEL="SDA" || METHOD_LABEL="SURGE (SDA)"
          for G in $(smoke_head "${SDA_GAMMAS[@]}"); do
            SDA_GAMMA="$G" \
              run_cell "$mdir/${SS}__M${M}__gamma$(slug "$G").csv" "$method/$SS/M$M/gamma=$G"
          done ;;
        dflow)
          METHOD_LABEL="D-Flow SGLD"
          local base_lam="${DFLOW_LAMBDAS[0]}"
          # eta sweep at base lambda
          for ETA in $(smoke_head "${DFLOW_ETAS[@]}"); do
            DFLOW_STEP_SIZE="$ETA" DFLOW_LAMBDA="$base_lam" DFLOW_NOISE_SCALE="$DFLOW_S" \
              run_cell "$mdir/${SS}__M${M}__eta$(slug "$ETA")_lam$(slug "$base_lam").csv" \
                       "$method/$SS/M$M/eta=$ETA lam=$base_lam"
          done
          # lambda refinement at base eta (skip the base-lambda dup), unless SMOKE
          if [ "$SMOKE" != "1" ]; then
            local base_eta="${DFLOW_ETAS[1]}"   # 5e-3, the NS-tuned eta
            for LAM in "${DFLOW_LAMBDAS[@]}"; do
              [ "$LAM" = "$base_lam" ] && continue
              DFLOW_STEP_SIZE="$base_eta" DFLOW_LAMBDA="$LAM" DFLOW_NOISE_SCALE="$DFLOW_S" \
                run_cell "$mdir/${SS}__M${M}__eta$(slug "$base_eta")_lam$(slug "$LAM").csv" \
                         "$method/$SS/M$M/eta=$base_eta lam=$LAM"
            done
          fi ;;
        fig)
          METHOD_LABEL="Guided FM (FIG)"
          for K in $(smoke_head "${FIG_KS[@]}"); do
            for C in $(smoke_head "${FIG_CS[@]}"); do
              FIG_K="$K" FIG_C="$C" \
                run_cell "$mdir/${SS}__M${M}__k${K}_c$(slug "$C").csv" "$method/$SS/M$M/k=$K c=$C"
            done
          done ;;
        *) echo "[urban-tune] unknown method '$method'"; exit 2 ;;
      esac
    done
  done
}

# ---- dispatch ---------------------------------------------------------------
ALL_METHODS=(flowdas sda dflow fig surge_flowdas surge_sda)
WHICH="${1:-}"
if [ -z "$WHICH" ]; then
  echo "usage: $0 <method|all>   (methods: ${ALL_METHODS[*]})"; exit 1
fi
echo "[urban-tune] START $(date) | which=$WHICH DRY_RUN=$DRY_RUN SMOKE=$SMOKE dev=$DEVICE" | tee -a "$LOG"
if [ "$WHICH" = all ]; then
  for m in "${ALL_METHODS[@]}"; do sweep_method "$m"; done
else
  sweep_method "$WHICH"
fi
echo "[urban-tune] ALL DONE $(date)" | tee -a "$LOG"
