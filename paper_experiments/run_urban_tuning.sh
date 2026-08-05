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
#   steps M   : 25 50 100 250                 (the SDE/ODE step axis -- MUST match
#                                              run_urban_grid.sh STEPS, restructured
#                                              2026-07-25 from {50,100,250,500})
#
# POSTERIOR RUN (per request): ONE urban test trajectory (+test_index=$TRAJ), ONE
# seed, ensemble_size=8, num_physical_steps=6 = 5-step seeded history + 1
# assimilation step (n_assim = 6 - len_field_history(5) = 1). Cheap enough to
# rank configs; re-validate winners at E=64 / full history before the final table.
# NOTE the urban test set is `files (170, 178)` -> 9 trajectories, so test_index
# must be in 0..8; the paper grid evaluates 1..5, which leaves 0 / 6 / 7 / 8 free
# to tune on without tuning on an evaluation trajectory.
#
# USAGE
#   bash paper_experiments/run_urban_tuning.sh <method>     # one method
#   bash paper_experiments/run_urban_tuning.sh baselines    # the 4 non-SURGE ones
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
NP="${NP:-6}"                     # num_physical_steps = 5 history + 1 DA step
TRAJ="${TRAJ:-6}"                 # ONE urban test trajectory; must be 0..8 (9 test
                                  # sims) and ideally OUTSIDE the paper grid's 1..5
SEED=0                            # one seed
DEVICE="${DEVICE:-cuda}"          # cuda for the real sweep; DEVICE=cpu to smoke
REQUIRE_W="${REQUIRE_W:-true}"    # udales checkpoints carry model.pth -> true
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"               # 1 -> only the first value of each swept axis
OUT="${OUT:-paper_experiments/results/urban/tuning}"

# ---- sweep matrix (env-overridable) -----------------------------------------
STEPS="${STEPS:-25 50 100 250}"                        # M (SDE/ODE step axis)
if [ -n "${SCENARIOS:-}" ]; then IFS='|' read -r -a SCEN_ARR <<< "$SCENARIOS";
else SCEN_ARR=("sparse 5%" "sparse 1.5625%"); fi       # urban: sparse only

mkdir -p "$OUT"
LOG="$OUT/run_urban_tuning.log"

slug() { echo "$1" | sed -E 's/[^A-Za-z0-9.+-]+/_/g; s/^_+//; s/_+$//'; }
# SMOKE mode keeps only the first element of a grid array (fast wiring check);
# otherwise returns the whole grid unchanged.
smoke_head() { if [ "$SMOKE" = "1" ]; then echo "$1"; else echo "$@"; fi; }

# ---- per-method hyperparameter grids -----------------------------------------
# RECENTRED 2026-07-30 ON THE NS SPARSE TUNED CELLS. The urban `<method>.yaml`
# blocks currently hold those NS sparse rows COPIED VERBATIM, so the NS optimum is
# also urban's incumbent -- these grids exist to confirm or move it, which means
# they must BRACKET it tightly rather than span decades. Every grid below contains
# the incumbent exactly and walks ~one half-decade either side of it.
#
# The NS sparse tuned values over the urban M ladder {25,50,100,250} are:
#   FlowDAS zeta   sparse_5  0.002 (flat) | sparse_1p5  0.004,0.004,0.004,0.002
#   SDA gamma_sda  sparse_5  1e-3,1e-3,1e-3,1e-4 | sparse_1p5  1e-4 (flat)
#   D-Flow         eta 5e-3 (flat in scenario AND M), s 1e-3, lambda 1e-4
#   FIG            k=3, c=10 (both flat across scenario and M)
# (The old grids were built around the M=500 column, which the urban ladder no
#  longer has -- that is why zeta went up to 1.0 and gamma up to 0.1.)

# FlowDAS zeta: WIDENED DOWNWARD 2026-07-30 after the first 12 cells of this very
# sweep came back MONOTONE in zeta with the winner sitting on the grid's lower edge:
#   sparse_5   M=25 / M=50 / M=100  rmse
#     0.0005   0.0582  0.0554  0.0544   <- best, and at the edge
#     0.001    0.0781  0.0914  0.1024
#     0.002    0.3296  0.3919  0.4079   <- the NS sparse_5 incumbent, ~7x worse
#     0.004    0.5016  0.6036
#     0.02     1.0029  1.1621           <- diverged
# So the urban optimum is at or below 5e-4 and the NS-copied cell in
# configs/method/flowdas.yaml is badly wrong for urban, not marginally mistuned --
# the ~4x larger residual (5% of 4x128x128 dofs vs NS's 5% of 1x128x128) bites hard.
# The old top end (0.004, 0.02) is dropped as conclusively diverging; 0.002 stays
# purely as the incumbent reference. zeta=0 is the NO-GUIDANCE control: with rmse
# monotone toward zero it is the only way to tell whether FlowDAS guidance helps on
# urban at all, or whether the best cell is just the prior with guidance switched off.
FLOWDAS_ZETAS=(0 0.00001 0.00005 0.0001 0.0002 0.0005 0.001 0.002)
# SDA gamma_sda: incumbent 1e-4 / 1e-3, both in the grid. 1e-5 is the lower anchor,
# 1e-2 the released-code default (NS found it clearly worse on sparse).
SDA_GAMMAS=(0.00001 0.0001 0.0005 0.001 0.01)
# D-Flow: eta is the dominant knob on NS (tuned 5e-3; the paper's 5e-2 was ~2x
# worse there, so it is dropped), lambda nearly inert. Coordinate sweep: eta grid
# at base lambda, then lambda at base eta.
# NB: base_eta below is DFLOW_ETAS[1] -- keep 0.005 (the NS tune) in slot 1.
DFLOW_ETAS=(0.002 0.005 0.01 0.02)
DFLOW_LAMBDAS=(0.0001 0.001)          # base lambda = first entry (1e-4, the NS tune)
DFLOW_S=1e-3                          # noise scale s: NS default, held fixed
# FIG: k structural (NS sparse best k=3 > k=1 > k=2), c the numeric knob (NS sparse
# c=10 flat in M). k=1 kept as the cheap structural anchor / FIG's own SR setting.
FIG_KS=(1 3)
# c=2 added 2026-07-30 as a low anchor -- cheap insurance after FlowDAS turned out to
# want a much smaller guidance strength on urban than on NS. FIG's corrector
# normalises by the GLOBAL residual norm, so it should be far less sensitive to
# urban's 4x dof count than FlowDAS's raw score is; this just confirms that.
FIG_CS=(2 5 10 20)

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
# `baselines` = the four bare baselines, i.e. every method WITHOUT its SURGE layer.
# For FlowDAS/SDA that means the standalone method: zeta / gamma_sda are properties
# of the guidance term itself, so tuning them bare and reusing the winner under
# SURGE is the cheap path (it halves the sweep). Run surge_flowdas / surge_sda
# separately later if the SMC layer turns out to shift the optimum.
# Ordered cheapest-first (measured at M=25, E=8, 1 DA step: fig k=1 ~3s, flowdas
# ~7s, dflow ~73s of sampling) so the dear one cannot starve the other three.
BASELINE_METHODS=(flowdas sda fig dflow)
WHICH="${1:-}"
if [ -z "$WHICH" ]; then
  echo "usage: $0 <method|baselines|all>   (methods: ${ALL_METHODS[*]})"; exit 1
fi
echo "[urban-tune] START $(date) | which=$WHICH DRY_RUN=$DRY_RUN SMOKE=$SMOKE dev=$DEVICE" | tee -a "$LOG"
if [ "$WHICH" = all ]; then
  for m in "${ALL_METHODS[@]}"; do sweep_method "$m"; done
elif [ "$WHICH" = baselines ]; then
  for m in "${BASELINE_METHODS[@]}"; do sweep_method "$m"; done
else
  sweep_method "$WHICH"
fi
echo "[urban-tune] ALL DONE $(date)" | tee -a "$LOG"
