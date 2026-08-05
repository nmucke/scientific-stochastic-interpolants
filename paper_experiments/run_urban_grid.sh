#!/bin/bash
# =============================================================================
# Urban (uDALES) REDUCED-GRID master run  (2026-07-01; restructured 2026-07-13)
#
# Case 3 reduced grid, writing into results/urban/ (see results/README.md).
# Same structure and saving policy as run_ns_grid.sh; the differences are
# intrinsic to the case, not stylistic:
#   * GENERATIVE-ONLY -- urban has no in-repo CFD solver, so there is NO
#     `classical` group (no EnKF/PF) and NO KL reference (no ground-truth
#     posterior), hence no kl_reference_states argument.
#   * SPARSE-ONLY scenarios (no super-resolution observation operator).
#   * SURGE-ONLY baselines -- see METHOD GROUPS below.
#   * NO energy-spectrum metric: the radially-averaged KE spectrum presumes a
#     periodic fluid box, which a building array is not.
#
# GRID
#   trajectories : test_index 1..5   (one seed each; seeds=[0])
#   scenarios    : sparse 1.5625% + sparse 0.78125%  (2026-08-04; 5% off the lineup)
#   steps M      : 50 ONLY            (2026-08-03 -- M=25,100,250 deferred; see the
#                                     STEPS= line below for why and how to add them)
#   Ours modes   : jacfree (dps_jacobian_free) + shared (inflated_shared)
#   E=64, num_physical_steps=55 (5 history + 50 DA steps -- RAISED 2026-08-03)
#
# METHOD GROUPS (each is one run.py call per (traj, scenario, M)):
#   ours_jacfree     : Ours (SI-SDE/DM-SDE/FM-ODE), likelihood_mode=dps_jacobian_free
#   ours_shared_k<K> : Ours (SI-SDE/DM-SDE/FM-ODE), likelihood_mode=inflated_shared
#                      with Jacobian refresh cadence k=K. One group per cadence
#                      (ours_shared_k1 / _k5 / _k10) -- pick the cost/accuracy point
#                      by picking the group; each writes its own files so cadences
#                      can coexist in one results tree. lambda (jacobian_damping)
#                      stays PER-SCENARIO in the method YAMLs.
#   baselines        : SURGE (FlowDAS), SURGE (SDA), Guided FM (FIG)
#   dflow            : D-Flow SGLD alone -- same lineup, own group, because it is
#                      by far the dearest baseline (on NS it ran ~4x the other
#                      baselines COMBINED), so it is worth being able to pause,
#                      re-run or move it to another box on its own. It IS in the
#                      default GRPS, so coverage is unchanged; drop it from GRPS
#                      to run everything else first.
#
# SURGE-ONLY BASELINES (2026-07-25, user request). Where a baseline has a SURGE
# variant, urban runs the SURGE variant ONLY -- so bare `FlowDAS` and `SDA` are
# NOT in `baselines`; `SURGE (FlowDAS)` and `SURGE (SDA)` stand in for them.
# They remain wired in the driver, so run_urban_tuning.sh can still name them.
#
# SAVING
#   save_states  : $SAVE_TRAJ ONLY (default traj1), ALL groups incl. BOTH Ours
#                  modes (variant is in the filename so jacfree/shared never collide).
#   save_per_step: ALWAYS -> per-step metric curves for every trajectory.
#   timings      : seconds + NFE are on every metric row and every per-step row.
#
# Launch: setsid nohup bash paper_experiments/run_urban_grid.sh >run_urban_grid.log 2>&1 & disown
# Track : .venv/bin/python paper_experiments/status.py --case urban
# Env overrides: TRAJ, SAVE_TRAJ, SCENARIOS(|-sep), STEPS, GRPS, E, NP, DEVICE, REQUIRE_W.
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
PY=.venv/bin/python
ROOT="${ROOT:-paper_experiments/results/urban}"   # override for smoke tests
MET=$ROOT/metrics
PS=$ROOT/per_step
mkdir -p "$MET" "$PS" "$ROOT/states"

# ---- knobs (env-overridable) ------------------------------------------------
TRAJ="${TRAJ:-1 2 3 4 5}"
# Which trajectory gets its raw ensembles written out (states are big, so only one).
# MUST be a member of TRAJ or nothing is saved (guarded below).
SAVE_TRAJ="${SAVE_TRAJ:-1}"
# M LADDER TRIMMED TO {25,50,100} (2026-08-02, user decision). M=250 is dropped for
# now, not abandoned -- add it later with a single resumable pass:
#     STEPS=250 bash paper_experiments/run_urban_grid.sh
# The grid is skip-if-exists and every cell writes its own file, so that fills only
# the M=250 column and touches nothing already produced.
#
# WHY: M=250 is by far the dearest rung and it dominated the budget. Cost here is
# driven by the shared-Jacobian group, whose per-DA-step work scales with the
# refresh count M/k (k=10) -- so M=250 alone is 25 refreshes against 3+5+10=18 for
# the whole rest of the ladder COMBINED. Measured per-DA-step (uncontended, urban,
# 3 methods per cell):
#     sparse 5%        M=25 11.0 min | M=50 18.8 | M=100 36.7 | M=250 91.7
#     sparse 1.5625%   M=25  3.8 min | M=50  6.4 | M=100 12.6 | M=250 31.4
# Over 15 DA steps x 5 trajectories x 2 scenarios that puts the run at roughly
# 7 days for {25,50,100} against ~17 days with M=250 included -- i.e. the last rung
# was ~60% of the total. (D-Flow is separately excluded from GRPS; see below.)
#
# NOTE make_urban_figures.py URBAN_STEPS and status.py URBAN_STEPS still list all
# four rungs, so the M=250 column will simply read as missing in the figures and
# the coverage report until it is run -- visibly absent rather than silently
# dropped, which is the intended behaviour.
# 2026-08-03: trimmed AGAIN, to {25,50}. M=100 joins M=250 in the deferred pile --
# both are one resumable pass away (`STEPS=100 bash ...`, `STEPS=250 bash ...`), and
# both fill only their own column. This second trim pays for the DA-step change
# below: 30 assimilation steps is exactly 2x the work per cell, so dropping M=100
# (which by the refresh-count argument above is ~53% of the {25,50,100} budget)
# roughly cancels it.
# 2026-08-03 (later the same day): narrowed again to M=50 ONLY, alongside the move to
# 50 DA steps below. M=25 joins {100,250} in the deferred pile; each is one resumable
# pass (`STEPS=25 bash ...`) filling only its own column.
STEPS="${STEPS:-50}"
E="${E:-64}"
# num_physical_steps = len_field_history(5) + n_assim, so 55 -> 50 DA steps.
# RAISED FROM 20 (15 DA steps) on 2026-08-03, via 35, to 55. Cost is exactly linear
# in n_assim (every step is one full sampler pass), so this is 3.33x the original
# per-cell cost -- which is why STEPS was narrowed to a single rung in the same edit.
#
# HEADROOM: the urban test trajectories are 200 steps long (raw files are 250, with
# starting_time=50 discarding spin-up; measured sample['x'].shape = (4,128,128,200)),
# and n_assim = NP - len_field_history(5), so the ceiling is NP=200 / 195 DA steps.
# 50 uses a quarter of that. Do NOT exceed 200: prepare_truth_and_obs slices with
# `traj[..., :num_physical_steps]`, which silently truncates, then loops
# `range(num_physical_steps)` over it -- so an over-large NP dies with an IndexError
# deep in the observation loop after the model has loaded, not with a clear message.
#
# NOTE this makes the urban horizon DIFFER FROM NS, which stays at num_physical_steps
# =20 / 15 DA steps (configs/case/navier_stokes.yaml, run_ns_grid.sh). The two cases
# were deliberately matched before, so per-step curves and any "at step k" statement
# are no longer directly comparable across cases -- and results/urban_dasteps_15/
# (the 15-step run, backed up 2026-08-03) is NOT comparable to what this produces.
# Aggregation keys on (method, scenario, metric, E, M, variant) and does NOT carry
# n_assim, so mixing the two trees in one aggregate would silently average different
# horizons. Keep them separate.
NP="${NP:-55}"                       # num_physical_steps (5 history + 50 DA)
DEVICE="${DEVICE:-cuda}"
REQUIRE_W="${REQUIRE_W:-true}"       # hard-fail if no trained weights
# Divergence safety net: abort a cell whose ensemble RMSE exceeds this (well above
# any healthy value) and NaN-pad the rest, rather than crashing the whole run.
DIV_GUARD="${DIV_GUARD:-10.0}"

# FlowDAS Monte-Carlo chunking -- REQUIRED ON URBAN, and the reason the `baselines`
# group OOM-ed on 2026-08-02 (2 grid cells + 2 timing cells, all with
#   torch.OutOfMemoryError ... this process has 22.65 GiB in use
# out of a 23.59 GiB card). FlowdasGaussianLikelihood.score materialises J
# one-step predictions as a [J,B,C,H,W] block and pushes J*B samples through the
# observation operator in ONE batch. Urban: J=25, B=E=64, C=4, H=W=128 -> a 0.39 GiB
# block and a 1600-sample operator batch. NS survives untouched only because its
# C=1 makes every one of those tensors 4x smaller.
#
# mc_chunk=5 splits J=25 into 5 passes of 5 members: peak transients go from ~4*J*S
# to J*S + O(k*S), and the operator sees 320 samples instead of 1600. Same number of
# UNet evaluations (2 forwards + 1 backward) -- the extra cost is one cheap chunked
# pass through the observation operator.
#
# EXPORTED HERE RATHER THAN PUT IN THE YAML on purpose: chunking regroups a float sum
# over J, and float addition is not associative, so a genuine split moves the score by
# <= ~1.4 ULP. Setting it only for urban leaves the already-produced NS FlowDAS /
# SURGE-FlowDAS numbers reproducible BIT-FOR-BIT. 5 divides 25 evenly; any value >= J
# (or unset) is a no-op and stays bitwise identical to the old code.
export FLOWDAS_MC_CHUNK="${FLOWDAS_MC_CHUNK:-5}"
#
# NOTE lambda (jacobian_damping) is NOT set here. It is PER-SCENARIO and lives in
# the method YAMLs (configs/method/{si_sde,dm_sde,fm_ode}.yaml) as [case][scenario][M]
# tables. Urban's cells were NEVER SWEPT: they currently carry the NS *sparse*
# optimum (0.95), on the argument that urban's scenarios are sparse too -- but the
# NS divergence cliff sits at 1.0 on sparse, so 0.95 is one grid point away from it
# and urban's own cliff has never been located. If urban shared cells NaN out, drop
# the urban rows of those tables to 0.9. To sweep a value on purpose, override on
# the CLI for one run:  +jacobian_damping=0.9
#
# The BASELINE hyperparameters are in the same position (2026-07-25): every
# configs/method/*.yaml now carries an `urban:` block, but each one is the NS
# *sparse* row COPIED VERBATIM, not an urban sweep -- results/urban/tuning/ is still
# empty. That is a much better starting point than the old `default` fallback (it
# moves FlowDAS zeta 1.0 -> 0.002, SDA gamma 0.01 -> 1e-3, FIG k 1 -> 3), but the
# cells are unvalidated on urban. Re-sweep with run_urban_tuning.sh before the
# headline table; env vars (FLOWDAS_ZETA, SDA_GAMMA, DFLOW_*, FIG_K/FIG_C) override
# the tables without editing them.

# Scenarios as a bash array (canonical labels). Urban is sparse-only.
if [ -n "${SCENARIOS:-}" ]; then IFS='|' read -r -a SCEN_ARR <<< "$SCENARIOS";
else SCEN_ARR=("sparse 0.78125%"); fi
# else SCEN_ARR=("sparse 1.5625%" "sparse 0.78125%"); fi
# THE URBAN LINEUP IS NOW THE TWO SPA012//RSEST REGIMES (2026-08-04): 1.5625% (1/64) and
# 0.78125% (1/128). `sparse 5%` is OFF the lineup -- it stays fully wired (scenario
# config, SCENARIO_CONFIG_NAME entry, and hyperparameter rows in every method YAML are
# all intact), so it runs on demand with one resumable pass:
#     SCENARIOS="sparse 5%" bash paper_experiments/run_urban_grid.sh
# Cells are per-scenario files, so that fills only the 5% rows and touches nothing.
#
# sparse_0p78's hyperparameters are the sparse_1p5 rows COPIED VERBATIM in every
# method YAML (zeta 5e-4, gamma 5e-4, FIG k=3/c=160, D-Flow eta 5e-3/s 1e-3/lambda
# 1e-3, jacobian_damping 0.95). That transfer is better supported than the earlier
# NS->urban one: each of those knobs measured FLAT between sparse_5 and sparse_1p5,
# i.e. across a 3.2x change in observation count, so extending it across a further
# 2x reduction interpolates inside a demonstrated plateau.
#
# This is also the CHEAPER half by a wide margin, which is worth knowing when
# budgeting the deferred pass: shared-mode cost scales with the OBSERVATION COUNT
# (one JVP per observation per Jacobian refresh), and sparse 1.5625% sees
# N_y ~ 650 against sparse 5%'s ~2081 -- ~3.2x. Measured per-DA-step, 3 methods:
#     sparse 1.5625%   M=25  3.8 min | M=50  6.4
#     sparse 5%        M=25 11.0 min | M=50 18.8
# So this run is ~1/4 of the two-scenario budget, and the deferred 5% pass will
# cost roughly 3x what this one does.

# SHARED-MODE GROUPS -- one per Jacobian refresh cadence k. Pick the cadence by
# picking the group; each writes its own files (variant is stamped in), so several
# cadences can coexist in one results tree and be compared directly.
#   ours_shared_k1   k=1  exact per-step Jacobian. BEST rmse on every NS scenario.
#   ours_shared_k5   k=5  ~3-4x cheaper.
#   ours_shared_k10  k=10 ~5-7x cheaper.
# Measured on NS (SI-SDE), the lag is nearly free on the super-res cells but costs
# +15% (sparse 5%) to +49% (sparse 1.5625%) rmse at k=5 -- and urban is SPARSE-ONLY,
# i.e. it sits on the expensive side of that trade. lambda was tuned at k=1, so a
# lagged group is not re-tuned.
#
# k is read straight OUT OF the group name (ours_shared_k<k>), so any cadence works
# with no second list to keep in sync -- e.g. GRPS="... ours_shared_k20 ..." just runs.
# GRPS="${GRPS:-ours_jacfree ours_shared_k1 ours_shared_k5 baselines dflow}"
GRPS="${GRPS:-ours_jacfree ours_shared_k10 baselines dflow}"

# Cadences requested this run: every ours_shared_k<k> in GRPS, k parsed from the name.
SHARED_KS=$(echo "$GRPS" | tr ' ' '\n' | sed -nE 's/^ours_shared_k([0-9]+)$/\1/p')
# Catch a typo'd shared group (e.g. ours_shared_k5x, ours_shared) instead of silently
# skipping it -- a group that matches nothing would otherwise just never run.
for g in $GRPS; do
  case "$g" in
    ours_shared_k[0-9]*) echo "$g" | grep -qE '^ours_shared_k[0-9]+$' || {
        echo "[urbangrid] FATAL: malformed shared group '$g' (expected ours_shared_k<int>)" >&2; exit 1; };;
    ours_shared) echo "[urbangrid] FATAL: group 'ours_shared' is gone -- use ours_shared_k1 (or _k5/_k10)." >&2; exit 1;;
  esac
done

OURS='["Ours (SI-SDE)","Ours (DM-SDE)","Ours (FM-ODE)"]'
# SURGE-only: bare "FlowDAS" / "SDA" are deliberately absent (see the header).
BASELINES='["SURGE (FlowDAS)","SURGE (SDA)","Guided FM (FIG)","D-Flow SGLD"]'
# DFLOW='["D-Flow SGLD"]'
# -----------------------------------------------------------------------------

slug() { echo "$1" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//'; }
has_group() { echo " $GRPS " | grep -q " $1 "; }

LOG="$ROOT/run_urban_grid.log"
# Fail loudly rather than silently saving no states at all.
if ! echo " $TRAJ " | grep -q " $SAVE_TRAJ "; then
  echo "[urbangrid] FATAL: SAVE_TRAJ=$SAVE_TRAJ is not in TRAJ=[$TRAJ] -- no states would be saved." >&2
  exit 1
fi
echo "[urbangrid] START $(date) | E=$E NP=$NP dev=$DEVICE | traj=[$TRAJ] steps=[$STEPS] groups=[$GRPS] save_states=traj$SAVE_TRAJ" | tee -a "$LOG"

# run_cell <outfile> <group-tag> <extra run.py args...>
run_cell() {
  local outfile="$1"; shift
  local tag="$1"; shift
  if [ -f "$outfile" ]; then echo "[urbangrid] SKIP (exists) $outfile" | tee -a "$LOG"; return; fi
  local psfile="$PS/$(basename "${outfile%.csv}").csv"
  echo "[urbangrid] RUN  $tag -> $(basename "$outfile") $(date +%T)" | tee -a "$LOG"
  # Trajectory N -> test sample N (test_sample_indices=[1..5]); WITHOUT this every
  # traj would rerun the default sample and the trajectory aggregation is a no-op.
  $PY -u paper_experiments/run.py case=urban seeds=[0] \
      ensemble_size=$E case.num_physical_steps=$NP \
      case.require_weights=$REQUIRE_W case.device=$DEVICE \
      +test_index=$N \
      +save_per_step=true "+per_step_file=$psfile" \
      +divergence_rmse_threshold=$DIV_GUARD \
      results_file="$outfile" "$@" >> "$LOG" 2>&1 \
    && echo "[urbangrid] OK   $tag $(basename "$outfile") $(date +%T)" | tee -a "$LOG" \
    || echo "[urbangrid] FAIL $tag $(basename "$outfile") $(date +%T)" | tee -a "$LOG"
}

for N in $TRAJ; do
  # states for SAVE_TRAJ only; ALL groups + both Ours modes.
  if [ "$N" = "$SAVE_TRAJ" ]; then SAVE_STATES=true; STATES_ROOT="$ROOT/states/traj${SAVE_TRAJ}"; else SAVE_STATES=false; STATES_ROOT="$ROOT/states/_unused"; fi
  echo "[urbangrid] ===== traj$N (save_states=$SAVE_STATES) $(date +%T) =====" | tee -a "$LOG"

  for SCEN in "${SCEN_ARR[@]}"; do
    SS=$(slug "$SCEN")

    for M in $STEPS; do
      # ours jacfree
      if has_group ours_jacfree; then
        run_cell "$MET/${SS}__M${M}__traj${N}__ours_jacfree.csv" "ours_jacfree/$SS/M$M/traj$N" \
          "+urban_methods=$OURS" "+urban_scenarios=[\"$SCEN\"]" num_steps=$M \
          likelihood_mode=dps_jacobian_free \
          +save_states=$SAVE_STATES "+states_root=$STATES_ROOT"
      fi
      # ours shared -- one cell per requested refresh cadence k. lambda still comes
      # from the per-scenario table in configs/method/*.yaml; only k is set here.
      # k=1 keeps the historical `ours_shared` / variant=shared naming; k>1 is
      # stamped as ours_shared_jac<k> / variant=shared_jac<k> so they never collide.
      for K in $SHARED_KS; do
        if [ "$K" = "1" ]; then
          KOUT="ours_shared"; KVAR="shared"
        else
          KOUT="ours_shared_jac${K}"; KVAR="shared_jac${K}"
        fi
        run_cell "$MET/${SS}__M${M}__traj${N}__${KOUT}.csv" "${KOUT}/$SS/M$M/traj$N" \
          "+urban_methods=$OURS" "+urban_scenarios=[\"$SCEN\"]" num_steps=$M \
          likelihood_mode=inflated_shared \
          +jacobian_refresh_every=$K "+variant_override=$KVAR" \
          +save_states=$SAVE_STATES "+states_root=$STATES_ROOT"
      done
      # baselines (ignore likelihood_mode; run once per M)
      if has_group baselines; then
        run_cell "$MET/${SS}__M${M}__traj${N}__baselines.csv" "baselines/$SS/M$M/traj$N" \
          "+urban_methods=$BASELINES" "+urban_scenarios=[\"$SCEN\"]" num_steps=$M \
          likelihood_mode=dps_jacobian_free \
          +save_states=$SAVE_STATES "+states_root=$STATES_ROOT"
      fi
      # D-Flow SGLD alone -- its own group/file purely so the dearest baseline can
      # be paused or re-run without touching the other three. Aggregation globs
      # every CSV and keys on (method, variant, scenario, metric, M), so the split
      # is invisible downstream and the two halves rejoin on their own.
      if has_group dflow; then
        run_cell "$MET/${SS}__M${M}__traj${N}__dflow.csv" "dflow/$SS/M$M/traj$N" \
          "+urban_methods=$DFLOW" "+urban_scenarios=[\"$SCEN\"]" num_steps=$M \
          likelihood_mode=dps_jacobian_free \
          +save_states=$SAVE_STATES "+states_root=$STATES_ROOT"
      fi
    done
  done
  echo "[urbangrid] ===== traj$N done $(date +%T) =====" | tee -a "$LOG"
done
echo "[urbangrid] ALL DONE $(date)" | tee -a "$LOG"
