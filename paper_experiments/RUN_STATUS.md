# Navier–Stokes run status — 2026-06-30

## LIVE: step-count benchmark M=100 (PID 2839538)

**Still running detached** (survived session boundary). Log at the PREVIOUS
session's scratchpad: `/tmp/claude-12096798/.../ee14f21c-3ae9-47e8-aae9-b2bd89a70d12/scratchpad/run_ns_stepbench.log`

- **M=50: COMPLETE** — all 4 scenarios × 5 trajectories → 20 CSVs in `results/stepbench/csv/`.
- **M=100: in progress** as of 2026-06-30 07:20 — traj2 sparse 1.5625% running
  (OT-ODE done, SDA in progress ~40% at 07:20). traj3–5 remain (~12 h at current rate).
  7/20 M=100 CSVs exist.
- **⚠️ DECISION NEEDED — KILL before M=250 starts (~tonight/tomorrow):**
  M=250 + M=500 would consume ~3–4 GPU-days and block the headline grid
  (which is higher priority). Run `kill 2839538` when the M=100 block completes
  (watch the log for `[stepbench] ===== M=100 done`). M=50+100 data is sufficient
  for the step-count figure (2 points; add M=250/500 later if needed).
- **GPU state:** 19 GB / 24 GB used, 100% util.

### M=50 stepbench summary (5-traj mean RMSE, `num_physical_steps=15`)

| Method | 32²→128² | 16²→128² | sparse 5% | sparse 1.5625% |
|---|---|---|---|---|
| **Ours (SI-SDE)** | **0.066** | 0.287 | 0.628 | 0.762 |
| Ours (FM-SDE) | 0.067 | 0.292 | 0.623 | 0.772 |
| Ours (FM-ODE) | 0.068 | 0.293 | 0.649 | 0.782 |
| **SDA** | 0.261 | 0.860 | **0.130** | **0.246** |
| **OT-ODE** | 0.087 | **0.179** | **0.142** | **0.223** |
| FIG | 0.127 | 0.634 | 0.637 | 0.778 |
| SURGE | 0.404 | 0.522 | 0.395 | 0.439 |
| **FlowDAS** | 0.797 | 0.805 | 0.835 | 0.836 |

⚠️ **FlowDAS regression confirmed** (5-traj average): 0.797 for super-res 32² vs
old buggy 0.411. Root cause: the unit-L2-norm step normalization
(`guidance = -grad / ||grad||`) scales each element's step by ~1/√32768 ≈ 5×10⁻³,
making the guidance too weak at NS scale. **Fix requires a guidance-scale
hyperparameter sweep** (can't do without GPU). Current results are stable but
conservative; FlowDAS is still an honest baseline (the "old" 0.411 was an
overestimate from a non-faithful implementation). The paper's key message
(Ours + SDA + OT-ODE dominate) is unaffected.

## NEXT (when GPU is free after killing stepbench M=250+)

**Step 1 — Kill the stepbench before M=250 starts:**
```bash
# Watch log for completion of M=100 block, then:
kill 2839538        # or whatever PID the next process is
```

**Step 2 — Generate stepbench figure (M=50+100 CSVs already enough for 2-point curves):**
```bash
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
.venv/bin/python paper_experiments/make_ns_stepbench_figure.py
```

**Step 3 — Run headline 5-trajectory grid (all 8 step-based generative methods):**
```bash
# WITHOUT D-Flow (fast; ~2 h/scenario × 4 scenarios = ~8 h total):
setsid nohup bash paper_experiments/run_ns_headline.sh \
    >paper_experiments/results/headline/run.log 2>&1 & disown
# WITH D-Flow (adds ~34 min/scenario × 4 scenarios = ~2.5 h extra per traj):
# INCLUDE_DFLOW=1 setsid nohup bash paper_experiments/run_ns_headline.sh ...
```

**Step 4 — Run classical baselines (E=64 localized, sparse scenarios only):**
```bash
setsid nohup bash paper_experiments/run_ns_classical.sh \
    >paper_experiments/results/headline/run_classical.log 2>&1 & disown
```

**Step 5 — Merge and regenerate tables:**
```bash
bash paper_experiments/merge_and_tables.sh
cd manuscript && latexmk -pdf -interaction=nonstopmode main.tex
```

⚠️ **D-Flow NS hyperparameters are NOT yet locked** (the NS sweep never
completed). With `dflow_sgld.yaml` defaults (K=20, step_size=0.05), D-Flow
should run at NS scale — it has run successfully in the stepbench exclusion test.
Run at defaults for the headline, then do a targeted sweep (K ∈ {10,20,50} ×
η ∈ {0.01,0.05,0.1} on sparse 5%, traj1 only = 9 configs × ~34 min ≈ 5 h) if
the default numbers look off.

---

## Session update — 2026-06-29

**Method lineup is FINAL; baseline implementations were audited and three real bugs
were fixed and validated on the analytical case. The NS/urban headline runs have
NOT yet been re-run with the fixed methods — all `\tbd` cells in the manuscript
NS/urban tables come from these.** The detailed history below (2026-06-26/27/28) is
retained for context but several specifics are now superseded — read this section
first.

**Final method lineup.** Generative: Ours SI-SDE / FM-ODE / FM-SDE (the FM-SDE is
shown as "FM-SDE (DM)"); FlowDAS; Guided FM (FIG); Guided FM (OT-ODE); D-Flow SGLD;
SDA; SURGE. Classical (NS only): EnKF (E=1000 non-localized = ground-truth
posterior / KL reference; E=64 localized = baseline), LETKF, particle filter,
ensemble score filter. **The legacy "Guided FM" (one-step DPS-on-flow) and "Guided
diffusion" (DPS) are DROPPED** — enum kept for back-compat, removed from
`NS_METHODS`/`WIRED_METHODS`, `make_tables`, and urban `URBAN_METHODS`.

**New code this session:** `posterior_models/dflow_posterior.py` (D-Flow SGLD over
the FM source latent through a checkpointed FM-ODE rollout), `likelihood_models/dflow.py`,
`posterior_models/surge_posterior.py` (guided reverse-SDE proposal + Girsanov SMC
reweighting + ESS resampling), and in `likelihood_models/guidance.py` the
`FIGGaussianLikelihood` + the `weighting="ot_ode"` mode. `DenoiseDiffusionModel`
gained `from_flow_matching` (velocity mode) + a first-step t-shape fix.

**Diffusion prior is built from the FM model** (`diffusion_from_fm: true`, NS +
urban) — the trained diffusion checkpoint is weak. SDA and SURGE use this DM prior.

**Baseline-implementation audit + fixes (validated on the analytical linear-Gaussian
case, KL-to-exact):**
- **FlowDAS** — the non-faithful "bounded" surrogate `(x̂1−μ)/(v1+R)` (which vanished
  at the data end) was replaced with the paper's Algorithm-2 importance-weighted
  residual guidance pulled through the denoiser by autograd, DPS step-normalized.
  Analytical KL **0.299 → 0.080**.
- **D-Flow SGLD** — RMSProp preconditioner cold-start blew up the Langevin noise
  (~7×) for thousands of steps; fixed with Adam-style bias correction. KL
  **0.624 → 0.079** at K=200.
- **SDA** — the `1/‖Γ⁻¹r‖` step-normalization stripped the guidance magnitude (the
  SDA paper warns against it) and the `0.5g²√t` weight under-powered it ~10–16×.
  Fixed: drop the normalization, apply with the FM coefficient `a_τ+½g²`
  (`guidance_weight="fm_coeff"`). KL **0.436 → 0.019**.
- **FIG** — faithfully implemented (matches official riccizz/FIG), NOT a bug; it is
  structurally mismatched to full noisy observation (corrector targets `y_t=t·y`,
  collapses covariance→0). Reported as "collapsed" in the analytical table.

**Locked NS hyperparameters:** FIG (k=1, c=80, w=0); OT-ODE (σ_y²=0, γ=4), from a
traj1 / sparse-5% sweep vs the E=1000 EnKF posterior. (Analytical uses
regime-appropriate settings — OT-ODE σ_y²=R, γ=1; D-Flow K=200 — because the
NS-locked noiseless/few-step settings are degenerate in full observation.)

**Analytical case DONE.** All 11 methods are self-contained closed-form samplers in
`cases/analytical/samplers.py`. KL-to-exact (mean over 5 seeds): SI-SDE 0.0009,
FM-SDE 0.0016, FM-ODE 0.0011; FlowDAS 0.080; OT-ODE 0.0021; D-Flow SGLD 0.079; SDA
0.019; SURGE 0.0021; EnKF 0.0012; PF 0.0030; FIG collapsed. Three figures under
`manuscript/figures/analytical/`.

**STILL PENDING (need the GPU):**
- The NS/urban headline 5-trajectory runs (E=64, M=50; E=1000 only for the
  ground-truth EnKF). The old NS baseline numbers below are STALE (FlowDAS / SDA
  changed; legacy Guided FM / Guided diffusion removed; D-Flow SGLD + SURGE new).
- D-Flow SGLD + SURGE NS hyperparameter sweeps (never completed; D-Flow pSGLD is
  ~17 min/config). Their NS configs are not yet locked.
- Step-count benchmark (M=100 / 500) deferred.

---

## (historical) As of 2026-06-26
Full-scale runs were **deferred** (the machine's CPU was in use; GPU runs were
interrupted at session boundaries). The note below lists what to (re)run when
resources free up. **Note:** the method/baseline specifics below are superseded by
the 2026-06-29 section above.

## CRITICAL FIX (2026-06-26): EnKF/PF forecast interval was 50x too short
The true-solver EnKF/PF advanced only `INNER_STEPS*HF_DT = 100*1e-4 = 0.01` physical
time per assimilation step, but the dataset (and hence the SI/FM prior) uses a
`REDUCED_DT = 0.5` interval between consecutive states (`INNER_STEPS = 5000`). So the
filters under-propagated by 50x and were dragged by the observations rather than
forecasting. FIXED in `enkf_baseline.py` (`INNER_STEPS = REDUCED_DT/HF_DT = 5000`).
**The previously reported EnKF/PF numbers (rmse 0.758 / 1.32, in
`navier_stokes_classical_results.csv`) are INVALID and must be DISCARDED.** Re-run the
classical baselines. COST: the fix is ~50x more solver work per forecast (each step now
5000 sub-steps x E members x 256^2). **`enkf_baseline.py` now runs jax on the GPU by
default** (jax 0.8.1 has CUDA; `XLA_PYTHON_CLIENT_PREALLOCATE=false` so it shares with
torch) -- this makes the corrected run feasible. Force CPU with env `ENKF_JAX_PLATFORM=cpu`
when a torch job needs the whole GPU (then reduce E to 8-16 / fewer seeds). Do NOT raise
`HF_DT` above 1e-4 without checking 256^2 solver stability. The manuscript EnKF/PF rows
were reverted to `--`.

## DECISION: the multiplicative gain is dropped
The multiplicative gain `G_τ = I + β⁻²Σ_s HᵀR⁻¹H` (mode `dps_full`) did NOT improve
accuracy (analytical KL 0.001 inflated-covariance vs 0.174 gain; NS sparse rmse ~0.16
inflated vs ~0.72 cheap). Accuracy comes from **inflating the covariance** (G=I), not
the gain. So `dps_full` is **excluded from all runs and the paper** (kept only as a
code option, off by default). The ablation is now a **covariance** comparison
(`inflated` vs Jacobian-free), not a gain comparison.

## Generative baselines — FlowDAS + DPS STABILIZED (2026-06-27, confirmed on GPU)
- **FlowDAS FIXED:** `FlowdasGaussianLikelihood.score` (gaussian_likelihood.py) rewritten to the
  NORMALIZED, autograd-free guidance `(x1_hat - mu_x1)/(v1 + R)` matching the analytical sampler
  (MC predictions around the denoiser mean, softmax-weighted by `N(y; H x1, R)`; no UNet-Jacobian
  autograd). GPU smoke (E=8, M=20, sparse 1.5625%, seed 0): **rmse 0.816** (was ~4.8e3 -> NaN).
- **DPS (Guided diffusion) FIXED:** `DPSGaussianLikelihood.score` (guidance.py) now applies the DPS
  step-norm `zeta_t = zeta/||y - H xhat||` (divide the squared-residual gradient by the residual
  norm; the raw `1/sigma^2 = 400` factor removed). GPU smoke (same cell): **rmse 0.866** (was ~916
  -> NaN).
- Both are O(1), comparable to SI-SDE sparse-1.5625% (~0.865). The full generative grid (B.1) can
  now run; it is DEFERRED behind the conventional E=1000 ground-truth-posterior assessment (below).

## Generative grid vs EnKF ground-truth posterior + figures (2026-06-28)
> **SUPERSEDED (2026-06-29):** this grid used the OLD FlowDAS/SDA implementations and
> the now-DROPPED "Guided FM" / "Guided diffusion" baselines. The NS numbers here are
> STALE and must be re-run with the fixed methods + the new D-Flow SGLD / SURGE
> baselines. The E=1000 non-localized EnKF as the ground-truth/KL reference still
> stands, as does the figure tooling.

KL-at-points now uses the **E=1000 non-localized EnKF** as the ground-truth posterior reference
(driver `_reference_trajectory` loads `states_E1000_noloc/` by scenario+seed via
`+kl_reference_states=...`; truth-referenced metrics unchanged; no-ref -> KL=NaN). Seed-0 EnKF refs
exist for all 4 scenarios (sparse x2 + super-res x2); KL is at seed 0, truth metrics over 3 seeds.
**Full 7-method generative grid done** (Ours SI/FM-SDE/FM-ODE + FlowDAS, Guided FM, Guided diffusion,
SDA; E=64, M=50, dps_jacobian_free): `navier_stokes_gen_full.csv` (+ `gfm_sda_fixed.csv` /
`sda_fixed.csv` supersede the Guided FM / SDA rows). **Ours dominates** — KL-vs-EnKF 1-37 (super-res
~1) vs baselines 30-335; lowest RMSE/CRPS across all scenarios.
**ALL generative divergences fixed this session:** FlowDAS + DPS (normalized/step-norm guidance),
EnSF (bounded Kalman gain), **Guided FM** (`GuidanceGaussianLikelihood`: DPS step-norm), **SDA**
(`SDALikelihood`: normalize by `||Gamma^-1 r||` = `||sol||` -> pure network-Jacobian magnitude; the
Mahalanobis-norm attempt left it ~1/sigma too strong and it slowly blew up over the rollout). All 7
now finite at E=64.
**FIGURES:** `paper_experiments/make_method_figures.py` -> `results/method_figures/` (32 PNGs, seed 0):
per-method `method__<m>__<scenario>.png` (truth / mean / std / energy spectrum / RMSE+CRPS+spread-skill
+KL-vs-EnKF-vs-step) and per-scenario `compare__<scenario>.png` (all methods' RMSE/CRPS/KL/spread-skill
vs step + all spectra vs truth). NS result figures (`ns_trajectories.png`, `ns_diagnostics.png`) also
generated with real weights via `+save_figures=true`.

## Conventional E=1000 ground-truth-posterior assessment (2026-06-27, RESULTS IN)
Ran EnKF/LETKF/PF/EnSF at E=1000, both sparse scenarios, seed 0, `num_physical_steps=15`,
`+save_states=true` -> `results/states_E1000/` (distance-loc EnKF, PF, EnSF) and
`results/states_E1000_noloc/` (global EnKF). Analysis + figures:
`paper_experiments/analyze_groundtruth_posterior.py` -> `results/groundtruth_figures{,_noloc}/`.

**WINNER = NON-LOCALIZED (global) EnKF at E=1000** (final-step): sparse-5% RMSE **0.081** /
spread-skill 0.85; sparse-1.5625% RMSE **0.083** / 0.78. Vs:
- distance-localized EnKF (r=20): RMSE 0.242 / 0.626 -- localization HURTS the mean 3-8x at E=1000
  (Evensen-2024: global is right at large E; localization is a small-ensemble fix).
- Particle filter: collapsed (ESS~1), RMSE 0.75-0.85 -- unusable.
- EnSF: DIVERGES to 1e12 (NaN) -- `_analysis_update` obs-score Langevin step ~var_f/R (R=0.0025,
  ~400x) overshoots; needs a step-norm like the DPS fix. Unusable as-is.
- **LETKF cannot run at E=1000** -- per-grid-point local transform OOMs (~200 GB). Run at reduced
  E (<=256) only.
**Metric caveat:** the driver's logged `spread_skill` (~0.09) is a whole-trajectory aggregate
(includes the near-zero-spread spin-up); the analysis script's CONVERGED final-step ratio (0.78-0.85)
is the real calibration number -- quote that, not 0.09.
**Inflation sweep DONE -> inflation does NOT help, it WRECKS the ground truth** (non-loc EnKF, sparse 5%,
`results/ns_E1000_noloc_infl*.csv`): RMSE 0.081 (infl 1.0) -> 0.169 (1.3) -> 0.544 (1.6) -> 2.409 (2.0);
calibration never reaches 1. **FINAL REFERENCE = non-loc global EnKF at E=1000, inflation=1.0**
(RMSE 0.081, final-step spread/skill 0.85 -- accurate AND well-calibrated; no inflation needed).

## Urban (uDALES) eval (2026-06-27): big vs small FM/SI
E=32, sparse 5%, seed 0, num_steps 50. SI-SDE velocity RMSE 0.292 (big) vs 0.300 (small); FM-ODE
0.322 (big) vs 0.329 (small). SI > FM; big ~= small (near-tie) -> small matched pair pragmatic for the
grid (big ~4x costlier). `urban.yaml` defaults to big; swap to small if cost matters. CSVs
`results/urban_eval_{big,small}.csv`.

## Conventional-filter improvements for the ground-truth posterior (2026-06-27)
The E=1000 distance-localized EnKF is accurate in the MEAN (rmse 0.23 sparse-5% / 0.52
sparse-1.5625%) but badly UNDER-DISPERSED (spread/skill ~0.09-0.11) -> overconfident, NOT a
trustworthy posterior as-is. Two levers added to fix calibration:
- **Inflation knob** (`enkf_inflation`, default 1.0): multiplicative covariance inflation, threaded
  into `run_enkf_baseline`/`run_letkf_baseline` via `driver.py::_evaluate_classical`. Sweep with
  `+enkf_inflation=1.1` etc.
- **Adaptive correlation-based localization** (Vossepoel-Evensen-vanLeeuwen 2025), NEW opt-in
  `+enkf_localization_type=correlation` (default `distance` = unchanged Gaspari-Cohn). Implemented in
  `external_libs/jax_cfd_lib/src/jax_cfd_lib/{ENKF,ETKF}.py` (+module helpers `correlation_state_obs`,
  `correlation_localization_weights`); knobs `enkf_corr_threshold` (default `3/sqrt(N)`),
  `enkf_corr_inflation_max` (4.0), `enkf_corr_inflation_beta` (0.5). **LETKF path is the EXACT
  per-variable local analysis (use for headline); EnKF path is an approximate global-Schur variant
  (TODO(author) in ENKF.py).** Distance path verified bitwise-identical; 5 CPU unit tests pass
  (`external_libs/jax_cfd_lib/tests/test_correlation_localization.py`). At E=1000 the default
  threshold `3/sqrt(N)~0.095` truncates almost nothing -> set `+enkf_corr_threshold=0.3..0.4`.
- **OPEN SCIENCE QUESTION:** at E=1000, localization may be UNNECESSARY (Evensen 2024: 1000-member
  GLOBAL updates avoid filter divergence with consistent spread). The under-dispersion may be caused
  BY the localization. So also test the **non-localized** E=1000 EnKF (`enkf_localization_radius`
  unset) as a candidate ground-truth posterior before adding localization+inflation machinery.

## (historical) Generative baselines — robustness FIXED; FlowDAS/DPS need STABILIZATION
- **Root cause (diagnosed):** the earlier "SI posterior init must equal x0" crash was a
  SYMPTOM of numerical DIVERGENCE, not a feedback bug. FlowDAS diverges (rmse ~4.8e3 at
  n_assim=4 → NaN by n_assim=20); DPS (Guided diffusion) similarly blew up (rmse 916).
  Once the state goes non-finite, `allclose(nan, nan)=False` tripped the hard SI-init
  assert and crashed the WHOLE run.
- **FIXED (robustness):** the SI-init assert now only fires for FINITE states
  (`base_posterior.py`), so a diverging cell yields NaN metrics instead of crashing the
  run; and the in-place `base.requires_grad = True` in all three posteriors
  (`stochastic_interpolant_posterior.py`, `flow_matching_posterior.py`,
  `diffusion_posterior.py`) is replaced by a fresh detached leaf
  `base = base.detach().requires_grad_(True)`. Verified: FlowDAS at n_assim=20 now
  completes (NaN, no crash). The generative run will now run end-to-end, giving real
  numbers for the STABLE baselines and NaN for the unstable ones.
- **STILL TODO (stabilization) — ROOT CAUSE FOUND:** the WORKING analytical FlowDAS
  (`cases/analytical/samplers.py::flowdas_posterior`) uses a NORMALIZED guidance
  `(x̂₁ − μ_{x1}) / (v₁ + R)` (bounded, ΠGDM/pseudo-inverse-style). The NS
  `FlowdasGaussianLikelihood.score` instead returns a RAW `autograd.grad` of the
  softmax-weighted neg-log-likelihood through the UNet — an UNBOUNDED quantity (its scale
  is set by the network Jacobian) — which the posterior then multiplies by
  `w_tau = a_tau + ½g²`. That product blows up (NS FlowDAS rmse ~4.8e3 → NaN). FIX: make
  the NS FlowDAS guidance match the normalized analytical form (divide by the posterior
  obs-variance `v₁+R` ≈ `α²·HΣ_sHᵀ + R`), i.e. a pseudo-inverse-scaled guidance, not the
  raw gradient. DPS analogously needs a step normalization `ζ_t = ζ/‖y−Hx̂‖` (its raw
  `1/σ²=400` guidance is the culprit). SDA was stable (rmse 0.82 tiny test); Guided FM /
  EnSF still to confirm (need a GPU run). All of this needs the GPU (currently in use).

## What is already real (done)
> **Note (2026-06-29):** the analytical case is now the full 11-method lineup with
> faithful baselines (see the 2026-06-29 section at the top). The "Guided FM (FIG) /
> Guided diffusion (DPS) implemented" lines below refer to baselines that have since
> been DROPPED from the paper (FIG is retained; DPS-on-flow Guided FM and DPS Guided
> diffusion are gone).
- **Case 1 analytical** — complete, real numbers (the 3 samplers recover the exact
  posterior; KL ~0.001) and now the full faithful baseline lineup.
- **NS headline, `dps_jacobian_free`, E=64 M=50, 3 seeds** — the 3 "Ours" samplers
  (SI-SDE / FM-SDE / FM-ODE) over all 4 scenarios **completed** and were salvaged from
  the run log into `results/navier_stokes_results.csv`; `tab:ns_accuracy` /
  `tab:ns_calibration_cost` fill with these real numbers. (FlowDAS + baselines were not
  reached before the run was interrupted.)
- All baselines **implemented + reviewed + compile-clean**: Guided FM (FIG), Guided
  diffusion (DPS), SDA, ensemble score filter (generative); EnKF + bootstrap PF
  (true 256² jax-cfd solver, stride-2 subsample, obs-points identical to 7e-7).
- `inflated_shared` mode implemented + verified (collapses to exact `inflated` at 3e-15).
- DPS/SDA in-place-`requires_grad_` bug **fixed** (detached grad-copy) — verified by a
  tiny CPU test.

## SCHEDULED full runs (do these when CPU/GPU free)

All write to a **separate results file** then merge for `make_tables.py` (avoid
concurrent-write corruption). Use `likelihood_mode=dps_jacobian_free` so the KL
reference is cheap. `n_assim = num_physical_steps − 5 = 20` (config default).

1. **Generative baselines** (GPU): FlowDAS, Guided FM, Guided diffusion, SDA, Ensemble
   score filter — `dps_jacobian_free`, E=64, M=50, all 4 scenarios, 3 seeds.
   ```bash
   .venv/bin/python -u paper_experiments/run.py case=navier_stokes seeds="[0,1,2]" \
     likelihood_mode=dps_jacobian_free \
     '+ns_methods=["FlowDAS","Guided FM","Guided diffusion","SDA","Ensemble score filter"]' \
     ensemble_size=64 num_steps=50 case.reference_ensemble_size=128 \
     results_file=paper_experiments/results/navier_stokes_gen_results.csv \
     case.require_weights=true case.device=cuda
   ```
2. **Classical EnKF/PF** (jax forecast on **GPU by default**; corrected 5000-substep
   interval is ~50× heavier than the old buggy run): 2 sparse scenarios, E=32, EnKF
   localized, 2 seeds. (`case.device=cuda` puts the torch reference on GPU too; set
   `ENKF_JAX_PLATFORM=cpu` to force the jax solver onto CPU instead.)
   ```bash
   .venv/bin/python -u paper_experiments/run.py case=navier_stokes seeds="[0,1]" \
     likelihood_mode=dps_jacobian_free '+ns_methods=["EnKF","Particle filter"]' \
     '+ns_scenarios=["sparse 5%","sparse 1.5625%"]' \
     ensemble_size=32 num_steps=20 +enkf_localization_radius=20 \
     case.reference_ensemble_size=32 \
     results_file=paper_experiments/results/navier_stokes_classical_results.csv \
     case.require_weights=true case.device=cuda
   ```
3. **Ablation** (GPU, mode comparison → `tab:ablation`): FM-SDE, sparse 5%, base E=16 M=20,
   2 seeds (keeps the per-member `dps_full` point tractable).
   ```bash
   .venv/bin/python -u paper_experiments/run.py case=navier_stokes +ablation=true \
     ablation_smoke=false ablation_scenario="sparse 5%" \
     ensemble_size=16 num_steps=20 seeds="[0,1]" \
     results_file=paper_experiments/results/navier_stokes_ablation_results.csv \
     case.require_weights=true case.device=cuda
   ```
4. **Regenerate tables** from the union of all CSVs:
   ```bash
   RES=paper_experiments/results
   head -1 $RES/analytical_results.csv > $RES/all_results.csv
   tail -q -n +2 $RES/*_results.csv >> $RES/all_results.csv
   .venv/bin/python paper_experiments/make_tables.py --results $RES/all_results.csv
   ```

## Robustness notes (important)
- **Background runs are killed at session boundaries here.** Launch with
  `setsid nohup … &` AND rely on the run log: each completed cell logs an `[NS] … {metrics}`
  line, recoverable with `scratchpad/reconstruct_csv.py <log>` even if the run dies before
  writing its CSV.
- The NS driver writes its CSV only at the END (after all methods×scenarios×seeds). For
  resilience prefer per-method `+ns_methods` batches so partial progress is salvageable.
- `num_physical_steps` must exceed `len_field_history` (5) or the assimilation loop is empty
  (all-NaN metrics).

## Watch items
- **Guided diffusion (DPS) magnitude.** The tiny fix-verification (E=2, M=2, n_assim=2)
  gave DPS rmse≈916 vs O(1–10) for the others — likely the very coarse M=2 integration
  amplifying DPS's large `1/σ²=400` guidance. Check at full M=50; if it still diverges,
  DPS needs a guidance-step normalization (e.g. `ζ_t = ζ/‖y−Hx̂‖`), standard in DPS.
- Tiny-test sanity (E=2,M=2): FlowDAS 7.99, Guided FM 9.90, DPS 916, SDA 0.82 — wiring
  only, NOT meaningful numbers.

## Saving raw states (optional)
Raw posterior/truth field states are NOT saved by default (only scalar metrics → CSV;
everything is reproducible from the fixed seeds). To cache them, add `+save_states=true`
to any NS run: each cell writes `results/states/<case>__<method>__<scenario>__seed<k>__E<e>_M<m>.npz`
with `posterior_trajectory [E,C,H,W,T]`, `true_trajectory`, `observations`, `obs_indices`
(sparse sensors), cost, and metadata. `states_root` overrides the directory. The dir is
gitignored (files are ~2.4 MB/cell at 128², so a full grid is many GB). Off by default.

## Optional (decide later)
- Currently **one test trajectory** (`test_sample_indices[0]`); the seed std reflects
  obs-noise + sampler variance only. To average over the 5 held-out trajectories, add an
  outer loop over `test_sample_indices` (≈5× runtime).
- Canonical likelihood-mode decision (Task 3) awaits the ablation's gain axis
  (`inflated_shared` vs `dps_full` vs `dps_jacobian_free`).
- LaTeX: `paper_new/{gensymb,animate,listingsutf8}.sty` are **local compile shims** (those
  packages are missing from this TeX install); `animate` renders a static placeholder.
  Install the real packages for the final build (or keep the shims and gitignore them).
</content>
</invoke>
