# Project handoff — Observation-Interpolant Data Assimilation

**Date:** 2026-06-30. **Branch:** `sync-with-paper`. **Repo:**
`/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants`.
**Audience:** the next agent/engineer taking over. Read this top-to-bottom; it is
self-contained and current. Companion docs (older but still useful):

---

## Session update — 2026-06-30

**What's new since 2026-06-29:** The step-count benchmark (launched ~15:01 on
2026-06-29 via `setsid nohup`) survived the session boundary and is still running
as PID 2839538. M=50 is **fully complete** (all 4 scenarios × 5 traj = 20 CSVs).
M=100 is **in progress** (~12 h remaining). **M=250/500 must be killed before they
start** — they would consume 3–4 GPU-days and block the headline grid. See
`paper_experiments/RUN_STATUS.md` for the exact kill command and timing.

**Stepbench M=50 results (5-traj mean RMSE, `num_physical_steps=15`):**
- Super-res 32²: Ours 0.066, OT-ODE 0.087, FIG 0.127, SURGE 0.404, SDA 0.261, FlowDAS 0.797.
- Sparse 5%: SDA **0.130**, OT-ODE **0.142**, Ours 0.623–0.649, FIG 0.637, SURGE 0.395, FlowDAS 0.835.
- Sparse 1.5625%: SDA **0.246**, OT-ODE **0.223**, Ours 0.762–0.782, FIG 0.778, SURGE 0.439, FlowDAS 0.836.
- Key finding: **SDA and OT-ODE are the strongest baselines at sparse obs** (not just Ours).
  The fixed SDA transfers from the analytical case — it is now genuinely competitive.

**FlowDAS regression confirmed:** the faithful Algorithm-2 implementation gives
RMSE 0.797 (super-res 32²) vs the old buggy 0.411. Root cause: the unit-L2-norm
step normalization (`guidance = -grad / ||grad||_2`) scales each element's step by
~1/√32768 ≈ 5×10⁻³, making the guidance too weak at field scale. Fix requires a
guidance-scale hyperparameter sweep. **Current implementation is correct (faithful
to the paper's algorithm) — the paper's message is unaffected** (SDA + OT-ODE +
Ours dominate; FlowDAS is a faithful but weak baseline at NS scale).

**New scripts prepared (2026-06-30):**
- `paper_experiments/run_ns_headline.sh` — the headline 5-traj grid (8 step-based
  generative methods + optionally D-Flow, `num_physical_steps=25`, M=50, E=64,
  per-(scenario × traj) CSV files). Run with `INCLUDE_DFLOW=0` to defer D-Flow.
- `paper_experiments/run_ns_classical.sh` — classical baselines (EnKF E=64 loc
  r=20, LETKF, PF, EnSF; sparse scenarios only; all 5 traj).
- `paper_experiments/merge_and_tables.sh` — merge all headline CSVs into
  `all_results.csv` and regenerate LaTeX.

**NOTE: stepbench uses `num_physical_steps=15` (n_assim=10); headline grid uses
the default `num_physical_steps=25` (n_assim=20). THESE ARE DIFFERENT RUNS. The
headline table (`tab:ns_accuracy`) must use the headline grid, not the stepbench.**

---

## Session update — 2026-06-29

The method lineup is **final**, all baselines are **implemented and audited**, and
the analytical case is **DONE**. The NS/urban headline runs are **PENDING the GPU**.
Parts A/B below are largely current; the bullets here override any stale specifics.

**Final method lineup (9 generative + classical), grouped by (prior, sampler):**
- **Ours (unified family):** SI-SDE, FM-ODE, FM-SDE (shown in the paper as
  "FM-SDE (DM)" — a diffusion-model-style SDE on the FM prior).
- **SI prior + SDE:** FlowDAS.
- **Flow-matching prior + ODE:** Guided FM (FIG), Guided FM (OT-ODE), D-Flow SGLD.
- **Diffusion-model prior + SDE:** SDA, SURGE.
- **Classical (Navier–Stokes ONLY — urban has no solver):** EnKF (E=1000
  non-localized = ground-truth posterior / KL reference; E=64 localized = baseline),
  LETKF, particle filter, ensemble score filter.
- **DROPPED from the paper:** the legacy "Guided FM" (one-step DPS-on-flow) and
  "Guided diffusion" (DPS). Enum entries kept for back-compat but removed from
  `NS_METHODS`/`WIRED_METHODS`, `make_tables`, urban `URBAN_METHODS`.

**New code this session:** `posterior_models/dflow_posterior.py` (D-Flow SGLD over
the FM source latent through a gradient-checkpointed FM-ODE rollout),
`likelihood_models/dflow.py`, `posterior_models/surge_posterior.py` (guided
reverse-SDE proposal + Girsanov-corrected SMC reweighting + ESS resampling), and in
`likelihood_models/guidance.py` the `FIGGaussianLikelihood` + the `weighting="ot_ode"`
mode of `GuidanceGaussianLikelihood`. `DenoiseDiffusionModel`
(`models/diffusion_model.py`) gained a velocity-mode `from_flow_matching` constructor
+ a first-step t-shape fix.

**Diffusion prior is built from the FM model** (`DenoiseDiffusionModel.from_flow_matching`,
velocity mode) because the trained diffusion checkpoint is weak; controlled by
`case.checkpoints.diffusion_from_fm: true` (NS + urban). SDA and SURGE use this DM
prior; the FM-SDE sampler is its ODE/SDE sibling.

**Baseline-implementation audit + fixes (three real bugs, validated on the analytical
linear-Gaussian case where KL-to-exact is computable):**
- **FlowDAS:** the non-faithful "bounded" surrogate `(x̂1−μ)/(v1+R)` (vanished at the
  data end) → the paper's Algorithm-2 importance-weighted residual guidance pulled
  through the denoiser by autograd, DPS step-normalized. KL **0.299 → 0.080** (Chen
  et al. 2025, arXiv:2501.16642).
- **D-Flow SGLD:** RMSProp preconditioner cold-start (step-1 second moment ≈ 0 ⇒
  P=1/(λ+√V) explodes ⇒ ~7× too much Langevin noise) → Adam-style bias correction
  `V̂=V/(1−ρ^k)`. KL **0.624 → 0.079** at K=200 (Parikh et al., arXiv:2602.21469).
- **SDA:** the `1/‖Γ⁻¹r‖` step-normalization stripped the guidance magnitude and the
  `0.5g²√t` weight under-powered it ~10–16× → drop the normalization, apply with the
  FM coefficient `a_τ+½g²` (`DiffusionPosterior` `guidance_weight="fm_coeff"`). KL
  **0.436 → 0.019** (Rozet & Louppe, arXiv:2306.10574).
- **FIG:** faithfully implemented (matches official riccizz/FIG), NOT a bug — but
  structurally mismatched to full noisy observation (corrector targets `y_t=t·y`,
  collapses covariance→0). Reported as "collapsed" in the analytical table.

**Locked NS hyperparameters:** FIG (k=1, c=80, w=0); OT-ODE (σ_y²=0, γ=4), from a
traj1 / sparse-5% sweep vs the E=1000 EnKF posterior. (Analytical uses
regime-appropriate settings — OT-ODE σ_y²=R, γ=1; D-Flow K=200 — because the
NS-locked noiseless/few-step settings are degenerate in full observation.)

**Analytical case DONE.** All 11 methods are self-contained CLOSED-FORM samplers in
`cases/analytical/samplers.py` (they do NOT use the `scisi/src` posterior classes —
the linear-Gaussian prior velocity/score/drift are closed-form). KL-to-exact (mean
over 5 seeds): SI-SDE 0.0009, FM-SDE 0.0016, FM-ODE 0.0011; FlowDAS 0.080; OT-ODE
0.0021; D-Flow SGLD 0.079; SDA 0.019; SURGE 0.0021; EnKF 0.0012; PF 0.0030; FIG
collapsed. Three figures under `manuscript/figures/analytical/`: analytical_case.pdf,
analytical_kl_vs_steps.pdf, analytical_covariance_ablation.pdf (appendix).

**Manuscript (`manuscript/`).** Compiles clean. Tables list the full lineup with NO
per-row `\cite` (methods cited in prose + a new appendix "Method descriptions"
section, `sections/appendix_methods.tex`). New bib entries: yan_fig_2025,
ben-hamu_d-flow_2024, parikh_d-flow_2026, wei_surge_2026. Analytical table is filled;
NS (`tab:ns_accuracy`) + urban tables have many `\tbd` cells pending GPU runs.

**Urban (prepared, not run).** Generative-only, no conventional baselines, no KL, NO
energy/enstrophy — only per-variable RMSE + CRPS + spread-skill. **Sparse 5% and
sparse 1.5625% ONLY (no super-res).** `urban.yaml`:
si_run=stochastic_interpolant_big_gamma1, fm_run=flow_matching_big,
diffusion_from_fm=true, test_sample_indices [1..5]; data `data/udales/*.nc` +
`mask.npz` (test split sims 170–178); models under `checkpoints/udales/`.

**PENDING (all need the GPU):**
- **Kill stepbench before M=250** (PID 2839538 as of 2026-06-30 morning). M=50+100
  are sufficient for the step-count figure (2 data points). M=250/500 would take 3–4
  GPU-days and block everything else.
- **NS headline 5-trajectory grid** (`run_ns_headline.sh`, ready to launch). Runs all
  8 step-based generative methods (optionally D-Flow) at `num_physical_steps=25`,
  M=50, E=64, seeds=[0], per-(scenario × traj) CSVs under `results/headline/`. This
  fills `tab:ns_accuracy` / calibration table `\tbd` cells. Est. ~8–10 h (without
  D-Flow), ~12 h (with D-Flow, K=20 defaults).
- **NS classical baselines** (`run_ns_classical.sh`, ready to launch). EnKF E=64
  localized r=20, LETKF, PF, EnSF; sparse scenarios only; 5 traj × 2 scenarios.
- **D-Flow NS hyperparameters**: not yet locked (sweep never completed). The default
  config (K=20, step_size=0.05) should run at NS scale; try defaults in the headline
  grid first, then do a 9-config sweep (K∈{10,20,50} × η∈{0.01,0.05,0.1}) on
  sparse 5% traj1 only if the defaults look off (~5 h on GPU).
- **Urban headline grid**: generative-only (no classical), sparse 5% + 1.5625% only,
  5 traj, `urban.yaml` (big matched pair; diffusion_from_fm=true).
- **Step-count figure** (`make_ns_stepbench_figure.py`): ready to run once M=100 CSVs
  are complete; generates 4-panel per-scenario RMSE/CRPS/spread-skill/KL vs M figures.

---
`CODE_AND_EXPERIMENTS_OVERVIEW.md` (full code/experiment reference),
`paper_experiments/RUN_STATUS.md` (run commands + watch items),
`paper_new/GAP_ANALYSIS.md` + `paper_new/EXPERIMENTS_IMPLEMENTATION_SPEC.md` (original plan/spec),
`paper_experiments/HANDOFF_GPU.md` (original GPU brief — partly superseded by THIS file).

> **Golden rules for this environment**
> 1. **Always use `.venv/bin/python`** (bare `python` has no torch). torch 2.9.1+cu128, CUDA on.
> 2. **Background jobs are killed at session boundaries.** Launch long runs with
>    `setsid nohup … & disown`, AND rely on the log-salvage scripts (below): every cell logs an
>    `[NS] … {metrics}` line, so a run that dies before writing its CSV is fully recoverable.
> 3. **The NS driver writes its CSV only at the END.** For resilience, run per-method/scenario
>    batches and/or point each run at its OWN `results_file=` (concurrent writers corrupt one file;
>    `ResultsWriter` is append-only).
> 4. **`num_physical_steps` must exceed `len_field_history` (=5)** or the assimilation loop is empty
>    → all-NaN metrics. `n_assim = num_physical_steps − 5`.
> 5. Commit/push only when the author asks. Nothing has been committed this session.

---

## 0. One-paragraph project summary
The method turns any pre-trained flow-based generative prior (stochastic interpolant **SI** or
flow matching **FM**) into a posterior sampler for data assimilation, without retraining, by
conditioning the affine Gaussian probability path on the observation. The conditional drift is
`b_post = b_prior + w_τ·S̄`, with `w_τ = a_τ + ½g_τ²` and `S̄` a closed-form interpolant-likelihood
score. The equal-marginal SDE family gives three samplers — **SI-SDE** (native diffusion),
**FM-SDE** (lifted diffusion = a denoising-diffusion sampler), **FM-ODE** (deterministic) — that
share one observation-interpolant likelihood and differ only in `g_τ`. The science thesis: SI / FM /
SDE-flow-matching are ONE family, and the paper makes explicit which efficiency-vs-accuracy choices
matter and when. Three test cases: analytical linear–Gaussian (done), stochastic Navier–Stokes
(main, partly done), urban airflow (stub, data pending).

---

# PART A — CURRENT STATE (what's implemented, what ran, how to run)

## A.1 The library `src/scisi/`

### A.1.1 Likelihood — `likelihood_models/gaussian_likelihood.py::InterpolantGaussianLikelihood` ★
The heart of the method. Builds the closed-form interpolant-likelihood score
`S̄ = (Σ_s/σ_τ²) Hᵀ Σ̄⁻¹ (ȳ_τ − μ̄_τ)` with covariance `Σ̄ = β²R + H Σ_s Hᵀ`, observation interpolant
`ȳ_τ = α_τ H a₀ + β_τ y`, and source moments (SI Wiener / FM Tweedie). **Four `likelihood_mode`s:**

| mode | covariance `Σ_s` | gain `G_τ` | cost/pseudo-step | role |
|---|---|---|---|---|
| `inflated` | full, **per member** (`_build_HSHt`) | I | `O(E·N_y)` JVPs | exact for analytical; **intractable at NS scale** |
| **`inflated_shared`** | full, **shared at ensemble mean** (`_build_HSHt_shared`) | I | `O(N_y)` JVPs | tractable approx; ~hours/cell at NS |
| `dps_jacobian_free` | isotropic `ρI` | I | `O(N_uN_y)`, **no JVP** | cheap; **the headline mode used so far** |
| `dps_full` | full, per member | `I + β⁻²Σ_s HᵀR⁻¹H` | `O(E·N_y)` JVPs | **RETAINED but OFF** (see A.1.2) |

- **`inflated_shared`** evaluates the source-covariance Jacobian ONCE at the ensemble mean (shared),
  so `Σ̄` is one `N_y×N_y` matrix factorised once and reused for all members; `HΣ_sHᵀ` built by a
  chunked column-batched JVP (`shared_sigma_chunk=64`). It collapses EXACTLY to `inflated` when the
  ensemble members coincide — **verified to 3e-15** (`scratchpad/verify_inflated_shared.py`,
  reviewed independently). Use this for the inflated covariance at field scale.
- **`_math_sdpa()`** context manager (top of file) forces the math SDPA attention backend around the
  JVPs: on CUDA the flash/mem-efficient attention kernels have NO double-backward, which crashes
  `torch.autograd.functional.jvp`. Do not remove.
- `FlowdasGaussianLikelihood` (same file): the FlowDAS baseline (MC one-step predictions,
  softmax-weighted, `autograd.grad` through the net). **Diverges at NS scale — see B.1.**

### A.1.2 The gain `G_τ` was DROPPED (author decision, this session)
Empirics: the multiplicative gain `G_τ = I + β⁻²Σ_s HᵀR⁻¹H` (mode `dps_full`) did **not** help.
Analytical KL: **0.001** (inflated, G=I) vs **0.174** (gain) vs 0.104 (cheap). NS sparse-1.5625%
rmse: **0.155** (inflated_shared) vs 0.716 (cheap); the gain was killed mid-measure (it's the slow
per-member path). **Accuracy comes from inflating the covariance, not the gain.** Therefore:
- Paper: the "Multiplicative correction" Theorem/subsection was removed; drift is `b_prior + w_τ S̄`;
  the Jacobian-free Corollary recast as a *covariance* approximation.
- Experiments: `dps_full` dropped from all runs; the ablation "gain axis" → **covariance axis**
  (`inflated` vs Jacobian-free), `make_tables` recast (`ABLATION_ROWS`).
- Code: `dps_full` / `_apply_gain` are **kept as an option but OFF by default** (`apply_gain` is only
  true for `dps_full`; every default mode is `G=I`). See the comment at `self.apply_gain = …`.

### A.1.3 Posteriors — `posterior_models/`
`StochasticInterpolantPosterior` (SI-SDE, FlowDAS), `FlowMatchingPosterior` (FM-SDE, FM-ODE, Guided FM
FIG/OT-ODE), `DiffusionPosterior` (SDA, with `guidance_weight="fm_coeff"`), `DFlowPosterior` (D-Flow
SGLD), `SurgePosterior` (SURGE), `base_posterior.py` (shared `sample`/`sample_trajectory` loop).
(The legacy DPS-on-flow "Guided FM" and DPS "Guided diffusion" are dropped from the paper.)
**Fixed in an earlier session:** all three posteriors set `base.requires_grad = True` IN PLACE, which
corrupts the autoregressive feedback for grad-requiring likelihoods; replaced by a fresh detached
leaf `base = base.detach().requires_grad_(True)`. Also the SI-init assert in `base_posterior.py`
(`"SI posterior init must equal x0"`) now fires ONLY for finite states — a diverging sampler
produces NaN and `allclose(nan,nan)=False` used to crash the WHOLE run; now it yields NaN metrics
for that cell without blocking others.

### A.1.4 Other library pieces (unchanged this session unless noted)
- `models/interpolations.py` — affine-path schedules, `velocity_score_coeff` (a_τ), velocity↔score
  duality (general in α,β,γ; the trained SI uses **quadratic-β**, `β=τ²`).
- `models/follmer_stochastic_interpolant.py` (SI prior), `models/flow_matching_model.py` (FM prior +
  general `score`).
- `likelihood_models/observation_operators.py` — `LinearObservationOperator` (sparse `random`,
  super-res `super_res`/block-average). `obs_indices_on_grid` marks observed grid points (all points
  for super-res). `obs_indices` = sparse sensor flat indices.
- `likelihood_models/guidance.py` — `GuidanceGaussianLikelihood` (Guided FM/FIG baseline) +
  **`DPSGaussianLikelihood`** (NEW: faithful DPS, autograd through the Tweedie denoiser; **diverges
  at NS scale — see B.1**).
- `likelihood_models/sda.py` — **`SDALikelihood`** (NEW: single-window SDA adaptation; stable in
  the tiny test, rmse 0.82).
- `particle_filter/ensemble_score_filter.py` — **`EnsembleScoreFilter`** (score-based analysis:
  empirical-Gaussian ensemble score + tempered obs score). Its `_analysis_update` is reused by the
  true-solver `run_ensf_baseline` (see below). **EnSF now forecasts with the CONVENTIONAL true solver
  (jax-cfd), NOT the learned prior** (author decision, this session) — so it is grouped with the
  conventional filters (EnKF/LETKF/PF), not the generative baselines. The original learned-prior
  `EnsembleScoreFilter.run` is kept but no longer dispatched.
- `metrics/` — `ensemble_mean_rmse`, `radial_kinetic_energy_spectrum`, `crps` (supports a `mask`),
  `spread_skill`, `kl_at_points`, cost counters.

## A.2 The experiments harness `paper_experiments/`

- `run.py` (Hydra entrypoint), `common/runner.py` (loops methods×scenarios×seeds, `seed_everything`
  per cell → fully reproducible; writes tidy CSV), `common/seeding.py` (`SEED_LIST`, `mask_seed`,
  `obs_seed` — identical truth+obs+mask across methods), `common/aggregation.py` (**fixed this
  session:** `statistics.stdev` crashed on NaN; now NaN-safe).
- `results_schema.py` — tidy columns `case,method,scenario,metric,value,std,E,M,seed,NFE,seconds`.
  `Metric` enum now includes **`crps_observed` / `crps_unobserved`** (NEW). `likelihood_mode` is NOT
  a column → mode comparison lives in the ABLATION (scenario tags), not the headline.
- `configs/` — `benchmark.yaml` has top-level `likelihood_mode: null` (override on CLI);
  `case/navier_stokes.yaml` (checkpoints, `require_weights`, `device`, `num_physical_steps=25`,
  `num_steps=50`, `variance=0.0025`, `reference_ensemble_size=1024`, `test_sample_indices=[1..5]`);
  `method/*.yaml`, `scenario/*.yaml`.
- `cases/navier_stokes/`:
  - `driver.py::NavierStokesRunner` — `WIRED_METHODS` (generative: SI/FM-SDE/FM-ODE, FlowDAS, Guided
    FM (FIG), Guided FM (OT-ODE), D-Flow SGLD, SDA, SURGE), `CLASSICAL_METHODS` (**EnKF, LETKF, PF,
    Ensemble score filter** —
    ALL true-solver — via `_evaluate_classical`). Runs its OWN method×scenario grid; restrict with
    `+ns_methods=["…"]` / `+ns_scenarios=["…"]`. `evaluate_ablation` = covariance axis + g/M/E sweeps
    on FM-SDE. NEW: `_save_states` (off by default), and `crps_observed/unobserved` in
    `NS_FIELD_METRICS`.
  - `_ns_pipeline.py` — `load_prior` (SI+FM checkpoints), `build_obs_operator`,
    `prepare_truth_and_obs`, `build_posterior` (method+mode → posterior), `run_assimilation`,
    `build_reference_trajectory` (large-E SI-SDE reference for KL), `compute_metrics` (NEW: split
    CRPS).
  - `enkf_baseline.py` (NEW) — ALL true-solver classical baselines via the jax-cfd 256² solver,
    **stride-2 subsample** to 128² so observation points are IDENTICAL to the torch H (verified 7e-7),
    normalization std=3.09969, jax on **GPU by default** (`ENKF_JAX_PLATFORM`, default `cuda`).
    `run_enkf_baseline` (`LocalizedSpectralEnKF`), **`run_letkf_baseline`** (`LocalizedSpectralETKF`,
    inherently localized — defaults `localization_radius=20` if None), `run_particle_filter_baseline`,
    and **`run_ensf_baseline`** (true-solver forecast + the score-based `_analysis_update` from
    `EnsembleScoreFilter`, with 256²↔128² up/down-sampling and autoregressive analysis→forecast
    feedback). `INNER_STEPS=5000` (= one training interval; the 50× bug is fixed).
- `make_tables.py` — tidy CSV → LaTeX snippets (`generated/tab_*.tex`). Run after rebuilding
  `all_results.csv` (union of the per-case CSVs).

## A.3 Results that exist NOW (real numbers, in `paper_experiments/results/`)

> **Superseded by the 2026-06-29 update at the top:** the analytical case now holds
> the full 11-method faithful lineup (numbers up top); the NS "Ours"/baseline numbers
> below are STALE (baseline bug fixes; dropped/added methods) and must be re-run.

| file | contents |
|---|---|
| `analytical_results.csv` | **complete** — 3 samplers + 5 baselines, KL & sliced-W2 to the exact posterior |
| `navier_stokes_results.csv` | **3 "Ours" samplers** (SI/FM-SDE/FM-ODE), `dps_jacobian_free`, E=64, M=50, 3 seeds, all 4 scenarios |
| `navier_stokes_classical_results.csv` | **EnKF + bootstrap PF** — ⚠️ **INVALID, discard** (forecast interval was 50× too short; see B.10). Re-run after the fix. |
| `all_results.csv` | union of the above (rebuild before `make_tables.py`) |

**Headline NS numbers (dps_jacobian_free, mean/seed):** SI-SDE rmse 0.066 (super-res 32²) / 0.744
(sparse 5%) / 0.865 (sparse 1.5625%); FM-SDE 0.064/0.736; FM-ODE 0.065/0.759. Cost: SI-SDE NFE 50
~5 s/step; FM NFE 100 ~16 s/step. **Classical (sparse 5%):** EnKF rmse 0.758 (spread-skill ~0.95,
well-calibrated), PF rmse 1.32 (weights collapse) — **⚠️ these classical numbers are INVALID (B.10)
— forecast interval was 50× too short; re-run needed.** **inflated_shared sanity (sparse 1.5625%):** rmse
~0.14–0.16 vs ~0.72 for the cheap mode — the inflated covariance is much more accurate when
observations are sparse.

> **Important metric caveat (KL-at-points, NS):** the NS KL reference is a large-E ensemble drawn by
> our OWN SI-SDE sampler (no closed-form NS posterior exists). So NS `kl_points` measures *agreement
> with SI-SDE*, biased in our favor; it is NOT a fidelity ranking and should NOT be used to claim a
> distributional win over EnKF/PF (their high NS KL just means "different from SI-SDE"). The
> analytical KL IS vs the exact posterior. See B.8 for the recommended fix. (For NS, rank cross-method
> on RMSE / energy-spectrum / CRPS / spread-skill, which ARE truth-referenced.)

## A.4 The manuscript `manuscript/` (the live paper)

> **2026-06-29:** the tables now list the full final lineup with **NO per-row `\cite`**
> (methods cited in prose + a new appendix "Method descriptions" section,
> `sections/appendix_methods.tex`, one subsection per method). New bib entries:
> yan_fig_2025, ben-hamu_d-flow_2024, parikh_d-flow_2026, wei_surge_2026. The
> analytical table is filled with real numbers; the NS (`tab:ns_accuracy`) + urban
> tables have many cells marked `\tbd` (a new macro) pending GPU runs. The main-text
> analytical figure is a 2-panel `figure*` (2D case | KL-vs-steps); the
> covariance-ablation is a full-width appendix figure.

A restructured copy of `paper_new/` (reuses its macros/style/bib), compiles clean. Build:
```bash
cd manuscript && latexmk -pdf -interaction=nonstopmode main.tex   # → main.pdf
```
- **Local LaTeX shims** `manuscript/{gensymb,animate,listingsutf8}.sty` are present because the TeX
  install lacks those packages (pre-existing). `animate` renders a static placeholder. Install the
  real packages for the final build, or keep the shims.
- Structure (see `manuscript/PAPER_SPEC.md`, the writing bible): Intro (+ **Fig 1 visual abstract**,
  TikZ `figures/visual_abstract.tex`), Related work (merged), Preliminaries (Bayesian DA; SI; FM as
  SI-without-Wiener with ODE→SDE/diffusion bridge), Methodology (posterior sampling SDE+ODE +
  exactness theorem; interpolant likelihood + 2 lemmas; **[gain removed]**; comparison of 3 samplers
  + Tables; approximations subsubsections; summary + **Fig 2 decision guide**, TikZ
  `figures/decision_guide.tex`), Comparison to alternatives, Implementation (Algorithm 1 in body,
  2–4 in appendix), Results (analytical real; NS Ours real + EnKF/PF; generative baselines `--`),
  Conclusion (intentionally near-empty). The two TikZ figures are real (not placeholders).
- The page target is **10**; it's at **15**. The gain removal already shaved a page; reaching 10
  needs author decisions on what to cut (see B.6).
- `manuscript/sections/appendix_extra.tex` holds relocated floats (Algs 2–4, `tab:si_vs_fm`,
  `tab:approximations`, the placeholder NS/urban figures, urban tables, ablation table).

## A.5 Environment, data, weights
- venv `.venv` (uv). Data `data/stochastic_navier_stokes/data.npz` (200,100,128,128) raw vorticity,
  **downsampled from 256² by stride-2 subsample** (the EnKF relies on this). Preprocesser
  normalization: mean 0, **std 3.09969**. Trained priors:
  `checkpoints/stochastic_navier_stokes/{stochastic_interpolant_small,flow_matching}/model.pth`
  (SI and FM use DIFFERENT UNet sizes — a fairness caveat for SI-vs-FM, spec §5).
- jax-cfd true solver in `external_libs/jax_cfd_lib` (256², viscosity 1e-3, drag 0.1, Kolmogorov k=4,
  hf_dt 1e-4, inner_steps 100, REDUCED_DT 0.5).

---

# PART B — WHAT STILL NEEDS DOING (prioritized)

## B.1 ★ Generative baselines — IMPLEMENTED + AUDITED; run them at headline scale
**STATUS (2026-06-29):** All generative baselines are implemented and the FlowDAS / D-Flow SGLD / SDA
guidance bugs were found and fixed (validated on the analytical case — see the top-of-file update). The
DPS "Guided diffusion" baseline was DROPPED from the paper. What remains is **running** the fixed
methods at headline NS scale on the GPU and locking the D-Flow SGLD / SURGE NS hyperparameters.

> **(historical, 2026-06-27, superseded)** FlowDAS + DPS were "stabilized" with a normalized/step-norm
> surrogate (FlowDAS rmse 0.816, DPS 0.866). That FlowDAS surrogate was later found NON-FAITHFUL (it
> vanished at the data end) and replaced with the paper's Algorithm-2 guidance; DPS was dropped. Fixes:
- `FlowdasGaussianLikelihood.score` (gaussian_likelihood.py): normalized autograd-free guidance
  `(x1_hat − mu_x1)/(v1 + R)` (MC predictions around the denoiser mean, softmax-weighted by
  `N(y;Hx1,R)`) — matches the analytical sampler; no UNet-Jacobian autograd.
- `DPSGaussianLikelihood.score` (guidance.py): DPS step-norm `zeta_t = zeta/‖y−Hx̂‖` (divide the
  squared-residual gradient by the residual norm; dropped the raw `1/σ²=400`).
The full generative grid can now run — **DEFERRED** behind the conventional E=1000
ground-truth-posterior assessment (author request 2026-06-27; see RUN_STATUS.md). SDA stable
(0.82); Guided FM still to confirm in the grid.

**(historical) Original status:** FlowDAS and DPS (Guided diffusion) **numerically diverge** at NS
scale (FlowDAS rmse ~4.8e3 → NaN; DPS ~916). SDA was stable (0.82); Guided FM + EnSF unconfirmed.
The run no longer crashes (robustness fixed) but those two cells produce NaN.
**Root cause (diagnosed):** the working analytical FlowDAS uses a NORMALIZED guidance
`(x̂₁ − μ)/(v₁ + R)`; the NS `FlowdasGaussianLikelihood.score` returns a RAW `autograd.grad` through
the UNet × `w_τ` — unbounded → blows up. DPS likewise has no step normalization (raw `1/σ²=400`).
**Fix:** make the NS FlowDAS guidance match the normalized form (divide by `v₁+R ≈ α²HΣ_sHᵀ+R`,
a pseudo-inverse scaling); add a DPS step-norm `ζ_t = ζ/‖y−Hx̂‖`. **MUST be tested on GPU** (the
divergence only shows with the real UNet at scale) — don't implement blind.
**Then run** (when GPU free), generative baselines, `dps_jacobian_free`, E=64, M=50, all scenarios,
3 seeds → `navier_stokes_gen_results.csv`. Command in `paper_experiments/RUN_STATUS.md`.

## B.1c ★ Ground-truth posterior assessment + conventional-filter findings (2026-06-27)
**Author goal:** pick the conventional filter that gives the most trustworthy E=1000 posterior, to
use as the ground-truth reference for the generative runs (replaces the self-drawn SI-SDE KL ref;
addresses the B.8 caveat). Ran EnKF/LETKF/PF/EnSF at **E=1000**, both sparse scenarios, seed 0,
`num_physical_steps=15`, `+save_states=true`. New analysis script
`paper_experiments/analyze_groundtruth_posterior.py` (loads the saved `.npz`, recomputes per-method
final-step RMSE + spread/skill + CRPS + rank histogram, emits field+diagnostic figures to
`results/groundtruth_figures{,_noloc}/`).

**RESULTS (final-step, E=1000):**
| candidate | sparse 5% RMSE | spread/skill | sparse 1.5625% RMSE | verdict |
|---|---|---|---|---|
| **EnKF — non-localized (global)** | **0.081** | 0.85 | **0.083** | **WINNER: accurate + ~calibrated** |
| EnKF — distance-localized (r=20) | 0.242 | 0.84 | 0.626 | localization HURTS mean 3–8× at E=1000 |
| Particle filter | 0.750 | 0.00 | 0.853 | collapsed (ESS≈1) — unusable |
| Ensemble score filter | NaN | — | NaN | DIVERGES (1e12) — unusable as-is |

**Key takeaways:**
- **★ FINAL REFERENCE: non-localized global EnKF at E=1000, inflation=1.0** (Evensen-2024 result: at
  large E, global updates avoid filter divergence; localization is a SMALL-ensemble fix and here it
  *degrades* accuracy 3–8×). It is BOTH accurate (RMSE 0.081) AND well-calibrated at convergence
  (final-step spread/skill **0.85** sparse-5% / 0.78 sparse-1.5625%) — no inflation needed.
- **Inflation does NOT help — it WRECKS the ground truth** (sweep done, sparse 5%, `ns_E1000_noloc_infl*.csv`):
  RMSE 0.081 (infl 1.0) → 0.169 (1.3) → 0.544 (1.6) → **2.409 (2.0)**; calibration never reaches 1.
  Multiplicative inflation pumps the covariance each step and destabilizes the already-near-optimal
  global filter in this chaotic system. **Use inflation=1.0.** (The earlier "under-dispersion" worry
  was the misleading driver-aggregate `spread_skill`≈0.09, NOT the converged final-step 0.85.)
- **LETKF CANNOT run at E=1000** — the per-grid-point local transform OOMs (tries to allocate ~200 GB
  = 65536 points × 1000² matrices). Per-point localization is inherently small-ensemble. Run it (and
  the correlation-loc LETKF, same memory profile) only at REDUCED E (~256), or chunk the vendored lib.
- **EnSF diverges** — `EnsembleScoreFilter._analysis_update` (ensemble_score_filter.py:104-114): the
  obs-score Langevin step scales as `var_f/R` with `R=0.0025` (≈400×), past the explicit-Euler
  stability limit → overshoots, autoregressive feedback amplifies to 1e12. Needs a step-norm like the
  DPS fix (B.1) before it is usable.
- **Metric caveat:** the driver's logged `spread_skill` (~0.09) is a whole-trajectory aggregate
  (includes the spin-up transient where spread≈0); the analysis script's CONVERGED final-step ratio
  (0.78–0.85) is the meaningful calibration number. Don't quote the 0.09.

**NEW conventional-filter code (this session, opt-in, defaults unchanged):**
- **Inflation knob** `enkf_inflation` (default 1.0) → `driver.py::_evaluate_classical` →
  `run_enkf_baseline`/`run_letkf_baseline`. Sweep with `+enkf_inflation=1.2`.
- **Adaptive correlation-based localization** (Vossepoel-Evensen-vanLeeuwen 2025): opt-in
  `+enkf_localization_type=correlation` (default `distance` = unchanged GC, verified bitwise-identical).
  `external_libs/jax_cfd_lib/{ENKF,ETKF}.py` + helpers `correlation_state_obs`,
  `correlation_localization_weights`; knobs `enkf_corr_threshold` (def `3/sqrt(N)`),
  `enkf_corr_inflation_max` (4.0), `enkf_corr_inflation_beta` (0.5). **LETKF path = exact per-variable
  local analysis (headline); EnKF path = approximate global-Schur (TODO(author) in ENKF.py).** 5 CPU
  unit tests pass (`external_libs/jax_cfd_lib/tests/test_correlation_localization.py`). At E=1000 the
  default threshold `3/sqrt(N)≈0.095` truncates ~nothing → set `+enkf_corr_threshold=0.3..0.4`.
  NOTE: the correlation-loc LETKF will ALSO OOM at E=1000 (same per-point structure); use it at the
  small operational E (≤256), which is where localization is actually needed.

## B.2 ★ Full-scale NS grid + ablation (GPU)
Only `dps_jacobian_free` Ours + classical sparse have run. Still to do:
- The 3 Ours at **full headline scale** if more than 3 seeds / the spec settings are wanted.
- **EnKF/PF on the super-res scenarios** (they only ran sparse; localization is incoherent for
  block-average super-res — run non-localized or omit those columns).
- The **ablation** (covariance axis `inflated_shared` vs Jacobian-free + g/M/E sweeps) — needs
  `+ablation=true` (NOT `ablation=true`; it's not in the config struct). Fills `tab:ablation`.

## B.3 inflated_shared NS at scale (the accurate mode)
The inflated covariance is the more accurate method on sparse obs but costs ~hours/cell
(`O(N_y)` UNet-JVPs/step; exact `inflated` is `O(E·N_y)` = days/cell, do NOT run it at NS scale).
Decide the scale (reduced E/M/seeds, sparse scenarios) and run so the paper can report the
inflated-covariance numbers, not just `dps_jacobian_free`. This is also the canonical-mode decision
(Task 3): present inflated_shared vs dps_jacobian_free as the accuracy/cost trade-off.

## B.4 Urban Case 3 — IMPLEMENTED (2026-06-27); data + models arrived
**STATUS: the urban (uDALES) case is now WIRED and CPU-smoke-verified.** Data `data/udales/` (179
`sim_*.nc`, ~82 MB each, + `mask.npz`); models `checkpoints/udales/` (FM big/small, SI
big_gamma1/small_gamma1/small_original + archive). Implemented this session (driver `UrbanRunner`,
`cases/urban/_urban_pipeline.py`, `configs/case/urban.yaml`, README), reusing the NS driver pattern:
loader, fluid-cell masking (`fluid_keep_mask`, fluid-restricted sparse sensors), per-variable RMSE +
split CRPS + spread-skill (NO KL). CPU smoke passes (velocity RMSE 0.09 / temperature 0.045, finite).
- **DATA IS 4-CHANNEL `(u, v, w, thl)`, NOT 3 `(u, v, T)`** — the trained UNets are `in/out=4`; `thl`
  is temperature (~295 K), `qt` dropped. velocity RMSE = (u,v,w); temperature = thl. **The paper's
  "(u,v,T)" wording must be updated.** (`UDalesDataset.num_channels=5` is a stale hardcode; the driver
  derives C=4 from the loaded trajectory.) Splits: train(0,150)/val(151,170)/test(170,178);
  starting_time=50 → 200 steps. Mask `1=fluid (10409)`, `0=solid (5975)`; keep=fluid.
- **BEST MODELS (held-out one-step loss + an E=32 sparse-5% assimilation eval, seed 0):** SI beats FM;
  BIG ≈ SMALL on assimilation RMSE (SI-SDE velocity 0.292 big vs 0.300 small; FM-ODE 0.322 vs 0.329).
  `urban.yaml` defaults to the **matched BIG pair** (`stochastic_interpolant_big_gamma1` +
  `flow_matching_big`, same UNet size = SI-vs-FM fairness) but BIG is ~4× costlier for a near-tie →
  **small matched pair is the pragmatic choice for the full grid.** `small_original` is legacy
  (old normalization, sims 0–40) → EXCLUDE.
- **Remaining urban TODOs:** per-channel obs noise R (currently scalar 0.0025 in normalized space —
  `TODO(spec Section 6)`); `fig:urban_fields` (no urban figure module yet); a multi-seed FM/SI
  confirmation if the big-vs-small near-tie matters for the headline.
**SCENARIOS (2026-06-29): sparse 5% and sparse 1.5625% ONLY — NO super-resolution for urban.**
**GENERATIVE-ONLY comparison (author decision):** the urban CFD has NO true forward solver here, so
the conventional filters (EnKF, LETKF, bootstrap PF, ensemble score filter — all true-solver) CANNOT
be run for urban. **Do NOT add EnKF / LETKF / PF / EnSF to the urban case.** `URBAN_METHODS` excludes
them — urban runs only our 3 samplers + FlowDAS, Guided FM (FIG), Guided FM (OT-ODE), D-Flow SGLD, SDA,
SURGE (deep generative, sharing the prior); the manuscript urban tables had their classical rows removed. (Conventional baselines remain
valid for analytical (Case 1) and NS (Case 2), which DO have a known/true solver.)
**Urban metrics (author decision):** urban has only a ground-truth STATE, not a ground-truth
posterior, so **KL-at-points is NOT computed for urban** (it needs a reference posterior). Calibration
is assessed by the **spread--skill ratio and CRPS** (scored against the ground-truth state), split
observed/unobserved as for NS. `URBAN_METRICS` (urban/driver.py), the `make_tables` urban specs, and
the manuscript urban tables were updated accordingly (accuracy = velocity/temperature RMSE only;
calibration = CRPS + spread--skill + cost).

## B.5 Figures
- NS field/diagnostic figures: the functions exist (`_ns_figures.save_ns_trajectories` /
  `save_ns_diagnostics`) but were never generated with real weights. Run a cell with
  `+save_figures=true` (or use a saved `.npz` from `+save_states=true`), check they look right, then
  replace the `\figbox{}`/appendix placeholders in `manuscript/sections/results.tex` +
  `appendix_extra.tex` with `\includegraphics`.
- Analytical figures (`fig:analytical_panels`) similarly need generation + insertion.
- The two TikZ figures (visual abstract, decision guide) ARE done.

## B.6 Page budget (15 → 10) — needs author decision
A trim agent already moved Algs 2–4, `tab:si_vs_fm`, `tab:approximations`, urban, and placeholder
figures to the appendix. Reaching 10 from 15 needs cutting results or moving more theory (e.g. the
preliminaries SI/FM derivations, or the source-moment lemmas) to the appendix. Author must say what's
expendable.

## B.7 Multi-trajectory averaging (currently ONE trajectory)
`make_context` uses `test_sample_indices[0]` only; `run()` loops seeds, NOT trajectories. So the seed
std reflects obs-noise + sampler variance, not trajectory spread. To average over the 5 held-out
trajectories (180–200 split), add an outer loop over `test_sample_indices` and key aggregation on
`(method, scenario, seed, test_index)` (~5× runtime). Author deferred this ("keep one for now").

## B.10 ★ Re-run the classical EnKF/PF with the corrected forecast interval
**Bug found + fixed (2026-06-26):** `enkf_baseline.py` advanced the true solver only
`100×HF_DT = 0.01` physical time per assimilation step, but the data/training interval is
`REDUCED_DT = 0.5` (`INNER_STEPS = 5000`). The filters under-propagated 50× → the reported EnKF/PF
numbers are **invalid and were reverted** (CSV + manuscript). `INNER_STEPS` is now
`int(REDUCED_DT/HF_DT) = 5000`. It is now ~50× more solver work per forecast (5000 sub-steps × E ×
256²), so **run it on GPU**: `enkf_baseline.py` now runs jax on the **GPU by default** (jax 0.8.1 has
CUDA; `XLA_PYTHON_CLIENT_PREALLOCATE=false` so it shares the GPU with torch). Force CPU with env
`ENKF_JAX_PLATFORM=cpu` (e.g. when a torch job needs the whole GPU). **Re-run** the classical
baselines on GPU; if the GPU is busy, fall back to CPU with reduced E (8–16) / fewer seeds (slow).
Do NOT raise `HF_DT` above 1e-4 without checking 256² solver stability. Sanity-check after the fix:
the EnKF forecast of a member from the truth IC (no obs) should track the truth's next state, not lag it.

## B.8 Manuscript number-sync + KL caveat (after baselines land)
- Sync `manuscript/sections/results.tex` tables with the latest CSVs once the generative + ablation
  numbers exist (EnKF/PF are already in; generative are `--`). Consider auto-`\input`-ing the
  generated `tab_*.tex` instead of hardcoding, OR regenerate by hand.
- **Soften the NS KL@points claim** (currently implies a distributional win over EnKF). Recommended:
  present NS `kl_points` as a self-consistency check among the generative samplers only; rank
  EnKF/PF on truth-referenced metrics. Add the split-CRPS (`crps_observed`/`crps_unobserved`, now
  computed) to the tables if wanted (currently recorded but not in any paper table).
- Add the **inflated-covariance** rows once B.3 runs; state the canonical-mode decision.

## B.9 Documentation to keep current
- `paper_experiments/RUN_STATUS.md` — the live run log / watch items; update as runs complete.
- `CODE_AND_EXPERIMENTS_OVERVIEW.md` — the full code/experiment reference (still accurate; predates
  the gain removal + the new metrics/save_states — refresh those sections).
- THIS file — update as B.1–B.8 land.

---

# PART C — FILE-BY-FILE CHANGE LOG (this session)

**Continuation session (2026-06-27) — additional changes:**
- **LETKF baseline (NEW):** `enkf_baseline.py::run_letkf_baseline` (wraps `jax_cfd_lib.ETKF.LocalizedSpectralETKF`;
  inherently localized → defaults `localization_radius=20` via `DEFAULT_LETKF_LOCALIZATION_RADIUS`);
  `Method.LETKF` added to `results_schema.py` + `NS_METHODS` + `CLASSICAL_METHODS`; dispatched in
  `_evaluate_classical`; `make_tables.py` label + group. Paper: LETKF row + `hunt_efficient_2007` bib
  entry (`manuscript/library_NTM.bib`). **Vendored library fix:** `external_libs/jax_cfd_lib/.../ETKF.py`
  `gaspari_cohn` could return a tiny negative weight (~−2e-7) at the support boundary → `sqrt(w+1e-10)`
  NaN poisoned the whole field; clamped to `[0, ∞)` (genuine correctness fix, not test-only).
- **EnSF → conventional forecast:** `enkf_baseline.py::run_ensf_baseline` (true-solver jax forecast +
  `EnsembleScoreFilter._analysis_update` score-analysis). `_evaluate_classical` now routes EnSF here
  (NOT the old learned-prior `EnsembleScoreFilter.run`). EnSF regrouped with conventional filters in
  the paper; **removed from `URBAN_METHODS`** + urban paper tables (urban = generative-only, no solver).
- **Analytical covariance ablation (NEW, appendix):** `analytical/driver.py::run_ablation`/`evaluate_ablation`
  runs the 3 samplers × {`inflated`, `inflated_shared`, `dps_jacobian_free`} vs the exact posterior;
  `samplers.py` now recognises `inflated_shared` (= `inflated` in the state-independent linear case).
  Run via `+ablation=true case=analytical`. Numbers in `manuscript` `tab:analytical_ablation`
  (individual ≈ shared KL ≈ 1e-3; isotropic ≈ 0.04–0.07).
- **Urban metrics:** KL-at-points dropped (no GT posterior); spread-skill + split CRPS instead
  (`URBAN_METRICS`, `make_tables` urban specs, urban paper tables).
- **Autoregressive rollout VERIFIED** for all NS methods (generative via `sample_trajectory`;
  EnKF/LETKF/PF/EnSF via the carried analysis ensemble) — posterior@n feeds forecast@n+1, no truth leakage.

**Modified (`git status` M):**
- `src/scisi/likelihood_models/gaussian_likelihood.py` — `_math_sdpa` (CUDA double-backward fix);
  `inflated_shared` mode (+`_build_HSHt_shared`, `_bcast` broadcast, shared-Jacobian build in
  `score`); gain marked retained-but-off.
- `src/scisi/likelihood_models/guidance.py` — NEW `DPSGaussianLikelihood`; DPS/SDA detached-grad-copy.
- `src/scisi/posterior_models/{base_posterior,stochastic_interpolant_posterior,flow_matching_posterior,diffusion_posterior}.py`
  — robust SI-init assert (finite-only); in-place `requires_grad` → fresh detached leaf.
- `paper_experiments/cases/navier_stokes/_ns_pipeline.py` — split CRPS (`crps_observed`/`unobserved`).
- `paper_experiments/cases/navier_stokes/driver.py` — wire DPS/SDA/Guided FM (generative) + EnSF/EnKF/PF
  (classical); ablation gain→covariance axis; `_save_states`; new CRPS metrics in `NS_FIELD_METRICS`.
- `paper_experiments/common/aggregation.py` — NaN-safe mean/std.
- `paper_experiments/make_tables.py` — ablation rows recast (covariance, not gain).
- `paper_experiments/results_schema.py` — `CRPS_OBSERVED`/`CRPS_UNOBSERVED`.
- `paper_experiments/configs/benchmark.yaml` — top-level `likelihood_mode: null`.
- `paper_new/sections/methodology.tex` — approximations subsection + `tab:approximations` (the
  OLD paper; the live rewrite is in `manuscript/`).
- `.gitignore` — ignore `paper_experiments/results/states/`.

**New (`git status` ??):**
- `manuscript/` — the rewritten paper (the deliverable).
- `paper_experiments/cases/navier_stokes/enkf_baseline.py` — true-solver EnKF/PF.
- `src/scisi/likelihood_models/sda.py`, `src/scisi/particle_filter/ensemble_score_filter.py` — baselines.
- `paper_experiments/RUN_STATUS.md`, `CODE_AND_EXPERIMENTS_OVERVIEW.md`, `PROJECT_HANDOFF.md` (this).
- `paper_new/{gensymb,animate,listingsutf8}.sty` — LaTeX shims (also copied into `manuscript/`).

**Salvage/verify scripts (in the session scratchpad — copy to repo if you want them durable):**
`reconstruct_csv.py` (parse a headline log → CSV), `salvage_classical.py` (parse EnKF/PF log → CSV),
`verify_inflated_shared.py` (the 3e-15 collapse test). The scratchpad is
`/tmp/claude-…/scratchpad/` and is EPHEMERAL.

---

# PART D — EXACT NEXT STEPS (copy-paste; mind the golden rules)

Updated 2026-06-30. See also `paper_experiments/RUN_STATUS.md` for the live log.

```bash
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
PY=.venv/bin/python

# (0) Kill stepbench before M=250 starts:
#   Watch: tail -f <previous-scratchpad>/run_ns_stepbench.log
#   When you see "[stepbench] ===== M=100 done", run:
kill 2839538    # (re-check PID first: ps aux | grep run_ns_stepbench | grep -v grep)
#   Then generate the step-count figure:
$PY paper_experiments/make_ns_stepbench_figure.py

# (1) Headline 5-trajectory grid — 8 step-based generative methods, all 4 scenarios:
#     (D-Flow included by default; set INCLUDE_DFLOW=0 to defer)
setsid nohup bash paper_experiments/run_ns_headline.sh \
    >paper_experiments/results/headline/run.log 2>&1 & disown
#     Watch: tail -f paper_experiments/results/headline/run.log
#     ETA: ~8-12 h (with D-Flow). CSVs land in results/headline/*.csv

# (2) Classical baselines (sparse scenarios, 5 traj, E=64 localized r=20):
#     CAN run concurrently with step (1) if the torch job is light enough to share the GPU
#     with jax (XLA_PYTHON_CLIENT_PREALLOCATE=false lets them coexist).
#     Safer: run after (1) finishes, or force jax to CPU with ENKF_JAX_PLATFORM=cpu.
setsid nohup bash paper_experiments/run_ns_classical.sh \
    >paper_experiments/results/headline/run_classical.log 2>&1 & disown

# (3) D-Flow hyperparameter sweep (if the defaults in step 1 look off):
#     9 configs × ~34 min = ~5 h on GPU (sparse 5%, traj1 only)
for K in 10 20 50; do
    for ETA in 0.01 0.05 0.1; do
        $PY paper_experiments/run.py case=navier_stokes \
            +test_index=1 seeds=[0] likelihood_mode=dps_jacobian_free \
            '+ns_methods=["D-Flow SGLD"]' '+ns_scenarios=["sparse 5%"]' \
            ensemble_size=64 num_steps=50 \
            '+kl_reference_states=paper_experiments/results/multitraj/states/traj1/gt' \
            "method.likelihood_model.num_optim_steps=$K" \
            "method.likelihood_model.step_size=$ETA" \
            case.require_weights=true case.device=cuda \
            results_file="paper_experiments/results/dflow_sweep_K${K}_eta${ETA}.csv"
    done
done

# (4) Urban headline grid:
setsid nohup $PY -u paper_experiments/run.py case=urban \
    seeds="[0,1,2]" likelihood_mode=dps_jacobian_free \
    case.require_weights=true case.device=cuda \
    results_file=paper_experiments/results/urban_headline_results.csv \
    >paper_experiments/results/urban_headline.log 2>&1 & disown

# (5) Merge + regenerate tables:
bash paper_experiments/merge_and_tables.sh
cd manuscript && latexmk -pdf -interaction=nonstopmode main.tex && cd ..

# Salvage a run that died before writing its CSV (per-cell metrics survive in the log):
#   Look for "[NS] <method> | <scenario> | seed=0 | E=64 M=50 | {metrics}" lines.
#   The reconstruct_csv.py pattern from earlier sessions works for these too.
```

**If a run crashes:** check the log for the `[NS]` lines (salvageable) and the traceback. Common
causes already handled: NaN aggregation, SI-init assert on divergence, CUDA SDPA double-backward,
`num_physical_steps ≤ 5`. New crashes → read the traceback; don't assume.

**FlowDAS note (2026-06-30):** the faithful implementation gives RMSE ~0.80 at NS scale
(unit-norm step too weak at d=32768). The result is stable (not NaN) and honest. No
code change needed before the headline run — include it as-is. A guidance-scale sweep
can be done later if the author wants to close the gap.
</content>
