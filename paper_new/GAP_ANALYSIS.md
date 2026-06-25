# Paper ↔ Implementation Gap Analysis

Comparison of the manuscript in `paper_new/` against the current `src/scisi/` implementation.
Produced by a three-agent audit covering (1) the prior generative models / affine-Gaussian-path
interface, (2) the posterior-sampler core / guidance machinery, and (3) the experiments,
baselines, and metrics. Per-area detail lives in `scratchpad/discrepancies_{1,2,3}_*.md`; this file
is the consolidated overview and plan.

---

## 1. Headline assessment

The codebase is **mid-migration** toward the paper's framework. The paper's central claim is a
*single unified family*

```
b_post = b_prior + w_tau · G_tau · Sbar          w_tau = a_tau + ½ g_tau²
```

instantiated as three samplers (SI-SDE, FM-SDE, FM-ODE) that share identical observation handling
and differ only in `g_tau`, `w_tau`, and the source. **This structure is nowhere implemented
faithfully today.** The SI path has the right skeleton but is missing the load-bearing weight
`w_tau` (replaced by a hand-tuned constant `correction_multiplier=3.0`); the FM path is a legacy
FlowDAS/FIG guidance baseline, not the paper's method; FM-SDE does not exist; and the evaluation
can fill only ~⅓ of the promised result cells (every table in `results.tex` is still a placeholder).

The three layers have a strict dependency chain: **experiments need the samplers, the samplers need
the prior models to expose `a_tau` and `fm_score`.** Fixes must proceed bottom-up.

### Severity tally

| Layer | Critical | Major | Minor |
|---|---|---|---|
| Prior models (affine path) | 1 | 4 | 5 |
| Posterior sampler core | 3 | 5 | 3 |
| Experiments / baselines / metrics | 4 | 8 | 7 |

---

## 2. Layer 1 — Prior generative models (the affine-Gaussian-path interface)

*Foundation. Source: `discrepancies_2_prior_models.md`.* The SI score recovery (Eq. `SI_score`) and
SI drift-loss target are implemented correctly and generally — the one solid piece. The gaps are
general-vs-special-case mismatches that make the code correct only for the two shipped configs.

| # | Sev | Gap | Where | Fix |
|---|---|---|---|---|
| L1 | **CRIT** | FM prior exposes **no score method**; Eq. `fm_score` recovery hard-coded to rectified flow (`x+(1-t)·drift`, `(1-t)/t`) with general formula commented out | `models/flow_matching_model.py` (no `score`), `likelihood_models/guidance.py:73,90` | Add general `FlowMatchingModel.score(x,t)` from `interpolation.{alpha,beta,*_diff}`; route guidance through it |
| L2 | MAJ | Velocity–score coefficient `a_tau` (`vscoef`) **never computed anywhere** — the deterministic part of `w_tau` | absent in `src/scisi/` | One `velocity_score_coeff(t)` helper per interpolation: SI `σ=γ√τ`, FM `σ=α` |
| L3 | MAJ | SI trained schedule is **quadratic-β (`β=t²`)**, not the paper's rectified-flow running example — every "rectified-flow reduces to…" SI statement is inapplicable to the trained model | `config/stochastic_navier_stokes.yaml:62`, `interpolations.py:173-240` | Decide: document quadratic-β in paper, or switch config to linear SI. **Needs user call.** |
| L4 | MAJ | Full source-covariance Jacobian term (`J_b`/`∇s`) never implemented — only the Jacobian-free Corollary gain is available, never full `G_tau`; no FM `α²` covariance path | `gaussian_likelihood.py:96-97` | Implement full `r_cov`, or explicitly scope paper to Jacobian-free gain |
| L5 | MAJ | Velocity↔score identity hand-rolled in **4 divergent places** (no single source of truth) | `follmer_stochastic_interpolant.py`, `diffusion_model.py:65-80`, `guidance.py`, missing FM | Factor into one shared helper on the interpolation classes |
| L6 | min | Additive denominator guards (`1e-6/1e-4/1e-3`) bias `A_tau`/gains O(1) near `τ=0`; inconsistent across copies | `follmer_*:226`, `gaussian_likelihood.py:97,117` | Clamp `τ` away from {0,1} consistently |
| L7 | min | No endpoint-vanishing `g_tau` for FM-SDE lift; FM score singular at `τ→1` unguarded | config `fm.yaml` (ode only) | Add `g_τ ∝ √(αβ)` + finiteness assert when FM-SDE is built |
| L8 | min | SI `_drift_with_prior_score` default diffusion is literal `1-t`, not `self.interpolation.gamma` | `follmer_*:239` | Default to `self.interpolation.gamma` |
| L9 | min | SDE "Heun" reuses one Wiener increment (not marginal-preserving); ODE solvers offer only Euler | `sampling/sde_solvers.py`, `ode_solvers.py` | Restrict Heun/RK4 to ODE; add them there |

---

## 3. Layer 2 — Posterior sampler core (the method itself)

*The heart of the paper. Source: `discrepancies_1_sampler_core.md`.* Three structurally
inconsistent posterior classes sit on two different likelihood models; the unified family is not
realized.

| # | Sev | Gap | Where | Fix |
|---|---|---|---|---|
| P1 | **CRIT** | Guidance weight `w_tau = a_tau + ½g²` **missing from all samplers**; SI substitutes constant `correction_multiplier=3.0` | `stochastic_interpolant_posterior.py:91`, `flow_matching_posterior.py:70` | Form `w_tau` from `a_tau` (L2) + `½g²`; multiply corrected score by it; drop `correction_multiplier` |
| P2 | **CRIT** | FM posterior does **not** use the interpolant likelihood — it's a DPS/FIG one-step prediction with raw `R`: no obs interpolant, no `Σ̄` inflation, no source moments, no `G_tau` | `flow_matching_posterior.py:52-70`, `guidance.py:74,108-119` | Route FM through an interpolant likelihood (a0=0 ⇒ `ȳ=βy`, `μ_s=-α²s`, FM gain), score via `fm_score` |
| P3 | **CRIT** | **FM-SDE (Sampler 2) missing**: no diffusion term, no Brownian increment, no score recovery for the lift; FM-ODE has wrong weight/score | `flow_matching_posterior.py:37-72` | Parameterize `_one_step` by `diffusion_term` (None⇒ODE, callable⇒SDE); add `g∝√(αβ)`, retain Brownian |
| P4 | MAJ | Covariance inflation isotropic/Jacobian-free only (SI); absent (FM uses raw `R`) | `gaussian_likelihood.py:95-98` | `jacobian_free` flag; full `H Σ_s Hᵀ`; FM inflates `β²R + α²HHᵀ` |
| P5 | MAJ | `G_tau` Jacobian-free-only, scalar-collapsed, entangled with tuning multiplier; FM gain absent | `gaussian_likelihood.py:116-134` | Separate `G_tau` from `w_tau`; full vs Jacobian-free selectable; add FM gain |
| P6 | MAJ | `grad_xbar_mu` computed via autograd through `μ_s`/network instead of `≈ H` — different, costlier score | `gaussian_likelihood.py:181-195` | Closed-form `Sbar = Hᵀ Σ̄⁻¹(ȳ-μ̄)` with covariances detached |
| P7 | MAJ | Singularity at `τ=0` masked by ad-hoc epsilons instead of starting guidance at `τ=dτ`; loop bound `num_steps-1` drops final step; SI init not asserted `=x0` | `base_posterior.py:181`, `gaussian_likelihood.py` | Start guidance at `τ=dτ`; fix loop bound; assert SI init |
| P8 | MAJ | `DiffusionPosterior` scales likelihood by `½g²·√t` — no paper counterpart | `diffusion_posterior.py:78-80` | Use `w_tau`, or label clearly as non-paper VP baseline |
| P9 | min | SI bundles SMC particle-filter resampling not in methodology; configs disagree (`resample` true vs false) | `stochastic_interpolant_posterior.py:105-146` | Default `resample=false` in paper configs; document as optional |
| P10 | min | Operator-split update obscures `b_prior + w_tau G Sbar` structure | `stochastic_interpolant_posterior.py:85-91` | Assemble one combined EM update |
| P11 | min | FM anchor `a0=0` (⇒ `ȳ=βy`) not implemented in its likelihood | `guidance.py` | Resolved by P2 |

---

## 4. Layer 3 — Experiments, baselines, metrics

*The evaluation. Source: `discrepancies_3_experiments.md`.* `results.tex` is 100% placeholders;
current code can fill ~⅓. Three test cases, 7 baselines, 10 metrics, 4 observation scenarios.

**Test-case readiness**

| Case | Data | Train | Assimilate |
|---|---|---|---|
| 1. Analytical linear–Gaussian | n/a (synthetic) | n/a (closed form) | ✅ but 2D only, not the 3 samplers |
| 2. Stochastic Navier–Stokes | ✅ (~1.2 GB) | ✅ | ✅ SI/FM/FlowDAS, strided-grid obs only |
| 3. Urban airflow (uDALES) | ❌ empty dir | path exists | path exists — **no data, no generator** |

**Baselines:** FlowDAS ✅ · Guided FM/FIG ⚠ partial/unlabelled · DPS ⚠ partial, off-benchmark ·
EnKF ⚠ present but off-pipeline (separate JAX pkg) · bootstrap PF ❌ (empty stub) · SDA ❌ ·
ensemble score filter ❌.

**Metrics:** RMSE ⚠ (per-member not ensemble-mean) · energy-spec RMSE ⚠ (enstrophy form) ·
CRPS ✅ (biased estimator) · spread–skill ❌ · rank histogram ❌ · KL-at-points (fields) ❌ ·
sliced-W2 ⚠ (is W1) · NFE ❌ · wall-clock ❌.

**Observation operators:** super-res block-average (32²→128², 16²→128²) ❌ (only strided/random
point-selection) · sparse 5% / 1.5625% ⚠ present but commented out, masks unseeded.

| # | Sev | Gap | Fix |
|---|---|---|---|
| E1 | **CRIT** | Super-resolution (block-average) obs operator missing — 4 of 6 main-table column groups | Add `AverageDownsampleObservationOperator` (avg_pool2d) + transpose; register; add configs |
| E2 | **CRIT** | uDALES (Case 3) data + generator absent; channel-count inconsistency | Acquire/generate uDALES runs; place `.nc`+`mask.npz`; fix channels; mask solid cells everywhere |
| E3 | **CRIT** | No tidy results file / LaTeX table-filler | Define `case,method,scenario,metric,value,std,E,M,seed,NFE,seconds` schema + `make_tables.py` |
| E4 | **CRIT** | Three distinct samplers not separated (only one FM variant) | Wire SI-SDE/FM-SDE/FM-ODE as 3 rows; assert the 2 FM correctness checks |
| E5 | MAJ | 4 baselines absent (DPS, PF, SDA, ensemble score filter) | Implement each as `method/*.yaml` + posterior/likelihood |
| E6 | MAJ | EnKF off-pipeline (own JAX pkg, own truth/obs) | Wrap into benchmark sharing truth+obs+mask |
| E7 | MAJ | Spread–skill ratio (+`√((E+1)/E)`) absent | Add `spread_skill`; report `|1-ratio|` |
| E8 | MAJ | Rank/Talagrand histogram absent | Add computation + plot |
| E9 | MAJ | KL-at-points on fields absent (estimator exists, Case 1 only) | Reuse `kl_divergence.py` at obs/unobs points vs large-E reference |
| E10 | MAJ | NFE + wall-clock not logged | NFE counter on each net call; `perf_counter` per step |
| E11 | MAJ | Energy-spec RMSE not in paper's log-KE-spectrum form | Compute `√(mean_k(log Ek[x̄]-log Ek[x*])²)` |
| E12 | MAJ | Ablation knobs (full/JF/no `G_tau`, g sweep, M, E) not all exposed | Config switches + sweep driver |
| E13 | min | RMSE per-member not ensemble-mean; CRPS biased `1/2M²`; sliced-W1 labelled W2; masks unseeded; hardcoded `.to("cuda")`; NS generator/loader key mismatch; tiny committed `ensemble_size:4` | Per-item fixes in Phase 4 |

---

## 5. Unified gap-closing plan (dependency-ordered)

### Phase 0 — Foundation: the affine-path interface (Layer 1)
1. **L1** Add `FlowMatchingModel.score()` (general Eq. `fm_score`, in terms of `α,β,*_diff` — no rectified-flow hard-coding).
2. **L2/L5** Add one `velocity_score_coeff(t)` (`a_tau`) + a single shared velocity↔score helper on the interpolation classes; delete the 4 divergent copies. **All schedule-derived quantities must be general in `α,β,γ`** (per the quadratic-β decision — see §6).

### Phase 1 — The unified sampler (Layer 2)
4. **P1** Introduce `w_tau = a_tau + ½g²`; multiply corrected score by it; remove `correction_multiplier`.
5. **P5/P6/P4/P7/P10** Make the SI likelihood canonical: closed-form `Sbar` with detached covariances, `G_tau` separated from `w_tau`, full-vs-Jacobian-free flag, `τ≥dτ` guard, single combined EM step, fix loop bound.
6. **P2/P3/P11** Rebuild FM on the unified path: FM interpolant likelihood (`ȳ=βy`, `μ_s=-α²s`, FM gain), `diffusion_term`-parameterized `_one_step` (None⇒FM-ODE, callable⇒FM-SDE with `g∝√(αβ)`); retire `GuidanceGaussianLikelihood` to baseline status.
7. **P8/P9 / L7** Reconcile `DiffusionPosterior` (relabel or fix), default SI `resample=false`, add endpoint-vanishing `g_tau` + the 2 FM correctness asserts.

### Phase 2 — Make the headline results real (Layer 3 core)
8. **E4** Run all three samplers on Case 1 + Case 2.
9. **E1** Block-average super-resolution operator (unblocks 4 table column groups).
10. **E3** Tidy results schema + `make_tables.py` LaTeX emitter (unblocks every table).

### Phase 3 — Metrics to fill the tables
11. **E7** spread–skill, **E8** rank histogram, **E9** KL-at-points, **E10** NFE+wall-clock, **E11** energy-spec RMSE; fix RMSE (ensemble-mean) and CRPS estimator (E13).

### Phase 4 — Baselines
12. **E6** wrap EnKF into the shared benchmark; **E5** add bootstrap PF, DPS, then SDA + ensemble score filter.

### Phase 5 — Case 3 + ablations + hardening
13. **E2** acquire/place uDALES data, fix channels + solid-cell masking.
14. **E12** wire the four ablation axes.
15. **E13** seed+persist masks, seed list + mean±std over seeds, remove hardcoded CUDA, reconcile NS generator/loader, W1→W2.

---

## 6. Author decisions (resolved 2026-06-25)

- **SI schedule (L3): keep quadratic-β (`β=t²`); do not change the paper.** Consequence: every
  helper — `a_tau`, `A_tau`, source moments, `G_tau` — **must be implemented generally in terms of
  the `α,β,γ` schedules and their derivatives**, never hard-coded to rectified flow. This raises the
  bar on L1/L2/L5 (full generality is now a hard requirement, not a nicety) and removes the L3
  decision/retraining work.
- **Full vs Jacobian-free `G_tau` (L4/P5): implement BOTH, config-selectable.** Add a config switch
  (e.g. `gain: full | jacobian_free`) so either gain can be chosen per run; the full source-covariance
  Jacobian term must be built (enables the full-vs-JF ablation row).
- **uDALES (E2): author will provide the data.** Scope reduces to wiring the loader, solid-cell
  masking, channel-count fix, and metrics around the supplied `.nc`+`mask.npz`; no CFD generator
  needed in-repo.
- **Baseline scope (E5): decide after Phase 1** (once samplers + metrics work). Plan Phase 4
  accordingly; default working assumption is FlowDAS + EnKF + bootstrap PF + DPS first, SDA +
  ensemble score filter as a second wave.
