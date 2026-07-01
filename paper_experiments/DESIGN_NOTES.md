# Design notes — durable architecture & empirical rationale

Salvaged (2026-07-01) from the now-archived `CODE_AND_EXPERIMENTS_OVERVIEW.md` and
`PROJECT_HANDOFF.md`. This file keeps only the content that is **still accurate and
not captured elsewhere**: the code↔math mapping for `src/scisi`, the cost/accuracy
trade-off, and the empirical decisions behind the KL reference, the dropped gain,
and the true-solver forecast interval. For *how to run* see `results/README.md`
(layout) + `RUN_STATUS.md` (status); for the method math see the manuscript
(`sections/methodology.tex`, `sections/appendix_methods.tex`).

> Naming: the sampler formerly "FM-SDE" is now **DM-SDE** (diffusion-model-style SDE
> on the FM prior). The underlying class is still `FlowMatchingPosterior` and the
> prior is still `fm_model` — only the method label changed.

---

## 1. Code ↔ math map (`src/scisi/`)

### Affine Gaussian path — `models/interpolations.py`
`AffineGaussianPathMixin` (≈L34) is schedule-general; concrete classes set α, β, γ:

| class | α(t) | β(t) | σ/γ | used by |
|---|---|---|---|---|
| `LinearDeterministicInterpolation` (L145) | 1−t | t | — | FM (rectified flow) |
| `QuadraticStochasticInterpolation` (L302) | — | **t²** | γ(1−t), σ=γ√t | **trained SI prior** |

Key helpers (all clamp t off {0,1}):
- `velocity_score_coeff(t)` → `a_τ` (L57): `a_τ = σ·(β'σ − σ'β)/β`  (σ=γ√t for SI, σ=α for FM)
- velocity↔score duality (single source of truth), `score_from_velocity` (L110) / `velocity_from_score` (L76):
  ```
  s = (β v − β' x − (α'β − β'α) a0) / (β a_τ)
  v = (β'/β) x + (α' − β'α/β) a0 + a_τ s
  ```

### Priors
- **SI** `models/follmer_stochastic_interpolant.py` — `drift`=trained `b_θ` (L48); score recovery
  `_prior_score` (L209): `A = t·γ·(β'γ − βγ')`, `score = (β b_θ − c)/(A+1e-6)`, `c = β'x + (βα'−β'α)a0`.
- **FM** `models/flow_matching_model.py` — `drift`=trained `v_θ` (L43); `score` (L64) =
  `interpolation.score_from_velocity(x, v, t, a0=0)` (general in α,β; GAP L1, previously hard-coded to
  rectified flow). NS FM prior is a dedicated checkpoint (`flow_matching/`) + `LinearDeterministicInterpolation`.

### Posteriors — `posterior_models/`
All realize `b_post = b_prior + w_τ·G_τ·S̄`; the likelihood returns the corrected score, the posterior
multiplies by `w_τ = a_τ + ½g²`.
- **SI-SDE** `StochasticInterpolantPosterior` (L19): source a₀=x₀ (point mass), native `g_τ=γ_τ`, init at x₀;
  guidance starts at τ=Δτ (avoids the τ=0 singularity).
- **DM-SDE / FM-ODE** `FlowMatchingPosterior` (L43): a₀=0 (so ȳ_τ=β_τ y), latent init N(0,I).
  `diffusion_term=None` ⇒ FM-ODE; a callable ⇒ DM-SDE with the endpoint-vanishing schedule
  `endpoint_vanishing_diffusion` (L16): `g_τ = scale·√(α_τ β_τ)` (vanishes at endpoints so the lifted
  score stays finite as τ→1). DM-SDE prior drift is `v_θ + ½g²s`.

### Likelihood — `likelihood_models/gaussian_likelihood.py` (the heart of the method)
`InterpolantGaussianLikelihood` (L38) builds the closed-form interpolant-likelihood score:
```
Σ̄_τ = β_τ² R + H Σ_s Hᵀ                              (obs-space covariance)
S̄   = (Σ_s/σ_τ²) Hᵀ Σ̄_τ⁻¹ (ȳ_τ − μ̄_τ)               (covariances detached)
```
with obs-interpolant `ȳ_τ = α_τ H a₀ + β_τ y` and source moments (SI Wiener `ρ_τ=γ_τ²t`; FM Tweedie
`ρ_τ=α_τ²`, `μ_s=−α²s`). The full source-cov operator `_build_full_sigma_s_apply` (L≈610) applies the
network Jacobian as a JVP: SI `Σ_s = γ²t·I + γ⁴t²A_τ(βJ_b − β̇I)`; FM `Σ_s = α²·I + α⁴∇_x s`.

---

## 2. The likelihood-covariance modes (`likelihood_mode`)

| mode | Σ_s in the solve | gain G_τ | cost / pseudo-step | role |
|---|---|---|---|---|
| `inflated` | full, **per member** | I | `O(B·N_y)` JVPs | exact (ΠGDM); analytical gate |
| **`inflated_shared`** | full, **shared** (ensemble mean) | I | `O(N_y)` JVPs | tractable approx; field scale (`shared` variant) |
| `dps_jacobian_free` | isotropic ρI | I | `O(N_u N_y)`, **no JVP** | Corollary cheap drift; cheapest (`jacfree` variant) |
| `dps_full` | full, per member | `I + β⁻²Σ_s HᵀR⁻¹H` | `O(B·N_y)` JVPs | multiplicative-gain DPS surrogate — **DROPPED from the paper, off by default** |

Two implementation notes that were hard-won:
- **`_math_sdpa()` context manager** (L12): `torch.autograd.functional.jvp` (double-backward) has **no
  CUDA kernel** for flash/mem-efficient SDPA attention; both JVP closures must run under
  `sdpa_kernel(SDPBackend.MATH)`. Numerically identical, but a hard blocker for any full-Σ_s GPU run.
- **`inflated_shared`**: builds Σ_s **once at the ensemble-mean state** and forms the single `[N_y,N_y]`
  matrix via a chunked column-batched JVP (`_build_HSHt_shared`, L316, chunk 64). It **collapses exactly
  to `inflated` when ensemble members coincide** — verified to 3e-15 (float64), chunk-invariant, incl.
  FM path + non-square super-res operator.

### 2.1 Cost / accuracy trade-off (measured, sparse 1.5625%, N_y=256)

| mode | rmse | crps | s/assim-step | scaling |
|---|---|---|---|---|
| `dps_jacobian_free` | 0.72 | 0.44 | ~1 | `O(N_u N_y)`, no JVP — cheap, full-scale OK |
| exact `inflated` | (blows up at M=4) | — | 114 | `O(B·N_y)` JVPs — **days/cell, intractable** |
| **`inflated_shared`** | **0.137** | **0.081** | 85 (E=8, M=10) | `O(N_y)` JVPs — ~hours/cell |

Takeaways: (1) the inflated covariance is dramatically more accurate on sparse obs (0.137 vs 0.72);
(2) exact `inflated` is intractable at NS scale (`B·N_y·M·n_assim ≈ 6.5×10¹⁰` UNet-equiv/cell — batching
can't close a constant factor); `inflated_shared` removes the B factor (~64× at E=64) → hours/cell.
This is why the two "Ours" grid variants are `jacfree` (headline, cheap) and `shared` (field-tractable
inflated); exact per-member `inflated` is analytical-gate only.

---

## 3. Why the KL reference is the non-localized E=1000 EnKF (empirical, 2026-06-27)

Assessed EnKF/LETKF/PF/EnSF at **E=1000** as the ground-truth posterior (final-step, sparse):

| candidate | sparse 5% RMSE | spread/skill | sparse 1.5625% RMSE | verdict |
|---|---|---|---|---|
| **EnKF non-localized (global)** | **0.081** | 0.85 | **0.083** | **WINNER: accurate + calibrated** |
| EnKF distance-localized (r=20) | 0.242 | 0.84 | 0.626 | localization HURTS the mean 3–8× at E=1000 |
| Particle filter | 0.750 | 0.00 | 0.853 | collapsed (ESS≈1) — unusable |
| Ensemble score filter | NaN | — | NaN | diverges (1e12) — unusable as-is |

- **Reference = non-localized global EnKF, E=1000, inflation=1.0** (Evensen-2024: at large E global updates
  avoid divergence; localization is a small-ensemble fix and here *degrades* accuracy 3–8×). Accurate AND
  calibrated at convergence (final-step spread/skill 0.85 / 0.78) — no inflation needed.
- **Inflation WRECKS it** (sweep, sparse 5%): RMSE 0.081 (1.0) → 0.169 (1.3) → 0.544 (1.6) → **2.409 (2.0)**.
- **LETKF cannot run at E=1000** — per-grid-point transform OOMs (~200 GB); reduced E (≤256) only.

These are the `results/navier_stokes/reference/traj<N>/gt/` files (see `results/README.md`).

---

## 4. Two more decisions worth keeping

- **The multiplicative gain `G_τ` was dropped** (author decision). `G_τ = I + β⁻²Σ_s HᵀR⁻¹H` (mode
  `dps_full`) did **not** help: analytical KL **0.001** (inflated, G=I) vs **0.174** (gain) vs 0.104 (cheap);
  NS sparse-1.5625% rmse **0.155** (inflated_shared) vs 0.716 (cheap). **Accuracy comes from inflating the
  covariance, not the gain.** `dps_full`/`_apply_gain` remain in code but OFF by default.
- **True-solver forecast interval (EnKF/PF).** The jax-cfd filters must advance **one full training
  interval** per assimilation step: `INNER_STEPS = REDUCED_DT/HF_DT = 5000` (an earlier 50×-too-short
  interval made the classical numbers invalid). The solver runs at 256², **stride-2 subsampled to 128²** so
  obs points match the torch operator (verified 7e-7), preprocesser normalization **std = 3.09969**. jax on
  GPU by default (`ENKF_JAX_PLATFORM`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`). Localization is incoherent for
  block-average super-res → run EnKF non-localized there.

---

## 5. Gotchas (still true)

- Always `.venv/bin/python` (bare python has no torch).
- Full-Σ_s JVPs must run under the math SDPA backend on CUDA (handled by `_math_sdpa`).
- `num_physical_steps > len_field_history (=5)` or the assimilation loop is empty → all-NaN metrics
  (the grid uses 20).
- The NS/urban drivers ignore Hydra's `method=`/`scenario=`; restrict with `+ns_methods` / `+ns_scenarios`
  (mind the quoting for spaces/parens: `+ns_methods=["Ours (SI-SDE)"]`).
- `likelihood_mode` is a global (top-level) override. The two Ours modes ARE distinguished in the tidy
  output now, via the `variant` column (jacfree/shared) — compare them directly, not only in an ablation.
