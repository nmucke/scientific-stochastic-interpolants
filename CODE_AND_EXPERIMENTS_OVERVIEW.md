# Implementation & Experiments Overview

> ## Session update — 2026-06-29
>
> This reference predates the final method lineup, the baseline audit, and the gain
> removal. Read `PROJECT_HANDOFF.md` (its 2026-06-29 section) first. Corrections:
> - **Final lineup (9 generative + classical):** Ours SI-SDE / FM-ODE / FM-SDE
>   ("FM-SDE (DM)"); FlowDAS; Guided FM (FIG); Guided FM (OT-ODE); D-Flow SGLD; SDA;
>   SURGE. Classical (NS only): EnKF (E=1000 non-loc = ground-truth/KL reference;
>   E=64 loc = baseline), LETKF, particle filter, ensemble score filter. The legacy
>   "Guided FM" (one-step DPS-on-flow) and "Guided diffusion" (DPS) are **DROPPED**.
> - **The multiplicative gain `G_τ` is dropped** — there are no longer "four modes"
>   for accuracy; the axis is the covariance (`inflated` / `inflated_shared` vs
>   isotropic Jacobian-free). `dps_full`/`G_τ` are kept off-by-default in code only.
> - **New code:** `posterior_models/{dflow_posterior,surge_posterior}.py`,
>   `likelihood_models/dflow.py`, `likelihood_models/guidance.py`
>   (`FIGGaussianLikelihood` + `weighting="ot_ode"`), and
>   `models/diffusion_model.py::DenoiseDiffusionModel.from_flow_matching`.
> - **SDA/SURGE diffusion prior is built from the FM model** (`diffusion_from_fm:
>   true`); the FM prior is its own trained checkpoint.
> - **Analytical case DONE** (all 11 methods as closed-form samplers; real numbers
>   below in §2.3 are updated). **NS + urban headline runs PENDING the GPU.** Urban is
>   generative-only, sparse-only, no KL/energy.
> - **Three baseline bugs fixed + validated** on the analytical case: FlowDAS, D-Flow
>   SGLD, SDA (details in `PROJECT_HANDOFF.md` / `RUN_STATUS.md`).

> **Audience:** the author (internal reference). A full, self-contained overview of
> *what the code does* and *how the experiments are set up*, for the paper
> *"A General Observation-Interpolant Method for Data Assimilation with Flow-Based
> Generative Models."*
>
> **Branch:** `sync-with-paper`. **Scope:** the `src/scisi/` library (Part 1) and
> the `paper_experiments/` harness (Part 2). Code references are `file:line`
> (line numbers are approximate after edits — search by symbol name if they drift).
>
> Companion docs: `paper_new/GAP_ANALYSIS.md` (the why), `paper_new/EXPERIMENTS_IMPLEMENTATION_SPEC.md`
> (the spec), `paper_experiments/HANDOFF_GPU.md` (GPU run guide).

---

## 0. The one-paragraph summary

The method samples from a Bayesian data-assimilation posterior by running a *unified*
family of guided generative samplers. A prior is a flow-based generative model
(stochastic interpolant or flow matching) viewed as an **affine Gaussian probability
path** `x_τ = α_τ a₀ + σ_τ z + β_τ x₁`. Conditioning that path on a linear-Gaussian
observation `y = H x₁ + η` yields a posterior drift of the single form

```
b_post = b_prior + w_τ · S̄ ,      w_τ = a_τ + ½ g_τ²
```

instantiated as three samplers — **SI-SDE**, **FM-SDE**, **FM-ODE** — that differ only
in the diffusion `g_τ`, the weight `w_τ`, and the source. `S̄` is a closed-form
**interpolant-likelihood score** built from an *observation interpolant*
`ȳ_τ = α_τ H a₀ + β_τ y`. (The multiplicative gain `G_τ` was DROPPED — see the
2026-06-29 update; the accuracy axis is the covariance, not a gain.) The likelihood
covariance is config-selectable (`inflated` / `inflated_shared` / isotropic
Jacobian-free), trading accuracy vs cost (Section 1.5). The experiments (Part 2)
evaluate this on a closed-form analytical case, stochastic Navier–Stokes, and urban
airflow.

---

# PART 1 — The `src/scisi/` library

## 1.1 Architecture: three layers

```
Layer 1  models/                  affine Gaussian path interface + prior models
            interpolations.py        schedules α,β,γ; a_τ; velocity↔score duality
            follmer_stochastic_interpolant.py   SI prior (drift, score recovery)
            flow_matching_model.py              FM prior (drift, score recovery)

Layer 2  posterior_models/        the unified posterior sampler (the method)
            base_posterior.py        sampling loop
            stochastic_interpolant_posterior.py   Sampler 1 (SI-SDE)
            flow_matching_posterior.py            Samplers 2 & 3 (FM-SDE / FM-ODE)
         likelihood_models/        the observation likelihood + obs operators
            gaussian_likelihood.py   InterpolantGaussianLikelihood (4 modes) + FlowDAS
            observation_operators.py LinearObservationOperator (grid/random/super-res)

Layer 3  sampling/   solvers (Euler–Maruyama, Heun, Euler)
         metrics/    RMSE, KE-spectrum RMSE, CRPS, spread-skill, KL-at-points, NFE, time
         architectures/  UNet + attention   data/  datasets   bin/  train/infer entrypoints
```

Dependency chain: experiments need the samplers; the samplers need the priors to expose
`a_τ` and the score recovery. Everything is **general in `(α,β,γ)`** and their derivatives
— never hard-coded to rectified flow — because the trained SI model uses a **quadratic-β
schedule** (`β=τ²`).

## 1.2 Affine Gaussian path interface — `models/interpolations.py`

`AffineGaussianPathMixin` (≈L34) provides the schedule-general identities; concrete
classes set the schedules:

| class | α(t) | β(t) | σ/γ | used by |
|---|---|---|---|---|
| `LinearDeterministicInterpolation` (L145) | 1−t | t | — | FM (rectified flow) |
| `QuadraticDeterministicInterpolation` (L183) | 1−t | t² | — | — |
| `LinearStochasticInterpolation` (L221) | — | t | γ(1−t), σ=γ√t | — |
| `QuadraticStochasticInterpolation` (L302) | — | **t²** | γ(1−t), σ=γ√t | **trained SI prior** |

Key schedule-general helpers (all clamp `t` away from {0,1}):

- **`velocity_score_coeff(t)`** → `a_τ` (the deterministic part of `w_τ`), L57:
  ```python
  a_τ = σ * (β' σ − σ' β) / β        # σ = γ√t for SI, σ = α for FM
  ```
- **`score_from_velocity(x, v, t, a0)`** (L110) and **`velocity_from_score(x, s, t, a0)`** (L76):
  the single source of truth for the velocity↔score duality
  ```python
  s = (β v − β' x − (α'β − β'α) a0) / (β a_τ)            # score_from_velocity
  v = (β'/β) x + (α' − β'α/β) a0 + a_τ s                  # velocity_from_score
  ```

## 1.3 Prior models

**SI — `models/follmer_stochastic_interpolant.py`** (`FollmerStochasticInterpolant`, L19).
`drift(x,t,field_history,…)` returns the trained drift `b_θ` (L48). Score recovery
`_prior_score` (L209) uses the `A_τ` coefficient:
```python
A = t·γ·(β'γ − βγ');  A = 1/(A + 1e-6)
c = β'·x + (β α' − β' α)·a0
score = A·(β·b_θ − c)
```

**FM — `models/flow_matching_model.py`** (`FlowMatchingModel`, L19). `drift` (L43) returns
the trained velocity `v_θ`; **`score(x,t,…)`** (L64) is the general fm-score recovery
(GAP L1 — previously hard-coded to rectified flow):
```python
v = drift_model(x,t,…)
score = interpolation.score_from_velocity(x, v, t, a0=0)   # general in α,β
```
The NS FM prior is a *dedicated* trained checkpoint (`flow_matching/`), built from its own
`config.yaml` and paired with `LinearDeterministicInterpolation`.

## 1.4 Posterior samplers — `posterior_models/`

All samplers realize `b_post = b_prior + w_τ · G_τ · S̄`. The `likelihood_model.score(...)`
returns the corrected score (`S̄` for inflated/jacobian-free, `G_τ S̄` for dps_full); the
posterior multiplies by the weight `w_τ`.

**SI-SDE — `StochasticInterpolantPosterior`** (L19). Source `a₀ = x₀` (point mass), native
diffusion `g_τ = γ_τ`, initialised exactly at `x₀`. One Euler–Maruyama step (L82):
```python
drift = model.drift(base, t, …)
if t ≥ MIN_TIME:                      # guidance starts at τ = Δτ (avoids the τ=0 singularity)
    corrected_score, loglik = likelihood_model.score(obs, base, t, drift=drift, …)
    w_τ = a_τ(t) + ½ g(t)²
else: corrected_score, w_τ = 0, 0
base += (drift + w_τ · corrected_score)·dt + g(t)·√dt·z
```

**FM-SDE / FM-ODE — `FlowMatchingPosterior`** (L43). `a₀ = 0` (so `ȳ_τ = β_τ y`), latent
init `N(0,I)` (exact: `Φ₀` is constant). `diffusion_term=None` ⇒ FM-ODE (deterministic);
a callable ⇒ FM-SDE with the **endpoint-vanishing** schedule
`endpoint_vanishing_diffusion` (L16): `g_τ = scale·√(α_τ β_τ)` — vanishes at the endpoints
so the lifted score stays finite as `τ→1` (asserted in `_one_step`, L115). FM-SDE prior
drift is `v_θ + ½ g² s`.

## 1.5 Likelihood model — `likelihood_models/gaussian_likelihood.py` ★

`InterpolantGaussianLikelihood` (L38) is the heart of the method and the most important file
to understand. It builds the closed-form interpolant-likelihood score

```
Σ̄_τ  = β_τ² R + H Σ_s Hᵀ                          (likelihood covariance in obs space)
S̄    = (Σ_s / σ_τ²) Hᵀ Σ̄_τ⁻¹ (ȳ_τ − μ̄_τ)          (interpolant score, covariances detached)
```

from the **observation interpolant** `ȳ_τ = α_τ H a₀ + β_τ y` (`_interpolate_observations`,
L239) and the **source moments** `μ_s, ρ_τ` (`_source_mean_si/_fm`, `_source_cov_diag_si/_fm`,
L256–288): SI uses Wiener moments (`ρ_τ = γ_τ² t`), FM uses Tweedie moments (`ρ_τ = α_τ²`,
`μ_s = −α² s`). `μ̄_τ = H x − H μ_s`.

### The FOUR `likelihood_mode` values

| mode | `Σ_s` in the solve | gain `G_τ` | cost / pseudo-step | role |
|---|---|---|---|---|
| `inflated` | full, **per member** | I | `O(B·N_y)` JVPs | exact (PiGDM); **analytical gate** |
| **`inflated_shared`** ★new | full, **shared** (ensemble mean) | I | `O(N_y)` JVPs | tractable approx; **NS full-scale** |
| `dps_full` | full, per member | `I + β⁻²Σ_s HᵀR⁻¹H` | `O(B·N_y)` JVPs | paper's multiplicative-gain DPS surrogate |
| `dps_jacobian_free` | isotropic `ρI` | I | `O(N_u N_y)`, **no JVP** | Corollary cheap drift; cheapest |

The mode flags are set in `__init__` (≈L150): `use_full_sigma_s ∈ {inflated, inflated_shared,
dps_full}`, `apply_gain = (dps_full)`, `share_sigma_s = (inflated_shared)`. Legacy keys
(`gain: full/jacobian_free`, `correct_likelihood_score`) still map for back-compat.

`_interpolant_score` (L346) branches on the mode:
```python
if sigma_s_apply is None:                    # dps_jacobian_free: isotropic Σ_s = ρI
    Σ̄ = β²R·I + ρ·(H Hᵀ);  solve once (shared across ensemble)
elif self.share_sigma_s:                     # inflated_shared
    HSHt = self._build_HSHt_shared(x, sigma_s_apply)    # single [N_y,N_y]
    Σ̄ = β²R·I + HSHt;  solve for all B member RHS
    return sigma_s_apply(Ht_sol) / σ_τ²
else:                                         # inflated / dps_full: per member
    HSHt = self._build_HSHt(x, sigma_s_apply)           # [B,N_y,N_y]  ← N_y JVPs/member
    Σ̄ = β²R·I + HSHt;  solve per member
```

### The full source-covariance operator (the expensive Jacobian)

`_build_full_sigma_s_apply` (L≈610) returns `v ↦ Σ_s v`, where `Σ_s` carries the network
Jacobian applied as a JVP:
- SI: `Σ_s = γ²t·I + γ⁴t²A_τ(β J_b − β̇ I)`, `J_b = ∇_x b_θ`
- FM: `Σ_s = α²·I + α⁴ ∇_x s`

The JVP uses `torch.autograd.functional.jvp` (double-backward). A `_bcast` helper broadcasts
the captured state/conditioning to the tangent batch — a **no-op on the per-member path**, and
the enabler of column-batching and the shared-state path:
```python
def jvp_fn(v):
    k = v.shape[0]
    xk, tk, fhk = _bcast(x,k), _bcast(t_net,k), _bcast(field_history,k)
    with _math_sdpa():                       # ★ see below
        _, jv = torch.autograd.functional.jvp(lambda inp: self.model.drift(inp,tk,fhk,…), (xk,), (v,))
    return jv.detach()
```

### ★ Changes made this session (in this file)

1. **`_math_sdpa()` context manager** (L12). `torch.autograd.functional.jvp` uses the
   double-backward trick, but on **CUDA** the flash/mem-efficient SDPA attention kernels have
   **no double-backward** (`derivative for aten::_scaled_dot_product_efficient_attention_backward
   is not implemented`). The math SDPA backend supports it (it is what CPU silently used). Both
   JVP closures now run under `sdpa_kernel(SDPBackend.MATH)`. Numerically identical; this was a
   hard blocker for any full-Σ_s run on GPU.

2. **`inflated_shared` mode** — the tractable approximation of exact `inflated`:
   - `__init__`: new `share_sigma_s` flag + `shared_sigma_chunk = 64`.
   - `score()`: when sharing, build `Σ_s` **once at the ensemble-mean state** (mean of `x` and of
     the conditioning; pseudo-time `τ` is identical across the ensemble so the schedule scalars are
     unchanged):
     ```python
     sigma_s_apply = self._build_full_sigma_s_apply(
         x=_mean0(x), t=t_grid[:1], t_net=t_net[:1],
         field_history=_mean0(field_history), …, rho=rho[:1])
     ```
   - `_build_HSHt_shared` (L316): builds the single `[N_y,N_y]` matrix via a **chunked
     column-batched JVP** (chunk 64) instead of an `N_y`-long Python loop:
     ```python
     for start in range(0, N_y, chunk):
         block = cols[start:start+chunk]                    # [k,C,H,W]
         HSHt[:, start:start+k] = obs_operator(sigma_s_apply(block)).transpose(0,1)
     ```
   - **Why:** exact `inflated` does `B·N_y` network-Jacobian evals/step (days/cell at NS scale —
     intractable, see Part 2 §2.8). Sharing drops the `B` factor (~64× at E=64). It **collapses
     exactly to `inflated` when the ensemble members coincide** — the basis of the numerical test.
   - **Verification:** `verify_inflated_shared.py` — fully-identical ensemble ⇒ `inflated_shared
     == inflated` to **3e-15** (float64); chunk-size invariant to 0; independent review +
     extra tests (FM path, non-square super-res operator, distinct-member-vs-reference) all pass
     at machine precision.

3. **`_bcast` broadcasting** added to `_build_full_sigma_s_apply` so one code path serves both the
   per-member (no-op) and shared (batch-1→K expand) cases.

`FlowdasGaussianLikelihood` (L≈713) is the FlowDAS baseline (Monte-Carlo: draw one-step `x₁`
predictions, softmax-weight by `N(y; H x₁, R)`) — *not* the paper's method.

## 1.6 Observation operators — `likelihood_models/observation_operators.py`

`LinearObservationOperator` (L162) with three `type`s, all exposing `forward` (`H@x`, L326) and
`transpose` (`Hᵀ@y`, adjoint, L344), both batched over the leading dim:

| type | builder | `N_y` (128² grid) | scenario |
|---|---|---|---|
| `grid` | `get_grid_observation_matrix` (L8) | per spacing | — |
| `random` | `get_random_observation_matrix` (L39) | 819 (5%), 256 (1.5625%) | sparse; **seeded mask** |
| `super_res`/`avg_pool` | `get_block_average_observation_matrix` (L78) | 1024 (32²), 256 (16²) | super-res; **block-average**, adjoint spreads `1/f²` |

The device-resident matrix is cached (`_matrix_on`) so the hot path does no per-call CPU→GPU
copy. `save_mask`/`load_mask` (L255/272) persist a fixed sparse mask for cross-method sharing.

## 1.7 Solvers — `sampling/`

`sde_solvers.py`: `euler_maruyama_step` (L8), `heun_step` (L26, ODE-restricted per GAP L9).
`ode_solvers.py`: `euler_step` (L7). The case pipeline maps `stepper: "sde"|"ode"` to these.

## 1.8 Metrics — `metrics/`

| metric | function (file) | definition |
|---|---|---|
| ensemble-mean RMSE | `ensemble_mean_rmse` (accuracy.py L40) | `√mean_i (x̄_i − x*_i)²`, `x̄ = mean_e` |
| KE-spectrum RMSE | `radial_kinetic_energy_spectrum` (spectral.py L53) | RMSE of log-E(k) on a fixed radial grid |
| CRPS | `crps` (calibration.py L17) | unbiased pairwise estimator |
| spread–skill | `spread_skill` (calibration.py L68) | `√((E+1)/E)·spread` vs skill; report `|1−ratio|` |
| KL-at-points | `kl_at_points` (distributional.py) | 1-D marginal KL at observed/unobserved points vs reference |
| sliced-W2 | `sliced_wasserstein_w2` (distributional.py) | avg 2-Wasserstein over random 1-D projections |
| NFE / seconds | `NFECounter`, `StepTimer` (cost.py L19/57) | net-eval count / wall-clock, per assimilation step |

## 1.9 Architectures / data / training

`architectures/u_net.py` — `UNet` (L66) with conditioning + attention bottleneck;
`architectures/attention.py` — `Attention` (L119) via `torch.nn.functional.scaled_dot_product_attention`
(the kernel behind the `_math_sdpa` fix). `data/datasets.py` — `StochasticNavierStokesDataset`
(L20) yields sliding `(history, target)` windows. `bin/main_train.py` / `bin/main_posterior.py`
— Hydra train / inference entrypoints. **No FM training needed** — both NS priors are trained.

## 1.10 Session change-set (summary)

| file | change | why |
|---|---|---|
| `likelihood_models/gaussian_likelihood.py` | `_math_sdpa()` context manager around both JVPs | CUDA double-backward through flash/efficient SDPA is unimplemented |
| `likelihood_models/gaussian_likelihood.py` | `inflated_shared` mode (shared-Jacobian + chunked column JVP + single solve) + `_bcast` | exact `inflated` is `O(B·N_y)` ⇒ intractable at NS scale |
| `configs/benchmark.yaml` | top-level `likelihood_mode: null` | `likelihood_mode=…` CLI override (run_gpu_ns.sh) hit a Hydra struct error |
| `cases/navier_stokes/driver.py` | full-ablation `inflated` → `inflated_shared` | the no-correction/sweep ablation points were using the intractable exact mode |

---

# PART 2 — The experiments (`paper_experiments/`)

## 2.1 Harness core

- **`run.py`** — Hydra entrypoint. Dispatches `case` → runner (`analytical`/`navier_stokes`/
  `urban`, L36). `ablation=true` calls `run_ablation()` instead of `run()` (L73). Resolves the
  results path against the original CWD (Hydra chdir-safe).
- **`common/runner.py`** — `ExperimentRunner` base. `run()` loops `methods × scenarios × seeds`,
  calls `evaluate(ctx)`, aggregates. `RunContext` carries `(case, method, scenario, seed, E, M,
  extra)`. `_cfg_get(key, default)` reads a top-level config key (this is how `likelihood_mode`,
  `ensemble_size`, `num_steps`, `ns_methods`, `ns_scenarios` overrides are read).
- **`common/seeding.py`** — `SEED_LIST = (0,1,2,3,4)`; `derive_seed` (SHA256-stable);
  `obs_seed(case,scenario,test,seed)` for obs noise; `mask_seed(case,scenario)` for the sensor
  mask (shared across methods/seeds). Guarantees identical truth+obs+mask per scenario across all
  methods (spec §9).
- **`common/aggregation.py`** — `aggregate_over_seeds` groups by `(case, method, scenario, metric,
  E, M)` → mean `value`, std `std`, `seed=-1`.
- **`results_schema.py`** — the tidy `ResultRecord`: columns `case, method, scenario, metric,
  value, std, E, M, seed, NFE, seconds`. **`likelihood_mode` is NOT a column** — so the mode
  comparison lives in the *ablation* (encoded via `scenario` tags), not the headline. Enum string
  values are the table keys, e.g. `Method`: `"Ours (SI-SDE)"`, `"Ours (FM-SDE)"`, `"Ours (FM-ODE)"`,
  `"FlowDAS"`, …; `Scenario`: `"32^2->128^2"`, `"16^2->128^2"`, `"sparse 5%"`, `"sparse 1.5625%"`,
  `"analytical"`; `Metric`: `rmse, energy_spec_rmse, crps, spread_skill, kl_points, sliced_w2,
  nfe, seconds, rmse_velocity, rmse_temperature`.

## 2.2 Config system — `configs/`

`benchmark.yaml` defaults (`case: navier_stokes`, `method: si_sde`, `scenario: superres_32`),
plus top-level `seeds`, `ensemble_size` (E=64), `num_steps` (M=50), and **`likelihood_mode: null`**
(null ⇒ each method's own default; override to force one mode across all methods).

**Case configs:**
- `analytical.yaml` — `dimensions [2,10,100]`, `plot_dimension 2`, `obs_variance/prior_variance 1.0`,
  `diffusion_base 1.0`, `n_eval_samples 4096`, `likelihood_mode: inflated`.
- `navier_stokes.yaml` — `checkpoints.si_run: stochastic_interpolant_small`,
  `checkpoints.fm_run: flow_matching`, `require_weights` (false laptop / **true GPU**), `device`,
  `test_sample_indices [1..5]`, `ensemble_size 64`, `num_physical_steps 25`, `num_steps 50`,
  `variance 0.0025` (σ=0.05 vorticity), `reference_ensemble_size 1024`. **Constraint:**
  `num_physical_steps > len_field_history (=5)` so `n_assim = num_physical_steps − L > 0` (else the
  loop is empty and all metrics are NaN — a real footgun hit during sanity).
- `urban.yaml` — stub; per-variable variance, channels `[u,v,T]`, author-provided `.nc` + `mask.npz`.

**Method configs** (`_target_` + `stepper` + `likelihood_model` + `posterior_model`):
`si_sde` (SDE, `g=γ`, `InterpolantGaussianLikelihood model_class=si`, `StochasticInterpolantPosterior`),
`fm_sde` (SDE, endpoint-vanishing `g`, `model_class=fm`, `FlowMatchingPosterior`),
`fm_ode` (ODE, `g=0`, `FlowMatchingPosterior`), `flowdas`. The baselines
`guided_fm_fig`, `guided_fm_otode`, `dflow_sgld`, `surge`, `sda`, `enkf`,
`particle_filter`, `ensemble_score_filter` are all implemented (2026-06-29; LETKF is
dispatched in the driver rather than via its own method yaml). The
legacy `guided_diffusion` (DPS) config is retained but dropped from the paper lineup.

**Scenario configs:** `superres_32` (`super_res low=32 high=128`, N_y=1024), `superres_16` (N_y=256),
`sparse_5` (`random percent=0.05`, N_y=819), `sparse_1p5` (N_y=256).

## 2.3 Case 1 — analytical linear–Gaussian (`cases/analytical/`)

Closed-form correctness probe, no training. `samplers.py` defines `GaussianSystem` (exact posterior
moments + samples), the SI/FM prior drifts/scores in closed form, and the three likelihood-guidance
modes. `driver.py` draws truth+obs, runs each sampler, and reports **KL** and **sliced-W2** to the
exact posterior (4096 samples). **This case is DONE with real numbers** (`results/analytical_results.csv`):

KL→exact (mean over 5 seeds): SI-SDE 0.0009, FM-SDE 0.0016, FM-ODE 0.0011; FlowDAS 0.080; Guided FM
(OT-ODE) 0.0021; D-Flow SGLD 0.079; SDA 0.019; SURGE 0.0021; EnKF 0.0012; PF 0.0030; Guided FM (FIG)
collapsed (KL degenerate; sliced-W2 0.733).

Key finding: **with faithful baseline implementations essentially every method recovers the
linear-Gaussian posterior** — this case is an exactness check (the baselines are strong, not
strawmen); methods separate on the nonlinear fluid cases. FIG is faithful but structurally collapses
on full noisy observation. (The OLD table here — FlowDAS 0.299, Guided FM 0.104, Guided diffusion
0.174 — is obsolete; it predates the baseline bug fixes and the dropped DPS baselines.)

## 2.4 Case 2 — stochastic Navier–Stokes (`cases/navier_stokes/`)

The main benchmark. `driver.py` (`NavierStokesRunner`) drives its **own** method×scenario grid
(CLI `method=`/`scenario=` are ignored; use `+ns_methods=[…]`/`+ns_scenarios=[…]` to restrict).
`_ns_pipeline.py` does the heavy lifting:

- **`load_prior`** — loads SI from `checkpoints.si_run` and FM from `checkpoints.fm_run` (FM model
  built from its *own* `config.yaml`; falls back to reusing the SI drift only if `fm_run` is null).
  `require_weights=true` hard-fails on a missing `model.pth`. **Verified:** both real checkpoints
  load cleanly, including the previously-unexercised FM `load_state_dict`. (Note: SI and FM
  checkpoints use *different* UNet sizes — SI hidden `[8,16,32,64]`/emb 128, FM `[16,32,64,128]`/
  emb 256 — a fairness caveat for the SI-vs-FM comparison, spec §5.)
- **`build_obs_operator`** — scenario → operator; sparse masks seeded via `mask_seed`.
- **`prepare_truth_and_obs`** — normalises a test trajectory, builds field history (L frames),
  truncates to `num_physical_steps`, draws `y = Hx + η` (seeded via `obs_seed`).
- **`build_posterior`** — `(method, likelihood_mode)` → `(model, posterior, stepper)`.
- **`run_assimilation`** — autoregressive loop over `n_assim = num_physical_steps − L` steps;
  attaches an NFE counter; returns `[E,C,H,W,T]` ensemble + per-step NFE/seconds
  (`seconds_per_step = elapsed / n_assim`).
- **`build_reference_trajectory`** — the large-E reference ensemble for KL-at-points, drawn once
  per `(scenario, seed)` by SI-SDE at the run's `likelihood_mode` (cached).
- **`compute_metrics`** — RMSE, KE-spectrum RMSE, CRPS, spread-skill, KL-at-points, NFE, seconds.

**Wired methods (2026-06-29):** SI-SDE, FM-SDE, FM-ODE, FlowDAS, Guided FM (FIG), Guided FM (OT-ODE),
D-Flow SGLD, SDA, SURGE + the classical filters (EnKF, LETKF, PF, EnSF). Headline runs PENDING the GPU.
**Scenarios:** main `{32²→128², sparse 5%}`, appendix `{16²→128², sparse 1.5625%}`.

**Ablation** (`run_ablation`/`evaluate_ablation`) fills `tab:ablation` on FM-SDE. The gain axis was
recast to a **covariance axis** (`inflated_shared` vs isotropic Jacobian-free — the multiplicative gain
`dps_full` is dropped from the paper, off-by-default in code); plus `gdiff` (g=0/0.5/1), `steps`
(M=10/50/100), `ensemble` (E=16/64/256).

## 2.5 Case 3 — urban airflow (`cases/urban/`)

**Implemented; headline runs pending the GPU (2026-06-29).** Data is **4-channel `(u, v, w, thl)`**
(NOT 3). Generative-only (no true solver → no conventional filters), **sparse 5% + sparse 1.5625% only
(no super-res)**, **no KL** (no ground-truth posterior) and **no energy/enstrophy** — only per-variable
RMSE + split CRPS + spread-skill against the ground-truth state. Solid-cell masking is applied in the
obs operator AND all metrics. `urban.yaml`: si_run=stochastic_interpolant_big_gamma1,
fm_run=flow_matching_big, diffusion_from_fm=true, test sims 170–178. Reuses the NS driver pattern.

## 2.6 Tables & figures — `make_tables.py`

Maps tidy `(method, scenario, metric)` triples to LaTeX cells via `TABLE_SPECS`:

| `tab:*` (results.tex) | rows × columns |
|---|---|
| `tab:analytical_results` | ours+baselines × {KL, sliced-W2} @ analytical |
| `tab:ns_accuracy` | methods × {rmse, energy_spec_rmse, kl_points} × {32², 5%} |
| `tab:ns_calibration_cost` | methods × {crps, spread_skill} × {32², 5%} + {nfe, seconds} |
| `tab:urban_accuracy` / `tab:urban_calibration_cost` | urban analogues |
| `tab:ablation` | FM-SDE × ablation tags × {rmse, crps, spread_skill} |

Emits the `tabular` **body** (rows + `\midrule`) to `generated/tab_*.tex` for the paper to
`\input`. `--demo` proves the pipeline with synthetic data.

## 2.7 Run commands

```bash
PY=.venv/bin/python   # torch 2.9.1+cu128, CUDA

# Headline (cheap, tractable mode) — driver runs its full grid; one invocation:
$PY paper_experiments/run.py case=navier_stokes seeds="[0,1,2]" \
    likelihood_mode=dps_jacobian_free ensemble_size=64 num_steps=50 \
    case.reference_ensemble_size=128 case.require_weights=true case.device=cuda

# Restrict to one cell (sanity): +ns_methods=["Ours (SI-SDE)"] +ns_scenarios=["sparse 1.5625%"]
#   NB: num_physical_steps must exceed len_field_history (5): case.num_physical_steps=8

# Ablation (mode comparison incl. inflated_shared):
$PY paper_experiments/run.py case=navier_stokes ablation=true ablation_smoke=false \
    ablation_scenario="sparse 5%" ensemble_size=… num_steps=… case.require_weights=true case.device=cuda

# Tables:
head -1 paper_experiments/results/analytical_results.csv > paper_experiments/results/all_results.csv
tail -q -n +2 paper_experiments/results/*_results.csv  >> paper_experiments/results/all_results.csv
$PY paper_experiments/make_tables.py --results paper_experiments/results/all_results.csv
```

## 2.8 Practical findings on cost (important)

Measured on this GPU (sparse 1.5625%, N_y=256):

| mode | rmse | crps | s/assim-step | scaling |
|---|---|---|---|---|
| `dps_jacobian_free` | 0.72 | 0.44 | ~1 | `O(N_u N_y)`, no JVP — cheap, full-scale OK |
| exact `inflated` | (blows up at M=4) | — | 114 | `O(B·N_y)` JVPs — **days/cell, intractable** |
| **`inflated_shared`** | **0.137** | **0.081** | 85 (E=8,M=10) | `O(N_y)` JVPs — ~hours/cell |

Two headline takeaways:
1. **The inflated covariance is dramatically more accurate** (RMSE 0.137 vs 0.72 for the cheap
   mode) — consistent with the analytical finding. This is the central evidence for the deferred
   inflated-vs-DPS decision.
2. **Exact `inflated` is intractable at NS scale** — the FLOPs are `B·N_y·M·n_assim ≈ 6.5×10¹⁰`
   UNet-equivalents/cell. Batched-JVP alone can't close that (constant factor). `inflated_shared`
   removes the `B` factor (shared Jacobian) → ~hours/cell, runnable at reduced scale. A further
   `O(N_y)→O(k)` matrix-free (CG/GMRES) cut is possible if minutes/cell is needed, but the
   non-symmetric `Σ_s` makes it more fragile.

**Practical plan:** headline tables in `dps_jacobian_free` (full scale, the paper's presented
method); the inflated covariance studied via `inflated_shared` in the ablation / at reduced scale.

---

## Appendix — environment & gotchas

- Python: `.venv/bin/python` (torch 2.9.1+cu128, CUDA available). `import torch` fails on the bare
  system python — always use the venv.
- Weights: `checkpoints/stochastic_navier_stokes/{stochastic_interpolant_small,flow_matching}/model.pth`.
- CUDA SDPA: full-Σ_s JVPs must run under the math SDPA backend (handled by `_math_sdpa`).
- `num_physical_steps > 5` (else empty assimilation loop → all-NaN metrics).
- The NS driver ignores CLI `method=`/`scenario=`; restrict with `+ns_methods`/`+ns_scenarios`
  (note the Hydra quoting for strings with spaces/parens: `+ns_methods=["Ours (SI-SDE)"]`).
- `likelihood_mode` is global (top-level); it is not a results column, so compare modes via the
  ablation, not the headline.
- The paper does not compile in this environment (TeX install missing `gensymb`, `animate`,
  `listingsutf8`, …) — pre-existing, unrelated to content edits.
</content>
</invoke>
