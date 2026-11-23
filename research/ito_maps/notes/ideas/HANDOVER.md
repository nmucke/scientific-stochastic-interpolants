# Handover: improving few-step conditional Itô maps

Written 2026-11-23 for the agent who will implement and test these recommendations on a GPU machine.

Your job is to implement the methods below in `scientific-stochastic-interpolants`, test them on the stochastic Navier–Stokes (NS) case, and report what holds up. NS is the test bed, not the target: every method must be written so that it applies to other forecasting problems (Section 2).

This document is self-contained. Supporting material in the same folder:

- `ito_map_improvement_ideas.tex`: the short research note this handover expands. The compiled PDF and `figures/*.pdf` are git-ignored; rebuild with `uv run python make_figures.py` and `pdflatex`.
- `agent_reports/`: the five source reports, the small numerical check scripts (`checks/`) and the 1D toy study (`toy/`).
- `../key_findings.md`: consolidated lab notes on the existing method, data and known code pitfalls. Read it before touching code.

---

## 0. Ground rules

### What is and is not established

Nothing below has been run on a trained NS map. The evidence is of three kinds, and each claim in this document is tagged:

- **[derived]**: derivation only.
- **[toy]**: confirmed on a scalar or 1D/2D toy problem.
- **[data]**: measured on the NS data archive.
- **[unverified]**: plausible, not checked.

Treat every recommendation as a hypothesis with a stated falsifier. If a falsifier fires, stop that line, record the result, and move on. A clean negative result is a useful outcome.

### Repository workflow

Follow `AGENTS.md` at the repo root. In short:

- Branch from `origin/dev` and open PRs into `dev`, never `main`.
- General, reusable method code goes in `src/scisi/` with tests in `tests/`. It must not import from `research/`.
- Experiment scripts, NS-specific launchers, reports and their tests go under `research/ito_maps/` (tests in `research/ito_maps/tests/`).
- Do not commit or push unless the user asks.

### Reporting

- Report failures and partial results as plainly as successes, with the numbers.
- Never tune on the test split. Select everything on validation (Section 6).
- Keep one results table per experiment that includes the baseline rerun under the same protocol.
- If you change a default that affects existing behaviour, say so and add a regression test.

---

## 1. Background

### 1.1 Setting and notation

Two clocks. Physical time advances in steps: given a history `h` (the last `p` states, `p = 5` for NS), forecast the next state `y`. Generative time `t ∈ [0, 1]` runs inside one forecast.

- Base point `c = c(h)`: the latest observed state. The generative process starts from this point mass.
- Interpolant: `I_t = α_t c + β_t y + γ_t √t z`, `z ~ N(0, Id)`, with `α_t = 1 − t`, `β_t ∈ {t, t²}`, `γ_t = a(1 − t)`.
- Forecasting SDE: `dX_t = b_t(X_t; h) dt + γ_t dW_t`, `X_0 = c`. `X_t` has the law of `I_t`, so `X_1 ~ p(y | h)`.
- Drift target for regression: `R_t = α'_t c + β'_t y + γ'_t √t z`. Do not differentiate `√t`.
- Martingale: `M_t = ∫_0^t γ_r dW_r`. Variance clock: `q(t) = ∫_0^t γ_r² dr`. For `γ = a(1 − t)`, `q(t) = a²[1 − (1 − t)³]/3`.
- Itô map: `Φ_{s,t}(x) = x + (t − s) G_θ(s, t, x, h; Ψ_{s,t}) + (M_t − M_s)`. `G` is the average drift over `[s, t]`; `Ψ_{s,t}` is a finite description of the Brownian path on `[s, t]`. On the diagonal `G_{t,t} = b_t` and `Ψ = 0`.
- Training today: diagonal regression of `G(t, t, I_t; 0)` on `R_t`, plus an off-diagonal self-distillation loss with an EMA copy as target. The default is progressive self-distillation (PSD): for `s < u < t` on one shared path, `G(s,t,x)` is regressed on `λ G_ema(s,u,x) + (1 − λ) G_ema(u,t,x_u)`, `λ = (u − s)/(t − s)`.

Code map:

| Piece | File |
|---|---|
| Map interface, sampling, rollout | `src/scisi/models/base_map_model.py` |
| Itô map | `src/scisi/models/follmer_ito_map_model.py` |
| Interpolants | `src/scisi/models/interpolations.py` |
| Brownian drivers and path features | `src/scisi/sampling/brownian_path.py` |
| Two-time and path conditioning of the backbone | `src/scisi/architectures/map_conditioning.py` |
| Map trainer and off-diagonal losses | `src/scisi/training/map_trainer.py` |
| Base trainer (draws `t` and `noise` in `_prepare_batch`) | `src/scisi/training/trainer.py` |
| Solver baseline steppers | `src/scisi/sampling/sde_solvers.py` |
| Configs | `config/models/ito_map.yaml`, `config/training/map.yaml`, `config/stochastic_navier_stokes.yaml` |
| NS experiment scripts | `research/ito_maps/experiments/` (see its `README.md`) |

### 1.2 Diagnosis

The map's only random input is a per-pixel white martingale. Its covariance does not match the forecast uncertainty.

- **[data]** Per-mode power of the one-step increment `y − c` (an upper bound on the conditional variance), relative to the white endpoint noise `Var(M_1) = a²/3`, by wavenumber shell:

  | Shell | 1 | 4 | 7 | 11 | 16 | 32 | 48 | 64 |
  |---|---|---|---|---|---|---|---|---|
  | increment / white | 0.31 | 20 | 190 | 59 | 9.8 | 0.59 | 0.066 | 0.012 |

  The value of `a` used for this table was not recorded. Changing `a` shifts every entry by a constant factor; the four-decade range is what matters. Recompute it in Phase 0.
- **[derived]** A one-step map from a point mass is `Φ_{0,1} = c + G_{0,1}(Ψ) + M_1`, whose squared-error optimum is `E[X_1 | Ψ]`. The explicit noise supplies white spread `a/√3` per degree of freedom. All other spread must come from the learned `Ψ`-dependence of `G`, which enters through a zero-initialised projection, is zero on the diagonal, and is only bootstrapped by self-distillation. Where the driver has too little variance the network must amplify noise; where it has too much, the network must cancel it.
- **[toy]** A map that ignores the path is underdispersed. For a Gaussian surrogate (`a = 0.1`, `σ = 0.17`, `β = t²`) the spread ratio is 0.34, 0.58, 0.65, 0.78, 0.88, 0.97 at 1, 2, 4, 8, 16, 64 steps. The multi-step solver reference is therefore underdispersed at finite step counts too.
- **Noise amplitude.** The current training grid uses `a = 1` (default in `config/models/ito_map.yaml`; `config/stochastic_navier_stokes.yaml` does not set `gamma_multiplier`). Other configs use `0.1`. Confirm the value in every checkpoint you evaluate.

This mechanism would explain the known failures: underdispersion, high-wavenumber spectral error, PSD being weak without teacher rollouts, and the Gaussian-base flow map winning at one step.

Other measured facts you will rely on:

- **[data]** The NS flow decorrelates fast: field autocorrelation 0.62, 0.33, 0.07, 0.017 at lags 1, 2, 5, 10. Beyond about 8 physical steps only the stationary law of the model chain can be judged.
- **[data]** Energy and enstrophy budgets close on the archive to about 0.5% and are consistent with viscosity `ν = 1e-3`, drag `α_d = 0.1`, forcing amplitude `ε = 1`, snapshot spacing `Δ = 0.5`. A second estimate on the test split closed the enstrophy balance to 3.4%. This is evidence for the parameters, not a recovered manifest.
- **[toy]** With `K = 4` global Karhunen–Loève features, the share of the last step's noise increment that the features do not explain is 0.03% at 1 step but 15%, 29%, 80% at 2, 4, 8 steps.
- **[toy]** With the current split `u ∈ [s + 0.25(t−s), s + 0.75(t−s)]`, a parent interval's four local increments are not functions of the children's eight (`R²` 0.75–0.91).

---

## 2. Generality requirements

The methods rest on general facts about SDE maps with additive Gaussian noise. Keep the implementation at that level.

**Rule.** Code in `src/scisi/` must not assume: a periodic domain, a square grid, one channel, vorticity, NS physical parameters, a history length, a particular `γ` or `β`, or a particular dataset. Each method takes a small interface; NS supplies one implementation of it, configured from `config/` or `research/`.

| Concept | General interface (in `src/scisi/`) | NS instance | Other instances to keep possible |
|---|---|---|---|
| Driver covariance | `NoiseCovariance`: `sqrt(z)`, `apply(x)`, optional `solve` | Fourier-diagonal spectrum on a periodic grid | Identity; per-channel scale; spherical-harmonic diagonal (global weather); DCT or patch-diagonal (non-periodic or masked domains); low-rank plus diagonal |
| Schedule | `β` derived from any decreasing `γ` through `q(t)` | closed form for `γ = a(1 − t)` | numerical quadrature for other `γ` |
| Variance clock | `interpolation.variance_clock(s, t)` | closed form | any deterministic `γ` |
| Linear invariant | `LinearInvariant`: `project_out(x)`, `lift(value)`, `value(x)` | spatial mean per channel | divergence-free projection; mass or tracer totals; boundary values |
| Quadratic budget | `QuadraticBudget`: weight operator, linear dissipation operator, injection rate, physical step | enstrophy and energy with `D = α_d − νΔ` | any system `du = (N(u) − D u) dτ + ε dξ` where `⟨u, 𝒲 N(u)⟩ = 0` |
| Path features | `PathFeatures`: `features(path, s, t, K)` with an exact parent-from-children relation | Legendre moments | local increments, KL (existing) |
| Distribution scores | functions of sample tensors and optional feature maps | pixel CRPS plus spectral features | any feature map supplied by the caller |
| Observation operator | existing `LinearObservationOperator` in `src/scisi/likelihood_models/` | sparse points | any linear `H` |

Further requirements:

- **Identity reproduces today's behaviour.** With `NoiseCovariance = Identity`, no invariant, and the existing schedule and features, new code paths must reproduce current outputs. Add regression tests that assert this with fixed seeds.
- **Estimate, do not hard-code.** Anything fitted to data (covariance spectrum, feature statistics, climatology) is produced by a generic estimation script that takes a dataset config, and is stored as a file referenced from the config. No NS constants in `src/`.
- **Channels and masks.** Write operators for `[B, C, *spatial]` tensors with arbitrary `C` and arbitrary spatial rank where practical. `MapTrainer` currently rejects non-trivial masks; do not make that worse. If a method cannot support masks, raise a clear error and note it.
- **Validate generality cheaply.** For each new component, include a unit test on a non-NS shape (for example `C = 3`, non-square grid). If time allows after the NS results, run the best configuration once on a second dataset already in the repo (for example KTH, `config/kth.yaml`) to check nothing is NS-specific. This is a smoke test, not a benchmark.
- **Document the assumption each method needs.** Each new class docstring states its mathematical assumption (for example "additive, state-independent diffusion").

---

## 3. Phase 0: diagnostics on existing checkpoints (no training)

Run these first. They test the diagnosis and decide what is worth building. Put scripts under `research/ito_maps/experiments/`, outputs under `research/ito_maps/results/phase0/`.

Use existing checkpoints under `checkpoints/stochastic_navier_stokes/` and the training-grid runs under `research/ito_maps/results/training_grid/`. Record, for each checkpoint: `gamma_multiplier`, `β`, features type, `K`, and training mode.

| # | Diagnostic | How | What it decides |
|---|---|---|---|
| D1 | Uncertainty spectrum against the driver | Per Fourier mode: power of `y − c`, power of the ensemble anomaly of the best solver reference (SI with ≥ 64 steps), and `a²/3`. Radial shells and full 2D. | Confirms or refutes the spectral mismatch. Also yields the first estimate of the covariance for WP1. |
| D2 | Per-wavenumber spread ratio | Ensemble spread spectrum of Itô-1, Itô-8 against SI-128 on validation. | Prediction: ratio below 1 at forced scales; flat excess tail at high wavenumber. |
| D3 | Path-feature ablation | Zero `path_features` at sampling time in an Itô checkpoint; compare spread per shell. | Measures how much spread the feature branch actually creates. |
| D4 | Noise-leak regression | Regress high-pass(sample − ensemble mean) on high-pass(`M_1`) per pixel. | A non-zero coefficient is uncancelled white noise. |
| D5 | Rollout decomposition | Per lead (1–10 and a long run): pooled member variance over climatology; mean member–member and member–truth pattern correlation (equal if calibrated); injected against propagated spread (rerun one step with one shared Brownian path for all members). | Separates "too little stationary variance" from "members diverge too slowly". |
| D6 | Budget residual | WP7's label-free residual on existing SI and Itô rollouts, including beyond lead 95. | Prediction: underdispersed models show a negative mean residual. |
| D7 | Stationary statistics of long runs | Pooled enstrophy, spectrum, lag-1 autocorrelation (truth 0.62) from existing long rollouts. | If all are within the data's trajectory-bootstrap band, WP6c is unnecessary. |

Notes:

- Equal seeds do not give equal Brownian paths across step budgets. For same-path comparisons share one path object explicitly (`key_findings.md` §1).
- Compute spectra on denormalised fields in float64, using the conventions in `key_findings.md` §4.
- Write D1 generically: it takes a dataset, a basis transform, and optionally an ensemble; NS passes the FFT.

**Gate.** If D1 shows the uncertainty spectrum within a factor of about 3 of white across scales, the central diagnosis is wrong for this data. Report that and re-plan before WP1.

---

## 4. Work packages

Order and dependencies are in Section 7. Each package lists the idea, the mathematics, a general interface, implementation notes, tests, the NS experiment and a falsifier.

### WP1. Covariance-matched driver and matched schedule (highest priority)

**Idea.** Choose the noise covariance and `β` so that the martingale itself carries the forecast uncertainty. The drift then has little state dependence, the average drift little path dependence, and the map has little to learn about the path.

**Mathematics [derived, toy].** Let `C` be a covariance operator and

```
I_t = α_t c + β_t y + γ_t C^{1/2} √t z,      M_t = C^{1/2} ∫_0^t γ_r dW_r,      Var(M_t) = q(t) C.
```

Proposition. If `y | h ~ N(m, Σ)` and

```
C = Σ / q(1),        β_t² = (q(t) − t γ_t²) / q(1),
```

then `b_t(x; h) = α'_t c + β'_t m` for every `x`, `X_t = α_t c + β_t m + M_t`, and `G_{s,t}` is independent of the path. Every step count is exact with no path features.

Proof. `R_t` and `I_t` are jointly Gaussian given `h`. `Cov(R_t, I_t) = β β' Σ + γ γ' t C = (β β' q(1) + t γ γ') C`. Differentiating `β² q(1) = q − tγ²` gives `2ββ' q(1) = −2tγγ'`, so the covariance is zero and `E[R_t | I_t] = E[R_t]`. Equivalently `Var(I_t | h) = q(t) C = Var(M_t)`.

- `q(t) − tγ_t² = −2 ∫_0^t r γ_r γ'_r dr`, which is non-negative and increasing whenever `γ` is decreasing. Implement `β` from this formula for any such `γ`, with numerical quadrature as the fallback.
- For `γ = a(1 − t)`: `β_t = t √(3 − 2t)`, `β'_t = 3(1 − t)/√(3 − 2t)`, independent of `a`. `β'(0) = √3`, `β'(1) = 0`.
- The scale is then fixed by `C`: per mode, `q(1) C_k = Σ_k`.

The forecast law is not Gaussian in general. The proposition says what is removed (the Gaussian part of the path dependence); the path features still carry the non-Gaussian and history-dependent remainder.

**Toy evidence [toy].** Fraction of `Var(X_1)` not explained by `M_1` alone, exact drift, scalar targets:

| Driver variance / target variance | β = t² | β = t | matched β |
|---|---|---|---|
| 1, Gaussian target | 5.0% | 1.2% | 0.1% |
| 1, skewed / bimodal | 5.2 / 5.7% | 1.3 / 1.4% | 1.2 / 0.9% |
| 2 | 7.7% | 3.5% | 1.2% |
| 4 | 10.6% | 6.9% | 4.0% |
| 100 | 20–22% | – | 19–21% |
| 0.5 | 3.2% | 1.3% | 5.1% |
| 0.25 | 2.4% | 4.6% | 12.9% |
| 0.01 | 13–17% | – | 64–69% |

Reading: matching the covariance is the main lever. The matched `β` helps when the driver variance is 1–4 times the target variance and hurts when the driver is under-noised. Therefore **err on the side of too much noise**: aim for `q(1) C ≈ 1–2 × Σ̄`, and always run `β = t` with the same `C` as a comparison.

**General interface.**

```python
class NoiseCovariance(nn.Module):
    """Symmetric positive semi-definite operator C acting on [B, C, *spatial]."""
    def sqrt(self, z): ...        # C^{1/2} z
    def apply(self, x): ...       # C x
    def solve(self, x): ...       # optional: C^{+} x, for whitening and WP8

class IdentityCovariance(NoiseCovariance): ...          # reproduces today's behaviour
class FourierDiagonalCovariance(NoiseCovariance): ...   # spectrum buffer, periodic grids
```

Because `C^{1/2}` is linear and acts on the spatial index only, it commutes with everything the Brownian driver does in time. The clean implementation is therefore: draw white Gaussians exactly as now, apply `C^{1/2}` at the draw.

**Implementation.**

1. `src/scisi/sampling/` (or a new `src/scisi/noise/`): `NoiseCovariance` classes. `FourierDiagonalCovariance` takes a real, non-negative spectrum on the `rfft` grid, per channel; uses `norm="ortho"`; guarantees a real output.
2. `src/scisi/sampling/brownian_path.py`: the base class draws through one method (`_randn`). Give the path an optional `covariance` and apply `covariance.sqrt` there. All bridge logic is unchanged. Check both `WeightedBrownianPath` and `KarhunenLoeveBrownianPath`.
3. `src/scisi/models/interpolations.py`:
   - Add a `MatchedStochasticInterpolation` (or a `beta="matched"` option) with `β` and `β'` computed from `γ`.
   - Add `variance_clock(s, t)` to the interpolants so drivers stop hard-coding `γ = a(1 − t)`. `FollmerItoMapModel.__init__` currently probes for that exact form; keep the check but route the variance through the interpolation.
   - The interpolation holds the `NoiseCovariance`, so that the model, trainer and sampler all read it from one place.
4. Every place that draws interpolant or sampler noise must use the same `C`: `Trainer._prepare_batch` (`batch["noise"]`), the fresh `torch.randn_like` in `MapTrainer._compute_loss`, `make_brownian_path`, and the solver steppers in `sde_solvers.py` for the SI baseline. Grep for `randn` and audit each hit.
5. Path features stay at their physical scale (the toy study found unit-variance normalisation harmful, Section 4 WP3).
6. A generic estimation script (under `research/` or `scripts/`) that takes a dataset config and writes the spectrum file:
   - Preferred estimate: residuals `e = y − m̂(h)` of a deterministic regressor trained with MSE on the training split, evaluated on the **validation** split (training-split residuals underestimate the variance). `Σ̄_k = mean |ê_k|²`.
   - Alternative: ensemble anomalies of an existing multi-step SI model.
   - First pass only: the increment spectrum of `y − c`. It is an upper bound, so it over-noises.
   - Then `C_k = κ Σ̄_k / q(1)` with `κ ∈ [1, 2]`, a floor relative to the maximum (start at `1e-4`), smoothing across neighbouring modes, and zero in directions of exact invariants (WP2).
   - Keep the full spectrum, not only the radial average; the NS forcing is anisotropic.
7. Do not learn `C` through the diagonal loss; the objective changes with `C` and the problem is ill-posed.

**Tests (`tests/`).**

- `IdentityCovariance` reproduces current samples, paths and losses bit-for-bit under fixed seeds.
- Empirical covariance of `M_t − M_s` matches `(q(t) − q(s)) C` for a small synthetic spectrum; additivity over sub-intervals is exact.
- `β(0) = 0`, `β(1) = 1`, `β` monotone; the closed form equals the quadrature form.
- Gaussian exactness: with an analytic Gaussian target and the exact drift, a one-step map with zero features has the right mean and covariance; with `β = t²` it does not.
- Non-NS shapes: `C = 3`, non-square grid.

**NS experiments.**

1. *No map training.* Retrain only the SI drift on three processes: (white, `t²`) as today, (coloured, `t`), (coloured, matched). Then:
   - Repeat the nested Monte Carlo path-compression test (`research/ito_maps/experiments/ns_path_compression.py`): omitted endpoint-variance fraction at `K = 1, 4`. The lab-note value for the white process is about 9.7% at `K = 4`.
   - Euler sampler with exact integrated noise at `N = 1, 2, 4, 8` against its own `N = 128`: spread ratio, CRPS, spectra.
2. *Map training* on the best process, with `K ∈ {0, 2, 4}` features. `K = 0` is meaningful here: it tests how far the martingale alone goes.
3. Sensitivity to `κ ∈ {1, 2, 4}`.

**Falsifier.** If, under the coloured processes, the `K = 4` omitted fraction does not fall well below the white-process value, or the few-step Euler sampler is still strongly underdispersed relative to its own 128-step run, the non-Gaussian part dominates and this package does not deliver. Report and continue with WP3–WP5 on the white process.

**Risks.** The conditional covariance depends on the history and is not stationary in space; a single stationary `C` is a first approximation. A history-dependent scalar scale `C(h) = λ(h)² C_0` is allowed (the history is fixed during generation) and connects to WP7. The process changes, so all baselines must be retrained on the same interpolant for a fair comparison. Smooth noise changes what the backbone sees as input.

---

### WP2. Invariant-closed maps and driver

**Idea.** Exact linear invariants should hold for any step count and survive composition and self-distillation. Low novelty, cheap, do it with WP1.

**Mathematics [derived].** Let `L` be linear with `L y = ℓ(h)` almost surely, and let the driver satisfy `L C^{1/2} = 0`. Then `L I_t = ℓ_t := α_t L c + β_t ℓ(h)` and the exact map satisfies `L Φ_{s,t}(x) = L x + ℓ_t − ℓ_s`. The parametrisation

```
G_θ = (Id − L⁺L) N_θ + L⁺ (ℓ_t − ℓ_s)/(t − s)
```

enforces this for every `θ` and every step count. The class is closed under composition, convex combination (PSD targets) and the Lagrangian identity. With a white driver, `L b_t` contains the singular term `−(L x − ℓ_t)/(1 − t)`, which a few-step map cannot integrate; that is why such maps leak invariants.

Affine invariants need `α_t + β_t = 1` only if you want `L I_t` constant; the formula above handles the general case through `ℓ_t`.

**General interface.**

```python
class LinearInvariant(nn.Module):
    def value(self, x): ...        # L x
    def project_out(self, x): ...  # (Id − L⁺L) x
    def lift(self, v): ...         # L⁺ v
```

NS instance: spatial mean per channel (zero to about 1e-6 in the data). Other instances to keep possible: Leray projection for velocity fields, total mass.

**Implementation.** Apply in `BaseMapModel.average_velocity` output (or a wrapper module around `map_model`), and zero the invariant direction in `NoiseCovariance` (for the spatial mean: `C_0 = 0`). Target `ℓ(h)` is supplied by the invariant object from the history.

**Tests.** Invariant error at `N = 1, 2, 4, 8` is at machine precision after the change and non-zero before; holds along a 500-step rollout; non-NS shape.

**Falsifier.** None needed; it is exact. Check only that forecast scores do not change.

---

### WP3. Path features and interval sampling (adopt; prior art)

These are not research contributions (Legendre features are in arXiv:2606.01086 and 2610.29605) but the toy study found them the clearest practical gain.

**[toy] results** (1D two-mode target, `W1` to the true law; floor 0.014; three seeds):

| Variant | 1 step | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Current method (local increments, PSD) | 0.126 | 0.030 | 0.022 | 0.015 | 0.014 |
| + terminal energy-score anchor | 0.083 | 0.023 | 0.024 | 0.018 | 0.015 |
| Legendre features, physical scale | 0.039 | 0.026 | 0.031 | 0.028 | 0.022 |
| Legendre features + anchor | 0.030 | 0.019 | 0.025 | 0.018 | 0.015 |
| Anchor only, no self-distillation | 0.024 | 0.033 | 0.046 | 0.076 | 0.100 |

Legendre features cut one-step error by a factor of 2–3 in every toy setting and halved the pathwise error against a same-path reference. Unit-variance feature normalisation hurt many-step accuracy. This is 1D evidence only.

**Definition.** For `k = 0 … K − 1`,

```
φ_k(s, t) = ∫_s^t P_k( (2r − s − t)/(t − s) ) dM_r,
```

with `P_k` the Legendre polynomials. `φ_0 = M_t − M_s`. Keep the physical scale (no normalisation), so features vanish as `t → s` in agreement with the zero-feature diagonal.

Key property **[toy, residual 3.6e-15]**: a polynomial of degree `< K` on the parent interval restricts to a polynomial of degree `< K` on each child, so the parent's moments are an exact linear function of the two children's moments for any split point. The parent's information is contained in the children's, which the progressive loss needs and the current local increments do not provide.

**Implementation.**

- Add a third `brownian_features` option beside `local` and `karhunen_loeve`.
- Simplest correct route: sample `M` on a fixed fine grid of `n` points (64–256) per path, restrict `s, u, t` to grid points, and compute moments as weighted sums of fine increments. Memory is `B × C × spatial × n`; for large fields provide a lazy variant later. Exact joint Gaussian sampling of the moments is the alternative.
- Whether moments are taken against `dM` or `dW` was not separated cleanly in the toy for Legendre features; for plain increments, `M`-increments beat `W`-increments at one step. Default to `dM`, and compare once.
- Interval sampling: add a midpoint-split option (`u = (s + t)/2`), needed by WP4b; consider lowering the minimum interval length towards zero so that the off-diagonal support meets the diagonal.

**Tests.** Parent-from-children linear relation; `φ_0` equals the increment; covariance of the moments against the analytic value; features vanish as `t → s`.

**NS experiment.** `local K=4` against `Legendre K ∈ {2, 4}` on the same process, same budget. Report one-step and multi-step scores and monotonicity in step count.

**Falsifier.** No gain at one step on validation: keep local increments.

**Not recommended: the bridge (integrating-factor) form of the map.** It rewrites `Φ_{s,t}(x) = (1−t)/(1−s) x + [β_t − β_s(1−t)/(1−s)] Ȳ_{s,t} + a(1−t)(W_t − W_s)` so that explicit noise vanishes at `t = 1`. The identity is correct **[toy]**, but in the toy it was worse than the current form at every step count in four of five settings, and a repaired variant only matched the current form. Do not spend time on it unless WP1 fails and D4 shows a large noise leak.

---

### WP4. Off-diagonal objectives

Three objectives to compare against PSD at equal compute, starting from the same diagonal-trained checkpoint. Implement each as a new entry in `OFF_DIAGONAL_LOSSES` in `map_trainer.py`, sharing the interval and path sampling.

#### WP4a. Picard self-distillation

**Idea.** Teacher rollouts helped because they evaluate the data-trained drift along the path. The integral form of the Lagrangian identity does that without simulation.

**Mathematics [derived].** `(t − s) G_{s,t}(x; W) = ∫_s^t b_u(Φ_{s,u}(x; W)) du`. Estimate the integral with one random node `u ~ U(s, t)`:

```
target = sg[ b_ema(u, Φ_ema,s,u(x; W)) − ( β'_u − (β_t − β_s)/(t − s) ) · y ]
L = E ‖ G_θ(s, t, x; Ψ_{s,t}) − target ‖²
```

- `b_ema(u, ·) = G_ema(u, u, ·; 0)` is the diagonal of the EMA map.
- The `y` term has zero mean over `u` for any field that does not depend on `u`, so it is a control variate: unbiased, and it removes the dominant deterministic variation of the drift target in `u`. `y` is the sample's own target. Write it with the interpolation's `beta` and `beta_diff` so it works for any schedule. If `α` is not linear in `t`, add the analogous `c` term.
- Cost: two EMA evaluations and one student evaluation, the same as PSD. No JVP. Works with any feature type.

**Why it might help [derived, unverified for SGD].** Every target queries the diagonal, which is anchored to data. The target operator is of Volterra type; in idealised function-space iteration its error contracts factorially. The PSD operator passes child errors to the parent with unit gain and never touches the diagonal.

**Risks.** Monte Carlo variance over `u` (consider 2 nodes, or stratified `u`). The drift's Lipschitz constant grows like `1/(1 − u)` near `t = 1`; this is milder under WP1.

**Falsifier.** No gain over PSD in one-step spread or CRPS at equal budget.

#### WP4b. Weak closure: energy-score self-distillation conditional on the coarse path

**Idea.** Be pathwise on the resolved features and match in law on the rest, so that information the features do not carry turns into sampled variability instead of lost variance.

**Mathematics [derived].** Give the network one extra white latent field `η` (zero-initialised input branch, scaled so it vanishes on the diagonal, for example by `√(t − s)`). With student `S(η) = G_θ(s, t, x, h; Ψ_{s,t}, η)` and the composed EMA target `T` from two half-steps (each with its own latent, features from the same path):

```
L = E[ ½ Σ_{i=1,2} ‖S(η_i) − T‖ − ½ ‖S(η_1) − S(η_2)‖ ]
```

This energy score is strictly proper, so at the optimum `Law(S | x, Ψ_{s,t}) = Law(T | x, Ψ_{s,t})`. With nested features (midpoint split or Legendre moments), the true conditional kernels satisfy `Q_{s,t}(· | x, Ψ_{s,t}) = E[Q_{u,t} ∘ Q_{s,u} | Ψ_{s,t}]`, so they are an exact fixed point at any finite `K`. The squared-error PSD loss has no such fixed point. Uniqueness given the diagonal is unverified.

A data anchor fits the same loss: `Law(Φ_{0,t}(c; W, η)) = Law(I_t | h)` for every `t`, scored against a fresh interpolant sample; `t = 1` is the ordinary score against truth.

**Implementation notes.**

- Scores must be general functions of sample tensors: per-degree-of-freedom fair CRPS plus an energy score on caller-supplied feature maps. The plain energy score is weakly discriminative in high dimension. NS supplies shell spectra and patch features; do not hard-code them in `src/`.
- Two student passes per example; budget accordingly.
- Requires nested features (WP3).

**Control.** Terminal energy-score fine-tuning alone (score one- and two-step outputs against data, alongside PSD). This is known practice (for example arXiv:2509.25631, 2605.06916) and helped moderately in the toy (one-step `W1` down 25–40%, about 2.3× training cost). Used without any self-distillation it gives the best one-step law but a map that degrades with more steps, so always keep a consistency loss on.

**Falsifier.** At equal budget, compare PSD, PSD + terminal anchor, and weak closure. Drop the weak closure if it does not improve rank histograms over PSD, or adds nothing over the terminal anchor at 2–4 steps.

---

### WP5. Interpolant refresh

**Idea.** A one-call corrector that leaves the forecast law unchanged and forgets small errors. Use it on every forecast before it enters the history, and build training histories with the same kernel.

**Definition.** For a state `y`, history `h` with base `c`, and `s ∈ (0, 1)`:

```
x_s = α_s c + β_s y + γ_s C^{1/2} √s z          (re-noise with the interpolant)
y'  = Φ_{s,1}(x_s, h; W')                         (map back with a fresh path on [s, 1])
```

Call this kernel `R_s`. `Φ_{s,1}` started from interpolant states is exactly what the off-diagonal loss already trains.

**Properties [derived].**

- *Invariance.* With the exact drift, if `y ~ p(· | h)` then `y' ~ p(· | h)`: the interpolant and the SDE share the marginal at time `s`.
- *Forgetting.* `KL(R_s(y, ·) ‖ R_s(y + e, ·)) ≤ β_s² ‖e‖²_{C⁻¹} / (2 γ_s² s)`. For white noise and `β = t²` this is `‖e‖² s³ / (2a²(1 − s)²)`. Errors below the noise floor `γ_s √s / β_s` are forgotten. The bound grows with dimension, so read it per degree of freedom.
- *Mode-selective shrinkage* (Gaussian mode `N(m, σ_f²)`, `a = 1`, `β = t²`): the variance is preserved and a perturbation is transmitted with factor `κ_s`:

  | s | σ_f = 0.05 | 0.1 | 0.3 |
  |---|---|---|---|
  | 0.8 | 0.16 | 0.32 | 0.72 |
  | 0.9 | 0.38 | 0.64 | 0.93 |
  | 0.95 | 0.68 | 0.88 | 0.98 |

  Directions where the forecast law is narrow, where artefacts live, are pulled to the conditional mean; uncertain directions pass through.

The invariant law of the learned kernel is the model's, not the truth's; refresh removes artefacts, it does not repair a wrong forecast law.

**Use A, inference.** `y_{n+1} = R_s(Φ_{0,1}(c, h; W))`: two network calls per physical step. Implement as an option of `sample_trajectory` (for example `refresh_time=None`). It is generic: it needs only `training_pair`, `make_brownian_path` and `map`.

**Use B, training.** Replace the last `r` history frames by `sg[R_s(truth frame | preceding history)]`, applied sequentially, keeping the true target and the usual losses. With Use A on at inference, training and test histories differ only through the raw artefact, bounded by the KL above. `s → 1` is teacher forcing; smaller `s` approaches free running. Needs dataset windows of length `p + 1 + r`.

**NS experiments.**

1. No training: validation rollouts of 10 leads and about 500 leads with the 1-step map (i) plain, (ii) as a 2-step same-path map, (iii) with refresh at `s ∈ {0.8, 0.9, 0.95}`. Report spectra against lead, pooled enstrophy, budget residual (WP7), spread–skill and lead-1 CRPS.
2. Fine-tune with Use B from the same checkpoint; evaluate with and without Use A.

**Falsifier.** Drop Use A if refresh does not beat the 2-step same-path map at equal cost for any `s`, or degrades lead-1 CRPS.

**Risk.** `Φ_{s,1}` for `s` near 1 lies close to the diagonal; make sure the interval sampler covers those short intervals.

---

### WP6. Rollout training

A one-call map makes unrolled training affordable. Build one generic rollout fine-tuning loop under `research/ito_maps/` on top of general scoring functions in `src/scisi/`. Use `model.map` (not `sample`, which runs under `no_grad`), a fresh `make_brownian_path` per physical step, and truncated backpropagation through the last `j` steps.

#### WP6a. Baseline: unrolled energy-score fine-tuning

Score the `ℓ`-step rollout ensemble against the truth at lead `ℓ`. This is established practice (Swift, FGN, AIFS-CRPS) and is the baseline every rollout method must beat. Not novel; do not present it as such.

#### WP6b. Chapman–Kolmogorov closure with direct-lead teachers

**Idea.** Train the same network, with a lead embedding, as a direct lead-`ℓ` interpolant `D_ℓ` (`c = y_n → y_{n+ℓ}`). It is simulation-free and targets `p(y_{n+ℓ} | h_n)` exactly, with no compounding, whether or not the history is a sufficient state. Then make the law `A_ℓ` of `ℓ` composed one-step maps agree with it.

```
L = Σ_ℓ w_ℓ [ ES(A_ℓ(h), y_{n+ℓ}) + λ ED(A_ℓ(h), sg D_ℓ(h)) ] + w_∞ ED(A_L(h), μ̂)
```

`ES` is the fair energy score against the single truth, `ED` the energy distance between ensembles, `μ̂` a sample from the climatological pool (states from other trajectories), used once `L` exceeds the mixing time.

- The direct kernel supplies a whole target ensemble per history instead of one truth sample, which lowers gradient variance when trajectories are few (160 for NS).
- Choose leads from the data's decorrelation, not hard-coded: for NS, `ℓ ∈ {2, 3, 4, 6}` and `L ≥ 8`. Provide a helper that estimates the mixing time from the autocorrelation.
- The lead embedding goes through the existing `pars_cond` path. Note the known `pars_cond` shape bug in `sample_trajectory` (`key_findings.md` §2).

**Stage A (one training run).** Lead-conditioned model with `ℓ ∈ {1, 2, 3, 4, 6}`. On validation compare `D_ℓ` against `A_ℓ` at equal lead: spread–skill, rank histograms, CRPS, spectra.

**Falsifier.** If `D_ℓ` is not better calibrated than `A_ℓ` at `ℓ = 4–6`, the teacher is useless: drop it and keep WP6a.

**Stage B.** Fine-tune with the full loss against WP6a at equal compute.

**Risks.** Multi-lead training may cost one-step accuracy (fallback: a separate teacher network). The closure is not exactly zero for the truth if the history is insufficient, so `λ` trades one-step against multi-lead fidelity.

#### WP6c. Stationary window matching on persistent chains

Run only if D7 shows a stationary-statistics gap.

**Fact [derived].** For an order-`p` kernel with a unique stationary law, equality of the stationary law of `(p + 1)`-frame windows with the data's implies equality of the whole stationary process. Matching only the marginal invariant measure does not.

**Method.** Keep a buffer of free-running model histories on the GPU; each iteration advance it one map call with gradient, and minimise the energy distance between features of model windows and data windows, with the one-step losses kept on and a small weight. Reseed part of the buffer from data with some probability. Features are supplied by the caller (NS: shell spectra per frame, spectra of increments, patch features, scalar invariants).

**Risk.** Optimising the chosen features at the expense of others; check statistics that were not in the loss.

---

### WP7. Itô-budget consistency

**Idea.** For a stochastically forced system the pointwise equation residual is the forcing and does not vanish. The correct residual is a conditional-moment identity for observables whose evolution closes.

**General statement [derived].** Consider `du = (N(u) − D u) dτ + ε dξ`, with `D` linear, self-adjoint, non-negative; `ξ` a Wiener process with covariance `Q`; and a self-adjoint weight `𝒲` commuting with `D` such that the nonlinearity conserves the quadratic form: `⟨u, 𝒲 N(u)⟩ = 0`. For `q_𝒲(u) = ½⟨u, 𝒲 u⟩`, Itô's formula gives

```
d q_𝒲 = ( −⟨u, 𝒲 D u⟩ + ½ ε² tr(𝒲 Q) ) dτ + ε ⟨u, 𝒲 dξ⟩.
```

Integrate over one physical step `Δ` from `c` to `y` and approximate the dissipation integral by the trapezoid rule (error `O(Δ³)`):

```
q_𝒲(y) − q_𝒲(c) ≈ −(Δ/2) [ ⟨c, 𝒲 D c⟩ + ⟨y, 𝒲 D y⟩ ] + ½ ε² Δ tr(𝒲 Q) + martingale term.
```

Collect the `y` terms on the left and the `c` terms on the right. With `q^±_𝒲(x) = ½⟨x, 𝒲(Id ± Δ D) x⟩` and taking the conditional expectation:

```
E[ q⁺_𝒲(y) | h ] = q⁻_𝒲(c) + ½ ε² Δ tr(𝒲 Q).
```

The reference check script is `agent_reports/checks/budget_check.py`.

Because `q⁺` is quadratic, `E q⁺(y) = q⁺(E y) + ½ tr(𝒲(Id + ΔD) Cov(y | h))`, hence

```
tr( 𝒲(Id + ΔD) Cov(y | h) ) = 2 [ q⁻_𝒲(c) + ½ ε² Δ tr(𝒲 Q) − q⁺_𝒲(E[y | h]) ].
```

Each such identity fixes one weighted trace of the forecast covariance: a constraint on ensemble spread. A second-moment identity also holds: the residual's conditional variance is approximately `ε² Δ ⟨u, 𝒲 Q 𝒲 u⟩`.

Before relying on the exact coefficients, re-derive the trapezoid form and confirm it numerically on the data (that is the first test below). The published check reproduced the archive to about 0.5%.

**NS instance [data].** `D = α_d − νΔ_x`; `𝒲 = Id` (enstrophy) and `𝒲 = (−Δ_x)⁻¹` (energy). On the archive: stationary enstrophy balance 1.991 against 2.000; energy balance 0.03783 against 0.03800; per-pair residual mean −0.5% ± 0.5% of injection, uncorrelated with the state; residual standard deviation 0.722 against 0.720 predicted. Injection per step is about 1.0 against mean enstrophy 4.78, so the spread term is a well-conditioned effect of about 20%.

**General interface.**

```python
class QuadraticBudget:
    """weight 𝒲, dissipation D, injection rate ½ε² tr(𝒲Q), physical step Δ."""
    def residual(self, y_next, y_prev): ...   # q⁺(y_next) − q⁻(y_prev) − injection, per sample
```

NS constructs two instances from `(ν, α_d, ε, Q, Δ)` in a config. Other systems supply their own `D`, `𝒲` and injection. Operate on denormalised fields.

**Uses, in order.**

1. *Label-free diagnostic.* Per member and lead, the residual `r` must be a mean-zero martingale difference with known variance. It tests the kernel on the model's own histories and beyond the last lead with truth. This is D6; implement it first.
2. *Training loss.* On two or more samples from the same history, use the unbiased U-statistic `(1/(E(E − 1))) Σ_{e ≠ e'} r_e r_{e'}`. Do **not** use `(mean r)²`: it penalises `Var(q(y))` and would worsen underdispersion. The loss needs no target, so it can be applied on rolled-out histories. Use a small weight.
3. *Driver scale.* A history-dependent scale `λ(h)` for `C(h) = λ² C_0` in WP1, chosen to close the budget.

**Tests.** The data archive closes the budget within tolerance (this doubles as a check on the parameters and the derivation); a synthetic linear SPDE with known answer; the U-statistic is unbiased under a calibrated synthetic ensemble.

**Falsifier.** On existing rollouts, the known coverage deficit of the Itô maps should appear as a negative mean residual. If the residual does not separate calibrated from underdispersed models, keep it as a sanity check only.

**Risks.** Each identity constrains one scalar; a biased mean can be compensated by wrong spread. It needs the physical parameters; for NS they are corroborated, not confirmed. The 128² data are a stride-2 subsample of 256² without anti-aliasing, which affects the palinstrophy-dependent term.

---

### WP8. Martingale Kalman conditioning (after WP1)

**Idea.** A few-step map has no long trajectory along which to apply guidance. But the one-step map is `y = c + G_θ(c, h; Ψ) + S Ψ`, with the Gaussian driver `Ψ` entering linearly through a known operator `S`. Observations can be imposed on the driver.

**Method [derived].** Observations `o = H y + η`, `η ~ N(0, R)`; driver prior covariance known from `C` and the variance clock.

1. Freeze `G` at its current value. The linear-Gaussian posterior of the driver is sampled exactly by Matheron's rule: `Ψ⁺ = Ψ + K (o + η̃ − H y(Ψ))`, with gain `K = Cov(Ψ, Hy)(q(1) H C Hᵀ + R)⁻¹` and `η̃ ~ N(0, R)`.
2. Correct exactly with a preconditioned Crank–Nicolson step relative to that Gaussian reference, accepting on the ratio of the true to the frozen-`G` likelihood. One network call per step, no gradients.

In the Gaussian limit of WP1 the acceptance is one and the samples are independent exact posterior draws. With a white driver the frozen-`G` reference is poor, which is why this depends on WP1.

Hard constraints (`R → 0`) by iterating the update are an approximation, not an exact posterior draw.

**Implementation.** A new posterior class beside `src/scisi/posterior_models/`, reusing `LinearObservationOperator`. `H C Hᵀ` comes from `NoiseCovariance.apply`. Generic in `H` and `C`.

**Experiments.** A linear-Gaussian toy with the Kalman posterior as ground truth, then NS with sparse point observations against the existing multi-step `StochasticInterpolantPosterior`: acceptance rate, posterior CRPS, cost.

**Falsifier.** Acceptance below about 10% with the coloured driver.

---

### WP9. Forcing-revealed latents (data requirement only, for now)

If the dataset is regenerated, **log the physical forcing increments** per stored interval. That enables conditioning the map on a few coefficients of the physical noise path, sampled exactly at inference, with the generative driver modelling only the remainder. Do not build the method yet. The simulator-only falsifier is: fix `c` and the first `J` coefficients, resample the rest, and measure `Var(y | c, ψ_{<J}) / Var(y | c)` for `J = 1 … 8`; the idea fails if less than 80% is explained at `J ≤ 4`.

When regenerating data, also save: all physical parameters and the generator revision, conditional reference ensembles for a set of histories (true conditional calibration checks), and anti-aliased downsampling.

---

## 5. Things considered and rejected

Do not re-derive these unless you find new evidence.

- Larger `K` with the current local increments: `K = 16` did not beat `K = 4` in the initial stage. Change the feature type instead.
- `(t − s)²` endpoint weighting: starves short intervals; does not touch the mechanism.
- Post-hoc variance inflation or scaling the noise at inference: not a valid change of law; breaks spectra.
- Eulerian / MeanFlow identities for Itô maps: admit zero-loss solutions that ignore the path (Pan et al., arXiv:2606.11156, Prop. 2).
- Extending the map to a physical-time semigroup: distils the model's own rollout; faster, not more faithful.
- Pathwise shadowing of a finer reference across physical steps: imposes spurious contraction, given the fast decorrelation.
- Regressing on the truth from model-generated histories: the optimum blurs for a stochastic model.
- Brownian paths correlated across physical steps: breaks the one-step law.
- Noised-history conditioning: known, and destroys small-scale information; WP5 gets the benefit without it.
- Pointwise PDE residual losses: for a forced SPDE the residual is the forcing.
- Coupling the generative noise to the physical forcing inside the interpolant: breaks the independence the drift identity needs.
- State-dependent diffusion: loses exact martingale sampling.
- Rotation equivariance for NS: a 90° rotation is not a symmetry of this forcing. Translations and sign-flipping reflections only.

---

## 6. Experimental protocol

Follow `key_findings.md` §4. The essentials:

- **Splits.** Whole trajectories: train 0–159, validation 160–179, test 180–199. History length 5.
- **Selection.** Every hyperparameter and checkpoint is chosen on validation by a rule written down beforehand. Open the test split once, at the end, with frozen configs. Label anything test-informed as exploratory.
- **Baselines, rerun under the same protocol.** Multi-step SI solver at 16, 64, 128 steps; the same solver with exact integrated noise (to separate map gains from noise-quadrature effects); the Gaussian-base flow map; the current Itô map (white driver, PSD, local `K = 4`). When the process changes (WP1), retrain the SI baseline on the new process as well.
- **Step budgets.** 1, 2, 4, 8 for maps, one checkpoint for all budgets.
- **Metrics.**
  - Fair CRPS and energy score (unbiased pair terms; do not clamp).
  - RMS spread with the finite-ensemble correction, spread–skill, rank histograms, extreme-rank fraction. Close CRPS can hide a clear calibration loss.
  - Shell spectra of members (not of the ensemble mean), per-wavenumber spread ratio.
  - Rollouts: all of the above against lead up to the last lead with truth (95 for NS); beyond that only stationary statistics and the budget residual.
  - Network calls and synchronised wall-clock latency at equal ensemble size and precision.
- **Uncertainty.** Average origins within a trajectory, bootstrap whole trajectories, and report paired differences on shared resamples. Use at least three training seeds for any headline comparison; keep seed variation separate from trajectory variation.
- **Equal compute.** Compare training objectives at equal updates and report wall-clock. Count any simulation-based refinement in the cost.
- **Precision.** Use full precision for map losses (the AMP gradient-clipping path has a known issue). Dropout must be 0 for map training.
- **Write the protocol down first.** Before the first training run, commit a short protocol file under `research/ito_maps/` with splits, origins, ensemble size, budgets, metrics and the selection rule.

---

## 7. Order of work and decision points

```
Phase 0  diagnostics D1–D7 (no training)
   │  gate: D1 confirms spectral mismatch?
   ▼
WP1 step 1  SI drift on (white,t²), (coloured,t), (coloured,matched)   ── with WP2
   │  gate: path compression and few-step Euler spread improve?
   ├── yes → WP1 map training on the best process, K ∈ {0,2,4}
   └── no  → stay on the white process
   ▼
WP3  Legendre features, nested splits           (independent of the gate)
WP5  refresh at inference on the best 1-step map (no training)
   ▼
WP4  PSD vs Picard vs terminal anchor vs weak closure, equal compute
   ▼
WP6  rollout fine-tuning: energy-score baseline, then CK closure, then
     refresh training (WP5 use B); WP6c only if D7 shows a gap
WP7  budget loss, once the diagnostic is validated
   ▼
WP8  path-space conditioning (needs WP1)
```

Suggested first deliverable: Phase 0 results plus WP1 step 1. Those two decide most of the rest and need no map training.

If compute is tight, the minimum informative set is: D1, D2, D5, D6; WP1 step 1; WP3; WP5 Use A.

---

## 8. Known pitfalls in the current code

From `key_findings.md` §2; check each is still true before working around it.

- `MapTrainer` raises for any non-trivial mask.
- `sample_trajectory` handles `pars_cond` only as one scalar per future step.
- `BaseMapModel.sample` runs under `no_grad` and returns CPU tensors; use `map` for anything differentiable.
- A seed fixes a query sequence, not a path. Share the path object for same-path comparisons.
- The Itô map checks `γ = a(1 − t)` at construction and the drivers hard-code that variance clock.
- A trained map is tied to its diffusion schedule and, after WP1, to its covariance. Neither can be changed after training.
- Validation in `MapTrainer` uses the full off-diagonal weight; early stopping is held until the ramp ends.
- `EarlyStopping` treats small degradations as improvements; AMP clips without unscaling.
- The NS dataset window count is `T − L − 1`; keep it for comparability or change it for all models together.
- The lab notes' NS provenance section lists the physical parameters as unknown; the budget check now corroborates them, but treat them as provisional.

---

## 9. What to report back

For each work package attempted:

1. What was implemented, where, and which interface it implements.
2. The tests added and their status.
3. The experiment run: configs, seeds, compute.
4. The result table with baselines, and whether the falsifier fired.
5. Anything that turned out to be NS-specific and how it was isolated.
6. Open issues and what you would do next.

Keep a running `research/ito_maps/STATUS.md` update, and state plainly which claims in this document held and which did not.

---

## Appendix A. References and their verification status

Opened and read during the literature pass: Pan et al., "Itô maps for any-step SDEs", arXiv:2606.11156; Strong Stochastic Flow Maps, arXiv:2606.01086; improved SSFM, arXiv:2610.29605; Kolmogorov–Chapman maps, arXiv:2611.12880; StocBench, arXiv:2608.22309; Stochastic Transition-Map Distillation, arXiv:2605.07661; source-space pCN with flow maps, arXiv:2610.01034.

Identified from search results or abstracts only; verify before citing: 2305.17209 (functional flow matching), 2502.02483 (distributional diffusion), 2503.07565 (IMM), 2509.25631 (Swift), 2605.06916 (Tyche), 2505.13447 (MeanFlow), 2512.15702 (Resampling Forcing), 2503.18731 (Thermalizer), 2510.25769 (Neural Stochastic Flows), 2402.04467 (DySLIM), 2504.05852 (energy-consistent interpolants), EqualSNR.

"Not found" in this document means not found in the searches that were run. One discrepancy is open: improved SSFM states that Pan et al. add the increment `W_t − W_s` to the KL features, while the version of Pan et al. that was read says the increment is withheld from the network.

## Appendix B. Novelty map

| Package | Status |
|---|---|
| WP1 matched schedule and path-independence criterion | not found in prior work; coloured noise itself is known |
| WP2 invariant closure | low novelty; driver null-space condition and closure under self-distillation appear new |
| WP3 Legendre features | prior art (2606.01086, 2610.29605) |
| WP4a Picard target | not found, from one search round |
| WP4b weak closure | scoring rules are well covered; conditioning on a shared coarse path inside two-time self-distillation appears new |
| WP4 terminal anchor | prior art; control |
| WP5 refresh | training half exists for video (2512.15702); conditional invariance, matched use and the bound appear new |
| WP6a unrolled scoring | prior art; baseline |
| WP6b closure with direct-lead teachers | related losses exist in other settings; this form not found |
| WP6c window matching | modest; marginal matching is known |
| WP7 budget consistency | no conditional-moment budget loss found |
| WP8 path-space conditioning | related latent-space samplers exist; analytic gain from a known driver covariance for a forecast kernel not found |
