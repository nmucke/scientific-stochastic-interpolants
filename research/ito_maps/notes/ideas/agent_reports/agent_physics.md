# Physics consistency for few-step conditional Itô maps: ranked proposals

Scope: research only, no repo changes. Numerical checks were run on `data/stochastic_navier_stokes/data.npz` and on scalar toys; scripts are in this scratchpad directory (`budget_check.py`, `spec_check.py`, `toy_matched.py`, `toy_sens.py`).

Notation: interpolant `I_t = α_t c + β_t y + γ_t C^{1/2} W_t`, SDE `dX = b dt + γ_t C^{1/2} dW`, `X_0 = c`, martingale `M_t = C^{1/2}∫γ dW ~ N(0, q(t) C)`, `q(t) = ∫_0^t γ²`. Today `C = Id`, `γ = a(1−t)`, `a = 1`, `β = t²`. Map: `Φ_{s,t}(x) = x + (t−s) G_θ + (M_t − M_s)`. Physical SPDE: `dω = (−u·∇ω + νΔω − αω)dτ + ε dξ`, `Δτ = 0.5`, forcing on `K = {(6,0),(7,0),(5,5),(8,8)}`.

## Organising thesis

In an Itô map the only random object is a Gaussian martingale that enters **additively and linearly**. Everything the network does is deterministic given the path. So the natural place for physics is the martingale:

- its **null space** carries exact invariants (idea 5),
- its **covariance** carries the physical conditional uncertainty (idea 1),
- **Gaussian conditioning** of it carries observations (idea 3),
- **Itô budgets** of the physical SPDE fix its weighted trace (idea 2),
- and part of it can be the **physical forcing itself** (idea 4).

## Two empirical findings from this session (both new, both checkable)

**F1. The energy and enstrophy Itô budgets close on the existing dataset to about 0.5%, and they corroborate the "unknown" parameters.**
With `Z = ½⟨ω²⟩`, `P = ½⟨|∇ω|²⟩`, `E = ½⟨ω(−Δ)^{-1}ω⟩` (area means, all 200×100 fields, 128²):

| Check | Data | Theory (ν=1e-3, α=0.1, ε=1) |
|---|---|---|
| Stationary enstrophy balance `2αZ̄ + 2νP̄` | 1.991 | 2.000 |
| Stationary energy balance `2αĒ + 2νZ̄` | 0.03783 | 0.03800 |
| `(α, ν)` solved from the two balances with ε=1 | (0.1005, 1.004e-3) | (0.1, 1e-3) |
| Std of the per-pair enstrophy-budget residual | 0.722 | 0.720 (quadratic variation `ε²Δ Σ_{k∈K}|ω̂_k|²`) |
| Mean per-pair trapezoid residual / injection | −0.5% ± 0.5% (enstrophy), −0.6% ± 0.5% (energy) | 0 |
| Correlation of residual with `Z(c)`, `P(c)`, `E(c)` | all within ±0.011 | 0 |

The balances fix `ν/ε²` and `α/ε²`; the martingale variance fixes `ε²Δ`. Together they support ν=1e-3, α=0.1, ε=1, Δ=0.5 for the archived data. This is evidence, not a recovered manifest.

**F2. The white driver is mismatched to the conditional uncertainty by about four decades across scales.**
In normalised coordinates, per-mode power of the one-interval increment `y − c` (an upper bound on conditional variance) relative to the white endpoint noise `Var(M_1) = a²/3`:

| Shell `|k|` | 1 | 4 | 7 | 11 | 16 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|
| increment / white | 0.31 | 20 | 190 | 59 | 9.8 | 0.59 | 0.066 | 0.012 |

The forced shells are under-noised by ~10², the small scales over-noised by ~10². A one-step map therefore has to cancel per-pixel noise exactly at high `k` (`G ≈ −M_1 + …`) and has too little noise where the physics is random.

---

## Ranked ideas

### 1. Covariance-matched martingale (physical driver colour + matched schedule)

**Pitch.** Choose the driver covariance and `β` so that the martingale already *is* the conditional uncertainty. The drift then becomes (nearly) state-independent, the average drift `G` (nearly) path-independent, and the finite-feature compression error, which is the documented source of underdispersion, vanishes to first order.

**Math.**
Proposition (Gaussian exactness; proved, and checked numerically). Let `y | h ~ N(m, Σ)`, `C = Σ / q(1)`, and

```
β_t² = (q(t) − t γ_t²) / q(1) = −(2/q(1)) ∫_0^t r γ_r γ̇_r dr.
```

Then `b_t(x; h) = α̇_t c + β̇_t m` for every `x`, `X_t = α_t c + β_t m + M_t`, and `G_{s,t} = [(α_t−α_s)c + (β_t−β_s)m]/(t−s)` does not depend on the path. Any step count is exact with zero path features.
Proof sketch: with `u = I_t − E I_t` and `r = R_t − E R_t`, `Cov(r,u) = (β β̇ q(1) + t γ γ̇) C = 0` by the definition of `β`, so `E[r | u] = 0`.
For `γ = a(1−t)`: `β_t = t√(3−2t)` (so `β² = 3t² − 2t³`, the smoothstep), independent of `a`; `β̇(0) = √3`, `β̇(1) = 0`, bounded.

Design criterion in general: minimise `∫ E‖∇_x b_t(I_t)‖² dt` over `(C, β)`; path dependence of `G` is bounded by `∫ Lip(b_r)‖δX_r‖ dr`. This generalises "straight trajectories" for ODE flow maps: an Itô map is one-step exact iff its drift is spatially constant along the path. (General non-Gaussian bound: unverified beyond the toy.)

**Mechanism here.**
- `C` = Fourier-diagonal mean conditional covariance `Σ̄_k = E_h Var(ŷ_k | h)`, estimated from residuals of an MSE regressor `m(h)` (its residual is exactly `y − E[y|h]`) or from anomalies of existing SI-64 ensembles. Keep the full 2D `(k_x,k_y)` spectrum (forcing is anisotropic), floor it, set `C_0 = 0`.
- Per-mode SNR `β²σ_k² / (γ² t κ_k²)` becomes scale-independent, so all scales are "decided" at the same generative time.
- History-dependent `C(h)` is allowed (h is frozen during generation), e.g. a scalar `λ(h)²` from the budget of idea 2. `C` must not be learned through the diagonal loss (the objective changes with `C`; ill-posed).
- The exact variance clock, bridges, KL features, PSD and LSD are unchanged; `C^{1/2}` acts on the spatial index only.

**Toy evidence** (exact drift for Gaussian-mixture targets, fraction of `Var(X_1)` not explained by `M_1` alone, i.e. K=1):

| Driver variance / target variance | β=t² | β=t | matched β |
|---|---|---|---|
| 1 (Gaussian target) | 5.0% | 1.2% | 0.1% (theory 0) |
| 1 (skewed / bimodal) | 5.2 / 5.7% | 1.3 / 1.4% | 1.2 / 0.9% |
| 2 | 7.7% | 3.5% | 1.2% |
| 4 | 10.6% | 6.9% | 4.0% |
| 100 (today's high-k regime) | 20–22% | – | 19–21% |
| 0.5 | 3.2% | 1.3% | 5.1% |
| 0.25 | 2.4% | 4.6% | 12.9% |
| 0.01 (today's forced-shell regime) | 13–17% | – | 64–69% |

Readings: (i) covariance matching is the large lever (≈20% → ≈1%); (ii) matched `β` is best for variance ratio in [1, 4] and harmful when the driver is under-noised; rule: over-noise, `C ≈ 1–2 × Σ̄`, or keep `β = t`. (iii) The 100× row (20% at K=1, 3–6% at K=4) is the same order as the lab-note nested Monte Carlo (28% at K=1, 9.7% at K=4).

**Novelty.** Closest work and differences:
- EqualSNR (Falck et al. 2025): colours diffusion noise by the *unconditional* data spectrum for image models. Here: *conditional* covariance, point-mass source, plus the schedule identity and the path-independence consequence specific to Itô maps.
- Lim et al. 2024 (probability paths for forecasting), TORNADO (2611.10251, state-dependent latent diffusion), Chen et al. 2024 (Föllmer changes `g`, keeps white noise): none targets path dependence of the average drift.
- Pan et al. 2026 and Strong Stochastic Flow Maps (2606.01086): white identity-covariance noise only.
Not found in search: the schedule `β² = (q − tγ²)/q(1)` or "state-independent drift" as a design criterion for stochastic flow maps.

**Risks.** Conditional covariance is history-dependent and non-stationary in space; under-noised directions hurt with matched `β` (table). Smooth inputs change what the U-Net sees. The NS conditional law may be far from Gaussian beyond one turnover time (Δ=0.5 vs turnover ≈0.3).

**Minimal falsifying experiment.** Repeat the lab-note nested Monte Carlo (omitted endpoint-variance fraction at K=1, 4) with the *solver* teacher under (white, t²), (coloured, t), (coloured, matched). No map training needed, only one retrained SI teacher per interpolant. Falsified if the K=4 omitted fraction does not fall well below 9.7%. Then: one-step map spread–skill and extreme-rank fraction.

**Implementation sketch.** Spectral filter at the single white-noise draw of `_WeightedPathBase._randn` (`src/scisi/sampling/brownian_path.py`) and in the interpolant noise; a `MatchedStochasticInterpolation` with `beta = t·sqrt(3−2t)` in `src/scisi/models/interpolations.py`; the `gamma` probe in `FollmerItoMapModel` stays valid.

---

### 2. Itô-budget consistency: the weak physics residual for a stochastic PDE

**Pitch.** For an SPDE the right residual is not pointwise; it is the Dynkin identity for observables whose generator closes. Energy and enstrophy do, because the Euler nonlinearity drops out. Their Itô corrections give two exact, label-free constraints on the forecast ensemble **spread**.

**Math.** With `D = α − νΔ ≥ 0`: `dZ = (−⟨ω, Dω⟩ + ½ε² tr Q) dτ + ε⟨ω, dξ⟩`, and the same with weight `(−Δ)^{-1}` for energy. A trapezoid in τ (error `O(Δ³)`, verified in F1) gives, for `𝒲 ∈ {Id, (−Δ)^{-1}}`,

```
E[ q⁺_𝒲(y) | h ] = q⁻_𝒲(c) + ½ ε² Δ tr(𝒲Q),     q^±_𝒲(x) = ½⟨x, 𝒲(Id ± ΔD) x⟩.
```

Because `q⁺` is quadratic, `E q⁺(y) = q⁺(E y) + ½ tr(𝒲(Id+ΔD) Cov(y|h))`, so

```
tr(𝒲(Id+ΔD) Cov(y|h)) = 2[ q⁻_𝒲(c) + ½ε²Δ tr(𝒲Q) − q⁺_𝒲(E[y|h]) ].
```

Physics pins two differently weighted traces of the conditional covariance (enstrophy: all scales; energy: large scales). A second-moment budget also holds: `Var(residual | h) ≈ ε²Δ Σ_{k∈K} |ω̂_k|²` (matched 0.722 vs 0.720).
Numbers: injection per step is 1.0 in `Z` units against `Z̄ ≈ 4.78`, so the spread term is a well-conditioned ~20% effect.

**Mechanism.**
- *Label-free diagnostic, usable today:* per member and lead, `r = q⁺(y_{n+1}) − q⁻(y_n) − inj` must be a mean-zero martingale difference with known variance. This tests the kernel on the model's own encountered histories and beyond lead 95, where no truth exists.
- *Training loss:* few-step maps give real samples in one differentiable call, so the constraint is on samples, not on a denoised mean (no Jensen gap). Use the unbiased U-statistic `L = (1/(E(E−1))) Σ_{e≠e'} r_e r_{e'}` (E=2 suffices). The naive `(mean r)²` penalises `Var(q(y))` and would worsen underdispersion. It needs no target, so it can be applied on rolled-out histories.
- *Inference:* budget-closing inflation `λ²(h) = 2[ρ(c) − q⁺(ȳ)] / tr_q(Ĉov)`, or `λ(h)` as the driver scale `C(h) = λ² C_0` of idea 1.

**Novelty.** PIDM (Bastek et al.), DiffusionPDE, CoCoGen, PCFM/ECI use deterministic pointwise residuals or hard constraints, which do not vanish for an SPDE. The energy-consistent interpolant paper (Mücke and Sanderse, 2504.05852) applies an Itô budget to the *interpolant in pseudo-time* for a deterministic PDE with `k_τ = 0`; here the budget is the *physical-time* SPDE budget with the injection term and constrains ensemble covariance. CRPS plus spectral losses (2509.09599) are generic scores that need truth. No Dynkin-type conditional-moment loss was found in search.

**Risks.** Two scalars constrain two traces only; a biased mean can be compensated by wrong spread, so use a small weight and always report the diagnostic. Requires known `ν, α, ε, Q`. Anti-aliased downsampling would add a small cut-off flux (`k_d ≈ 35 < 64`; energy budget is insensitive). Shell-wise budgets need the cubic transfer term; trapezoid closure for those is unverified.

**Minimal falsifying experiment.** Zero training: evaluate `r` on existing SI-64, Itô-1, Itô-8 rollouts. Falsified as a useful signal if the known ~5 pp coverage deficit of Itô maps does not appear as a negative mean residual.

**Implementation sketch.** `research/ito_maps/` script with FFT quadratic forms on denormalised float64 fields; optional term in `MapTrainer._compute_loss` using two forks of one history through `model.map(c, 0, 1, path)`.

---

### 3. Martingale Kalman conditioning (observation and constraint consistency in path space)

**Pitch.** Maps have no trajectory to guide, but the one-step map is `y = c + G_θ(c,h;Ψ) + SΨ` with a Gaussian `Ψ` entering linearly through `S` (the sum of increments). Condition the driver in closed form and correct the rest with a prior-reversible MCMC that costs one network call per step and no gradients.

**Math.** Observations `o = Hy + η`, `η ~ N(0,R)`, `Ψ ~ N(0, D⊗C)`, `Σ_j D_j = q(1)`.
- Reference measure: freeze `G` at `g_0 = G(Ψ_0)`. The linear-Gaussian posterior `μ_ref = N(μ_p, Σ_p)` is sampled exactly by Matheron's rule:
  `Ψ⁺ = Ψ + (D1 ⊗ C Hᵀ)(q(1) H C Hᵀ + R)^{-1}(o + η̃ − H(c + g_0 + SΨ))`.
- Exact correction: pCN with respect to `μ_ref`, `Ψ' = μ_p + √(1−ρ²)(Ψ − μ_p) + ρ ξ_p`, accept with the residual likelihood ratio
  `ℓ(Ψ) = exp(−½‖o − HΦ(Ψ)‖²_R + ½‖o − H(c + g_0 + SΨ)‖²_R)`.
  If `G` is path-independent (idea 1's Gaussian limit) acceptance is 1 and `ρ = 1` gives i.i.d. exact posterior samples.
- Hard linear constraints (`R → 0`): Picard iteration of the Matheron update gives `HΦ(Ψ*) = o` exactly when `‖H ∂_ΨG K‖ < 1`; this is an RML-type approximation, not an exact posterior draw (flagged).

**Novelty.** Pan et al. steer with Monte Carlo h-transform estimators (Itô-G, Itô-GF, BEL) in multi-step controlled rollouts; no conditioning of the path. D-Flow and noise-space Langevin (Purohit et al. 2024) need gradients through the generator and treat the latent generically. Latent EnKF, EnFF and FlowEF use sampled ensemble covariances. PCFM/ECI project along a multi-step flow. New here: the explicit additive martingale gives an analytic gain with the *known* prior covariance, exactness in the Gaussian limit, and a dimension-robust exact correction.

**Risks.** With the white driver `∂G/∂Ψ ≈ −S` at small scales, so the frozen-Jacobian reference is poor; the idea depends on idea 1. pCN mixes slowly for dense, accurate observations. The posterior inherits map error.

**Minimal falsifying experiment.** Linear-Gaussian toy where the answer is the Kalman posterior, then NS with sparse point observations: compare acceptance rate and posterior CRPS against the existing multi-step `StochasticInterpolantPosterior`; falsified if acceptance stays below ~10% with the coloured driver.

**Implementation sketch.** New posterior class beside `src/scisi/posterior_models/`, reusing `LinearObservationOperator`; `H C Hᵀ` from the stationary kernel (inverse FFT of the spectrum).

---

### 4. Forcing-revealed latents: tie the stochastic input to the physical Wiener path (needs the regenerated dataset)

**Pitch.** The true randomness is an 8-dimensional Wiener path per interval. Log it. Condition the map on a few of its Wiener-chaos coefficients and sample them exactly at inference; the generative driver only models the residual.

**Math.** `ψ_{k,j} = Δ^{-1/2}∫ e_j(τ) dξ_k(τ)`, i.i.d. `N(0,1)`, independent of `h`. Then `p(y|h) = E_ψ p(y | h, ψ)` exactly. Feed `ψ` as fields `f_j(x) = Σ_k ψ^c_{k,j} cos(k·x) + ψ^s_{k,j} sin(k·x)`; `ε√Δ f_0` is the exact physical martingale increment `ε(ξ_{τ+Δ} − ξ_τ)` and can be added in closed form. Residual `Var(y | h, ψ_{<J})` sets a much smaller generative `a`.

**Mechanism and payoff.** Spread is carried by exactly sampled physical Gaussians; data assimilation becomes inference over ~8J numbers (learned weak-constraint 4D-Var; idea 3 in a tiny space); common random numbers for twin experiments.

**Novelty.** Neural SPDE (Salvi et al.), SPDEBench and TRIE learn deterministic maps of the *full* noise path; Wiener-chaos latent models (2602.11794) infer forcing for linear SPDEs. A hybrid with truncated physical chaos coefficients plus a generative residual was not found.
Discarded variant: coupling the generative `W` to the physical forcing inside the interpolant breaks the independence that the SDE drift identity needs.

**Risks.** Only for simulators that log forcing. Δ exceeds a turnover time, so within-interval timing matters and `J` may need to be large.

**Minimal falsifying experiment.** Simulator only: fix `c` and `ψ_{<J}`, resample the rest, measure `Var(y | c, ψ_{<J}) / Var(y | c)` for `J = 1, 2, 4, 8`. Falsified if below 80% explained at `J ≤ 4`.

**Implementation sketch.** Store `dW` sums in `external_libs/jax_cfd_lib/.../navier_stokes_forward_model.py::stochastic_explicit_terms`; pass `f_j` through the existing `field_cond`.

---

### 5. Invariant-closed maps (exact linear invariants for every step count)

**Pitch.** Low novelty, zero risk, do it anyway: it answers "what is preserved under composition".

**Math.** Let `L y = ℓ(h)` a.s. and `L C^{1/2} = 0`. Then `L I_t = ℓ_t := α_t Lc + β_t ℓ(h)`, `L b_t = ℓ̇_t`, and `L Φ_{s,t}(x) = Lx + ℓ_t − ℓ_s`. The parametrisation

```
G_θ = (Id − L⁺L) N_θ + L⁺ (ℓ_t − ℓ_s)/(t − s)
```

satisfies this for every θ and every step count, and the class is closed under composition, convex combination (PSD targets) and the Lagrangian identity.
With the white driver instead, `L X_t − ℓ_t = γ_t L W_t` and `L b_t(x) = ℓ̇_t − (Lx − ℓ_t)/(1−t)`: a singular pull that a few-step map must integrate exactly; this is why white-driver maps leak invariants.
Nonlinear invariants are not preserved along an affine interpolant; only `t = 1` matters, so a projection built into terminal maps keeps closure.

**Here.** Mean vorticity (verified zero to 1.5e-6 in the data): remove `k = 0` from `C` and from `G`. Velocity formulations: Leray-projected driver and output.

**Novelty.** PCFM/ECI (inference-time, multi-step), Beucler et al. (architecture constraints), PMosFM (2609.40287, one-step deterministic manifold decoder). New: the driver null-space requirement and closure under self-distillation for stochastic maps.

**Falsifier.** Invariant error at N = 1, 2, 4, 8 is machine zero after the change and non-zero before.

---

## Discarded

- Pointwise PDE-residual loss (PIDM-style): the residual is the forcing, non-zero by construction.
- Gaussian likelihood on `y − S_Δ^{det}(c)`: Δ exceeds a turnover time; needs a 5,000-step differentiable solver.
- Rank-8 driver equal to the physical `Q`: unforced directions get no noise, giving point-source collapse at `t = 0`.
- Learning `C` through the diagonal loss: ill-posed.
- State-dependent or multiplicative diffusion (TORNADO-like): loses exact martingale sampling and additive cancellation in PSD.
- Mirror/reflected diffusion: no inequality constraints here; breaks the additive structure.
- DPS/ECI/PCFM guidance transplanted to 1–8 steps: too few steps; terminal Gauss–Newton is a projection, not a posterior.
- Pan et al.'s control estimators: already published; multi-step.
- Equivariance as a contribution: known. Do the free augmentation only; note the symmetry group is translations × reflections with `ω → −ω`; 90° rotation is **not** a symmetry because `K` contains `(6,0),(7,0)` but not `(0,6),(0,7)`.
- Stationarity moment matching: implied by, and weaker than, the conditional budgets.
- Fokker–Planck regularisers on the generative SDE: generative-time consistency, not physics.
- Shell-wise spectral budgets: cubic transfer term, closure unverified; revisit later.

## Suggested order

1. Budget diagnostic on existing rollouts (hours, no training).
2. Nested Monte Carlo path-compression test with coloured driver (one teacher retrain).
3. Invariant closure together with the coloured driver.
4. Map training with matched interpolant; then path-space assimilation.
5. Log forcing in the regenerated dataset regardless (cheap, enables idea 4).

## Sources

- Pan et al., Itô maps for any-step SDEs: https://arxiv.org/abs/2606.11156
- Strong Stochastic Flow Maps: https://arxiv.org/abs/2606.01086
- Boffi et al., flow map matching: https://arxiv.org/abs/2406.07507
- Falck et al., A Fourier Space Perspective on Diffusion Models (EqualSNR): https://arxiv.org/abs/2505.11278
- Lim et al., probability paths for forecasting: https://arxiv.org/abs/2410.03229
- TORNADO: https://arxiv.org/abs/2611.10251
- Mücke and Sanderse, energy-consistent stochastic interpolants: https://arxiv.org/abs/2504.05852
- PCFM: https://arxiv.org/abs/2506.04171 ; ECI sampling: https://arxiv.org/abs/2412.01786
- PMosFM: https://arxiv.org/abs/2609.40287
- Purohit et al., noise-space Langevin posterior sampling: https://arxiv.org/abs/2410.02078
- Ensemble Flow Filter: https://arxiv.org/abs/2508.13313 ; FlowEF: https://arxiv.org/abs/2609.28015
- Neural SPDEs: https://arxiv.org/abs/2110.10249 ; SPDEBench: https://arxiv.org/abs/2505.18511 ; TRIE: https://arxiv.org/abs/2607.00196 ; Wiener-chaos latent SPDEs: https://arxiv.org/abs/2602.11794
- CRPS + spectral loss emulators: https://arxiv.org/abs/2509.09599 ; Beucler et al.: https://arxiv.org/abs/1906.06622

Literature caveat: novelty statements rest on a handful of targeted searches and abstract-level reads (Pan et al. read in more depth, first 100k characters). PMosFM, FlowEF, EnFF and TORNADO were not read in full.
