# Few-step conditional Itô maps: diagnosis and ranked proposals

Research only; nothing in the repo was modified. Scratch scripts used for the numerical checks are next to this file (`kern.py`, `integ.py`, `matched.py`).

Status labels used below: **[verified]** = derived and checked numerically here on a scalar Gaussian toy; **[derived]** = derived by hand, not checked numerically; **[conjecture]** = mechanism argument, untested on NS.

---

## 1. Diagnosis from first principles

### 1.1 What a 1-step Itô map from a point mass is

With `X_0 = c(h)` deterministic, `Φ_{0,1}(c;W) = c + G_{0,1}(c,h;Ψ) + M_1` is a **noise-to-data generator** whose only stochastic inputs are the path features. `M_1` is Ψ-measurable, so the MSE optimum is `Φ_{0,1} = E[X_1 | Ψ]`. Two consequences:

- The explicit martingale contributes spread `a/√3` per pixel, white in space. Everything else must come from the **learned Ψ-dependence of G**, which enters through a zero-initialised 1×1 branch and is zero on the diagonal. It is never supervised by data, only bootstrapped by self-distillation.
- Any shortfall in the learned Ψ-dependence is a regression towards `E[G | x]`, a path-ignorant ("frozen-drift", Euler-like) map.

### 1.2 The amplification factor (the core finding)

Take the Gaussian surrogate `y | h ~ N(m, σ²)` per mode. The SDE is linear, `b_t(x) = A_t (x − mean_t) + mean_t'`, and `X_1 = m + ∫_0^1 k(r) dW_r`, with `k(r) = γ_r exp(∫_r^1 A)` and `∫k² = σ²` **[verified]**. A path-ignorant map has endpoint std `a/√3` only. So the learned path dependence must supply the factor

    amplification = σ / (a/√3).

For NS, `a = 0.1` (`config/stochastic_navier_stokes_pde_transformer.yaml`). From the notes' Itô-1 CRPS (≈0.305 physical, scale 3.1, Gaussian CRPS = σ/√π) I infer `σ ≈ 0.17` normalised (rough, unverified), so amplification ≈ 3 in std, i.e. **≈ 90% of the forecast variance must be created by the bootstrapped Ψ-branch**. The error field is red and the noise is white, so the factor is scale dependent: ≫ 1 at low wavenumber (drift must amplify noise) and ≪ 1 at high wavenumber (drift must cancel white noise).

Path-ignorant limit, exact Gaussian drift, β = t², a = 0.1, σ = 0.17 (endpoint std / σ) **[verified]**:

| N steps | 1 | 2 | 4 | 8 | 16 | 64 |
|---|---|---|---|---|---|---|
| Euler drift + exact noise | 0.34 | 0.58 | 0.65 | 0.78 | 0.88 | 0.97 |

This single mechanism predicts, with the correct sign:

- **Underdispersion** of every under-learned map (the missing part is positively correlated with the noise).
- **High-wavenumber spectral error**: uncancelled white noise where amplification < 1. A 5% cancellation error is invisible in MSE and dominant in a steep spectrum.
- **"PSD alone gave half the spread; teacher rollout helped a lot"**: rollout supplies the amplified target directly, while PSD must bootstrap it through ~7 dyadic levels with shrinkage at each.
- **K16 did not beat K4**: information is not the bottleneck. In the Gaussian surrogate, K=4 increments of `M` already capture > 99% of endpoint variance for β = t² and σ/a in 1–3 **[verified]**; the measured NS loss (≈10% at K4) is therefore mostly nonlinear. The bottleneck is learning a large gain through the feature branch.
- **The Gaussian-base flow map wins at one step**: there the noise is the *state*; the Gaussian part of the transport is a linear map of the network input, and the MSE target is a deterministic function of the inputs (no `Var(G|Ψ)` loss).
- **The solver reference is itself underdispersed at finite N** (0.88 at 16, 0.97 at 64 in the toy), and a 32-Euler-step rollout teacher caps the student below 1.

Non-monotonicity in N is also reproducible in the path-ignorant limit for some schedules (β = t² with a = √3σ: 1.00, 0.53, 0.74, 0.85 at N = 1, 2, 4, 8 **[verified]**), though for the present a/σ the toy is monotone. Ideal conditional-mean maps improve monotonically in N for a linear SDE, so the observed non-monotonicity is a learning artefact (intermediate intervals less trained, off-policy start states), not compression.

**Sharp falsifiable prediction (no training needed):** the per-wavenumber spread ratio of Itô-1 vs SI-128 is < 1 at low k and > 1 at high k, and zeroing Ψ at sampling time in an existing checkpoint drops low-k spread towards `a/√3`.

### 1.3 Secondary observations

- **KL features, train/sample mismatch [derived]:** training draws `x = I_s` independently of the global coefficients ξ, but at sampling `X_s` depends on `W_{[0,s]}`, which is correlated with ξ. The exact map is unaffected; the learned `E[G | x, ξ]` is taken under the wrong joint. Pan et al. report local KL on `[s,1]` works better, consistent with this.
- **Exact semigroup in law needs nested features [derived]:** conditional kernels `Law(X_t | x, Ψ_{s,t})` satisfy a Chapman–Kolmogorov identity only if the parent features are a function of the children's (midpoint split, K even). The current `u ∈ [0.25, 0.75]` split is not nested.
- **Loss weighting [derived]:** in the integrating-factor form (Idea 4), G-space MSE equals denoiser-space MSE times `1/(1−s)²` (β = t). The current loss therefore weights late-start intervals up to 10⁴× more than `(0,1)` in endpoint terms.

---

## 2. Ranked ideas

### Idea 1 — Gaussian-exact (variance-clock-matched) Itô maps

**Pitch.** Do not ask the network to learn a 3–30× scale-dependent noise gain. Choose the interpolant so that, for the Gaussian surrogate of the forecast law, the average drift is *exactly path independent*. The exact, free, explicit martingale then carries the whole Gaussian part of the spread, and Ψ only has to encode non-Gaussian and state-dependent corrections.

**Math [verified on the scalar toy, derived for commuting covariances].** Let `I_t = (1−t)c + β_t y + γ_t C^{1/2} W_t`, `q(t) = ∫_0^t γ²`, target `N(m, Σ)`. The drift is state independent iff `Cov(R_t, I_t) = 0`, i.e.

    β_t β_t' Σ = −γ_t γ_t' t C   ⇔   β_t² Σ = [q(t) − t γ_t²] C.

With `β_1 = 1, γ_1 = 0` this forces `q(1) C = Σ` and `β_t² = [q(t) − tγ_t²]/q(1)`. For `γ = a(1−t)`:

    β_t = t √(3 − 2t),   β_t' = 3(1−t)/√(3−2t),   a_k = √3 σ_k per Fourier mode.

Then `b_t = −c + β_t' m`, `G_{s,t}` has no path dependence, and `Φ_{0,1} = m + M_1 ~ N(m, Σ)` exactly with K = 0. Path-ignorant spread ratio is 1.00 at every N (toy); with the same matched `a` but β = t² it is 1.00, 0.53, 0.74, 0.85.

**Why it helps.** Removes the systematic sign of the dispersion error (failure 1), removes white noise at wavenumbers where the data have none (spectral error), makes Euler exact for the Gaussian part (better teacher and solver baseline, monotone in N), and shrinks `Var(G|Ψ)` to the non-Gaussian remainder (K matters less).

**Novelty.** Coloured or GP noise in diffusion and flow forecasting is known (functional flow matching 2305.17209; data-dependent couplings 2310.03725). Pan et al. (2606.11156) pick `σ_t = √(2(1−t))` only to remove a singularity. I found no work that selects `(β, γ, C)` by the criterion "average drift path independent for the Gaussian surrogate", nor the amplification diagnosis. Moderate-to-high confidence it is new in this context.

**Risks.** Forecast-error covariance is state dependent, so residual amplification ≠ 1 remains (handled by Ψ). `σ_k` must be estimated (spread spectrum of existing SI-128 ensembles, or residual of a deterministic regression); a floor is needed at high k. It changes the process, so SI baselines must be retrained with the same interpolant. Chen et al.'s regularity assumptions under coloured noise are unchecked.

**Falsifying experiment.** (a) Measure `σ_k/(a/√3)` from SI-128 ensembles; if ≈ 1 across k the diagnosis is wrong. (b) Train only the SI drift with the matched interpolant; if its Euler sampler at N = 1, 2, 4 is still strongly underdispersed relative to its own N = 128, the non-Gaussian part dominates and the idea fails.

**Implementation.** `src/scisi/models/interpolations.py`: new interpolation class (β above, optional radial Fourier multiplier for noise). `src/scisi/sampling/brownian_path.py`: apply the same multiplier to every Gaussian drawn in `sample_like` and the bridges (colouring is linear and commutes with the time bridge). `FollmerItoMapModel.__init__`: the γ check already passes. `MapTrainer._compute_loss`: colour `torch.randn_like` and `batch["noise"]`. Baseline stepper diffusion term coloured likewise.

---

### Idea 2 — Weak closure: conditional energy-score self-distillation with a latent η

**Pitch.** Be strong (pathwise) on the K resolved features and weak (in law) on everything else. Give the network one extra white field η and replace the off-diagonal MSE by a conditional energy score against the same composed EMA target. Shrinkage towards the conditional mean then becomes impossible at the optimum: under-learned path dependence turns into mis-coupling, not lost variance.

**Math [derived].** Student `S(η) = G_θ(s,t,x,h;Ψ_{s,t},η)`, target `T` from two EMA half-steps (each with its own η, features from the bridge refinement):

    L = E[ ½Σ_{i=1,2} ‖S(η_i) − T‖ − ½‖S(η_1) − S(η_2)‖ ].

Strictly proper, so the optimum has `Law(S | x,Ψ_{s,t}) = Law(T | x,Ψ_{s,t})`. With nested features (midpoint split) the true kernels satisfy `Q_{s,t}(·|x,Ψ_{s,t}) = E[Q_{u,t} ∘ Q_{s,u} | Ψ_{s,t}]` by the Markov property and independent increments. The true kernels are therefore an exact fixed point at every finite K, which MSE-PSD does not have (OU counterexample in the notes). Uniqueness given the diagonal is **unverified**. Data anchors in the same loss: `Law(Φ_{0,t}(c;W,η)) = Law(I_t | h)` for every t, scored against `I_t = α c + β y + γ√t z` with fresh z; t = 1 is the usual CRPS-on-truth.

**Why it helps.** Underdispersion from finite features and from bootstrapped shrinkage; restores a weak semigroup guarantee (reviewer item P1).

**Novelty.** Scoring-rule training is well covered: distributional diffusion (2502.02483), IMM (2503.07565), VarFlow, energy-distance distillation for weather (2608.27728), CRPS fine-tuning of one-step forecasters (Swift 2509.25631, Tyche 2605.06916), stochastic consistency distillation with a Gaussian student (SCott 2403.01505), STMD (2605.07661, fresh latent plus bridge kernel, in law only). What differs: the score is *conditional on a shared coarse Brownian path*, with exact bridge refinement supplying the teacher's conditional randomness, and it is used for self-distillation of a two-time map. Terminal CRPS alone is **not** novel and should be run as the control.

**Risks.** Energy score is weakly discriminative in 16k dimensions (use per-pixel fair CRPS plus a spectral or patch term). Two student passes per example. η must vanish on the diagonal (scale by √(t−s), zero-init).

**Falsifying experiment.** Fine-tune an existing K=4 checkpoint for the notes' 500-update budget with (i) MSE-PSD, (ii) ES-PSD + η, (iii) terminal CRPS only. If (ii) does not improve rank histograms over (i), or adds nothing over (iii) at 2–4 steps, drop it.

**Implementation.** `map_conditioning.py`: η as extra path channels. `map_trainer.py`: new entry in `OFF_DIAGONAL_LOSSES`, midpoint `u` in `_intervals`, an `s = 0` anchor term. `follmer_ito_map_model.py`: draw η in `average_velocity`.

---

### Idea 3 — Picard self-distillation (randomised-quadrature Lagrangian)

**Pitch.** Teacher rollout worked because it evaluates the *data-anchored diagonal* along the path. Do the same without simulation: the integral form of the Lagrangian identity, estimated with one random node.

**Math [derived].** `(t−s) G_{s,t}(x;W) = ∫_s^t b_u(Φ_{s,u}(x;W)) du`, hence

    L = E_{u~U(s,t)} ‖ G_θ(s,t,x;Ψ_{s,t}) − sg[ b_ema(u, Φ_ema,s,u(x;W)) − (β_u' − (β_t−β_s)/(t−s)) y ] ‖².

The bracketed control variate has zero u-mean for any u-independent field, so using the sample's own `y` is unbiased and removes the dominant deterministic u-variation. Cost: 2 EMA + 1 student evaluation, same as PSD; no JVP, valid for local features and bf16. The target operator is Volterra: `e(s,t) ≤ (t−s)^{-1} ∫ L_u (u−s) e(s,u) du`, so idealised Picard iteration converges factorially for any horizon, whereas the PSD operator passes child errors with unit gain and never reaches the anchored diagonal (gap floor 0.01). The contraction claim is for function-space iteration, **unverified** for SGD.

**Why it helps.** "PSD alone insufficient", the support gap (small-gap map errors enter multiplied by the gap), and Lagrangian instability.

**Novelty.** MeanFlow (2505.13447) states the integral but differentiates it; Boffi et al.'s LSD and Pan et al.'s loss are differential; SplitMeanFlow and Euler Mean Flows (2602.02571) are JVP-free via the semigroup. I found no random-node integral target; confidence moderate (one search round).

**Risks.** Monte Carlo variance over u; Lipschitz constant ~ `1/(1−u)` near t = 1 (mild under Idea 1, where the Gaussian part of b is state independent).

**Falsifying experiment.** Equal-budget PSD vs Picard from the same diagonal checkpoint: teacher-map validation loss (the 0.034 plateau) and Itô-1 spread. No gain over PSD falsifies it.

**Implementation.** One new function in `map_trainer.py` (`u` already sampled; pass `batch["target"]` and `interpolation.beta_diff`).

---

### Idea 4 — Bridge-form (integrating-factor) map, for the existing white-noise process

**Pitch.** If the process is kept as is, make noise damping structural instead of learned.

**Math [verified pathwise to 1e-5 on a nonlinear toy].** `b_t(x) = (γ'/γ) x + (β' − βγ'/γ) ŷ_t(x)` with `ŷ_t = E[y | I_t = x]` (the c term cancels for α = 1−t, γ = a(1−t)). Integrating `d(X/γ)`:

    Φ_{s,t}(x) = (1−t)/(1−s) · x + [β_t − β_s (1−t)/(1−s)] · Ȳ_{s,t} + a(1−t)(W_t − W_s),

where `Ȳ_{s,t}` is the `(β/γ)'`-weighted average of `ŷ_r(X_r)` on `[s,t]`. The noise term vanishes at t = 1; `Ȳ_{s,1} = X_1`; `Ȳ_{t,t} = ŷ_t`; diagonal loss is plain denoiser regression on y.

**Why it helps.** High-k spectral error (no white noise to cancel in any final step), stiffness as t → 1, and it puts outputs in data space for Idea 2. It does **not** by itself fix underdispersion: the population optimum is unchanged and a path-ignorant 1-step gives the conditional mean.

**Novelty.** Stochastic analogue of the CTM (2310.02279) and TCD (2402.19159) exponential-integrator parametrisations; Pan et al. use the average-drift form only. Incremental.

**Relation to Idea 1.** Alternatives for the same problem. Under matched forcing the G-form is already the natural one; use Idea 4 only if the process must stay fixed.

**Falsifying experiment.** In an existing checkpoint, regress high-pass(sample) on high-pass(`M_1`); a zero coefficient means cancellation is already exact and the idea is moot.

**Implementation.** `base_map_model.map` and `noise_increment` (needs `W` increments alongside `M`), `training_pair` target → `y`.

---

## 3. Considered and discarded

- **Unweighted W-increment or response-kernel features.** I expected the linear-response kernel to be flat; the calculation shows it is front-loaded and M-increments are near optimal (β = t², σ/a in 1–3: K=4 loss 0.5–0.8% for M vs 1.4–2.3% for W). Falsified.
- **Larger K / learned linear path features.** Information is not the bottleneck.
- **Terminal CRPS fine-tuning alone.** Likely effective, not novel (Swift, Tyche, 2608.27728). Keep as control.
- **KL-optimal Föllmer process + posterior sampler + bridge kernel.** Transitions become data-samplable, but this is STMD / distributional diffusion and abandons pathwise maps. Strong baseline, not a contribution.
- **Endpoint-weighted `(t−s)²` loss.** Starves short intervals; does not touch the mechanism.
- **Post-hoc variance inflation.** Breaks spectra; unprincipled.
- **Eulerian / MeanFlow identities.** Pan et al. Prop. 2: W-independent zero-loss solutions.
- **Probability-flow flow map from `N((1−t0)c, ·)`.** Is the existing flow-map baseline with an informative base.
- **Normalised features or feeding `M_t − M_s` directly.** Discontinuous at the diagonal; Pan et al. keep the increment out of the input.
- **Adversarial anchoring (CTM, SCott).** Works, heavier and less stable than a scoring rule.
- **Dyadic interval sampling, on-policy start states, gap floor → 0.** Sensible hygiene (Shortcut 2410.12557, FlowMap-OPD 2609.37851); not novel. Midpoint splits are required by Idea 2 anyway.

## 4. Suggested order

1. Run the two no-training diagnostics of §1.2 on existing checkpoints (amplification spectrum; Ψ-ablation).
2. Idea 1 on the SI drift only (cheap, decisive), then the map with K ≤ 2.
3. Idea 3 as the off-diagonal loss; Idea 2 on top if rank histograms are still U-shaped.
4. Idea 4 only if the process must remain the published one.

## Sources

- Pan et al., Itô maps for any-step SDEs, arXiv:2606.11156
- Boffi et al., arXiv:2406.07507 and arXiv:2505.18825
- De Bortoli et al., Distributional diffusion models with scoring rules, arXiv:2502.02483
- Zhou et al., Inductive moment matching, arXiv:2503.07565
- STMD, arXiv:2605.07661; SCott, arXiv:2403.01505; CTM, arXiv:2310.02279; TCD, arXiv:2402.19159
- Swift, arXiv:2509.25631; Tyche, arXiv:2605.06916; Diffusion distillation for weather ensembles, arXiv:2608.27728
- MeanFlow, arXiv:2505.13447; Euler Mean Flows, arXiv:2602.02571; flow-map training stability, arXiv:2610.20674; FlowMap-OPD, arXiv:2609.37851
