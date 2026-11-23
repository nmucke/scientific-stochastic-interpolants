# Long autoregressive rollouts with conditional Itô maps: ranked ideas

Research only; nothing in the repository was modified. Two small scripts in this scratchpad produced the numbers marked **[checked]**: `toy_check.py` (Gaussian toy) and `budget.py` (statistics of the 20 test trajectories).

Notation: generative time s,t ∈ [0,1]; interpolant I_t = α_t c + β_t y + γ_t W_t with α=1−t, β=t², γ=a(1−t); map Φ_{s,t}(x) = x + (t−s)G_θ + (M_t−M_s); h = 5-frame history, c = last frame; K_θ(h,·) = law of one forecast; σ_f = conditional (one-step forecast) standard deviation in normalised units.

---

## 0. Three facts from the data that reframe the problem [checked]

Measured on test trajectories 180–199 (`data.npz`, 128², float64):

| Quantity | Value |
|---|---|
| Field autocorrelation at lag 1 / 2 / 5 / 10 / 20 snapshots | 0.62 / 0.33 / 0.07 / 0.017 / 0.005 |
| RMS one-step increment / RMS field (normalised) | 0.86 / 0.99 |
| Mean enstrophy Z, mean palinstrophy P, k_rms = √(P/Z) | 4.67, 499, 10.3 |
| Stationary enstrophy budget 2ν⟨P⟩ + 2α⟨Z⟩ (ν=1e-3, α=0.1) | 0.999 + 0.933 = **1.93** (theory 2ε² = 2 for ε=1) |
| RMS of the spatial mean of vorticity | 4.5e-7 |

Consequences:

1. **The flow decorrelates in about five snapshots.** Any statement at lead ≥ 8–10 is a statement about the model chain's *stationary law*, not about forecasting. "Spread/skill ≈ 0.63 at lead 10" therefore means one of two things: the chain's stationary variance is too small, or the members do not diverge from each other fast enough. These need different fixes (Experiment 0 below separates them).
2. **A 6-frame window (5 history + 1 target) spans the whole memory of the process** (autocorrelation 0.07 at lag 5). Matching the stationary law of 6-frame windows is therefore nearly a complete long-horizon criterion (Idea 3).
3. **The enstrophy budget closes to 3.4 % on the subsampled data.** Itô's formula for the forced vorticity equation gives d E[Z]/dτ = −2ν E[P] − 2α E[Z] + 2ε² (four modes, eight unit Wiener coefficients; Z = ½⟨ω²⟩, P = ½⟨|∇ω|²⟩). This corroborates the manuscript's ν, α, ε (the notes list them as unverified) and yields a truth-free scalar test valid at any lead, including beyond lead 95. The 3.4 % gap is presumably the stride-2 subsampling (aliased, truncated palinstrophy); that attribution is unverified.

Also: `config/models/ito_map.yaml` defaults `gamma_multiplier` to 1.0 and the NS config does not override it, so a = 1 appears to be the value in use (verify). Then the explicit martingale has per-pixel variance a²/3 = 0.33, against σ_f² ≈ 0.03 (σ_f ≈ 0.17 inferred from the notes' CRPS ≈ 0.305 physical; rough). The network must cancel roughly 90 % or more of the white noise it is handed (Idea 4).

### Experiment 0 (diagnostic, run before anything else)

Uses existing checkpoints and the 10-lead / 4,096-lead outputs. Per lead ℓ:

- **Amplitude:** pooled per-member variance (or mean member enstrophy) divided by the truth's climatological value.
- **Divergence:** mean member–member pattern correlation versus mean member–truth pattern correlation. For a calibrated ensemble these are equal at every lead (truth is exchangeable with a member).
- **Common-noise split:** because the map is a deterministic function of (h, W), rerun step ℓ for all members with one shared path W (variance left = propagated from history) and for one member's history with independent paths (variance = injected). Compare the split between the 1-step map and the 8-step map on shared path objects.
- **Stationary checks:** budget residual 2ν⟨P⟩+2α⟨Z⟩−2, shell spectrum, spatial mean (random-walk drift of the mean under the identity skip is possible; if present, project it out — the data mean is zero to 1e-7).

Outcome → which idea matters: amplitude deficit → Ideas 3 and 1; member–member correlation above member–truth → Idea 2; flat high-k tail or noise leak → Ideas 4 and 1.

---

## Ranked ideas

| Rank | Idea | Cost to falsify | Novelty | Addresses |
|---|---|---|---|---|
| 1 | Interpolant refresh closure | Hours, no training | Moderate (new combination and guarantees) | Own errors in history, base pass-through, spectral growth, part of spread deficit |
| 2 | Chapman–Kolmogorov closure with ensemble teachers | ~1–2 days training | Moderate–high | Underdispersion growing with lead |
| 3 | Stationary window matching on persistent chains | Diagnostic in hours; training ~1 day | Modest | Everything beyond lead ~8 |
| 4 | Exact linear path response | Diagnostic in hours; retraining | Moderate; mechanism unverified | One-step noise leak, spectral error at source |

---

## Idea 1 — Interpolant refresh: a law-preserving corrector, used identically at training and inference

**Pitch.** The point-mass-base interpolant has an explicit forward kernel Q_s (given c and y) and the two-time map gives the SDE transition P_{s,1} in one call. Their composition R_s = Q_s P_{s,1} leaves the conditional forecast law invariant, forgets errors below an explicit noise floor, and costs 1 NFE. Apply it to every forecast before it enters the history, and build training histories by applying the *same* kernel to truth frames. The train/test history gap then has an explicit relative-entropy bound.

**Math.**

- Refresh kernel: x_s = α_s c + β_s y + γ_s √s z, z ~ N(0,I); y′ = Φ^θ_{s,1}(x_s, h; W′) with a fresh path on [s,1].
- *Invariance (exact drift).* Let π = p(·|h). The interpolant marginal is ρ_s = πQ_s, and the SDE has Law(X_s) = ρ_s and Law(X_1) = π. Hence πR_s = ρ_s P_{s,1} = π. No reversibility is needed.
- *Forgetting (data processing).* For inputs y and y+e: KL(R_s(y,·) ‖ R_s(y+e,·)) ≤ β_s²‖e‖² / (2γ_s² s) = ‖e‖² s³ / (2a²(1−s)²). The noise floor is a(1−s)/s^{3/2} per degree of freedom. The bound is extensive (it scales with field dimension), so it is a per-pixel rate, not a small total-variation distance.
- *Gaussian mode surrogate [checked].* For a mode with conditional law N(m, σ_f²), R_s acts as y′ − m = κ_s (y − m) + fresh noise, with variance exactly preserved. A variance deficit shrinks by κ_s² per application, and a perturbation is transmitted with factor κ_s:

| s | noise floor (a=1) | κ_s at σ_f=0.05 | σ_f=0.1 | σ_f=0.3 |
|---|---|---|---|---|
| 0.8 | 0.28 | 0.16 | 0.32 | 0.72 |
| 0.9 | 0.117 | 0.38 | 0.64 | 0.93 |
| 0.95 | 0.054 | 0.68 | 0.88 | 0.98 |

  So R_s is a mode-selective shrinkage: directions where the conditional law is narrow (high wavenumbers, where model artefacts live) are pulled to the conditional mean; directions carrying real uncertainty pass through. At s = 0.9 and σ_f ≈ 0.1–0.2 one refresh removes roughly 30–60 % of a variance deficit.

**Use A, inference.** y_{n+1} = R_s(Φ_{0,1}(c,h;W)); 2 NFE per physical step. This is a different 2-NFE sampler from the same-path composition 0→½→1, which the notes found no better than 1 step.

**Use B, training.** Replace the last r history frames by ŷ_k = sg[R_s^θ(y_k | ĥ_{k−1})], sequentially, with the true y_{n+1} as target and the usual diagonal + PSD losses (base ĉ = ŷ_n). The loss stays simulation-free in generative time; the extra cost is r no-grad forwards. With Use A on at inference, a history frame is R_s(raw model sample) at test time and R_s(truth sample) at training time; they differ only through the raw artefact e, bounded by the KL above. The dial s interpolates continuously between teacher forcing (s→1) and free running (s→0).

**Mechanism.** Without it, the identity skip makes off-manifold error neutrally stable (e_{n+1} ≈ e_n + ε_{n+1}, a random walk, matching spectral error that grows with lead). With it, error is scrubbed at the source each step and the kernel is trained on histories carrying exactly the scrubber's residual artefacts.

**Closest prior work and what differs.**
- Resampling Forcing (arXiv:2512.15702, video): noise a truth frame to t_s, let the online model finish denoising, detach, keep clean targets. This is Use B. It does not resample at inference, has no invariance statement and no bound; its alignment with truth is justified empirically.
- Multistep consistency sampling and Restart sampling (Xu et al. 2023): renoise–denoise inside one sample, unconditional in the autoregressive sense.
- PDE-Refiner (Lippe et al. 2023): refines a deterministic prediction through denoising steps; not a law-invariant kernel for a stochastic forecast.
- Thermalizer (Pedersen et al. 2025): a separate diffusion model pulls states toward the *unconditional* invariant measure. R_s preserves the *conditional* law p(y|h) and uses the forecaster itself.
- Stable Video Infinity (arXiv:2510.09212): banks model errors and injects them into clean inputs.
- New here: the conditional invariance for point-base interpolants, the matched use on both sides with a bound, the mode-shrinkage reading, and 1-NFE cost because Φ_{s,1}(I_s) is exactly what the off-diagonal loss already trains (off-diagonal samples start at x = I_s, with probability mass on t = 1).

**Risks.** The invariant law of R_s^θ is the model's, not the truth's. Φ_{s,1} near the diagonal sits in the trained support gap (gaps ≥ 0.01, fine for s ≤ 0.97). Use B biases the kernel toward over-dispersion of order (1−κ_s²) times the history sensitivity; s too small breaks target validity. Invariance and the κ table are exact only for exact drift / Gaussian modes.

**Minimal falsifying experiment (no training).** On validation trajectories, roll out 10 and ~500 leads with the 1-step map: (i) plain, (ii) 2-step same-path, (iii) 1-step + refresh at s ∈ {0.8, 0.9, 0.95}. Report spectral discrepancy versus lead, pooled enstrophy, budget residual, spread–skill, and lead-1 CRPS (should not degrade). Kill Use A if (iii) does not beat (ii) at equal NFE for any s. Then test Use B by fine-tuning from the checkpoint.

**Implementation sketch.** In `research/ito_maps/experiments/`: `refresh(model, y, history, s)` = `x, _ = model.training_pair(history[..., -1], y, s, randn)`; `path = model.make_brownian_path(x)`; `model.map(x, s, 1.0, history, brownian_path=path)`. Insert after each step in a copy of the `long_horizon_ns.py` loop. For Use B, a `MapTrainer` subclass under `research/` that overrides `_compute_loss` to rewrite `batch["field_history"]` and `batch["base"]`, plus a dataset wrapper returning windows of length p+1+r.

---

## Idea 2 — Chapman–Kolmogorov closure with ensemble teachers

**Pitch.** Train the same network, with a lead embedding, as a *direct* lead-ℓ interpolant (c = y_n → y_{n+ℓ}; simulation-free, no compounding). Then make the autoregressive composition of the one-step map agree in law with the direct kernel at the same history, and with climatology once ℓ exceeds the mixing time. A 1-NFE map makes the unroll affordable; the direct kernel supplies a whole target ensemble per history instead of one truth sample.

**Math.**
- A_ℓ^θ(h) = law of y_{n+ℓ} under ℓ compositions of K_θ; D_ℓ^θ(h) = direct lead-ℓ kernel, trained with the usual diagonal + PSD losses on pairs (h_n, y_{n+ℓ}).
- By definition D_ℓ targets p(y_{n+ℓ} | h_n) *exactly*, whether or not five frames are a sufficient state. A_ℓ is the order-p surrogate. Chapman–Kolmogorov, A_ℓ = D_ℓ, holds for the truth only under sufficiency; minimising the defect selects the best order-p surrogate for multi-lead laws.
- Loss, with fair energy score ES and energy distance ED over M members:
  L_CK = Σ_ℓ w_ℓ [ ES(A_ℓ^θ(h_n), y_{n+ℓ}) + λ ED(A_ℓ^θ(h_n), sg D_ℓ(h_n)) ] + w_∞ ED(A_L^θ(h), μ̂), L ≥ ℓ_mix.
- Here ℓ_mix ≈ 8 (autocorrelation 0.017 at lag 10), so ℓ ∈ {2,3,4,6} for the direct teacher and pooled snapshots from *other* trajectories for μ̂ (independent trajectories give exact samples of the infinite-lead law).
- Gradients through the last j steps only (truncated), or the full unroll for ℓ ≤ 4.

**Mechanism.** The direct kernel is as calibrated at lead ℓ as the one-step kernel is at lead 1, because nothing is composed. The defect term transfers that calibration to the composition, on exactly the histories where the simulation-free loss is silent (model-generated ones). The teacher ensemble lowers gradient variance relative to a single truth with only 160 training trajectories.

**Closest prior work and what differs.**
- Swift (arXiv:2509.25631), FGN, AIFS-CRPS: unrolled CRPS against truth. The ES term here is that and is **not novel**; it is the baseline every idea must beat.
- Self Forcing (arXiv:2506.08009): few-step unroll with a video-level distribution-matching loss.
- Neural Stochastic Flows (arXiv:2510.25769): KL regulariser toward the Chapman–Kolmogorov property for normalising-flow SDE kernels. KC-Maps (arXiv:2611.12880): Kolmogorov–Chapman objective with known drift, molecular dynamics.
- Continuous Ensemble Forecasting (arXiv:2410.05431): lead-conditioned diffusion, no tie to the iterated kernel. Long-Range Distillation (arXiv:2512.22814): autoregressive teacher → long-step student, the opposite direction. Stormer: deterministic multi-interval averaging.
- New here: direct → autoregressive distillation in law at the same history, for a stochastic field emulator, with the ergodic limit as the ℓ = ∞ member of the same family.

**Risks.** Multi-lead training may cost one-step accuracy (fallback: separate teacher network). D_ℓ may itself be underdispersed through the same PSD compression. The CK defect is nonzero for the truth if history is insufficient, so λ trades one-step against multi-lead fidelity. Distribution distances with M ≈ 8 members in 16,384 dimensions are weak; add spectral/patch feature maps.

**Minimal falsifying experiment.** Stage A (one training run): lead-conditioned model, ℓ ∈ {1,2,3,4,6}. On validation, compare D_ℓ with A_ℓ at equal lead: spread–skill, rank histograms, CRPS, spectra. If D_ℓ is not better calibrated than A_ℓ at ℓ = 4–6, the teacher is useless: drop it and keep ES only. Stage B: fine-tune with L_CK versus ES-only at equal compute.

**Implementation sketch.** Lead embedding through the existing `pars_cond` path of the U-Net; a research dataset wrapper that samples ℓ and returns (history, y_{n+ℓ}, ℓ); a fine-tuning script that unrolls with `model.map` (not `sample`, which is `no_grad`), fresh `make_brownian_path` per step, fair energy score and energy distance in `research/`. All under `research/ito_maps/`, nothing in `src/`.

---

## Idea 3 — Stationary window matching on persistent chains, with exact budget metrics

**Pitch.** Beyond lead ~8 only the chain's stationary law matters, and beyond lead 95 nothing else can be judged. Keep a buffer of free-running model histories, advance it one 1-NFE step per iteration, and match the law of (p+1)-frame windows to data windows. Judge with the exact enstrophy budget.

**Math.**
- *Completeness.* For an order-p kernel K with a unique stationary law, let ν_K be the stationary law of (h, y) windows. If ν_K = ν_data, then the invariant law on histories agrees and K(h,·) = p_data(·|h) for almost every h, so the whole stationary process agrees. Marginal invariant-measure matching does not have this property; window matching does. Here a 6-frame window also spans the physical memory.
- *Loss.* ED_φ(windows from buffer ∪ new step, data windows) over feature maps φ: log shell spectra per frame; spectra of increments y_{k+1} − y_k and of lag-2 differences (dynamics); random-convolution patch features; scalars Z, P. Gradient through the newest step only; one-step losses stay on to pin the kernel.
- *Exact tests.* Stationary: 2ν⟨P⟩ + 2α⟨Z⟩ = 2ε² (1.93 on data). Conditional (drop the viscous term): E[Z_{n+1} | h_n] ≤ e^{−2αΔ} Z_n + (ε²/α)(1 − e^{−2αΔ}), i.e. Z contracts toward at most 10 with factor 0.905 per step.

**Mechanism.** It trains where the model actually lives at long lead, which no paired loss can reach, on the statistics by which long runs are evaluated.

**Closest prior work and what differs.** DySLIM (arXiv:2402.04467) and Jiang & Lu 2023 regularise the invariant *marginal* of mostly deterministic emulators. Thermalizer corrects at inference. Forced-dissipative stochastic interpolants (arXiv:2611.09317) obtain a Foster–Lyapunov condition by architecture; Mücke & Sanderse (arXiv:2504.05852) constrain ensemble energy in the interpolant. Different here: window law (complete for order-p stochastic kernels), persistent chains made affordable by 1 NFE, and the forcing-specific budget identity as metric. Novelty is modest; relevance is high because of the short memory.

**Risks.** Goodhart on the chosen features; weak two-sample power; a drifting buffer needs reseeding from data with some probability; conflict with the one-step loss.

**Minimal falsifying experiment.** From the existing 4,096-lead runs compute pooled Z, spectrum, lag-1 autocorrelation (truth 0.62), and the budget residual. If all are within the trajectory-bootstrap band of the data, the idea is unnecessary. Otherwise fine-tune with a small weight and check that lead-1 CRPS does not move.

**Implementation sketch.** Buffer tensor [N,1,128,128,5] on GPU; per iteration one `model.map` step with gradient, features with `torch.fft`, energy distance against a random batch of training windows. Metrics added to a copy of `report_long_horizon.py`.

---

## Idea 4 — Exact linear path response (hard-wire the martingale cancellation)

**Pitch.** For a point-mass base with γ_1 = 0, the average drift contains a known linear functional of the path that almost entirely cancels the explicit martingale. Put it in the parameterisation so the network learns only the small physical response.

**Math.**
- Point-target limit (y fixed): with D = X − α c − β y, d(D/γ) = dW, so X_t = m_t + (γ_t/γ_s)(x − m_s) + γ_t (W_t − W_s). Therefore
  (t−s) G_{s,t} = [deterministic] − A_{s,t}, A_{s,t} = ∫_s^t (γ_r − γ_t) dW_r = a ∫_s^t (W_r − W_s) dr.
- In general, the first-order Itô–Taylor path term is ∫_s^t ∇b_r · (M_r − M_s) dr; in the narrow-target regime ∇b_r ≈ (γ̇_r/γ_r) I, which reproduces −A.
- Gaussian target N(m, σ_f²) [checked]: X_1 − m = ∫_0^1 w(r) dW_r with ∫w² = σ_f². The exact map removes all of M_1 (variance a²/3 = 0.33) and leaves variance σ_f² (0.01–0.03).
- Parameterisation: Φ_{s,t}(x) = x + (t−s) Ĝ_θ + γ_t (W_t − W_s). At t = 1 no white noise is added; all spread comes through the features. The diagonal is unchanged (A/(t−s) = O(√(t−s))). The PSD identity carries over with A_{s,t} = A_{s,u} + A_{u,t} + (γ_u − γ_t)(W_u − W_s).
- K local increments of M represent A exactly on intervals ending at 1 and leave 1 % (K=4) to 11–22 % (K=1) of Var(A) unexplained elsewhere [checked]. So for the one-step map the gain is exact cancellation instead of cancellation learned through a U-Net in bf16, not extra information.

**Mechanism.** A relative cancellation error of 1 % leaves white noise of std ≈ 0.006 per pixel, a flat spectrum that can exceed the true high-wavenumber enstrophy and then passes through the identity skip at every step. The regression target is also dominated by the cancellation term, which may starve the small stochastic response of gradient (a candidate cause of underdispersion).

**Closest prior work and what differs.** Pan et al. (arXiv:2606.11156) keep the martingale explicit, feed Karhunen–Loève coefficients, and report shortcut learning when M_t − M_s is given as an input feature. Improved Strong Stochastic Flow Maps (arXiv:2610.29605) use a Legendre basis carrying "higher-order local information"; its first coefficient is essentially ∫(W_r − W_s) dr, but as a learned input. I found no hard-wired analytic response; I did not read those papers' full texts, so this is unconfirmed. The response is specific to forecasting interpolants with a point base.

**Risks.** The mechanism is unverified: the flow map must cancel its Gaussian base in the same way and was the better one-step model. With no explicit noise at t = 1, a network that ignores the features collapses the ensemble (the shortcut Pan et al. warn about becomes fatal rather than hidden). Needs a joint (M, W) sampler.

**Minimal falsifying experiment (no training).** For the existing 1-step map, regress per pixel the residual against the 8-step map on a shared path (or against the ensemble mean) on the four feature channels: the leak coefficient and the leaked variance fraction. Check for a flat high-k tail in one-step outputs that is absent at 8 steps. If leaked variance is below a few percent of the residual, drop the idea.

**Implementation sketch.** Research subclass of `FollmerItoMapModel` overriding `noise_increment` to return γ_t ΔW. Extend the path cache to the bivariate Gaussian Markov pair (W_t, M_t) with 2×2 bridge conditioning (Cov(ΔW, ΔM) = a[(1−s)² − (1−t)²]/2, Var ΔW = t−s, Var ΔM = V(s,t)); about 40 lines beside `WeightedBrownianPath`, kept under `research/`.

---

## Discarded

| Idea | Reason |
|---|---|
| Plain unrolled CRPS / energy-score fine-tuning | Correct and likely effective but not novel (Swift, FGN, AIFS-CRPS, Self Forcing). Kept as the baseline inside Idea 2. |
| Physical-time semigroup: one map Φ_{σ,τ} on an extended clock τ = n + t | Sound (X is continuous across physical steps, γ is 1-periodic, PSD holds across integers with state (x,h)), but it distils the model's own rollout: faster, not more faithful. Data at long leads enters only through extra scoring. Needs ℓK path channels. Useful later as an accelerator only. |
| Pathwise shadowing of a finer-budget reference across physical steps (stochastic pushforward on shared paths) | Imposes contraction toward the reference trajectory. With lag-1 autocorrelation 0.62 the chains separate physically within a step or two, so the loss would remove legitimate propagated spread. |
| Error transplant (1-step minus 8-step output on a shared path, added to truth frames) | Captures only budget error, not error against truth; state-dependent mismatch; prior art (Stable Video Infinity). Subsumed by Idea 1. |
| Truth as regression target for model-generated histories (classical pushforward) | Wrong for a stochastic model: the generated history is a different realisation, so the optimum blurs. |
| Brownian paths correlated across physical steps | Changes the law of y_{n+1} given h_n (the previous path is partly identifiable from the history); buys spread by breaking one-step calibration. |
| Noise temperature (scale W at inference) | Not a valid change of law for a nonlinear map; inflates white noise, not physical spread. |
| Foster–Lyapunov enstrophy penalty | Pre-empted (arXiv:2611.09317; Mücke & Sanderse), and blow-up is not the observed failure. The inequality is kept as a metric in Idea 3. |
| Noised-history conditioning / diffusion forcing | Known (ACDM, GenCast-style); destroys small-scale information; not map-specific. Idea 1 gets masking without conditioning on noisy frames. |
| Jacobian matching to the multi-step teacher | The teacher's sensitivity to off-manifold histories is equally unconstrained by data; expensive. |
| Exact innovation inference (the path that reproduces the observed y) | Needs the KL-optimal Föllmer variant, where the conditioned process is a reference bridge; the repository's γ-diffusion variant has no such property. |
| Noise optimisation to make a rollout track truth | The white path has enough freedom to hit any target; ill-posed without a path prior. |
| Zero-mean projection of the output | Not an idea, just do it if Experiment 0 shows drift of the spatial mean. |

---

## Suggested order

1. Experiment 0 and the two no-training falsifiers (Idea 1 Use A, Idea 4 leak regression): one day, existing checkpoints.
2. Idea 3 diagnostics on the existing 4,096-lead runs.
3. One lead-conditioned training run (Idea 2 Stage A).
4. Fine-tuning comparison at equal compute: ES-only baseline vs L_CK vs Use B of Idea 1, each with and without refresh at inference.

## What is not verified

- Any effect on the actual models: no checkpoint was run.
- a = 1 and σ_f ≈ 0.17 are inferred from configs and old notes.
- The 3.4 % budget gap is attributed to subsampling without a check on the 256² companion.
- Invariance, κ_s and the linear response are exact only under exact drift / Gaussian or point targets.
- Novelty claims rest on abstracts and search summaries for arXiv:2606.01086, 2610.29605, 2611.09317, 2611.12880 and 2510.25769; full texts were not read. Pan et al. was read to ~70 % of its HTML.

## Sources

- Pan et al., Itô maps for any-step SDEs — https://arxiv.org/abs/2606.11156
- Strong Stochastic Flow Maps — https://arxiv.org/abs/2606.01086
- Improved Training of Strong Stochastic Flow Maps — https://arxiv.org/abs/2610.29605
- Boffi et al., How to build a consistency model (flow maps) — https://arxiv.org/abs/2505.18825
- Swift — https://arxiv.org/abs/2509.25631
- Self Forcing — https://arxiv.org/abs/2506.08009
- Resampling Forcing (self-resampling) — https://arxiv.org/abs/2512.15702
- Stable Video Infinity — https://arxiv.org/abs/2510.09212
- Thermalizer — https://arxiv.org/abs/2503.18731
- PDE-Refiner — https://arxiv.org/abs/2308.05732
- DySLIM — https://arxiv.org/abs/2402.04467
- Jiang & Lu, invariant measures of chaotic attractors — https://arxiv.org/abs/2306.01187
- Neural Stochastic Flows — https://arxiv.org/abs/2510.25769
- Kolmogorov–Chapman Maps — https://arxiv.org/abs/2611.12880
- Continuous Ensemble Weather Forecasting — https://arxiv.org/abs/2410.05431
- Long-Range Distillation — https://arxiv.org/abs/2512.22814
- Mücke & Sanderse, energy-consistent stochastic interpolants — https://arxiv.org/abs/2504.05852
- Stable probabilistic emulation of partially observed dynamical systems — https://arxiv.org/abs/2611.09317
- Benchmarking autoregressive conditional diffusion models (ACDM) — https://arxiv.org/abs/2309.01745
