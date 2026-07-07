# Report & Plan: Fine-tuning a deterministic time-stepper into an Itô map

**PR ordering:** this is **PR 3**, implemented after the Itô maps plan (`docs/plans/ito-maps-implementation-plan.md`, **PR 1**) and the deterministic-models plan (`docs/plans/deterministic_models_and_trainer_refactor.md`, **PR 2**). It depends on both: PR 1's `models/ito_maps/` module + `ItoMapTrainer`, and PR 2's `DeterministicModel`.

**Goal:** given a deterministic next-step model `x_{n+1} = F_θ(x_n, history, params)` trained with plain MSE (PR 2), post-train / fine-tune it into an Itô map `X̂_{s,t}(x, W)` (PR 1) — an any-step stochastic flow map that produces sharp, diverse samples of `x_{n+1}` in one (or few) network evaluations.

---

## 1. Summary of the recommendation

The proposal rests on one identity: **an MSE-trained deterministic model is the conditional mean, and the conditional mean exactly pins down the Brownian-averaged, full-span drift of the Itô map.** Concretely, `F(x_n) ≈ E[X₁ | x_n, h]`, and the true Itô map satisfies `E_W[G_{0,1}(x_n, W)] = F(x_n) − x_n`. Fine-tuning therefore never has to relearn the mean dynamics — only the *fluctuations around a known mean*. Everything below is a way of baking this identity into the initialization and the parametrization so it is preserved rather than merely approximated.

Three methods, in recommended priority:

1. **Method 1 (primary): mean-anchored residual Itô map.** Freeze `F`. Define the residual `R = X₁ − F(x_n)`, normalize it by a calibrated scale `ρ`, and train a *new, typically smaller* Itô map on the residual process only (base = point mass at 0, Föllmer-style — or Gaussian base for the flow-matching variant). The composite `x̂_{n+1} = F(x_n) + ρ·ItôMap(residual)` is itself, exactly, a full-state Itô map under a closed-form change of variables (derived in §5). This is CorrDiff's regression + residual-diffusion decomposition lifted to any-step stochastic flow maps; the sea-ice results of Finn et al. suggest residual generative models match from-scratch ones at a fraction of the difficulty. Modular, stable (no catastrophic forgetting — `F` is frozen), cheap, and PR 1's `ItoMapTrainer` applies verbatim.

2. **Method 2 (unified variant): weight-surgery warm start.** Initialize the Itô map network *from* `F`'s weights (copy backbone; zero-init the new `(s,t)` time embedding and Brownian-feature input channels; arrange the output head so that `X̂_{0,1}(x_n) = F(x_n)` **exactly at initialization** — §6 gives the identities for both values of PR 2's `residual` flag). Then fine-tune with PR 1's from-scratch objective under staged unfreezing. One unified network, full capacity, but risks degrading the mean dynamics; mitigated by the staged schedule and the (optional) weight-EMA already in `BaseTrainer`.

3. **Method 3 (optional accelerator for either): analytic Gaussian-shell teacher.** Under the approximation `p(X₁|x_n) ≈ N(F(x_n), ρ²I)`, the interpolant's velocity, score, and hence the diagonal drift target are **fully closed-form functions of `F`** (§7). Use this analytic teacher for the first few epochs of fine-tuning — regression onto a deterministic function is fast and low-variance — then switch to the exact single-sample data targets, which removes the Gaussian bias. Plugs into PR 1's teacher mode.

**Novelty statement:** the components have precedent — regression-mean + residual diffusion ([CorrDiff, arXiv:2309.15214](https://arxiv.org/abs/2309.15214); [Finn et al. 2024, JAMES](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004395)), deterministic warm starts for conditional diffusion ([arXiv:2507.09212](https://arxiv.org/abs/2507.09212)), teacher-guided Itô maps ([arXiv:2606.11156](https://arxiv.org/abs/2606.11156)) — but deterministic-to-*any-step-stochastic-flow-map* fine-tuning is new, and the mean-anchor identity (§4) plus the exact residual change of variables (§5) are what make it principled rather than heuristic.

---

## 2. Related work

- **[CorrDiff — Residual Corrective Diffusion Modeling (arXiv:2309.15214)](https://arxiv.org/abs/2309.15214)**: UNet regression predicts the conditional mean; a diffusion model learns the residual. Signal decomposition inspired by Reynolds decomposition. Recovers correct spectra/distributions where regression alone blurs. Multi-step diffusion sampling; no flow-map/one-step structure, no two-time consistency.
- **[Finn et al. 2024 — Generative diffusion for sea-ice surrogates (JAMES)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004395)**: residual diffusion on top of a deterministic surrogate performs on par with diffusion trained from scratch, for all variables and lead times — direct evidence that the residual task is not harder than the full task, while being much cheaper to fine-tune.
- **[Warm Starts Accelerate Conditional Diffusion (arXiv:2507.09212)](https://arxiv.org/abs/2507.09212)**: a deterministic network predicts an informed prior `N(μ̂, diag(σ̂²))`; the generative process only traverses the remaining distance, cutting NFEs to ~4–6. Same insight — the deterministic prediction removes most of the transport — but applied to the *prior*, not to the map or its training targets.
- **[Itô Maps for Any-Step SDEs (arXiv:2606.11156)](https://arxiv.org/abs/2606.11156)** (PR 1): provides the map parametrization, the diagonal + LSD/PSD losses, and a teacher-guided mode for *velocity-field* teachers. This plan adds a new teacher species (a deterministic mean model) that the paper does not cover.
- One-step distillation with student-from-teacher weight initialization (e.g. [SlimFlow, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10822.pdf)) — precedent for the weight-surgery pattern of Method 2, already mirrored by PR 1's `from_flow_matching` constructors.

---

## 3. Setup and notation

Matches PR 1/PR 2 and the repo:

- State `x_n ∈ ℝ^{C×H×W}`; conditioning `h` = `field_history` (whose last slice **is** `x_n`), `field_cond`, `pars_cond`. Target `X₁ ~ p(x_{n+1} | x_n, h)`.
- **Deterministic model (PR 2):** `F(x_n, h) := DeterministicModel._step(x_n, h, ...)`. MSE-trained ⇒ `F ≈ μ(x_n) := E[X₁ | x_n, h]` (the MSE minimizer is the conditional expectation).
- **Interpolant (Föllmer, point-mass base):** `I_t = α(t)x_n + β(t)X₁ + γ(t)z`, `z ~ N(0,I)`, with `α(0)=β(1)=1`, `α(1)=β(0)=γ(0)=γ(1)=0` (repo: `LinearStochasticInterpolation` etc.).
- **Itô map (PR 1):** `X̂_{s,t}(x, W) = x + (t−s)Ĝ_{s,t}(x, Ψ(W)) + (M_t − M_s)`, with `M_t = ∫₀ᵗ σ_u dW_u`, Brownian features `Ψ(W)` (KL or dyadic), and diagonal drift target `G_{t,t}(x) = b_t(x) + (σ_t²/2)s_t(x)` (velocity + score correction), trained with `ℒ_SI + λℒ_LSD`.

## 4. The anchor identity

Let `Φ_{s,t}` be the *true* Itô map of the interpolant SDE. With point-mass base `X₀ = x_n`, the endpoint of the SDE has law `p(x_{n+1} | x_n, h)`, so averaging the map over Brownian realizations:

```
E_W[Φ_{0,1}(x_n, W)] = E[X₁ | x_n, h] = μ(x_n)
⟹  E_W[G_{0,1}(x_n, W)] = μ(x_n) − x_n ≈ F(x_n) − x_n .
```

The martingale term `M₁ − M₀` averages to zero, so the identity constrains only the drift `G`. Interpretation: **the deterministic model is the zeroth Brownian moment of the Itô map at full span.** Fine-tuning must supply (i) the dependence on the Brownian features (the fluctuations), (ii) the interior-time behavior `s, t ∈ (0,1)` (where the score term matters), while the `(0,1)`-mean is already solved. Each method below hard-codes this identity in a different place: Method 1 in the *parametrization*, Method 2 in the *initialization*, Method 3 in the *training targets*.

## 5. Method 1 (primary): mean-anchored residual Itô map

### 5.1 Residual process

Define the residual and its normalization:

```
R := X₁ − F(x_n, h),        R̃ := R / ρ,
```

with `ρ` a per-channel scale estimated once, offline, from the frozen `F` on training/validation data (§5.4). If `F` is unbiased, `E[R̃ | x_n] ≈ 0` and `R̃` is O(1) — a well-conditioned generative target. (If `F` is biased, `R̃` has non-zero conditional mean; the residual Itô map learns it — no assumption of zero mean is needed anywhere below.)

Build the interpolant **on the normalized residual with base ≡ 0** (Föllmer with a zero point mass — the repo's stochastic interpolant classes accept `base = zeros` unchanged):

```
J_t = β(t) R̃ + γ(t) z .
```

This is *identically* (not just in law) the full-state interpolant in moving coordinates: with the deterministic anchor path

```
φ(t) := α(t) x_n + β(t) F(x_n),      one checks      I_t − φ(t) = β(t)(X₁ − F) + γ(t)z = ρ · J_t .
```

### 5.2 The composite model is exactly a full-state Itô map

Let the residual Itô map be `Ĵ_{s,t}(j, W) = j + (t−s)ĝ_{s,t}(j, Ψ(W)) + (M_t − M_s)` with its own drift network `ĝ_φ` (conditioned on `h`, `field_cond`, `pars_cond` exactly as PR 1's model — the residual distribution depends on the state through the conditioning). Since `x_n` is fixed along a step, `X_t = φ(t) + ρJ_t` is an affine time-dependent change of variables, and if `dJ = g dt + σ dW` then `dX = (φ'(t) + ρ g) dt + ρσ dW`. Hence the composite

```
X̂_{s,t}(x, W) = φ(t) + ρ · Ĵ_{s,t}( (x − φ(s))/ρ , W )
```

is a full-state Itô map with drift `Ĝ_{s,t}(x,W) = [φ(t)−φ(s)]/(t−s) + ρ·ĝ_{s,t}((x−φ(s))/ρ, W)` and diffusion `ρσ_t`. The first term — the analytically known mean transport — carries the anchor identity by construction: at `(s,t)=(0,1)`, `φ(1)−φ(0) = F(x_n) − x_n`, so

```
X̂_{0,1}(x_n, W) = F(x_n) + ρ · Ĵ_{0,1}(0, W) :   deterministic prediction + generated residual.
```

### 5.3 Training

- **Freeze `F`.** Train only `ĝ_φ` with PR 1's `ItoMapTrainer`, *unchanged*: the wrapper model's target computation replaces `(base, target)` by `(0, R̃)` (computed with a `no_grad` forward of `F`) and everything else — diagonal target via `forward_diff + (σ²/2)·score`, LSD/PSD, Brownian features, time sampling — applies verbatim to the residual process.
- The residual network can be **smaller** than `F`'s backbone (the residual field is lower-amplitude and higher-frequency; CorrDiff and Finn et al. both use modest correctors). Architecture stays config-driven.
- **Optional joint polish:** after residual training converges, unfreeze `F` at ~10× lower LR for a few epochs with the same loss (targets recomputed against the *current* `F`, so `R` shrinks as `F` improves). Off by default.

### 5.4 Noise-scale calibration (Stage 0, no gradients)

The stochasticity of PDE time-stepping is small and structured; unit-scale noise would swamp the residual. One pass over validation data with the frozen `F`:

```
ρ_c = std over (samples, H, W) of  R_c = (X₁ − F(x_n))_c        per channel c,
```

persisted next to the checkpoint. `R̃ = R/ρ` then justifies unit-scale `γ(t)` and PR 1's default `σ_t = √(2(1−t))` in residual coordinates — no interpolant retuning needed. (Optionally also record the residual mean per channel as a bias diagnostic; do **not** subtract it — the model learns it.)

### 5.5 Flow-matching variant

With base ≡ 0 **and** `σ ≡ γ ≡ 0` the residual map is deterministic and can only output `E[R̃]` — useless. The FM variant instead uses a **Gaussian base for the residual**: `J_t = (1−t)ε + tR̃`, `ε ~ N(0,I)` (repo: `LinearDeterministicInterpolation` with `base = ε`), `σ ≡ 0` in the Itô map ⇒ a deterministic residual *flow map*: `x̂_{n+1} = F(x_n) + ρ·FlowMap_{0,1}(ε)`. This is one-step conditional flow matching on the residual — the same config-level switch (`sigma_schedule: zero`, `gaussian_base: true`) PR 1 already plans; no special-case code.

## 6. Method 2 (unified variant): weight-surgery warm start

Produce a single `ItoMapModel` whose network is initialized from `F`'s network — the same pattern as PR 1's `from_flow_matching` / `from_stochastic_interpolant` constructors, with one new wrinkle: the teacher was never a drift model.

**Initialization identities.** Let `net` be `F`'s network (PR 2 contract: `net(x, cond, h, field_cond, pars_cond)`; `DeterministicModel._step` passes `cond = zeros(B,1)` and returns `x + net(...)` if `residual=True` else `net(...)`).

- Copy all backbone weights. The state-channel input shapes match (`I_s` has the same shape as `x_n`).
- **Zero-init the new pathways:** the extra column(s) of the two-time `(s,t)` embedding (PR 1 extends `cond` to `[B,2]` backward-compatibly) and the Brownian-feature input convolutions. At init the network is therefore *blind* to `(s,t)` and `W` and computes exactly what it computed as `F`.
- **Output head:** define the map drift as
  - `residual=True` teacher: `Ĝ := net(x, ...)` (the net already outputs `μ(x) − x`);
  - `residual=False` teacher: `Ĝ := net(x, ...) − x` (fixed skip in the wrapper).

  Either way, at initialization `X̂_{0,1}(x_n, W) = x_n + Ĝ(x_n) + (M₁ − M₀)`, whose Brownian mean is exactly `F(x_n)` — the anchor identity holds at init, and `X̂_{s,t}(x) ≈ x + (t−s)(μ(x) − x)`, i.e., linear transport toward the deterministic endpoint — a controlled starting point whose remaining error is precisely the score/noise structure to be learned. Note `F` receives `x_n` also via `field_history`'s last slice, so its skill survives even when the state channel holds a noisy interpolant `I_s`.

**Fine-tuning schedule (staged unfreezing).**
1. Epochs `0..k`: train **only** the new parameters (time-embedding rows, Brownian input convs, final output block) with PR 1's from-scratch objective (`ℒ_SI + λℒ_LSD`; optionally Method 3's analytic teacher as the diagonal target). The frozen backbone protects the mean dynamics.
2. Epochs `k..`: unfreeze everything at reduced LR (~0.1×). Enable `BaseTrainer`'s optional weight-EMA (PR 1) — it doubles as the stop-grad target for LSD and as forgetting insurance.

**When to prefer Method 2 over Method 1:** when a single unified network is wanted (deployment simplicity, full-capacity fine-tuning, later use as a teacher itself), and the training budget allows re-touching the whole backbone. Method 1 remains the default recommendation.

## 7. Method 3 (optional accelerator): analytic Gaussian-shell teacher

Approximate `p(X₁ | x_n, h) ≈ N(F(x_n), ρ²I)` (with the calibrated `ρ`; per-channel diagonal is fine). Then `X₁ = F + ρε`, and with `u := x − α(t)x_n − β(t)F(x_n)` the interpolant marginal is Gaussian with `Var(I_t | x_n) = (β²ρ² + γ²)I`, giving **closed forms** for everything the diagonal target needs:

```
score:      s_t(x)     = − u / (β²ρ² + γ²)
E[ε | I_t=x] =  βρ u / (β²ρ² + γ²) ,     E[z | I_t=x] = γ u / (β²ρ² + γ²)
velocity:   b_t(x)     = α'x_n + β'F + (β'βρ² + γ'γ) · u / (β²ρ² + γ²)
teacher:    G^gauss_{t,t}(x) = b_t(x) + (σ_t²/2)·s_t(x)
           = α'x_n + β'F + [β'βρ² + γ'γ − σ_t²/2] · u / (β²ρ² + γ²) .
```

This is a deterministic function of `(x, t, x_n, F)` — a valid drift teacher for PR 1's teacher-guided mode (Algorithm 4), requiring only one frozen forward pass of `F` per batch. Its value: early fine-tuning regresses onto a *noise-free* target (fast, low gradient variance) instead of the single-sample stochastic targets. Its cost: a Gaussian/unimodal bias. Therefore: **warm-up only** — use `G^gauss` as the diagonal/LSD target for the first `teacher_warmup_epochs`, then switch to the exact data targets, which are unbiased and remove the assumption. Applicable to both Method 1 (in residual coordinates, where `F ≡ 0`, `ρ ≡ 1` and the formulas simplify further) and Method 2.

## 8. Risks and open questions

- **Residual multimodality.** Methods 1 and 2 make *no* distributional assumption — the Itô map learns the full conditional law of `R`. Only Method 3 assumes Gaussianity, and only as a warm-up. Validation should include a case with known non-Gaussian step statistics.
- **Mean-model bias.** A biased `F` puts a conditional mean into `R̃`, which the residual map absorbs; the optional joint-polish stage (§5.3) reduces it at the source. Not a correctness issue, only an efficiency one.
- **Autoregressive distribution shift.** `F` and the residual map are trained on data-manifold inputs; rollouts drift off-manifold. Same exposure as every model in the repo; rollout-in-the-loop fine-tuning is explicitly out of scope for PR 3 (future work).
- **Calibration sensitivity.** `ρ` off by an order of magnitude degrades conditioning of the residual task (too small: targets blow up; too large: noise swamps signal). The calibration is cheap; make it automatic (Stage 0 inside the entry point), not a hand-set hyperparameter.
- **LSD through two networks (Method 1).** The `jvp` in LSD only touches `ĝ_φ` — the frozen `F` enters targets under `no_grad` — so the finite-difference fallback story from PR 1 is unchanged.

---

## 9. Implementation plan (PR 3)

Prerequisites: PR 1 and PR 2 merged. All work is additive except three small, flagged extensions to PR 1 files.

### 9.1 New files

- **`src/scisi/models/ito_maps/residual_ito_map.py`** — `ResidualItoMapModel(BaseModel)` (Method 1):
  - holds `mean_model: DeterministicModel` (frozen; `requires_grad_(False)`, excluded from `state_dict` optimizer params but included in checkpoint), an inner `ItoMapModel` on the normalized residual, and the calibration stats `ρ` (registered buffer).
  - `forward(base, target, ...)` computes `R̃ = (target − F(base, h))/ρ` under `no_grad`, then delegates to the inner model with `(base=0, target=R̃)` — so `ItoMapTrainer._compute_loss` works unchanged.
  - `map` / `sample` / `sample_trajectory`: composite formula of §5.2 (`φ(t)` shift + `ρ` rescale), delegating rollout to `BaseModel._sample_trajectory` as everywhere else. `sample` at `(0,1)` returns `F(x_n) + ρĴ_{0,1}(0, W)`.
  - `classmethod from_deterministic(det_model, ito_map_cfg, residual_stats)`.
- **`src/scisi/models/ito_maps/calibration.py`** — `estimate_residual_stats(det_model, dataloader, per_channel=True) -> ResidualStats` (mean + std per channel, one `no_grad` pass); `save/load` next to the checkpoint (`residual_stats.pt`).
- **`src/scisi/models/ito_maps/analytic_teacher.py`** — `GaussianShellTeacher` (Method 3): implements PR 1's teacher-drift interface (`drift(x, t, field_history, field_cond, pars_cond)`) from `(F, ρ, interpolant, σ)` using the §7 formulas. **Requires PR 1's teacher interface to be duck-typed** (any object with a `drift(...)` method), not `isinstance`-gated — flagged as a forward-compatibility note to PR 1.
- **`config/ito_map_from_deterministic_residual_navier_stokes.yaml`** (Method 1), **`config/ito_map_from_deterministic_warmstart_navier_stokes.yaml`** (Method 2), plus an FM variant of the residual config (`sigma_schedule: zero`, Gaussian residual base, §5.5).
- **`tests/test_ito_map_finetuning.py`** (§9.4).

### 9.2 Extensions to PR 1 files (small, additive)

- **`models/ito_maps/ito_map_model.py`**: add `classmethod from_deterministic(det_model, sigma_schedule, ...)` (Method 2 weight surgery; mirrors `from_flow_matching`; branches on PR 2's `residual` flag for the output-head identity of §6).
- **`training/ito_map_trainer.py`**: three config options — `freeze: [module names]`, `unfreeze_at_epoch: int | null`, `teacher_warmup_epochs: int = 0` (switches the diagonal/LSD target from `teacher` to data targets after N epochs). Staged unfreezing needs an epoch-boundary hook: add a no-op `BaseTrainer._on_epoch_start(epoch)` called at the top of the epoch loop (behavior-preserving one-liner in PR 1's `base_trainer.py`; if PR 1 has merged without it, add it here).
- **`bin/main_train.py`**: extend PR 1's `pre_trained_model` block with `type: deterministic` and `init_mode: residual | warm_start`; when `residual`, run Stage-0 calibration automatically before building the model (or load persisted stats).

### 9.3 No changes to PR 2 files

`DeterministicModel` is consumed as-is; the only coupling is its `_step` signature and `residual` flag semantics, both fixed in the PR 2 plan.

### 9.4 Tests

1. **Init identities:** Method 1 — untrained `ResidualItoMapModel.sample` at `(0,1)` with zeroed residual net output equals `F(x_n)` exactly. Method 2 — after `from_deterministic`, `X̂_{0,1}(x_n, W) − (M₁−M₀) == F(x_n)` to float precision, for both `residual=True/False` teachers; perturbing `(s,t)` or Brownian features at init changes nothing (zero-init check).
2. **Change-of-variables:** on a Gaussian toy with analytic residual drift, composite full-state drift equals `φ' + ρ·g((x−φ)/ρ)` (§5.2), and the composite endpoint law matches direct simulation of the full-state SDE.
3. **Gaussian-shell teacher:** §7 formulas vs. Monte-Carlo estimates of `E[ε|I_t]`, `E[z|I_t]`, score, on a 2-D Gaussian toy across a `t`-grid.
4. **Calibration:** `estimate_residual_stats` recovers a known injected `ρ` on synthetic data; persists/loads round-trip.
5. **Trainer options:** `freeze` keeps listed modules' grads `None`; `unfreeze_at_epoch` flips them; `teacher_warmup_epochs` switches targets (assert via loss-value discontinuity on a fixed batch).
6. **Smoke fine-tunes:** 2 epochs, tiny synthetic dataset, all three of {Method 1 stochastic, Method 1 FM-variant, Method 2}; loss decreases; one-step sample + 6-step rollout shapes and finiteness.

### 9.5 Validation experiment (uses existing infra)

On Navier-Stokes: (a) train `DeterministicModel` (PR 2 config); (b) fine-tune via Method 1 and Method 2, each ± the Gaussian-shell warm-up; (c) compare against PR 1's from-scratch Itô map and SI-distilled Itô map using `metrics/` (CRPS, spread-skill, rank histograms, sliced-W2, energy spectra) **and total training cost** (deterministic pre-training + fine-tuning vs. from-scratch). The claim to test: fine-tuning reaches from-scratch quality at a fraction of the generative-training cost, mirroring Finn et al.'s residual-diffusion finding. Also verify the composite spectra are sharp (residual restores the high-frequency power that MSE-trained `F` blurs — the CorrDiff effect).

### 9.6 Implementation order

1. `calibration.py` + tests (no dependencies).
2. `ResidualItoMapModel` + init-identity and change-of-variables tests (Method 1 core).
3. Trainer options (`freeze` / `unfreeze_at_epoch` / `teacher_warmup_epochs`) + `_on_epoch_start` hook.
4. `ItoMapModel.from_deterministic` weight surgery (Method 2) + init-identity tests.
5. `GaussianShellTeacher` + formula tests (Method 3).
6. Configs + `main_train.py` extension + smoke tests.
7. Validation experiment.

---

## 10. Alignment with PR 1 and PR 2

| Touchpoint | Nature |
|---|---|
| `models/ito_maps/ito_map_model.py` (PR 1) | PR 3 adds one classmethod (`from_deterministic`); no existing code paths change. |
| `training/ito_map_trainer.py` (PR 1) | PR 3 adds three config-defaulted options; default behavior identical. |
| `training/base_trainer.py` (PR 1) | PR 3 needs a no-op `_on_epoch_start(epoch)` hook — ideally added in PR 1 (one line), else added here. |
| PR 1 teacher interface | Must be duck-typed on `.drift(...)` so `GaussianShellTeacher` plugs in — noted as a design requirement for PR 1 Phase 3. |
| `bin/main_train.py` | PR 3 extends the `pre_trained_model` block PR 1 introduces (`type`, `init_mode` keys). |
| `deterministic_models/` (PR 2) | Consumed read-only; depends on `_step` signature and `residual` flag semantics as specified in the PR 2 plan. |

Two forward-compatibility notes should be reflected in the earlier plans (PR 1: duck-typed teacher + `_on_epoch_start` hook; PR 2: none) so PR 3 lands without retrofits.
