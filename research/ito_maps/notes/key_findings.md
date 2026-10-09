# Itô maps for stochastic Navier–Stokes forecasting: key findings (consolidated)

Consolidated 2026-10-08 from all 27 former `notes/` files and 6 former `reviews/` files (written 2026-10-07). The originals are archived; each bullet cites its origin as `(source: ../archive/notes/<file>)` or `(source: ../archive/notes/reviews/<file>)`. The `results/` tree referenced below now lives in `../archive/results/`.

**Scope:** The project now covers only the stochastic Navier–Stokes (NS) case. Video forecasting (KTH/CLEVRER) is dropped. Every number from the initial research stage, including the `results/` tree, is **superseded**. Section 5 keeps those numbers only as qualitative, initial-stage lessons; none of them should go in the paper.

## TL;DR

- **Object learned:** a conditional Itô map `Φ(s,t,x;h,W) = x + (t−s)G + (M_t−M_s)` for the repository's *forecasting* SDE. The SDE uses γ(t)=a(1−t) and starts from the latest observed field. This is **not** the KL-optimal Föllmer process, despite the class name `FollmerStochasticInterpolant`. The diagonal target is `α'c + β'Y + γ'√t z`: never differentiate √t.
- **Brownian driver:** the driver is exact. `M` is a Brownian motion on the variance clock `q(t)=a²(t−t²+t³/3)`, sampled through lazy Gaussian bridges. The approximation is elsewhere: the network sees only K local increments, so the nonlinear drift's dependence on the path is compressed. Finite-feature PSD therefore has **no identifiability or exact-semigroup guarantee**, even with ideal optimisation.
- **Training:** training has two stages. Stage 1 is teacher-anchored PSD (simulation-free). Stage 2 is same-path teacher-rollout refinement (simulation-based). The rollout teacher uses **Euler drift plus exact integrated noise**, which is not the baseline's Euler–Maruyama sampler. Any comparison must disclose this. An exact-diffusion solver control is still missing.
- **NS data provenance is incomplete.** Snapshot spacing *was* recovered: 0.4999 stored-time units, with the 128² data being a stride-2 subsample of a 256² companion. Viscosity, drag, forcing amplitude, generator revision and teacher pretraining records are **unknown**. Only 95 truth frames follow a 5-frame history.
- **Initial-stage lessons (superseded):**
  - The deterministic flow map (Gaussian base, 1 step) was the strongest fast baseline.
  - Itô maps had close proper scores but were consistently **underdispersed**.
  - Teacher-rollout refinement helped a lot over PSD alone.
  - K16 did not beat K4, even though a diagnostic showed K16 carries much more path information.
  - More map steps did not improve quality monotonically.
- **Metrics:** close CRPS can hide clear calibration loss, because spread errors move CRPS only to second order. Empirical 5–95% coverage from E members cannot reach 90%, even for a calibrated ensemble. Use RMS spread, extreme-rank fraction and rank histograms, and use paired trajectory bootstraps for differences.
- **Pitfalls in current code:**
  - Masks are rejected in `MapTrainer`.
  - Trajectory `pars_cond` is broken for general `[B,D]` parameters.
  - Maps cannot change their diffusion after training.
  - A common seed does not give a common Brownian path across budgets.
  - Inherited `Trainer`, `EarlyStopping` and AMP have known bugs.

---

## 1. Methodology and mathematical decisions

### Interpolant and SDE
- Two clocks:
  - Physical lead Δ is separate from generative time t∈[0,1].
  - The history h is fixed during all map calls of one forecast.
  - After each physical forecast, shift the history and draw a fresh, independent driving path.
  (source: ../archive/notes/methodology.md)
- Interpolant: `I_t = α c + β Y + γ B_t`, with α=1−t, β=t or t² (the NS config uses quadratic), γ=a(1−t), and `B_t=√t z`. The SDE is `dX = b dt + γ dW`, with `X_0 = c` (the latest observed field). (source: ../archive/notes/methodology.md, ../archive/notes/navier_stokes_provenance.md)
- **Diagonal target:** `R_t = α'c + β'Y + γ'√t z`, which equals `−c+2tY−a√t z` for quadratic β.
  - Do **not** differentiate √t. That gives the probability-flow velocity, not the SDE drift.
  - Do not copy the Itô-map paper's `X1−2X0` target; it assumes a Gaussian source and a different diffusion.
  - `sigma` in `AffineGaussianPathMixin` (γ√t) is not the sampler diffusion (γ).
  (source: ../archive/notes/methodology.md, ../archive/notes/flow_maps_and_code_design.md, ../archive/notes/reviews/adversarial_method_math_2026-10-07.md)
- Paired interpolant samples `(I_s, I_t)` are not SDE sample paths. Regressing on them directly does not train the Markov Itô map. (source: ../archive/notes/methodology.md)

### FSI-γ versus KL-optimal Föllmer
- The repository class `FollmerStochasticInterpolant` uses diffusion γ(t). The source paper's KL-optimal Föllmer process uses a *different* `g_F` together with a score-corrected drift `b^g = b + ½(g²−γ²)·score`. Its schedules are:
  - β=t: `g_F² = a²(1−t²)`.
  - β=t²: `g_F² = a²(1−t)(3−t)`, so `g_F(0) = √3·a ≠ γ(0)`.
  (source: ../archive/notes/methodology.md)
- A learned map is tied to its diffusion schedule. Swapping the diffusion after training is mathematically wrong, because the whole average-drift/path relation changes. A strict-Föllmer map would need a `b^g_F` diagonal, an `M = ∫g_F dW` driver, and careful endpoint-score handling. Treat it as a separate ablation only. (source: ../archive/notes/methodology.md, ../archive/notes/methodology_implementation_audit.md, ../archive/notes/reviews/adversarial_method_math_2026-10-07.md)

### Brownian driver (variance clock and bridges)
- For additive deterministic γ: `M_t = ∫γ dW` is Brownian motion on the clock `q(t) = a²(t−t²+t³/3)`, with `V(s,t) = a²[(1−s)³−(1−t)³]/3`.
  - Evaluate `V` as a factored difference of cubes; this avoids cancellation near t=1.
  - Endpoint variance is `M_1 ~ N(0, a²/3)`.
  (source: ../archive/notes/methodology.md)
- Implemented driver (`WeightedBrownianPath`) uses lazy bridges. For cached `l<t<r`:
  - Mean: `M_l + V(l,t)/V(l,r)·(M_r−M_l)`.
  - Variance: `V(l,t)V(t,r)/V(l,r)`.
  - This gives the exact joint law at any finite set of query times. It supersedes the earlier fixed-grid, Haar, cosine and KL-feature recommendations, which remain research alternatives.
  (source: ../archive/notes/methodology.md, ../archive/notes/code_equation_audit.md)
- Features: each map call receives K chronological raw increments over equal subdivisions of *its own* interval, as K·C extra channels. These sum to the explicit martingale residual `M_t−M_s`, which is therefore represented exactly. (source: ../archive/notes/methodology.md, ../archive/notes/code_equation_audit.md)
- `γ(s)(W_t−W_s)` is only an approximation of `∫_s^t γ dW`. Do not present it as exact noise. (source: ../archive/notes/flow_maps_and_code_design.md)
- A seed fixes a *query sequence*, not a query-order-independent Brownian tree. Equal seeds at different budgets therefore do **not** share paths. Same-path diagnostics must share one path object explicitly. (source: ../archive/notes/code_equation_audit.md, ../archive/notes/implementation_record.md, ../archive/notes/final_benchmark_protocol_audit.md)

### Composition: PSD versus Lagrangian
- Velocity-form PSD, for `s<u<t` with `λ = (u−s)/(t−s)`:
  - Target: `G(s,t) ≈ stopgrad[λ·G(s,u,x) + (1−λ)·G(u,t,X_u)]`.
  - `X_u` is the *generated* intermediate state (including `M_u−M_s`), never the observed Y.
  - All terms use one path and one history. Independent noise draws in the three terms are wrong.
  (source: ../archive/notes/methodology.md, ../archive/notes/flow_maps_and_code_design.md)
- **Interval weighting:** the endpoint residual equals `(t−s)` times the average-drift residual. Unweighted average-drift MSE is therefore endpoint MSE weighted by `1/(t−s)²`. It shares exact zeros with the endpoint loss but is not the same objective, and the difference matters when a zero residual is unattainable. The Itô-map paper calls the two "equivalent"; that is only true at the zeros. (source: ../archive/notes/flow_maps_and_code_design.md, ../archive/notes/methodology.md, ../archive/notes/reviews/adversarial_method_math_2026-10-07.md)
- PSD was chosen over Lagrangian or Eulerian training. PSD avoids JVPs through attention and second derivatives, and it reuses the trained drift networks.
  - Lagrangian: `G + (t−s)∂_t G = b(t, Φ)`. It is valid only if path features that depend on t are differentiated consistently.
  - MeanFlow/Eulerian identities do not transfer to Itô maps, because of the pathwise initial-time obstruction.
  (source: ../archive/notes/methodology.md, ../archive/notes/flow_maps_and_code_design.md)
- **Deterministic flow map:**
  - It must start from the Gaussian base Z of `FlowMatchingModel`, with history used only as conditioning. Setting γ=0 in a point-source FSI collapses the ensemble.
  - Never regress a deterministic map from c to random next states: the MSE optimum is the mean.
  - Direct FMM training needs reverse maps and symmetric time pairs. Do not call every endpoint regression "FMM".
  (source: ../archive/notes/flow_maps_and_code_design.md, ../archive/notes/methodology.md)

### Teacher rollout versus baseline Euler–Maruyama
- Refinement target: `X_{j+1} = X_j + δ·b_T(X_j,t_j;h) + [M_{t_{j+1}} − M_{t_j}]`. This is Euler drift plus **exact integrated diffusion**. Average-drift target = (endpoint − x − total noise) / (t−s). (source: ../archive/notes/code_equation_audit.md, ../archive/notes/methodology_implementation_audit.md, ../archive/notes/reviews/adversarial_method_math_2026-10-07.md)
- Baseline SI sampler: `+ γ(t_j)√δ ξ_j` (left-point Euler–Maruyama).
  - With zero drift, its endpoint variance is `ε²(1/3 + 1/(2n) + 1/(6n²))`.
  - Relative excess over the exact `ε²/3` is ≈9.6%, 4.7%, 2.4% and 1.2% at n=16, 32, 64, 128.
  - Noise quadrature alone can therefore shift apparent spread. This is not a fluid prediction.
  (source: ../archive/notes/reviews/adversarial_method_math_2026-10-07.md, ../archive/notes/reviews/editor_method_verification_2026-10-07.md)
- "32-step teacher" must not be read as "the baseline's 32-step sampler". (source: ../archive/notes/reviews/adversarial_method_math_2026-10-07.md)

### Finite-feature compression limits
- With finite features Z, squared-error regression learns `E[Y|Z]`, leaving an irreducible same-path error `E Var(Y|Z)`. A conditional-mean map can lose spread. This is a property of same-path regression, **not** a lower bound on CRPS or W2 for arbitrary transports. (source: ../archive/notes/broader_literature.md, ../archive/notes/code_equation_audit.md)
- Analytical OU example (`dX = −X dt + dW`): leftover variance is 0.0328 given W1 alone and 0.0022 given 4 increments. (source: ../archive/notes/broader_literature.md)
- Even exact conditional-mean predictors violate composition. With interval-total features, the direct and composed OU predictors have different increment coefficients for every h>0. So a positive PSD plateau is not optimiser failure, a low PSD loss does not prove SDE fidelity, and zero loss cannot be promised at finite K. (source: ../archive/notes/reviews/adversarial_method_math_2026-10-07.md, ../archive/notes/code_equation_audit.md)
- Child intervals resolve within-parent detail that the parent student cannot see. With an EMA target, the population target also drifts during training. (source: ../archive/notes/code_equation_audit.md)
- Error decomposition (diagnostic only, not a theorem): W2(student, P*) ≤ √(map approximation + compression) + W2(numerical teacher, continuous teacher) + W2(teacher, data). (source: ../archive/notes/methodology.md, ../archive/notes/reviews/adversarial_method_math_2026-10-07.md)

### Identifiability and assumption caveats (now in the paper)
- **Strong solutions:** state the assumptions explicitly. These are a regular drift version, a pathwise unique strong solution, finite second moments, and the diagonal as a continuous extension where it exists. Counterexample: c=0, y=0 gives `b = −x/(1−t)`, whose Lipschitz constant blows up as t→1. A bounded regression target does not make the drift regular. (source: ../archive/notes/reviews/adversarial_method_math_2026-10-07.md, ../archive/notes/reviews/adversarial_method_math_response_2026-10-07.md)
- **Finite-history surrogate:** autoregression samples the learned order-p kernel. It equals the true trajectory law only if the history is sufficient and the kernel is right on the histories actually encountered. Hidden-Θ Bernoulli counterexample: P(1|1) = 0.82 versus the true 0.890 after two ones. Long-run boundedness or spectra do not certify the joint law. (source: ../archive/notes/reviews/adversarial_method_math_2026-10-07.md)
- **Support gap:** off-diagonal intervals have gap ≥0.01 (direct) or ≥0.0025 (child), while the diagonal is trained at gap 0. No refinement to zero is enforced. (source: ../archive/notes/reviews/adversarial_method_math_2026-10-07.md)
- The interval distribution is not uniform over the ordered time triangle, so population uniqueness statements do not transfer. (source: ../archive/notes/methodology_implementation_audit.md)
- Propositions in the paper are elementary diagnostics (coupling bound, projection identity, Gaussian score illustration), not new theorems. No bound links the PSD loss to forecast-law error. Claim boundary: this is a conditional same-path adaptation and an approximation study. (source: ../archive/notes/reviews/adversarial_method_math_2026-10-07.md, ../archive/notes/final_implementation_review.md, ../archive/notes/broader_literature.md)
- NS forcing must be stated as spatially correlated or discretised forcing with unverified covariance. Continuum space-white forcing need not have finite enstrophy. (source: ../archive/notes/reviews/adversarial_method_math_2026-10-07.md)

---

## 2. Implementation facts and known pitfalls

### Architecture and classes
- `BaseMapModel` is a separate `nn.Module`, **not** a `BaseModel` subclass, so `isinstance(…, BaseModel)` checks fail. Subclasses:
  - `FlowMapModel`: Gaussian base.
  - `FollmerItoMapModel`: latest-field base plus explicit martingale.
  (source: ../archive/notes/map_api_compatibility.md, ../archive/notes/implementation_record.md)
- `MapConditionedNetwork` wraps the teacher UNet:
  - `TwoTimeEncoder`: `orig(s) + (t−s)·gap_mlp(t−s)`, with the last layer zero-initialised.
  - `PathConditionedInit`: a bias-free, zero-initialised 1×1 projection of the path channels, added after the original init/history block.
  - Effect: the student reproduces the teacher's diagonal exactly at initialisation. Teacher weights are transplanted with key/shape checks and the teacher is not saved in student checkpoints.
  - Do not just enable `field_cond_channels`; that inserts an extra history block.
  (source: ../archive/notes/flow_maps_and_code_design.md, ../archive/notes/code_equation_audit.md, ../archive/notes/implementation_record.md)
- Converting K4 to K16 while preserving the function: reshape the projection to `[hidden,4,C]`, **repeat** (do not divide) each coefficient over its 4 child increments, and restart the optimiser. (source: ../archive/notes/pilot_assessment_ito_psd_2000.md, ../archive/notes/ns_path_compression_assessment.md)
- Dropout must be 0 for map training; otherwise consistency targets contain dropout noise. Fused SDPA may not support forward-mode AD, which matters if JVP losses are revisited. (source: ../archive/notes/flow_maps_and_code_design.md, ../archive/notes/implementation_record.md)

### Settings actually run (initial stage)

| Setting | Stage 1 PSD | Stage 2 teacher rollout |
|---|---|---|
| Updates, batch | 2,000 (20×100), batch 8 | 500 (10×50) |
| Optimiser | AdamW 1e-4, cosine to 1e-5 | 5e-5 to 1e-5 |
| Diagonal and off-diagonal weights | 1 / 1, ramp 100 updates | 1 / 1, no ramp |
| Interval curriculum | Max gap grows 0.125→1 over 500 updates | Full range immediately |
| Endpoint-atom probability | 0.25 | 0.5 |
| Target | EMA, decay 0.95 | Frozen teacher, 32 Euler substeps (an EMA copy at 0.99 exists but is *not* the target) |
| Diagonal-only warmup | 0 | 0 |

- Other fixed settings: K=4, widths 16/32/64/128, teacher-anchored diagonal (the teacher replaces raw `R_t`).
- Interval sampler:
  - Draw length ℓ ~ U[0.01L, L], then s ~ U[0, 1−ℓ].
  - With the endpoint probability, set (s,t) = (0,L).
  - Draw split u = s + f(t−s), with f ~ U[0.25, 0.75].

(source: ../archive/notes/reviews/adversarial_method_math_response_2026-10-07.md, ../archive/notes/methodology_implementation_audit.md, ../archive/notes/validation_iteration_log.md, ../archive/notes/code_equation_audit.md)
- Config defaults versus run values were not re-verified against the current `config/*.yaml` (`follmer_ito_map_…yaml` currently has uncommitted edits). Check before relaunching. Map `num_steps` defaults to 1 (the solver default is 100). (source: ../archive/notes/map_api_compatibility.md, ../archive/notes/final_validation.json)
- Validation and checkpoint selection used a fixed-RNG, same-path teacher-rollout loss, even when training used pure PSD. It covered the full range of interval lengths, without curriculum or ramp. This loss measures **teacher fidelity, not forecast quality**, and it is not comparable across model families. (source: ../archive/notes/code_equation_audit.md, ../archive/notes/methodology_implementation_audit.md, ../archive/notes/validation_iteration_log.md)

### API compatibility limits
- **Masks:** `MapTrainer` raises `ValueError` for any non-trivial mask (guard added after the protocol freeze). Map sampling multiplies increments by the mask, but legacy Euler/EM steppers ignore it and training targets are unmasked. Masked domains need a new methodology. (source: ../archive/notes/map_api_compatibility.md, ../archive/notes/post_freeze_api_guard.json)
- **Parameter conditioning in `sample_trajectory`:** it inherits `pars_cond[:, i:i+1]`, which works only for one scalar per future step (`[B,T]`). General `[B,D]` vectors get truncated and `[B,T,D]` keeps a stray dimension. Use repeated single-step calls instead. The solver helper has the same bug. (source: ../archive/notes/map_api_compatibility.md)
- Time-varying spatial conditioning `[B,C,H,W,T_future]` is indexed from the first generated forecast, and its length is not validated. (source: ../archive/notes/map_api_compatibility.md)
- `sample` is history-first, while FSI's `sample` is base-first, so use keyword arguments. `num_physical_steps` in `sample_trajectory` is the *total* output length, including history. `sample` does not call `eval()`. (source: ../archive/notes/map_api_compatibility.md)
- Maps expose no `stepper`, `diffusion_term`, `.drift` or `.score`, so guidance and posterior wrappers cannot use them. Only the affine γ=a(1−t) Wiener interpolant is supported; there is no state-dependent or coloured diffusion. Generative noise is per-pixel independent and distinct from the physical forcing. (source: ../archive/notes/map_api_compatibility.md, ../archive/notes/final_implementation_review.md)
- The teacher/student compatibility check covers only interpolation class and amplitude; it is not a general proof of compatibility. (source: ../archive/notes/final_implementation_review.md)
- If base and target normalisation differ, or targets are tendencies, outputs cannot be appended to the history directly. NS is safe because it uses one scalar normalisation (mean 0, std 3.09969). (source: ../archive/notes/map_api_compatibility.md, ../archive/notes/final_benchmark_protocol_audit.md)

### Resume behaviour
- `MapTrainer` resumes at epoch boundaries and restores model, EMA, optimiser, scheduler, clipping and RNG state. Persistent DataLoader worker RNG is **not** serialised, so exact replay needs deterministic workers. (source: ../archive/notes/map_api_compatibility.md)
- Long-horizon runner:
  - Writes atomic `restart.pt` files (history, all RNG streams, journal, raw-state hashes, protocol hash).
  - Resume verifies provenance, can extend the horizon, and rolls back to the last committed lead.
  - It cannot resume through a non-finite failure.
  (source: ../archive/notes/long_horizon_runner.md)

### Inherited bugs and risks (legacy code)
- `Trainer.__init__` assumes a scheduler, and `_print_info` assumes a tracker. (source: ../archive/notes/flow_maps_and_code_design.md)
- `Trainer._compute_val_loss` runs under `no_grad`, which breaks derivative losses. (source: ../archive/notes/flow_maps_and_code_design.md)
- AMP clips gradients without `scaler.unscale_`. Use full precision for map losses. (source: ../archive/notes/flow_maps_and_code_design.md)
- `EarlyStopping` uses `val < best + delta`, so small degradations count as improvements. Use a stricter rule. (source: ../archive/notes/flow_maps_and_code_design.md)
- `StochasticNavierStokesDataset` crashed with `preprocesser=None`. A narrow fix exists. Its window count is `T−L−1` instead of `T−L` (kept for comparability with the teachers). (source: ../archive/notes/flow_maps_and_code_design.md, ../archive/notes/implementation_record.md)
- `BaseModel.to` handles device only; the mask is not a buffer in the legacy model. `sample` returns CPU arrays, so synchronise CUDA when timing. (source: ../archive/notes/flow_maps_and_code_design.md)
- Pretrained teachers differ in width: the FM config is 16/32/64/128, and the "small" SI config is 8/16/32/64. Never mix them silently. (source: ../archive/notes/flow_maps_and_code_design.md, ../archive/notes/experiment_protocol.md)
- The validation loader is shuffled with a fixed isolated seed so that a bounded validation budget covers all trajectories. (source: ../archive/notes/implementation_record.md)

### Evaluation code semantics
- `forecast_metrics` works in normalised coordinates; physical values are ×3.09969 for CRPS, RMSE, energy and spread. CRPS and energy use unbiased `E(E−1)` pair terms, which can go negative, so do not clamp them. The energy score divides by √dim. (source: ../archive/notes/code_equation_audit.md, ../archive/notes/final_benchmark_protocol_audit.md)
- Spread definitions:
  - Legacy "spread/skill" is `√((M+1)/M)` × the mean of pointwise population standard deviations. It is not RMS spread, and the docstring's claim that calibration implies a ratio of 1 is wrong.
  - Also report RMS spread (unbiased variance) with the `√((E+1)/E)` correction.
  - The reported per-case ratio mean is not the ratio of the means.
  (source: ../archive/notes/code_equation_audit.md, ../archive/notes/experiment_protocol.md, ../archive/notes/final_benchmark_protocol_audit.md)
- The legacy "relative enstrophy/spectral error" is an L2 ratio between the ensemble-averaged radial spectrum and the single observed spectrum, using unnormalised FFT shell averages. It is not comparable to the new shell-sum spectra (§4). (source: ../archive/notes/code_equation_audit.md, ../archive/notes/spatial_enstrophy_spectrum_definition.md)
- NFE is counted by a backbone hook (batched E members count as 1 call). Timing synchronises CUDA and includes path construction and CPU transfer, but excludes scoring. Warm-up is only one step. Per-lead time is per transition, and cumulative cost must be summed within a trajectory before averaging. (source: ../archive/notes/code_equation_audit.md, ../archive/notes/final_benchmark_protocol_audit.md)
- Autoregression replicates the history to E once. Each member then keeps its own generated history (no mean replacement, no truth), giving a fixed-size ensemble rather than a branching tree. Truth for lead l is at index `origin+l−1`. (source: ../archive/notes/final_benchmark_protocol_audit.md)

---

## 3. Review findings (method and maths)

| Item | Objection | Status |
|---|---|---|
| P1 | Strong existence, pathwise uniqueness and endpoint regularity were assumed but not stated (counterexample c=y=0) | Resolved in text: assumptions added in §3 and App. A.1; endpoint transition treated as an extra assumption |
| P1 | A one-step order-p kernel is not the true trajectory law (hidden-Θ counterexample) | Resolved in text: "finite-history surrogate" plus a sufficient-history condition |
| P1 | Finite-feature PSD is not an identified exact-semigroup problem (OU composition counterexample; gap support) | Resolved in text beside the objective. **Remains an intrinsic limitation** |
| P2 | Teacher uses exact integrated diffusion while the comparator uses Euler–Maruyama | Resolved in text (eq. for the numerical teacher). **Exact-diffusion solver control is still open** |
| P2 | Propositions are coupling/projection diagnostics, not loss guarantees | Resolved: section renamed "Approximation Errors and Diagnostic Bounds" |
| P2 | Average-drift MSE implies `1/(t−s)²` endpoint weighting | Resolved: stated, with the full objective table in App. A.2 |
| P3 | W must be a Brownian *process* independent of (h,y); only *increments* after s are independent of the state at s; moment conditions; warmup was 0 (text implied otherwise); covariance identity is at fixed (s,t,x,h); forcing covariance | All resolved in text |

(source: ../archive/notes/reviews/adversarial_method_math_2026-10-07.md, ../archive/notes/reviews/adversarial_method_math_response_2026-10-07.md, ../archive/notes/reviews/editor_method_verification_2026-10-07.md)
- The reviewer also confirmed as sound:
  - The diagonal target (no √t derivative).
  - Fixed h across child maps.
  - The bridge mean and variance.
  - Additive-noise cancellation.
  - Independence between the interpolant start state and the future path.
  - Zeroed branches on the diagonal.
  - A Gaussian base for the flow map.
  - Correct use of the Föllmer naming caveat.
  (source: ../archive/notes/reviews/adversarial_method_math_2026-10-07.md)
- The editor verification was *not independent*: it only checked that the corrections appear in the PDF and match `map_trainer.py` and `sde_solvers.py`. (source: ../archive/notes/reviews/editor_method_verification_2026-10-07.md)
- An independent pre-results code review found no correctness blocker for the *unmasked, full-state, equal-normalisation, unconditioned* NS setup only. It certifies nothing beyond that setup. (source: ../archive/notes/final_implementation_review.md, ../archive/notes/methodology_implementation_audit.md)

---

## 4. NS data, provenance and protocol lessons for the new experiments

### Data and provenance
- `data/stochastic_navier_stokes/data.npz` holds only `state`: 200 trajectories × 100 snapshots × 128², scalar vorticity, no metadata. The companion `data_file.pt` holds `[states 200×100×256², times(100)]`. Every 128² field is **exactly** `full[..., ::2, ::2]` (stride subsampling, not pooling). (source: ../archive/notes/navier_stokes_provenance.md, ../archive/notes/long_horizon_reference_feasibility.md)
- Times run from 49.99 to 99.48, with mean spacing 0.49989997, nominally **0.4999**. This is consistent with 4,999 substeps of 1e-4 and floor division. It supersedes the earlier "spacing unrecoverable" statement. (source: ../archive/notes/long_horizon_reference_feasibility.md, ../archive/notes/navier_stokes_provenance.md)
- **Still unknown:** viscosity, drag, forcing amplitude, generator revision, spin-up and absolute time origin. The current jax_cfd script (256², ν=1e-3, drag 0.1, dt 1e-4, saved spacing 0.5, 5 initial conditions, writes `trajectory.npz`) does not match the archive. It also uses Euler–Maruyama despite a stale comment mentioning CN-RK4. Its forcing has 8 Gaussian coefficients on modes (6,0), (7,0), (5,5), (8,8). Treat it as corroborating context only. (source: ../archive/notes/navier_stokes_provenance.md, ../archive/notes/long_horizon_reference_feasibility.md)
- Report lead time in **stored-snapshot intervals**; a time axis (lead × 0.4999) is secondary. Do not claim a physical solver comparison until the generation manifest is recovered. (source: ../archive/notes/navier_stokes_provenance.md, ../archive/notes/long_horizon_protocol.md)
- Teacher pretraining logs, cost and seed variation are **unavailable**; report them as such. Only student cost can be measured. (source: ../archive/notes/code_equation_audit.md, ../archive/notes/final_selection_plan.md, ../archive/notes/final_implementation_review.md)

### Protocol lessons
- Split by whole trajectories (train 0:160, validation 160:180, test 180:200), never by windows. Fit normalisation on training data only; whether 3.09969 came from training data is unverified. History length is 5. (source: ../archive/notes/experiment_protocol.md, ../archive/notes/flow_maps_and_code_design.md)
- Select every hyperparameter and checkpoint (including K) on validation, using a predeclared rule. Freeze configs and hashes before opening test. Label any test-informed change as exploratory. Use one checkpoint for all budgets. (source: ../archive/notes/experiment_protocol.md, ../archive/notes/final_selection_plan.md, ../archive/notes/final_benchmark_protocol_audit.md)
- Predetermine origins, ensemble size, seeds and qualitative case. Never pick figures or members by appearance. Keep incomplete runs visible. (source: ../archive/notes/final_benchmark_protocol_audit.md, ../archive/notes/final_selection_plan.md)
- Aggregation and uncertainty:
  - Average origins within a trajectory, then bootstrap whole trajectories; never pixels.
  - Use **paired** differences on shared resamples, not overlap of separate confidence intervals.
  - Lead curves should reuse one index matrix; their bands are pointwise, not simultaneous.
  - Student-seed variation is separate from trajectory variation. Never pool ensemble members across trained models.
  - No equivalence margin was preregistered, so a zero-crossing CI does not show equivalence.
  (source: ../archive/notes/final_benchmark_protocol_audit.md, ../archive/notes/heldout_three_seed_assessment.md, ../archive/notes/code_equation_audit.md)
- Report NFE **and** synchronised latency at the same E and precision. Map inputs are slightly larger than the teacher's, so equal NFE does not mean equal FLOPs. Include a high-step reference (128 steps) to check solver saturation; 64 steps is not "converged". (source: ../archive/notes/methodology.md, ../archive/notes/experiment_protocol.md, ../archive/notes/final_selection_plan.md)
- Calibration metrics:
  - Expected coverage of an empirical 5–95% interval under perfect calibration is 0.9(E−1)/(E+1): 70.0%, 79.4% and 84.6% at E=8, 16, 32. It never reaches 90%.
  - Ideal extreme-rank fraction is 2/(E+1).
  - Give rank histograms and fair CRPS more weight than raw coverage.
  (source: ../archive/notes/code_equation_audit.md, ../archive/notes/experiment_protocol.md, ../archive/notes/final_benchmark_protocol_audit.md)
- Gaussian toy: a 16% spread deficit raises expected CRPS by only 0.69% but drops population-90% coverage to 83.3%. A low mean RMSE can coexist with collapsed spread; the FM1 baseline was the extreme case. (source: ../archive/notes/gaussian_score_calibration.md, ../archive/notes/heldout_seed0_assessment.md)
- Count simulation-based refinement and the shared PSD source in cumulative training cost. Label smoke runs, pilots and full experiments separately. (source: ../archive/notes/validation_iteration_log.md, ../archive/notes/experiment_protocol.md)

### Long-horizon rollout lessons
- Truth is limited: from a 5-frame history, only **95** future truth frames exist (leads 1–95). RMSE, CRPS and spread–skill must be null beyond lead 95 — never padded or extrapolated. Long runs can only assess boundedness and distribution of a single trajectory. They do not show accuracy, invariant-measure correctness or stability for other initial conditions. (source: ../archive/notes/long_horizon_protocol.md, ../archive/notes/long_horizon_reference_feasibility.md)
- Compute all diagnostics on **denormalised** vorticity in float64. Area-mean enstrophy is `Z = ½·mean(ω²)` (no domain area assumed). Implementation check: `mean_e Z_e = Z(mean field) + (E−1)/(2E)·S²`. The enstrophy of the mean decays even when members keep their energy, so plot the two separately. (source: ../archive/notes/long_horizon_protocol.md, ../archive/notes/long_horizon_runner.md)
- **Spatial enstrophy spectrum:**
  - Full FFT with `norm='forward'`; modal power `½|ω̂|²`.
  - Bin into the nearest-integer shell `k = floor(√(kx²+ky²)+0.5)`, giving shells 0–91, and **sum** within each shell. This satisfies Parseval exactly.
  - Keep corner shells above 64 (mark them as incomplete annuli). Keep DC in the data but omit it from log plots.
  - The model curve is the mean of member spectra, with 5–95% member bands. The spectrum of the ensemble-mean field is separate and must not be substituted.
  - The frequency unit is cycles per grid domain.
  (source: ../archive/notes/spatial_enstrophy_spectrum_definition.md)
- Historical context band:
  - Pool the 20 test trajectories × 100 snapshots.
  - Pooled enstrophy: mean ≈4.67, 5–95% range ≈3.15–6.53. Use this as a sanity range only.
  - The band is descriptive: it is not a prediction interval or future truth, and stationarity is unverified.
  - The 2,000 fields are temporally correlated, so do not build an i.i.d. confidence interval from them.
  (source: ../archive/notes/long_horizon_reference_feasibility.md, ../archive/notes/long_horizon_protocol.md)
- **Physical reference feasibility:**
  - Assumed-parameter CPU Euler–Maruyama costs ≈6.3 s per stored interval at 256² (7 h per 4,096-lead path; 14–57 CPU-h for 8 members).
  - 128² is cheaper but changes the dynamics.
  - No continuation was run. The helper `simulate_assumed_ns_reference.py` labels its output `is_verified_dataset_continuation: false`.
  - A paired discretisation study would need the same Wiener increments aggregated across meshes.
  (source: ../archive/notes/long_horizon_reference_feasibility.md)
- Runtime and procedure:
  - Initial-stage CPU profiling gave ≈1.5 s per lead (map, 1 step) and ≈12 s per lead (Itô, 8 steps); use the GPU.
  - Long-run wall times on a shared GPU are diagnostic only and must not be used for speedup claims.
  - Stop only on non-finite output; record enstrophy threshold crossings (10/100/1000×) as flags.
  - Do not clip, renormalise or reset to truth.
  (source: ../archive/notes/long_horizon_runner.md, ../archive/notes/long_horizon_runtime_context.md, ../archive/notes/long_horizon_protocol.md)
- Equal seeds across models or budgets do not give paired Brownian paths. A single initial condition does not provide independent calibration cases. (source: ../archive/notes/long_horizon_protocol.md)

---

## 5. Qualitative lessons from initial-stage experiments (**initial-stage, not for the paper**)

All of these came from the old pipeline: one teacher pair, fixed pretrained teachers, E=8 on 4 validation trajectories or E=16 on 20 test trajectories, 3 student seeds. **Every number below is superseded.**
- **PSD alone was insufficient.**
  - After 2,000 PSD updates, Itô 1-step was badly underdispersed (CRPS ≈29% worse than SI64, about half the spread).
  - 1 and 2 steps were similarly poor, which points to off-diagonal learning as the bottleneck.
  - The validation teacher-map loss plateaued at ≈0.034.
  (source: ../archive/notes/pilot_assessment_ito_psd_2000.md, ../archive/notes/validation_iteration_log.md)
- **Teacher-rollout refinement helped substantially.** 500 updates with 32 teacher steps lifted Itô-1 validation CRPS from 0.366 to 0.305 and restored much of the spread. It still did not match SI64 on any validation trajectory. Caveat: this conflates the change of objective with extra compute and teacher simulation. (source: ../archive/notes/validation_iteration_log.md, ../archive/notes/methodology_implementation_audit.md)
- **K16 did not beat K4,** despite better path information:
  - Nested Monte Carlo on one history (trajectory 160, origin 45) gave an omitted endpoint-variance fraction of ≈28% at K1, ≈9.7% at K4 and ≈0.66% at K16. The ideal retained standard deviation was ≈95% at K4 and 99.7% at K16.
  - So compression explains only a small part of the spread loss.
  - With matched ancestry and 500 updates, K16 was slightly *worse* (validation CRPS 0.309 vs 0.305) and ~16% slower at 1 step.
  - More information was not learned within that budget.
  (source: ../archive/notes/ns_path_compression_assessment.md, ../archive/notes/validation_iteration_log.md, ../archive/notes/final_selection_plan.md)
- **The deterministic flow map (Gaussian base) was stronger at one step.**
  - Flow-map-1 was within ≈0.6% of FM64 CRPS at ≈56× lower latency, and within ≈0.1% at lead 10.
  - It had better spread than Itô-1 and beat Itô-1 in every student seed.
  - At 8 steps the two maps were close.
  - There was no evidence that the stochastic map is better.
  (source: ../archive/notes/heldout_seed0_assessment.md, ../archive/notes/heldout_three_seed_assessment.md, ../archive/notes/final_implementation_review.md)
- **Itô maps: close proper scores, but underdispersed.**
  - Itô-8 versus SI64: CRPS ≈+1.3% at one step and ≈+1.7% at lead 10.
  - Coverage was ≈5 pp lower and the extreme-rank fraction ≈4–5 pp higher, consistently across seeds.
  - Itô-1 rollout spread/skill fell to ≈0.63 by lead 10.
  - The spectral error of Itô-8 at lead 10 was ≈15% worse, so one-step spectral closeness did not persist.
  (source: ../archive/notes/heldout_three_seed_assessment.md, ../archive/notes/heldout_seed0_assessment.md)
- **Itô-1 beat SI16** (≈5% better CRPS, better calibration, ≈13× faster). This "beats a moderately resolved solver" result does not extend to SI64 or SI128. (source: ../archive/notes/heldout_seed0_assessment.md, ../archive/notes/heldout_three_seed_assessment.md)
- **Step count was not monotone.** For both maps, 1→2→4 steps often got worse and 8 recovered; spread contracted at intermediate budgets. Flow-map-8 was worse than flow-map-1. The lesson: say the budget is selectable, not that quality improves with steps. (source: ../archive/notes/heldout_seed0_assessment.md, ../archive/notes/validation_iteration_log.md, ../archive/notes/methodology_implementation_audit.md)
- **Latency ratios (order of magnitude):** a 1-step map gives ≈50–60× over a 64-step solver; an 8-step map gives ≈7×. Path construction adds little. Student training took ≈19 min per student for both stages, excluding teacher training. (source: ../archive/notes/heldout_three_seed_assessment.md, ../archive/notes/final_implementation_review.md)
- **Ten-lead rollouts:** all learned models were far better than persistence, errors plateaued after the early leads, and nothing blew up. Two 4,096-lead map runs stayed finite (the solver references were incomplete when the notes were written). This shows limited-horizon boundedness only. (source: ../archive/notes/heldout_seed0_assessment.md, ../archive/notes/long_horizon_runner.md, ../archive/notes/long_horizon_reference_feasibility.md)
- **Student-seed variation was small** (CRPS SD ≈0.001–0.003) and the ordering held across seeds. This says nothing about teacher-seed variation. (source: ../archive/notes/heldout_three_seed_assessment.md)

---

## 6. Open questions and TODOs for the new experiments

1. **Exact-diffusion solver control:** run SI with exact integrated noise (and Euler drift) at matched NFE, so that map-learning gains can be separated from the noise-quadrature effect. (source: ../archive/notes/reviews/adversarial_method_math_response_2026-10-07.md)
2. **Underdispersion:** find the main cause among PSD self-consistency bias, teacher-target choice, the `1/(t−s)²` weighting, the EMA target, the support gap near the diagonal, and K. Candidate fixes:
   - Spread-aware or CRPS-type fine-tuning.
   - Endpoint-weighted loss.
   - Longer refinement for K16 (more information did not translate under a 500-update budget).
   (source: ../archive/notes/validation_iteration_log.md, ../archive/notes/ns_path_compression_assessment.md, ../archive/notes/reviews/adversarial_method_math_2026-10-07.md)
3. **Stochastic versus deterministic map:** state what advantage the Brownian structure is supposed to deliver (pathwise coupling, calibration, rollouts?) and design a test for it. In the initial stage the flow map matched or beat the Itô map. (source: ../archive/notes/heldout_three_seed_assessment.md, ../archive/notes/broader_literature.md)
4. **Strict KL-optimal Föllmer variant (`g_F`, score-corrected drift):** only as a properly derived ablation with endpoint handling. (source: ../archive/notes/methodology.md)
5. **NS provenance:** recover viscosity, drag, forcing and generator revision, or generate a *new* fully documented dataset with conditional reference ensembles. That would also enable true conditional calibration checks (as in StocBench) and a physical long-horizon reference. Decide whether the 128²-from-256² subsampling (no anti-aliasing) is acceptable. (source: ../archive/notes/navier_stokes_provenance.md, ../archive/notes/long_horizon_reference_feasibility.md, ../archive/notes/broader_literature.md)
6. **Teachers:** retrain SI and FM teachers with matched widths and logged cost and seeds, so the comparisons are fair and teacher-seed variance can be estimated. (source: ../archive/notes/experiment_protocol.md, ../archive/notes/final_selection_plan.md)
7. **Sufficient history:** test whether 5 frames suffice, for example by varying the history length. Long rollouts only assess the finite-history surrogate. (source: ../archive/notes/reviews/adversarial_method_math_2026-10-07.md)
8. **Preregistration:** set an equivalence or noninferiority margin before the test if "comparable quality" is to be claimed. (source: ../archive/notes/final_selection_plan.md, ../archive/notes/final_implementation_review.md)
9. **Code fixes to consider before relaunch:**
   - The `EarlyStopping` criterion.
   - AMP `unscale_`.
   - Trajectory `pars_cond` shapes.
   - Validation of the `field_cond` sequence length.
   - The `T−L−1` window count (change it only for all models together).
   - Persistent worker RNG on resume.
   (source: ../archive/notes/flow_maps_and_code_design.md, ../archive/notes/map_api_compatibility.md)
10. **Related work:** position against RecFM, Swift, PixelFlowCast, StocBench, DYffusion and GenCast. Few-step probabilistic fluid forecasting is not novel in itself. A head-to-head against consistency distillation or RecFM would strengthen the paper. (source: ../archive/notes/broader_literature.md)

---

## Original file → what was kept / why the rest is archive-only

| File | Kept here | Archive-only because |
|---|---|---|
| ../archive/notes/methodology.md | Interpolant, diagonal target, variance clock and bridges, PSD form, Föllmer distinction, error decomposition, acceptance checks | Pseudocode and alternative global-feature designs superseded by the implemented lazy bridges |
| ../archive/notes/flow_maps_and_code_design.md | Flow-map base requirement, FMM/Lagrangian notes, warm-start wrapper design, 12 legacy-code hazards, PSD weighting | Design proposals already implemented |
| ../archive/notes/code_equation_audit.md | Equation→code table facts, finite-feature PSD argument, metric semantics, coverage bound, two-stage summary | Handover gaps and file-level detail |
| ../archive/notes/broader_literature.md | Related-work list (condensed), claim boundary, OU compression numbers | Link details and source hashes |
| ../archive/notes/experiment_protocol.md | Splits, selection, baselines, metric and timing rules, finite-E caveat | Old budget lists |
| ../archive/notes/final_benchmark_protocol_audit.md | Autoregressive semantics, units, CRPS/energy estimators, timing and aggregation rules | Frozen-test specifics and report command lines |
| ../archive/notes/final_implementation_review.md | No-blocker scope, API boundaries, interpretation rules | Three-seed checklist, macro audit, old numbers |
| ../archive/notes/finalize_long_horizon.md | — (procedure only) | Finalizer CLI for old runs |
| ../archive/notes/final_selection_plan.md | Selection-rule lessons, K16 motivation | Old frozen-test plan |
| ../archive/notes/final_validation.json | Settings-check pointer only | Test count and hash list |
| ../archive/notes/gaussian_score_calibration.md | CRPS versus coverage sensitivity result | Script, Monte Carlo details, figure |
| ../archive/notes/heldout_seed0_assessment.md | Qualitative lessons (§5) | Superseded seed-0 tables and confidence intervals |
| ../archive/notes/heldout_three_seed_assessment.md | Qualitative lessons, cost scale (§5) | Superseded three-seed tables |
| ../archive/notes/implementation_record.md | Class list, scope decisions | Early test counts and smoke-run outputs |
| ../archive/notes/long_horizon_protocol.md | Truth limit, enstrophy and spread definitions, identity, interpretation limits | Old run configuration and GPU sharing notes |
| ../archive/notes/long_horizon_reference_feasibility.md | Timestamp recovery, context band, PDE cost, helper and RNG notes | Audit hashes and commands |
| ../archive/notes/long_horizon_runner.md | Resume semantics, stop rule, diagnostics, CPU cost | Queue PIDs and scheduling chatter |
| ../archive/notes/long_horizon_runtime_context.md | "Wall times are diagnostic" lesson | Execution history |
| ../archive/notes/map_api_compatibility.md | All API limits (masks, `pars_cond`, normalisation, resume) | — (fully absorbed) |
| ../archive/notes/methodology_implementation_audit.md | Interval sampler, validation semantics, interpretation limits | — |
| ../archive/notes/navier_stokes_provenance.md | Archive facts, simulator mismatch, unknown parameters | Byte sizes |
| ../archive/notes/ns_path_compression_assessment.md | Estimator idea, K1/K4/K16 lesson, K16 initialisation | Exact tables and teacher hash |
| ../archive/notes/pilot_assessment_ito_psd_2000.md | PSD-only failure mode, next-step reasoning | Pilot table |
| ../archive/notes/post_freeze_api_guard.json | Mask guard exists | Hashes |
| ../archive/notes/source_snapshot.json | — | Hash list only |
| ../archive/notes/spatial_enstrophy_spectrum_definition.md | Full spectrum definition and conventions | Preview file paths and commands |
| ../archive/notes/validation_iteration_log.md | Settings run, refinement and K16 outcomes (§5) | Per-step tables |
| ../archive/notes/reviews/adversarial_method_math_2026-10-07.md | All P1–P3 objections, counterexamples, sound points | Line references and suggested wording |
| ../archive/notes/reviews/adversarial_method_math_response_2026-10-07.md | Resolutions, run-settings table, open exact-diffusion control | Page-layout notes |
| ../archive/notes/reviews/editor_method_verification_2026-10-07.md | Non-independence of the editor pass; code match confirmed | Page and PDF-hash mapping |
| ../archive/notes/reviews/checkout_paper_build.json | — | Video paper-build hash record (video dropped) |
| ../archive/notes/reviews/final_video_pdf_inspection.json | — | Video results (dropped) |
| ../archive/notes/reviews/final_video_pdf_inspection.md | — | Video results and layout (dropped) |
