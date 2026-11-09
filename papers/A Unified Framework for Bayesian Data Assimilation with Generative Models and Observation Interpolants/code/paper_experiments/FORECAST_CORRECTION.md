# Forecast-residual correction for sequential assimilation

This note describes a correction for the bias in the paper's sequential (multi-step)
assimilation, how to implement it in this code base, and how to test it. It assumes no
access to the session in which it was written.

Status: **implemented in the library (section 4); checked in linear-Gaussian models
only.** The numbers in section 6 come from scratch studies with closed-form transition
kernels and, where marked, the trained Lorenz 96 priors. They are preliminary.

Branch `review/forecast-correction`, based on `review/lorenz96` (which carries the
Lorenz 96 case, its trained-prior pipeline and the PF reference). PRs target
`ICLR-2027-review`.

## 1. The problem

For each member i, Algorithm 1 samples p(x^{k+1} | x^k_i, y^{k+1}) and feeds the sample
into the next step. The filtering posterior is

    p(x^{k+1} | y^{1:k+1}) = ∫ p(x^{k+1} | x^k, y^{k+1}) p(x^k | y^{1:k+1}) dx^k,

so each member should also carry the weight p(y^{k+1} | x^k_i), the predictive likelihood
of the new observation given the member's previous state. Algorithm 1 drops it.

Consequences, established on `review/multistep-linear-gaussian`:

- After the first step the method converges to a stable but biased approximation of the
  filtering posterior. In the scalar case A=Q=H=R=1 the steady variance is 2/3 and the gain
  is 0.5, against Kalman's 0.618 for both. More sampler steps or members do not close the
  gap; a perfect per-member sampler has it too.
- **Spread-skill cannot detect it.** In the linear-Gaussian case the biased chain has
  spread equal to skill exactly. The usable symptom is positive lag-1 autocorrelation of
  the innovations y^{k+1} − H E[x^{k+1}].
- The bias grows when observations are informative relative to the transition noise in
  the observed directions, under advection, and for unobserved directions with weak
  cross-covariance to the sensors.

Why not weights: with E=64 at high dimension all importance-weighting schemes failed.
Global weights leave an ESS of about 2-7 at d=1024. SURGE path weights are worse
(their expected weight is p(y|x^k_i), so their variance is at least that of the exact
weights). Localised resampling splices fields and leaves seams. A learned weight only
re-estimates the same degenerate weights.

## 2. The correction

Split the forecast covariance as P_f = Cov_i(μ_i) + Q, with μ_i = E[x^{k+1} | x^k_i] and Q
the transition-noise covariance. The guided sampler already handles Q, exactly and without
Gaussian assumptions. What it misses is the spread of the μ_i across members. The
correction supplies that part with a Kalman-type step:

    x''_i = x'_i + (K_ens − K_q) (y + e_i − H f_i)

| Symbol | Meaning |
| --- | --- |
| x'_i | the unchanged guided sample of member i |
| f_i | an **unguided** forecast from the same x^k_i, with noise independent of x'_i |
| e_i | a fresh draw from N(0, R) |
| K_ens | P_f H' (H P_f H' + R)^-1, with P_f the localised sample covariance of the f_i |
| K_q | Q̂ H' (H Q̂ H' + R)^-1, the gain the sampler already applies |
| Q̂ | an estimate of Q, built offline (section 4.3) |

### Why it is exact in the linear-Gaussian case

The sampler draws x'_i = μ_i + K_q (y − H μ_i) + ξ_i with ξ_i ~ N(0, (I − K_q H) Q). Take
f_i = μ_i + η_i, η_i ~ N(0, Q) independent of ξ_i, and D = K_ens − K_q, S_q = H Q H' + R.

- **Mean:** x''_i has mean μ_i + K_ens (y − H μ_i), the Kalman (perturbed-observation) mean.
- **Covariance:** the added noise is D (e_i − H η_i), with covariance D S_q D'. Using
  K_q S_q = Q H', the total within-member covariance is
  Q − K_ens H Q − Q H' K_ens' + K_ens S_q K_ens', the Joseph form. Across members the
  μ-part has covariance (I − K_ens H) Cov(μ) (I − K_ens H)', so the ensemble has the
  Kalman analysis covariance.

The independence of f_i from x'_i is what makes the covariance come out right.

### Why not use x'_i in the innovation

y − H x'_i is the residual after the sampler has already pulled towards y. In the scalar
case it equals (1 − K_q)(y − μ_i) − ξ_i. The total gain becomes K_q + (K_ens − K_q)(1 − K_q),
which under-corrects (0.559 instead of 0.618 in the example above), and the noise ξ_i
appears in both terms, so the covariance is wrong too. The mean can be fixed by rescaling
the gain to (K_ens − K_q)(I − H K_q)^-1, but no matching noise term has been worked out.
`tests/test_forecast_correction_lg.py` checks both statements.

### What it keeps and what it does not

- H and R stay free at test time, and nothing is retrained.
- It works with SI-SDE, DM-SDE and FM-ODE, because it sits outside the sampler and uses
  only samples.
- With E=64 it tolerates rank deficiency better than an EnKF, because the ensemble only
  has to carry Cov(μ) while the sampler handles Q at full rank.
- **It is a Gaussian closure on the across-member part.** When p(y | x^k) varies in a
  strongly non-Gaussian or multimodal way across members, it removes only part of the
  bias. With a Gaussian transition kernel it is about as good as an EnKF, not better.

### Relation to "unguided prior SDE + EnKF update"

That alternative forecasts with the unguided prior and applies a full EnKF update. The
correction differs in that the Q-part is conditioned by the generative sampler, at full
rank and without Gaussian assumptions; only Cov(μ) goes through the low-rank Gaussian
step. Expect an advantage only where the transition is non-Gaussian, Q is a large share
of P_f, or E is small for the dimension. This alternative **must** be run as a baseline
(section 5).

## 3. Variants

| Variant | Forecast f_i | Extra cost per member per step | Status |
| --- | --- | --- | --- |
| **A. Forecast sample (recommended)** | one unguided sampler run from x^k_i | one prior run (M passes, no Jacobians) | exact in LG; checked |
| B. Predicted mean | μ̂_i from one network pass, plus a residual draw from the Q̂ bank | 1 pass | cheaper; not checked independently |
| C. Anchor shift | move x^k_i by a one-lag smoother update before the sampler | 1 pass | exact in LG; worse on Lorenz 96 |

For variant B: for flow matching the τ=0 velocity gives E[x¹ | x⁰] directly. For the
repository's quadratic stochastic interpolant β̇(0)=0, so the τ=0 drift carries no
information and the pass must be at τ > 0. The scratch studies used τ between 0.05 and
0.3 and disagreed on the best value; measure it.

## 4. Implementation

Implemented in `src/scisi/posterior_models/forecast_correction.py` (variant A), with
tests in `tests/test_forecast_correction.py`. The docstrings are the reference; this
section summarises the design and where it departs from the original outline.

### 4.1 Where it goes

The correction runs inside `BasePosterior.sample`, right after `_post_sample` and
before the new state is appended, so `field_history[..., -1]` is still each member's
x^k_i (helper `BasePosterior._apply_forecast_correction`, identity when off). It is
configured through the attribute `posterior.forecast_correction` (default `None`), or
per call with `sample_trajectory(..., forecast_correction=corr)`, which sets the
attribute for that call and restores it. Consequences:

- Callers that drive `sample` directly (the multistep linear-Gaussian case) get it by
  setting the attribute. When enabled, **every** `sample` call is corrected, including
  single-step runs. It is opt-in, so that is intended.
- Free-running steps of `sample_trajectory` call `self.model.sample` and are never
  corrected.
- It runs after the subclass `_post_sample`, so subclass overrides need not call it.
  SMC posteriors (SURGE, SI with `resample=True`) raise: they resample `base` and
  `field_history` together, so the pairing survives, but their weights already use
  the observation, and a correction on top would count it twice in either order.
- `DFlowPosterior` overrides `sample` and calls the helper itself. Any other override
  must do the same; `sample_trajectory` raises if an enabled correction was not
  applied at an assimilation step.
- Default off is bit-for-bit identical (outputs and global RNG state), checked
  against outputs saved before the change and in the tests.

The original outline started the forecast from `prev_base` kept from before
`self.sample`. That is unsafe for SI: `sample` writes the SI `base` in place
(`base[sl] = ...`), so the kept tensor already holds x'_i. The implementation starts
the forecast from `field_history[..., -1]`.

The forecasts f_i (and the online second forecast f'_i) are drawn under `no_grad`
after the guided sample, in chunks of the sampler's `batch_size`, from the global
stream, or with `forecast_seed` from forked generators (CPU and the devices in use)
seeded with a hash of `(forecast_seed, call counter)`. The hash keeps them apart from
a driver that seeds `torch` with the same number, and the counter is never reset, so
every call and trajectory gets fresh forecast noise. They use `forecast_num_steps`
(default: the sampler's `num_steps`; D-Flow: its `ode_steps`) and `forecast_stepper`
(default: the sampler's stepper).

**The forecast is the posterior's own SDE/ODE without guidance**
(`draw_unguided_forecast`): SI uses `model.sample` with the posterior's diffusion,
FM-ODE the FM ODE from N(0, I), FM-SDE (DM-SDE) the lifted reverse SDE
`v + 1/2 g^2 s`, and D-Flow its midpoint ODE. FM forecasts honour
`drift_time_shift`. An ODE stepper with an SI or FM-SDE posterior raises, because the
forecast would be deterministic (SI from its point mass: zero online residuals,
K_q = 0) or noiseless.

**Finding: the free-running branch is wrong for FM-SDE (DM-SDE).** It calls
`FlowMatchingModel.sample(diffusion_term=g)`, which integrates `v dt + g dW`, without
the score lift `1/2 g^2 s`. In the closed-form linear-Gaussian case at 50 steps its
forecast variance is 1.25-1.30 Q, against 0.94-0.97 Q for the lifted SDE, which the
correction uses. The free-running branch itself is unchanged, to keep existing
numbers; this affects only DM-SDE runs with `assimilate_every > 1`.

### 4.2 The correction module

`ForecastCorrectionConfig` fields, with defaults: `enabled=False`,
`forecast_num_steps=None`, `forecast_stepper=None`, `localisation=None`,
`gain_scale=1.0`, `q_source="online"`, `residual_bank=None`, `q_variance=None`,
`group_size=None`, `update_cap=None`, `post_update=None`, `forecast_seed=None`.
`sample_trajectory` and `posterior.forecast_correction` take a `ForecastCorrection`
(or `None`).

- **Units and H, R.** Everything is in model (standardised) units. H is
  `likelihood_model.obs_operator` and R is `likelihood_model.original_variance`
  (scalar or a diagonal `[m]`), which every likelihood in the repository holds in model
  units. `ForecastCorrection.correct(x, f, y, operator, obs_noise_var, f2=...)` takes
  them explicitly. The update assumes a linear (or linearised) H: observation
  coordinates are H c / H 1, and the bank/fixed Q̂H' and the cap apply H to
  increments. The online Q̂H' uses (H f − H f')/√2.
- **No d x d.** Per group, it forms the m x m innovation covariances (Cholesky) and
  applies the d x m gains in row chunks; one taper block serves every group and both
  gains. Without localisation the gain goes through the E anomalies, with the product
  order that avoids an E x E matrix at large E. The linear algebra is float64 (on the
  CPU when the state is on MPS, which has neither float64 nor `cholesky_solve`); H is
  applied on the state's device and dtype. For the 32²→128² operator (m=1024) one
  group costs about E d m multiply-adds, and the cached bank K_q is one d x m matrix
  per (H, R).
- **Groups.** `group_size=E` treats each block of E contiguous members as an
  independent ensemble. The last group takes the remainder (a single leftover member
  joins the previous group). Observations may be `[1, m]`, one row per group, or one
  row per member.
- **Cap and hook.** `update_cap` (opt-in) limits sqrt(u' S⁻¹ u / m) with u = H δ_i and
  S the localised H P_f H' + R. `post_update(x_new, x_guided)` runs after the update
  (for projection or masks).
- **Diagnostics.** `corr.diagnostics.records` holds one dict per corrected step, with
  per-group innovation mean, mean squared normalised innovation, lag-1 product, update
  RMS, forecast spread `spread_rms`, `q_rms` (so the Q share of the forecast variance
  is (q_rms/spread_rms)²) and cap fraction. `lag1_autocorrelation()` and `as_dict()`
  summarise them.

**Localisation.** `CoordinateLocalisation(state_coords, kernel, state_shape, mask)`
provides `state_obs(H, rows, ref)` / `obs_obs(H, ref)`. Observation coordinates are
derived from H as `H c / H 1`, which gives the sensor location for selection operators
and the block centre for block averages, so H stays free. Tapers are cached per
operator on the compute device, in float64. `grid_localisation(shape, radius,
periodic, channel_axis=True, mask=None)` covers 1D (`(1, 1, d)`, identical to
`enkf.periodic_taper` when periodic), 2D and 3D grids, with channels at one site at
distance 0 and an optional mask (zero taper rows, e.g. solid cells). Gaspari-Cohn uses
chordal distance on periodic axes. The isotropic form is PSD by theory only in at most
3 embedding dimensions; on a 2D torus (4D embedding) it was PSD numerically for the
radii tested.

### 4.3 The Q̂ source

- `q_source="online"` (default): a second unguided forecast per member gives
  r_i = (f_i − f'_i)/√2 each step. It needs no artefact, follows the state, and uses
  the same solver as K_ens. It costs one more prior run per member and step, and Q̂ has
  rank E. It reuses f_i, which also enters the innovation: an O(1/E) bias.
- `q_source="bank"`: `build_residual_bank(posterior, states, num_steps, stepper, ...)`
  returns r = (f¹ − f²)/√2 as a tensor (save it with `torch.save`; `residual_bank`
  takes the tensor or the path). Q̂H' and HQ̂H' are formed from H applied to the bank
  at use time and cached per (H, R). A known factor Q = B'B is the bank √N_b B.
- `q_source="fixed"`: a known diagonal Q (`q_variance`, scalar or per state entry);
  K_q is formed in closed form and not localised.

**Few-step forecasts are approximate prior draws.** In the closed-form case, Euler
FM-ODE at 5 steps gives variance 0.58 Q (0.23 Q at 2 steps), and SI-SDE at 2 steps
gives 1.1 Q. This biases P_f. The online Q̂ inherits the same error, so K_ens and K_q
stay consistent. A bank must be built with the forecast steps used at test time. That
consistency is the reason for the online default.

### 4.4 Calibration against the realised gain

The derivation assumes the sampler applies exactly K_q. The paper's samplers approximate the
likelihood (Jacobian-free, inflated), so their realised gain can differ. If the sampler
already over-applies the gain, the correction over-corrects: in a linear-Gaussian check with
Q_eff = 4Q, plain per-member was already close (RMSE/Kalman 1.07) and the correction made it
worse (1.10-1.13).

Calibrate on validation sequences only: choose `gain_scale` (or a per-variant K_q) so that the
lag-1 autocorrelation of innovations is close to zero, and report it.

### 4.5 Physical constraints

- Navier-Stokes (vorticity): a localised update is not a combination of whole member
  anomalies, so it can shift the domain-mean vorticity; the case removes the domain mean of
  the increment (`zero_mean_increment`, `cases/navier_stokes/_ns_correction.py`). Check
  enstrophy spectra per band (`diagnostics: true`): a global update cut member enstrophy to
  42-50% in a worst-case check, while localisation at 8 sites kept 85-103%.
- uDALES: the state is a horizontal slice of a 3D LES and is not horizontally
  divergence-free (du/dx + dv/dy = -dw/dz), so no divergence projection is applied (an
  increment projection was tried and removed: stronger than the physics requires). The
  obstacle mask is applied (`mask_obstacles`: zero taper in solid cells and a
  `post_update` keeping their guided values).
- Cap the update size in Mahalanobis units as a guard, and log how often the cap binds.

## 5. Test plan

Run in this order. Stop and reassess if a stage fails.

1. **Unit check (no network).** `pytest tests/test_forecast_correction_lg.py`. Exactness in
   the linear-Gaussian case, and the failure of the x'-based innovation.
2. **Lorenz 96 with the trained priors** (`cases/lorenz96/`, see `GPU_HANDOFF.md`).
   d=16, PF reference with N=4e6. Rows: plain SI-SDE / FM-ODE / DM-SDE; the same with
   variant A; with variant B; **unguided prior + localised EnKF** (the alternative in
   section 2); EnKF with the true model; the `oracle` cell (exact per-member kernel) with and
   without correction. Metrics as in the handoff, plus lag-1 innovation autocorrelation.
   Tune localisation radius, `gain_scale` and τ for variant B on validation only.
3. **Measure before fixing on Navier-Stokes and uDALES.** Run with `gain_scale=0
   forecast_seed=0` (grid: `FC=on FC_OVERRIDES="gain_scale=0 forecast_seed=0"`; the
   trajectory equals the plain run's) and read the lag-1 innovation autocorrelation and the
   Q share `(q_rms / spread_rms)^2` from the per-cell diagnostics JSON. If autocorrelation
   is near zero and Q dominates, the correction is not warranted there; say so.
4. **Navier-Stokes and uDALES with the correction** (`FC=on`, settings via `FC_OVERRIDES`;
   wiring in `common/forecast_correction.py`, switches in `cases/navier_stokes/README.md`),
   E=64, on the paper's scenarios, with the spectrum and obstacle checks from 4.5, and the
   alternative in section 2 (not yet wired for these cases).

What would count against the method:

- On Lorenz 96, corrected SI/FM do not match localised EnKF on mean error against the PF.
- Unguided prior + EnKF matches the corrected sampler everywhere. Then the paper should say
  so rather than present the correction as an advantage.
- On Navier-Stokes the plain per-member bias is already at the E=64 sampling floor
  (mean error near 0.125 posterior std), or the correction damages spectra or constraints.

## 6. Evidence so far (preliminary)

All from scratch studies on a Mac. Closed-form transition sampling stands in for a perfect
prior unless marked "trained". Mean error is the RMS error of the ensemble mean in
reference-posterior std units; the E=64 floor is about 0.125. These scripts are not in the
repository.

Linear-Gaussian, periodic 1D field, every 8th site observed, R=0.25, E=64, K=20, 5 seeds,
localised:

| Case | plain | localised EnKF | corrected |
| --- | --- | --- | --- |
| d=1024, A=I | 0.27-0.70 | 0.17-0.26 | 0.13-0.21 |
| d=1024, advection | 0.27-0.30 | 0.17-0.21 | 0.13-0.14 |
| d=32, advection | 0.36-0.59 | 0.20 | 0.16 |

Without localisation at d=1024, the correction applied to the anchors scored 0.168 against
0.610 for EnKF.

An independent check (d=8, advection with damping, every 4th site observed, E=100000,
12 cycles, variant A): corrected mean error at most 0.02 posterior std with variance ratios
0.995-1.007, against 0.90 and 1.20-1.39 for plain.

Lorenz 96, d=16, against the N=4e6 PF:

| Method | mean error | RMSE |
| --- | --- | --- |
| exact per-member, plain | 0.87-1.00 | 3.58-3.73 |
| exact per-member, corrected after the sampler | 0.38-0.47 | 3.01-3.17 |
| corrected before the sampler (variant C) | 0.55-0.61 | 3.11-3.27 |
| localised EnKF, true model | 0.38-0.48 | 3.03-3.20 |
| trained SI-SDE, plain → corrected (1 seed, Jacobian-free, M=25) | 0.66 → 0.41 | 3.31 → 2.94 |
| trained FM-ODE, plain → corrected (1 seed) | 0.63 → 0.41 | 3.30 → 3.00 |

Ranges span the three independent studies, which used different sequence subsets. The
trained corrected ensembles were under-dispersed. "Unguided trained prior + EnKF" was not run.

Non-Gaussian 1D checks (E=64, W1 against an exact grid reference, in posterior-std units):
bimodal transition, EnKF 1.09, plain 0.48, corrected about 0.40, exact weights 0.16;
double well with weak observations, EnKF 0.23, plain 0.60, corrected about 0.19.

## 7. Open questions

1. Does the realised gain of the Jacobian-free and inflated samplers match K_q? (4.4)
2. Is a state-averaged Q̂ good enough, or is an online K_q needed?
3. Variant B: which τ, and how much does network error in μ̂_i matter?
4. How large is the bias on Navier-Stokes and uDALES in the first place? (5, stage 3)
5. Where, if anywhere, does the corrected sampler beat an EnKF on generative forecasts?
