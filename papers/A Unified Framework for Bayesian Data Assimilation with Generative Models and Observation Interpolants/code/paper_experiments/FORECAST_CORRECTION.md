# Forecast-residual correction for sequential assimilation

This note describes a correction for the bias in the paper's sequential (multi-step)
assimilation, how to implement it in this code base, and how to test it. It assumes no
access to the session in which it was written.

Status: **designed and checked in a linear-Gaussian model only. Not implemented in the
library.** The numbers in section 6 come from scratch studies with closed-form transition
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

## 4. Implementation outline

### 4.1 Where it goes

`BasePosterior.sample_trajectory` in `src/scisi/posterior_models/base_posterior.py` loops
over physical steps. At an assimilation step it calls `self.sample(...)`, which returns
`(base, field_history)`. `base` is the new state `[E, C, H, W]` (and is the start of the
next step when `gaussian_base` is false); the last slot of `field_history` holds the same
state. The free-running branch already shows how to draw an unguided forecast:
`self.model.sample(...)` with the same `field_history`, `num_steps` and stepper.

Add an optional argument, defaulting to `None`, so existing behaviour stays bit-for-bit:

```python
def sample_trajectory(..., analysis_correction: ForecastCorrection | None = None):
    ...
    if _is_assim_step(phys_idx):
        prev_history = field_history                      # state x^k for every member
        base, field_history = self.sample(...)            # guided x'_i
        if analysis_correction is not None:
            with torch.no_grad():
                f, _ = self.model.sample(                 # unguided f_i, fresh noise
                    base=None if self.gaussian_base else prev_base,
                    field_history=prev_history, num_steps=num_steps,
                    batch_size=batch_size, return_field_history=True,
                    stepper=stepper, field_cond=..., pars_cond=..., **free_run_extra)
            base = analysis_correction(x=base, f=f, y=cond["observations"])
            field_history = torch.cat([field_history[..., :-1], base.unsqueeze(-1)], -1)
```

Keep the noise of the unguided run independent of the guided one: draw it after the guided
sample, or from a separate generator.

### 4.2 The correction module

New file `src/scisi/posterior_models/forecast_correction.py`:

```python
class ForecastCorrection:
    def __init__(self, obs_operator, obs_noise_var, q_bank, localisation, gain_scale=1.0): ...
    def __call__(self, x, f, y):
        # x, f: [E, C, H, W] in model (standardised) units; y: [E, m], one row per member
        Hf  = H(f)                                  # [E, m]
        Af  = f - f.mean(0);  AHf = Hf - Hf.mean(0)
        PHt = loc_xy * (Af.T @ AHf) / (E - 1)       # [d, m], localised P_f H'
        HPH = loc_yy * (AHf.T @ AHf) / (E - 1)      # [m, m]
        K_ens = PHt @ inv(HPH + R)
        K_q = QHt @ inv(HQH + R)                    # from the bank, see 4.3; cache per H, R
        e = sqrt(R) * randn(E, m)
        return x + gain_scale * (y + e - Hf) @ (K_ens - K_q).T
```

Notes:

- Work in the model's standardised units, and scale R accordingly, as the Lorenz 96 case
  does (`_l96_generative.Standardisation`).
- Never form d×d matrices. Use d×m and m×m: the observed dimension m is small for sparse
  observations. For the super-resolution operators m is larger (32² = 1024 for 32²→128²);
  then use an observation-space or local (LETKF-style) solve.
- `LinearObservationOperator.forward` gives Hx as `[B, num_obs]`. `obs_indices_on_grid`
  gives sensor locations for localisation. For block-average operators, use block centres.
- Localisation: Gaspari-Cohn (`src/scisi/particle_filter/enkf.py`, `gaspari_cohn`) on the
  distance between grid points and sensors for `loc_xy`, and between sensors for `loc_yy`.
  Tune the radius on validation data only.
- `gain_scale` defaults to 1. It exists for the calibration in 4.4.

### 4.3 The Q̂ bank

K_q needs Q̂ H' and H Q̂ H', which depend on H. To keep H free at test time, store residual
samples rather than a matrix:

1. From validation (or training) states x, draw two unguided forecasts f¹, f² from the same
   x. Then r = (f¹ − f²)/√2 has covariance Q(x).
2. Store N such residual fields (N of a few hundred to a few thousand), in model units.
3. At test time compute Q̂ H' = R_bank' (H R_bank) / N and H Q̂ H' = (H R_bank)' (H R_bank) / N,
   localised as above, and cache them per (H, R).

Caveat: Q is state-dependent for nonlinear dynamics, and the bank gives an average. An
alternative is to estimate K_q online from a second unguided forecast per member, at one
more prior run.

### 4.4 Calibration against the realised gain

The derivation assumes the sampler applies exactly K_q. The paper's samplers approximate the
likelihood (Jacobian-free, inflated), so their realised gain can differ. If the sampler
already over-applies the gain, the correction over-corrects: in a linear-Gaussian check with
Q_eff = 4Q, plain per-member was already close (RMSE/Kalman 1.07) and the correction made it
worse (1.10-1.13).

Calibrate on validation sequences only: choose `gain_scale` (or a per-variant K_q) so that the
lag-1 autocorrelation of innovations is close to zero, and report it.

### 4.5 Physical constraints

- Navier-Stokes (vorticity): localised updates are linear combinations of member anomalies
  and keep the zero-mean constraint. Check enstrophy spectra per band: a global update cut
  member enstrophy to 42-50% in a worst-case check, while localisation at 8 sites kept
  85-103%.
- uDALES: localised updates break the divergence-free condition and can put mass into
  obstacles. Apply the update to a potential or stream function, or project after it, and
  reapply the obstacle mask.
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
3. **Measure before fixing on Navier-Stokes and uDALES.** On existing runs, compute lag-1
   innovation autocorrelation and the share of P_f that is Q (from the bank against the
   forecast spread). If autocorrelation is near zero and Q dominates, the correction is not
   warranted there; say so.
4. **Navier-Stokes and uDALES with the correction**, E=64, on the paper's scenarios, with the
   spectrum, divergence and obstacle checks from 4.5, and the alternative in section 2.

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
