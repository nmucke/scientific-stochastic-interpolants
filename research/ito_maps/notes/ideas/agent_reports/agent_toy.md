# Itô-map toy study: derivation check and three candidate improvements

All code, logs and outputs are in
`research/ito_maps/notes/ideas/agent_reports/toy/`
(`lib.py`, `claims.py`, `train.py`, `run_all.py`, `run_extra.py`, `run_extra2.py`, `ref_check.py`, `report.py`;
outputs `claims_log.txt`, `results/*.json`, `ref_check.json`, `results_table.md`, `fig_metrics_vs_steps.png`, `fig_histograms.png`).
The repo was not modified. 120 training runs, 20k steps, batch 512, 3 seeds each; about 9.5 core-hours, roughly 75 min wall on 9 workers,
which is well over the 40-minute target (runs took 190 s without and 430-500 s with the energy-score term, and I added control runs).

## Short version

- Claims 1-3 are correct. Claim 4 is correct but the effect is small for a one-step map with K=4 (residual std 0.010a); it is larger for the last step of a multi-step sampler.
- **The denoiser-average form (V1) as specified did not help; it was worse than the baseline at every step count** and its quality got worse with more steps.
- **Legendre path features were the one clear win**, but only at physical scale and independent of the map form: baseline form + raw Legendre K=4 cut one-step W1 from 0.126 to 0.039 and one-step pathwise RMSE from 0.44 to 0.22.
- **The energy-score anchor helps the one-step law moderately** (W1 0.126 -> 0.083 on the baseline) but does not fix it while PSD is active, and does nothing for pathwise accuracy. Anchor-only gives the best one-step law and no usable multi-step map.
- Unit-variance normalisation of path features hurts many-step accuracy in both forms.

## A. Derivation check

Symbolic (sympy) and numeric (float64) checks are in `claims.py`, log in `claims_log.txt`.

**Claim 1 - correct.** With E[a sqrt(t) z | I_t=x] = (x - (1-t)c - beta*yhat)/(1-t) the drift is kappa*yhat - x/(1-t), kappa = beta' + beta/(1-t); the c terms cancel. Symbolic residual 0 for beta=t and t^2. Numerically, quadrature of the conditional expectation over y against the closed form: max error 3.9e-13 over 400 random (t, x, c, a).

**Claim 2 - correct, with three qualifications.**
- w(s,t) matches for both schedules (symbolic). Fine EM of the velocity form and of the Z = X/(1-t) form on the same Brownian path: the residual of the identity is 2.3e-2, 5.7e-3, 1.4e-3, 3.6e-4 for N = 512, 2048, 8192, 32768, i.e. clean O(h); the same for the 3-mode target with c=0.5, a=0.7 and for beta=t^2.
- "Convex combination" holds only for beta=t. For beta=t^2 the coefficients sum to 1 - (t-s)(1-t) < 1.
- At t=1 the "path average" is degenerate: int_s^t kappa/(1-r) dr diverges as t->1, so all weight sits at r=1 and Ybar_{s,1} = X_1. Numerically RMS(Ybar_{0,t} - X_1) = 0.235, 0.062, 0.017 for 1-t = 1/4, 1/16, 1/64. So a one-step V1 network predicts the sample itself, and PSD at t=1 has lambda = 0: the target is purely D(u,1,x_u), a consistency-model-type bootstrap.
- G_{0,1} = Ybar_{0,1} - c - M_1 is an identity, but "must cancel white noise" is misleading as a difficulty argument. X_1 is strongly positively correlated with M_1 (the noise picks the mode), so corr(G_{0,1}, M_1) = +0.60 and Var(G_{0,1}) = 1.24 < Var(X_1) = 2.28 on the two-mode target. The baseline's regression target has the smaller variance.

**Claim 3 - correct.** Additivity of A(s,t) = int kappa/(1-r) yhat dr gives the composition; the weights lambda = (1-t)/(1-u) w(s,u)/w(s,t) and w(u,t)/w(s,t) sum to 1 for both schedules (symbolic); numeric residual 1e-15. For beta=t, lambda = (1-t)(u-s)/((1-u)(t-s)). The noise terms compose exactly.

**Claim 4 - correct, quantified.** m_n(1)/a = sqrt(2)/w_n^2 with w_n = (n-1/2)pi, so the captured fraction of Var M_1 = a^2/3 is sum_n 6/w_n^4.

| K | captured fraction | residual variance | residual std |
|---|---|---|---|
| 1 | 0.98553 | 4.82e-3 a^2 | 0.069 a |
| 2 | 0.99770 | 7.66e-4 a^2 | 0.028 a |
| 4 | 0.99969 | 1.04e-4 a^2 | 0.010 a |
| 8 | 0.99996 | 1.33e-5 a^2 | 0.004 a |

Monte Carlo with the repo's `KarhunenLoeveBrownianPath` (n=20000) agrees: residual variance 4.83e-3 (K=1), 1.02e-4 (K=4). For one step with K=4 this is negligible next to a mode width of 0.3. It matters more for multi-step KL-feature sampling: the last step's increment M_1 - M_{1-1/n} is unexplained by xi_1..4 to 15% (n=2, std 0.078a), 29% (n=4, std 0.039a), 80% (n=8, std 0.023a), and nothing after the last step can remove it.

## B. Toy experiments

### Setup and deviations from the brief

- Main setting S0: p(y) = 0.5 N(-1.5, 0.3^2) + 0.5 N(1.5, 0.3^2), c=0, a=1, beta=t. Secondary: 3-mode (weights .5/.3/.2, means -2/0.5/2.5, stds .3/.5/.2), c=0.5, a=0.5, beta=t^2, and a 2D 3-mode mixture.
- MLP 3x128 SiLU, Fourier embeddings of s and t, Adam 1e-3 with cosine decay, EMA 0.99, intervals as in `MapTrainer._intervals`, PSD (and ES) weight ramped linearly over the first 2000 steps. The EMA network is evaluated.
- Brownian functionals are sampled exactly rather than on a fixed grid: 4 Gauss-Legendre nodes with independent Gaussian weights in every gap between queried times reproduce the exact joint law of all increments of W and M and all Legendre moments up to degree 3 (self-test in `claims_log.txt`).
- Evaluation: 50k samples, the same 2048-cell Brownian paths for every run, reference = 2048-step EM with the exact drift. That reference itself has W1 0.014 to the truth (iid 50k sample: 0.004), so W1 below about 0.015 is at the floor.
- "Monotone" = W1 does not increase by more than 0.005 from one step count to the next.
- Variants I added: `V1s` (D = x + (1-s)F, which for beta=t is the baseline velocity parametrisation with the Claim-2 noise split and PSD weights), and feature-scale controls (`-rawfeat`, `-normfeat`, `-legraw`).
- "Bell" diagonal weight = 0.2 + 4.8 t(1-t). ES two-step uses u ~ U(0.05, 0.95).
- Not run: V0 anchor-only; V1+ES1 on the 3-mode, a=0.5 and 2D settings (the 3-mode runs were lost to a killed launcher); c=0.5 has only V0 and V1; K=1,2,4 was compared for the denoiser form only.

### Main table (S0, mean±std over 3 seeds)

W1 to the true p(y):

| variant | 1 step | 2 | 4 | 8 | 16 | monotone |
|---|---|---|---|---|---|---|
| V0 baseline (vel, 4 raw M-increments, PSD) | 0.126±0.008 | 0.030±0.003 | 0.022±0.004 | 0.015±0.006 | 0.014±0.006 | 3/3 |
| V0 + ES (weight 1) | 0.083±0.003 | 0.023±0.001 | 0.024±0.003 | 0.018±0.001 | 0.015±0.005 | 2/3 |
| V0, unit-variance M-increments | 0.118±0.003 | 0.039±0.002 | 0.028±0.004 | 0.024±0.002 | 0.025±0.003 | 3/3 |
| V0, raw Legendre K=4 | 0.039±0.002 | 0.026±0.004 | 0.031±0.008 | 0.028±0.010 | 0.022±0.006 | 1/3 |
| V0, raw Legendre K=4 + ES 1 | 0.030±0.002 | 0.019±0.002 | 0.025±0.002 | 0.018±0.004 | 0.015±0.001 | 1/3 |
| V1 (denoiser, 4 unit W-increments) | 0.179±0.013 | 0.079±0.015 | 0.068±0.023 | 0.073±0.029 | 0.084±0.030 | 0/3 |
| V1, bell diagonal weight | 0.172±0.010 | 0.054±0.009 | 0.034±0.010 | 0.042±0.008 | 0.055±0.014 | 1/3 |
| V1, raw W-increments | 0.266±0.008 | 0.116±0.008 | 0.106±0.011 | 0.108±0.014 | 0.101±0.015 | 2/3 |
| V1 K=1 / K=2 (increments) | 0.543 / 0.367 | 0.134 / 0.095 | 0.064 / 0.056 | 0.067 / 0.064 | 0.079 / 0.077 | 0/3, 1/3 |
| V2 Legendre K=2 (unit) | 0.249±0.016 | 0.079±0.017 | 0.067±0.009 | 0.071±0.008 | 0.071±0.007 | 2/3 |
| V2 Legendre K=4 (unit) | 0.063±0.018 | 0.050±0.014 | 0.045±0.014 | 0.056±0.023 | 0.066±0.041 | 1/3 |
| V2 K=4 + ES 1 | 0.031±0.003 | 0.022±0.004 | 0.020±0.004 | 0.020±0.003 | 0.028±0.012 | 1/3 |
| V3 = V1 + ES 0.1 | 0.148±0.001 | 0.048±0.003 | 0.048±0.002 | 0.073±0.008 | 0.089±0.006 | 0/3 |
| V3 = V1 + ES 1 | 0.108±0.001 | 0.027±0.006 | 0.038±0.004 | 0.060±0.004 | 0.090±0.003 | 0/3 |
| V1 anchor only (diag + ES, no PSD) | 0.024±0.002 | 0.033±0.007 | 0.046±0.011 | 0.076±0.010 | 0.100±0.021 | 0/3 |
| V1s (skip form, unit W-increments) | 0.177±0.006 | 0.046±0.005 | 0.022±0.006 | 0.015±0.004 | 0.016±0.004 | 3/3 |
| V1s, raw W-increments | 0.208±0.007 | 0.045±0.004 | 0.025±0.005 | 0.020±0.006 | 0.015±0.004 | 3/3 |
| V1s, raw Legendre K=4 | 0.042±0.003 | 0.028±0.003 | 0.029±0.005 | 0.027±0.004 | 0.020±0.003 | 2/3 |
| V1s + ES 1 | 0.104±0.002 | 0.026±0.002 | 0.025±0.003 | 0.019±0.004 | 0.018±0.005 | 3/3 |
| EM with exact drift (no learning) | 1.042 | 0.343 | 0.143 | 0.075 | 0.042 | |

Std ratio and pathwise RMSE against the same-path reference (1 / 4 / 16 steps):

| variant | std ratio 1 | 4 | 16 | pathwise RMSE 1 | 4 | 16 | mode flips at 1 step |
|---|---|---|---|---|---|---|---|
| V0 | 0.950 | 0.994 | 0.995 | 0.437 | 0.234 | 0.127 | 5.3% |
| V0 + ES 1 | 0.970 | 1.000 | 0.999 | 0.438 | 0.251 | 0.132 | 5.3% |
| V0 unit-variance features | 0.951 | 0.984 | 0.986 | 0.457 | 0.314 | 0.257 | 5.8% |
| V0 raw Legendre K=4 | 0.985 | 0.990 | 0.994 | 0.223 | 0.166 | 0.091 | 1.7% |
| V0 raw Legendre K=4 + ES 1 | 0.992 | 0.993 | 0.996 | 0.231 | 0.190 | 0.100 | 1.8% |
| V1 | 0.952 | 1.020 | 1.045 | 0.598 | 0.389 | 0.355 | 8.6% |
| V1 raw W-increments | 0.870 | 0.936 | 0.937 | 0.582 | 0.294 | 0.179 | 8.3% |
| V2 Legendre K=4 (unit) | 1.010 | 1.011 | 1.035 | 0.306 | 0.323 | 0.349 | 2.4% |
| V1 + ES 1 | 0.958 | 1.013 | 1.051 | 0.601 | 0.415 | 0.376 | 8.6% |
| V1 anchor only | 0.995 | 0.981 | 0.945 | 1.96 | 2.01 | 1.95 | 43% |
| V1s | 0.927 | 0.992 | 0.998 | 0.585 | 0.346 | 0.269 | 8.6% |
| V1s raw W-increments | 0.908 | 0.987 | 0.993 | 0.573 | 0.280 | 0.137 | 8.2% |
| V1s raw Legendre K=4 | 0.984 | 0.987 | 0.993 | 0.224 | 0.172 | 0.083 | 1.7% |
| EM with exact drift | 0.376 | 0.912 | 0.976 | 1.100 | 0.474 | 0.221 | |

Mode weights are right for every PSD variant (max error 0.001-0.009). The one-step defect is a bridge of mass between the modes, not a wrong mode split: the fraction of samples more than 3 sd from every mode is 0.3% in truth, 9.6% for V0, 11.8% for V1, 7.3% for V0+ES1, 3.2% for the raw-Legendre variants, 3.0% for anchor-only (see `fig_histograms.png`). Full tables for every setting are in `toy/results_table.md`.

### Secondary settings (W1 at 1 / 4 / 16 steps; pathwise RMSE at 1 / 16 steps)

| setting | V0 | V1 | V1s | V0 raw Legendre K=4 |
|---|---|---|---|---|
| 3-mode | 0.122 / 0.036 / 0.023; 0.446 / 0.067 | 0.202 / 0.111 / 0.144; 0.630 / 0.313 | 0.170 / 0.045 / 0.054; 0.612 / 0.176 | 0.053 / 0.031 / 0.024; 0.136 / 0.054 |
| c=0.5 | 0.126 / 0.022 / 0.013; 0.437 / 0.130 | 0.188 / 0.050 / 0.068; 0.596 / 0.346 | - | - |
| a=0.5 | 0.353 / 0.087 / 0.041; 0.883 / 0.275 | 0.384 / 0.073 / 0.034; 0.972 / 0.396 | 0.359 / 0.046 / 0.020; 0.972 / 0.372 | 0.170 / 0.078 / 0.038; 0.480 / 0.269 |
| beta=t^2 | 0.090 / 0.032 / 0.027; 0.353 / 0.131 | 0.182 / 0.082 / 0.084; 0.465 / 0.322 | 0.141 / 0.036 / 0.024; 0.439 / 0.203 | 0.051 / 0.031 / 0.024; 0.273 / 0.117 |
| 2D (sliced W1) | 0.083 / 0.023 / 0.017; 0.468 / 0.137 | 0.120 / 0.065 / 0.090; 0.660 / 0.487 | 0.121 / 0.031 / 0.032; 0.636 / 0.309 | 0.041 / 0.024 / 0.017; 0.312 / 0.121 |

beta=t^2 with V1 + ES 1: W1 0.080 / 0.028 / 0.057, std ratio 0.971 / 1.005 / 1.033.

c is not an independent test for beta=t: X - (1-t)c obeys the c=0 SDE, so the exact problem is a time-dependent shift (the EM numbers for c=0 and c=0.5 are identical), and the learned results match within seed noise.

### What helped

1. **Raw-scale Legendre path features (K=4).** One-step W1 falls 2-3x in every setting (0.126 -> 0.039 in S0, 0.353 -> 0.170 for a=0.5), one-step std ratio rises from 0.95 to 0.985 (0.83 -> 0.92 for a=0.5), one-step pathwise RMSE halves or better, mode flips drop from 5.3% to 1.7%. Many-step quality is unchanged or slightly better pathwise. The gain is the same in the velocity form and the skip form, so it is a feature effect, not a form effect. A likely reason, not tested: with K >= 2 the Legendre span contains both the W and the M increment of the interval, and smooth global kernels describe the path functional better than four piecewise-constant ones.
2. **Energy-score anchor, weight 1.** One-step W1 0.126 -> 0.083 (V0), 0.179 -> 0.108 (V1), 0.039 -> 0.030 (raw Legendre); std ratio 0.95 -> 0.97. Weight 0.1 does little. It costs about 2.3x the training time here.
3. **More path features.** For the denoiser form, one-step W1 is 0.54 / 0.37 / 0.18 for K = 1 / 2 / 4 increments and 0.25 / 0.06 for Legendre K = 2 / 4; Legendre beats increments at equal K at one step.

### What did not help

1. **V1 as specified.** Worse than V0 at every step count in S0, 3-mode, beta=t^2 and 2D, and W1 rises again beyond 4 steps (0/3 seeds monotone) with std ratio drifting above 1. Only for a=0.5 at 4+ steps is it level with or slightly ahead of V0. Two separable causes:
   - Raw denoiser output with a uniform (or bell) loss: X_t gets the error of D times w = (t-s)/(1-s), so late-time errors are amplified, and the last step outputs D directly. The skip parametrisation `V1s` removes the many-step degradation (W1 0.016 at 16 steps).
   - Unit-variance features: they stay O(1) as t-s -> 0 although their effect vanishes, and the diagonal is trained with zero features. The control confirms this in both forms: 16-step pathwise RMSE 0.127 -> 0.257 for V0 when its features are normalised, 0.269 -> 0.137 for V1s and 0.355 -> 0.179 for V1 when theirs are not.
   - With both fixed (`V1s` raw features), the Claim-2 form matches V0 at many steps and is still worse at one step (W1 0.208 vs 0.126) with W-increments. With Legendre features the two forms are indistinguishable. So the removal of the explicit M_1 term bought nothing measurable.
2. **The anchor does not repair pathwise accuracy and does not fully repair the law.** Pathwise RMSE and flip rates are unchanged by ES. With PSD on, the one-step bridge between modes shrinks but stays (7.3% vs 0.3% true), because the PSD regression still pulls towards the conditional mean of the two-step output. On V1 the anchor makes 8-16 step sampling slightly worse.
3. **Anchor only.** Best one-step law (W1 0.024, std ratio 0.995) but the map is unrelated to the SDE solution (pathwise RMSE about 2, 43% of samples in the other mode than the same-path reference) and quality falls steadily with more steps.
4. **Unit-normalised Legendre features (V2 as specified).** Good at one step, but pathwise error does not improve with steps (0.31 -> 0.35) and seed variance is large; the raw-scale version is better and stable.

### Monotonicity

V0, V1s and the raw-feature variants are monotone in W1 up to the reference floor; the "1/3" entries for the raw-Legendre runs are wiggles of 0.005-0.01 around W1 0.02-0.03, since those start much closer to the floor. V1, V2 and everything built on them are genuinely non-monotone. Pathwise RMSE of raw-Legendre variants is flat or slightly worse from 1 to 2 steps (0.223 -> 0.229; 0.136 -> 0.140 for 3-mode) before falling.

### Caveats

- 1D (plus one 2D mixture), small MLPs, well-separated modes. The denoiser here is trivially learnable; nothing is said about high-dimensional fields.
- 3 seeds; differences below about 0.01 in W1 are within seed noise and the reference floor.
- Each V variant bundles several choices. The controls separate form, feature type and feature scale on S0 only; the secondary settings have the baseline, V1, V1s and raw-Legendre only.
- K was varied for the denoiser form only; the baseline was not run with K = 1, 2 or with more than 4 Legendre moments.
- Pathwise error is measured against a 2048-step EM reference and is dominated by mode flips on this target (a flip costs a distance of 3).
- The explanations for the feature-scale and Legendre effects are hypotheses consistent with the controls, not established mechanisms.
