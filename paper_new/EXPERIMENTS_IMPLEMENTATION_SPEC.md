# Experiments Implementation Spec — Observation-Interpolant Data Assimilation

This document specifies the experiments for the paper *"A General Observation-Interpolant
Method for Data Assimilation with Flow-Based Generative Models."* It is written for an
engineer/agent who has the codebase but should not need to re-derive anything from the
paper. Every number and figure produced here maps to a specific placeholder in
`sections/results.tex`; the mapping is given in Section 8.

Three test cases:
1. **Analytical linear–Gaussian** (closed-form posterior, no training) — correctness probe.
2. **Stochastic incompressible Navier–Stokes** (learned prior, high-dim, chaotic) — main benchmark.
3. **Urban airflow CFD over a building array** (multi-variable, applied) — realism.

> **Convention.** Physical time steps use superscripts (`x^n`); generative pseudo-time uses
> subscripts (`x_tau`), `tau in [0,1]`. `E` = ensemble size, `M` = number of pseudo-time
> integration steps, `NFE` = number of network evaluations per assimilation step.

---

## 1. Methods to implement

### 1.1 Our three samplers (one shared loop)
All three are members of the unified family (paper Eq. `unified_posterior_drift`), differing
only in the diffusion `g_tau`, the guidance weight `w_tau = a_tau + 0.5*g_tau^2`, the source,
and whether a Brownian increment is added. Implement the shared loop (paper Algorithm 1) once
and parameterise it.

Per pseudo-time step at state `x` (drop the `n` superscript; `x0 = x^{n-1}`, `y = y^n`):

```
# given trained model -> velocity v_tau(x) and/or score s_tau(x); schedules alpha,beta,sigma
b_prior   = v_tau(x) + 0.5 * g_tau**2 * s_tau(x)         # prior drift  (= SI drift for SI)
ybar_tau  = alpha_tau * H @ a0 + beta_tau * y            # observation interpolant (Eq. interpolated_observation)
mu_s, Sigma_s = source_moments(x, x0, tau)               # Lemma "Source conditional moments"
mu_bar    = H @ x - H @ mu_s                             # Eq. (bias);   = H E[x1|x_tau] pushed through H
Sigma_bar = beta_tau**2 * R + H @ Sigma_s @ H.T          # Eq. (cov_correction)
Sbar      = grad_xbar_mu.T @ inv(Sigma_bar) @ (ybar_tau - mu_bar)   # interpolant-likelihood score
G_tau     = I + (1/beta_tau**2) * Sigma_s @ H.T @ inv(R) @ H        # multiplicative gain (Thm)
drift     = b_prior + w_tau * (G_tau @ Sbar)             # Eq. posterior_drift_multiplicative
x        += drift * dtau + (g_tau * sqrt(dtau) * randn() if g_tau>0 else 0)
```

- **`grad_xbar_mu`**: exact value is `H @ Sigma_s / sigma_tau^2`; the **Jacobian-free**
  choice (Corollary `cheap_drift`) uses `grad_xbar_mu ≈ H` and `Sigma_s ≈ sigma_tau^2 I`,
  giving `G_tau ≈ I + (rho_tau/beta_tau^2) H^T R^{-1} H`, `rho_tau = gamma^2 tau` (SI) or
  `alpha^2` (FM). Precompute `H^T R^{-1} H` once when `H,R` are time-independent.
- **Singularity at `tau=0`** (`beta_0=0`): start the guidance at `tau = dtau`, not `0`.
- `a_tau` (velocity–score coef.), the score-recovery formulas, and the source moments differ
  by model class — see the table below.

| Quantity | SI (Sampler 1) | FM (Samplers 2 & 3) |
|---|---|---|
| Interpolant | `alpha*x0 + beta*x1 + gamma*W_tau` | `alpha*z + beta*x1`, `z~N(0,I)` |
| Source process `s_tau` | `gamma_tau * W_tau` (`sigma=gamma*sqrt(tau)`) | `alpha_tau * z` (`sigma=alpha`) |
| Learned object | SDE drift `b_theta = E[R_tau\|x_tau]` | velocity `v_theta` |
| Score from model | `A_tau[beta*b_theta - c_tau]` (Eq. SI_score) | `(beta*v_theta - bdot*x)/(alpha(bdot*alpha - adot*beta))` (Eq. fm_score) |
| `a_tau` | `sigma(bdot*sigma - sdot*beta)/beta`, `sigma=gamma*sqrt(tau)` | `alpha(bdot*alpha - adot*beta)/beta` |
| `mu_s` | `-gamma^2 tau A_tau(beta*b_theta - c_tau)` | `-alpha^2 * s_tau` |
| `Sigma_s` | `gamma^2 tau I + gamma^4 tau^2 A_tau(beta J_b - bdot I)` | `alpha^2 I + alpha^4 grad s_tau` |

with `A_tau = [tau*gamma*(bdot*gamma - beta*gdot)]^{-1}`,
`c_tau = bdot*x + (beta*adot - bdot*alpha)*x0`, `J_b = grad_x b_theta`.

**Sampler specifics**
- **SI-SDE** (Alg. 2): `a0 = x0`, `g_tau = gamma_tau` (native). Init `x = x0` (point mass;
  no reweighting). Brownian increment retained. Score already in `b_theta`.
- **FM-SDE** (Alg. 3): `a0 = 0`, `x0` as conditioning input. `g_tau > 0` free; **use an
  endpoint-vanishing schedule `g_tau ∝ sqrt(alpha_tau*beta_tau)`** — this is required to keep
  `0.5*g_tau^2*s_tau` finite as `tau→1` (the recovered FM score diverges there). Init
  `x ~ N(0,I)` — **this is exact, no importance reweighting needed** (`Phi_0^obs` is constant
  because at `tau=0` the latent is independent of `x1`). Recover score via Eq. `fm_score`.
  Brownian increment retained.
- **FM-ODE** (Alg. 4): `a0 = 0`, `g_tau = 0`, `w_tau = a_tau`. Init `x ~ N(0,I)` (exact). No
  noise term; use a deterministic solver (Heun/RK4 acceptable since dynamics are smooth).
  Randomness enters only through the initial latent.

> **Two FM correctness checks to assert in code:** (a) `Phi_0^obs` independent of the latent
> ⇒ FM init needs no reweighting; (b) with `g_tau ∝ sqrt(alpha*beta)`, the FM-SDE drift stays
> bounded as `tau→1`.

### 1.2 Baselines
Share the **same trained prior** for the generative baselines so comparisons isolate the
assimilation mechanism.

1. **FlowDAS** (`chen_flowdas_2025`): SI-SDE with the Monte-Carlo likelihood estimate (draw
   `J` samples of `x1 ~ p(x1|x_tau,x0)` by forward-integrating the interpolant SDE, softmax
   weight by `p(y|x1)`). Use the authors' `J` (report it).
2. **Guided FM / FIG** (`yan_fig_2024`): guided probability-flow ODE with the measurement
   interpolant (coincides with our observation interpolant for `a0=x0`). Velocity correction
   scaled by the path's velocity–score coefficient.
3. **Guided diffusion / DPS** (`chung_diffusion_2023`): reverse-SDE with the DPS likelihood
   `N(y; H E[x0|x_t], R)` and gradient through the denoiser. Use a diffusion prior of matched
   capacity, or reuse our score with a VP/VE schedule — state which.
4. **EnKF / ESMDA** (`evensen_data_2022`): stochastic EnKF for the field cases; same ensemble
   size `E`. Localisation/inflation tuned and reported.
5. **Bootstrap particle filter** (`carrassi_data_2018`): same `E`; report effective sample
   size and resampling scheme.
6. **SDA** (`rozet_score-based_2023`): score-based DA with the all-at-once trajectory score.
7. **Ensemble score filter** (`bao_ensemble_2024`).

> Classical baselines (EnKF, PF) need the forward model for propagation. For the analytical
> case the forward model is known; for NS/urban, propagate with the numerical solver (EnKF/PF
> are allowed the true solver — note this in the paper as an advantage they have over the
> generative methods, which use a learned prior).

### 1.3 Shared sampler configuration
Hold these fixed across methods unless a metric varies them:
- Ensemble size `E` (e.g. 64; sweep in ablation).
- Integration steps `M` (e.g. 50; sweep in ablation).
- Schedules `alpha_tau, beta_tau, gamma_tau`: state the exact functional forms used at
  training (e.g. rectified-flow `alpha=1-tau, beta=tau` for FM; the SI schedule from the
  trained model). The samplers **must** use the same schedules as training.
- Observation noise covariance `R` (per case below).
- Seeds: fix a list of seeds; report mean ± std over seeds.

---

## 2. Observation operators (field cases)

Both linear (matching paper Section `obs_interpolation` assumptions).

- **Super-resolution `Lr^2 → Hr^2`**: `H` = average-pool / strided downsample from the model
  grid `Hr^2` to `Lr^2` (define exactly: e.g. `Lr=32, Hr=128` ⇒ `4x4` block average). The
  observation is the low-res field; `N_y = Lr^2` (× n_vars). Scenarios: `32^2→128^2`, `16^2→128^2`.
- **Sparse sensors (fraction `f`)**: `H` selects `round(f * Hr^2)` grid points. Scenarios:
  `f = 5%` and `f = 1.5625%` (= 1/64). For the urban case place sparse sensors at physically
  plausible locations (street level, façades); for NS use a fixed random mask (fixed seed,
  shared across methods).
- **Noise**: additive Gaussian, `R = sigma^2 I`. NS: `sigma = 0.05` (vorticity). Urban:
  per-variable `sigma` (TODO — set with the user; suggest 1–5% of each variable's std).

---

## 3. Metrics (exact definitions)

Let ground truth `x* ∈ R^d`, ensemble `{x_e}_{e=1}^E`, ensemble mean `xbar = mean_e x_e`.
Average every metric over assimilation steps and test trajectories; report mean ± std over seeds.

**(a) Point accuracy**
- **RMSE** (ensemble mean): `sqrt( mean_i (xbar_i - x*_i)^2 )`. Per variable for the urban case.
- **Energy-spectrum RMSE**: radially-averaged power spectrum `Ek` of `xbar` vs `x*`:
  `sqrt( mean_k (log Ek[xbar] - log Ek[x*])^2 )` (log to weight scales evenly; state if linear).

**(b) Probabilistic calibration**
- **CRPS** per grid point, averaged spatially; use the ensemble CRPS estimator
  `CRPS = mean_e |x_e - x*| - 0.5 * mean_{e,e'} |x_e - x_{e'}|`.
- **Spread–skill ratio**: `mean_i ensStd_i / RMSE`, where `ensStd_i = sqrt(var_e x_{e,i})`.
  Report `|1 - spread/skill|` in tables (0 = perfectly calibrated). Apply the
  `sqrt((E+1)/E)` finite-ensemble correction.
- **Rank (Talagrand) histogram**: per point, rank of `x*_i` among `{x_{e,i}}`; aggregate. Flat
  = calibrated, U = under-dispersed, ∩ = over-dispersed. (Figure only.)

**(c) Distributional fidelity**
- **KL at points**: at a fixed set of observed and unobserved grid points, fit 1-D Gaussians
  (or KDE) to the sampled marginal and compute `KL(sampled || reference)`. Reference =
  analytical posterior (Case 1) or a large-`E` reference ensemble / long MCMC (Cases 2–3).
  Report observed and unobserved points separately if possible.
- **Sliced-Wasserstein `W2`** (Case 1, full joint): average `W2` over random 1-D projections
  between sampled set and exact-posterior samples.

**(d) Cost**
- **NFE**: network evaluations per assimilation step (count guidance-gradient evals too).
- **Wall-clock**: seconds per assimilation step at matched `E`, fixed hardware (log GPU/CPU).

---

## 4. Case 1 — Analytical linear–Gaussian

**Setup** (paper Appendix `simple_test_case`):
```
x^1 = x^0 + w,   w ~ N(0, I)
y^1 = H x^1 + e, e ~ N(0, R),  H = I, R = I
```
**Exact posterior**: `N(x^0 + K(y^1 - H x^0), I - K H)`, `K = H^T(H H^T + R)^{-1} = 0.5 I`.

**Closed-form SI drift** (no training; Prop. B.9 of `chen_probabilistic_2024`, specialised to
target mean `m1(x0)=x0`, `Cov1=I`):
```
b(x,x0,tau) = adot*x0 + bdot*m1(x0) + (beta*bdot + tau*gamma*gdot) * Cbar_tau^{-1} (x - mbar_tau)
mbar_tau    = alpha*x0 + beta*m1(x0)
Cbar_tau    = (beta^2 + tau*gamma^2) I
```
(Note the second term uses `m1(x0)=x0`, not `mbar_tau`.) For FM, derive the analytic velocity
for the Gaussian target similarly (or set `alpha=1-tau, beta=tau` and use the known Gaussian
flow). Implement all three samplers and the generative baselines on this analytic prior.

**Dimensionality**: use `d=2` for the density plots and `d∈{2,10,100}` for the convergence
study (KL still tractable since everything is Gaussian — compare sample mean/cov to exact).

**What to produce**
- **Fig. `analytical_panels`**: (a) prior conditional, (b) likelihood, (c) exact posterior,
  (d) sampled posterior (one sampler), (e) KL vs diffusion strength `g_tau` (SDE samplers),
  (f) KL vs steps `M` (all samplers), (g) 1-D density slices sampled vs exact. (2-D case.)
- **Table `analytical_results`**: KL and sliced-`W2` to exact posterior for SI-SDE, FM-SDE,
  FM-ODE, FlowDAS, Guided FM, Guided diffusion, EnKF, particle filter. Matched `E`, `M`.

**Pass criteria** (sanity, report in text): all samplers → exact mean/cov as `M→∞`;
multiplicative correction reduces KL vs `G=I`; FM-ODE matches the SDEs at convergence.

---

## 5. Case 2 — Stochastic incompressible Navier–Stokes

**PDE** (vorticity form on torus `[0,2π]^2`):
```
dω + (u·∇)ω dt = ν Δω dt − α ω dt + ε dξ
u = ∇⊥ψ = (−∂_yψ, ∂_xψ),   −Δψ = ω
```
`dξ` = temporally white forcing on selected Fourier modes. **TODO (confirm with user / codebase):**
`ν` (viscosity), `α` (linear drag), `ε` (forcing amplitude), forced wavenumber band, grid
`128^2`, time step `dt`, physical assimilation interval `Δt` between observations, trajectory
length `N`, number of test trajectories.

**Data generation**: spin up to statistical stationarity, discard transient; generate train /
val / test splits of trajectories with **independent forcing realisations**. The stochastic
forcing is essential — the prior `p(ω^n|ω^{n-1})` is a genuine distribution.

**Prior model**: train the SI model (SI-SDE) and the FM model (FM-SDE, FM-ODE) to sample
`p(ω^n|ω^{n-1})` (paper Eqs. `SI_drift_loss`, `fm_loss`). **Use the same architecture, data,
and schedules for both** so sampler comparisons are fair. Record training config.

> **Trained weights (GPU machine).** Both priors are already trained and live at
> `checkpoints/stochastic_navier_stokes/stochastic_interpolant_small/` (SI) and
> `checkpoints/stochastic_navier_stokes/flow_matching/` (FM), each holding a `model.pth` +
> `config.yaml`. The `paper_experiments` NS driver loads them via the
> `checkpoints.si_run` / `checkpoints.fm_run` keys in `configs/case/navier_stokes.yaml`
> (see `paper_experiments/HANDOFF_GPU.md`). The laptop repo has no `model.pth`, so runs there
> use random weights (smoke-scale only).

**Observation scenarios**: `32^2→128^2`, `16^2→128^2` (super-res); `5%`, `1.5625%` (sparse).
`R = 0.05^2 I`.

**Assimilation protocol**: start each test trajectory from the same prior state; assimilate
`y^1..y^N` autoregressively (feed each posterior sample back as `x^{n-1}`); ensemble size `E`,
steps `M`. Run all methods on identical truth + observation sequences (shared seeds/masks).

**What to produce**
- **Fig. `ns_trajectories`**: truth / prior forecast / posterior (mean ± spread) snapshots,
  one scenario.
- **Fig. `ns_diagnostics`**: (a) RMSE vs assimilation step, (b) energy spectra truth/prior/
  posterior, (c) rank histogram.
- **Table `ns_accuracy`**: vorticity RMSE, energy-spec RMSE, KL at points × {`32^2→128^2`,`5%`}.
- **Table `ns_calibration_cost`**: CRPS, spread–skill, NFE, s/step × same scenarios.
- Put `16^2→128^2` and `1.5625%` columns in an appendix table (same format).
- Methods: our three + FlowDAS, Guided FM, Guided diffusion, SDA, ensemble score filter,
  EnKF, particle filter.

---

## 6. Case 3 — Urban airflow CFD over a building array

**Physics**: incompressible airflow + heat transport over an array of buildings (bluff bodies)
on a structured grid. Coupled fields: **velocity** (components `u, v`, or speed) and
**temperature** `T`. Boundary layers, wakes, recirculation behind obstacles; anisotropic,
spatially heterogeneous statistics. **TODO (specify with user / from dataset):**
- Domain size and grid (target `128^2` to match NS pipeline, or state actual).
- Building layout / mask (footprints), boundary conditions (inflow profile, wall thermal BCs).
- Solver and turbulence treatment (LES or RANS), `Re`, `Pr`, buoyancy/Boussinesq if used.
- Source of stochasticity (inflow perturbations, thermal forcing) — needed for a non-trivial prior.
- Variables actually modelled and their normalisation; per-variable noise `sigma`.
- Train/val/test trajectory counts; assimilation interval.

**Prior model**: same generative setup as NS, but multi-channel (velocity + temperature).
Condition on the previous multi-variable state. Mask out / handle solid cells consistently in
the model, the observation operator, and all metrics (exclude solid cells from RMSE/CRPS).

**Observation scenarios**: super-res `32^2→128^2` and sparse `5%`; sparse sensors at
physically plausible locations (street-level + façade). Observe velocity and temperature
(state whether jointly or separately).

**What to produce**
- **Fig. `urban_fields`**: geometry + truth/prior/posterior for velocity and temperature; mark
  building footprints and sensor locations.
- **Table `urban_accuracy`**: velocity RMSE, temperature RMSE, KL at points × {`32^2→128^2`,`5%`}.
- **Table `urban_calibration_cost`**: CRPS, spread–skill, NFE, s/step.
- Same method list as NS. Pay attention to reconstruction quality in **unobserved wake regions**
  (report KL at unobserved points there).

---

## 7. Ablations (Navier–Stokes, one scenario)

Fill **Table `ablation`** (RMSE, CRPS, spread–skill):
1. **Multiplicative correction**: full `G_tau` vs Jacobian-free `G_tau` (Cor. `cheap_drift`)
   vs none (`G_tau = I`). Tests the value of the second-order curvature term (paper Section
   `multiplicative_correction`); expect full ≥ Jacobian-free ≥ none, and quantify the gap.
2. **Diffusion strength `g_tau`** (FM-SDE): low / medium / high; include `g=0` (= FM-ODE).
   Maps the accuracy↔calibration trade-off.
3. **Steps `M`**: e.g. 10 / 50 / 100.
4. **Ensemble `E`**: e.g. 16 / 64 / 256.

Optional extra: DPS-style raw-`R` surrogate vs the ΠGDM-style inflated covariance
`R + H Cov(x1|x_tau) H^T` (available for free since `Cov(x1|x_tau) = Sigma_s/beta^2`) — this
directly tests the framing point raised in the paper's Section `multiplicative_correction`.

---

## 8. Output format and mapping to the LaTeX

Emit one tidy results file (CSV or JSON) per case with columns:
`case, method, scenario, metric, value, std, E, M, seed, NFE, seconds`. A small script should
turn this into the LaTeX table cells. Mapping of deliverables to `sections/results.tex`:

| Result | LaTeX label | File slot |
|---|---|---|
| Analytical panels | `fig:analytical_panels` | replace `\figbox{...}` with `\includegraphics` |
| Analytical distances | `tab:analytical_results` | fill rows |
| NS truth/prior/posterior | `fig:ns_trajectories` | figures |
| NS diagnostics | `fig:ns_diagnostics` | figures |
| NS accuracy/distributional | `tab:ns_accuracy` | fill cells |
| NS calibration/cost | `tab:ns_calibration_cost` | fill cells |
| Urban fields | `fig:urban_fields` | figures |
| Urban accuracy | `tab:urban_accuracy` | fill cells |
| Urban calibration/cost | `tab:urban_calibration_cost` | fill cells |
| Ablations | `tab:ablation` | fill cells |

To insert a figure: replace `\figbox{<w>}{<h>}{<text>}` with
`\includegraphics[width=<w>]{figures/results/<path>}` and delete the `\figbox` macro
definition at the top of `results.tex` once all are replaced. Every slot needing data is
marked `TODO` in `results.tex`.

---

## 9. Reproducibility checklist
- [ ] Fixed seed list; mean ± std over seeds in every table.
- [ ] Identical truth + observation sequences + sensor masks across all methods per scenario.
- [ ] SI and FM priors share architecture, data, schedules; training configs logged.
- [ ] Generative baselines reuse the shared prior; classical baselines use the true solver (noted).
- [ ] Solid cells excluded from all urban metrics.
- [ ] NFE and wall-clock logged on fixed hardware at matched `E`.
- [ ] Assert: FM init needs no `Phi_0^obs` reweighting; FM-SDE drift bounded at `tau→1` with `g∝sqrt(alpha*beta)`.
- [ ] Reference posteriors for KL: analytic (Case 1), large-`E`/MCMC (Cases 2–3).
