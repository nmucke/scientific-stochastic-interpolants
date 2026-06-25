# Case 1 — Analytical linear–Gaussian

Correctness probe (spec Section 4). Closed-form posterior, **no network trained**,
so it isolates the posterior-sampling machinery (observation-interpolant
likelihood + multiplicative correction) from model error.

## System

```
x^1 = x^0 + w,   w ~ N(0, I)
y^1 = H x^1 + e, e ~ N(0, R),   H = I, R = I
```

Exact posterior `N(x^0 + K(y^1 - H x^0), I - K H)` with gain `K = 0.5 I`. The SI
drift is known in closed form (Prop. B.9 of `chen_probabilistic_2024`); for FM use
the analytic Gaussian-target velocity (`alpha=1-tau, beta=tau`).

Dimensionality: `d=2` for the density plots; `d ∈ {2, 10, 100}` for the
convergence study (KL stays tractable — compare sample mean/cov to exact).

## Deliverables

- **`tab:analytical_results`** — KL and sliced-`W2` to the exact posterior for
  SI-SDE, FM-SDE, FM-ODE, FlowDAS, Guided FM, Guided diffusion, EnKF, particle
  filter, at matched `E`, `M`. Emitted by `make_tables.py` →
  `generated/tab_analytical_results.tex`.
- **`fig:analytical_panels`** — (a) prior conditional, (b) likelihood, (c) exact
  posterior, (d) sampled posterior, (e) KL vs diffusion strength `g_tau`,
  (f) KL vs steps `M`, (g) 1-D density slices. *(Figure code: separate TODO.)*

## Pass criteria (report in text)

All samplers → exact mean/cov as `M → ∞`; the multiplicative correction reduces
KL vs `G = I`; FM-ODE matches the SDEs at convergence.

## Tidy rows this case emits

`case=analytical`, `scenario=analytical`, `metric ∈ {kl_points, sliced_w2}`,
one row per (method, seed) before aggregation.

## Implementation (GAP E4 — done)

- `samplers.py` — compact closed-form vector SDE/ODE integrators on the analytic
  Gaussian prior for the three samplers (SI-SDE, FM-SDE, FM-ODE), the three
  config-selectable likelihood modes (`inflated` / `dps_full` /
  `dps_jacobian_free`), and the baselines (FlowDAS Monte-Carlo likelihood, a
  stochastic EnKF, and a bootstrap particle filter — all exact here since the
  forward model is known). Guidance weights and scores are derived in closed form
  and verified against the exact posterior (mean 3, cov 0.5 for `x0=5, y=1`).
- `driver.py::AnalyticalRunner.evaluate` — runs each method, computes Gaussian KL
  and sliced-`W2` to the exact posterior, emits the tidy rows. Truth + obs are
  seeded identically across methods.
- `figures.py` — the 7 panels of `fig:analytical_panels` →
  `figures/results/analytical/an_{prior,like,true,sampled,kl_diff,kl_steps,slices}`.
- `convergence_study.py` — the d ∈ {2,10,100} × mode ablation (`inflated`
  converges; `dps_full` / `dps_jacobian_free` plateau) →
  `figures/results/analytical/an_dim_convergence`.

KL / sliced-`W2` estimators are reused from
`paper/scripts/analytical_utils/kl_divergence.py`.

## How to run

```
python paper_experiments/run.py case=analytical                 # tidy results
python -m cases.analytical.figures                               # fig panels
python -m cases.analytical.convergence_study                     # dim ablation
python paper_experiments/make_tables.py \
    --results paper_experiments/results/analytical_results.csv \
    --out paper_experiments/generated                            # LaTeX table
```
