# Case 1 — Analytical linear–Gaussian

**Status (2026-06-29): DONE — real numbers, all 11 methods, three figures generated.**

Correctness probe (spec Section 4). Closed-form posterior, **no network trained**,
so it isolates the posterior-sampling machinery (the observation-interpolant
likelihood) from model error. The headline message: with **faithful** baseline
implementations essentially every method recovers the linear-Gaussian posterior
(this case is an exactness check, not a place where the baselines are strawmen);
methods separate on the nonlinear fluid cases.

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
  the full lineup at matched `E`, `M`. Emitted by `make_tables.py` →
  `generated/tab_analytical_results.tex`. **Filled with real numbers** (mean over 5
  seeds, KL to exact):

  | method | KL→exact |
  |---|---|
  | Ours SI-SDE / DM-SDE / FM-ODE | 0.0009 / 0.0016 / 0.0011 |
  | FlowDAS | 0.080 |
  | Guided FM (OT-ODE) | 0.0021 |
  | D-Flow SGLD | 0.079 |
  | SDA | 0.019 |
  | SURGE | 0.0021 |
  | EnKF (E=1000 ref) | 0.0012 |
  | Particle filter | 0.0030 |
  | Guided FM (FIG) | collapsed (KL degenerate; sliced-W2 0.733) |

  FIG is faithfully implemented (matches official riccizz/FIG) but is structurally
  mismatched to full noisy observation — its corrector targets the measurement
  interpolant `y_t = t·y` and concentrates the ensemble onto the measurement
  (covariance→0), so its KL is degenerate. Reported as "collapsed".
- **Three figures** under `manuscript/figures/analytical/`:
  - `analytical_case.pdf` — 2D prior / likelihood / posterior contours.
  - `analytical_kl_vs_steps.pdf` — KL vs `M ∈ {20,50,100,250,500}`, all methods,
    table-consistent protocol.
  - `analytical_covariance_ablation.pdf` (appendix) — our 3 samplers ×
    {per-member exact, shared, isotropic Jacobian-free}; per-member ≡ shared in
    the linear case, Jacobian-free plateaus.

## Pass criteria (report in text)

All samplers → exact mean/cov as `M → ∞`; the multiplicative correction reduces
KL vs `G = I`; FM-ODE matches the SDEs at convergence.

## Tidy rows this case emits

`case=analytical`, `scenario=analytical`, `metric ∈ {kl_points, sliced_w2}`,
one row per (method, seed) before aggregation.

## Implementation (GAP E4 — done)

- `samplers.py` — **all 11 methods are self-contained CLOSED-FORM samplers** here;
  they do NOT use the `scisi/src` posterior classes, because the linear-Gaussian
  prior velocity / score / drift are available in closed form. Covers the three
  ours samplers (SI-SDE, DM-SDE, FM-ODE), and the faithful baselines: FlowDAS
  (paper Algorithm-2 importance-weighted residual guidance), Guided FM in both
  FIG and OT-ODE modes, D-Flow SGLD (Adam-style bias-corrected RMSProp
  preconditioner), SDA (DiffusionPosterior `fm_coeff` weighting, no DPS
  step-norm), SURGE (guided reverse-SDE + Girsanov SMC reweighting), a stochastic
  EnKF, and a bootstrap particle filter. Guidance weights and scores are derived
  in closed form and verified against the exact posterior.
- **Regime-appropriate hyperparameters** are used here (OT-ODE `σ_y²=R, γ=1`;
  D-Flow `K=200`) because the NS-locked noiseless / few-step settings are
  degenerate in the full-observation analytical regime.
- `driver.py::AnalyticalRunner.evaluate` — runs each method, computes Gaussian KL
  and sliced-`W2` to the exact posterior, emits the tidy rows. Truth + obs are
  seeded identically across methods.
- The three manuscript figures (see Deliverables) are generated under
  `manuscript/figures/analytical/`.
- `convergence_study.py` — the covariance-mode ablation (per-member exact ≡
  shared in the linear case; isotropic Jacobian-free plateaus).

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
