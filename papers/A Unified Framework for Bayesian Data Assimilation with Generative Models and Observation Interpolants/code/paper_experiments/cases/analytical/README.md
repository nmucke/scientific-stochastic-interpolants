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

## Nonlinear observation variant (`analytical_nonlinear`)

Same one-step system, observation `y = g(x1) + e`, `e ~ N(0, R I)`, `g` elementwise
(`PointwiseObservationOperator`, `L = I`). The linear case above is untouched
(bit-identical outputs); the `nonlinear:` block of
`configs/case/analytical_nonlinear.yaml` switches `AnalyticalRunner.run()` to
`nonlinear_driver.py`.

- **Exact reference** (`nonlinear_reference.py`): the posterior factorises per
  coordinate, so each marginal `N(x; x0_i, prior_var) N(y_i; g(x), R)` is integrated
  by trapezoidal quadrature on an adaptive 200 001-point grid covering every region
  within 60 nats of the mode (both branches for `square`; raises if the window
  cannot be closed). Gives log-density, moments, entropy, inverse-CDF sampling and
  the exact-quantile marginal W2. Validated in `tests/test_nonlinear_reference.py`:
  identity `g` = Kalman posterior to the tested 1e-8; 1e6-particle SNIS agrees for
  arctan/cubic/square within 5 s.e.
- **Metrics** (`nonlinear_metrics.py`; definitions, floors and the KL caveat in its
  docstring), all coordinate-wise against the exact target: `mean_err` (RMS mean
  error in posterior-sd units), `var_err` (RMS relative variance error),
  `marginal_w2` (W2 to the exact marginal / its sd, RMS over coordinates),
  `mode_weight_err` (square/abs), `kl_marginal_knn` (per-coordinate 1-D k-NN
  KL(p||q); biased LOW when q under-covers p -- read with the moment metrics), and
  the diagnostic `unique_frac` (distinct-row fraction: particle degeneracy).
  Identity cells also emit the linear table's legacy `kl_points` / `sliced_w2`.
  Every (cell, seed) emits an `Exact reference` row (exact samples scored the same
  way) = the metric floor. Each (cell, method, variant) has `n_failed` / `n_seeds`
  rows: the metric means are over the surviving seeds only.
- **Cells**: d in {2, 10} x R in {0.01, 0.25} x {arctan c in {1, 4} (x0 ~ 2 +
  0.5 z), cubic (x0 ~ z), square (x0 ~ 0.5 + 0.5 z)}, plus an identity control at
  every (R, x0) used: 28 cells x 5 seeds.
- **Methods**: the 11 linear methods + Guided FM (OT-ODE), standalone SURGE, DPS
  (`Guided diffusion`). The Ours rows run all three covariance modes
  (`jacfree`, `shared`, `individual`). EnKF uses `h(member)` (stochastic EnKF,
  statistical linearisation); the particle filter weights by `N(y; h(x), R)`.
  M = 50, 4096 samples per draw (the tidy `E` column stays the nominal 64).
- **Baseline hyperparameters** are R-dependent (`tuned_hparams` in the case YAML),
  selected by `nonlinear_tuning.py` on the identity cell (d = 2, same R) with
  tuning seeds 100-101 by minimum `marginal_w2`; table in
  `results/analytical_nonlinear/tuning/`. DPS has no knob; the Ours samplers are
  not tuned (`jacobian_damping` stays 1.0).
- Every draw is wrapped in `try/except`; failures go to `failures_<tag>.json`.

Run the whole sweep (from `<PE>`):

```
uv run python -m cases.analytical.nonlinear_driver \
    --results-file results/analytical_nonlinear/results.csv
# or: uv run python "<PE>/run.py" case=analytical_nonlinear \
#         'results_file=${results_root}/analytical_nonlinear/results.csv'
```

Filtered runs: `--cells arctan_c1_ --dims 2 --seeds 0 1 2 --methods EnKF`.
Outputs: `results/analytical_nonlinear/{raw,results,failures,summary}_<timestamp>.*`
(+ `results.csv` / `summary.md` with `--results-file`). Re-tune with
`uv run python -m cases.analytical.nonlinear_tuning [--stage probe]`.
