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

## Multistep extension (sequential filtering)

`multistep.py` runs `K` cycles of `x^{k+1} = A x^k + w_k`, `y^k = H x^k + e_k`, with
member `i` of a method's posterior ensemble at cycle `k` as the anchor of member `i`
at cycle `k + 1` (the `sample_trajectory` convention of the field cases, and the
paper's "feeding each posterior sample back"). `A, Q, H, R` live under `system:` in
`configs/case/analytical_multistep*.yaml`, defaulting to the paper's identities; the
reference is the Kalman filter from `x^0` (`linear_system.py`). `driver.py` is
untouched. With `num_cycles=1` and the defaults, every method's KL and sliced
distance match the current single-step `driver.py` at equal `n_eval_samples`
(`tests/test_analytical_multistep.py`). The parity is with this code, not with every
published number: SURGE (SDA) gives a KL of about 3e-4 at `K = 1` in both drivers,
where the manuscript's table, produced with older `src/` and config, reports 0.173.
The manuscript's table uses 100000 samples, which is also the grid default.

| Config | System | Why |
|---|---|---|
| `analytical_multistep` | `A = Q = H = R = I` | the paper's system |
| `analytical_multistep_partial` | damped rotation `A`, `Q = diag(1, 0.25)`, `H = [1, 0]`, `R = 0.5` | an unobserved coordinate, non-identity `A`, anisotropic `Q`, slower covariance transient |

A further system needs its own YAML and `scenario_name`: the label is what identifies
a system in the results, and the figure script picks up every label it finds.

### What a per-member update targets

An exact sampler of `p(x^{k+1} | x^k_i, y^{k+1})` applied to every member does NOT
give the filtering posterior once the ensemble has spread. The filter is

    p(x^{k+1} | y^{1:k+1})  propto  sum_i p(y^{k+1} | x^k_i) p(x^{k+1} | x^k_i, y^{k+1})

and the per-member update drops the weights `p(y^{k+1} | x^k_i)`. Its law is still
Gaussian and available in closed form (`LinearGaussianSSM.member_chain_step`), with
the fixed gain `Q H^T (H Q H^T + R)^{-1}` where the filter's gain adapts to the
ensemble spread. For scalar `A = Q = H = R = 1` the steady variance is `2/3` against
the filter's `(sqrt(5) - 1) / 2 = 0.618`, and the gain `0.5` against `0.618`. This
is a property of the update rule, not a sampler error, so it does not shrink with
`M` or `E`. Methods with cross-member weights (EnKF, particle filter, the SURGE rows,
whose SMC weights include the member's marginal likelihood) do not have it.

The output separates the two effects:

* `kl_points`, `sliced_w2` -- to the Kalman posterior, the quantity of interest.
  `kl_points` uses a sampled reference, as the single-step case does; `kl_exact` is
  the same KL against the exact Kalman moments, with half the Monte-Carlo bias.
* `kl_chain` -- to the member-chain law. For the full-covariance Ours rows it is the
  accumulated sampler error. For rows that do not target that law (EnKF, particle
  filter, SURGE, `adapted_shared`) it only restates the gap between chain and filter.
* `Exact per-member chain` rows -- the score of a perfect per-member sampler. Not a
  lower bound: a biased sampler can land closer to Kalman by accident.
* `Kalman filter (sampled)` rows -- the Monte-Carlo floor of the estimators for iid
  members. EnKF and particle-filter members interact, so their floor can differ.
* `adapted_shared` variant -- Ours (shared) with members first resampled by the
  closed-form `p(y^{k+1} | x^k_i)`. This is a diagnostic: for a learned prior that
  predictive likelihood is not available in closed form.

`sliced_w1` here is the single-step `sliced_w2` column: `wasserstein_distance`
averages 1-D W1 distances, which is why the appendix table says sliced-`W_1`. In this
case `sliced_w2` is a true sliced W2, so compare the two files on `sliced_w1`.

### SURGE around Ours

The grid's `<setting>_surge` cells hold SI-SDE and DM-SDE inside the SURGE SMC layer,
as variants of the Ours rows: `surge_shared`, `surge_jacfree`, and `surge_unguided`
(the same SMC, proposing from the prior; the ablation of the wrapped sampler). The
likelihood is the guidance, so with the weights ignored a SURGE row moves members
exactly as the plain row does (tested). `ess_threshold` is the existing rows'. FM-ODE
is left out: SURGE's weight is a ratio of path measures with noise `g dW`, and adding
noise to FM-ODE makes it DM-SDE. `NFE` counts two prior evaluations per step.

SURGE supplies the missing weight without computing it. Its log-weight telescopes to

    log p(y | x_1) - int u . dW - 1/2 int |u|^2 dt,    u = (w / g) G

the likelihood at the endpoint times the Girsanov ratio from proposal to prior path,
normalised over all members. For any proposal, the expected weight of member `i` is
`E[p(y | x_1) | x^k_i] = p(y^{k+1} | x^k_i)`, so resampling across members applies the
term in expectation. It needs `y`, `R`, the injected noise and the guidance, the same
quantities a learned prior has, and not the closed-form predictive likelihood that
`adapted_shared` uses. The prior in the target is the Euler chain of the model at `M`.

The expectation is not the realisation. A Doob transform of the prior would make the
weight exactly `p(y | x^k_i)`, with `w = g^2`. Ours injects `w = a + 1/2 g^2`, which
preserves marginals but is not that transform (`w / g^2` is 1 to 50 for SI, 2500 to 1.5
for DM over `tau`), so `int |u|^2 dt` adds weight variance that a better sampler for
the per-member conditional does not remove. In low dimension the prior proposal is
therefore at least as good as Ours inside SURGE: the gain comes from the weights, not
from the wrapped sampler. See `diagnose_analytical_multistep_surge.py`.

This is a particle filter, and it inherits the limit: terminal ESS falls with the
observed dimension for every proposal, and the closed-form predictive weight too. The
d = 2 rows do not show what happens at field scale; the same script runs d = 8 and 32.

### Forecast-residual correction

`FORECAST_CORRECTION.md` (variant A, `scisi.posterior_models.forecast_correction`)
adds the missing across-member part of the gain after each guided draw:
`x''_i = x'_i + (K_ens - K_q)(y + e_i - H f_i)`, with `f_i` an unguided forecast
from `x^k_i`. It is off by default, and off leaves every row bit-for-bit as it was.
Corrected Ours rows are extra variants:

| Variant | Meaning |
|---|---|
| `<jacfree\|shared>_fc<S>` | correction with `forecast_num_steps = S`, online `Q_hat` (a second forecast) |
| `..._bank` | `Q_hat` from a residual bank, 4096 pairs from `N(0, I)` states at the same `S` |
| `..._bank<B>` | the same with `fc_bank_size = B` pairs; the size is in the tag only when it is not 4096, so the stored `_bank` tags keep their meaning |
| `..._g<E>` | ensemble statistics over independent groups of `E` contiguous members (the last group takes the remainder); the pooled `N` members are scored |
| `adapted_shared_g<E>`, EnKF `g<E>` | the comparators on the same groups (`LINEUPS=group`) |

Switches (case YAML or `+key=value` on `run.py`): `forecast_correction` (add the
corrected rows to the lineup), `fc_only` (a cell with only them), `fc_steps`
(list of `S`; default `M`), `fc_variants`, `fc_q_source`, `fc_group_size`,
`fc_bank_size`, `group_only`. The grid takes `LINEUPS=fc` (one cell per `S` in
`FC_STEPS`) and `LINEUPS=group`, with `FC_STEPS`, `FC_VARIANTS`, `FC_Q_SOURCE`,
`FC_GROUP`, `FC_BANK_SIZE` (with `FC_Q_SOURCE=bank`; a non-default size is in the cell
name too). A blank `FC_STEPS` gives no fc cells; the grid logs it as a failure and
goes on with the other lineups. The tables and figures take the group sizes `E` of
the grouped rows from the data.

```bash
SETTINGS=paper LINEUPS=fc FC_VARIANTS=shared FC_STEPS="1 2 5 10 25 50" bash "$EXPERIMENT_DIR/run_analytical_multistep_grid.sh"
SETTINGS=paper LINEUPS="fc group" FC_GROUP=64 FC_STEPS=5 SEEDS=0 bash "$EXPERIMENT_DIR/run_analytical_multistep_grid.sh"
```

`NFE` per cycle is `M + 2 S` (online) or `M + S` (bank; the bank's own cost is not
counted). Cycle 1 shares the plain row's seed for the guided draw. The groups are
the library's `group_size` partition (contiguous, the last group takes the remainder,
a single leftover member joins the previous group); the grouped comparators use the
same partition (`_groups`), so `..._g<E>` and `adapted_shared_g<E>` / EnKF `g<E>`
rows share their groups. The library picks the unlocalised gain order by size
(`Z' (HX' X)` here, never an `E x E` matrix at `E = 10^5`). Rows stored before the
library rewrite still hold: bank rows reproduce bit-for-bit, online rows to round-off
(about 1e-7 in a 2000-member, 3-cycle check).

### Differences from the single-step code path

* From cycle 2 on, SDA and SURGE (SDA) use `FullFirstStepDiffusionModel`.
  `DenoiseDiffusionModel._compute_first_step` omits the velocity, so the first Euler
  step loses `mu dt` and the result drifts with the size of the state. That is small
  near the origin and grows on a random walk. `src/scisi` is unchanged because fixing
  it there moves the single-step SDA rows; cycle 1 keeps the original for parity.
* `E` is the true ensemble size, whereas the single-step rows record the unused
  `ensemble_size` (64) and draw `n_eval_samples` members.
* A cycle-mean row skips NaN cycles. A collapsed ensemble gives a NaN KL at that cycle;
  `n_traj` in `per_step.csv` counts the seeds with a finite value.

### Reading the baselines

Baseline hyperparameters are the single-step `analytical` cells of
`configs/method/*.yaml`, chosen for one cycle of the identity system. They are not
tuned for sequential use on either system. FlowDAS at `M = 50` has `zeta = 0.004`, so
guidance is close to off and its rows are near a free-running forecast; D-Flow also
under-corrects. D-Flow and FIG never use `R`. On the partial system D-Flow's
unobserved latent has no data gradient, so its Langevin noise accumulates with `M`.

### Running it

```bash
EXPERIMENT_DIR="$PWD/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments"
PY="$PWD/.venv/bin/python"
bash "$EXPERIMENT_DIR/run_analytical_multistep_grid.sh"   # env: SETTINGS LINEUPS SEEDS M N_EVAL CYCLES ROOT PY
"$PY" "$EXPERIMENT_DIR/aggregate_analytical_multistep.py"
"$PY" "$EXPERIMENT_DIR/make_analytical_multistep_figures.py"
"$PY" "$EXPERIMENT_DIR/diagnose_analytical_multistep_surge.py"   # weights behind the SURGE rows
```

Outputs: `results/analytical_multistep/aggregated/{all,per_step}.csv`, `table_*.md`,
`generated/tab_analytical_multistep.tex`, and
`figures/analytical_multistep/analytical_multistep_{error,surge}_vs_step.{pdf,png}`; with
corrected rows also `analytical_multistep_{correction_vs_step,error_vs_forecast_steps}`,
`aggregated/correction_*.md` and `generated/tab_analytical_multistep_correction.tex`. In the
per-step file `step` is 0-based, as in `common/per_step_io.py`; tables and figures
use `k = step + 1`. `per_step_file` holds one seed, so a manual `run.py` call needs
`seeds=[s]` and a `traj<s>` filename.
