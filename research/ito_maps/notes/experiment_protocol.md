# Forecast comparison protocol (specified before results)

Date: 2026-10-07.

## Questions
1. Can conditional Brownian-path maps preserve the predictive quality of the existing Föllmer-class stochastic interpolant using 1, 2, 4, or 8 evaluations?
2. How do deterministic flow maps compare under the same forecasting protocol?
3. Does fewer evaluations translate to lower synchronized GPU latency, accounting for path generation and larger conditioning input?
4. Are ensemble diversity, calibration, spatial spectra and autoregressive stability preserved?

## Data and selection
Use existing stochastic Navier–Stokes data and preprocessing, 128x128 fields, history length 5. Split by entire trajectories: train 0:160, validation 160:180, test 180:200. No random window split. Student objectives may sample interpolant states using training future fields. Evaluation must use only history as conditioning. Select training hyperparameters/checkpoints on validation. Keep test trajectory results separate from development results; any subsequent protocol revisions must be disclosed.

## Baselines
- Existing pretrained FollmerStochasticInterpolant at 1,2,4,8,16,32,64,128 Euler–Maruyama steps (subset of steps for pilots).
- Existing pretrained FlowMatchingModel at corresponding Euler steps.
- FlowMapModel at 1,2,4,8 maps, stochasticity only in Gaussian initial state.
- FollmerItoMapModel at 1,2,4,8 maps, deterministic observed initial state and fresh Brownian path per forecast.
- Persistence for a deterministic sanity baseline.

Teacher architectures and provenance must be reported. Large SI and FM teacher configs have matching widths [16,32,64,128]; small SI teacher [8,16,32,64] is useful for development but should not be silently substituted in matched comparisons. Preserve original checkpoint files. Report teacher pretraining cost as unavailable when logs do not exist; student training wall time is measured.

## Evaluation
Development pilots on validation trajectories. Final evaluation across all 20 held-out trajectories, with predetermined temporal origins spread across trajectory and at least 16 ensemble members if runtime permits. Report normalized and physical-unit conventions explicitly.

Primary measure: repository fair ensemble CRPS averaged spatially; aggregate and bootstrap at trajectory level (pixels and adjacent time points are not independent replicates). Secondary: ensemble-mean RMSE, spread/skill, central interval coverage, rank histogram, radial enstrophy spectrum discrepancy. Consider multivariate energy score to complement marginal CRPS. For autoregressive forecasts, initialize with five observed frames, then append each member's own forecast, without future truth inputs.

Use matched initial Gaussian seeds for FM/flow-map comparisons and shared Brownian paths where same-path fidelity is measured. A single aggregate Brownian increment is not sufficient for an exact nonlinear SDE map; disclose finite path features and quantify path approximation separately from learned map error.

Measure wall time after warmup, synchronize CUDA before and after, use equal ensemble sizes and precision. Include CPU/GPU transfer and path generation consistently. Count actual network evaluations using hooks where possible. Report both per-ensemble latency and NFE; avoid asserting speed solely from NFE.

## Result labeling
Smoke tests, short development pilots, and full experiments are separate. Do not state the central sample-quality claim unless measurements support it. An undertrained map is evidence about that run, not a general impossibility result. Validation-selected configurations evaluated on test are frozen for primary reporting; any test-informed change is exploratory and must be labeled.

## Finite ensemble calibration caveat (identified before test evaluation)
The existing repository spread/skill statistic uses the mean of pointwise standard deviations, which is not the RMS spread convention. Keep it for compatibility, and additionally report RMS unbiased ensemble spread with the finite-ensemble factor sqrt((E+1)/E). The empirical 5–95% interval is a diagnostic, not a calibrated 90% coverage target at small E: its endpoints lie inside the ensemble minimum and maximum, whose calibrated expected coverage is only (E−1)/(E+1). Rank histograms and fair CRPS therefore carry more weight than a naive comparison of empirical coverage to90%. These issues do not change the saved old metrics; new auxiliary metrics are added explicitly.
