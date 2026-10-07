# Frozen NS teacher: conditional Brownian feature compression

Date: 2026-10-07. Evidence: `research/ito_maps/results/ns_path_compression/results.json`. Script: `research/ito_maps/experiments/ns_path_compression.py`. Validation only; no student training or test evaluation.

## Question and estimator

For a fixed forecast history, how much of a frozen stochastic-interpolant teacher's endpoint depends on Brownian fluctuations that a map with K local increment features cannot see?

The experiment samples G=8 coarse weighted-Brownian paths and R=4 independent fine bridge refinements conditional on each coarse path. The same fixed history is used in every call. The teacher integrates its learned drift using 64 Euler substeps, adding exact increments of M_t=∫_0^t(1-r)dW_r. The coarse feature counts K=1,4,16 form nested partitions and divide the teacher grid.

Let endpoints be X_gr, with group mean Xbar_g. Averaging pixelwise unbiased variances over state coordinates gives

- W = mean_g Var_r(X_gr), estimating E[Var(X_1|coarse path)].
- B = Var_g(Xbar_g), containing between-path variation plus W/R from finite replication.
- Explained variance estimate = B−W/R.
- Total variance estimate = B+(1−1/R)W.

The first quantity is the irreducible population same-path MSE for a predictor restricted to these features, estimated under this finite-step teacher and fixed history. It is not a forecast-score lower bound: another transport with the same random input dimension could reproduce an endpoint distribution without preserving the teacher's coupling.

Forks copy cached path observations but generate independent subsequent Gaussian bridges. Parent caches contain only the corresponding coarse points; solver refinements on children never leak back into conditioning. CPU tests verify conditional bridge covariance/independence, exact known linear-SDE risk, and zero within-group variance when every solver increment is given.

## Results

| K | W: omitted variance | Standard error across 8 coarse groups | Total variance estimate | Omitted fraction | Irreducible RMS same-path error |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.05595703 | 0.00379153 | 0.19683848 | 28.4279% | 0.23655239 |
| 4 | 0.01711263 | 0.00189376 | 0.17669714 | 9.6847% | 0.13081527 |
| 16 | 0.00105534 | 0.00009366 | 0.16091162 | 0.6558% | 0.03248594 |

Conditioning on sixteen rather than four increments reduces W by a factor of approximately 16.2. Every coarse group exhibits lower within-group variance. This directly supports evaluating a finer Brownian representation on this problem rather than inferring usefulness only from the analytic OU example.

The ideal conditional-mean endpoint retains approximately 95.0% of the teacher's standard deviation at K4 and 99.7% at K16 under these estimates. Thus K4 compression is measurable, but its estimated effect is much smaller than the initial PSD student's severe spread reduction. A successful K16 ablation would need to demonstrate improved learned forecast behavior, not merely point to these conditional variance estimates.

## Limits and comparison caveats

This result concerns trajectory160, origin45, normalized vorticity coordinates, and one frozen teacher checkpoint (SHA256 `cdb28038e89a6a49757260169fe4043f39e7959d61d4ab6ac363089c1c84e810`). It cannot establish a dataset-wide risk floor or a conditional law of the true physical system.

The true unconditional forecast variance does not depend on K; estimated totals vary from 0.1609 to 0.1968 because only eight coarse groups and four replicas are used. Omitted fractions therefore have meaningful Monte Carlo uncertainty. Standard errors use coarse groups, never independent-pixel assumptions; the recorded values do not quantify variability across forecast histories.

The diagnostic uses 64 Euler drift steps and exact integrated diffusion. Current student supervision uses 32 such teacher steps. Existing SI baseline inference uses its existing Euler–Maruyama implementation. These related numerical processes are not identical, so the diagnostic's MSE values cannot be directly subtracted from the student's aggregate validation losses.

Local features and supervised same-path regression suggest conditional-mean targets. Progressive self-distillation has a different finite-feature objective, and its optimum need not equal that conditional mean. The compression experiment separates an information constraint from architecture and optimization, but it does not guarantee that adding channels removes the measured practical error.

## Controlled follow-up

Refine the K4 PSD source to K16 by repeating every coarse projection coefficient on its four child raw increments. Since children sum to the parent increment, the initial map function is preserved for the same underlying path. Run the same 500 teacher-rollout updates, teacher grid, learning-rate schedule, and validation contexts as the K4 comparison. Count the shared 2,000-update PSD source in cumulative training cost. Record whether forecast CRPS, spread/calibration, spectra, latency, and teacher-map error improve together. Do not select a preferred conclusion before observing that result.
