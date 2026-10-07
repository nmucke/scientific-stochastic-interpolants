# Code-to-equation audit

Review date: 2026-10-07, branch `research/ito-map-forecasting`. Reviewed current source read-only while the initial GPU runs were in progress. Later changes must be reconciled with this audit. No evaluation results below are held-out-test claims.

## Mathematical objects and implemented locations

| Mathematical object | Implementation | Audit finding |
|---|---|---|
| Conditional interpolant `I_t=alpha c+beta y+gamma sqrt(t)z` and SDE drift target `R_t=alpha'c+beta'y+gamma' sqrt(t)z` | `FollmerItoMapModel.training_pair`, existing interpolation `forward` / `forward_diff` | Existing interpolator reused. No derivative of `sqrt(t)` is added. Default family uses the baseline forecasting SDE, not the distinct optimal Föllmer diffusion. |
| Deterministic interpolant from random base | `FlowMapModel.training_pair` | Base is sampled noise, target is future data, and observed history remains a separate condition. This is required for nondegenerate probabilistic forecasts. |
| `Phi=x+(t-s)G+M_t-M_s` | `BaseMapModel.map`, `FollmerItoMapModel.noise_increment` | Correct additive decomposition. An optional mask multiplies both drift and martingale increments; current NS experiments use no nontrivial mask, and masked domains need a separately stated model. |
| Diagonal `G(t,t)=b_t` | `FollmerItoMapModel.average_velocity`, `MapTrainer._compute_loss` | Equal-time path features are zero. Raw-data mode regresses to `R_t`; with any teacher, the target is frozen teacher drift instead. Do not call the actual teacher-distilled run fresh data-only drift learning. |
| Two-time neural condition | `MapConditionedNetwork`, `TwoTimeEncoder` | Existing starting-time embedding plus `(t-s)*gap_network(t-s)`; last gap layer initialized zero. All other backbone weights are copied from teacher when requested. |
| Path inputs | `PathConditionedInit` | K raw local increments per state channel, chronological feature-major order, projected with a bias-free 1×1 convolution after the original initial/history processing block. Projection initialized zero. At initialization map is a drift Euler step plus exact noise increment, not an already trained finite-time map. |
| Martingale law | `WeightedBrownianPath` | `q(t)=a²(t-t²+t³/3)`. Each state coordinate has an independent Gaussian driver. Lazy bridge queries give exact joint finite-dimensional laws; integrated variance uses a factored difference of cubes for numerical stability. |
| Coupling | Reuse one `WeightedBrownianPath` per PSD example or physical forecast | Parent and both child maps query the same cached path. Repeated exact times return cached values; noise increments telescope. |
| Velocity-form PSD target | `MapTrainer._compute_loss` | Target is `lambda G_EMA(s,u,x)+(1-lambda)G_EMA(u,t,Phi_EMA(s,u,x))`, detached. Student receives the full interval. The noise cancels algebraically only because the path is shared. |
| Interval curriculum | `MapTrainer._intervals` | Interval lengths grow to 1; an explicit initial-to-current-maximum endpoint atom later trains `[0,1]`. Split fraction is uniform on `[.25,.75]`. This is an implementation sampling choice, not the unweighted continuous integral of the reference theory. |
| Fine-grid supervised alternative | `objective=teacher_rollout` | Euler teacher drift with exact M increments on each interval; finite drift integration error remains. No claim that teacher rollout is an exact SDE solution. |
| Validation | `MapTrainer._compute_val_loss` and `_compute_loss` | Fixed validation RNG, full interval range. With a teacher present, validation uses teacher-rollout targets even for PSD training; checkpoint is selected by this teacher-agreement loss. Current launch uses teacher_steps=32. |
| Sampling | `BaseMapModel.sample` | Uniform artificial-time partition; K Brownian features constructed separately for each interval but from one path object. One network evaluation per map interval. |
| Physical rollout | evaluation loop and `sample_trajectory` | History appends each member's own prediction; new Brownian path per future physical forecast. Gaussian-flow baseline obtains new base noise each physical step. |

## Finite-feature PSD is a conditional regression, not an exact semigroup theorem

The full Brownian-path solution satisfies pathwise composition. The network does not observe the full path. For a parent interval with K local increments, two child intervals have 2K local increments at generally different locations. A random split need not make either grid a subset of the other; all values are jointly valid on one path, but the child target generally reveals within-parent variation absent from the student's feature vector.

Fix the teacher parameters for one optimization step and call its composed average-drift target `T`. Let `U` collect the student's observed state, history, two times, and K parent features. Population squared-error regression minimizes at `E[T|U]`, with unavoidable residual `E||T-E[T|U]||²`. This is ordinary conditional expectation, not an optimization failure. A random unobserved split can add target variability when the finite-feature teacher is not exactly composition-consistent. EMA changes the target over training, so there is not even a single static population target across all updates.

If the teacher used exact full-path maps, `T` would be the true finite-time drift and independent of the split, but hidden within-parent Brownian information would still generally create irreducible conditional variance. If instead the teacher itself uses compressed maps, bias in its two transitions is added to that information loss. Thus: low PSD loss does not prove true SDE fidelity; a positive plateau does not by itself indicate failure; zero PSD loss cannot be promised for K=4; equality of student and teacher *endpoint distributions* is not guaranteed by diagonal correctness or shared increments alone.

No new theorem is claimed. The elementary projection identity in the paper makes this limitation visible, and the OU example isolates it without neural approximation. Same-path validation versus a sufficiently resolved teacher, predictive spread/calibration, K ablation, and comparison against deterministic maps are complementary evidence.

## Evaluation code and reporting interpretation

`main_map_model_test.forecast_metrics` evaluates normalized coordinates. CRPS is the repository fair ensemble estimator; energy score uses Euclidean distances divided by sqrt(number of state coordinates), i.e. RMS-normalized multivariate distance. RMSE is the ensemble-mean score. Spread/skill and central 5–95% empirical quantiles are diagnostics. For small finite ensembles, empirical quantile coverage need not equal exactly 90% under a perfectly calibrated continuous distribution; report ensemble size and avoid overinterpreting small coverage gaps.

The relative enstrophy metric compares the *ensemble-averaged radial spectrum* against the single observed future spectrum, using an L2 ratio. It is not a comparison to an exact conditional expected spectrum, nor a proof of an invariant measure. The paper should give that formula if the metric is in a results table.

Aggregation averages temporal origins within trajectories and bootstraps entire trajectories. This respects the basic dependence structure better than bootstrapping pixels. The current summary computes separate model confidence intervals. Claiming a statistically resolved model difference should additionally use paired trajectory-level differences, not compare overlap of separate CIs.

Network hooks count batched backbone forward calls; this corresponds to NFE per ensemble member when every call processes the whole ensemble. Timings synchronize CUDA and include path feature construction and model.sample's transfer to CPU; data loading, target preprocessing, and metric computation are outside the interval. Report these boundaries explicitly.

`seed_everything` repeats common seeds across budgets. Lazy bridge path values depend on query order for a fixed seed: identical seeds at different sampling budgets do not mean shared Brownian realizations. The evaluator already comments this correctly. Same-path diagnostics must explicitly share one path object; cross-budget independent or common-seed distributional comparisons remain valid.

## Handover gaps to resolve before final reporting

- Initial research README still claimed no implementation/results; update it after code/experiment milestones.
- Store final selected checkpoint identifiers, hashes, full resolved configuration, launched command, raw output, and interpretation together. Current evaluation wrapper records hashes; ensure custom comparison scripts do too.
- Record actual student updates, effective optimization budget, and initial/final checkpoint selection; planned 2,000 updates are not necessarily completed updates.
- Report teacher pretraining cost/provenance as unavailable when original logs cannot be recovered.
- Keep validation pilot artifacts separate from final held-out test results. Freeze selected training and sampling choices before interpreting test results.
- Keep data-generation provenance uncertainty explicit; the current simulator cannot certify the supplied archive's physical time interval.


## Metric caveat follow-up

The paper now states the exact existing spread definition: sqrt((M+1)/M) times the spatial mean of population ensemble standard deviations. The code docstring's claim that calibration implies ratio one is not mathematically exact. Population-variance bias, the nonlinear square root, spatially varying uncertainty, and the random denominator all matter. A conventional squared RMS spread based on unbiased sample variances obeys an expected second-moment equality under conditionally iid truth/ensemble; even its finite-sample ratio need not have expectation one. No baseline is called underdispersed solely from the compatibility ratio or empirical coverage.

An empirical 5–95% interval lies inside the ensemble min/max, whose expected coverage is (M−1)/(M+1) for exchangeable continuous forecasts. Therefore empirical coverage is at most 77.78% for M=8 or 88.24% for M=16 even for a calibrated distribution. The exact expected coverage for uniform iid truth/ensemble with linear quantile interpolation is 0.9(M−1)/(M+1): 70.00%, 79.41%, 84.55% at M=8,16,32. The appendix gives the short order-statistic derivation. This caveat does not invalidate matched-M comparisons, but rules out comparison to 90% as an exact calibration criterion. Rank histograms and fair CRPS provide complementary evidence.

## Two-stage experimental method update

The study progressed from the initial PSD-only training design to teacher-diagonal PSD followed by direct same-path teacher-rollout refinement. The launch scripts are `experiments/train_psd_pilots.sh` (20×100=2,000 updates) and `experiments/train_rollout_stage.sh` (10×50=500 updates). The second stage initializes from the validation-selected first-stage model, uses 32 teacher integration steps, and keeps diagonal teacher supervision. All intervals are available immediately, with an endpoint sampling probability of 0.5; learning rate is 5e-5. Its loss is simulation-based and must be counted that way. Validation still uses fixed-noise same-path teacher rollout on four batches, separate from four-trajectory/eight-member forecast development pilots. Final test results remain separate and were untouched at the time of this update.
