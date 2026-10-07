# Partial held-out seed-0 assessment

**Status: complete for seed-0 one-step curves; overall assessment remains partial.** All four one-step model curves are now complete, including flow matching. Ten-lead rollouts and student training seeds 1 and 2 remain pending. No model, feature count, hyperparameter, or forecast case was changed after viewing these scores.

Sources are `results/test_one_step/{ito_rollout_seed0,flow_rollout_seed0,stochastic_interpolant,flow_matching}/evaluation.json`. Their checkpoint hashes match the selections in `results/frozen_protocol.json`. Each budget uses all 20 test trajectories 180–199, origins 5/45/85, E=16, and the frozen sampling seed. Thus there are 60 matched one-step forecasts per budget. Scores below use the evaluator's normalized coordinates; times are seconds per E=16 ensemble forecast.

Paired intervals below were recomputed from raw cases using `experiments/paired_comparison.py`'s `compare` function with 10,000 trajectory bootstrap resamples and seed 712. Origins were averaged within trajectories first. These intervals describe test-trajectory variability for the fixed seed-0 models, not training-seed variability, and are not multiplicity-adjusted hypothesis tests.

## Main assessment

The completed results support **similar proper forecast scores with substantially fewer neural evaluations**, and a clear improvement over a moderately resolved SI16 baseline. They do **not** support an unqualified claim that the Itô map preserves calibration or outperforms the deterministic flow map on every metric.

| Model | NFE | Fair CRPS | Fair energy | Mean RMSE | Relative spectral error | Ensemble seconds |
|---|---:|---:|---:|---:|---:|---:|
| Itô map | 1 | 0.27287 | 0.33609 | 0.47036 | 0.27037 | 0.04609 |
| Itô map | 8 | 0.27052 | 0.33356 | 0.47253 | 0.25155 | 0.34054 |
| Flow map | 1 | 0.26656 | 0.32854 | 0.46864 | 0.26290 | 0.04348 |
| Flow map | 8 | 0.26905 | 0.33161 | 0.46258 | 0.25667 | 0.31607 |
| Föllmer SI | 16 | 0.28802 | 0.35467 | 0.48627 | 0.32089 | 0.61293 |
| Föllmer SI | 64 | 0.26780 | 0.33078 | 0.47446 | 0.24716 | 2.43447 |
| Föllmer SI | 128 | 0.26718 | 0.32992 | 0.47608 | 0.25190 | 4.89958 |
| Flow matching | 64 | 0.26501 | 0.32673 | 0.47152 | 0.27751 | 2.44906 |
| Flow matching | 128 | 0.26481 | 0.32644 | 0.47322 | 0.28659 | 4.94788 |

**Itô1 versus SI16:** CRPS improves by 5.26%, energy score by 5.24%, ensemble-mean RMSE by 3.27%, and relative spectral error by 15.74%, with 16 times fewer backbone evaluations and a measured 13.30-fold latency reduction. The paired CRPS difference is −0.01516 [−0.02089, −0.00949]. The energy difference is −0.01859 [−0.02549, −0.01187]; the RMSE difference is −0.01591 [−0.02369, −0.00822]. All inspected calibration diagnostics also improve for this comparison.

**Itô8 versus SI64:** CRPS is 1.02% higher and energy score 0.84% higher; RMSE is 0.41% lower and relative spectral error 1.78% higher. This uses eight times fewer backbone evaluations and is 7.15 times faster in measured ensemble latency. Paired CRPS difference: +0.00272 [−0.00129, +0.00668]; energy difference: +0.00278 [−0.00207, +0.00761]; RMSE difference: −0.00194 [−0.00836, +0.00451]; spectral difference: +0.00440 [−0.01110, +0.01845]. These small score differences are consistent with useful score retention; intervals crossing zero do not establish formal equivalence or noninferiority.

**Itô8 versus SI128:** CRPS is 1.25% higher and energy 1.10% higher, with 16 times fewer neural evaluations and 14.39-fold measured latency reduction. The paired CRPS difference is +0.00334 [−0.00179, +0.00888]. Relative spectral errors are almost equal (0.25155 versus 0.25190). This stronger solver reference remains important even though its test CRPS gain over SI64 is only approximately 0.23%: its calibration improves appreciably.

## Calibration is the clearest qualification

| Model | NFE | RMS spread | Corrected RMS spread/skill | Empirical 90% coverage | Extreme-rank fraction |
|---|---:|---:|---:|---:|---:|
| Itô map | 1 | 0.33153 | 0.76478 | 65.56% | 24.84% |
| Itô map | 8 | 0.36503 | 0.84052 | 69.70% | 20.89% |
| Flow map | 1 | 0.37355 | 0.85878 | 71.07% | 19.61% |
| Flow map | 8 | 0.31858 | 0.74116 | 64.16% | 25.93% |
| Föllmer SI | 16 | 0.30377 | 0.67755 | 59.55% | 30.36% |
| Föllmer SI | 64 | 0.40856 | 0.93576 | 74.89% | 16.35% |
| Föllmer SI | 128 | 0.43483 | 0.99360 | 77.35% | 14.40% |
| Flow matching | 64 | 0.41259 | 0.94388 | 75.28% | 15.97% |
| Flow matching | 128 | 0.43249 | 0.98602 | 77.18% | 14.41% |

Against SI64, Itô8 has **10.66% less RMS spread**, coverage lower by **5.19 percentage points**, and extreme-rank frequency higher by **4.54 percentage points**. The paired differences are:

- RMS spread: −0.04354 [−0.04583, −0.04122].
- Coverage: −5.19 pp [−6.03, −4.39].
- Extreme-rank fraction: +4.54 pp [+3.75, +5.32].

Against SI128, the spread deficit is 16.05%, the coverage deficit is 7.65 pp [−9.02, −6.38], and the extreme-rank excess is 6.48 pp [+5.33, +7.72]. These are meaningful and consistent signs of residual underdispersion, despite close CRPS/energy values.

Coverage uses empirical finite-ensemble 5th/95th percentiles, so the population target 90% should not be treated as an exact finite-E reference. The comparison still uses the same E throughout. The exchangeable continuous-sample reference for the extreme-rank fraction is 2/(E+1)=11.76%, and the corrected RMS ratio is closest to one for SI128. Both complementary diagnostics support the same underdispersion conclusion.

The Itô1/SI16 comparison has the opposite direction: coverage rises by 6.01 pp [4.96, 7.10], extreme-rank fraction falls by 5.52 pp [−6.66, −4.43], and RMS spread rises by 9.14%. Thus the one-step map improves both scores and calibration relative to SI16; that result should not be generalized to the more resolved SI64/128 references.

## Deterministic flow-map comparison and step-count behavior

At one step, the deterministic flow map has lower point estimates for CRPS/energy and stronger spread/coverage than the Itô map. The Itô-minus-flow CRPS difference is +0.00631 [−0.00199, +0.01444], while Itô coverage is 5.51 pp lower [−7.25, −3.78]. At eight steps, the two maps have close CRPS/energy estimates, but the Itô map has larger spread and 5.54 pp higher coverage [4.33, 6.68]. Its ensemble-mean RMSE is also worse by 0.00995 [0.00057, 0.01930], approximately 2.15%. Both maps remain probabilistic forecasters: the deterministic flow map receives a Gaussian random initial state.

Neither map exhibits monotone score improvement with more steps. Itô CRPS across 1/2/4/8 steps is 0.27287/0.27449/0.27865/0.27052; flow-map CRPS is 0.26656/0.27052/0.27490/0.26905. Intermediate-budget spread contracts before recovering at eight steps. The paper should therefore say that step count is selectable, not that quality is guaranteed to improve monotonically with step count.

The spectral comparison does not introduce a strong additional contradiction: Itô8 is close to both high-step SI references, with paired intervals crossing zero. Itô1 is clearly better than SI16 on the reported mean-spectrum discrepancy. This metric concerns the ensemble-average radial spectrum, not every individual sample's physical fidelity.

## Completed flow-matching comparison

The full flow-matching solver CRPS curve at 1/4/16/64/128 steps is 0.34622/0.31291/0.27195/0.26501/0.26481. The corresponding ensemble latencies are 0.04090/0.15625/0.61307/2.44906/4.94788 seconds. Its CRPS gain from 64 to 128 steps is only about 0.077%, although spread and calibration continue to change.

**Flow-map1 versus FM64 is a strong baseline result:** CRPS is 0.58% higher, with paired difference +0.00155 [−0.00031, +0.00319], and energy score is 0.56% higher, +0.00182 [−0.00046, +0.00381]. This requires 64 times fewer backbone evaluations and is **56.33 times faster** in measured ensemble latency. Ensemble-mean RMSE is 0.61% lower, difference −0.00287 [−0.00581, −0.00028], and spectral discrepancy is 5.26% lower, difference −0.01461 [−0.02744, −0.00286]. Nevertheless, RMS spread is 9.46% lower, coverage falls by 4.21 pp [−4.51, −3.91], and extreme-rank frequency rises by 3.63 pp [3.34, 3.92]. Thus the deterministic map exhibits the same proper-score/calibration distinction as the Itô map.

**Flow-map8 versus FM64 does not improve on the one-step flow map:** CRPS is 1.52% higher, +0.00404 [0.00141, 0.00657], and energy is 1.49% higher, +0.00488 [0.00178, 0.00784]. RMS spread is 22.78% lower and coverage falls by 11.11 pp [−11.77, −10.51]. The measured speedup is 7.75 times. Its lower mean RMSE (−1.90%) coexists with worse probabilistic scores and underdispersion; lower mean error alone must not be presented as better sample quality.

**Itô8 versus FM64** has CRPS 2.08% higher, +0.00551 [−0.00175, +0.01236], energy 2.09% higher, and approximately equal RMSE (+0.21%). Its latency is 7.19 times lower. Coverage is 5.58 pp lower [−7.05, −4.15], RMS spread is 11.53% lower, and extreme-rank fraction is 4.91 pp higher [3.60, 6.24]. The spectral point estimate is 9.35% lower, but its paired interval crosses zero. This reinforces that the Itô approach is a competitive fast stochastic forecaster, not a demonstrated universal quality improvement over deterministic alternatives.

The completed low-budget FM results also explain why uncertainty diagnostics matter: FM1 has mean RMSE 0.45454, lower than FM64's 0.47152, but CRPS 0.34622 and only 10.23% empirical interval coverage, versus 0.26501 and 75.28% for FM64. It is severely underdispersed (corrected RMS spread/skill 0.101 and extreme-rank fraction 87.17%). A mean-only accuracy comparison would conceal this failure.

All 18 seed-0 one-step model/budget summaries and the full curves are now reported in `results/final_report/one_step/`. The report uses 10,000 paired trajectory resamples (seed 712), has explicit test/E16/20-trajectory labels, and includes quality, calibration, RMS calibration, timing/spectrum, parameter and provenance tables. The PDF/PNG NFE and latency figures were visually inspected. The shaded intervals are pointwise marginal score intervals, not confidence intervals of model differences; use the paired comparison table for the latter.

## Appropriate provisional claim

A defensible description of these partial results is: **a one-step conditional Itô map improves proper scores over a 16-step Föllmer sampler at roughly 13-fold lower latency; an eight-step map approaches the proper scores of 64- and 128-step references while retaining less ensemble spread and weaker calibration.** The current numbers do not show general superiority over deterministic flow maps. Full conclusions should await the frozen ten-lead and additional-student-seed results without revising model selection from these test outcomes.
