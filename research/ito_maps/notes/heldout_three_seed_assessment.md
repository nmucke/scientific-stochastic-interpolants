# Complete held-out three-student-seed assessment

All frozen training and evaluations are complete. `results/confirmation_audit.json` records a successful audit of 4,360 forecast rows, the selected checkpoint hashes and twelve completed training stages. This note supersedes the provisional training-seed conclusions in `heldout_seed0_assessment.md`; its seed-0 full-budget curves remain valid and separately reported. No tuning followed the held-out evaluations.

Sources are the two `results/final_report/{one_step,rollout}/student_training_seeds/training_seed_report.json` files, their `per_student_seed.csv` and `fixed_teacher_comparison.{json,csv,tex}` companions, and the audited raw evaluation JSONs. All results use normalized dataset coordinates, E=16 and test trajectories 180–199. One-step forecasts use origins 5/45/85; ten-lead autoregression uses origin 45. Both map models were trained with seeds 0, 1 and 2, each with 2,000 PSD updates and 500 rollout-refinement updates. The pretrained teachers and hyperparameters were fixed.

## Uncertainty and interpretation

Each score is computed within an E=16 forecast ensemble, then averaged over origins within a trajectory, then trajectories. Across-seed means average these separately computed scores; ensemble members from different trained models are **never pooled**. The reported seed SD is the sample standard deviation of the three trained students' score means, and the range is their observed minimum/maximum. It is descriptive variability from three student trainings, not an estimate of teacher-training uncertainty or a confidence interval for future training runs.

Paired 95% data intervals use 10,000 trajectory bootstrap draws (seed 712) after averaging the per-trajectory scores over the fixed three students. They describe test-trajectory uncertainty conditional on those trained checkpoints and the fixed teacher; they do not include training-seed variation. Intervals are pointwise, not simultaneous across leads or adjusted for inspecting several metrics. A small or zero-crossing difference does not establish formal equivalence.

## Proper scores and training-seed variation

| Forecast | Model | Steps | CRPS mean ± student SD | Student range |
|---|---|---:|---:|---:|
| One step | Föllmer Itô map | 1 | 0.27439 ± 0.00290 | [0.27256, 0.27773] |
| One step | Föllmer Itô map | 8 | 0.27139 ± 0.00168 | [0.27033, 0.27334] |
| One step | Flow map | 1 | 0.26663 ± 0.00076 | [0.26590, 0.26741] |
| One step | Flow map | 8 | 0.26870 ± 0.00088 | [0.26769, 0.26935] |
| Lead 10 | Föllmer Itô map | 1 | 0.56167 ± 0.00199 | [0.55937, 0.56284] |
| Lead 10 | Föllmer Itô map | 8 | 0.54222 ± 0.00148 | [0.54052, 0.54316] |
| Lead 10 | Flow map | 1 | 0.53353 ± 0.00048 | [0.53301, 0.53397] |
| Lead 10 | Flow map | 8 | 0.54138 ± 0.00160 | [0.53965, 0.54280] |

| Forecast | Model | Steps | Energy mean ± SD | Mean RMSE ± SD | Spectral error mean ± SD |
|---|---|---:|---:|---:|---:|
| One step | Föllmer Itô map | 1 | 0.33799 ± 0.00356 | 0.47237 ± 0.00399 | 0.27514 ± 0.00457 |
| One step | Föllmer Itô map | 8 | 0.33462 ± 0.00205 | 0.47384 ± 0.00273 | 0.25237 ± 0.00075 |
| One step | Flow map | 1 | 0.32863 ± 0.00099 | 0.46883 ± 0.00104 | 0.26438 ± 0.00311 |
| One step | Flow map | 8 | 0.33124 ± 0.00112 | 0.46284 ± 0.00053 | 0.25652 ± 0.00100 |
| Lead 10 | Föllmer Itô map | 1 | 0.70578 ± 0.00241 | 0.96257 ± 0.00178 | 0.58584 ± 0.02469 |
| Lead 10 | Föllmer Itô map | 8 | 0.68160 ± 0.00178 | 0.96330 ± 0.00096 | 0.44612 ± 0.00645 |
| Lead 10 | Flow map | 1 | 0.66881 ± 0.00044 | 0.96897 ± 0.00135 | 0.35524 ± 0.00072 |
| Lead 10 | Flow map | 8 | 0.67947 ± 0.00189 | 0.96356 ± 0.00025 | 0.41948 ± 0.01399 |

The seed SDs are small relative to the score means, and the main ordering is consistent across the three trainings. This supports repeatability under the selected teacher/configuration, not broad robustness across teacher seeds, architectures, PDEs or datasets. At one step, Itô1 CRPS by seeds 0/1/2 is 0.27287/0.27256/0.27773; Itô8 is 0.27052/0.27033/0.27334. At lead 10, Itô8 is 0.54316/0.54298/0.54052. Flow-map1 has lower CRPS than Itô1 in each training seed at both endpoints; eight-step map scores are much closer.

## Paired comparison to fixed 64-step teachers

Negative differences favor the map for CRPS. Reference CRPS is 0.26780 for one-step SI64, 0.26501 for one-step FM64, 0.53293 for lead-10 SI64 and 0.53308 for lead-10 FM64.

| Forecast | Map | Steps | Mean CRPS difference [95% data CI] | Relative change |
|---|---|---:|---:|---:|
| One step | Föllmer Itô map | 1 | +0.00659 [-0.00059, +0.01395] | +2.46% |
| One step | Föllmer Itô map | 8 | +0.00360 [-0.00024, +0.00744] | +1.34% |
| One step | Flow map | 1 | +0.00161 [+0.00001, +0.00303] | +0.61% |
| One step | Flow map | 8 | +0.00368 [+0.00118, +0.00597] | +1.39% |
| Lead 10 | Föllmer Itô map | 1 | +0.02874 [+0.01873, +0.03913] | +5.39% |
| Lead 10 | Föllmer Itô map | 8 | +0.00929 [+0.00270, +0.01636] | +1.74% |
| Lead 10 | Flow map | 1 | +0.00045 [-0.00236, +0.00317] | +0.08% |
| Lead 10 | Flow map | 8 | +0.00829 [+0.00259, +0.01390] | +1.56% |

**Itô8 retains a favorable speed/score tradeoff but does not retain every measure of quality.** Relative to SI64, one-step CRPS is 1.34% higher and energy 1.16% higher; their paired data intervals include zero. At lead 10, CRPS is 1.74% higher and energy 1.82% higher, with positive paired intervals: CRPS +0.00929 [0.00270, 0.01636], energy +0.01219 [0.00391, 0.02134]. Its spectral discrepancy at lead 10 is 14.90% higher, difference +0.05785 [0.01228, 0.09533], so spectral proximity at one step should not be generalized to the rollout endpoint.

**The moderate-step SI comparison remains clearly favorable for one-step generation.** Three-seed mean Itô1 CRPS improves on SI16 by 4.73%, paired difference −0.01364 [−0.01797, −0.00929]. Itô8 versus the stronger SI128 reference is +1.58%, difference +0.00422 [−0.00091, 0.00972]. These two supplemental paired calculations use the same raw cases, averaging trained-student scores within each trajectory before the same 10,000 seed-712 bootstrap draws; SI16 and SI128 have no frozen ten-lead results.

**Flow-map1 is a strong competing method.** Its one-step mean CRPS is only 0.61% above FM64. The paired interval narrowly excludes zero (+0.00161 [0.00001, 0.00303]); this small effect should be described by magnitude, not used to turn a nearly identical score into a categorical failure claim. At lead 10 the difference is +0.08%, +0.00045 [−0.00236, 0.00317]. Flow-map8 is weaker than flow-map1 here: relative to FM64 its CRPS is 1.39% higher one-step and 1.56% higher at lead 10, both with positive paired data intervals.

A direct comparison of the three-seed map means gives Itô1 minus flow-map1 one-step CRPS +0.00776 [−0.00137, 0.01686], or +2.91%; at lead 10 it is +0.02814 [0.01747, 0.03891], or +5.27%. At eight steps the differences are +0.00270 [−0.00299, 0.00831] one-step and +0.00084 [−0.00550, 0.00726] at lead 10. These supplemental intervals pair the mean map scores on the same trajectories; they do not assume student-seed pairing establishes a random-effects training experiment. There is no evidence of a general quality advantage for the stochastic map over the deterministic alternative.

## Calibration persists as the main limitation

| Forecast | Model | Steps | RMS spread/skill mean ± SD | Empirical 90% coverage mean ± SD (pp) | Extreme ranks mean ± SD (pp) |
|---|---|---:|---:|---:|---:|
| One step | Föllmer Itô map | 1 | 0.759 ± 0.009 | 65.15 ± 0.60 | 25.23 ± 0.56 |
| One step | Föllmer Itô map | 8 | 0.839 ± 0.002 | 69.57 ± 0.12 | 21.01 ± 0.11 |
| One step | Flow map | 1 | 0.861 ± 0.004 | 71.20 ± 0.25 | 19.51 ± 0.23 |
| One step | Flow map | 8 | 0.750 ± 0.012 | 64.78 ± 0.75 | 25.37 ± 0.72 |
| Lead 10 | Föllmer Itô map | 1 | 0.633 ± 0.019 | 60.06 ± 1.31 | 30.10 ± 1.31 |
| Lead 10 | Föllmer Itô map | 8 | 0.787 ± 0.009 | 69.66 ± 0.45 | 20.95 ± 0.44 |
| Lead 10 | Flow map | 1 | 0.963 ± 0.010 | 78.60 ± 0.40 | 13.47 ± 0.38 |
| Lead 10 | Flow map | 8 | 0.801 ± 0.017 | 70.57 ± 1.02 | 20.22 ± 0.89 |

Itô8's mean coverage is below SI64 by 5.32 percentage points one-step [−6.11, −4.54] and 4.12 pp at lead 10 [−5.07, −3.25]. Its extreme-rank fraction is higher by 4.66 pp [3.88, 5.45] and 3.56 pp [2.80, 4.43], respectively. At lead 10, Itô1's corrected RMS spread/skill is only 0.633, with coverage 60.06% and extreme ranks 30.10%. These patterns recur across training seeds and contradict an unqualified calibration-retention claim.

Flow-map1 also has less spread than FM64: coverage is lower by 4.08 pp one-step [−4.34, −3.81] and 3.69 pp at lead 10 [−4.08, −3.30]. Nonetheless, it has substantially stronger rollout spread/coverage than the Itô maps, with corrected RMS spread/skill 0.963 at lead 10. Flow-map8 and Itô8 are similarly underdispersed at the rollout endpoint. Empirical 5th/95th percentiles from E=16 do not have an exact 90% finite-ensemble reference; the exchangeable continuous-rank extreme-bin reference is 2/17=11.76%. These caveats do not remove the matched-model spread and rank differences.

## Measured cost and training overhead

Timings are averages over trained students, not timings of a pooled ensemble. The speedup references are the corresponding fixed 64-step teachers on the same GPU. Full ten-lead costs sum all ten transitions within each trajectory before averaging.

| Model | Steps per transition | One-step ensemble seconds | One-step speedup | Ten-lead ensemble seconds | Ten-lead speedup | Ten-lead NFE |
|---|---:|---:|---:|---:|---:|---:|
| Itô map | 1 | 0.04630 | 52.58× | 0.44819 | 54.88× | 10 |
| Itô map | 8 | 0.34006 | 7.16× | 3.38834 | 7.26× | 80 |
| Flow map | 1 | 0.04349 | 56.32× | 0.42122 | 58.92× | 10 |
| Flow map | 8 | 0.31602 | 7.75× | 3.14726 | 7.89× | 80 |

All teacher references in this table use 64 NFE per physical transition, hence 640 for ten leads. Itô1's favorable SI16 comparison uses a different baseline and gives about 13.24× one-step latency reduction. The speed gains should be reported alongside the quality/calibration tradeoffs rather than as a claim that all budgets preserve quality equally. Seed-to-seed timing variation is small, but sequential runs on one GPU are not an independent hardware benchmarking study. The persistence baseline remains far worse in proper scores and is excluded from speedup comparisons.

Training elapsed times are recorded in `results/training/{ito,flow}_{psd,rollout}_seed{0,1,2}/history.json`. Summing the final elapsed clock of both stages gives Itô totals 1139.70/1143.17/1140.13 seconds and flow-map totals 1111.81/1109.68/1106.53 seconds. These are approximately 19.02 and 18.49 minutes per student respectively, including the recorded training/validation workflow. They exclude pretrained-teacher training, prior validation/ablation exploration, final evaluation, and surrounding process startup; they are not pure GPU-kernel compute times.

## Defensible final conclusion

The three frozen student seeds confirm useful acceleration of conditional probabilistic forecasting. The eight-step Itô map runs about seven times faster than SI64 while incurring roughly 1–2% higher proper scores, smaller ensemble spread, and worse final spectral discrepancy over ten physical leads. A one-step Itô map improves on a 16-step SI reference, but its longer-rollout underdispersion is substantial. The deterministic one-step flow map is especially competitive and prevents claiming that stochastic maps are generally superior. The evidence supports a practical speed/quality tradeoff for the proposed conditional map framework, with explicit limits from fixed teachers, one PDE benchmark, three student seeds and twenty held-out trajectories.
