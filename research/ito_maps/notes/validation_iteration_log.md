# Validation-only iteration log

## Initial PSD pilot (2026-10-07)
Hypothesis: frozen-teacher diagonal anchoring plus same-path PSD can learn useful few-step maps without trajectory targets during training. Teacher rollouts are used for checkpoint selection only.

Fixed design:2000updates, batch8, all160training trajectories, widths16/32/64/128, four Brownian features for Itô; full interval coverage after500updates; consistency ramp100updates; EMA0.95; AdamW1e-4 cosine to1e-5; held-out teacher32-step validation. Forecast development subset: validation trajectories160:164, origins5/45/85, ensemble8. No test data used.

Itô result:8maps CRPS0.3056 vs16-stepSI0.3177, measured0.212s vs0.375s per ensemble (1.77x). All4trajectory-average CRPS differences favor8maps. Compared with64-stepSI CRPS0.2830, the8map CRPS remains8.0% worse on average and worse on each trajectory. One-map CRPS0.3661, insufficient forecast spread. These are pilot estimates without a general superiority claim.

Interpretation: useful finite-budget acceleration but incomplete distillation. One and two maps have similar poor scores, suggesting off-diagonal learning may be a bottleneck. Four-feature compression could also matter; this result alone does not distinguish causes.

Next planned controlled intervention: same-path numerical teacher-rollout supervision, initialized from the PSD checkpoint. Keep4features first to isolate loss choice.500updates, teacher32substeps, endpoint probability0.5, no interval curriculum, learning rate5e-5 to1e-5. Apply same stage to deterministic flow map. If residual spread loss persists, refine Brownian features4→16 with a function-preserving weight transfer and compare under matched training. These choices are made before evaluating held-out test trajectories.

## Frozen teacher-rollout refinement (2026-10-07)

Completed the planned controlled change of training objective: both students received 500 additional teacher-rollout updates, initialized from their respective 2,000-update PSD checkpoints. Each update uses 32 teacher substeps. The source PSD cost therefore remains part of each method's cumulative cost; these are not standalone 500-update models. All development forecasts use the same four validation trajectories, three origins, and eight ensemble members as the initial pilot.

| Sampler | Map steps | CRPS | RMSE | Energy score | RMS spread | Coverage90 | Relative enstrophy error | Ensemble latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Itô rollout, K4 | 1 | 0.304729 | 0.522163 | 0.374230 | 0.327195 | 0.533203 | 0.331910 | 0.028116 |
| Itô rollout, K4 | 2 | 0.305364 | 0.517983 | 0.374982 | 0.309680 | 0.512868 | 0.322267 | 0.054167 |
| Itô rollout, K4 | 4 | 0.307306 | 0.511168 | 0.377039 | 0.280373 | 0.476176 | 0.328514 | 0.106685 |
| Itô rollout, K4 | 8 | 0.294158 | 0.513650 | 0.361303 | 0.355409 | 0.575877 | 0.302084 | 0.211590 |
| Flow rollout | 1 | 0.288475 | 0.508376 | 0.354322 | 0.366065 | 0.589315 | 0.273240 | 0.027011 |
| Flow rollout | 2 | 0.293018 | 0.501112 | 0.359845 | 0.312079 | 0.525487 | 0.273795 | 0.051392 |
| Flow rollout | 4 | 0.298635 | 0.496598 | 0.366132 | 0.272482 | 0.469864 | 0.279795 | 0.100316 |
| Flow rollout | 8 | 0.293202 | 0.501647 | 0.359491 | 0.314711 | 0.525553 | 0.274114 | 0.198360 |

The same-path teacher target substantially improves one-step Itô CRPS (0.36607 to 0.30473) and spread. Itô1 now beats SI16 (0.31773) on each trajectory, but remains worse than SI64 (0.28302) on each trajectory: paired differences are +0.024973, +0.007882, +0.031537, +0.022446. Itô8 closes part of that gap (mean CRPS 0.29416) while remaining worse on all four trajectories. Thus changing the supervision resolves a substantial optimization/self-consistency limitation, but does not yet deliver matched SI64 quality.

The deterministic map is currently the stronger one-step forecast baseline: flow1 CRPS 0.28848 is close to FM64's 0.28653, with paired differences +0.001718, +0.007397, +0.000724, −0.002058. Itô1 minus flow1 differences are +0.029665, +0.032920, −0.002696, +0.005131. There is no empirical basis here to claim that the stochastic map is superior to the deterministic map. More map steps do not monotonically improve either student, underscoring that trained any-step behavior must be measured rather than assumed.

The final teacher-map validation losses are 0.019374 (Itô K4) and 0.005784 (flow). Additional stage wall times were approximately 521.8 and 514.0 seconds, respectively. Frozen teacher-map validation compares different reference dynamics between these models, so the numerical loss values alone are not a cross-model forecast-quality ranking.

Artifacts are archived in `research/ito_maps/results/training/{ito_rollout_seed0,flow_rollout_seed0}`: resolved configuration, run metadata, complete stage history, and selected model checkpoint SHA256. Forecast evidence is in `research/ito_maps/results/validation_rollout`.

## Conditional Brownian-information diagnostic (2026-10-07)

Before choosing the K16 ablation, a frozen-teacher experiment measured information lost by conditioning on finite Brownian features. It uses validation trajectory 160 at origin45 only, eight coarse path groups, four independently refined bridges per group, and 64 Euler drift substeps with exact weighted-noise increments. It ran on CPU in 679.7 seconds. No neural student, training updates, or test trajectories were involved.

| Features K | Mean within-group endpoint variance | Omitted variance fraction estimate | Ideal retained standard deviation fraction |
|---:|---:|---:|---:|
| 1 | 0.0559570 | 0.284279 | 0.846003 |
| 4 | 0.0171126 | 0.096847 | 0.950343 |
| 16 | 0.00105534 | 0.006558 | 0.996715 |

This is direct evidence that four features leave measurable same-path information unresolved for this teacher/history, and sixteen resolve much more of it. The within-group variance falls by about 16.2 times from K4 to K16, with a reduction in every one of the eight coarse groups. It motivates a controlled K16 trial. However, the ideal conditional-mean standard-deviation reduction at K4 is only around 5% in this diagnostic, so compression alone does not account for the much larger initial PSD spread loss. Teacher approximation and optimization remain relevant.

The percentages use finite-sample hierarchical variance estimates. Their total-variance denominators fluctuate between 0.1609 and 0.1968 across feature counts; the true unconditional teacher variance is invariant to conditioning, and this variability reflects limited Monte Carlo sampling. A single-history experiment does not establish an average irreducible error over the dataset. See `notes/ns_path_compression_assessment.md` for the estimator and limitations; raw metadata and per-group values are in `results/ns_path_compression/results.json`.

Next intervention: K16 initialized from the same PSD K4 source with a function-preserving raw-increment projection expansion, then the same 500 teacher-rollout updates as K4. This isolates additional path information while retaining equal training initialization and update budget. K16 forecast results are pending; test-set evaluation remains reserved until method selection is frozen.

## K16 outcome and pre-test freeze (2026-10-07)

The controlled K16 refinement completed and did not improve forecast quality: one-step validation CRPS0.308603, compared with0.304729 for K4; eight-step CRPS0.295606 versus0.294158. One-step latency increased from0.02812s to0.03264s per ensemble. Both variants used the same PSD source function, 500 additional teacher-rollout updates and matched forecast cases. These measurements do not invalidate the conditional-compression diagnostic: additional available information need not be learned successfully under a fixed training budget. K4 was retained by the predeclared validation-CRPS selection rule.

Configuration/checkpoint/source hashes and the complete confirmation protocol were frozen at2026-10-07T07:42:18Z in `results/frozen_protocol.json`, before the held-out test sequence began. Further seed1/2 runs use unchanged settings. Test forecasts and independent training replications are now serialized under root session53576; see `STATUS.md` and `results/confirmation.log` for progress. There is no test-informed feature-count change.

A subsequent public-API audit prompted a root-authorized constructor guard rejecting unsupported nontrivial masks during map training. The selected NS mask is identically one, so frozen loss/sampling arithmetic is unchanged. Parameter-trajectory limitations were documented without changing behavior. The original frozen manifest remains immutable; `notes/post_freeze_api_guard.json` records the exact source-hash changes and targeted test evidence.
