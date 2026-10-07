# Validation-only iteration log

## Initial PSD pilot (2026-10-07)
Hypothesis: frozen-teacher diagonal anchoring plus same-path PSD can learn useful few-step maps without trajectory targets during training. Teacher rollouts are used for checkpoint selection only.

Fixed design:2000updates, batch8, all160training trajectories, widths16/32/64/128, four Brownian features for Itô; full interval coverage after500updates; consistency ramp100updates; EMA0.95; AdamW1e-4 cosine to1e-5; held-out teacher32-step validation. Forecast development subset: validation trajectories160:164, origins5/45/85, ensemble8. No test data used.

Itô result:8maps CRPS0.3056 vs16-stepSI0.3177, measured0.212s vs0.375s per ensemble (1.77x). All4trajectory-average CRPS differences favor8maps. Compared with64-stepSI CRPS0.2830, the8map CRPS remains8.0% worse on average and worse on each trajectory. One-map CRPS0.3661, insufficient forecast spread. These are pilot estimates without a general superiority claim.

Interpretation: useful finite-budget acceleration but incomplete distillation. One and two maps have similar poor scores, suggesting off-diagonal learning may be a bottleneck. Four-feature compression could also matter; this result alone does not distinguish causes.

Next planned controlled intervention: same-path numerical teacher-rollout supervision, initialized from the PSD checkpoint. Keep4features first to isolate loss choice.500updates, teacher32substeps, endpoint probability0.5, no interval curriculum, learning rate5e-5 to1e-5. Apply same stage to deterministic flow map. If residual spread loss persists, refine Brownian features4→16 with a function-preserving weight transfer and compare under matched training. These choices are made before evaluating held-out test trajectories.
