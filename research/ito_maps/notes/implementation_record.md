# Implementation and validation record

2026-10-07; initial manuscript compiled before implementation began.

## Implemented interfaces
- `BaseMapModel`: common two-time map, batched sampling, physical autoregression.
- `FlowMapModel`: deterministic map with random Gaussian forecast initialization.
- `FollmerItoMapModel`: latest-observed-field initialization, shared weighted Brownian path and explicit martingale displacement.
- `MapConditionedNetwork`: wraps original U-Net scalar-time embedding and input/history block; adds zero-initialized interval and path branches. Every original teacher parameter is copied with explicit shape/key checking. Teachers are not stored in student checkpoints.
- `WeightedBrownianPath`: exact finite-query Gaussian bridge construction for gamma(t)=a(1-t), local equal-subinterval path increments. Seeds identify query sequences, not query-order-independent Brownian trees.
- `MapTrainer`: Trainer subclass, teacher/raw diagonal anchoring, PSD or numerical teacher rollout, EMA targets, interval curriculum, offline checkpoints. With a teacher, validation always uses fixed-randomness full-interval teacher targets independent of curriculum.
- Hydra train/test entrypoints and shared Navier–Stokes configs.

## Scope decisions
- Support current affine gamma and Wiener-process interpolation explicitly; reject incompatible stochastic interpolation schedules.
- U-Net reused without modifying its implementation. Dropout set to zero for map training so consistency targets do not include arbitrary dropout noise.
- Retain dataset's existing final-window omission for comparison with teacher training. Fix its existing `preprocesser=None` dictionary bug narrowly.
- Preserve existing normalized field coordinates and train/val/test trajectory splits.
- Validation loader shuffled with a fixed isolated RNG seed so a bounded validation budget covers trajectories instead of only the earliest windows of trajectory160.
- No external tracking service is enabled by default.

## Completed execution checks
- Existing metrics: 23 tests passed before new code.
- Initial combined suite: 55 tests passed, including model invariants, exact Brownian covariance/composition, map trainer gradients, frozen teacher, offline checkpointing, and real U-Net transplant.
- Real 128x128 Navier–Stokes GPU train smoke: 2 optimization batches with batch2, 2 training and2 validation trajectories; output `checkpoints/stochastic_navier_stokes/ito_smoke`. Best validation objective 0.0930367. This is an execution check, not evidence of forecast quality.
- Real test entrypoint smoke at map steps1,2, ensemble4, one validation history completed; output `results/smoke`. It is deliberately untrained and must not be used for conclusions.

## Active experiments
Validation solver baselines are running before student selection. Results, exact command, checkpoint hashes and raw trajectory-level scores are saved by `experiments/compare.py`.
