# Assessment: first 2,000-update Itô PSD pilot

Date: 2026-10-07. Validation only; no test-set claim. Sources: `../results/validation_psd/ito_psd_seed0/evaluation.json`, `../results/validation_baselines/{stochastic_interpolant,flow_matching}/evaluation.json`, and `checkpoints/stochastic_navier_stokes/ito_psd_seed0/history.json` from repository root.

The comparison uses four shared validation trajectories (160–163), three forecast origins (5,45,85), one physical forecast interval, eight ensemble members, and one trained student seed. The averages below are over twelve forecast contexts. They do not establish a population-quality ranking or multi-step stability.

| Sampler | Steps | CRPS ↓ | RMSE ↓ | Energy score ↓ | Mean pixel spread | Coverage90 | Relative enstrophy error ↓ | Ensemble latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Itô PSD | 1 | 0.36607 | 0.55878 | 0.44954 | 0.18851 | 0.31882 | 0.41158 | 0.02830 |
| Itô PSD | 2 | 0.36728 | 0.55973 | 0.45090 | 0.18741 | 0.31635 | 0.41170 | 0.05433 |
| Itô PSD | 4 | 0.34428 | 0.54611 | 0.42257 | 0.22588 | 0.38488 | 0.39253 | 0.10700 |
| Itô PSD | 8 | 0.30565 | 0.52693 | 0.37561 | 0.32477 | 0.54596 | 0.32677 | 0.21175 |
| SI | 4 | 0.41682 | 0.62010 | 0.51115 | 0.18075 | 0.27423 | 0.56628 | 0.09588 |
| SI | 16 | 0.31773 | 0.53413 | 0.38980 | 0.29263 | 0.49227 | 0.38058 | 0.37544 |
| SI | 64 | 0.28302 | 0.50651 | 0.34911 | 0.37958 | 0.63232 | 0.28006 | 1.45656 |
| FM | 16 | 0.29599 | 0.50088 | 0.36276 | 0.28528 | 0.50478 | 0.27386 | 0.37529 |
| FM | 64 | 0.28653 | 0.51331 | 0.35204 | 0.38631 | 0.62958 | 0.28137 | 1.46098 |

## What the measurements establish

Eight-step Itô maps improve mean CRPS by about 3.8% versus the 16-step SI baseline at about 1.77 times lower latency. The paired trajectory-mean CRPS differences (Itô8 − SI16) are −0.01103, −0.01767, −0.01336, and −0.00627: the direction agrees on all four trajectories. This is encouraging evidence of a finite-budget acceleration benefit.

Against SI64, Itô8 CRPS is about 8.0% worse, and the paired differences are positive on all four trajectories (+0.02071, +0.01715, +0.02479, +0.02786). The one-step Itô model is about 29.3% worse in CRPS, has roughly half SI64's mean pixel spread, and has markedly lower interval coverage. Its lower enstrophy indicates smoothing or lost variance. A claim of similar-quality one-step forecasting is not supported.

Coverage from only eight empirical ensemble members has substantial finite-ensemble limitations; SI64 itself is not calibrated to a nominal 90% band in these measurements. Interpret the relative degradation alongside rank and spread diagnostics rather than assign all miscalibration to the student.

The validation objective is a fixed-noise, simulation-based teacher-map error, although training is PSD. It drops from 0.3432 after 100 updates to 0.0351 at 1,300 updates and 0.03368 at 2,000 updates. Improvement slows considerably over the last 700 updates. The final cosine-decayed learning rate is near 1e-5.

## Focused next methodological step

The pilot does not identify whether insufficient off-diagonal training, PSD self-consistency bias, or Brownian compression dominates. Two map steps barely change one-step quality; additional shorter transitions eventually recover spread. This suggests an off-diagonal problem but does not uniquely diagnose its cause.

The recommended next controlled step is a short teacher-rollout-supervised run with K=4, initialized from the current PSD checkpoint, using a fresh optimizer and a documented finite teacher budget (e.g. 16 or 32 solver steps). This introduces an independent target to test whether self-distillation has settled on an inaccurate transition. It must be described as simulation-based distillation and its teacher computation counted. Repeating another long PSD run at the already-decayed learning rate is less diagnostic.

If direct teacher supervision cannot recover teacher-map accuracy or spread, expand the Brownian feature resolution to K=16. The analytic linear-SDE diagnostic demonstrates that four features can lose considerable path-dependent variance for strongly contracting dynamics, but this is not proof that the Navier–Stokes teacher exhibits that bottleneck. A K16 run should be treated as an ablation, not an assumed fix.

For cheap controlled initialization of K16 from K4: the current 1×1 path projection has weights W of shape [hidden,4*C,1,1]. Reshape to [hidden,4,C,1,1], repeat each feature's coefficient for its four child increments, and flatten to [hidden,16*C,1,1]. Do not divide by four: the four raw child martingale increments sum to their coarse parent. All other weights remain identical. On one shared Brownian path, this expansion preserves the original predictions up to floating-point arithmetic before optimization. Optimizer state should restart, and the ablation must disclose its warm start and cumulative training budget.

A deterministic flow-map pilot with the same training budget is pending. Its result is needed before selecting the strongest affordable map baseline or making a general claim about stochastic versus deterministic map quality.
