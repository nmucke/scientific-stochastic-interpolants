# Final benchmark protocol audit

Audit date: 2026-10-07. This is a code/protocol audit; it contains no new experiment outcomes.

## Proposed frozen test protocol

- Select the model checkpoint and Brownian feature count using validation, including the planned K=16 ablation, before opening final test scores.
- One-step: all test trajectories 180–199, ensemble size 16, origins 5/45/85. Map budgets 1/2/4/8; differential-solver budgets 1/4/16/64/128.
- Autoregression: the same 20 test trajectories, origin 45, ten physical forecast leads, ensemble size 16. Map budgets 1/8 per physical transition; solver budget 64.
- Retain all predetermined cases and budgets. Use raw per-origin results, checkpoint hashes and fixed seeds. Do not choose figures or members from their quality.

## Verified autoregressive semantics

`evaluate_model` initializes conditioning from indices `origin-history_length:origin`. Lead `l` is compared with truth index `origin+l-1`. A history of length five is shifted after every prediction; the truth slice is used only for scoring. The planned origin 45 and ten leads require fields through index 54 and are within the stored Navier–Stokes trajectories.

The first call replicates the one observed history across the E ensemble members. Subsequent calls receive an E-sized history; neither BaseMapModel nor BaseModel repeats it again. Each member retains its own generated fields. The evaluator never replaces these histories with their ensemble mean, and never substitutes the true next field. This is a fixed-size ensemble of E evolving paths, not an exponentially expanding branching tree.

Each physical transition creates fresh generative noise. Flow-matching and flow-map forecasts begin from independent Gaussian initial states. Föllmer SI begins from each member's latest state and draws new SDE increments; the Itô map begins from each member's latest state and constructs a new Brownian driver. Reuse of a Brownian path is required within one pseudo-time map composition, not across distinct physical transition draws. These choices implement the learned conditional transition model autoregressively.

## Units and uncertainty scores

The existing Navier–Stokes checkpoints use the same zero mean and scalar standard deviation 3.09969 for historical/base and target fields. Thus a generated normalized target can be placed directly into the normalized history. This would need explicit conversion if base and target preprocessing differed; the comparison runner now checks checkpoint preprocessing and history length against the reference configuration.

Evaluation JSONs report normalized coordinates. For this scalar common normalization, physical CRPS, RMSE, energy score and either spread definition equal their normalized values multiplied by 3.09969. Coverage, rank counts/fractions, spread/skill ratios and relative spectral error are unchanged. The qualitative plot script explicitly inverse-transforms every generated member to physical coordinates by default before calculating its displayed mean and standard deviation.

CRPS and energy score use unbiased distinct-pair spread terms with denominator E(E−1). Their finite-sample realizations need not be nonnegative; do not clamp them or switch denominators after viewing results. Energy score additionally divides Euclidean norms by the square root of field dimension.

The existing 90% coverage uses empirical 5th/95th percentiles. With E=16, finite-ensemble quantiles are noisy and systematically narrower in expectation than exact population quantiles. Comparisons at identical E are meaningful, but interpreting 90% as an exact finite-E calibration target is misleading. For continuous exchangeable truth/member draws, the ideal expected extreme-rank fraction is 2/(E+1)=2/17; it complements coverage without changing the stored coverage definition.

RMS spread is sqrt(mean of unbiased ensemble variances), and RMS spread/skill includes sqrt((E+1)/E). The evaluator reports means of the per-case ratios; this is not the same statistic as the ratio of globally averaged spreads and errors.

## Timing and aggregation

Each row's `seconds` and `nfe` measures only the transition at that physical lead. Cost to reach lead ten is the sum of ten transition costs. Final report orchestration computes cumulative cost within each trajectory/origin/budget before averaging origins and resampling trajectories; it never labels a lead-ten single-transition time as full-rollout cost.

Model sampling returns CPU arrays, so elapsed time includes Brownian construction, neural evaluations, and output/history transfers. CUDA is synchronized around timing. Scoring and spectra are outside the timed region. NFE is measured by a backbone hook, excludes warm-up and represents E-member batched evaluations, not E times that count. Hardware and software metadata accompany new evaluations. Warm-up uses only one step; isolated cold-start anomalies remain possible and should be visible in raw timing rows.

For score summaries, average origins within each trajectory and bootstrap complete trajectories. For model differences, use the same trajectory resamples and subtract paired trajectory means. For lead-dependent plots, reuse one bootstrap index matrix across every lead, retaining their dependence. Pointwise 95% intervals are not simultaneous confidence bands and do not quantify variation across training seeds. Twenty trajectories support a much more useful audit than the earlier four-trajectory development pilots, but claims should still describe the available sample size.

## Limits retained deliberately

The evaluator is currently scoped to unconditional-field Navier–Stokes conditioning: it does not forward time-varying external field/parameter conditions from other datasets. This does not affect the proposed benchmark. No claim of identical Brownian sample paths across different NFE budgets is made: repeated seeds give common random seeds, while path query order differs. Distillation training cost and checkpoint provenance must accompany inference speed comparisons.

## Reporting and sanity-baseline entry points

`experiments/final_report.py` reads completed seed-0 one-step and rollout evaluation JSONs. It rejects missing cases/budgets and checkpoint changes between the two tasks. It creates the primary budget curves, per-lead tables and plots, cumulative rollout-cost curves, and predeclared paired CRPS comparisons. The SI16 and SI128 reference comparisons are explicitly listed as skipped in rollout reporting because that frozen rollout protocol evaluates SI64 only; they are never silently replaced by another reference.

The optional `--seed-one-step` and `--seed-rollout` arguments add complete student training seeds 1 and 2, evaluated at budgets 1 and 8. `training_seed_report.py` reads the training seed from checkpoint metadata/configuration, never the sampling seed. Per-student trajectory CIs, across-student sample standard deviation/range, and a trajectory-cluster CI for the seed-averaged score are separate fields. With only seed 0, student-seed standard deviation is explicitly unavailable. The teachers are fixed, so these quantities do not describe teacher-training variation.

`persistence_baseline.py` requires the actual freeze manifest before test evaluation and runs entirely on CPU. It reuses the evaluator/metrics, copies each path's latest field, and records zero neural evaluations. One-step invocation uses origins 5/45/85, E16 and 20 trajectories; the rollout invocation uses origin 45 and horizon 10. Optional `--persistence-one-step` and `--persistence-rollout` inputs add these sanity checks to final-report score tables/rollout curves. Their CPU bookkeeping times are excluded from generative-model speedup comparisons.

Example reporting command, substituting the actual evaluation files:

```bash
.venv/bin/python research/ito_maps/experiments/final_report.py \
  --one-step ONE_STEP_ITO.json ONE_STEP_FLOW_MAP.json ONE_STEP_SI.json ONE_STEP_FM.json \
  --rollout ROLLOUT_ITO.json ROLLOUT_FLOW_MAP.json ROLLOUT_SI.json ROLLOUT_FM.json \
  --persistence-one-step research/ito_maps/results/test_persistence/one_step/evaluation.json \
  --persistence-rollout research/ito_maps/results/test_persistence/rollout/evaluation.json \
  --output research/ito_maps/results/final_report
```

Add the extra-seed inputs to the same command after those experiments complete. The initial seed-0-only report remains valid as a clearly labeled single-student result. The larger SI128 reference is retained because SI64 should not be described as a fully converged solver reference.
