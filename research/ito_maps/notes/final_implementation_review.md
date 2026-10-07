# Final implementation and scientific-claim review

Reviewed on 2026-10-07 while the frozen three-seed confirmation was still
running. This was an independent read-only CPU review of model, adapter,
Brownian, trainer, entry-point and manuscript code. No training numerics,
evaluation settings or checkpoints were changed, and no GPU jobs were started.

## Finding

No substantive correctness blocker was found for the frozen, unmasked,
full-state, equally normalized, unconditioned Navier–Stokes experiment.
This finding does not certify every configuration that the legacy framework
can express, statistical equivalence of forecast distributions, or ICML
acceptability.

The following implementation-to-manuscript contracts agree:

- The Föllmer-named model uses the repository forecasting SDE, with
  `gamma(t)=a*(1-t)` and the Wiener interpolant drift target containing
  `gamma'(t)*sqrt(t)*z`. It does not differentiate `sqrt(t)` or claim to
  implement the distinct KL-optimal diffusion.
- `WeightedBrownianPath` supplies exact finite-dimensional Gaussian laws in
  the variance clock through cached conditional bridges. Local interval
  features and the explicitly added martingale increment use the same object.
  Compression of the drift's path dependence remains an approximation.
- Composition targets use detached EMA parameters, retain the intermediate
  martingale displacement, and cancel the total additive noise only when
  forming the average-drift target. Interpolant starting states and future
  Brownian paths are independent as required for a conditional transition.
- The interval sampler includes an endpoint atom and expands the maximum
  interval during the first-stage curriculum. The frozen second stage uses
  the full interval range, with probability 0.5 on exactly `[0,1]`.
- A supplied teacher replaces the raw-data diagonal target. The refinement
  target uses 32 left-Euler drift steps and exact additive martingale
  increments; it remains a discretized teacher target. Neither stage's
  validation loss certifies the true conditional forecast distribution.
- Flow maps use independent Gaussian initial states, matching the inherited
  flow-matching construction. Removing stochastic increments yields the
  common deterministic map algebra; it does not turn a deterministic map of
  a fixed observed state into a probabilistic forecaster.
- Teacher backbone transplantation preserves the original diagonal drift:
  the additional time-gap and Brownian branches vanish on the diagonal.
- Forecast evaluation uses only past truth for the initial history, then
  shifts each member's generated history. Scoring RNG cannot affect later
  physical forecasts. Timings include path construction, and NFE counts only
  forecast backbone calls. Scores are compared on matched contexts and
  ensemble sizes.

The tests already cover the relevant algebra, nontrivial Gaussian bridge
covariances, conditional forks, real-UNet transplantation, teacher-gradient
isolation, analytic drift integration, history evolution, future-truth
exclusion, metric definitions, and full-state epoch-boundary resume. This
review did not rerun unchanged tests; the root agent owns the final complete
suite. The confirmation verifier was separately tested on seven synthetic
artifact scenarios and has not been run against incomplete live results.

## Supported API boundaries

The detailed contract remains in `map_api_compatibility.md`. In particular,
maps are a separate `BaseMapModel` abstraction with two-time forward calls;
they require `MapTrainer` and are not drop-in solver/guidance objects.
Single-step static field and vector conditioning are supported, whereas the
inherited trajectory parameter convention only supports a scalar future-time
series. Caller code must reconstruct tendencies or differing target/history
coordinates. `MapTrainer` rejects nontrivial masks. The present Brownian
driver supports the affine scalar diffusion schedule, not arbitrary
state-dependent or spatially correlated generative diffusion.

The entry point checks teacher/student interpolation class and diffusion
amplitude, but these checks alone are not a general proof of teacher
compatibility for arbitrary custom models, altered source conventions or
interpolation flags. The shipped frozen configurations satisfy the full
construction. Extending to such settings requires explicit compatible
teacher dynamics rather than relying only on the existing shape checks.

## Reviewer checklist to close after all three seeds finish

1. Run `experiments/verify_confirmation.py` successfully. Preserve its JSON
   record of the original four frozen model/config hashes, all 16 evaluation
   artifacts, exact cases and budgets, matched standalone/rollout lead-one
   scores, and all 12 training-stage archives with own-seed 2000-to-500 lineage.
2. Run the final CPU test suite and rebuild all paper artifacts from the
   completed raw evaluations. Check PDF compilation and ICML format/page
   requirements separately; those checks belong to the paper workflow.
3. Replace every pending-replication statement with the actual three-seed
   results. Report per-seed outcomes as well as summaries. Keep resampling of
   test trajectories distinct from variability over the three student seeds;
   three student seeds do not replicate teacher training or data generation.
4. Retain the conclusion supported by the complete evidence: reduced sampling
   cost with measured score/calibration tradeoffs. A small CRPS difference or
   an interval crossing zero is not a prespecified equivalence test. Preserve
   the strong deterministic baseline and any persistent uncertainty deficit.
5. Keep one-step and ten-lead results distinct, including accumulated latency
   and final spectral error. Do not turn ten observed stable leads into a
   general stability or invariant-distribution claim.
6. Keep development evidence labeled: the PSD-to-refinement comparison adds
   500 updates and teacher simulation; the K4/K16 comparison has matched
   ancestry and update budget but different realized random streams; the
   nested Brownian-compression diagnostic covers one history, not all NS
   contexts. These are explanatory diagnostics, not independent confirmation.
7. Retain the unresolved archive-generation provenance and unavailable
   teacher-pretraining cost. The current simulator configuration is not
   certified metadata for the supplied state array. Report lead time in
   stored snapshots and student distillation cost with its actual scope.

The final mathematical contribution should remain an accessible conditional
adaptation and documented empirical study. The Brownian bridge construction,
conditional-expectation variance decomposition, same-coupling Wasserstein
bound and Gaussian score illustration are established facts, not novel
theorems. Submission strength depends on the completed evidence and the
positioning relative to prior forecasting and map-distillation work.

## Completion evidence after three-seed integration

All frozen runs subsequently completed on 2026-10-07. The live
`results/confirmation_audit.json` reports success at
`2026-10-07T09:57:11.806064+00:00`: 4,360 raw forecast rows, 16 evaluation
artifacts, 12 complete training stages, and 40 exact standalone/rollout
lead-one matches for each of the six students. The four original frozen
model/config hashes passed. The root agent reports the final complete CPU
suite passed **105 tests**, with only the inherited CUDA-disabled warning.

After the paper agent signaled integration complete, I independently
recomputed all **72** `generated/replication_claims.tex` numeric macros from
the raw evaluation rows. These included per-seed and across-seed score
means, sample standard deviations, calibration values, all relevant paired
10,000-resample trajectory intervals, relative score changes, and cumulative
rollout latency ratios. Every displayed number matched its specified
rounding. All `generated/replication_sources.json` hashes matched their
current source files. The generated replication tables agree with those
values and the archived training histories.

Key independently checked results (means over student seeds 0, 1, 2):

| Comparison to fixed 64-step teacher | CRPS mean | Student-seed SD | Relative CRPS change | Measured latency ratio |
| --- | ---: | ---: | ---: | ---: |
| Itô map, 8 steps, one interval | 0.27139388 | 0.00168421 | +1.34303% | 7.15903 |
| Itô map, 8 steps, physical lead 10 | 0.54221859 | 0.00147725 | +1.74276% | 7.25918 |
| Flow map, 1 step, one interval | 0.26662567 | 0.00075751 | +0.60893% | 56.31716 |
| Flow map, 1 step, physical lead 10 | 0.53353189 | 0.00048451 | +0.08420% | 58.92186 |

Ratios at physical lead 10 use total latency through all ten leads. Positive
CRPS changes mean worse point estimates than the corresponding teacher.
The flow map's one-interval paired lower endpoint is
`0.000011517461389295456`, correctly reported as positive rather than an
interval crossing zero. Its final-lead interval crosses zero, with no
equivalence claim.

I also checked the manually stated three-seed coverage references (74.89%
and 73.78%), the Itô map's final spectral-discrepancy increase (14.90%),
the flow teacher's final RMS spread/skill ratio (1.056), and the reported
training-time ranges (18.99–19.05 minutes for Itô students and 18.44–18.53
minutes for flow students). All agree with their sources.

The integrated `main.tex` and `results.tex` contain no stale pending-seed
claims. They retain explicit seed-zero labels for complete budget curves,
separate training-seed variability from trajectory intervals, and preserve
the uncertainty, deterministic-baseline, data-provenance and compute-cost
limitations. No numerical or scientific-claim inconsistency was found.
The paper agent separately reports a clean build with main content ending
on page eight and the appendix beginning on page nine; that layout check
was not repeated in this numerical review.
