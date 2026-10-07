# Handover status

Updated 2026-10-07 after configuration freeze and launch of the serialized confirmation sequence. Read the live log before starting work: this snapshot records a running experiment, not its completion.

## Branch and authority

Branch: `research/ito-map-forecasting`. The user authorized extensive delegated research, manuscript development, implementation, tests and experiments. Root owns final integration/commits. All work remains local; no conference submission or publication has occurred. Do not start competing GPU jobs.

## Complete

- Full-text research on the three requested papers, conditional derivations, related-work/novelty review, and documented implementation decisions.
- Official ICML2026 template/guidelines checked and manuscript building; the intended future-cycle requirements remain unverified.
- `BaseMapModel`, `FollmerItoMapModel`, `FlowMapModel`, compatible two-time/path conditioning adapters, and exact finite-query weighted-Brownian bridges.
- `MapTrainer`: teacher-diagonal PSD and supervised same-path teacher-rollout objectives, offline artifacts, EMA, and tested epoch-boundary resume.
- Hydra train/test entry points and NS configurations, reusing existing datasets, preprocessing, U-Net, losses/optimizer/clipping and metrics.
- Latest completed whole-suite run:105 tests pass (CPU), including provenance, confirmation audits, paired seed reporting and the nontrivial-mask guard.
- Real GPU training/evaluation smoke and all validation-only development comparisons.
- Both seed0 PSD stages (2,000 updates each), followed by both teacher-rollout stages (500 updates each). Frozen teacher grids use32 drift substeps during supervised training/validation.
- Analytic/Monte Carlo OU compression experiment and actual frozen-NS conditional-bridge diagnostic (CPU).
- Controlled K4→K16 initialization and matching500-update K16 refinement ablation, plus64/128-step solver validation comparisons.
- Lightweight reproducibility archives under `results/training/`, including selected weight hashes and cumulative student-stage lineage/cost.

## Frozen selection and actual validation findings

The immutable pre-test manifest is `results/frozen_protocol.json` (freeze timestamp2026-10-07T07:42:18Z). Its historical `test_forecasts_started:false` records the state when frozen; it is not a live progress indicator.

Selected Itô representation:K4. The K16 ablation reduced the information-theoretic teacher path bottleneck in the separate diagnostic but did not improve trained one-step forecasts: validation CRPS0.308603 versus K4 0.304729, with greater latency. This negative result is retained. Do not change the selected feature count after viewing test data.

Validation-only evidence: after teacher-rollout refinement, one-step Itô CRPS0.304729 and eight-step0.294158; deterministic flow-map one-step0.288475 and eight-step0.293202. The deterministic map is stronger at one-step generation. Initial PSD and refinement histories, matched validation cases and paired trajectory differences are in `notes/validation_iteration_log.md`. No unsupported claim of stochastic-map superiority or fully simulation-free final training is warranted.

The NS conditional-path diagnostic at one validation history found approximately9.68% omitted teacher endpoint variance withK4 and0.66% withK16. It establishes a finite-feature same-path information limit for that teacher/history, not a guarantee that training exploits the added information or improves forecast scores.

## Live serialized confirmation job

Root tool session: **53576**. Script: `experiments/run_confirmation.sh`. Live log: `results/confirmation.log`. One RTX3090 is reserved for this sequence. Other agents should use CPU only. Sessions may be owned by the root tool context; inspect the log/process state rather than launching a duplicate job if the session is unavailable to a child agent.

The sequence is:

1. Seed0 forecasts on all20 held-out trajectories180–199, origins5/45/85, ensemble16. Maps use1/2/4/8 evaluations; existing SI/FM solvers use1/4/16/64/128 steps. Outputs: `results/test_one_step/`.
2. Seed0 autoregressive10-new-step physical forecasts on those20 trajectories from origin45, ensemble16. Maps use1/8 evaluations per physical step; solvers use64. Outputs: `results/test_rollout/`.
3. Saved qualitative forecast comparison, with outputs in `results/test_qualitative/`.
4. Complete independent student training for seeds1 and2, both model families and both stages, using unchanged frozen settings. Script: `experiments/train_replications.sh`. Teacher pretraining is shared and not rerun.
5. Seeds1/2 held-out evaluation at map budgets1/8 for the same one-step and10-step physical-forecast protocols, followed by provenance archiving.

At launch, root estimated roughly40minutes for initial seed0 evaluation and about75minutes for additional training, with further evaluation afterward; these estimates are not completion guarantees. The live log and output artifacts are authoritative.

### Latest completed test evidence (still seed0 only)

All one-step curves are complete. See `notes/heldout_seed0_assessment.md` and
`results/final_report/one_step/`. Itô8 CRPS0.27052 versusSI64 0.26780 at7.15x
lower latency; Itô1 CRPS0.27287 versusSI16 0.28802 at13.30x lower latency.
FlowMap1 CRPS0.26656 versusFM64 0.26501 at56.3x lower latency. Proper-score
proximity does **not** establish calibration retention: Itô8 has10.66% less
RMS spread thanSI64 and5.19percentage points less empirical coverage. Both
map families remain underdispersed relative to their high-step teachers.

All seed0 rollouts and the fixed qualitative example have completed. At physical lead10,
Itô8 CRPS0.54316 versusSI64 0.53293; full10-lead ensemble generation costs
3.386 versus24.597seconds. FlowMap1 lead10 CRPS0.53396 versusFM64 0.53308.
The full seed0 report is in `results/final_report/`; the manuscript builds to13
total pages with main text ending on8. Itô8's lead10 spectral discrepancy is
15.8% higher thanSI64, so broad physical-quality retention is also unproven.
Additional full training seeds remain pending. Both seed1 models have completed
and been archived. Seed2 Itô PSD reached1,700/2,000 updates at11:14 local time;
its refinement and both seed2 flow stages precede evaluation of all repetitions. Do not
present seed0 as a three-seed result. Root verified every map's initial rollout
forecast against its matching standalone origin45 forecast, with identical
proper scores/calibration metrics for all40 cases per map.

An optional user question is pending about the archive's original generation
settings (snapshot spacing, viscosity, forcing and creation script/revision).
Continue using stored-snapshot lead units unless actual provenance is supplied.

## Remaining work after the sequence

- Confirm every frozen run completed, inspect failures/nonfinite values, and archive all training metadata/history/configurations/hashes with `experiments/archive_training.py`. Run `experiments/verify_confirmation.py`, then `experiments/rebuild_confirmation_report.sh`, then `make -C research/ito_maps/paper replication-artifacts`.
- Aggregate measured test quality, calibration, spectra, latency and long-rollout behavior. Distinguish trajectory uncertainty from independently trained seed variation; no test-driven checkpoint or hyperparameter changes.
- Include the negative K16 ablation, conditional compression experiment and actual computational costs; pre-existing teacher pretraining cost remains unavailable.
- Reconcile manuscript and figures/tables against completed raw artifacts, compile the paper, review numerical/mathematical claims, and run final relevant checks.
- Root makes final commits and provides the user with the paper/code/results and remaining submission limitations. Manuscript submission is not authorized by this work.

## Important scientific and compatibility limits

- Repository default FSI gamma diffusion differs from the strict KL-optimal corrected Föllmer process. Maps here preserve the former.
- Stochastic diagonal target excludes the derivative of sqrt(t); supplied-teacher runs supervise against frozen teacher drift.
- Teacher rollouts add exact integrated generative noise plus Euler drift; the existing solver baseline uses Euler–Maruyama. State this distinction.
- Finite Brownian features can remove same-path information; exact queried Gaussian increments do not make the learned map exact.
- The original NS archive's physical timestep/forcing provenance is incomplete; lead times are stored snapshot intervals.
- The new map API is compatible with the unmasked normalized2D NS setup. Nontrivial masks and general parameter-conditioned trajectory inputs require further work; see `notes/map_api_compatibility.md`. Root authorized a post-freeze guard rejecting nontrivial masks in MapTrainer, plus API docstrings. The frozen mask1 NS numerics are unchanged; see `notes/post_freeze_api_guard.json` for source hashes and test evidence.
- Close related work includes StocBench, Recursive Flow Matching and Swift. Claims must concern the conditional path-preserving adaptation and measured quality/cost tradeoff, not first few-step scientific forecasting.
