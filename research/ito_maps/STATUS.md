# Handover status

Updated 2026-10-07 after the full frozen three-seed confirmation completed. All training and evaluation jobs finished successfully; no GPU job remains. The final manuscript includes the complete results and builds to14 total pages, with main text/impact ending on page8 and appendices starting on9. Numerical claims and key PDF pages were independently reviewed.

## Branch and authority

Branch: `research/ito-map-forecasting`. The user authorized extensive delegated research, manuscript development, implementation, tests and experiments. Root owns final integration/commits. All work remains local; no conference submission or publication has occurred. Do not start competing GPU jobs.

## Complete

- Full-text research on the three requested papers, conditional derivations, related-work/novelty review, and documented implementation decisions.
- Official ICML2026 template/guidelines checked and manuscript building; the intended future-cycle requirements remain unverified.
- `BaseMapModel`, `FollmerItoMapModel`, `FlowMapModel`, compatible two-time/path conditioning adapters, and exact finite-query weighted-Brownian bridges.
- `MapTrainer`: teacher-diagonal PSD and supervised same-path teacher-rollout objectives, offline artifacts, EMA, and tested epoch-boundary resume.
- Hydra train/test entry points and NS configurations, reusing existing datasets, preprocessing, U-Net, losses/optimizer/clipping and metrics.
- Latest whole-suite run:105 tests pass (CPU), including provenance, confirmation audits, paired seed reporting and the nontrivial-mask guard. The single warning is inherited CUDA autocast disabling during the explicitly CPU-only test run.
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

## Completed confirmation and principal findings

The frozen serialized sequence `experiments/run_confirmation.sh` completed with exit0. Root session53576 is closed. All six students (two families, seeds0/1/2) completed2,000 PSD plus500 teacher-rollout updates. All12 training stages are archived under `results/training/` with resolved configurations, histories, initialization lineage and selected checkpoint hashes. Original teacher pretraining was shared and not rerun.

`results/confirmation_audit.json` passed: all16 evaluation files,4,360 forecast records, exact cases/budgets/ensemble sizes, finite metrics,12 training-stage archives, and the four original frozen model/config hashes. Each student's matching standalone and rollout lead-one metrics agree exactly. The original frozen manifest remains unchanged.

The held-out protocol uses20 trajectories180–199 and16 ensemble members. One-step forecasts cover origins5/45/85; autoregressive forecasts start at45 and generate10 new snapshot intervals. Seed0 includes full budget curves (maps1/2/4/8, solvers1/4/16/64/128); all three student seeds have map budgets1/8. Rollout solver references use64 steps. Test scores were not used to change the frozen settings.

Across three independently trained students, with the same fixed teacher pair:

| Comparison | Student CRPS mean ± training-seed SD | Fixed teacher CRPS | Relative CRPS change |
|---|---:|---:|---:|
| Itô8, one step, versusSI64 |0.27139 ±0.00168|0.26780|+1.34%|
| FlowMap1, one step, versusFM64 |0.26663 ±0.00076|0.26501|+0.61%|
| Itô8, physical lead10, versusSI64 |0.54222 ±0.00148|0.53293|+1.74%|
| FlowMap1, physical lead10, versusFM64 |0.53353 ±0.00048|0.53308|+0.08%|

One-step ensemble latency averages0.3401s forItô8 versus2.4345s forSI64 (7.16x faster), and0.04349s forFlowMap1 versus2.4491s forFM64 (56.32x faster), on one RTX3090. These are measured ensemble-generation times, not end-to-end forecast scoring times. NFE counts actual network calls.

Against the moderate16-step SI budget, one-step Itô maps improve mean CRPS by4.73% at13.24x lower latency. The higher-step reference remains the principal comparison. The efficiency result is accompanied by residual distributional errors. Itô8 one-step RMS spread is10.64% belowSI64, and its physical-lead10 CRPS excess has a paired trajectory-bootstrap interval wholly above zero. Spectral discrepancy also worsens at later leads. The deterministic flow map is a strong comparator; the experiment does not establish an Itô advantage or distributional equivalence. Data-trajectory bootstrap intervals and student-seed SD/range are separate uncertainties; no48-member ensembles are formed by pooling seeds.

The full interpretation is in `notes/heldout_three_seed_assessment.md`; implementation/test hashes are in `notes/final_validation.json`. Detailed results are under `results/final_report/`, including full seed0 curves, all physical leads, persistence, per-seed values and fixed-teacher comparisons. The fixed qualitative example is test trajectory180, origin45, first member; no visual selection was performed. Development-only compression/K16 evidence remains labeled separately.

## Reproduce and verify completed artifacts

From the repository root:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
.venv/bin/python research/ito_maps/experiments/verify_confirmation.py
bash research/ito_maps/experiments/rebuild_confirmation_report.sh
make -C research/ito_maps/paper replication-artifacts
make -C research/ito_maps/paper
```

The report and audit commands use saved artifacts and do not run models. Large dataset/checkpoint files and the generated PDF remain local and ignored by Git; metadata, configs, numerical results, scientific figures and source are tracked. Do not rerun `run_confirmation.sh` over these archives merely to rebuild tables. The original teachers/data were preserved.

## Remaining human research and submission decisions

The user has not supplied the archive's original generation settings (snapshot spacing, viscosity, forcing, script/revision). Continue reporting physical leads in stored-snapshot intervals. Teacher-pretraining histories/costs are also unavailable. Broader external distillation baselines, additional datasets and larger physical conditional reference ensembles are not part of the completed experiment. These limits constrain ICML positioning; they are documented rather than replaced by unsupported claims.

The manuscript uses the official ICML2026 anonymous template. The intended future-cycle author instructions must be checked when available. Human authors must review scientific claims, authorship and substantial AI-assistance disclosure before any submission. No submission, push or publication has occurred.

## Important scientific and compatibility limits

- Repository default FSI gamma diffusion differs from the strict KL-optimal corrected Föllmer process. Maps here preserve the former.
- Stochastic diagonal target excludes the derivative of sqrt(t); supplied-teacher runs supervise against frozen teacher drift.
- Teacher rollouts add exact integrated generative noise plus Euler drift; the existing solver baseline uses Euler–Maruyama. State this distinction.
- Finite Brownian features can remove same-path information; exact queried Gaussian increments do not make the learned map exact.
- The original NS archive's physical timestep/forcing provenance is incomplete; lead times are stored snapshot intervals.
- The new map API is compatible with the unmasked normalized2D NS setup. Nontrivial masks and general parameter-conditioned trajectory inputs require further work; see `notes/map_api_compatibility.md`. Root authorized a post-freeze guard rejecting nontrivial masks in MapTrainer, plus API docstrings. The frozen mask1 NS numerics are unchanged; see `notes/post_freeze_api_guard.json` for source hashes and test evidence.
- Close related work includes StocBench, Recursive Flow Matching and Swift. Claims must concern the conditional path-preserving adaptation and measured quality/cost tradeoff, not first few-step scientific forecasting.
