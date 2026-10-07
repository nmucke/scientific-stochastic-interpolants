# Itô maps for probabilistic forecasting

Original confirmation branch: `research/ito-map-forecasting`; current extension branch: `research/video-map-forecasting`. Started2026-10-07. Root leads research/integration and owns final commits. [STATUS.md](STATUS.md) records the original frozen confirmation. For current video results and active long-horizon fluid jobs, read [the extension status](../video_forecasting/STATUS.md).

## Objective and research discipline

Adapt conditional stochastic-interpolant forecasting ([Chen et al.](https://arxiv.org/abs/2403.13724)) to Brownian-path-conditioned Itô maps ([Pan et al.](https://arxiv.org/abs/2606.11156)), with deterministic [flow-map matching](https://arxiv.org/abs/2406.07507) as a comparator. Reuse the existing architecture, datasets, normalization and training conventions. The research question is whether comparable forecast distributions can be produced with fewer evaluations and lower measured latency. Preserve negative and inconclusive findings.

Main research and the first manuscript draft preceded implementation. Keep exact Gaussian path queries distinct from finite-feature/neural approximations, preserve trajectory-level splits, and select configurations on validation only. Existing teacher training histories are incomplete and their pretraining cost must remain labeled unavailable.

## Completed frozen confirmation

Implementation and validation-only development are complete. All frozen held-out confirmations are also complete. The selection is frozen in [results/frozen_protocol.json](results/frozen_protocol.json), including checkpoint/config/source hashes. K4 was selected: controlled K16 refinement yielded worse validation one-step CRPS0.3086 versus0.3047 forK4. The deterministic flow map currently has stronger one-step validation quality. This is part of the result, not grounds to change the comparison after testing.

The serialized [confirmation sequence](experiments/run_confirmation.sh) completed successfully for all three student seeds in both model families. The [confirmation audit](results/confirmation_audit.json) verifies all4,360 forecast records,16 evaluation artifacts,12 training stages and original frozen checkpoint hashes. That original confirmation queue is complete. All20 held-out trajectories were evaluated with ensemble16, one-step forecasts at origins5/45/85, and10-new-step autoregressive forecasts from origin45.

The [final report](results/final_report/final_report_manifest.json) separates training-seed variability from trajectory-bootstrap uncertainty. Eight-step Itô maps have mean one-step CRPS0.27139 versus0.26780 forSI64 (+1.34%) at7.16x lower measured latency; one-step deterministic maps have0.26663 versus0.26501 forFM64 (+0.61%) at56.32x lower latency. The maps remain underdispersed, and later-lead Itô score/spectral degradation prevents a claim of full quality retention. The deterministic map is a strong baseline, not evidence of universal Itô superiority.

The integrated extension suite currently reports212 passing CPU tests (the original confirmation suite had105). The nontrivial-mask constructor guard and API documentation do not alter the frozen unmasked NS numerics; its source-hash addendum is [notes/post_freeze_api_guard.json](notes/post_freeze_api_guard.json).

## Video and long-horizon extension

All eight pixel-space KTH/CLEVRER models are trained, and all18 dataset/model/budget comparisons (including persistence) are evaluated and independently audited. See the [matched report](../video_forecasting/results/report/README.md) and [video handover](../video_forecasting/STATUS.md). The paper includes the video case, FVD table and matched truth/prediction frame sequences in its seven-page main text; both cases retain detailed appendices. The video findings show metric-dependent quality costs alongside faster sampling.

The 4,096-interval fluid extension has completed both maps; matched SI64 and FM64 references remain in the active serialized queue. Spatial enstrophy uses shell-summed Fourier power over space, with separate first/last-step log–log plots. Skill stops at the95 available truth leads. [Finalization instructions](notes/finalize_long_horizon.md) describe the independent audit and four-model report once both references finish. Do not describe the partial two-map preview as a completed four-model comparison.

The [independent mathematical review](reviews/adversarial_method_math_2026-10-07.md), [response](reviews/adversarial_method_math_response_2026-10-07.md), and [compiled video-paper inspection](reviews/final_video_pdf_inspection.md) record assumptions, limitations and validation. The final NS evidence will require another manuscript build and visual inspection.

## Layout

- `notes/`: derivations, methodological decisions, code/API audits, validation iteration history and venue requirements.
- `sources/`: primary paper downloads and extracted texts.
- `paper/`: current LaTeX manuscript, bibliography and official ICML2026 template.
- `experiments/`: reproducible training/evaluation/diagnostic/reporting scripts.
- `results/`: raw measurements, frozen protocol, run provenance and generated figures/tables.

The paper compiles with the official ICML2026 kit. That conference cycle has passed; the future intended submission cycle's requirements must be rechecked. No conference submission or public release has occurred.

## Confirmed environment and data

- Project `.venv`: Python3.13.9 and PyTorch2.9.1+cu128. System Python lacks project dependencies.
- One NVIDIA RTX3090 with24GiB. Run one GPU worker at a time.
- Dataset `data/stochastic_navier_stokes/data.npz`: key `state`, float32, shape `(200,100,128,128)`.
- Independent trajectory split: training0–159, validation160–179, test180–199.
- Five history fields; equal base/target normalization scale3.09969.
- Existing teachers: `checkpoints/stochastic_navier_stokes/{stochastic_interpolant,stochastic_interpolant_small,flow_matching}`.
- `pdflatex`, `latexmk` and `pdftotext` available.

## Implemented interface and boundaries

The new classes are `BaseMapModel`, `FollmerItoMapModel`, `FlowMapModel`, `MapTrainer` and `MapConditionedNetwork`. Train/test entry points are `scisi.bin.main_map_model_train` and `scisi.bin.main_map_model_test`. Existing NS datasets, preprocessing, U-Net and metrics are reused. `MapTrainer` supports PSD and an explicitly simulation-based teacher-rollout stage, offline checkpointing, and epoch-boundary resume.

Map sampling takes normalized2D histories and returns normalized CPU tensors. Use keyword arguments. `sample_trajectory` includes the supplied history in its requested total length. General parameter-conditioned rollouts, nontrivial masked training, arbitrary diffusion schedules and solver posterior wrappers are not drop-in supported. See [notes/map_api_compatibility.md](notes/map_api_compatibility.md) for exact tensor contracts and restrictions. The frozen NS configuration is inside the supported scope.

## Handover reading order

1. [STATUS.md](STATUS.md): completed evidence, audit/rebuild commands and remaining scientific limits.
2. [notes/methodology.md](notes/methodology.md) and [notes/flow_maps_and_code_design.md](notes/flow_maps_and_code_design.md): source papers, conditional construction and implementation choices.
3. [notes/methodology_implementation_audit.md](notes/methodology_implementation_audit.md), [notes/code_equation_audit.md](notes/code_equation_audit.md), and [notes/map_api_compatibility.md](notes/map_api_compatibility.md): equations, finite-feature caveats and supported usage.
4. [notes/validation_iteration_log.md](notes/validation_iteration_log.md): chronological decisions made before test exposure.
5. [notes/ns_path_compression_assessment.md](notes/ns_path_compression_assessment.md): actual teacher information diagnostic and its one-history/finite-teacher limits.
6. [notes/experiment_protocol.md](notes/experiment_protocol.md), [results/frozen_protocol.json](results/frozen_protocol.json), and [notes/navier_stokes_provenance.md](notes/navier_stokes_provenance.md): evaluation and data provenance.
7. [notes/heldout_three_seed_assessment.md](notes/heldout_three_seed_assessment.md) and [the compiled paper](paper/main.pdf): complete results and interpretation.
8. [notes/broader_literature.md](notes/broader_literature.md), [paper/guidelines.md](paper/guidelines.md), and [paper/main.tex](paper/main.tex): novelty boundary and evolving manuscript.
9. Executed scripts, archived configurations and raw results: verify actual completion, source checkpoint hashes and cumulative training cost before reporting outcomes.

## Useful reproducibility commands

Run these from the repository root with the project interpreter. Training/evaluation scripts require the appropriate scheduled device; the following archive/CPU diagnostics do not compete for the GPU.

```bash
.venv/bin/python research/ito_maps/experiments/archive_training.py \
  --checkpoint-dirs checkpoints/stochastic_navier_stokes/ito_rollout_seed0 \
  checkpoints/stochastic_navier_stokes/flow_rollout_seed0

.venv/bin/python research/ito_maps/experiments/brownian_compression.py \
  --mc-samples 100000
```

The archiver stores metadata/history/resolved config and selected model/EMA hashes without copying large weights. It follows initialization/refinement ancestry and includes all recorded student stages in cumulative cost. Analytic OU outputs are under `experiments/results/brownian_compression`; actual frozen-NS compression results are under `results/ns_path_compression`. Neither should be confused with held-out forecast quality. Smoke artifacts demonstrate execution only.
