# Interpreting the initial pixel-space video results

Prepared 2026-10-07, **before any held-out evaluation**. At this writing, no `results/{dataset}_protocol.json` or `results/test/**/evaluation.json` exists. The intended settings below are already specified in [STATUS.md](../STATUS.md) and the [freeze script](../experiments/freeze_video_protocol.py); the immutable checkpoint/case freezes will be created after training. This document is a reporting scaffold, not a results claim.

## Scope of the evidence

The operational objective is to train all four model families from dataset-specific video teachers, produce autoregressive forecasts, and compute reproducible metrics and examples. The study uses pixels and the most recent fixed N frames, as requested. It does not reproduce the reference's VQGAN representation, random historical conditioning, training compute, or FVD sample counts. CLEVRER is additionally reduced to 64² from the reference implementation's 128². These changes must appear next to results, rather than only in a footnote. [Primary-source audit](paper_video_protocol.md).

The intended experiment has one training seed, 20,000 updates for each newly initialized solver teacher, and 2,000 progressive plus 500 rollout-distillation updates for each map. Evaluation uses 128 fixed test clips and four completions: KTH 10→30 frames; CLEVRER 2→14 at stride five. Maps use 1/8 steps and solvers 16/64. Persistence uses the same cases. A successful run establishes that the method operates on these datasets; it does not establish convergence, state of the art, or reproducibility across training seeds.

## Evidence available now: validation only

The completed KTH stochastic-interpolant validation audit contains only **four clips**, each with four completions and 30 predicted frames. Its source-video-weighted averages are:

| Solver steps/frame | Mean fair CRPS ↓ | Mean LPIPS ↓ | Mean PSNR ↑ | Mean SSIM ↑ | Complete-ensemble generation seconds ↓ |
|---|---:|---:|---:|---:|---:|
| 16 | 0.04577 | 0.3716 | 21.40 | 0.4619 | 3.053 |
| 64 | 0.02283 | 0.1752 | 25.29 | 0.7458 | 12.40 |

These numbers come from the [saved validation report](../results/validation/kth_si_report/kth/README.md). They establish finite end-to-end rollouts and substantial sensitivity to solver resolution on these four cases. They do **not** show map quality, test generalization, solver convergence at 64 steps, or comparative performance on CLEVRER. Four-real-clip FVD is especially unstable; its numerical values should remain diagnostic rather than headline evidence. The untrained 100-update smoke checkpoint is not a research baseline.

At lead 30, this validation teacher's spread/skill ratios are approximately 0.60 and 0.66 for 16/64 steps. This is a reason to retain ensemble diagnostics and inspect motion over time, not to conclude calibration from four clips. Both proper scores and perceptual metrics matter: static backgrounds can make aggregate pixel scores look favorable while moving objects deteriorate.

## Fair comparisons in the final report

- **Compare each map with its own teacher**, at both solver budgets. Also compare the two map families directly, and include persistence. Keep every predefined budget; do not select the best test-time budget and present it as if predetermined.
- Report mean-over-lead quality and final-lead quality together. Plot all leads to expose accumulation of error. Measure generation seconds for the complete E=4 rollout, and actual backbone evaluations. Ratios of solver cost to map cost quantify acceleration; step ratios alone do not.
- Treat map training as teacher training **plus** distillation. Report the recorded costs separately. An inference speedup does not imply cheaper training or immediate amortization.
- Report pooled FVD (128 real/512 generated) and member FVD (128/128, mean and completion-index SD) with these counts. FVD includes the common observed prefix. It assesses a collection of completed clips, not conditional calibration for an individual history. Do not compare its scale across KTH/CLEVRER or against the reference table.
- Frame scores weight original source videos equally; FVD weights selected clips equally. Preserve this distinction, especially for KTH, where several annotated segments can share an original AVI. Source-video bootstrap intervals describe uncertainty over sampled sources, not training seeds or newly sampled people.
- Fair CRPS/energy, RMSE and spread use unbounded unit-pixel forecasts. PSNR/SSIM/LPIPS and I3D use clipped copies. Show out-of-range frequency. Mean-completion metrics are primary; oracle best-member scores are supplemental and depend mechanically on ensemble size.
- Four-member empirical 5–95% coverage is **not expected to equal 90% even for calibrated draws**. Do not label its distance from 90% a calibration error. Read coverage alongside spread, accuracy and member variability; these are limited-ensemble diagnostics.
- Show the fixed first selected cases and members, including weak forecasts. Inspect moving foregrounds, color retention, motion continuity, and late-frame degradation. Selected illustrations do not establish correct collision physics or coherent action dynamics across the dataset.

## Language for positive, mixed, or negative outcomes

**If quality remains close while cost falls:** “At [budget], the map changed [metric] by [absolute and relative difference] versus its [solver budget] teacher while reducing measured rollout time by [factor].” Use “close” descriptively with exact differences; an interval crossing zero is not a proof of equivalence. No equivalence margin was prespecified.

**If only some metrics improve:** “The map was faster and improved [metrics], while [metrics or later leads] deteriorated.” A lower FVD does not override worse proper scores, and close CRPS does not establish equal diversity or perceptual quality. Report the deterministic map's strengths even if the intended stochastic-map narrative is weaker.

**If maps lose quality:** “Few-step sampling reduced cost but incurred [observed losses] in this pixel-space pilot.” Preserve these results. Possible explanations—teacher quality, distillation budget, finite path features, or autoregressive error—are hypotheses unless tested independently. Do not attribute a failure to one mechanism from this experiment alone.

**If every learned model loses to persistence:** State that result directly. The operational pipeline can still be complete, but these training budgets have not yet demonstrated useful forecasting gains. Avoid using fast but weak forecasts as evidence for the main scientific claim.

After audits succeed, append a dated summary with links to the frozen protocols, result tables and fixed visualizations. Fill in actual model-specific comparisons only then; retain this pre-test scaffold as evidence of the intended interpretation rules.
