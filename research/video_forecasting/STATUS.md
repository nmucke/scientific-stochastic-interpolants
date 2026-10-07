# Pixel-space video forecasting handover

Branch: `research/video-map-forecasting`, branched from the completed Navier–Stokes study. User requests KTH and CLEVRER, all four model families, newly trained baselines, full pixel space and the last N consecutive frames as context. No latent autoencoder or randomly chosen conditioning frame is used.

## Active work

- Root coordinates implementation, GPU jobs, integration and commits. Only root launches GPU jobs.
- video_data owns official acquisition/preprocessing and manifests. KTH is complete. CLEVRER official train/validation/test downloads and automatic decoding are active in data-agent session88878; inspect `data/video_forecasting/clevrer_prepare.log` and ask that agent before restarting.
- video_models owns baseline training/configuration and CPU evaluation review/tests.
- video_protocol owns primary-source protocol, pretrained metrics and reporting/visualization.
- The full KTH SI baseline completed20,000 updates and a four-case validation rollout audit. Root GPU session87640 now runs `train_all_video_models.sh kth`: KTH flow matching20,000 updates, then both map pipelines. Main log `results/kth_pipeline.log`; per-stage logs are in `results/`. The earlier100-update GPU smoke completed separately under `checkpoints/video/smoke/kth_si` and is not a reported model.

## Implemented

`VideoFrameDataset` lazily reads uint8 THWC arrays and supplies the existing [C,H,W,N] history/base/target convention, with fixed [-1,1] scaling. All valid training windows are available without materializing them in memory. KTH uses subject1–18 training,19–20 validation,21–25 test, all six actions and official segment annotations. CLEVRER uses official10,000/5,000/5,000 splits and stride5.

`LocalTrainer` inherits the existing regression objective and adds offline artifacts, deterministic validation, bounded epochs and restart support. `main_video_model_train` dispatches map runs to the existing MapTrainer. Eight dataset/model configs share a64² UNet with widths16/32/64. N=10 and horizon30 for grayscale KTH; N=2 and horizon14 for RGB CLEVRER. CLEVRER64² short-side resizing and center cropping is an explicitly initial pixel-space deviation from the128² reference setup.

`metrics/video.py` implements fair pixel CRPS/energy, ensemble accuracy/calibration, PSNR/SSIM/LPIPS and400-dimensional pretrained Google-converted I3D FVD. Detector source/hash and primary-paper details are in `notes/`. Actual pretrained LPIPS and I3D weights are downloaded; random feature substitutes are prohibited.

Root's `main_video_model_test` performs autoregression without future truth or feedback clipping. Proper scores retain out-of-range unit-pixel predictions. Image/perceptual scores and I3D clip their own copies and record out-of-range frequency. Sampling finishes before scoring for each clip. Saved numerical records, features and preselected qualitative cases permit later reporting without resampling.

## Initial experiment protocol (before held-out model evaluation)

- One training seed0 per dataset/model; all teachers newly trained from random initialization.
- Baselines:20,000 updates, batch16, AdamW2e-4, cosine to2e-5, full precision. Fixed-noise validation chooses the checkpoint; no test-based selection.
- Maps: initialize from own selected teacher,2,000 progressive self-distillation updates; then500 teacher-rollout refinement updates with32 teacher drift steps. K=4 for Itô, same shared-path method as the NS implementation. Stage2 retains diagonal weight1, full interval sampling with endpoint atom0.5, AdamW5e-5 decaying to1e-5.
- Evaluation:128 seeded, fixed eligible test clips per dataset, four completions each. KTH10→30, CLEVRER2→14 at stride5. Map budgets1/8; solver budgets16/64. Persistence evaluated on identical cases. Final exact checkpoint hashes will be recorded before running tests.
- FVD includes the observed prefix. Report pooled FVD128real versus512generated, plus equal-count128-vs128 FVD for each completion index. Neither is the paper's256real/25,600generated benchmark. No VQGAN reconstruction correction applies.
- Future-only frame metrics, generation time and actual NFE are recorded at every lead. Bootstrap groups annotated segments by original source video; FVD completion-index SD is not a confidence interval.
- Save first fixed cases/members, without choosing images by quality. These are initial operational results, not a claim of convergence or reproduced paper performance.

## Current next steps

Current whole-suite check:160 tests passed; more provenance tests are being added. Complete source/test integration, train the remaining three teachers and all four map students serially, freeze selected evaluation artifacts, compute all metrics, inspect visual rollouts, prepare matched reports and reproducibility archives, then commit. Preserve all original NS data/checkpoints/results. Do not restart or overwrite active runs.
