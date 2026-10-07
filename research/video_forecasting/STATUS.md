# Pixel-space video forecasting handover

Branch: `research/video-map-forecasting`, branched from the completed Navier–Stokes study. User requests KTH and CLEVRER, all four model families, newly trained baselines, full pixel space and the last N consecutive frames as context. No latent autoencoder or randomly chosen conditioning frame is used.

## Active work

- **Active evaluation:** root session85180 runs both datasets through `evaluate_all_video_models.sh`; main log `results/evaluation_pipeline.log`. KTH SI both budgets are complete; KTH flow matching is active. All eight models and both pre-test freezes are complete. GPU-exclusive evaluation is required.
- **Pending NS queue:** shell PID2126968 (session32086) is SIGSTOPped. Both map children have completed4096 leads and released the GPU. Watcher session5222 (`results/ns_resume_watch.log`) waits for video process2150609 to exit, verifies both passed dataset audits and the NS process identity, then automatically sends SIGCONT to2126968 for both4096-step solver baselines. If the watcher fails, inspect its log and resume manually only after video GPU work ends. Do not leave this user-requested work pending at task completion.
- Root coordinates implementation, GPU jobs, integration and commits. Only root launches GPU jobs. Additional user-requested NS stability pipeline runs in root session32086 concurrently with video TRAINING; NS GPU work must finish or pause before timed video evaluation; inference timing will run with exclusive GPU access. See `../ito_maps/notes/long_horizon_protocol.md` and `../ito_maps/results/long_horizon/` logs.
- video_data completed both official acquisitions and preprocessing. Canonical manifests: `data/video_forecasting/kth_64/manifest.json` (2,391 segments from599 source videos) and `data/video_forecasting/clevrer_64/manifest.json` (20,000 videos; official10,000/5,000/5,000 splits). All23,391 decoded-file SHA-256 hashes, shapes, source-video/subject split separation and CLEVRER raw-frame indices passed the full audit; `notes/data_summary.json` records manifest/archive hashes. All CLEVRER clips have26 retained RGB frames. Watcher session88878 exited0; no acquisition jobs remain. About30GB disk remains. One recoverable KTH source-video decode defect is documented in `notes/data_provenance.md`.
- video_models owns baseline training/configuration and CPU evaluation review/tests.
- video_protocol owns primary-source protocol, pretrained metrics and reporting/visualization.
- All six KTH training stages completed successfully: both teachers20,000 updates, both maps2,000 PSD plus500 refinement. Root GPU session40828 runs `train_all_video_models.sh clevrer`; both CLEVRER teachers have completed20,000 updates, and both Itô stages are complete; FlowMap PSD is currently training. Main log `results/clevrer_pipeline.log`, stage logs in `results/`. Independent CPU review confirmed finite strictly loadable teacher checkpoints,200-epoch/20,000-update histories matching restart states, seed0 random initialization, RGB/N2 input contracts, and correct official train/validation splits. KTH selected-artifact review and pre-test freeze are complete. No held-out test forecasts have been generated yet. The earlier100-update GPU smoke is separate and is not a reported model.

- KTH protocol frozen at2026-10-07T11:48:11Z in `results/kth_protocol.json`; independent `--verify-only` passed. Six lightweight stage archives are in `results/training/kth_<stage>`. All completed checkpoints loaded strictly on CPU with finite tensors, matching restart histories and correct own-teacher/PSD lineage. Frozen evaluation uses128 eligible clips from83 source videos, covering all six actions and all five held-out subjects. Total recorded training/validation stage cost is2,686.99 seconds. Model/configuration, manifest, selected decoded clips and pretrained metric weights are hashed. Run `freeze_video_protocol.py --dataset kth --verify-only` before later evaluation; do not recreate or overwrite the freeze.

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
- Training wall times are recorded operational costs, not controlled training-speed comparisons: NS stability sampling began sharing the GPU around14:11 local time on2026-10-07, affecting the final portion of CLEVRER flow-matching training and subsequent map stages. See `notes/training_runtime_context.md`. Video inference timings will exclude concurrent NS GPU work.
- Save first fixed cases/members, without choosing images by quality. These are initial operational results, not a claim of convergence or reproduced paper performance.

## Current next steps

Current whole-suite check:195 tests passed (one expected CPU autocast warning). Complete source/test integration, finish CLEVRER map students serially, freeze selected evaluation artifacts, compute all metrics, inspect visual rollouts, prepare matched reports and reproducibility archives, then commit. Preserve all original NS data/checkpoints/results. Do not restart or overwrite active runs.
