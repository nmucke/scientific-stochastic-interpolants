# Video forecasting protocol: source audit and pixel-space adaptation

Audited 2026-10-07. This note separates the published experiment, available implementation, and the present project's deliberate changes. It is not a claim of numerical reproduction.

## What Chen et al. actually report

The reference uses a VQGAN representation, a previous-frame point-mass base for the stochastic model, and Gaussian-base flow matching as comparator. Conditioning includes the latest frame and a randomly selected earlier frame with its time index. Predictions are autoregressive. KTH uses grayscale 64 × 64 frames, subjects 21–25 for testing, and 10 observed followed by 30 generated frames. The reported quantitative video metric is FVD: 256 real test clips versus 100 completions per clip (25,600 generated clips). Their 250,000-update training uses AdamW, initial learning rate 2e-4, and approximately 1–2 days on four A100s per model. The paper also reports a VQGAN reconstruction FVD and subtracts it to obtain “shifted FVD.” [Paper, §4.3 and Appendix C.3](https://arxiv.org/html/2403.13724v2).

PSNR, SSIM, LPIPS, CRPS, energy scores, and calibration are **additional diagnostics in this project**, not metrics claimed to be in the reference's video table. A pixel model has no VQGAN reconstruction correction: report ordinary FVD, not a manufactured shifted score.

## Primary implementation audit

| Source | Pinned revision | Findings relevant to reproduction |
|---|---|---|
| [interpolants/forecasting](https://github.com/interpolants/forecasting) | `a5e4d8fbf2979a3b932fabb4995fa574edbe3a49` | Its README sends video work to a separate repository. |
| [marikgoldstein/video_interpolants](https://github.com/marikgoldstein/video_interpolants) | `5c087bbf9d424379888798d778cb1007286deb1b` | KTH is runnable in principle; the CLEVRER path asserts false. Published FVD computation is absent. The README labels preprocessing unfinished. |
| [Araachie/river](https://github.com/Araachie/river) | `9ecde339e4c19713485fc1ab31bcb524d6772f6e` | Complete dataset configurations clarify CLEVRER's temporal and spatial settings; its refinement evaluator includes the useful pretrained FVD detector wrapper. |

The Chen demo preprocessing is unsuitable as an authoritative split: its draft handles only jogging/running/walking, while its accompanying README puts scenarios d1/d2 in training, d3 in validation, and d4 in testing. These are explicitly demo choices and differ from the paper's person-disjoint test set. Preserve all six actions and follow the paper's test subjects. [Demo preprocessing README](https://github.com/marikgoldstein/video_interpolants/blob/5c087bbf9d424379888798d778cb1007286deb1b/work_in_progress/README.md), [draft converter](https://github.com/marikgoldstein/video_interpolants/blob/5c087bbf9d424379888798d778cb1007286deb1b/work_in_progress/avi_to_pngs.py).

The Chen config sets KTH to 40 consecutive frames and 10 conditioning frames. Its unreachable CLEVRER branch sets 16 sampled frames, two conditioning frames, 14 predictions, and `skip_frames=4`, but has inconsistent 64/128 spatial declarations. Consequently the CLEVRER resolution cannot be established reliably from that branch alone. [Chen configuration](https://github.com/marikgoldstein/video_interpolants/blob/5c087bbf9d424379888798d778cb1007286deb1b/configs.py).

RIVER's CLEVRER config uses RGB 128 × 128, 16 sampled frames, two context frames and 14 predictions. The loader turns `skip_frames=4` into stride **five**, requiring 76 raw frames for one 16-frame sample. It resizes the shorter spatial edge to 128 and then center-crops 128 × 128. KTH uses 64 × 64, 40 frames, stride one, 10 context and 30 predictions. Both loaders normalize uint8 pixels to [-1,1]. [CLEVRER config](https://github.com/Araachie/river/blob/9ecde339e4c19713485fc1ab31bcb524d6772f6e/configs/clevrer.yaml), [KTH config](https://github.com/Araachie/river/blob/9ecde339e4c19713485fc1ab31bcb524d6772f6e/configs/kth.yaml), [loader](https://github.com/Araachie/river/blob/9ecde339e4c19713485fc1ab31bcb524d6772f6e/dataset/video_dataset.py).

## Dataset provenance and split decisions

CLEVRER's official release contains 20,000 clips: indices 0–9999 train, 10000–14999 validation, 15000–19999 test. Each has 128 RGB frames at 480 × 320 and lasts about five seconds. The official README specifies CC0. Use these existing splits; the question-answer labels and object annotations are unnecessary for unconditional pixel forecasting given past frames. [Official data README](https://data.csail.mit.edu/clevrer/README.txt), [download page](https://clevrer.csail.mit.edu/).

For KTH, the reference establishes subjects 21–25 as the test set, but does not specify a validation partition. A defensible project choice is subjects 1–18 train, 19–20 validation, 21–25 test. Label this held-out validation decision explicitly. Do not use the common forecasting split 1–16/17–25 while calling it the Chen split. Preserve action, subject, scenario, source filename and clip boundaries in the manifest. The official website was unavailable to this audit (HTTP 503); record the actual download source and checksums separately. [KTH official site](https://www.csc.kth.se/cvap/actions/).

Never split sliding windows from one source video across train, validation, and test. For KTH, multiple evaluation windows from the same original AVI are correlated and must share a video/subject grouping identifier for uncertainty estimates.

## FVD implementation and pretrained weights

Use the **400-dimensional pre-softmax I3D outputs from the Google FVD model**, not generic 1024-dimensional I3D features or an R3D model renamed FVD. The original Google implementation uses the `deepmind/i3d-kinetics-400/1` TF-Hub model and bilinear resizing to 224 × 224; it extracts `RGB/inception_i3d/Mean:0`. [Google reference implementation](https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py).

A practical PyTorch path exists in the primary RIVER and StyleGAN-V repositories. Their detector is [the published TorchScript conversion](https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1). StyleGAN-V explicitly distinguishes its converted TF-Hub weights from the different original I3D-repository weights and notes a spatial-upsampling difference from TensorFlow. [StyleGAN-V FVD](https://github.com/universome/stylegan-v/blob/master/src/metrics/frechet_video_distance.py), [RIVER detector wrapper](https://github.com/Araachie/river/blob/9ecde339e4c19713485fc1ab31bcb524d6772f6e/refinement/fvd.py).

Recommended contract:

1. Save the detector locally, record URL, byte count and SHA-256, and put it in evaluation mode. Fail clearly if unavailable; never substitute random features.
2. Input layout is `[batch, 3, time, height, width]`. Replicate KTH grayscale into three identical channels. For input already normalized to [-1,1], use `rescale=False, resize=True, return_features=True`; do not normalize twice.
3. Include observed context followed by generated future in the video embedding, and use the corresponding complete real clip. This is our declared convention because the released Chen code does not expose the published FVD evaluator. It retains 16 frames for CLEVRER, sufficient for temporal I3D processing. Evaluate pixel/perceptual prediction errors on **future frames only**.
4. Record full-clip length, stride, real/generated counts, detector hash, resizing/backend, and clipping policy alongside every FVD. Cache real features and retain generated features so metrics can be recomputed without resampling.
5. Compute the Gaussian Fréchet distance in float64. A symmetric positive-semidefinite square-root formulation avoids silently discarding a large imaginary component from a nonsymmetric covariance product. Rank-deficient covariances with small samples are valid algebraically but yield noisy, biased estimates.
6. If only a small pilot set is used, label the value `pilot FVD`, always show sample counts, and do not compare its magnitude with the reference table. For final comparisons, fix the same cases and completion count for every model and budget. Report a matched-one-completion FVD as a separate diagnostic if desired; do not conflate it with the reference's asymmetric 256-vs-25,600 experiment.

No new trainable model weights are needed for MSE, MAE, PSNR, SSIM, pixel CRPS, or energy scores. LPIPS needs pretrained feature and calibration weights; use the official [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) implementation with an explicitly recorded network (for example AlexNet) and version. This is optional beyond the reference's required FVD, but useful if all image-quality diagnostics are requested. Pretrained metrics are feature evaluators, never forecasting teachers.

## Present project's deliberate changes

The user's requested method is pixel-space autoregression conditioned on a fixed rolling window of the last N frames. At forecast time, append each generated frame and remove the oldest context frame. No future truth enters the rolled-out history. At training time, use consecutive context-target windows; random selection of a training window is compatible with the user's instruction, which excludes randomly choosing the **conditioning frame**.

Use the existing Föllmer-named interpolant and flow-matching models as newly trained teachers, and distill the existing Itô map and deterministic flow map from their own dataset-specific teacher. The point-mass source is always the latest frame, even though the network sees all N context frames. The flow model uses its Gaussian source and the identical context. Map training shares Brownian paths across composition terms as in the completed Navier–Stokes implementation. No pretrained Navier–Stokes weights should be described as video teachers.

Suggested initial protocol: KTH N=10, horizon=30, 64² grayscale; CLEVRER N=2, horizon=14, stride=5. A 64² RGB CLEVRER pilot is computationally convenient but deviates from RIVER's 128² image space. Pixel space does not mean retaining the original camera resolution. Specify the actual resize/crop decision; preserving the full field by direct square resizing differs from the source's short-edge resize plus center crop.

Use one fixed evaluation manifest per dataset containing source video and raw start-frame indices. Freeze budgets, cases, completion count, selected checkpoint and preprocessing before evaluating test data. Start with tractable pilot counts while exposing a configuration for 256 cases and 100 completions. Preserve enough storage headroom by streaming feature extraction and writing only selected fixed qualitative clips plus scalar/per-case metrics; there is no need to store every full-resolution completion.

Report FVD, future-only per-lead PSNR/SSIM (and LPIPS when installed), ensemble-mean RMSE, fair CRPS and fair energy score, spread/coverage, generation latency and actual backbone evaluation count. Separate mean-over-completions metrics from best-of-E metrics; best-of-E improves mechanically with E and is not a proper forecast score. State whether range clipping is performed before scoring and whether clipped frames are fed back. Prefer unmodified dynamics internally with explicitly clipped display/image-metric copies; record the fraction outside the valid pixel range.

The first outcome is an executable, audited four-model benchmark with actual pilot results, not a claim of convergence or reproduction of the paper's much larger compute budget. Compare map costs with both their teacher costs and teacher-plus-distillation training expense. A weak teacher may make an apparently successful distillation scientifically uninformative; show teacher quality, persistence baselines, visual rollouts and distributional diagnostics together.
