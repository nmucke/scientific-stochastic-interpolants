# Independent CLEVRER result review

Completed 2026-10-07 after all nine frozen model/budget evaluations and the
canonical combined report finished. No issues were found. The result is
`research/video_forecasting/results/clevrer_independent_audit.json`.

Reproduction from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  research/video_forecasting/experiments/independent_video_audit.py \
  --dataset clevrer \
  --report research/video_forecasting/results/report/clevrer/summary.csv
```

The CPU-only checker imports no production dataset, metric, or report code.
It verifies the complete frozen model/budget set before writing success. The
original KTH audit and the earlier partial RGB validation remain unchanged.

- The frozen seed independently reproduces the 128 selected cases from all
  5,000 eligible test clips. All 128 source array hashes and the manifest hash
  match. Training, validation, and test source clips are disjoint. Each selected
  case has a different source video.
- All 27 saved NPZ examples (three cases for each of nine configurations)
  reproduce original RGB uint8 truth pixels exactly. They retain two observed
  frames and fourteen future frames at 64². Independent scoring uses only
  retained-frame indices 2–15, corresponding to original raw indices
  10, 15, …, 75 with stride five.
- Independent NumPy fair CRPS, energy score, RMSE, spread, spread/skill,
  coverage, out-of-range fraction, PSNR, and independent Gaussian SSIM match
  all **378 saved future rows** to **7.11e-15** maximum absolute error.
  The checked cases are `video_17574`, `video_18641`, and `video_16984`.
- Every contact sheet reconstructs pixel-exactly from fixed case selection,
  the first two ensemble members, and future leads 1, 4, 8, 11, 14. The displayed
  examples are not chosen by forecast score.
- All nine real I3D feature arrays are exactly identical (128×400); generated
  arrays have shape 128×4×400. Independent dense-covariance eigendecomposition
  reproduces pooled FVD and each member FVD with maximum absolute discrepancy
  **7.49e-8**. Pooled comparisons use 128 real versus 512 generated clips;
  member comparisons use 128 versus 128. Full-clip FVD includes the two observed
  frames and fourteen generated frames, as declared by the production evaluator.
- Every canonical report point estimate, including LPIPS aggregation and
  generation cost, independently recomputed from all saved scalar rows agrees
  to **7.11e-15**. The 128 distinct source videos receive equal weights.

Protocol SHA256:
`677bbc6c610729173ac7e6b56864c1ddea2672ab641cb75605070ef2fdefd592`.
Canonical CLEVRER summary CSV SHA256:
`f293923d3fcf4f3111d7dcd8a7472cb0e916d9995b16d84b2b6650dd476558ce`.
Evaluation, raw saved example, feature-file, and manifest hashes are recorded
in the audit JSON.

Scope: raw predictions are retained for three cases, so raw pixel arithmetic
is independently checked for these three cases rather than the 125 unsaved
cases. Report aggregation uses every saved scalar row. LPIPS and I3D neural
networks are not rerun, and bootstrap intervals are not independently
resampled. The existing strict audit additionally checks frozen checkpoint,
configuration, detector, and complete result consistency.

Metric coverage is complete for the requested comparison. Ordinary pooled FVD
is the numerical video metric in the reference; its autoencoder-shifted variant
has no counterpart in this pixel-space setup. The existing implementation also
provides proper scores, frame/perceptual metrics, ensemble calibration, and
cost diagnostics. See `fvd_primary_source_audit.md` for source-backed detector
verification and explicit differences from the reference's sample protocol.
