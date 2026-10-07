# Independent KTH result review

Reviewed 2026-10-07 on CPU using
`research/video_forecasting/experiments/independent_kth_audit.py`.
The compact result is
`research/video_forecasting/results/kth_independent_audit.json`.
The audit reads production outputs without changing them and imports no
production dataset, metric, or reporting functions. No issues were found.

Reproduction from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  research/video_forecasting/experiments/independent_kth_audit.py
```

- The original NumPy selection seed independently recreates the same 128
  cases from 452 eligible test segments. Manifest and all 128 selected source
  array hashes match. All selected subjects are 21–25 and disjoint from
  training/validation subjects. These segments represent 83 source videos.
- All 27 saved cases (three fixed cases across nine model/budget combinations)
  have ground truth that roundtrips exactly to original uint8 source pixels.
  Model normalization agrees with `2*pixel/255-1` within float32 rounding.
  Every one of the 810 saved future-frame scores uses truth indices 10–39,
  after the ten observed frames. Independent fair CRPS, energy score,
  RMSE, unbiased spread, corrected spread/skill, coverage, out-of-range
  fraction, clipped PSNR, and a Gaussian population-covariance SSIM formula
  agree with saved scores: largest absolute difference **7.11e-15**.
- The raw saved forecasts include values outside [-1,1]. Proper-score
  recomputation uses their unbounded unit-pixel transform, while image metrics
  clip their own copies. This distinction is numerically observable: replacing
  raw inputs by clipped inputs changes CRPS by up to **0.00236** in checked
  frames. The correct unclipped values match the recorded results. Thus this
  audit checks actual handling of out-of-range forecasts, not just labels.
- All nine feature archives contain exactly identical real features,
  shape 128×400, and generated features of shape 128×4×400. An independent
  dense covariance/eigendecomposition calculation reproduces pooled FVD and
  all four member FVDs to **5.75e-8** absolute difference. The pooled estimator
  uses 128 real versus 512 generated clips; each member estimator uses 128
  versus 128. As specified by the evaluator, feature extraction covers the
  shared ten observed frames plus thirty future frames. Frame scores cover
  future frames only. Member-FVD standard deviation is completion variability,
  not a confidence interval.
- All 27 saved contact sheets reproduce pixel-exactly from the corresponding
  raw NPZ: the fixed first three selected cases, first two members, future
  leads 1, 8, 16, 23, 30. No quality-based selection appears in these images.
- Every `summary.csv` point estimate (including LPIPS aggregation, temporal
  cost, and FVD summaries) independently recomputed from all saved score rows
  agrees to **7.11e-15**. Frame metrics use equal source-video weights and then
  equal lead weights, preserving the intended treatment of multiple segments
  from one source video.

Scope: saved raw predictions cover three of 128 cases, so independent pixel
arithmetic checks cover those three cases rather than unsaved forecasts.
All rows are included in the report-aggregation check. LPIPS/I3D networks are
not rerun; detector/model/config hashes and full feature/row consistency were
also checked by the existing strict audit (`results/kth_audit.json`). Bootstrap
confidence intervals are not independently resampled by this second audit.
The frozen protocol, evaluation JSON, feature files, raw examples, and report
CSV hashes are recorded in the new audit JSON.
