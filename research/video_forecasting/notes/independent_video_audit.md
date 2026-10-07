# Independent video audit workflow

The shared CPU implementation is
`research/video_forecasting/experiments/independent_video_audit.py`.
It supports both grayscale KTH and RGB CLEVRER without importing production
metric, dataset, or report code. The earlier `independent_kth_audit.py` command
remains a compatible wrapper. The original completed KTH audit JSON is
preserved unchanged; the generic implementation reproduced its checks on all
nine configurations with output sent to `/tmp/kth_generic_audit.json`.

After a dataset's full evaluation and report are complete:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  research/video_forecasting/experiments/independent_video_audit.py \
  --dataset clevrer \
  --report research/video_forecasting/results/report/clevrer/summary.csv
```

The default output is `results/clevrer_independent_audit.json`. A custom report
path can be passed when the final report includes both datasets. The script
requires every model/budget in the frozen protocol, complete evaluation JSON,
and the report CSV before writing a successful final artifact. Dataset history,
horizon, channel count, sample count, ensemble size, image selection, and source
counts are inferred from the frozen protocol and manifest. It checks source
split disjointness for both datasets and additionally held-out KTH persons.

## Interim RGB validation

While CLEVRER evaluation continued, all six available SI example NPZs (three
cases at 16 and 64 steps) were checked against the already saved per-frame rows
using the shared independent `metrics()` formula. All 84 future rows agree to
**7.11e-15** across CRPS, energy score, calibration, PSNR, and SSIM. Their saved
truth reconstructs the original RGB uint8 pixels exactly; all six contact sheets
reconstruct pixel-exactly from fixed first members and future leads 1, 4, 8, 11,
14. The partial record, with raw NPZ hashes, is
`results/clevrer_rgb_partial_validation.json`.

This interim artifact makes no claim about unfinished configurations, FVD, or
final report aggregation. Those are checked only by the full command above.

## Completed CLEVRER audit

All nine configurations subsequently passed the full independent audit against
`results/report/clevrer/summary.csv`. See `clevrer_independent_audit.md` and
`results/clevrer_independent_audit.json` for the final evidence. The partial RGB
record above remains a historical checkpoint, not the final audit.
