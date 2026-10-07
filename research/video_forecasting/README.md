# Pixel-space KTH and CLEVRER forecasting

This extension trains and evaluates all four existing model families in pixel
space: stochastic interpolants, flow matching, Itô maps and deterministic flow
maps. It preserves the original Navier–Stokes study on its earlier branch.

Read [STATUS.md](STATUS.md) before launching jobs. It identifies active downloads
and GPU work, the fixed initial budget and remaining tasks. This is an initial
operational study: no latent autoencoder, no random historical conditioning
frame, one training seed, and a smaller evaluation sample count than the paper.

## Main files

- [Primary-source protocol](notes/paper_video_protocol.md): reference horizons,
  splits, FVD definition, source-code inconsistencies and deliberate changes.
- [Data provenance](notes/data_provenance.md) and [audit summary](notes/data_summary.json).
- [Training integration](notes/model_training_integration.md): tensor contracts,
  checkpoints, configurations and individual commands.
- [Metric weights](notes/metric_weights.json): pretrained I3D and LPIPS hashes.
- [Training script](experiments/train_all_video_models.sh): both datasets by default,
  or pass `kth` / `clevrer` to run one. Existing completed stages are preserved.
- [Evaluation script](experiments/evaluate_all_video_models.sh): freezes cases and
  checkpoints, evaluates all four models plus persistence, and audits results.
- [Reporting script](experiments/report_video_results.py): checks matched cases
  before writing tables, per-lead curves, FVD/cost plots and fixed-example GIFs.

## Rebuild a complete report

Run from the repository root after both evaluation blocks complete:

```bash
.venv/bin/python research/video_forecasting/experiments/report_video_results.py \
  --evaluations research/video_forecasting/results/test/kth/*/evaluation.json \
                research/video_forecasting/results/test/clevrer/*/evaluation.json \
  --output-dir research/video_forecasting/results/report
```

`--allow-incomplete` is for clearly labeled development reports only. Use the
same selected cases, ensemble size, horizons and metric weights within each
dataset. Never rank KTH and CLEVRER together by raw FVD values. Test results,
training metadata and source hashes are tracked; large datasets, weights and
forecast arrays remain local. Saved features and first fixed qualitative cases
allow reporting to be rebuilt without resampling models.
