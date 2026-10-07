# Pixel-space model and training integration

All four model families reuse the existing conditional UNet and their original mathematical objectives. The only data adaptation is a `VideoFrameDataset` producing `base`, `target`, and `field_history`. A frame has shape `[C,H,W]`; the last N observed frames have shape `[C,H,W,N]`. The existing history conditioner internally flattens `(time,channel)` in chronological order. KTH defaults to ten grayscale history frames, CLEVRER to two RGB frames. No future frame enters conditioning. Both pilot datasets have resolution 64×64; CLEVRER thus differs from the source paper's 128×128 setup. CLEVRER preprocessing retains every fifth original frame.

The teachers are `FollmerStochasticInterpolant` with a latest-frame source and `FlowMatchingModel` with a fresh Gaussian source. Maps are `FollmerItoMapModel` and `FlowMapModel`, initialized from their corresponding newly trained teachers. No architecture or numerical behavior of the scientific benchmark models is changed. The common pixel UNet uses widths `[16,32,64]`, two ConvNeXt blocks, a 128-dimensional time embedding, zero padding, and bottleneck spatial attention. Four Itô path features require four input channels for grayscale and twelve for RGB; each feature retains an independent increment per state channel.

`LocalTrainer` subclasses the existing `Trainer` and reuses its drift regression objective and batch pseudo-time/noise generation. It adds offline model/best/restart checkpointing, bounded training epochs, sample-weighted losses, fixed-noise validation that restores the training RNG, local history, nonfinite-loss checks, and epoch-boundary restarts. Unlike the old entry point, no tracking-service identity is required. The validation loader shuffles under a fixed validation RNG, so capped validation covers dispersed windows rather than just the beginning of one clip. Validation and test splits remain separate. Existing `MapTrainer` handles both progressive and teacher-rollout training unchanged.

## Commands

Teacher example (100 epochs × 100 batches = 10,000 updates if the loader supplies enough batches):

```bash
.venv/bin/python -m scisi.bin.main_video_model_train \
  --config-name follmer_stochastic_interpolant_kth \
  trainer.num_epochs=100 trainer.max_train_batches=100
```

Replace the config with `flow_matching_kth`, `follmer_stochastic_interpolant_clevrer`, or `flow_matching_clevrer` for other teachers. Common defaults are batch 16, AdamW learning rate 2e-4, weight decay 1e-6, cosine decay to 2e-5, full precision, and seed zero. The default YAML only runs 2,000 teacher updates as a bounded initial pilot; executed experiments record their resolved overrides.

Progressive map example (2,000 updates):

```bash
.venv/bin/python -m scisi.bin.main_video_model_train \
  --config-name follmer_ito_map_kth trainer.num_epochs=20
```

The eight named model/dataset configs are in `config/`; maps can also use the preexisting `main_map_model_train` entry point. The common map default is 1,000 updates and 16 teacher steps. Every actual experiment must preserve its resolved configuration. Refinement is an explicit second training run, for example:

```bash
.venv/bin/python -m scisi.bin.main_video_model_train \
  --config-name follmer_ito_map_kth \
  initial_student_checkpoint=checkpoints/video/kth/follmer_ito_map/model.pth \
  trainer.checkpoint_dir=checkpoints/video/kth/follmer_ito_map_rollout \
  trainer.objective=teacher_rollout trainer.diagonal_weight=0 \
  trainer.interval_curriculum_steps=0 trainer.initial_max_interval=1 \
  trainer.consistency_ramp_steps=0 trainer.num_epochs=10 \
  trainer.max_train_batches=50 optimizer.lr=0.00005
```

Tests in `tests/test_video_models.py` verify both source laws' grayscale/RGB shapes, unchanged drift after teacher-to-map initialization, finite gradient and sampling behavior, chronological history updates, local checkpointing, deterministic validation with unchanged RNG, and exact CPU restart equivalence. These are implementation checks, not evidence of trained video forecast quality.

Actual full-KTH CPU entry-point smoke checks succeeded for the teacher and Itô map (one training and one validation batch each), with 145,747 valid training windows. Parameter counts are 2,920,192 for the KTH teacher and 2,937,024 for its Itô map. Smoke checkpoints live only under `/tmp/scisi_video_{teacher,map}_smoke` and are not research results. The manifest validation split is named `validation`.

## Pre-test freeze and completed-run audit

After both 20,000-update teachers and each map's 2,000-update PSD plus 500-update refinement have completed, run:

```bash
.venv/bin/python research/video_forecasting/experiments/freeze_video_protocol.py --dataset kth
```

The freezer checks exact recorded update schedules, seed, history, train/validation splits, own-teacher initialization and own-PSD refinement lineage. It archives configurations, histories, metadata and weight hashes under `results/training/kth_<stage>` without copying weights. It creates an immutable `results/kth_protocol.json` containing the manifest and metric-weight hashes, six trained stages, selected checkpoint/configuration hashes, and all 128 seeded eligible test clips with their decoded-file hashes. The fixed protocol uses E=4, KTH 10→30 or CLEVRER 2→14 frames, solver budgets 16/64, map budgets 1/8, and persistence budget 1. It refuses to freeze after any held-out JSON output exists, or to overwrite an existing protocol. `--verify-only` rechecks an existing freeze without editing it.

After all five evaluation directories are complete:

```bash
.venv/bin/python research/video_forecasting/experiments/verify_video_results.py --dataset kth
```

The verifier checks the full case/budget/lead grids; finite computed metrics including LPIPS; exact NFE and cumulative timings; equal-source-video frame summaries; recorded cases and forecast copies; frozen model, configuration, manifest, metric-weight, and selected-frame hashes; saved I3D feature shapes and hashes; and recomputed pooled and member FVD. Real-video features must agree across models/budgets. Successful verification writes `results/kth_audit.json`. Substitute `clevrer` for its independently frozen experiment. The ten synthetic artifact-audit tests do not create or modify the real research freezes.
