# Pixel video data provenance

Acquisition started 2026-10-07. Data are downloaded from the original dataset hosts; the source ZIPs are retained under `data/video_forecasting/archives`. Decoded arrays are uint8 `[T,H,W,C]` NumPy files, one per clip. Manifests record archive/member, source frame indices, split, frame count, and SHA-256 hashes of both archives and decoded arrays. The loader memory-maps only the requested consecutive history window, converts to float32, and applies fixed `2*x/255-1` scaling. It fits no dataset statistics. Future frames are never included in the conditioning history. Training visits every valid next-frame window, including the final available target. A configurable window stride may thin these windows.

## KTH

Original source: https://www.csc.kth.se/cvap/actions/ and its six action ZIPs (`boxing`, `handclapping`, `handwaving`, `jogging`, `running`, `walking`) and `00sequences.txt`. The source site describes 25 subjects, four scenarios, 160×120 video at 25 frames/s, and non-commercial use with citation to Schuldt, Laptev and Caputo (ICPR 2004). Downloaded all six original ZIPs, containing 599 available source videos, and extracted the official 2,391 action subsequences. Annotation endpoints are inclusive and one-based; preprocessing converts them to zero-based, half-open ranges. One annotated endpoint (`person01_boxing_d4_s03`) extends beyond decoded EOF; it is truncated and both endpoint values are recorded. The same original source video reports one recoverable MPEG4 damaged-macroblock diagnostic; its ZIP member CRC verifies, ffmpeg returns success, and the recovered pixels are retained consistently across all models. No other KTH decoding errors were logged. See `kth_source_decode_issues.json`.

We follow Chen et al.'s forecasting paper's **last five subjects as test**, rather than the original action-recognition subject split or the incomplete demonstration repository's scenario split. Subjects 1–18 form training, subjects 19–20 are a disjoint validation set chosen for this implementation, and subjects 21–25 are test. The resulting counts are 1,720 / 192 / 479 segments. For 10 observed + 30 future frames, segments shorter than 40 are excluded without padding: 1,649 / 184 / 452 eligible clips. Training next-frame windows may use shorter segments (minimum 11 frames).

All six actions are used. Frames are bilinearly resized directly to 64×64 grayscale, without temporal downsampling. This changes aspect ratio, following the forecasting repository's KTH resizing convention. Frames are sampled consecutively at the original frame rate. Manifest: `data/video_forecasting/kth_64/manifest.json`.

## CLEVRER

Original source: https://clevrer.csail.mit.edu/ links the video archives hosted at:

- https://data.csail.mit.edu/clevrer/videos/train/video_train.zip (12,354,893,389 bytes)
- https://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip (6,206,059,321 bytes)
- https://data.csail.mit.edu/clevrer/videos/test/video_test.zip (6,149,404,859 bytes)

The complete downloaded dataset uses the original 10,000 / 5,000 / 5,000 video splits. All 20,000 videos decoded successfully without decoder diagnostics, and each yields exactly 26 retained frames. No question/answer data or learned latent representations are used. Preprocessing selects decoded frame indices 0,5,10,... (the reference implementation's `skip_frames=4`), bilinearly resizes the short side to 64, and center-crops a 64×64 RGB square. This deliberately uses 64² pixels for the first full-space experiments; the reference RIVER setup uses 128². The spatial transform follows RIVER's resize-then-center-crop convention. Typical 128-frame source videos yield 26 retained frames, sufficient for two observed + fourteen future frames. Manifest: `data/video_forecasting/clevrer_64/manifest.json`.

Acquisition and preprocessing completed on 2026-10-07. All 23,391 decoded KTH/CLEVRER arrays passed full SHA-256 and shape checks; source-video split isolation, KTH subject separation, and CLEVRER raw-frame-index spacing also passed. Verified archive/manifest hashes, annotation-file hash, split counts and eligible-window counts are recorded in `data_summary.json`. All source archives are retained locally, and approximately 30 GB remains free on the workspace filesystem. Keep the pixel-space and fixed-last-N-frame protocol differences explicit; these experiments are not exact numerical reproductions of the latent-space paper.

## Reproduction

```bash
.venv/bin/python research/video_forecasting/data/prepare_videos.py --dataset kth
.venv/bin/python research/video_forecasting/data/prepare_videos.py --dataset clevrer
```

Downloads resume with HTTP ranges through curl. ZIP CRC checks run when each member is read. Per-video arrays and completed-archive indexes are written atomically. A completed archive index is reused only if the source archive hash is unchanged. Temporary decoded source files are removed after each clip. The download archives and arrays are ignored by Git; source code and compact provenance summaries are tracked.
