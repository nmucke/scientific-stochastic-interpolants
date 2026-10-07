# FVD definition and primary-source compatibility audit

Audited 2026-10-07. The existing `scisi.metrics.video` implementation already
provides the required FVD. No alternative detector, score definition, or
production-run change is needed.

## Published experiment and available evidence

Chen et al. evaluate 256 held-out clips with 100 generated completions each
(25,600 generated clips). Their KTH experiment observes 10 frames and predicts
30 at grayscale 64²; CLEVRER observes 2 and predicts 14 at RGB 128². Forecasting
uses a VQGAN latent representation and randomized earlier-frame conditioning.
Their table also subtracts autoencoder reconstruction FVD as a separate shifted
score. Our pixel-space experiment has no corresponding autoencoder correction.
These details are explicit in [§4.3 and Table 1](https://arxiv.org/html/2403.13724v2#S4.SS3).

The [paper's official code index](https://github.com/interpolants/forecasting)
links to `marikgoldstein/video_interpolants`. The latter repository's
[work-in-progress README](https://github.com/marikgoldstein/video_interpolants/blob/5c087bbf9d424379888798d778cb1007286deb1b/work_in_progress/README.md)
states that FVD was omitted and points to Google's reference implementation.
Thus the released code does not establish every detail of the paper's actual
feature-extraction pipeline. In particular, it does not independently certify
whether those reported numbers included the observed prefix. Our prefix choice
must be stated explicitly, without claiming an exact evaluator reproduction.

## Detector and preprocessing verified

The [Google reference](https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py)
uses the Kinetics-400 TF-Hub I3D model and the final 400-logit tensor
`RGB/inception_i3d/Mean:0`. Input frames are resized to 224² and normalized to [-1,1].
These are logits before softmax, rather than class probabilities or the
1024-dimensional hidden representation.

Our detector is the [StyleGAN-V Google conversion](https://github.com/universome/stylegan-v/blob/master/src/metrics/frechet_video_distance.py),
also used by the [RIVER FVD wrapper](https://github.com/Araachie/river/blob/9ecde339e4c19713485fc1ab31bcb524d6772f6e/refinement/fvd.py).
Its frozen SHA256 is
`bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2`.
StyleGAN-V explicitly documents spatial upsampling as a difference from the
TensorFlow reference and distinguishes these converted weights from the
original I3D repository's release. Thus our numerical backend is not claimed
to be bitwise equivalent to Google's TensorFlow implementation.

CPU inspection of the actual downloaded TorchScript `forward` confirmed:

- `return_features=True` returns the final 400-channel classifier output,
  averaged over its remaining time dimension, before softmax.
- `resize=True` resizes frames to 224² using PyTorch bilinear interpolation with
  `align_corners=False`. The time dimension is retained unchanged.
- `rescale=False` is correct for model frames already in [-1,1]. The repository
  wrapper clips a metric-only copy to that range and repeats grayscale KTH into
  three channels. It does not apply a second 255 rescaling.
- Generated clips concatenate the exact observed prefix with the generated
  future: 40 frames for KTH and 16 for CLEVRER. Real features use the corresponding
  full truth clip. Proper scores and image errors separately use future frames
  only. Generated feedback remains unclipped.

Local implementation: `src/scisi/metrics/video.py` (`I3DFeatures`) and
`src/scisi/bin/main_video_model_test.py` (prefix assembly and cached features).
The earlier detector provenance and CPU shape check remain in
`notes/i3d_provenance.json`. This audit does not mutate that historical record.

## Accessible mathematical definition

Let feature means be m_r and m_g, and unbiased empirical covariance matrices
be C_r and C_g. We report

```
FVD = ||m_r - m_g||² + trace(C_r + C_g
      - 2 (C_r^(1/2) C_g C_r^(1/2))^(1/2)).
```

This is the squared Gaussian Fréchet/Wasserstein distance; lower is better.
The implementation evaluates the covariance cross term through a low-rank SVD
of centered features, in float64. This is an algebraically equivalent way to
compute the displayed expression, including singular covariance matrices.
It neither changes the metric nor takes an extra outer square root. Feature
samples are clips; correlated completions are not treated as independent
real reference videos for uncertainty claims.

`tests/test_video_metrics.py` already checks translated distributions,
different sample counts, rank-deficient covariances, an independent PSD
covariance formula, and invalid inputs. An independent arithmetic audit of
all nine completed KTH configurations reproduces the FVD values using dense
covariance eigendecomposition to 5.75e-8 absolute error. No duplicate FVD
implementation is introduced into the production path.

## Reporting the actual pilot

Our frozen comparison uses 128 real clips and 4 completions per clip, hence 512
pooled generated features. Every model/budget shares the same cases, detector,
preprocessing, and counts. The supplemental memberwise score compares 128 real
with 128 generated clips for each of the four completion indices; its standard
deviation describes completion variability, not a confidence interval. Pooled
FVD is the primary estimator corresponding to the reference's asymmetric
real/generated comparison, at deliberately smaller sample counts.

Suggested table caption:

> Pooled FVD↓ uses Google-converted Kinetics-I3D-400 logits from the observed
> prefix plus forecast (KTH 10+30; CLEVRER 2+14), with 128 real clips and four
> completions each (512 generated). This 64² pixel-space pilot uses smaller
> sample counts and different conditioning from the 256×100 latent-space
> experiment of Chen et al.; its values are not a numerical reproduction.

The finite-sample score depends on sample counts and preprocessing. Do not
rank our values against the published table's absolute numbers. The current
framework already allows larger case/completion counts; changing the frozen
pilot counts requires a separately labelled evaluation, not relabelling the
existing outputs. No shifted or normalized FVD should be invented for the
pixel-space experiment.
