# Why similar CRPS can coexist with noticeably different calibration

This is an elementary illustrative calculation, not a new theorem, a fit to the NS results, or a replacement for the finite-ensemble diagnostics.

Suppose the true future is Y~N(0,1), while a forecast uses F_sigma=N(0,sigma²). The mean is correct for every sigma; only the forecast spread changes. Let X and X' be independent forecast draws, also independent of Y. From the definition

    CRPS(F_sigma,Y) = E_X |X-Y| − 1/2 E_(X,X') |X-X'|,

and E|N(0,v)|=sqrt(2v/pi), the expected score is

    S(sigma) = E_Y CRPS(F_sigma,Y)
             = sqrt(2(1+sigma²)/pi) − sigma/sqrt(pi).

It is minimized at sigma=1, with S(1)=1/sqrt(pi). Direct differentiation gives S'(1)=0 and S''(1)=1/(2sqrt(pi)). Consequently,

    S(sigma)−S(1) = (sigma−1)²/(4sqrt(pi)) + O((sigma−1)³).

The relative score penalty is therefore approximately (sigma−1)²/4 near the calibrated forecast. Small errors in spread have a second-order effect on the average score.

The forecast's central **population** 90% interval is [−z_0.95 sigma,+z_0.95 sigma], where z_0.95=1.64485362695. Its actual coverage under the true distribution is

    C(sigma) = P(|Y|≤z_0.95 sigma) = 2 Phi(z_0.95 sigma)−1.

Its derivative at the calibrated forecast is C'(1)=2 z_0.95 phi(z_0.95)≈0.339286, which is nonzero. Hence the coverage error changes to first order with spread. This difference in sensitivity explains why a small proper-score penalty can accompany a visible calibration loss.

| Forecast standard deviation sigma | Relative expected CRPS penalty | True coverage of population90% interval |
|---:|---:|---:|
| 0.50 | 8.1139% | 58.9166% |
| 0.70 | 2.6268% | 75.0431% |
| 0.84 | **0.6943%** | **83.2929%** |
| 0.90 | 0.2630% | 86.1225% |
| 1.00 | 0 | 90.0000% |
| 1.10 | 0.2380% | 92.9602% |
| 1.20 | 0.9072% | 95.1598% |

At sigma0.84, the standard deviation is16% too small but the expected CRPS penalty is below1%. Nevertheless, nominal population90% coverage falls by6.71 percentage points. Numerically S(0.84)=0.5681069893, compared with S(1)=0.5641895835.

## Independent checks and artifacts

`experiments/gaussian_score_calibration.py` derives the score from the two absolute Gaussian differences, then independently checks it by integrating the closed-form fixed-observation Gaussian CRPS against the true density using128-point Gauss–Hermite quadrature. All tabulated values agree at the script's1e-11 relative tolerance.

A paired Monte Carlo check uses one million common normal draws (seed4631). At sigma0.84, the exact score excess is0.00391741; the Monte Carlo excess is0.00380572 with standard error0.00012682. Monte Carlo coverage is83.2874%, versus the exact83.2929%.

Saved outputs are `experiments/results/gaussian_score_calibration/results.json` and the standalone two-panel PDF/PNG figure in the same directory. Everything is CPU-only and separate from model training or evaluation.

## Interpretation boundaries

CRPS is still proper: in this toy, the correct distribution has the uniquely best expected score. The example shows local insensitivity of its numerical value to small dispersion errors, not that proper scoring fails or that calibration should replace it. Bias, non-Gaussian tails, multimodality and spatial dependence may change the tradeoff.

The calculation concerns exact distribution quantiles. The NS evaluation instead uses interpolated empirical quantiles from16 ensemble draws; even a calibrated finite ensemble does not attain90% expected coverage under that estimator. Do not compare83.29% here directly with the NS empirical coverage values, or interpret sigma0.84 as an estimated NS variance multiplier. Under independent draws, the *fair* ensemble CRPS estimator has this same population expected score, but it has finite-sample variation.

An appropriate paper sentence is: "Close proper scores need not imply equally calibrated spread: in a centered Gaussian illustration, a16% spread deficit increases expected CRPS by only0.69%, while coverage of the forecast's population90% interval falls to83.3%. This illustration is distinct from our finite-ensemble empirical intervals." A compact appendix table or formula is sufficient; the full figure is optional.
