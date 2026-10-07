# Spatial enstrophy spectra at the first and last forecast steps

The requested horizontal coordinate is spatial frequency, not forecast lead. The new figures therefore show separate **lead 1** and **lead 4,096** spectra on logarithmic frequency and enstrophy axes. Scalar enstrophy-versus-lead diagnostics are retained as additional temporal diagnostics; they do not replace these spectra.

## Definition and units

For each raw-vorticity ensemble member on the 128×128 observed grid, compute the full complex FFT with `norm='forward'`. For mode `(kx,ky)`, modal enstrophy is

\[
P(k_x,k_y)=\tfrac12|\widehat\omega(k_x,k_y)|^2.
\]

Group modes into the nearest integer radial shell: `k = floor(sqrt(kx²+ky²)+0.5)`. The reported `Z(k)` is the **sum** of `P` in that shell, with no division by mode count or total power. This gives the exact discrete Parseval relation, up to floating-point rounding:

\[
\sum_{k=0}^{91} Z(k)=\tfrac12\operatorname{mean}_{x,y}\omega(x,y)^2.
\]

The field is denormalized with each run's frozen affine preprocessing before the FFT. This preserves absolute squared-vorticity units. No spatial mean is subtracted. The frequency unit is **cycles per grid-domain**; no physical domain length or inverse-metre unit is assumed. These spectra describe the archived 128² fields, which are stride-two subsamples of the companion 256² archive. No new anti-aliasing or smoothing is applied.

The complete square Fourier grid has 16,384 coefficients. Its axis Nyquist frequency is 64, but corner modes reach radial frequency `sqrt(64²+64²) ≈ 90.5097`, assigned to shell 91. Every coefficient is retained. Shells above 64 have incomplete annular support because of the square grid; the figures mark this region and the JSON retains shell mode counts. Shell 91 contains the single checkerboard corner coefficient. Silently cutting at 64 would break the complete Parseval accounting.

DC is retained in the numerical data and audits but omitted from the log-frequency figures. Nonpositive powers are omitted from logarithmic plotting without adding a numerical floor. Plotting excludes DC only; it does not renormalize the retained spectrum.

## Ensemble and reference conventions

Each generated member has its own spectrum. The main model curve is the mean of those member spectra; shaded bands are their pointwise 5th–95th percentiles, not confidence intervals. The spectrum of the ensemble-mean field is also saved separately: it is generally smaller because averaging fields cancels uncertain phases. It must not be substituted for the mean member spectrum.

The first-step figure can show the actual paired target, held-out trajectory 180 at archive frame index 5. There is no matched truth at lead 4,096. A separately labelled gray historical context band can show pointwise spectral quantiles over all 20 held-out trajectories × 100 stored fields. This is not final-step truth, a predictive interval, or an established stationary target. Those 2,000 fields are temporally correlated.

These absolute shell-sum figures use a different binning/normalization convention from older repository spectral score implementations. Existing tabulated spectral-error values are not recomputed or relabelled as errors of these new curves.

## Implementation, evidence, and reproduction

`experiments/spatial_enstrophy_spectra.py` is the sole FFT producer. Plotting and manuscript builds consume its compact JSON and need neither ignored raw arrays nor the original data downloads. Provenance includes normalized-raw file hashes, frozen run/model/config hashes, the archive hash, affine parameters, and the computation script hash.

The current two-map preview is `results/long_horizon/spatial_enstrophy_spectra_maps_preview.json`: two models × two leads, 92 shells including DC, all per-member spectra, ensemble summaries, first-step truth, and historical context summaries. It is explicitly marked `comparison_complete: false` until both ODE/SDE baselines are added. Eight tests verify constant/DC power, a known sine mode, random-field Parseval conservation, squared-amplitude scaling, retained corner-Nyquist power, the distinction between ensemble spectral averaging and field averaging, and invalid inputs. The actual-data maximum Parseval residual is 3.55e-15, including all 2,000 context fields. The context mean spectrum sums to 4.6653943657, matching its independently computed scalar enstrophy.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python \
  research/ito_maps/experiments/spatial_enstrophy_spectra.py \
  --run-dirs research/ito_maps/results/long_horizon/flow_map \
             research/ito_maps/results/long_horizon/follmer_ito_map \
  --output /tmp/ns_spatial_enstrophy_maps_preview.json
.venv/bin/python -m pytest tests/test_spatial_enstrophy_spectra.py -q
```

For the final four-family comparison, add both completed solver run directories and use a new output `results/long_horizon/spatial_enstrophy_spectra.json`. The default requested leads are 1 and 4,096. Missing saved endpoints or nonfinite fields cause an explicit failure; no final-step field or reference is fabricated.
