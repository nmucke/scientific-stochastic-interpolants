# Long autonomous Navier–Stokes rollout: pre-execution protocol

Specified 2026-10-07 in response to the request for much longer trajectories and stability diagnostics. This is an additional exploratory experiment using existing trained models, not a revision of the completed short-horizon confirmation study or its frozen protocol. No models are retrained or selected using the new trajectories.

## Fixed rollout

- Initial condition: original held-out trajectory **180**, five observed frames at indices **0–4**. Lead 1 predicts original index 5. The four models receive exactly the same initial history.
- Ensemble: **eight members**; training-seed-zero map checkpoints and the original solver teachers. Preserve exact selected checkpoint/configuration hashes in the run manifest.
- Models and budgets per new physical frame: stochastic interpolant **64** steps, flow matching **64**, Itô map **8**, deterministic flow map **1**. Map names describe their fictitious-time dynamics; the flow map still has a random Gaussian base.
- Horizon: **4,096 stored snapshot intervals**, with a recoverable checkpoint at **1,024**. Store every lead's scalar diagnostics, dense fields for the first 100 forecast leads, then fields every 16 leads, plus resumable current histories and random-number states.
- Entirely autoregressive after initialization: no resetting to truth, clipping, amplitude renormalization, or replacement of failing members. Preserve failures. Stop a model run only on nonfinite state, retaining its first failure lead and every preceding result; do not apply an arbitrary amplitude cutoff.
- Run the four stability models serially on the available GPU. The initial portion shares the GPU with CLEVRER training; finish all such work before controlled video evaluation. Record actual network evaluations and incremental/cumulative generation times as workload-dependent diagnostics, not a speed benchmark. CPU diagnostics and writing costs must have an explicitly recorded timing scope.

The run seed and saved RNG states make individual experiments reproducible. Equal numerical seeds across different budgets or model families do not imply identical Brownian realizations or mathematically paired stochastic paths. Each member retains its own generated rolling history.

## What “time” and “truth” mean here

The supplied coarse archive contains 200 trajectories of 100 vorticity snapshots at 128², and only a `state` array. **Pre-execution provenance recovery:** the companion `data_file.pt` contains 256² fields and a 100-entry time vector. A full-array audit verifies exact equality of every coarse field with stride-two spatial downsampling of its companion. Thus the saved times are now linked to the forecast data: approximately 49.989998–99.480095, with mean spacing **0.49989997016 stored simulation-time units**. Viscosity, forcing amplitude and generating revision remain unverified. [Full audit](../results/long_rollout/data_reference_audit.json).

The primary plot x-axis remains **forecast lead in stored snapshot intervals**. A complete generated frame advances one such interval; solver/map steps are artificial generative integration steps inside that frame. The 4,096-lead horizon corresponds to approximately 2,047.59 nominal saved-time units using the recovered mean spacing. Beyond the recorded time vector this conversion is an explicitly nominal extension, not another measured time sequence; do not call the units physical seconds without independent provenance.

From the selected initialization, **95 future truth frames** exist (archive indices 5–99). RMSE, CRPS, and spread/skill are valid only through lead 95. Beyond that boundary there is no reference continuation; these quantities must be null/absent, never extrapolated, padded, or evaluated against repeated truth. The longer record supports autonomous stability and distributional diagnostics, not demonstrated long-horizon accuracy.

The repository's current stochastic NS simulator is a plausible source of a *new* experiment, but not a verified continuation of this archive. Its current script uses a 256² grid and five initial conditions and saves a different filename/key; its diffusion/viscosity/drag cannot be assigned to the existing data merely because its spatial grid matches the recovered companion. Generating another trajectory with those settings could change the reference process. Defer a physical-solver comparison until the original data-generation configuration is recovered. The older [provenance audit](navier_stokes_provenance.md) predates the timestamp recovery and remains relevant to the unverified physical parameters.

## Enstrophy and spread definitions

All diagnostics use **denormalized archive vorticity**, obtained by the saved inverse target transform. The current models have mean zero and standard deviation 3.09969, so using normalized outputs directly would multiply enstrophy by the wrong factor. Do not alter internal normalized autoregressive histories merely to compute diagnostics.

For member e with P spatial grid cells, define area-mean enstrophy

\[
Z_e(h)=\frac{1}{2P}\sum_{p=1}^{P}\omega_{e,p}(h)^2.
\]

Report its ensemble mean, member range/quantiles, and the separate enstrophy of the ensemble-mean field. This is an area average, not the total integral over an assumed physical domain. It requires no domain-size assumption. If a domain area A is later verified, integral enstrophy is A times this quantity. Existing radial spectrum routines use unnormalized FFT shell averages and do not directly sum to this diagnostic.

Define unbiased RMS ensemble spread and ensemble-mean skill by

\[
S(h)^2=\frac1P\sum_p\frac1{E-1}\sum_e(\omega_{e,p}(h)-\bar\omega_p(h))^2,
\qquad
R(h)^2=\frac1P\sum_p(\bar\omega_p(h)-\omega_p^*(h))^2.
\]

Show raw S at all leads, and corrected spread \(S_c=\sqrt{(E+1)/E}\,S\), R, and \(S_c/R\) only on the truth-supported interval. A zero R gives an undefined ratio. The correction reflects finite-ensemble mean error under ideal exchangeable draws; a ratio near one is a diagnostic, not proof of calibration. With one initial condition, the trajectory is not a collection of independent calibration cases.

The exact finite-ensemble identity

\[
\frac1E\sum_e Z_e
= Z(\bar\omega)+\frac{E-1}{2E}S^2
\]

provides a useful implementation check and explains why enstrophy of the mean can decay even when individual members retain energy. Interpret the two enstrophy curves separately.

## Planned figures and interpretation

1. Enstrophy through lead 4,096, with each model's ensemble mean and member variability. Mark lead 95 as the end of available truth. Plot the actual selected truth only where available.
2. RMS spread through all generated leads. Growing spread alone may represent increasing forecast uncertainty, dynamical divergence, or unrealistic amplitude growth; read it alongside enstrophy and field amplitude.
3. Truth-supported corrected spread and RMSE, plus their ratio, for leads 1–95. Do not extend this panel beyond the reference data.
4. Maximum absolute vorticity, finite-member fraction, and generation cost over the long rollout. Explicitly mark any termination; omit post-failure values rather than averaging only surviving members.

An optional shaded enstrophy band may summarize the existing 20 held-out trajectories across their 100 stored snapshots. Label it **archive context**, with exact pooling and counts. It is neither a reference continuation nor a stationary confidence/prediction interval. The archive's stationarity has not been established, and temporal samples are correlated.

Finite numerical values and bounded diagnostics over one 4,096-lead run establish only empirical boundedness over that run. They do not prove stability for all initial conditions, invariant-measure accuracy, physical consistency, or a correct long-time distribution. Conversely, record sustained amplitude drift, excessive dissipation, spread collapse, and nonfinite failure without hiding unfavorable methods. Compare all four fixed checkpoints; do not infer stochastic-map superiority from the study design.
