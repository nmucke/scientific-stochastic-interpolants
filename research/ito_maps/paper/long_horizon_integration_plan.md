# Planned manuscript integration of long autonomous rollouts

## Automatic four-model integration

`build_long_horizon_figures.py` now keeps the verified two-map preview until both `results/long_horizon/independent_all_model_audit.json` and `results/long_horizon/spatial_enstrophy_spectra_all_models.json` exist. With both present it requires a passed audit for exactly SI64, FM64, Itô8 and FlowMap1, matching protocol/diagnostic hashes, matching spectral provenance, the prescribed trajectory180/origin5/E8 settings, and all-finite completed4096-lead diagnostics. A failed final pair raises an error rather than silently reverting to the favorable subset. A nonfinite or shortened teacher run requires an explicitly revised failure-aware presentation; this automatic completed-run path will not label it complete.

The final audit uses the existing map-audit schema: `status`, `runs`, and per-run `model_class`, `run_directory`, `protocol_sha256`, `diagnostics_sha256`, `all_scalar_rows_checked`, `saved_raw_files_checked`. Run paths are resolved against the repository root. The spectrum producer supplies the existing spectrum schema with `comparison_complete=true` and both endpoint leads for each family. After both audits are generated, `make -C research/ito_maps/paper` rebuilds all four spectral/skill curves, the table and provenance, and `generated/long_horizon_status.tex`. That status file switches main-text pending language and appendix captions and generates the actual teacher summary values from diagnostics. Existing map findings remain unchanged; final scientific interpretation still needs review after the teacher evidence is available.

Prepared while the four-model 4,096-lead study is running. **Status update (2026-10-07):** the user subsequently requested a manuscript update while experiments continue. The draft now includes the two completed, independently audited map runs in `long_horizon_results.tex`, explicitly labeled as a partial comparison with solver references pending. Timestamp provenance has been corrected, and `build_long_horizon_figures.py` regenerates the two-map table and figures from audited artifacts. The user clarified that the requested enstrophy plots are spatial-frequency spectra, so separate first-step and last-step log--log shell-summed spectra now replace the scalar-over-lead paper panel. Scalar-over-lead curves remain supplemental report files. The PDF has 19 pages with main text and references ending on page 8. The plan below remains the guide for the later complete four-model update; its original wait-for-all-results gate has been superseded for this clearly labeled interim draft only.

## Immediate provenance correction to make during integration

`main.tex`, currently around line 205, says the saved time increment cannot be certified. That statement is now outdated. The completed CPU audit verifies exact equality of all 200 × 100 coarse fields with stride-two downsampling of `data_file.pt`, which supplies the timestamps. Replace that paragraph with a similar-length account:

> A companion high-resolution archive was recovered and every field verified against the supplied array by exact stride-two downsampling. Its timestamps span approximately 49.989998–99.480095, with mean increment 0.49989997016 in saved simulation-time units. We retain stored snapshot intervals as the primary lead coordinate. The original forcing, viscosity, drag and generation revision remain unverified; the current simulator configuration alone does not certify them.

Appendix provenance can give the two archive hashes and full-equality audit path. Do not transfer the current simulator's physical coefficients into the recovered metadata, and do not call the units seconds. For the extended rollout, 4,096 intervals correspond to approximately 2,047.59 nominal saved-time units. That conversion beyond recorded timestamps is a declared extension, not a measured reference continuation. The reviewer-risk provenance row must be updated consistently.

## New appendix section

Add `long_horizon_results.tex`, included after the existing qualitative-results appendix, with a stable label `app:long-horizon`. Suggested title: **Long autonomous rollouts and available-reference skill**.

Keep the section to about one page of text/table and one page of figures:

1. **Protocol paragraph:** existing selected seed-zero maps and fixed teachers; one held-out history, trajectory 180 and frames 0–4; E=8; SI64, FM64, Itô8, FlowMap1; 4,096 sequential forecast intervals; independent generated histories, no clipping or resets. This is a new exploratory extension, not another three-seed confirmation. Give the frozen-protocol and completed-run audit provenance.
2. **Definitions paragraph:** area-mean enstrophy `Z = mean(omega²)/2` in denormalized archive units; distinguish mean member enstrophy from enstrophy of the ensemble mean. Define unbiased RMS spread and the `sqrt((E+1)/E)` correction already used in the metric appendix. The finite-ensemble decomposition explains ensemble-mean smoothing without claiming member dissipation.
3. **Truth boundary paragraph:** paired reference exists only for leads 1–95. RMSE and spread/skill terminate there. Later curves concern boundedness, spread and amplitude. The gray band pools 2,000 held-out archive fields and is contextual, not a continuation, stationary confidence interval, or target distribution proved valid at arbitrarily late times.
4. **Compact result table:** actual completed leads; any first nonfinite lead; mean member enstrophy and RMS spread over the final 512 recorded leads; optionally maximum absolute vorticity over the complete run. Print the averaging interval for early failures rather than relabeling an incomplete tail as the final 512 of 4,096. Avoid selecting a favorable endpoint from a fluctuating curve.
5. **Interpretation paragraph:** report the four models symmetrically. If a map stays finite but loses member enstrophy and spread, describe empirical dissipation/underdispersion, not accurate long-term stability. If the deterministic map retains a stronger amplitude/spread profile, say so. Do not attribute a difference specifically to Brownian compression, teacher error or training duration without a causal ablation. Near-one spread/skill over a single correlated temporal record is not proof of conditional calibration.

## Figures and reproducibility

Use the existing reporting outputs as the review source:

- `autonomous_enstrophy_spread.png`: all 4,096 leads, member-mean enstrophy, mean-field enstrophy, RMS spread and maximum amplitude; clearly mark the truth boundary. Member 5–95% bands are quantiles, not confidence intervals.
- `truth_supported_spread_skill.png`: corrected spread versus RMSE and their ratio, restricted to leads 1–95.

For the paper, generate PDF versions from the same numeric inputs, preserving the report's values and definitions. Avoid screenshots or manual tracing. Add one small builder, `build_long_horizon_figures.py`, or a PDF output option to the existing report script. Connect figure/table dependencies to the four completed `diagnostics.json` files and the contextual reference JSON in the Makefile only after these artifacts exist. Any generated claims should be computed from the saved records, not copied from progress updates.

The autonomous plot should retain raw curves. If visual clutter motivates an additional smoothed curve, label the window, preserve the raw values in the archive, and do not let smoothing hide excursions or a failure. Generation times are diagnostic wall times, potentially affected by concurrent workloads; do not add new speedup claims from this study. The controlled latency results already in the paper remain the speed evidence.

## Main-text edits, contingent on complete results

Keep the eight-page main-text limit. Prefer replacing a sentence in the existing limitations paragraph over adding a new paragraph or figure. A possible result-neutral structure is:

> An additional single-history 4,096-interval rollout separates numerical boundedness from distributional fidelity: amplitude and spread diagnostics are reported throughout, while reference-based skill is available for only 95 intervals (Appendix X).

Once actual results are known, replace the middle clause with the supported observation if warranted, explicitly naming any Itô-map loss of enstrophy/spread. Do not frame a finite but dissipative trajectory as retained sample quality. The abstract need not grow; its existing underdispersion and spectral-error caveats are consistent with investigating this limitation more deeply.

The existing ten-lead experiment stays intact: different origin, E=16, all 20 trajectories, and three training seeds. Do not pool its uncertainty estimates with this one-history E=8 extension. Update the reviewer-risk table's long-term-stability row from “ten leads only” to the actual new empirical scope, retaining the absence of an invariant-measure guarantee and long conditional reference ensembles.

## Completion gate

Before writing final values: verify all four checkpoint/data/history hashes, requested versus completed horizon, raw-state reconstruction at selected leads, finite-ensemble enstrophy identity, truth cutoff, and failure records. Rebuild the complete report and inspect both figures. Then update the appendix, provenance paragraph and reviewer-risk note; compile the paper, check the main text remains within eight pages, and inspect the new appendix pages. Preserve the old confirmation tables and selection rules unchanged.
