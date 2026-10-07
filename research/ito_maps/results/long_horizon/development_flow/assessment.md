# Preliminary flow-map diagnostic review

This review uses a fixed copy of the in-progress diagnostics through **lead 2,576**, not the final 4,096-lead result. The production run continued independently. See `input_snapshot/flow_map/` and `report_manifest.json` for the reviewed records.

All eight members remain finite in this snapshot. Ensemble-mean member enstrophy stays between 2.861 and 5.362. Its average is 3.859 over the first 95 forecast leads and 3.920 over the last 512 reviewed leads. RMS spread averages 2.798 over the latter interval; maximum absolute vorticity over all reviewed fields remains between 9.596 and 16.232.

This looks empirically bounded rather than progressively collapsing over the reviewed interval. The late mean enstrophy is approximately 16% below the pooled held-out archive mean of 4.665, suggesting reduced amplitude relative to that contextual distribution. The archive is not a matched long continuation, so this comparison does not prove a biased invariant distribution. Enstrophy of the ensemble-mean field averages only 0.487 late in the rollout; this is principally a different statistic from member enstrophy, not evidence that individual members have decayed to the same small amplitude.

Exactly 95 leads contain truth, RMSE and corrected spread/skill. All subsequent RMSE and ratio fields are null. The ratio equals sqrt(9/8) times unbiased RMS spread divided by RMSE to numerical precision; its average across the 95 correlated leads is 0.944. This single-history temporal average is not an independent-case calibration estimate.

Checks performed on CPU:

- The finite-ensemble enstrophy decomposition holds at every reviewed lead, with maximum absolute discrepancy 2.22e-15.
- Independently loading normalized raw states at leads 1, 95, 100, 1,024 and 2,560 and applying the recorded inverse affine transform reproduces enstrophy and RMS spread to approximately 1e-15.
- Both generated plots were visually inspected. The long-horizon panels show the end of available truth explicitly, and the separate spread/skill plot stops at lead 95.
- A report-schema mismatch was corrected: the production runner records `ensemble_size`, while the original reporting draft expected `E`. The reporting tool now accepts the production field. No production record, checkpoint, protocol or runner was changed.

The report is operationally compatible with the production artifacts. Comparing all four complete trajectories remains necessary before making a model comparison or updating the manuscript.
