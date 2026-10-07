# Completed map trajectories: partial model comparison

Both maps completed 4,096 forecast leads with eight finite members throughout. This is **not the complete four-model experiment**: the two solver references remain pending. No final manuscript result is inferred here.

| Diagnostic | Flow map, 1 step | Itô map, 8 steps |
|---|---:|---:|
| Member-mean enstrophy, average over leads 3,585–4,096 | 3.8366 | 1.4047 |
| Raw RMS spread, same late window | 2.7679 | 1.6605 |
| Enstrophy of ensemble mean, same late window | 0.4790 | 0.1942 |
| Member-mean enstrophy, range over all 4,096 leads | 2.8607–5.3623 | 0.9452–4.3991 |
| Maximum absolute vorticity observed anywhere | 16.7300 | 13.1287 |
| Mean member enstrophy over truth-supported leads 1–95 | 3.8592 | 1.9777 |
| RMS ensemble-mean error pooled over leads 1–95 | 3.0789 | 3.0148 |
| Corrected RMS spread pooled over leads 1–95 | 2.8901 | 2.0226 |
| Ratio of those pooled RMS quantities | 0.9387 | 0.6709 |
| Mean of the 95 separate corrected spread/skill ratios | 0.9442 | 0.6772 |

Pooling in this table means taking the square root after averaging the squared per-lead quantity. The final row is a different statistic and is labeled separately. Neither treats the correlated leads from this single initial history as independent calibration cases.

The Itô trajectory remains numerically bounded but loses substantial member enstrophy and spread. Its late enstrophy average is about 70% below the contextual archive mean of 4.6654; the flow map is about 18% below it. More directly, the available **paired truth** has mean enstrophy 4.3526 over leads 1–95, versus 1.9777 for Itô and 3.8592 for the flow map: approximately 55% and 11% lower, respectively. Thus the early Itô amplitude loss is visible within the reference-supported interval, not solely a comparison with unrelated late-time archive samples.

The flow map fluctuates around a substantially higher long-run amplitude and spread. The Itô map settles into a smaller-amplitude range rather than showing unbounded growth or complete extinction. The similar first-95 ensemble-mean RMSE values do not negate the marked spread difference: a low-amplitude ensemble mean can have modest RMSE while its ensemble misses forecast variability. No assertion of a correct invariant distribution follows for either model, and the pending solver runs are essential to distinguish map loss from inherited teacher behavior.

The two generated figures were inspected. Both show the actual raw diagnostics, with no temporal smoothing. The member bands are ensemble quantiles. Exactly 95 rows per model contain paired truth and skill; all subsequent skill values are null. The corrected ratios reproduce `sqrt(9/8) * unbiased_RMS_spread / RMSE` exactly. The finite-ensemble enstrophy identity has maximum residual below 3e-15 across all 8,192 model/lead records.

Independent CPU recomputation from saved raw fields at leads 1, 95, 100, 1,024 and 4,096 reproduces both enstrophy and spread within 2e-15. The inverse transform uses the recorded scale, not rounded handwritten values. Production protocols, checkpoints, runner and generated trajectories were not changed. Exact derived numbers are in `assessment_numbers.json`; input hashes are in `report_manifest.json`.
