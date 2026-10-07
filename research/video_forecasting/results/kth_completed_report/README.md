# KTH-only completed-dataset development report

**Scope:** all five KTH families and all prespecified KTH budgets are complete and audited. CLEVRER is still running; this is not the canonical two-dataset comparison and is not yet included as a complete manuscript result.

Four model families are evaluated using fixed rolling context, independently trained dataset-specific teachers, and matched forecast cases. These initial results establish an executable benchmark; they do not establish convergence or equivalence to the published latent-space results.

- [KTH results](kth/README.md)

FVD comparisons are valid only within a dataset and this recorded protocol. The combined CSV is for record keeping, not an aggregate score across datasets. Image metrics use clipped copies; proper scores and calibration retain out-of-range forecasts. Oracle best-member metrics in the CSV are explicitly separated from average-completion scores.
