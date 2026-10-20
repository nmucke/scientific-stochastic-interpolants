# Results plan: Navier--Stokes placeholders to fill

The initial-stage results are archived and not used (`../archive/results/`). The
new stochastic Navier--Stokes experiments will be run with the current
configuration of record (stage 1: 2,000 teacher-diagonal PSD updates, EMA 0.95,
interval curriculum 0.125 → 1 over 500 updates, endpoint probability 0.25;
stage 2: 500 same-path teacher-rollout updates, N = 32, full intervals,
endpoint probability 0.5). Every result-dependent item in the manuscript is a red
`\pending{}` / `\pendingbox{}`. Search `pending` in the `.tex` files to list them.

## Planned evaluation

- Data: 200 trajectories, 160/20/20 split, five-field history, 128 × 128 vorticity.
- Methods: SI and FM Euler samplers at 1/4/16/64/128 steps; Itô and flow maps at
  1/2/4/8 steps; persistence. Ten-lead rollouts at map budgets 1 and 8 and solver
  budget 64.
- One interval: 20 test trajectories × origins 5/45/85, 16 members. Ten leads:
  origin 45, 16 members, each member's own generated history.
- Students: seeds 0, 1, 2 with fixed teachers; seed 0 gives full budget curves.
- Metrics: fair CRPS (primary), energy score, ensemble-mean RMSE, RMS spread,
  corrected spread–skill, empirical C90, extreme-rank fraction, spectral
  discrepancy, synchronized ensemble latency. Paired 95% trajectory-bootstrap
  intervals (10,000 resamples) and student-seed SDs reported separately.
- Development (validation only): PSD versus refined checkpoints; K = 4 versus 16.
- Diagnostics: fixed-history Brownian compression (K = 1/4/16); 4,096-lead
  rollouts from trajectory 180, eight members, maps plus SI64/FM64.

## Placeholder locations

| Location | Pending item |
|---|---|
| Abstract | Headline quality–cost result |
| §3.4 `sections/ito_maps.tex` | Omitted-variance fractions at K = 4, 16 |
| §5.1 `sections/experiments.tex` | Table `tab:replication` (main comparison); main findings paragraph |
| §6 `sections/limitations.tex` | Long-horizon summary; result-specific limitations |
| App. A `appendix/implementation.tex` | Confirmation audit; Table `tab:replication-time` (training cost) |
| App. D `appendix/ns_results.tex` | Fig. `fig:quality-cost`, Table `tab:seed0-test`, Fig. `fig:rollout`; seed-zero, rollout, three-seed, ablation and cost paragraphs (incl. parameter counts) |
| App. E `appendix/replications.tex` | Tables `tab:individual-seeds`, `tab:replication-paired`; discussion |
| App. F `appendix/development_ablation.tex` | Table `tab:validation-development`; K ablation outcome |
| App. H `appendix/seed0_uncertainty.tex` | Table `tab:seed0-calibration`; paired calibration differences |
| App. I `appendix/qualitative_forecast.tex` | Fig. `fig:qualitative` |
| App. J `appendix/compression_diagnostic.tex` | Fig. `fig:ns-compression`; omitted variances, SEs, fractions; retained-SD statement |
| App. L `appendix/long_horizon.tex` | Table `tab:long-maps`; Figs. `fig:long-spectrum-first`, `fig:long-spectrum-last`, `fig:long-maps-skill`; behavior and spectra paragraphs |

## Interpretation rules (carry over)

- No forecast ensembles pooled across training seeds; student SDs and paired data
  intervals are distinct uncertainties; neither covers teacher-training variation.
- No equivalence margin is declared: near-zero differences are not equivalence.
- Empirical 5–95% coverage with 16 members is not expected to equal 90% even
  under calibrated sampling.
- Same-path teacher fidelity and forecast scores measure different objects.
