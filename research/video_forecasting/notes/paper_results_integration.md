# Video paper results integration

The qualitative comparison was fixed before completed all-model results; see
`paper_visual_selection.json` for the dated selection and the sole completed
source evaluation available at declaration. Use the first frozen case, member
zero, SI64/FM64/Itô8/Flow1/persistence1. Numerical tables retain every evaluated
budget. Do not change the qualitative choices in response to later scores.

Once all evaluations and both dataset audits have passed, first generate the
existing strict report in `research/video_forecasting/results/report` with
`research/video_forecasting/experiments/report_video_results.py`, including all
four models and persistence for both datasets. Then run, from the repository:

```sh
.venv/bin/python research/video_forecasting/experiments/verify_video_results.py --dataset kth
.venv/bin/python research/video_forecasting/experiments/verify_video_results.py --dataset clevrer
.venv/bin/python research/video_forecasting/experiments/report_video_results.py \
  --evaluations research/video_forecasting/results/test/{kth,clevrer}/{follmer_stochastic_interpolant,flow_matching,follmer_ito_map,flow_map,persistence}/evaluation.json \
  --output-dir research/video_forecasting/results/report
.venv/bin/python research/ito_maps/paper/build_video_results.py \
  --report-dir research/video_forecasting/results/report \
  --output-dir research/ito_maps/paper/figures/video_results \
  --snippet research/ito_maps/paper/generated/video_results.tex
```

This builder does no independent metric aggregation. It reads the reporter CSV,
checks evaluation/source-report hashes and the frozen-protocol audit, copies the
report's FVD-versus-cost figure, and renders raw saved qualitative NPZ arrays.
Clipping is restricted to displayed image copies. It creates
`paper/figures/video_results/` with three metric tables per dataset, cost figures,
fixed examples, unrounded CSV copies and a provenance manifest, and the snippet
`paper/generated/video_results.tex`. The existing `paper/video_case.tex` includes
the snippet only when it exists; otherwise it retains the pending-evaluation
paragraph. No complete-result snippet has been created during development.

The strict builder requires both datasets and all four model families plus
persistence. Missing/stale audits or changed raw evaluation JSON fail before
output creation. Missing qualitative artifacts also fail, without writing the
final snippet. `--allow-incomplete` requires separate, explicitly noncanonical
asset and snippet paths and labels the fragment as incomplete development
output. The dated selection record is hashed into the generated provenance.

Validation on 2026-10-07: eight focused CPU tests cover complete-artifact builds,
missing dataset/model/persistence rejection, development-report rejection,
stale protocol audit and changed evaluation rejection, canonical destination
protection, and first-case/common-truth qualitative identity. The completed KTH
SI report was rendered separately under `/tmp/video_paper_si_assets`; its figure
was visually reviewed and its LaTeX fragment compiled at the manuscript text
width without overfull/underfull warnings. These partial development assets are
not manuscript results.

## Main-text video comparison (user-requested update)

The same strict builder now also writes `generated/video_main_results.tex`
(default: beside `--snippet`) and `figures/video_results/main_fvd_table.tex`
plus `main_forecast_sequences.{pdf,png}`. `video_main.tex` includes the main
snippet only after it exists. Override with `--main-snippet` for an isolated
layout preview; development builds cannot target either canonical snippet.

The main table preserves pooled FVD as the estimator analogous to the original
paper and reports equal-count member-FVD mean/SD alongside it. It retains all
nine model/budget combinations for each dataset. Member SD is completion
variability, not a confidence interval. Pixel resolution, observed-prefix
inclusion and the smaller real/generated counts prevent numerical benchmark
reproduction claims.

The main figure has matched side-by-side KTH/CLEVRER panels. Each contains truth
and all four model families. A blue-bordered observed column repeats the last
observed frame; three future columns show first, middle and final leads from
the previously fixed sequence. The smaller display subset and layout were
recorded in `paper_main_visual_selection.json` before completed CLEVRER results.
Case zero, member zero and original model budgets remain unchanged. Predictions
are individual sampled completions, never ensemble-mean images. Retained-frame
and original-source frame indices are saved in the generated provenance.
Persistence and five future display times remain in the full appendix panel.

A real completed KTH-only layout preview lives under `/tmp/video_main_kth_preview`
and is explicitly incomplete. It is not included in the manuscript. CPU tests
also verify that the main panel actually uses member zero rather than averaging
members, produces both grayscale and RGB panels, and retains both datasets and
all budgets in the main FVD table.
