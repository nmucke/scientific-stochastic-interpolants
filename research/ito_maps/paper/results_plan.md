# Completed manuscript integration record

The frozen study and manuscript integration are complete. All six students completed both stages: 2,000 teacher-diagonal progressive self-distillation updates, followed by 500 same-path teacher-rollout refinement updates. The confirmation audit verifies all twelve training stages, frozen checkpoint hashes, and 4,360 forecast rows. There were no test-driven model or feature changes.

## Evidence and final interpretation

- Full seed-zero sampling curves: maps at 1/2/4/8 generative steps and both teachers at 1/4/16/64/128 steps. These appear separately from training-replication summaries.
- Confirmation: both map families trained with seeds 0, 1, and 2, evaluated at the frozen 1/8-step budgets. Teachers and hyperparameters stay fixed. The one-interval comparison uses all 20 test trajectories, origins 5/45/85, and 16 ensemble members. Ten-lead autoregression uses origin 45 and each member's own generated history.
- Eight-step Itô map versus SI64: three-student mean CRPS is 1.34% higher at one interval and 1.74% higher at physical lead ten, with 7.16× and 7.26× lower measured latency respectively. The final-lead paired trajectory interval excludes zero. Residual underdispersion and 14.90% higher final spectral discrepancy qualify quality retention.
- One-step Itô map versus SI16: mean CRPS improves by 4.73%, at about 13.24× lower latency. This does not imply improvement on the more resolved SI64/128 references.
- The deterministic one-step map is a strong comparator: mean CRPS is 0.61% above FM64 at one interval and 0.08% above it at lead ten, with 56.32× and 58.92× lower latency. The one-interval paired interval barely excludes zero; the final-lead interval crosses zero, without establishing equivalence. Its final calibration diagnostics are mixed, so lower coverage alone is not labeled worse calibration.
- The K16 development ablation improved teacher-supervised validation loss without improving forecast CRPS. Selection retained K4 before testing. Equal budgets and checkpoint ancestry were controlled; differently shaped projections consume different RNG streams, so this remains a single-seed exploratory ablation.

No forecast ensembles are pooled across training seeds. Mean scores average the separately scored students. Reported sample standard deviations describe three student trainings with fixed teachers. Paired 95% data intervals resample 20 trajectories after averaging the trained-student scores; they are distinct from training variation and exclude teacher-training uncertainty. Intervals are pointwise and unadjusted for multiple comparisons. Empirical 5–95% coverage with 16 members is not expected to equal 90% even under calibrated sampling.

## Source artifacts

Authoritative numerical sources are `results/final_report/{one_step,rollout}/`, especially `student_training_seeds/training_seed_report.json`, `per_student_seed.csv`, and `fixed_teacher_comparison.json`. These derive from the audited raw evaluation JSONs. `notes/heldout_three_seed_assessment.md` gives the complete independent interpretation. `notes/heldout_seed0_assessment.md` remains the detailed record for the seed-zero curves, not the final aggregate claim source.

`results/training/{ito,flow}_{psd,rollout}_seed{0,1,2}/history.json` supplies recorded stage times. These include training, validation and teacher evaluation after model/data initialization; they exclude startup, teacher pretraining, development exploration and final evaluation. Both stages total 18.99–19.05 minutes per Itô student and 18.44–18.53 minutes per flow student on the measured RTX 3090. They are not pure GPU-kernel times or full end-to-end job costs.

The fixed qualitative case is the tracked `results/test_qualitative/forecast_fields.png` and companion provenance JSON: trajectory 180, origin 45, first ensemble member, fixed checkpoints and sampling seed. No sample selection or resampling occurred during manuscript integration.

## Reproducible manuscript build

Run `make -C research/ito_maps/paper`. The build regenerates ignored figures and derived tables from tracked artifacts before LaTeX. `PYTHON` defaults to the repository virtual environment; override with a compatible NumPy/Matplotlib environment if needed.

- `build_validation_table.py` regenerates the development ablation table and source hashes.
- `build_compression_figure.py` regenerates the one-history conditional Brownian compression figure. This is a teacher-path diagnostic, not a forecast-quality guarantee.
- `build_seed0_results.py` produces selected seed-zero score and uncertainty tables, with raw-source hashes.
- `build_study_figures.py` regenerates the full seed-zero cost/quality and physical-rollout figures, and explicitly seed-zero numerical macros. Rollout speed ratios use summed transition costs.
- `build_replication_tables.py` strictly requires complete seeds 0/1/2, both horizons, 20 trajectories, E16, and all twelve stage histories before writing anything. It generates the aggregate and individual-student tables, paired comparisons, recorded stage times, final numerical macros, and source hashes. `make replication-artifacts` rebuilds this subset.

`generated/seed0_claims.tex` and `generated/replication_claims.tex` use distinct macro prefixes to avoid substituting seed-zero estimates for three-student claims. JSON/CSV source hashes are recorded beside generated outputs. The final report CSV files use LF line endings; manuscript source hashes were regenerated after normalization.

## Remaining scientific and submission limitations

There are no pending frozen experiments. Remaining limitations are substantive scope, not incomplete runs: one stochastic-fluid archive, three student seeds, fixed teachers, absent teacher-pretraining costs, uncertified exact physical archive generation settings, no direct external consistency/adversarial-distillation baseline runs, and no physical conditional reference ensembles. The stochastic map does not show a general advantage over deterministic maps. The theoretical Gaussian illustration and conditional-variance proposition explain distinctions; they are not novel general map-learning theory.

The main manuscript stays within eight pages before references/appendices and uses an explicit page break before the appendix. Keep the official style unmodified. Before actual submission, human authors must review all content and disclosure, verify the intended conference cycle's current kit and deadlines, check anonymity, and freeze the submission artifacts. See `guidelines.md` and `reviewer_risks.md`.
