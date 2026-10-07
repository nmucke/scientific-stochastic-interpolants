# Response to independent method and mathematics review

Revision date: 7 October 2026. The independent review `adversarial_method_math_2026-10-07.md` is preserved unchanged. This response records targeted manuscript changes, not a new independent review. No model code, checkpoints, frozen protocols, metric values or completed experimental results were changed.

Revised sources are `../paper/main.tex` and the new `../paper/method_assumptions.tex`. The latter is included within Appendix A. The scope remains a conditional forecasting adaptation and approximation study, with diagnostic identities rather than a new convergence theorem.

## P1: strong solution, drift versions and degenerate endpoints — addressed

The main map definition now assumes a regular drift version, a pathwise unique strong solution and finite second moments. Its diagonal statement is explicitly a continuous extension where that extension exists, rather than an evaluation of the undefined average-drift quotient at zero gap.

Appendix A.1 separates the ideal marginal-construction assumptions from strong-map well-posedness, gives familiar interior regularity conditions, and treats the forecasting endpoint transition as an additional assumption. It does not assert a continuous endpoint diagonal for arbitrary off-support states. The review's deterministic-zero example is used briefly to show that the nonsingular regression target does not itself ensure bounded drift regularity. Quantized video observations are explicitly not asserted to satisfy smooth-density assumptions without a continuous observation model.

## P1: one-step conditionals versus physical trajectory laws — addressed

The forecasting setup now calls repeated finite-history sampling a surrogate process. Appendix A.1 states the sufficient-history conditional-law identity and the need for a correct kernel on encountered histories and relevant time dependence. It identifies occlusion and coarsening as possible sources of omitted information. Fresh independent randomness correctly samples the chosen kernel; it does not establish that the kernel contains the system's full memory. Neither generative composition nor long-run boundedness is claimed to certify the physical joint law.

## P1: finite-feature PSD identification — addressed

The qualification now appears directly beside the training objective: compressed-feature PSD with restricted intervals is approximate self-distillation, with no asserted identifiability or zero population residual even under ideal optimization. Appendix A.2 explains the final direct-gap lower bound 0.01 and child-gap lower bound 0.0025, the absence of enforced refinement to zero, and the additional limitations of EMA targets and finite features. The existing fixed-conditioning compression caveats remain. No new theorem or experiment is used to circumvent this limitation.

## P2: numerical teacher differs from baseline diffusion integration — addressed; empirical control deferred

The main method now names Euler drift plus exact integrated-diffusion increments and points to the explicit update in Appendix A.2, equation `eq:numerical-teacher`. The comparator's left-point Euler–Maruyama noise term is stated separately. The appendix includes the zero-drift noise-variance calculation to make the numerical distinction concrete, explicitly without interpreting it as a prediction for fluid variance.

An exact-diffusion solver control remains future empirical work. The text acknowledges its importance for separating learned-map benefit from a cheap quadrature change. No frozen baseline is silently replaced, and no new result is claimed from this calculation. The model-selection description cross-references the recorded teacher rule.

## P2: coupling/projection identities are not optimized-loss guarantees — addressed

The main subsection is renamed **Approximation Errors and Diagnostic Bounds**. The endpoint proposition requires finite second moments, and its interpretation explicitly states that it does not bound forecast-law error by the implemented PSD objective. Appendix A.1 identifies the propositions as elementary coupling/projection diagnostics rather than new optimizer-specific convergence or calibration theorems. The optional additional Wasserstein decomposition was not added, as requested; the existing error distinctions are sufficient for the intended accessible presentation.

## P2: average-drift weighting and complete objective — addressed

The main objective now states its inverse-square interval weighting of endpoint residuals. Appendix A.2 gives the combined weighted objective, explains the cancellation and `(t-s)^(-2)` factor, and distinguishes PSD from teacher-regression stages.

Recorded settings were verified across all six NS PSD configurations and all six refinement configurations:

| Setting | PSD | Teacher refinement |
|---|---:|---:|
| Diagonal weight | 1 | 1 |
| Maximum off-diagonal weight | 1 | 1 |
| Weight ramp | 100 updates | None |
| Interval curriculum | 500 updates, maximum gap 0.125 → 1 | Full gap range immediately |
| Endpoint-atom probability | 0.25 | 0.5 |
| Diagonal-only warmup | 0 | 0 |
| Target | EMA, decay 0.95 | Frozen numerical teacher |

The refinement archive's EMA copy has decay 0.99, but it does not define the teacher-rollout target; the prose explicitly distinguishes that copy from the target. Average-drift units align finite-time and diagonal supervision without making the variable-interval objective identical to unweighted endpoint MSE.

## P3 clarifications — addressed

1. **Brownian independence:** `W` is now a standard Brownian process independent of the forecast pair, not merely a collection of Gaussian marginals.
2. **Future independence:** the statement concerns increments after the start, before conditioning on revealed future bridge values; absolute future Brownian values are not claimed independent of an adapted state.
3. **Diagonal limit:** the ideal extension is qualified, and learned diagonal regression is distinguished from exact equality to the population drift. Zero added branches do not imply a perfect backbone.
4. **Moments:** the endpoint proposition requires finite second moments; the compression proposition requires square-integrable predictors.
5. **Warmup:** the main text no longer implies a separate warmup occurred. Appendix A.2 records zero steps in both actual NS stages and distinguishes optional warmup, the loss ramp and interval growth.
6. **Covariance:** the identity and its spread interpretation explicitly fix `(s,t,x,h)`; they are not directly transferred to multi-step rollouts with different histories.
7. **Physical forcing:** the NS display is described as spatially correlated or discretized forcing with unverified covariance, rather than an unstated continuum space-white model.

## Validation and retained scope

`make -C research/ito_maps/paper` succeeds. The integrated PDF currently has 21 pages after the separately requested case-study restructuring; main text ends on page 5, references on page 6 and Appendix A starts on page 7. Space is reserved for the audited video-results table and forecast-sequence figure. The mathematics-only revision was also verified to fit the original eight-page main-text boundary before that restructuring. No undefined equation/cross-references or overfull boxes appear in the final LaTeX log; the existing template empty-anchor warning is unchanged. Labels for the new assumptions and numerical-teacher sections resolve. The earlier OU illustration remains covered by the linear-SDE appendix; redundant main-text exposition and the composition-derivative discussion were moved or condensed to preserve the eight-page limit without deleting the substantive caveats.

These revisions narrow theoretical interpretation and clarify the implemented numerical target. They do not establish an exact semigroup, correct long-time physical law, retained calibration, or complete attribution of speed–quality differences. Those limitations remain visible alongside the existing mixed empirical findings.
