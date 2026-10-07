# Editor verification of the method revisions

**Scope:** second pass by the editor who made the revisions, **not an independent review**. Checked the original adversarial review, its response, the current compiled PDF (including extracted text and resolved equation references), and relevant implementation/configuration statements. No manuscript, model, protocol or result was changed during this verification.

Snapshot: PDF SHA-256 `71afa5e069c0e665eeb550cf6fcbd58a646f815730662010a8c2e70ba0bebb74`; 21 pages, main text through page 5, references through page 6, appendix from page 7. Page numbers below refer to this snapshot and will change when the pending video figures enter the paper.

The original independent review remains unchanged: SHA-256 `16107bc2b1cef7fd8888426df9a73b0300dc9cb8d555efd08e103a8d37614cbf`.

| Review concern | Compiled evidence | Editor disposition |
|---|---|---|
| P1: strong-map existence and drift version | §3, p.2; Appendix A.1, p.7 explicitly require a regular version, pathwise unique strong solution and finite second moments | Addressed as assumptions, not established for the empirical dataset |
| P1: degenerate endpoints and diagonal | p.2 qualifies the continuous extension; A.1 restricts the regularity setting to interior times and separately assumes the forecast endpoint; zero-endpoint counterexample and quantized-video caveat are printed | No global off-support endpoint regularity is asserted |
| P1: finite-history versus true joint law | §2, p.1 and A.1, p.7 identify the surrogate and state sufficient history, correct encountered-history kernels and relevant time dependence | Addressed; no joint-physical-law guarantee added |
| P1: finite-feature PSD identification | §3.2, p.3 and A.2, p.8 explicitly reject identifiability/zero residual even under ideal optimization; minimum direct/child gaps are 0.01/0.0025 | Addressed; representation limitation is not misdescribed as optimizer failure alone |
| P2: numerical teacher versus comparator | §3.2, p.3 refers to A.2; equation (19), p.7 displays Euler drift plus exact integrated diffusion; adjacent text states left-point Euler–Maruyama comparator | Addressed; exact-diffusion solver control remains explicitly future work |
| P2: diagnostic rather than loss guarantee | §3.4, pp.3–4 is titled “Approximation Errors and Diagnostic Bounds”; text states no implemented-PSD-to-forecast-law bound | Addressed; no new convergence claim inferred from elementary identities |
| P2: average-drift objective weighting | §3.2, p.3 and A.2, p.8 state endpoint residual scaling and inverse squared interval weighting | Correct; variable-interval average-drift MSE is not called unweighted endpoint MSE |
| P3: independent Brownian process | §2.1, p.2 specifies a standard Brownian process independent of `(h,y)` | Addressed |
| P3: future increments versus absolute values | §3.1, p.3 and A.2, p.8 qualify independence before conditioning on a future bridge endpoint | Addressed |
| P3: moments | Propositions 1–2, pp.3–4 require finite second moments and square-integrable predictors | Addressed |
| P3: actual warmup/weights/EMA | A.2, p.8 prints zero warmup, unit weights, ramp100, curriculum500 from0.125, endpoint probabilities0.25/0.5, and EMA target0.95 in stage one only | Matches all twelve archived NS stages and trainer branches |
| P3: covariance conditioning | §3.4, p.4 fixes `(s,t,x,h)` and disclaims direct transfer to diverged rollout histories | Addressed |
| P3: physical forcing covariance | B.1, p.8 describes spatially correlated or discretized forcing with unverified covariance, not assumed continuum space-white forcing | Addressed |

Additional checks: the displayed numerical teacher matches `src/scisi/training/map_trainer.py`; the comparator matches `src/scisi/sampling/sde_solvers.py`. Summing the zero-drift left-point noise variance at 16, 32, 64 and 128 steps reproduces the appendix formula exactly in the CPU calculation. Direct inspection confirms the weighted loss, independent path construction, positive off-diagonal sampling gaps and frozen-teacher refinement branch. The new section/equation labels resolve in `main.aux`; no undefined-reference or overfull-box warnings occur in the build log. The existing template empty-anchor warning remains.

**Conclusion:** the requested substantive review points are resolved as manuscript assumptions, interpretation limits and implementation clarifications. No new material mathematics issue was found in this editor pass. This does not prove those assumptions for the data, identify the finite-feature PSD solution, validate the true physical joint law, or substitute for the pending exact-diffusion empirical control. The unchanged independent review remains the independent assessment; this document only checks that its requested corrections actually appear in the compiled manuscript.
