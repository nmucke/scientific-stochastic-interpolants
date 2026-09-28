# Stage 1A — Claims versus evidence

Scope: C1--C33 in `review/paper-map.md`, checked against the complete active PDF/source and the active aggregate results. Mathematical correctness and citation accuracy are not assessed here.

## Findings

### A1 — The front-page universality and exactness promise omits essential conditions

- **Location** — Abstract, p. 1; contribution 1, p. 2; Theorem 4.1, Eq. (9), p. 4. Quotes: “converts any pretrained flow-based generative model”; “samples the exact posterior when this intermediate score is known”; “turns any interpolant-based generative model into a family of posterior samplers.”
- **Severity** — major.
- **Problem** — C2, C4, and C19 state a result for “any” pretrained SI/FM/diffusion generator and condition exactness only on the intermediate likelihood score. Theorem 4.1 supports the result only for a model admitting the stated interpolant path with exact path velocity/score, a positive regular posterior-path density, and well-posed unique dynamics; Appendix A also requires nondegeneracy when recovering one field from the other. The unqualified abstract/contribution therefore promise more than the theorem proves.
- **Confidence** — high; checked against Theorem 4.1, Proposition A.2, and Appendix B. No further check is needed to establish the mismatch.
- **Fix** — Replace the two relevant abstract sentences with: “We introduce an observation-interpolant mechanism that converts pretrained stochastic-interpolant, flow-matching, or diffusion models satisfying the interpolant-path assumptions into posterior samplers without retraining. Under the regularity and well-posedness conditions of Theorem 4.1, the resulting dynamics sample the posterior of the learned prior when both its path fields and the intermediate likelihood score are exact.” Replace contribution 1 with: “Under Theorem 4.1’s assumptions, one guidance term with weight $w_\tau=\kappa_\tau+\tfrac12g_\tau^2$ yields a posterior-sampler family for an interpolant-based model.”

### A2 — “Essential for sparse observations” is not tested by the reported comparisons

- **Location** — Abstract, p. 1; Eq. (15), Table 1, Table 2, and empty Table 4. Quote: “source covariance, which is essential for sparse observations.”
- **Severity** — major.
- **Problem** — C7 is unsupported as written. Both the Jacobian-free and shared variants retain source-covariance inflation; Table 1 therefore tests the added Jacobian/prior-correlation terms, not source covariance versus no source covariance. Moreover, Table 2 shows only modest shared-versus-Jacobian-free changes in the other sparse-observation case, and the planned ablation in Table 4 contains no results.
- **Confidence** — high; all active tables and the covariance definitions in Eqs. (15)--(17) were checked. Confidence would change only if an unreported no-inflation ablation exists elsewhere in the outputs.
- **Fix** — Replace the clause with: “including prior-correlation terms that materially improve sparse Navier--Stokes assimilation.”

### A3 — The three named implementations do not differ *only* by diffusion and guidance weight

- **Location** — Introduction, unified-view paragraph, p. 2; Eqs. (6)--(7) and Eq. (9); Appendix D. Quote: “the samplers differ only in their diffusion coefficient and likelihood-score weighting.”
- **Severity** — major.
- **Problem** — C18 is only true for members chosen on one fixed interpolant path. The named/native implementations also use different source laws and prior parameterizations (point-mass/native SI drift versus Gaussian/FM velocity), and the field experiments use separate SI and FM prior networks. The current sentence collapses the theoretical one-path family and the experimentally named implementations.
- **Confidence** — high; the initial laws are explicit below Theorem 4.1 and the separate SI/FM architectures are explicit in Appendix D. A code/config audit could only refine which checkpoint serves which table row, not remove the source-law difference.
- **Fix** — Replace the sentence with: “All three use the same Gaussian likelihood surrogate; for a fixed interpolant path, family members differ through $g_\tau$ and its resulting guidance weight, while native SI and FM implementations use different source and prior parameterizations.”

### A4 — FIG and OT-ODE are precedents, not demonstrated members of this exact sampler

- **Location** — Introduction, unified-view paragraph, p. 2; Appendix F, pp. 18--19. Quote: “a deterministic guided ODE, including FIG and OT-ODE.”
- **Severity** — major.
- **Problem** — C17’s inclusion claim is not established. Theorem 4.1 contains a deterministic $g_\tau=0$ member, but Appendix F says FIG uses normalized residual correctors, a tuned magnitude, no covariance, and a corrector “not derived from a probability path”; OT-ODE also uses a different Tweedie-preconditioned weight. The paper establishes a related deterministic member, not literal recovery of those algorithms.
- **Confidence** — medium; the internal descriptions conflict with “including.” Reading the FIG and OT-ODE papers would be needed to raise confidence on literal algorithmic equivalence.
- **Fix** — Replace “including FIG and OT-ODE” with “with FIG and OT-ODE as related deterministic precedents.”

### A5 — The dense-observation “best accuracy” claim is contradicted by Table 1

- **Location** — Results §5.2, p. 7, and Conclusion, p. 9; Table 1. Quote: “our methods achieve the best accuracy under dense observations.”
- **Severity** — major.
- **Problem** — C24 is unqualified, but the $E=1000$ EnKF reference in Table 1 is better at $16^2\!\to128^2$ on both RMSE (1.442 versus best-ours 2.024) and CRPS (0.779 versus best-ours 1.062, in the table’s displayed units). The claim holds among the matched-$E=64$ comparisons, not among every row shown.
- **Confidence** — high; this is a direct numerical comparison within Table 1.
- **Fix** — In both §5.2 and the conclusion, replace with: “At matched ensemble size $E=64$, our methods achieve the best dense-observation RMSE and CRPS among the compared methods.”

### A6 — Urban “best” claims generalize from one trajectory without saying so

- **Location** — Conclusion, p. 9; Table 2 and Appendix I. Quote: “DM-SDE performs best on observed velocity and calibration.”
- **Severity** — major.
- **Problem** — C26 is numerically true for the sole reported urban trajectory, but Table 2 explicitly has `n_traj=1` and no between-trajectory uncertainty. The conclusion presents the ranking as a property of “the urban case,” which the evidence cannot distinguish from a trajectory-specific ordering; the same qualification is needed for the SI-SDE temperature ranking.
- **Confidence** — high; the caption, Appendix I, and active aggregate all state one held-out trajectory. Additional held-out trajectories would be required to support an unqualified ranking.
- **Fix** — Replace the sentence with: “On the single reported urban rollout, DM-SDE has the lowest observed-velocity error and calibration deviation, while SI-SDE has the lowest temperature RMSE.” Alternatively, run the already-described five-trajectory aggregate (roughly four additional full urban grids) before retaining the unqualified claim.

### A7 — “Prior correlations are decisive” is a causal claim not isolated by the ablation

- **Location** — Conclusion, p. 9; Eqs. (15)--(17), Tables 1--2, and Table 4. Quote: “prior correlations are decisive for sparse Navier--Stokes.”
- **Severity** — major.
- **Problem** — C28 attributes the performance difference causally to prior correlations, but shared-versus-Jacobian-free changes both the likelihood covariance and the mean-Jacobian front factor, and also introduces shared linearization/damping/cadence choices. The intended ablation table is empty; the urban half additionally rests on one trajectory. The tables support a descriptive shared-Jacobian performance gap, not this isolated causal conclusion.
- **Confidence** — high on the non-isolation; Eqs. (15)--(17) and the empty Table 4 make it explicit. A completed component ablation could raise confidence in the causal interpretation.
- **Fix** — Replace the final sentence with: “The shared-Jacobian approximation yields large SI-SDE gains in sparse Navier--Stokes and modest gains in the single urban rollout, so the best tested sampler/covariance pairing is problem-dependent.”

### A8 — Broad frequency/exhaustivity claims are supported only by examples

- **Location** — Introduction, pp. 1--2. Quotes: “learned forward surrogates themselves are typically deterministic”; “Two flow-based families dominate”; “developed model by model, each tied to whether the generator is stochastic or deterministic.”
- **Severity** — minor.
- **Problem** — C11, C12, and C14 infer prevalence or exhaustivity from a short list of examples. The paper contains no survey evidence for “typically,” “dominate,” or “each,” and the latter is especially stronger than the related-work paragraph, which first identifies a common score-decomposition principle.
- **Confidence** — medium; this is high-confidence as an internal evidence gap, but reading the cited literature would be required to determine whether an external survey supports the frequency claims.
- **Fix** — Use: “Many learned forward surrogates are deterministic and require an explicit model-error distribution for probabilistic DA”; “We focus on two flow-based families”; and “Existing posterior-conditioning methods are usually presented for a specific generator class.”

### A9 — Finite numerical agreement is described as exact recovery

- **Location** — Results §5.1, Figure 3, p. 7; Conclusion, p. 9; Table 3. Quote: “all three samplers recover the exact posterior.”
- **Severity** — minor.
- **Problem** — C23 overstates the numerical result: the reported finite-ensemble KL values are approximately $9\times10^{-4}$--$1.6\times10^{-3}$, not zero. The Gaussian closure and continuous-time idealization may be exact, but the experiment demonstrates close agreement under discretization and Monte Carlo error.
- **Confidence** — high; Table 3 reports the nonzero values directly.
- **Fix** — Replace with: “With the full covariance, all three samplers approach the exact linear--Gaussian posterior (KL $\approx10^{-3}$).”

### A10 — Sparse Navier--Stokes does not identify approximation *bias*

- **Location** — Limitations, p. 9; Figure 3 and Table 1. Quote: “bias, as seen for the Jacobian-free variant in the analytical and sparse Navier--Stokes cases.”
- **Severity** — minor.
- **Problem** — The load-bearing “can” in C30 is justified by the analytical case, where the exact posterior is known. In Navier--Stokes, Table 1 only shows an accuracy gap between approximations; without an exact posterior or isolating ablation, it does not identify that gap as bias caused by the Jacobian-free approximation.
- **Confidence** — high; the paper reports no exact Navier--Stokes posterior and Table 4 is empty. An exact/reference posterior diagnostic or completed component ablation would raise the causal confidence.
- **Fix** — Replace with: “These approximations can introduce bias, directly visible in the analytical case; the sparse Navier--Stokes performance gap is consistent with the same concern.”

No separate claims/evidence finding was found for C1, C3, C5--C6, C8--C10, C13, C15--C16, C20--C22, C25, C27, C29, or C31--C33.

## Hedge audit

- C2’s “can” is load-bearing but does not supply the structural conditions; see A1.
- C11’s “typically” is load-bearing prevalence language; see A8.
- C30’s “can introduce bias” is supported directly only by the analytical case; see A10.
- C33’s “may accumulate” is matched to the increasing urban temperature-error curves in Figure 19; no finding.
- Missing qualifications/hedges are the absolute terms “any”/“exact” (A1, A9), “essential” (A2), “only” (A3), “including” (A4), “best” (A5--A6), and “decisive” (A7).

## Per-claim status

| Claim | Status | Evidence assessment |
|---|---|---|
| C1 | Fully supported | Standard problem statement with DA references; no stronger quantitative assertion. |
| C2 | Partially supported | Eq. (3), Theorem 4.1, and Appendix A support compatible interpolant-path models, not unqualified “any pretrained” model; A1. |
| C3 | Fully supported | Proposition B.1, Eqs. (34)--(35). |
| C4 | Partially supported | Theorem 4.1 proves conditional exactness but requires exact path fields plus regularity/well-posedness, not only the intermediate score; A1. |
| C5 | Fully supported | Eq. (9) and its three displayed specializations. |
| C6 | Fully supported | Lemmas 4.2--4.3 and Eqs. (13)--(15). |
| C7 | Unsupported | No no-source-covariance comparison; reported comparison isolates a richer bundle, and urban gains are modest; A2. |
| C8 | Fully supported | Both approximations execute on $128^2$ field states; Tables 1--2 report finite costs. |
| C9 | Fully supported | §§5.1--5.3 and Appendices G--I. |
| C10 | Fully supported | Standard EnKF/PF characterization with DA references; internal tables are illustrative only. |
| C11 | Partially supported | Cited examples do not establish “typically” across learned surrogates; A8. |
| C12 | Unsupported | No evidence for the field-wide prevalence claim “dominate”; A8. |
| C13 | Fully supported | Eqs. (5)--(6), Proposition A.3, and Appendix A.1 provide the stated FM-score/SDE lift. |
| C14 | Partially supported | Related work gives model-specific examples but does not support the exhaustive “each tied” formulation; A8. |
| C15 | Fully supported | Eq. (3), Appendix A, and interpolant references. |
| C16 | Fully supported | Proposition B.1. |
| C17 | Partially supported | Theorem 4.1 supports a deterministic guided member, but literal inclusion of FIG/OT-ODE is not established and conflicts with Appendix F’s distinctions; A4. |
| C18 | Partially supported | One-path theory supports the $g_\tau/w_\tau$ distinction; named/native implementations also differ in source/prior parameterization; A3. |
| C19 | Partially supported | Theorem 4.1 supports the weighting under explicit assumptions, not the unqualified “any” formulation; A1. |
| C20 | Fully supported | Exact conditional moments in Lemmas 4.2--4.3 followed by the stated Gaussian closure. |
| C21 | Fully supported | §4.3 gives the reduced Jacobian cost and Tables 1--2 demonstrate field-scale execution. |
| C22 | Fully supported | Theorem 4.1, Eq. (15), and Algorithm 1. |
| C23 | Partially supported | Table 3 shows KL near $10^{-3}$, not literal finite-run equality; A9. |
| C24 | Contradicted elsewhere | The $E=1000$ EnKF reference beats ours in the dense $16^2$ columns of Table 1; A5. |
| C25 | Fully supported | Sparse SI-SDE RMSE improves from 6.969 to 1.731 and 8.500 to 3.854 in Table 1. |
| C26 | Partially supported | Ranking holds in Table 2 but only for one trajectory; A6. |
| C27 | Fully supported | Table 2 gives approximately 3.1--4.2× lower cost with small metric changes. |
| C28 | Partially supported | Tables show shared/Jacobian-free gaps, but do not isolate prior correlations causally; urban evidence is one trajectory; A7. |
| C29 | Fully supported | Theorem assumptions and §§4.2--4.3 state the ideal/practical split. |
| C30 | Partially supported | Analytical bias is directly measured; sparse Navier--Stokes bias is inferred from an accuracy gap; A10. |
| C31 | Fully supported | The practical surrogate assumes linear $H$ and Gaussian noise in Lemma 4.2. |
| C32 | Fully supported | §4.3 states $N_y$ JVPs and an $N_y\times N_y$ factorization/solve. |
| C33 | Fully supported | Eq. (2), Algorithm 1, and Figure 19 support the autoregressive limitation and hedged accumulation claim. |

## Unsupported claims requiring Stage-2 evidence search

1. **C7** — Search active outputs for a direct no-source-covariance ablation. Shared versus Jacobian-free is not that ablation; if none exists, narrow the abstract as in A2.
2. **C12** — Search the cited survey/foundational literature for evidence that SI and FM “dominate.” If no explicit survey supports prevalence, use the scope statement in A8.
