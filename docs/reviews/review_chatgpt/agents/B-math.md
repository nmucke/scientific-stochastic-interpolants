# B — Mathematical correctness

Scope: the complete active `main.tex` build, all active inputs, the compiled 36-page PDF, and the relevant sampler, likelihood, interpolation, analytical-control, and metric implementations. I checked mathematical correctness rather than notation aesthetics or experimental design. The exact interpolant identities are substantially sound; the findings below are the points that do not survive a literal reading or need an unstated assumption.

## Findings

### B1

- **Location —** Section 4.1, Eq. (8), p. 4: “`\middle|\,\bX_\tau=\bx,\bX_0=\bx_0`”.
- **Severity —** major.
- **Problem —** The second equality in Eq. (8) conditions on the random path source `\bX_0`, whereas the first equality conditions on the fixed physical context `\bx_0`. These are not the same for FM, where `\bX_0\sim\mathcal N(0,I)` and `\bx_0=\state^{n-1}`; conditioning additionally on `\bX_0=\bx_0` changes the conditional law of `\bx_1` and makes the displayed equality false.
- **Confidence —** high. Proposition B.1 and Eq. (37) use the intended conditioning on `\bx_0`, confirming that Eq. (8) is the outlier; no further check is needed.
- **Fix —** Replace the right-hand side of Eq. (8) by
  `\E[\conddist{\obs}{\bx_1}\mid \bX_\tau=\bx,\bx_0]`.

### B2

- **Location —** Section 4.1, Theorem 4.1, p. 4: “Let the prior model define an interpolant path”.
- **Severity —** major.
- **Problem —** The proof uses assumptions not stated in the theorem: observation noise must be jointly independent of `(\bx_1,\latentnoise)` conditional on `\bx_0`, and the conditioned endpoint must retain the moment/regularity conditions needed by Proposition A.2. Without the first condition, conditioning on `\obs` need not leave the Gaussian source independent of the retargeted endpoint, so Proposition B.1 and the `\vscoef_\tau` velocity shift can fail.
- **Confidence —** high. Appendix B explicitly invokes source/observation independence, while Theorem 4.1 does not. The synthetic experiments draw independent Gaussian observation noise and therefore satisfy this condition; checking the urban data-loading path would only be needed to document the same fact for that case, not to establish the theorem gap.
- **Fix —** Add after the first sentence of Theorem 4.1: “Assume the path satisfies the hypotheses of Proposition A.2, `\obs\perp\latentnoise\mid(\bx_1,\bx_0)`, and `\mathbb E[\|\bx_1\|^2\mid\obs,\bx_0]<\infty`.” Also state the observation-noise independence in Eq. (1), where the data model is introduced.

### B3

- **Location —** Section 4.3, Eq. (17), p. 5: “The implemented covariances are”.
- **Severity —** major.
- **Problem —** Eq. (14) is a genuine covariance only because an exact score has a symmetric Hessian. After substituting a learned drift/score Jacobian and damping it as in Eq. (17), neither symmetry nor positive semidefiniteness is guaranteed; the implementation confirms that it solves the raw, potentially nonsymmetric matrix by default. Thus the shared field-case object need not define a Gaussian covariance or Gaussian score, despite being presented as one.
- **Confidence —** high. The networks impose no integrability/symmetric-Jacobian constraint, and `sigma_bar_eig_floor` is false in the active configurations. Measuring eigenvalues on saved headline trajectories would quantify frequency/magnitude but is not needed to establish the mathematical possibility.
- **Fix —** Smallest honest text fix: replace the lead-in to Eq. (17) with “For inexact learned fields, the following damped operators need not remain symmetric positive semidefinite and are used as empirical guidance preconditioners rather than literal covariances:” and add this limitation to Section 6. If a literal Gaussian-covariance interpretation is required, symmetrise and project `\bar\Sigma_\tau` to the PSD cone, then rerun all shared-covariance field evaluations (high cost).

### B4

- **Location —** Appendix E, Eq. (52), p. 17: “We report `|1-\mathrm{ratio}|`, so `0` is perfectly calibrated.”
- **Severity —** major.
- **Problem —** The numerator averages pointwise standard deviations, whereas the denominator is a root-mean-square error. For a perfectly calibrated heteroskedastic field this ratio is `\mathbb E_i[\sigma_i]/\sqrt{\mathbb E_i[\sigma_i^2]}<1`; for example, equal numbers of points with standard deviations 1 and 3 give about 0.894, not 1. In addition, the code uses population variance (`unbiased=False`) but multiplies by `\sqrt{(E+1)/E}`, the correction appropriate to unbiased sample variance; population variance instead requires `\sqrt{(E+1)/(E-1)}`.
- **Confidence —** high. This follows analytically and is confirmed by `src/scisi/metrics/calibration.py`; the existing test covers only an almost-homoskedastic, large-`E` case and cannot detect either error.
- **Fix —** Define `s_i^2=(E-1)^{-1}\sum_e(x_i^{(e)}-\bar x_i)^2` and replace Eq. (52) by
  `\mathrm{ratio}=\sqrt{\frac{E+1}{E}\frac1{|\mathcal I|}\sum_{i\in\mathcal I}s_i^2}\,/\,\mathrm{RMSE}`.
  Change the implementation to unbiased variance and RMS spatial aggregation, then recompute Tables 1–2 and calibration curves. This is a cheap metric-only pass if member fields were retained; otherwise it requires rerunning the field evaluations. Recheck all “best calibration” and “underdispersed” statements afterward.

### B5

- **Location —** Section 4.2, Eq. (15), p. 5: “yielding the closed-form score”.
- **Severity —** minor.
- **Problem —** For state-dependent `\bar\Sigma_\tau(\bx)`, Eq. (15) is not the score of the moment-matched Gaussian: it omits the log-determinant and quadratic covariance-derivative terms. Section 4.3 discloses this immediately afterward, so the method is not hiding the approximation, but the literal derivation sentence is false.
- **Confidence —** high. Differentiating a Gaussian with state-dependent covariance produces the omitted terms; no further check is needed.
- **Fix —** Replace “yielding the closed-form score” with “and use its stopped-covariance score” and append “with `\bar\Sigma_\tau` held fixed during differentiation” to the sentence introducing Eq. (15).

### B6

- **Location —** Section 4.2, Lemma 4.2 and Eq. (15), pp. 4–5: “`\obsnoise\sim\mathcal N(0,\obscov)`”.
- **Severity —** minor.
- **Problem —** Eq. (15) uses `\bar\Sigma_\tau^{-1}`, but the lemma does not assume `\obscov\succ0` or otherwise guarantee invertibility. A degenerate Gaussian observation model can make `\bar\Sigma_\tau` singular; the stated density score then does not exist in the displayed form.
- **Confidence —** high. All reported experiments use strictly positive isotropic observation variance, so the issue is theorem scope rather than an experimental violation.
- **Fix —** Change the assumption to “`\obsnoise\sim\mathcal N(0,\obscov)` with `\obscov\succ0`.” If singular observations are intended, replace the inverse by a pseudoinverse and explicitly formulate the density on its support.

## Proof-audit table

| Result / derivation | Assumptions actually used | Omitted steps reconstructed | Audit |
|---|---|---|---|
| Lemma A.1, Eqs. (20)–(23) | `\sigpath_\tau>0`, Gaussian source independent of `\bm m_\tau`, finite endpoint second moment, differentiability of the Gaussian convolution | Differentiating the convolution gives `\nabla E[m\mid x]=\sigma^{-2}\operatorname{Cov}(m\mid x)`; scaling `\epsilon=(x-m)/\sigma` gives Eq. (23) | **Pass.** Gaussian smoothing supplies the needed open-interval smoothness. |
| Proposition A.2, Eqs. (24)–(25) | Lemma A.1, `\beta_\tau>0`, differentiable schedules; nonzero `\dot\beta\sigma-\dot\sigma\beta` only for inversion | Substitution of Eqs. (21)–(22) into Eq. (4), collection of affine and score terms, multiplication by `\beta_\tau` | **Pass.** Constants and signs agree. |
| Proposition A.3, Eqs. (27)–(30) | Continuity equation for the conditional-expectation velocity; positive spatially `C^2` density; scalar time-only `g`; existence and uniqueness for both SDE and Fokker–Planck problem | At `q=p`, diffusion contributes `+\frac12g^2\Delta p` and the added score drift contributes its exact negative | **Pass conditionally.** The conclusion is as strong as the explicit well-posedness/uniqueness assumption; `C^{1,2}` is enough for the actual cancellation. |
| Proposition B.1, Eqs. (33)–(35) | Joint source/observation-noise independence conditional on `\bx_0`; posterior finite moment; Proposition A.2 hypotheses for both paths | Factorising `p(\epsilon,x_1\mid y)` proves retained source independence; Bayes gives the score shift; subtracting the two velocity–score identities gives coefficient `\vscoef_\tau` | **Flag.** Algebra passes, but the independence and conditioned-moment assumptions are not stated with the result (B2). |
| Theorem 4.1, Eqs. (9), (40) | Proposition B.1 plus Proposition A.3 for the posterior path; endpoint continuity; correct initial tilt | Substitute `s^y=s+\nabla\log\Phi` and `v^y=v+\kappa\nabla\log\Phi`; at `\beta_0=0`, source independence makes the initial tilt constant | **Flag.** The direct argument goes through after B1/B2, but not under only the theorem’s written assumptions. The `g=0` SI case is correctly excluded by the theorem’s well-posedness clause: a unique deterministic flow cannot spread a point mass. |
| Lemma 4.2, Eqs. (12)–(13), (41)–(47) | Linear `H`; observation noise jointly independent of target/source conditional on context; finite covariance | Substitute the two interpolants, condition Eq. (44), and use conditional cross-covariance zero | **Pass.** Invertibility is needed only for the subsequent score (B6), not for these moment identities. |
| Lemma 4.3, Eqs. (14), (48) | Lemma A.1 open-interval hypotheses | Multiply the conditional mean and covariance of `\epsilon` by `\sigma` and `\sigma^2` | **Pass.** The resulting exact matrix is symmetric PSD even though its Hessian representation does not make this visually obvious. |
| SI network-to-score formula, Eq. (32) | `\sigma=\gamma\sqrt\tau`; genuine Wiener source; nonzero `\tau\gamma(\dot\beta\gamma-\beta\dot\gamma)` | `\dot\sigma\sigma=\dot\gamma\gamma\tau+\frac12\gamma^2`, so `b-v=\frac12\gamma^2s`; solving after multiplying by `\beta` gives `A_\tau` and `c_\tau` | **Pass.** Endpoint singularities are real and are avoided by initialization/terminal limits. |
| Analytical SI drift, Eqs. (56)–(57) | Gaussian target with unit covariance and Wiener-source path | Gaussian regression gives coefficient `(\beta\dot\beta+\tau\gamma\dot\gamma)/(\beta^2+\tau\gamma^2)` | **Pass.** |
| Analytical FM velocity, Eq. (58) | Rectified schedules and unit target covariance | `E[x_1\mid x]-E[\epsilon\mid x]=x_0+(2\tau-1)(x-\tau x_0)/(\tau^2+(1-\tau)^2)` | **Pass.** |
| Gaussian posterior, Eqs. (59)–(60) | Prior and noise covariance `I`, `H=I` in the actual case | Completing the square/Kalman conditioning gives mean `x_0+\frac12(y-x_0)` and covariance `\frac12I` | **Pass.** |
| Metric formulas, Eqs. (49)–(54) | Finite ensembles and nonsingular fitted covariances for KL | Checked RMSE, unbiased ensemble CRPS, both KL orientations, and finite-ensemble spread variance | **Flag.** Eq. (52) fails for heteroskedastic fields and uses the wrong implementation convention (B4); Eqs. (49), (51), (53), and (54) pass. |

## Section disposition within scope

- **Abstract, Introduction, Related work:** no additional mathematical-correctness findings; claim support and citation characterisation are outside this scope.
- **Preliminaries and Methodology:** findings B1–B3, B5–B6.
- **Results:** no additional derivation errors in the analytical posterior, SI drift, FM velocity, or Navier–Stokes equations; experimental inference/fairness is outside this scope.
- **Conclusion:** no additional mathematical finding beyond the calibration interpretation inherited from B4.
- **Appendices A–C:** proof status is recorded in the table; no algebraic sign, constant, or index error found beyond B2.
- **Appendix D:** the schedules satisfy the open-interval nondegeneracy conditions used by the displayed recovery formulas; endpoint singularities are handled numerically rather than by the theorem.
- **Appendix E:** finding B4; other displayed metric formulas pass.
- **Appendices F–I:** no additional mathematical derivation finding; baseline fidelity and experiment/code discrepancies are outside this scope.

## Possible proof gaps requiring independent Stage-2 verification

1. **Theorem 4.1 / Proposition B.1 assumption closure (B2):** independently verify the proof line by line after adding conditional source/observation independence and the posterior finite-moment hypothesis, including weak continuity at `\tau=0,1` for the SI point-source path.
2. **No algebraic proof gap found in Lemma A.1, Propositions A.2–A.3, or Lemmas 4.2–4.3.** B1 is a false conditioning expression, not a difficult proof gap; B3 is a practical learned-operator issue, not a gap in the exact covariance identity.
