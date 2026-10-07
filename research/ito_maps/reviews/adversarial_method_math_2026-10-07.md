# Independent adversarial review: method and mathematics

**Reviewed:** 7 October 2026, manuscript `research/ito_maps/paper/main.tex`, SHA-256 `56b9dde4762c20047bf1bd3b8e7f0174e860105929d7d9ab4dd2563a6fdf2315`; repository HEAD `3421033`. Equation numbers below are from the current `main.aux`; source lines refer to this snapshot. The review also inspected the compression appendix, forecast algorithm, model classes, Brownian sampler, map trainer, baseline sampler, and archived training configurations. No production code or manuscript was changed. No GPU work was performed.

**Scope:** Mathematical validity, identification of the object being learned, approximation claims, and correspondence between equations and implementation. Experimental rankings, new video results, and unfinished long-horizon results are deliberately excluded.

## Overall assessment

The central construction is mathematically coherent. I found **no fatal sign error, missing Itô correction, incorrect shared-noise composition formula, or false stated proposition**. Several pitfalls that would ordinarily undermine this paper are already addressed explicitly: the forecasting drift differs from the Gaussian-base drift; the repository's Föllmer naming does not imply KL-optimality; finite Brownian features do not identify the full path; and teacher refinement is simulation-based.

The strongest remaining weaknesses are **unstated regularity assumptions**, **the absence of a sufficient-history assumption for interpreting physical rollouts**, and **an insufficiently explicit definition of the numerical teacher in the main method**. The theoretical section gives valid diagnostics, but it does not prove that the actual PSD objective identifies the desired transition, controls endpoint distribution error, or preserves calibration. The manuscript mostly admits this; its contribution should remain an adaptation and careful approximation study, not a new general consistency theorem.

| Priority | Issue | Classification | Confidence |
|---|---|---|---|
| P1 | Existence of a strong Itô map and regularity at the degenerate endpoints are assumed but not stated | Substantive assumption gap; concrete counterexample to a possible stronger interpretation | High |
| P1 | Correct one-step last-p-history conditionals do not establish correct physical trajectory laws | Missing modeling assumption, especially relevant to video and long horizons | High |
| P1 | Finite-feature PSD is not an identified exact-semigroup learning problem | Known, correctly acknowledged limitation; still the main methodological validity threat | High |
| P2 | Numerical teacher uses exact integrated diffusion, while the solver comparator uses Euler–Maruyama diffusion | Definition/attribution gap; already disclosed in an appendix | High |
| P2 | The endpoint proposition is a coupling inequality, not a training-loss guarantee | Claim-strength and novelty issue, not a false theorem | High |
| P2 | Average-drift loss weights endpoint errors by inverse squared interval length | Underspecified objective interpretation | High |
| P3 | Brownian independence, integrability, and the domain of the diagonal need precise qualifiers | Formal clarity | High |

P1 means address before submission; it does not imply the current implementation is invalid. P2 means make the distinction explicit so a reviewer cannot reasonably infer a stronger claim. P3 is a small correction.

## 1. State the strong-solution and endpoint assumptions

**Location:** Section 2.1, Eqs. (1)–(4), `main.tex:47–72`; Section 3 opening and Eqs. (5)–(6), `main.tex:77–89`.

The sentence that a solution is determined by the initial state and Brownian realization assumes strong existence and pathwise uniqueness. Equality of one-time marginals from a Fokker–Planck argument does not, by itself, supply those properties for arbitrary drift versions. The conditional expectation defining the drift is initially defined only almost everywhere under the interpolant distribution. A transition map for arbitrary starting states needs a specified regular version and appropriate well-posedness assumptions.

The original forecasting paper assumes positive conditional densities and imposes schedule conditions beyond endpoint values. Its Theorem 3.1 supplies the marginal construction; its endpoint discussion is relevant here. These assumptions should be stated for the ideal population setting rather than imported only through “under the assumptions of” a citation. They do not literally describe an empirical collection of quantized video frames. [Chen et al., Sections 3.1–3.2 and Appendix B](https://arxiv.org/html/2403.13724v2).

**Concrete counterexample, derived independently.** Take scalar `c(h)=0`, deterministic target `y=0`, and the manuscript's schedule. Then

\[
I_t=\varepsilon(1-t)W_t,\qquad R_t=-\varepsilon W_t,
\qquad b_t(x)=-\frac{x}{1-t}\quad (0<t<1).
\]

The regression target has no singular coefficient and has bounded second moment, but the conditional drift has Lipschitz constant `1/(1−t)`, which diverges at the endpoint. For a transition beginning from arbitrary `x` at time `s`, its endpoint at time 1 is zero. Therefore

\[
G_{s,1}(x;W)=\frac{-x-(M_1-M_s)}{1-s}.
\]

For fixed nonzero `x`, this diverges as `s→1`. A continuous diagonal extension on the entire closed time triangle and all states does not follow from the bounded target. This example violates the positive-density assumption, so it is **not** a refutation of the cited forecasting theorem. It shows why those assumptions cannot be silently omitted, particularly once pixel videos are included.

**Correction:** Add a short assumption paragraph. For ideal identities, assume a regular version of `b_t(·;h)` with strong existence, pathwise uniqueness, finite second moments, and the continuity needed for the diagonal limit, on the stated time/state domain. Local Lipschitz continuity plus growth/nonexplosion conditions on compact interior time intervals is a familiar sufficient framework; explicitly handle any extension to `t=0,1`. Distinguish this ideal reference from the finite learned neural teacher actually integrated. Say that a nonsingular regression target is an optimization convenience, not a theorem that the population drift is globally regular.

Do not claim that quantized video data satisfy a smooth-density theorem without specifying a continuous observation model or dequantization. It is sufficient to say that finite-data training approximates a continuous predictive model and endpoint guarantees are idealized.

## 2. A one-step conditional kernel is not automatically the true trajectory law

**Location:** Section 2, `main.tex:40`; Algorithm 1; limitations relevant to the new long-horizon and video cases.

The algorithm correctly produces the joint law of its learned order-p transition kernel. To equate that with the true physical joint conditional law requires the selected history to be sufficient:

\[
\mathcal L(Y_{n+1}\mid Y_{\le n})
=\mathcal L(Y_{n+1}\mid Y_{n-p+1:n}),
\]

with a time-homogeneous kernel if absolute time or changing forcing is not included. Perfect one-step fitting under the observed-history distribution is insufficient otherwise. Generated histories also need not have the same distribution as the histories on which the kernel was fitted.

**Concrete counterexample, derived independently.** Draw one hidden variable `Θ`, equally likely to be `0.1` or `0.9`, for an entire sequence. Conditional on `Θ`, observations are independent Bernoulli(`Θ`). A perfect last-one-observation model has

\[
P(Y_{n+1}=1\mid Y_n=1)=0.82.
\]

But after observing two ones, the true next-one probability is

\[
\frac{0.9^3+0.1^3}{0.9^2+0.1^2}
=0.8902439\ldots.
\]

The last-one model predicts `0.82` again, despite being exactly correct for the specified one-step conditional. The discrepancy is physical memory, not generative integration error. Larger finite p reduces but does not generally eliminate it. Video occlusion and unresolved/coarsened fluid variables make this a material concern.

**Correction:** State explicitly that autoregression defines a finite-history surrogate process; correctness of its full joint trajectory distribution additionally requires sufficient history and suitable stationarity assumptions. Long-horizon boundedness, approximate spectra, or nonzero spread do not prove that requirement. Independent fresh random numbers are appropriate for sampling a conditional kernel; the issue is omitted information in the kernel, not the use of fresh noise.

## 3. Finite-feature PSD has no exact-semigroup identification guarantee here

**Location:** Section 3.2, Eqs. (9)–(13), `main.tex:110–137`; compression discussion, `main.tex:161–186`; implementation appendix interval distribution. Code: `map_trainer.py:254–271,303–322`, `brownian_path.py:161–186`.

This is the strongest intrinsic limitation, but the manuscript already states it honestly. The correct interpretation is a finite-feature regression heuristic anchored to a teacher, not an application of an exact population uniqueness theorem to the implemented network.

The source Itô-map identification argument requires a continuous full-path map, exact diagonal anchoring, and composition on the relevant complete domain; its learning corollary adds full support. The actual compressed features, EMA target, and restricted interval distribution do not meet that ideal statement. [Pan et al., Theorem 2 and Corollary 1](https://arxiv.org/html/2606.11156v1#S6.SS1).

**An explicit extension of the manuscript's OU example, derived independently.** For `dX=−X dt+dW`, suppose each interval retains only its total increment. The optimal same-path endpoint regression over an interval of length `ℓ` is

\[
F_\ell(x,z)=e^{-\ell}x+a(\ell)z,
\qquad a(\ell)=\frac{1-e^{-\ell}}{\ell}.
\]

With two intervals of length `h` and increments `z₁,z₂`, the direct and composed predictors are

\[
F_{2h}(x,z_1+z_2)=e^{-2h}x+a(2h)(z_1+z_2),
\]

\[
F_h(F_h(x,z_1),z_2)
=e^{-2h}x+e^{-h}a(h)z_1+a(h)z_2.
\]

The increment coefficients differ for every `h>0`. Thus even exact conditional-mean teacher predictors fail pathwise composition. Correct Brownian coupling does not repair this representation loss. A PSD fixed point need not equal the teacher conditional-mean predictor either, since the target changes with the model.

There is a second, separate support limitation: after curriculum completion, direct intervals exclude gaps below `0.01`, and child intervals have a minimum possible gap of `0.0025`. The diagonal is trained at gap zero. Hence there is a positive gap between supervised/composed off-diagonal evaluations and the diagonal; there is no refinement-to-zero sequence enforced by the final population objective. Early curriculum updates and neural smoothness are useful inductive biases, not that missing theorem.

**Correction:** Keep the current compression caveat. Add one clear sentence near Eq. (13), rather than relying only on the later limitations: “For the finite representation and interval sampling used here, PSD is an approximate self-distillation objective; we do not assert identifiability or zero population residual.” Do not describe the issue merely as an optimizer failing to find the correct solution: even an ideal optimizer can face incompatible finite-feature targets. The current later discussion recognizes this stronger point; make the earlier discussion consistent with it.

A matched teacher-rollout diagnostic can test practical fidelity. It cannot establish an exact semigroup or correct true conditional distribution. No urgent redesign or rerun is demanded by this review, since the paper does not currently claim such a theorem.

## 4. Define the numerical teacher used in refinement explicitly

**Location:** `main.tex:135` and two-stage training paragraph; `compression_diagnostic.tex` already discloses the distinction. Code: `map_trainer.py:324–337`, `sde_solvers.py:euler_maruyama_step`.

The frozen drift network is shared, but the numerical transition used for supervision is not the same as the existing SI sampler at the same step count. The trainer uses

\[
X_{j+1}=X_j+\delta b_{\rm T}(X_j,t_j;h)
+[M_{t_{j+1}}-M_{t_j}],
\]

where the last term integrates the diffusion exactly. The baseline uses

\[
X_{j+1}=X_j+\delta b_{\rm T}(X_j,t_j;h)
+\gamma(t_j)\sqrt\delta\,\xi_j.
\]

This is a legitimate, generally preferable choice for additive deterministic diffusion. It must be visible where the distilled reference is defined. “32-step teacher trajectory” otherwise sounds like a call to the baseline's 32-step sampler.

**Exact sanity calculation, derived independently.** For zero drift and `γ(t)=ε(1−t)`, exact endpoint noise variance is `ε²/3`. Uniform n-step left-point Euler–Maruyama instead gives

\[
\varepsilon^2\left(\frac13+\frac{1}{2n}+\frac{1}{6n^2}\right).
\]

Its relative excess is `3/(2n)+1/(2n²)`, approximately 9.57%, 4.74%, 2.36%, and 1.17% for n=16,32,64,128. These are **zero-drift analytic values, not predictions for the fluid model's endpoint variance**. They demonstrate that noise quadrature alone can affect comparisons and apparent spread retention.

**Correction:** Put the exact teacher update above in the method or implementation appendix, identify the Euler drift / exact diffusion combination, and cross-reference it from model selection. Continue treating 64/128-step baseline comparisons as numerical sensitivity checks. A future exact-diffusion solver control would separate benefits of map learning from a cheap numerical improvement, but it is not necessary to invalidate or discard current results. Do not retrospectively change frozen benchmarks silently.

## 5. The endpoint proposition does not connect the optimized loss to forecast accuracy

**Location:** Section 3.4, Proposition 1, `main.tex:148–159`; contribution wording, `main.tex:37,193`.

The coupling bound is correct and useful as a diagnostic, but holds for any jointly defined endpoints, including unrelated algorithms. Shared Brownian noise provides a meaningful coupling; it is not needed for the abstract inequality. The proposition does not bound endpoint error by PSD loss, diagonal loss, or the combined teacher-validation score. Proposition 2 is likewise the standard squared-error projection identity. Neither establishes a new map-learning convergence theorem.

The deterministic flow-map source contains specific loss-to-distribution analyses under its own assumptions; these cannot be inherited by citing its two-time map formulation while optimizing a different stochastic compressed-feature objective. [Boffi et al., Sections 3.3–3.7](https://arxiv.org/html/2406.07507v2#S3).

**Correction:** Keep “diagnostic” language, consider renaming the subsection to “Approximation errors and diagnostic bounds,” and say explicitly that no bound from the implemented PSD training loss to forecast-law error is claimed. Describe conditionalization by fixing h as a straightforward extension of existing identities. The paper's potential contribution is the forecasting adaptation, exact finite-query coupling, and empirical/computational assessment, rather than the novelty of these two elementary propositions.

**Optional accessible improvement:** For a fixed h and full endpoint, let `Y_m` be the numerical teacher endpoint, `Z` the student's features, `μ(Z)=E[Y_m|Z]`, and `fθ(Z)` the student endpoint. Define

\[
A=E\|f_\theta(Z)-\mu(Z)\|^2,\quad
C=E\|Y_m-\mu(Z)\|^2.
\]

Writing `P_*` for the true forecast law and `P_b` for the continuous learned-teacher SDE endpoint law, the existing projection identity, coupling bound, and triangle inequality give

\[
W_2(\mathcal L(f_\theta(Z)),P_*)
\le \sqrt{A+C}
+W_2(\mathcal L(Y_m),P_b)+W_2(P_b,P_*).
\]

This directly separates map approximation, compression, numerical teacher error, and teacher/data error without pretending any is controlled by PSD. All laws and expectations here are conditional on h, and finite second moments are required. This is optional explanatory algebra, not a new theorem worthy of a stronger novelty claim.

## 6. Explain the weighting implicit in average-drift regression

**Location:** Eq. (13), `main.tex:129–135`; teacher target in `map_trainer.py:337–340`.

After cancellation of the shared martingale, the endpoint composition residual is `(t−s)` times the average-drift residual. Hence

\[
\|\Phi_\theta-\Phi_{\rm composed}\|^2
=(t-s)^2\|G_\theta-\widetilde G\|^2.
\]

The implemented unweighted average-drift MSE therefore corresponds to endpoint squared error weighted by `1/(t−s)²`. This also applies to teacher-rollout regression. It is not equivalent to an unweighted endpoint MSE across a variable interval distribution, although exact zero residuals coincide in an ideal representable setting. That distinction is particularly relevant when zero residual is unattainable under compression.

**Correction:** Add one sentence naming this weighting and why average-drift units are used to align diagonal and finite-time supervision. Report the total loss as `λdiag Ldiag + λcomp Lcomp`, with actual weights, EMA rule, curriculum, and endpoint atom in one place. Do not claim equal numerical objectives simply from martingale cancellation. The current manuscript does not explicitly make that false equality, so this is clarification rather than a detected algebra error.

## 7. Small formal and implementation clarifications

1. **Independent Brownian process.** Eq. (1) currently specifies only the marginal `W_t ~ N(0,tI)`. State that `W` is a standard Brownian motion independent of `(h,y)`. Marginal Gaussianity alone does not specify a Brownian process. The implementation does sample independent regression noise.
2. **Meaning of future independence.** At `main.tex:106`, say “increments after s are independent of the state at s.” Absolute future Brownian values need not be independent of an adapted state. The implementation feeds local increments and is correct. Conditioning additionally on an already revealed future bridge endpoint also changes conditional independence, without changing the unconditional Brownian law.
3. **Diagonal as a limit.** Eq. (6) is undefined at `s=t`. Say `G_tt=b_t` is the continuous extension when it exists, and distinguish that exact-map property from a learned diagonal trained only approximately. Added network branches vanishing exactly on the diagonal does not make the backbone equal to the population drift after training.
4. **Moments.** Add finite second moments to Proposition 1; in Proposition 2 require square-integrable predictors, or explicitly use extended nonnegative expectations. This avoids treating an infinite bound as a meaningful guarantee.
5. **Warmup description.** `main.tex:135` says interval growth occurs after diagonal warmup, but both archived NS stages use `diagonal_warmup_steps: 0`. Describe warmup as optional and state that the teacher-initialized study uses no separate diagonal-only warmup. The consistency ramp and interval curriculum are distinct.
6. **Fixed-h covariance statement.** The total-covariance identity is correct for the fixed `(s,t,x,h)` setting of Proposition 2. Keep that conditioning explicit; it cannot automatically diagnose multi-step spread after generated histories have diverged.
7. **Physical stochastic forcing.** In the NS equation at `main.tex:199–201`, identify `B_τ` as a spatially correlated or discretized forcing process, with covariance unverified if necessary. A cylindrical space-white forcing in a continuum two-dimensional vorticity equation need not have finite enstrophy: even the linear heat example has a logarithmically divergent sum of `1/|k|²` variances. The finite grid is well defined; the displayed continuum notation should not imply an unstated covariance model.

## What is sound and should be retained

- The forecasting diagonal target is correct: differentiating the schedule multiplying Brownian motion gives `γ̇_t W_t`; the SDE diffusion accounts for the separate Brownian increment. Adding a derivative of `sqrt(t)` would be wrong here.
- The conditional extension fixes h throughout generative time, including both child maps. Physical history changes only after the full forecast transition.
- The variance clock, Brownian bridge mean/variance, local increment concatenation, and additive noise cancellation agree between equations and code. Future path sampling is independent of the interpolant draw used to select the training start state.
- The exact map's diagonal is noise-independent. The code zeros the interval and Brownian branches on that diagonal and uses the frozen teacher drift when configured.
- The deterministic comparator retains randomness through its Gaussian initial state. It is not a deterministic point-mass-to-nondegenerate-law transport. Its distinction from direct FMM training is stated.
- The OU conditional-variance formula and covariance decomposition are correct. The manuscript correctly avoids interpreting a same-path regression lower bound as a lower bound on CRPS, Wasserstein distance, or the expressive power of arbitrary stochastic generators.
- Same-path teacher validation measures teacher fidelity, not truth accuracy. The paper says so. It also separates simulation-free diagonal training from simulation-based refinement.
- The Föllmer terminology qualification is accurate and important. Changing diffusion while retaining the uncorrected drift would define a different marginal process.
- The calibration appendix distinguishes empirical quantile coverage, finite-ensemble corrections, pointwise ranks, and spatial dependence. Nothing in the method guarantees calibration, and the manuscript appropriately avoids such a theorem.

## Minimum revision before a new results review

1. Add strong-solution/regularity assumptions and qualify endpoint idealizations.
2. State the sufficient-history assumption and the finite-history surrogate interpretation of long rollouts.
3. Put the numerical teacher update and the finite-feature PSD caveat next to the actual training objective.
4. Explain interval weighting; add missing independence and moment qualifiers.
5. Retain the modest adaptation claim and diagnostic interpretation of the propositions.

These revisions do not require restarting current training or altering frozen results. A later empirical review should ask whether the measured quality–cost gains remain meaningful against an exact-diffusion solver control and whether rollout behavior reflects the intended physical distribution. This review deliberately does not prejudge those answers.

## Suggested concise replacement wording

**Before defining the exact Itô map:**

> For the ideal map identities, fix h and assume that the reference SDE has a pathwise unique strong solution with finite second moments. We use a regular version of its drift for which the drift integral has the continuous diagonal extension stated below; any extension to the generative endpoints is an additional assumption. A nonsingular regression target alone does not guarantee these properties. In experiments, the reference is the learned neural drift with the recorded numerical integration rule, rather than an exactly known population drift.

If assumptions are intended only on the interior, replace the endpoint sentence with an explicit restriction to compact subintervals of `(0,1)` and separately assume existence of the endpoint transition used for forecasting. Do not assert a continuous off-support endpoint diagonal for quantized data without justification.

**After introducing autoregressive physical forecasts:**

> Iteration defines a probabilistic process with the chosen finite history. It reproduces the true joint forecast law only if this history is sufficient for the next state, the conditional kernel is learned correctly on the histories encountered, and any relevant time dependence is included. Otherwise, even exact one-step conditional sampling yields a finite-history surrogate. Long-horizon diagnostics therefore assess this surrogate, not a guarantee implied by the generative-time map identity.

**Next to the teacher-refinement target:**

> Teacher trajectories use Euler updates for the frozen drift and exact increments of the integrated diffusion M. They are distinct from the repository's Euler–Maruyama sampler, which evaluates the diffusion at each left endpoint. The finite-feature, restricted-interval PSD loss is an approximate self-distillation objective; we claim neither exact semigroup identification nor a forecast-error bound from this loss.

**Effect on existing computations:** None of the findings above invalidates the completed optimization, generated forecasts, or metric arithmetic on the evidence reviewed. Items 1–3 narrow theoretical interpretation; item 4 clarifies which numerical process is distilled; items 5–7 constrain attribution and notation. I have not audited every result file or claimed those computations are independently certified by this review. No finding justifies silently replacing checkpoints, rerunning a frozen test with revised settings, or discarding an unfavorable result.
