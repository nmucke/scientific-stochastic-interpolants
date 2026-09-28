# Stage 2 — Independent verification of the unified-posterior proof

This audit was derived directly from the active TeX for Lemma A.1, Proposition A.3, Proposition B.1, and Theorem 4.1. I did not consult `review/agents/B-math.md` or rely on another reviewer's conclusions.

## Independent reconstruction

### 1. Conditioning and preservation of the auxiliary source

Condition on the fixed anchor $x_0$. Under the independence used in the proof,
\[
p(\varepsilon,x_1,y\mid x_0)
=p(\varepsilon)\,p(x_1\mid x_0)\,p(y\mid x_1,x_0).
\]
Therefore
\[
p(\varepsilon,x_1\mid y,x_0)
=p(\varepsilon)\,p(x_1\mid y,x_0),
\]
so $\varepsilon\mid(y,x_0)$ remains standard Gaussian and is independent of $x_1\mid(y,x_0)$. This proves the first part of Proposition B.1, but only after assuming $\varepsilon\perp y\mid(x_1,x_0)$ (equivalently here, observation noise independent of the auxiliary source). The proposition statement and the basic observation model do not explicitly state that assumption; it first appears inside the proof.

For $0<\tau<1$, Bayes' rule gives
\[
p_\tau^y(x\mid x_0)
=\frac{p(y\mid X_\tau=x,x_0)p_\tau(x\mid x_0)}{p(y\mid x_0)}
=\frac{\Phi_\tau^y(x,x_0)p_\tau(x\mid x_0)}{p(y\mid x_0)}.
\]
Taking a spatial log-gradient gives $s_\tau^y=s_\tau+\nabla\log\Phi_\tau^y$. Because the conditioned path has the same schedules and anchor, applying the velocity--score identity to prior and conditioned paths and subtracting gives
\[
v_\tau^y-v_\tau=\kappa_\tau(s_\tau^y-s_\tau)
=\kappa_\tau\nabla\log\Phi_\tau^y.
\]
The algebra is correct once the conditioned path is known to satisfy the hypotheses under which Proposition A.2 was proved.

### 2. Tweedie identities and the required moments

For $X=m+\sigma\varepsilon$,
\[
\nabla p_X(x)
=\int -\frac{x-m}{\sigma^2}\varphi_\sigma(x-m)\,P_m(\mathrm dm),
\]
hence $\mathbb E[m\mid X=x]=x+\sigma^2\nabla\log p_X(x)$. The identities for $\mathbb E[\varepsilon\mid X]$ and $\mathbb E[x_1\mid X]$ follow by affine substitution. Differentiating the conditional mean gives
\[
\nabla_x\mathbb E[m\mid X=x]=\sigma^{-2}\operatorname{Cov}(m\mid X=x),
\]
and therefore $\operatorname{Cov}(\varepsilon\mid X=x)=I+\sigma^2\nabla s(x)$. All signs, factors, and matrix orientations in Lemma A.1 are correct.

The appendix explicitly assumes $\mathbb E\|x_1\|^2<\infty$ before Lemma A.1 and Proposition A.2. Proposition B.1 then applies Proposition A.2 to the *conditioned* target $p(x_1\mid y,x_0)$, but neither Proposition B.1 nor Theorem 4.1 assumes the corresponding posterior second moment or otherwise imports the Appendix-A hypotheses for the conditioned path. A bounded likelihood (including the Gaussian likelihood used later) preserves the prior second moment, but Theorem 4.1 is stated for a general likelihood tilt.

### 3. Marginal-preserving dynamics

For a smooth compactly supported test function $f$, differentiating the explicit path under the expectation gives
\[
\frac{\mathrm d}{\mathrm d\tau}\mathbb E[f(X_\tau)]
=\mathbb E[\nabla f(X_\tau)\cdot\dot X_\tau]
=\int \nabla f(x)\cdot v_\tau(x)p_\tau(x)\,\mathrm dx.
\]
Integration by parts yields the continuity equation $\partial_\tau p_\tau=-\nabla\cdot(v_\tau p_\tau)$. The Fokker--Planck operator of
\[
\mathrm dX_\tau=[v_\tau+\tfrac12g_\tau^2s_\tau]\,\mathrm d\tau+g_\tau\,\mathrm dW_\tau
\]
evaluated at $p_\tau$ is
\[
-\nabla\cdot(v_\tau p_\tau)
-\tfrac12g_\tau^2\nabla\cdot(s_\tau p_\tau)
+\tfrac12g_\tau^2\Delta p_\tau
=-\nabla\cdot(v_\tau p_\tau),
\]
because $s_\tau p_\tau=\nabla p_\tau$. Thus Proposition A.3's cancellation is exact. Uniqueness of the Fokker--Planck solution then identifies the interior-time laws.

There is a formal regularity mismatch: Proposition A.3 asks for joint $C^2((0,1)\times\mathbb R^{N_u})$, whereas Theorem 4.1 assumes only parabolic $C^{1,2}$. The calculation needs one time and two spatial derivatives, so $C^{1,2}$ is the appropriate condition; changing Proposition A.3 fixes the mismatch without changing the argument.

### 4. Source laws and endpoints

With the active schedules:

- FM has $\alpha_\tau=0$, $\beta_\tau=\tau$, $\sigma_\tau=1-\tau$, hence $X_0=\varepsilon\sim\mathcal N(0,I)$.
- SI has $\alpha_0=1$, $\beta_0=0$, and $\sigma_0=\gamma_0\sqrt0=0$, hence $X_0=x_0$ and $p_0=\delta_{x_0}$.

Since $\beta_0=0$, $X_0$ is independent of $x_1$ and therefore of $y$ under the factorization in part 1. Thus $\Phi_0^y(x,x_0)=p(y\mid x_0)$ is constant on the support of $p_0$, and $p_0^y=p_0$ for both FM and SI. This part is correct once source/observation independence is explicit.

At $\tau=1$, $\alpha_1=\sigma_1=0$ and $\beta_1=1$, so the constructed conditioned path satisfies $X_1=x_1\sim p(x_1\mid y,x_0)$. The PDE calculation is made only on $(0,1)$. To transfer equality of laws to $\tau=1$, take $\tau_n\uparrow1$: schedule continuity gives the explicit path $X_{\tau_n}\to x_1$ almost surely, while a well-posed Itô SDE/ODE on $[0,1]$ has continuous sample paths, so $X^*_{\tau_n}\to X^*_1$ almost surely. Equality of the interior laws and bounded convergence then give $\mathcal L(X^*_1)=p(x_1\mid y,x_0)$. The same limiting argument covers $\tau=0$. This endpoint step is valid but omitted from Proposition A.3's proof.

The recovered score and $\kappa_\tau$ need only be defined almost everywhere on $(0,1)$; neither is generally defined at the endpoints. The SDE should therefore be read with interior-time coefficients and separately specified endpoint laws.

### 5. The $g=0$ case with an SI point source

If $g=0$ and the ODE is well posed and unique from $X_0=x_0$, its law at every later time is a point mass at the unique deterministic trajectory. The SI interpolant has $\sigma_\tau>0$ for $0<\tau<1$, so its marginal is non-degenerate. Consequently no well-posed unique deterministic ODE from $\delta_{x_0}$ can realize those SI marginals. The “whenever that ODE is well posed” clause in Proposition A.3 therefore excludes $g=0$ for the SI point source; it is not an SI posterior sampler covered by the theorem. The explicit sampler list is consistent—it assigns $g=0$ only to FM—but the exclusion is currently present only in a commented-out remark.

### 6. Exact-posterior conclusion

For the conditioned path,
\[
v_\tau^y+\tfrac12g_\tau^2s_\tau^y
=v_\tau+\tfrac12g_\tau^2s_\tau
+(\kappa_\tau+\tfrac12g_\tau^2)\nabla\log\Phi_\tau^y,
\]
which is exactly Eq. (9). Proposition A.3 then supplies the conditioned marginals, and the endpoint limit supplies the posterior terminal law. Thus the advertised conclusion follows after explicitly adding the source/observation independence and conditioned-path integrability hypotheses, aligning the regularity notation, and reading well-posedness as holding through the endpoints. Without the first two imported hypotheses, the theorem is not proved as stated.

## Pass/fail table

| Check | Result | Assessment |
|---|---|---|
| (i) Conditioning/independence | **Fail as stated; derivation passes with one added assumption** | The factorization is correct, but observation/source independence appears only inside the proof. |
| (ii) Moment/regularity assumptions | **Fail as stated** | The conditioned path is not assumed to satisfy Appendix A's moment hypotheses, and Proposition A.3's $C^2$ condition does not match Theorem 4.1's $C^{1,2}$. |
| (iii) Source law at $\tau=0$ | **Pass, conditional on (i)** | Active FM schedules give $\mathcal N(0,I)$; active SI schedules give $\delta_{x_0}$; the tilt is constant at the source. |
| (iv) Endpoint claims | **Pass with an omitted limiting step** | Interior Fokker--Planck equality extends to 0 and 1 by weak continuity of both paths under the stated well-posedness. |
| (v) $g=0$ with SI point source | **Not covered** | Well-posed uniqueness excludes a deterministic flow that splits $\delta_{x_0}$ into non-degenerate SI marginals. |
| (vi) Exact-posterior conclusion | **Fail literally as stated; pass after the assumption fixes** | The drift algebra and Fokker--Planck cancellation are correct; the proof's imported hypotheses are not all in the theorem/proposition statements. |

## Findings

### S2-1 — Observation/source independence is used but not assumed in Proposition B.1

- **Location** — Proposition B.1 and its proof, Eqs. (33)--(36), p. 15. Quote: “with $\eta$ independent of $(x_1,\varepsilon)$.”
- **Severity** — major.
- **Problem** — This independence is introduced only in the proof. Without $\varepsilon\perp y\mid(x_1,x_0)$, conditioning can change the Gaussian source and its independence from the posterior target, so Eqs. (33) and (35), and hence Theorem 4.1, need not hold.
- **Confidence** — high; direct factorization verifies necessity. A different joint-coupling definition stated before Proposition B.1 could also supply the assumption.
- **Fix** — Add to Proposition B.1: “Assume the auxiliary source $\varepsilon$ is independent of $(x_1,y)$ conditional on $x_0$ (equivalently, the observation noise is independent of $(x_1,\varepsilon)$ given $x_0$).” Also state this independence with Eq. (1).

### S2-2 — The conditioned path does not inherit the moment hypothesis automatically

- **Location** — Appendix A preamble and Proposition B.1, pp. 13 and 15. Quotes: “$\mathbb E\|x_1\|^2<\infty$”; “apply the duality (24) to both paths.”
- **Severity** — major.
- **Problem** — Proposition A.2 is proved under a finite second moment, but Proposition B.1 applies it to $p(x_1\mid y,x_0)$ without assuming that posterior moment is finite. A general continuous likelihood can reweight a finite-moment prior into a posterior outside that hypothesis; the later bounded Gaussian likelihood is safe, but Theorem 4.1 is general.
- **Confidence** — high as a mismatch between invoked and stated hypotheses. A new derivation of Eq. (35) under weaker local-integrability conditions could replace, rather than add, the moment condition.
- **Fix** — Add to Proposition B.1/Theorem 4.1: “Assume the prior and conditioned interpolant paths satisfy the hypotheses of Appendix A; in particular $\mathbb E[\|x_1\|^2\mid y,x_0]<\infty$.” A bounded-likelihood assumption is an alternative sufficient condition when the prior has a finite second moment.

### S2-3 — The regularity hypothesis used by Proposition A.3 does not match Theorem 4.1

- **Location** — Proposition A.3, p. 14, and Theorem 4.1, p. 4. Quotes: “suppose $p\in C^2$”; “$p_\tau^y$ is positive and $C^{1,2}$.”
- **Severity** — minor.
- **Problem** — The theorem invokes Proposition A.3 without satisfying its written joint-$C^2$ condition. The proof itself needs only one time derivative and two spatial derivatives, so this is a formal statement mismatch rather than a failed cancellation.
- **Confidence** — high; the displayed Fokker--Planck calculation shows exactly which derivatives are used. No external result is needed.
- **Fix** — Change Proposition A.3's condition to “$p\in C^{1,2}((0,1)\times\mathbb R^{N_u})$,” matching Theorem 4.1.

### S2-4 — The endpoint conclusion requires a limiting argument absent from the proof

- **Location** — Proposition A.3 proof, p. 14. Quote: “the law of $X_\tau$ equals $p_\tau$ for all $\tau$.”
- **Severity** — minor.
- **Problem** — The density and Fokker--Planck calculation is only on $(0,1)$, while the proposition and Theorem 4.1 claim equality on $[0,1]$, including the exact posterior at $\tau=1$. The conclusion is recoverable by weak continuity, but that step is not written.
- **Confidence** — high under the theorem's intended well-posed continuous SDE/ODE solution on $[0,1]$. If “well posed” were intended only on the open interval, endpoint existence would need an additional assumption.
- **Fix** — Add after the uniqueness step: “Equality at $\tau=0,1$ follows by weak continuity: the explicit interpolant converges almost surely to its endpoint variables, and the well-posed SDE/ODE has continuous paths on $[0,1]$.”

### S2-5 — A deterministic sampler is impossible for the SI point-mass source

- **Location** — Proposition A.3, p. 14, and sampler specializations below Theorem 4.1, p. 4. Quote: “Setting $g_\tau=0$ gives the probability-flow ODE whenever that ODE is well posed.”
- **Severity** — minor.
- **Problem** — The well-posedness caveat necessarily excludes $g=0$ with $p_0=\delta_{x_0}$ when SI marginals are non-degenerate. This is mathematically consistent with the FM-only ODE specialization, but leaving the exclusion implicit makes “every admissible diffusion schedule” easy to misread as including an SI-ODE.
- **Confidence** — high; a unique deterministic flow maps a point mass to a point mass. Only a non-unique/singular ODE or added initial randomness could evade the argument, both outside the theorem's assumptions.
- **Fix** — Add after the three sampler specializations: “For the SI point-mass source, $g_\tau=0$ is not admissible under well-posed uniqueness because a deterministic flow cannot generate the non-degenerate intermediate marginals; the ODE member here uses the non-degenerate FM source.”

No algebraic error was found in Lemma A.1, the velocity/score shift, the guidance weight $\kappa_\tau+\tfrac12g_\tau^2$, or the Fokker--Planck cancellation.

## Comparison with Stage 1 after independent derivation

Stage 1 treated Theorem 4.1 as conditionally supporting the ideal exactness claim while criticizing the abstract for omitting the theorem's conditions. This independent derivation qualifies that assessment: the theorem/proposition statements themselves also omit the observation/source-independence and conditioned-path moment hypotheses used by their proofs. I otherwise found no Stage-1 mathematical claim to dispute.
