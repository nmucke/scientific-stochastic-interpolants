# Why `InterpolantLikelihood` is biased even in the linear-Gaussian setting

**TL;DR.** The method is an exact computation — but of the **wrong
conditional**. It computes the posterior score under the
*interpolation joint* $(x_\tau, x_1)$ defined by Lemma 4.2/4.3, and
then uses that score to guide the reverse **SDE** sampler. In the
linear-Gaussian test case both objects are Gaussian and closed-form,
so nothing is numerically approximated — yet they disagree, and the
disagreement is exactly the $\sim 18\%$ mean slope error that shows
up at $\tau=0.5$ in the numerical experiments.

---

## 1. Two different conditional distributions of $x_1$ given $x_\tau$

There are two distinct joint laws of $(x_\tau, x_1)$ in the paper's
framework, and they have different conditionals.

### 1a. Interpolation joint (Lemma 4.2 / 4.3)

The forward process is defined algebraically:

$$
x_\tau \;=\; \alpha_\tau\,x_0 \;+\; \beta_\tau\,x_1 \;+\; \gamma_\tau\,W_\tau,
\qquad x_1\sim p_{\text{target}},\; W_\tau\sim\mathcal{N}(0,\tau I).
$$

Here $x_1$ and $W_\tau$ are **independent by construction**. Under
this joint, for isotropic target $p(x_1\mid x_0)=\mathcal N(x_0, cI)$,
the conditional mean is a linear function of $x_\tau$ with slope

$$
s^{\text{interp}}(\tau)
\;=\; \frac{\partial\,\mathbb{E}^{\text{interp}}[x_1\mid x_\tau,x_0]}{\partial x_\tau}
\;=\; \frac{\beta_\tau\,c}{\beta_\tau^{\,2} c + \tau\gamma_\tau^{\,2}}.
$$

For linear interpolation ($\alpha=1-\tau$, $\beta=\tau$, $\gamma=1-\tau$),
$c=1$, $\tau=0.5$: $s^{\text{interp}}(0.5) = 0.5 / 0.375 \approx 1.333$.

This is the slope implicit in `InterpolantLikelihood.mu_fn`: the
correction term $d_\tau = -\gamma_\tau H\,\mathbb E[W_\tau\mid x_\tau,x_0]$
is exactly what converts the raw interpolation inverse
$(x_\tau - \alpha_\tau x_0)/\beta_\tau$ (slope $1/\tau = 2$) into the
interpolation-conditional mean (slope $1.333$).

### 1b. SDE joint (Theorem 4.1 / Doob h-transform)

The interpolation also defines a **prior SDE** — this is the object
whose drift $b_\theta$ the model learns and whose reverse-time
trajectory the sampler integrates:

$$
dX_s \;=\; b_\theta(X_s, x_0, s)\,ds + \gamma_s\,dW_s,
\qquad X_0 = x_0.
$$

In the LG setting this is an affine SDE with slope
$a(s) = [\beta\beta' c + s\gamma\gamma']/[\beta^2 c + s\gamma^2]$,
and the transition from $\tau$ to $1$ has closed form

$$
X_1 \mid X_\tau, X_0 \;\sim\; \mathcal N\!\bigl(x_0 + \Phi_{1,\tau}(X_\tau - x_0),\; V_\tau^{\text{cond}}\,I\bigr),
\qquad \Phi_{1,\tau} = \exp\!\int_\tau^1 a(s)\,ds.
$$

Computing $\Phi_{1,0.5}$ by quadrature (as
`SDEConditionalLikelihood._phi_and_vcond` does) gives
$\Phi_{1,0.5}\approx 1.562$ — i.e. the slope of $x_1$ in $x_\tau$
**under the SDE joint** is $1.562$, not $1.333$.

### 1c. The two conditionals are different Gaussians

Even though the *marginal* distributions of $x_\tau$ and $x_1$ agree
(by construction, the interpolation was designed so that $p_\tau(x)$
matches the marginal of the SDE at time $\tau$), the **joint** laws
of $(x_\tau, x_1)$ do **not** agree. The reason:

* Under 1a, $x_1$ is drawn once at the beginning and $W_\tau$ is an
  independent Brownian bridge — the algebraic coupling says
  "$x_1$ and $W_\tau$ are independent, $x_\tau$ is built from both".
* Under 1b, $x_1$ is produced by *integrating* the SDE across
  $[\tau,1]$ starting from $x_\tau$ — the coupling is causal, and
  $x_1$ is a function of the future noise increments $dW_s$ for
  $s>\tau$, which are independent of the history that produced
  $x_\tau$.

These give different covariances between $x_\tau$ and $x_1$, hence
different regression slopes, hence different conditional means —
even though both marginals are identical. The mismatch is intrinsic
to the setup, not a numerical artifact.

---

## 2. Which conditional does Doob's h-transform need?

The reverse-time posterior sampler implements

$$
dX_s \;=\; \bigl[b_\theta(X_s,x_0,s) \;+\; \gamma_s^{\,2}\,\nabla_{x}\log h_{\text{SDE}}(X_s,s)\bigr]\,ds \;+\; \gamma_s\,dW_s,
$$

where the correction factor is

$$
h_{\text{SDE}}(x_\tau,\tau) \;=\; \mathbb{E}_{\text{SDE}}\!\bigl[p(y\mid X_1)\,\big|\,X_\tau=x_\tau,\,X_0=x_0\bigr].
$$

The **expectation is under the SDE joint** — not the interpolation
joint — because the object being corrected is an SDE trajectory, not
an interpolation trajectory. The sampler walks along SDE paths, so
the "probability that this path will hit the observation" must be
averaged over the SDE's own future, i.e. $p_{\text{SDE}}(x_1\mid x_\tau,x_0)$.

`InterpolantLikelihood` substitutes $p_{\text{interp}}(x_1\mid x_\tau,x_0)$
into this formula instead. In the LG test, that is a Gaussian with
the wrong mean slope ($1.333$ vs $1.562$) and (slightly) the wrong
variance. The resulting $h$ factor is exact for
$\mathbb E[p(y\mid X_1)\mid X_\tau, X_0]$ under the **interpolation**
joint — and that number is simply not the quantity that belongs in
the Doob h-transform of the SDE sampler.

This is why the error persists in the purely analytical setting:
nothing is being approximated. Both expectations can be computed in
closed form; they just aren't equal.

---

## 3. Why the intuition "but the marginals match, so it should be exact" is wrong

A very natural expectation is: *"the interpolation was chosen so that
the marginal density $p_\tau(x)$ agrees with the SDE marginal at time
$\tau$, so pushing the likelihood through Bayes' rule with either
joint should give the same answer."* This is false, and the reason
is subtle.

Matching marginals $p_\tau(x) = p_{\text{SDE},\tau}(x)$ for every
$\tau$ is a **much weaker** condition than matching joints
$p(x_\tau, x_1) = p_{\text{SDE}}(x_\tau, x_1)$. Two processes can
share every one-time marginal and disagree on every two-time
covariance. That is precisely what happens here:

* Under the interpolation, $\operatorname{Cov}(x_\tau, x_1)
  = \beta_\tau c$ (from $x_\tau = \alpha_\tau x_0 + \beta_\tau x_1 + \gamma_\tau W_\tau$
  with $x_1\perp W_\tau$).
* Under the SDE, $\operatorname{Cov}(x_\tau, x_1)
  = \Phi_{1,\tau}\,V_\tau$, with $V_\tau = \beta_\tau^2 c + \tau\gamma_\tau^2$.

At $\tau=0.5, c=1$: the first is $0.5$, the second is
$1.562\cdot 0.375 \approx 0.586$. Different by $\sim 17\%$, which is
exactly the mean slope error we see. The marginals agree — both
copies of $x_\tau$ have variance $0.375$, both copies of $x_1$ have
variance $1$ — but the **temporal coupling** is different.

The interpolant literature uses the matching-marginals property to
justify training the drift $b_\theta$ by matching scores against
$p_\tau$. That is fine for *prior sampling*: as long as the drift
reproduces the marginal at each $\tau$, the SDE generates samples
from the right target. **But posterior conditioning requires the
joint**, and the joint is not pinned down by the marginals.

---

## 4. Where in the code this manifests

### 4a. `InterpolantLikelihood.mu_fn` (likelihood.py:82–100)

```python
i_obs = alpha * H @ x_0 + beta * observations        # y_bar_tau
E_W   = -gamma * t * model_score                     # E[W_tau | x_tau, x_0]
d     = -gamma * H @ E_W                              # d_tau correction
return i_obs - x_obs - d                              # residual
```

The quantity `E_W` is $\mathbb{E}^{\text{interp}}[W_\tau\mid x_\tau, x_0]$,
derived by regressing $W_\tau$ on $x_\tau$ under the **algebraic**
joint. In the LG test this equals $-\gamma_\tau\tau/V_\tau\cdot(x_\tau-x_0)$,
and the residual becomes

$$
r \;=\; \bar y_\tau - H x_\tau - d_\tau
   \;=\; \beta_\tau\bigl(y - H\hat x_1^{\text{interp}}\bigr),\qquad
   \hat x_1^{\text{interp}} = x_0 + \frac{\beta_\tau c}{V_\tau}(x_\tau - x_0).
$$

The predictor $\hat x_1^{\text{interp}}$ has slope $\beta_\tau c/V_\tau
= 1.333$ at $\tau=0.5,c=1$ — the interpolation-joint slope, not the
SDE slope $\Phi_{1,\tau} = 1.562$.

### 4b. `SDEConditionalLikelihood.score` (likelihood.py:236–252)

The corrected class hard-codes the SDE-joint slope:

```python
phi, V_cond = self._phi_and_vcond(tau)        # Phi_{1,tau}, V_cond
x1_mean = x0 + phi * (x - x0)                  # SDE slope
Sigma   = V_cond * HHt + original_variance * I_y
```

This is identical in structure to 4a but with the **correct** slope.
In the numerical table this reduces the Wasserstein distance from
$\sim 0.70$ to $\sim 0.02$ at $\sigma^2=0.5$ — a factor of 35
improvement coming purely from replacing $1.333$ with $1.562$.

### 4c. Numerical evidence

From `main_analytical.py` with isotropic $c=1$, $x_0=(5,5)$,
$y=(1,1)$, linear interpolation:

```
                            sigma^2 = 0.5              sigma^2 = 1               sigma^2 = 2
                        Wasserstein      KL-div   Wasserstein      KL-div   Wasserstein      KL-div
---------------------------------------------------------------------------------------------------
Interpolant                +0.7035     +1.2038       +0.5328     +0.5168       +0.3582     +0.1694
SDE-conditional            +0.0198     +0.0846       +0.0180     +0.0509       +0.0292     +0.0151
Multi-step lin. drift      +0.0171     +0.0561       +0.0364     -0.0078       +0.0217     -0.0528
```

`SDE-conditional` and `Multi-step lin. drift` — both of which use
the SDE joint — agree to plotting accuracy. `Interpolant`, which
uses the interpolation joint with the same analytical drift and the
same $H$, $\sigma^2$, and reverse sampler, is systematically off.
Nothing in those rows is numerical noise: the difference is the
choice of joint distribution.

---

## 5. Sanity check: when does `InterpolantLikelihood` become exact?

Setting the two slopes equal gives
$\beta_\tau c/V_\tau = \Phi_{1,\tau}$, which rearranges to

$$
\Phi_{1,\tau}\,(\beta_\tau^{\,2} c + \tau\gamma_\tau^{\,2}) \;=\; \beta_\tau c.
$$

* **$\gamma_\tau \equiv 0$** (deterministic interpolant, no Brownian
  bridge). Then $V_\tau = \beta_\tau^{\,2}c$ and the slope collapses
  to $1/\beta_\tau$; the SDE reduces to an ODE whose transition
  factor is also $1/\beta_\tau$. Both joints coincide.
* **$\tau\to 1$.** $V_\tau\to c$, slope $\to \beta_\tau$, and
  $\Phi_{1,\tau}\to 1$. Both sides $\to 1$; the bias vanishes.
* **$\tau\to 0^+$.** Both slopes blow up but at different rates;
  the ratio stays finite. Empirically this is where the bias is
  largest.

Outside of $\gamma=0$ or endpoints, the two joints genuinely
disagree in the LG test, and `InterpolantLikelihood` is exact for
the wrong quantity.

---

## 6. Summary

| Question | `InterpolantLikelihood` | What the sampler needs |
| --- | --- | --- |
| What joint is the conditional taken under? | Interpolation joint (Lemma 4.2) | SDE joint (Theorem 4.1) |
| Slope of $\mathbb E[x_1\mid x_\tau,x_0]$ in LG at $\tau=0.5,c=1$ | $1.333$ | $1.562$ |
| $\operatorname{Cov}(x_\tau,x_1)$ | $\beta_\tau c$ | $\Phi_{1,\tau} V_\tau$ |
| Is the computation exact? | Yes — for the interpolation joint | Yes — for the SDE joint |
| Does it match the sampler's Doob h-transform? | **No** | Yes |

The method is mathematically well-defined and internally consistent;
it just conditions on the wrong joint. In the LG test that joint
mismatch produces a deterministic, reproducible $\sim 18\%$ slope
error — not a sampling artifact, not a numerical issue, and not
something that shrinks with more SDE steps. `SDEConditionalLikelihood`
fixes it by replacing the interpolation-conditional $\hat x_1$
predictor with the SDE-conditional one, and the bias disappears.

The cheap corrections in `cheap_interpolant_corrections.md` are all
different ways of *approximating* the same fix — replacing the
interpolation-joint predictor with an SDE-joint predictor — without
paying for a full Jacobian of $b_\theta$.

---

## 7. Analytical correction factor in the LG setting

Because both the interpolant likelihood and the SDE likelihood are
Gaussians in closed form in the linear-Gaussian test case, their
ratio is also closed-form. One can therefore derive an exact
correction factor $K(x_\tau,\tau)$ such that

$$
p_S(y\mid x_\tau,x_0) \;=\; K(x_\tau,\tau)\,\cdot\,p_I(\bar y_\tau\mid x_\tau,x_0),
$$

and, equivalently, the "corrected" score is

$$
\nabla_{x_\tau}\log p_S \;=\; \nabla_{x_\tau}\log p_I \;+\; \nabla_{x_\tau}\log K.
$$

Take isotropic target $p(x_1\mid x_0)=\mathcal N(x_0,\,cI)$, linear
interpolation, isotropic observation noise $\sigma^{2}I$, and
$H=I$ for clarity — the vector/matrix form is a routine extension.

### 7.1 Two Gaussians, four scalars

Introduce $\xi := x_\tau - x_0$ and $\eta := y - x_0$. Up to
constants that do not depend on $x_\tau$, both log-likelihoods are
quadratic forms of the same shape:

$$
\log p_\bullet \;=\; -\tfrac{\lambda_\bullet}{2}\,\bigl\|\eta - s_\bullet\,\xi\bigr\|^{2} \;+\; \text{const},\qquad \bullet\in\{I,S\},
$$

with the four scalars

| | slope $s_\bullet$ | precision $\lambda_\bullet$ |
| --- | --- | --- |
| Interpolant ($I$) | $s_I = \dfrac{\beta_\tau\,c}{V_\tau}$ | $\lambda_I = \dfrac{\beta_\tau^{\,2}}{v_I}$ |
| SDE ($S$) | $s_S = \Phi_{1,\tau}$ | $\lambda_S = \dfrac{1}{v_S}$ |

where $V_\tau=\beta_\tau^{\,2}c+\tau\gamma_\tau^{\,2}$,
$v_I=\beta_\tau^{\,2}\sigma^{2}+\gamma_\tau^{\,2}\tau$ (the
`InterpolantLikelihood` scalar variance without the Jacobian
correction; add the $C_\tau$ scalar if you want to include it),
and $v_S=V_\tau^{\text{cond}}+\sigma^{2}$ with
$V_\tau^{\text{cond}}=c-\Phi_{1,\tau}^{\,2}V_\tau$. The extra
factor $\beta_\tau^{\,2}$ in $\lambda_I$ appears because
`InterpolantLikelihood` writes its Gaussian in the *transformed*
observation $\bar y_\tau = \alpha_\tau x_0+\beta_\tau y$, so the
change of variables from $\bar y_\tau$ to $y$ contributes
$\beta_\tau^{\,2}$ to the effective precision in $y$.

The only discrepancy between the two likelihoods is the pair
$(s_I,\lambda_I)$ vs $(s_S,\lambda_S)$. That is the entire bias.

### 7.2 The multiplicative correction is a Gaussian in $x_\tau$

Subtract the two quadratics:

$$
\Delta(x_\tau,\tau) \;:=\; \log p_S - \log p_I
\;=\; \tfrac{1}{2}(\lambda_I - \lambda_S)\,\|\eta\|^{2}
\;+\;(\lambda_S s_S - \lambda_I s_I)\,\eta^{\!\top}\!\xi
\;-\;\tfrac{1}{2}(\lambda_S s_S^{\,2} - \lambda_I s_I^{\,2})\,\|\xi\|^{2}.
$$

Exponentiating gives the closed-form correction factor

$$
\boxed{\;K(x_\tau,\tau) \;=\; \exp\!\Bigl[\tfrac{1}{2}(\lambda_I - \lambda_S)\,\|\eta\|^{2} + (\lambda_S s_S - \lambda_I s_I)\,\eta^{\!\top}\!\xi - \tfrac{1}{2}(\lambda_S s_S^{\,2} - \lambda_I s_I^{\,2})\,\|\xi\|^{2}\Bigr]\;}
$$

This is a Gaussian kernel in $x_\tau$. All three coefficients are
available analytically from $\beta_\tau$, $\gamma_\tau$, $\tau$,
$c$, and $\sigma^{2}$ (plus the offline quadrature for
$\Phi_{1,\tau}$). Multiplying the interpolant Gaussian by $K$
reproduces the SDE Gaussian **exactly** — no remaining bias in
either mean or variance.

### 7.3 The additive score correction has two coefficients

Differentiating $\Delta$ with respect to $x_\tau$:

$$
\nabla_{x_\tau}\log p_S \;=\; \nabla_{x_\tau}\log p_I
\;+\; A_\tau\,(y-x_0)
\;-\; B_\tau\,(x_\tau - x_0),
$$

with

$$
A_\tau \;=\; \lambda_S s_S - \lambda_I s_I \;=\; \frac{\Phi_{1,\tau}}{v_S} \;-\; \frac{\beta_\tau^{\,3}\,c}{v_I\,V_\tau},
$$

$$
B_\tau \;=\; \lambda_S s_S^{\,2} - \lambda_I s_I^{\,2} \;=\; \frac{\Phi_{1,\tau}^{\,2}}{v_S} \;-\; \frac{\beta_\tau^{\,4}\,c^{\,2}}{v_I\,V_\tau^{\,2}}.
$$

So the correction is **one linear term in $y$ and one linear term
in $x_\tau$**, with two $\tau$-dependent scalar coefficients.
Both are precomputable on a $\tau$-grid from the same quadrature
that `SDEConditionalLikelihood._phi_and_vcond` already runs. At
inference, the correction is arithmetic: two scalar-times-vector
operations per reverse step, **no Jacobians, no extra drift
calls**.

**Worked example ($\tau=0.5,\ c=1,\ \sigma^{2}=1$, linear
interpolation).**
$\beta_\tau=\gamma_\tau=0.5$, $V_\tau=0.375$,
$\Phi_{1,\tau}\approx 1.5622$, $V_\tau^{\text{cond}}\approx 0.0850$,
$v_I=0.25\cdot 1 + 0.25\cdot 0.5 = 0.375$,
$v_S \approx 1.0850$. Plugging in:

$$
\lambda_I s_I = \tfrac{0.25}{0.375}\cdot\tfrac{0.5}{0.375} \approx 0.889,
\qquad
\lambda_S s_S = \tfrac{1.5622}{1.0850} \approx 1.440,
$$

$$
A_{0.5} \;\approx\; 1.440 - 0.889 \;\approx\; 0.551,
$$

$$
\lambda_I s_I^{\,2} \approx 0.889\cdot 1.333 \approx 1.185,
\quad
\lambda_S s_S^{\,2} \approx 1.440\cdot 1.5622 \approx 2.249,
\quad
B_{0.5} \;\approx\; 1.064.
$$

At $\tau=0.5$ the corrected score is the interpolant score **plus**
$0.551(y-x_0)$ **minus** $1.064(x_\tau-x_0)$. Precomputing
$(A_\tau,B_\tau)$ on a $\tau$-grid once and caching them gives an
exact LG correction at essentially zero runtime cost.

### 7.4 Can this be collapsed to a single scalar multiplier?

**No — not in general.** A single scalar $\kappa(\tau)$ acting on
the existing interpolant score would rescale the two components
$\{(y-x_0)\text{-part},\ (x_\tau-x_0)\text{-part}\}$ by the same
factor, but $A_\tau$ and $B_\tau$ are **not proportional**:

$$
\frac{A_\tau + \lambda_I s_I}{\lambda_I s_I} \;\ne\; \frac{B_\tau + \lambda_I s_I^{\,2}}{\lambda_I s_I^{\,2}}
\quad\Longleftrightarrow\quad s_S \ne s_I,
$$

which is exactly the mean-slope discrepancy that caused the bias in
the first place. The interpolant score mixes the two components in
the wrong ratio, and a scalar multiplier cannot change a ratio.

The best single-scalar approximation is the projection that
matches the $(x_\tau-x_0)$-component only,

$$
\kappa_\tau^{\star} \;=\; \frac{\lambda_S s_S^{\,2}}{\lambda_I s_I^{\,2}}
\;=\; \frac{\Phi_{1,\tau}^{\,2}\,v_I\,V_\tau^{\,2}}{\beta_\tau^{\,4}\,c^{\,2}\,v_S},
$$

which kills the dominant (self-interaction) part of the bias but
leaves a residual in the $y$-dependent term. In the worked example,
$\kappa_{0.5}^{\star}\approx 2.249/1.185\approx 1.90$. Multiplying
the interpolant score by $\sim 1.9$ at $\tau=0.5$ removes most of
the mean bias — which matches the regime where a stray $\times 2$
factor had previously been found empirically helpful. The full fix
needs both $A_\tau$ and $B_\tau$ separately.

### 7.5 Sanity checks

* **$\gamma_\tau\to 0$** (deterministic interpolant). Then
  $V_\tau=\beta_\tau^{\,2}c$ so $s_I=1/\beta_\tau$; the SDE becomes
  an ODE with $\Phi_{1,\tau}=1/\beta_\tau$. Thus $s_I=s_S$, and a
  short calculation gives $\lambda_I=\lambda_S$, so
  $A_\tau=B_\tau=0$. ✓
* **$\tau\to 1$.** $V_\tau\to c$, $s_I\to 1$, $\Phi_{1,\tau}\to 1$,
  $V_\tau^{\text{cond}}\to 0$. The two Gaussians coincide and
  $A_\tau,B_\tau\to 0$. ✓
* **Perfect observations $\sigma^{2}\to 0$.** Both precisions blow
  up; $A_\tau,B_\tau$ scale like $1/\sigma^{2}$. The correction
  grows in the same regime where the interpolant bias matters most,
  consistent with the observed growth of the `Interpolant`
  Wasserstein error as $\sigma^{2}$ decreases.

### 7.6 Practical recipe

1. Offline, on a $\tau$-grid, compute $\Phi_{1,\tau}$ and
   $V_\tau^{\text{cond}}$ using the quadrature in
   `SDEConditionalLikelihood._phi_and_vcond`.
2. Offline, tabulate $A_\tau$ and $B_\tau$ from the formulas above.
3. At inference, inside `PosteriorModel.sample`, after the
   interpolant score is computed, add
   $A_\tau(y-Hx_0) - B_\tau H^{\!\top}H(x_\tau-x_0)$ (restoring $H$
   in the non-$H=I$ case).

Step 3 is a pure linear post-processing of the existing score: no
extra drift calls, no Jacobians. In the LG test it reduces the
`Interpolant` row of the metrics table to the accuracy of
`SDE-conditional` — by construction, since the correction is
exact.

### 7.7 Caveat: LG-exact, not method-general

The derivation above relies on the isotropic Gaussian prior with
$\text{target\_mean}(x_0)=x_0$. For a learned drift or a
non-Gaussian prior, the same *structure* (correcting the
interpolant with a quadratic in $x_\tau$) still applies, but the
coefficients $A_\tau$ and $B_\tau$ are no longer globally valid —
they must be re-derived around a local linearization, which is
precisely what `LinearizedDriftLikelihood` and
`MultiStepLinearizedDriftLikelihood` do implicitly. The LG
correction factor is therefore best viewed as a **calibration
target** for the cheap corrections of
`cheap_interpolant_corrections.md`: any cheap recipe should, at
minimum, reproduce $(A_\tau, B_\tau)$ in the LG limit.
