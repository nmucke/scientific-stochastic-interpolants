# Cheap corrections to `InterpolantLikelihood`

The "correct-but-expensive" route (linearizing $b_\theta$ and integrating
the resulting linear SDE) is impractical for high-dimensional,
strongly nonlinear settings: the Jacobian `jacrev` costs $O(d)$
backward passes, and $K$-step re-linearization multiplies that by $K$.
Those costs are exactly what motivated the original interpolant
likelihood in the first place.

This note collects **cheap** ways to perturb or rescale the existing
`InterpolantLikelihood` score so that it better approximates the true
Doob h-transform
$h_{\text{SDE}}(x_\tau,\tau)=\mathbb{E}_{\text{SDE}}[p(y\mid X_1)\mid X_\tau,X_0]$,
without ever computing a full Jacobian.

---

## 1. Where the bias comes from

The current class assembles a Gaussian likelihood from

$$
r(x_\tau) \;=\; \bar y_\tau - H x_\tau - d_\tau(x_\tau),
\qquad
\Sigma_\tau \;=\; \beta_\tau^{\,2}\sigma^{2} I + \gamma_\tau^{\,2}\,H\,C_\tau\,H^{\!\top},
$$

with $d_\tau$ and $C_\tau$ derived from the **interpolation joint**
$(x_\tau,x_1)$ via Lemmas 4.2/4.3. Compared with the true Doob
$h$-transform of the **prior SDE**, two separate errors appear:

1. **Mean error.** The "implicit $x_1$ estimator" hidden inside
   $d_\tau$ uses the interpolation-conditional mean slope
   $\partial\mathbb{E}[X_1\mid x_\tau,x_0]/\partial x_\tau
   = 1/(1-\tau+\tau^2)\;I$ in the LG test, whereas the SDE conditional
   has slope $\Phi_{1,\tau}\;I$. These differ by $O(1)$ for
   $\tau\in(0,1)$.
2. **Covariance error.** $C_\tau$ is the conditional Wiener variance
   under the interpolation joint. The SDE joint gives a different
   conditional variance and requires a Jacobian of $b_\theta$ that
   the class currently computes with `jacrev`.

Numerical evidence from the LG test case: at $\tau=0.5$, the **mean**
is wrong by $\sim18\%$ (slope $1.33$ vs $1.56$), while the
**covariance** is off by only a few percent. **Most of the posterior
error comes from the mean.** That makes mean corrections the
highest-leverage place to spend a small amount of extra compute.

---

## 2. Cheap mean corrections

These corrections replace or perturb the mean of the Gaussian
likelihood without touching the covariance block.

### 2.1 Scalar time-warp $\tilde\tau(\tau)$  — **essentially free**

Define a one-dimensional warping $\tilde\tau(\tau)$ such that the
interpolation-conditional mean at time $\tilde\tau$ equals the true
SDE-conditional mean at time $\tau$. Concretely, if you have a rough
surrogate (e.g. an isotropic Gaussian prior fit to low-order moments
of the target), you can precompute the SDE slope $\Phi_{1,\tau}$
offline and invert the interpolation slope function to find
$\tilde\tau(\tau)$. Inside the score computation, replace
$(\alpha_\tau,\beta_\tau,\gamma_\tau)$ with
$(\alpha_{\tilde\tau},\beta_{\tilde\tau},\gamma_{\tilde\tau})$
everywhere except the observation $\bar y_\tau = \alpha_\tau H x_0 + \beta_\tau y$.

* **Cost:** one 1-D table lookup per reverse step.
* **Works when:** the bias is primarily a time-axis reparametrization,
  which it is for roughly-Gaussian priors.
* **Fails when:** the SDE dynamics are strongly nonlinear in ways not
  captured by a time rescaling alone — then the warp is per-sample,
  not per-time.

### 2.2 Scalar correction factor $\lambda_\tau$ on $d_\tau$  — **free**

Rewrite the current residual as
$r = (\bar y_\tau - H x_\tau) - d_\tau$ and introduce a scalar
multiplier $\lambda_\tau \in \mathbb{R}$:

$$
r_{\lambda}(x_\tau) \;=\; (\bar y_\tau - H x_\tau) - \lambda_\tau \,d_\tau.
$$

$d_\tau$ is built from the learned score $s_\theta(x_\tau,\tau,x_0)$
(via `_compute_score_from_drift` in the analytical drift), so the
correction is just a rescaling of the existing quantity — no new
drift calls, no Jacobians.

#### 2.2.1 What $\lambda_\tau$ actually does

To see the effect of $\lambda_\tau$, rewrite the residual as
$r_\lambda/\beta_\tau = y - H\,\hat x_1^{\lambda}(x_\tau)$, where
$\hat x_1^{\lambda}$ is the implicit "$x_1$ estimator" that the
likelihood is regressing $y$ against. For the linear-Gaussian
test case (linear interpolation $\alpha=1-\tau,\beta=\tau,\gamma=1-\tau$,
target mean $x_0$, isotropic prior covariance $cI$), the closed-form
score gives $d_\tau = -(\gamma_\tau^{\,2}\tau/V_\tau)\,H\,(x_\tau - x_0)$
with $V_\tau = \beta_\tau^{\,2}c + \tau\gamma_\tau^{\,2}$, and the
implicit predictor is

$$
\hat x_1^{\lambda}(x_\tau)
\;=\; x_0 \;+\; \underbrace{\frac{1}{\tau}\!\left(1 - \lambda_\tau\,\frac{\gamma_\tau^{\,2}\tau}{V_\tau}\right)}_{=:\;s^{\text{interp}}_\lambda(\tau)}\,(x_\tau - x_0).
$$

So **$\lambda_\tau$ controls the slope of the $x_1$-predictor with
respect to $x_\tau$**, linearly interpolating between two limits:

* $\lambda_\tau = 0$: raw interpolation inverse, slope $1/\tau$.
* $\lambda_\tau = 1$: current `InterpolantLikelihood`, slope
  $\beta_\tau c/V_\tau$. At $\tau=0.5$, $c=1$ this is $\approx 1.333$.

The **target slope** is $\Phi_{1,\tau}$, the Doob transition factor of
the prior SDE (see `SDEConditionalLikelihood`). At $\tau=0.5$ in the
LG test, $\Phi_{1,\tau}\approx 1.562$, which sits **between** $1.333$
and $1/\tau = 2$. The interpolation overshoots the correction and
$\lambda_\tau < 1$ is needed to pull the slope back toward the raw
inverse.

#### 2.2.2 Closed-form $\lambda_\tau$ from a Gaussian surrogate

Set $s^{\text{interp}}_\lambda(\tau) = \Phi_{1,\tau}$ and solve:

$$
\boxed{\;\lambda_\tau \;=\; \frac{V_\tau\,(1 - \tau\,\Phi_{1,\tau})}{\gamma_\tau^{\,2}\,\tau}\;}
$$

(linear-interpolation case; the analogous formula for other
schedules replaces $\tau$ with $\beta_\tau$ where appropriate).

**Worked example, $\tau=0.5$, $c=1$:**
$V_\tau = 0.375$, $\Phi_{1,\tau}\approx 1.5622$,
$\gamma_\tau^{\,2}\tau = 0.125$, so

$$
\lambda_{0.5} \;=\; \frac{0.375\,(1 - 0.5\cdot 1.5622)}{0.125}
\;\approx\; 0.657.
$$

Plugging this back into the slope formula gives
$s^{\text{interp}}_{\lambda}(0.5) = 1.562 = \Phi_{1,0.5}$ ✓.

#### 2.2.3 How to get $\Phi_{1,\tau}$ when you don't know the prior

The formula above needs $\Phi_{1,\tau}$, which in turn needs *some*
statement about the prior. Three options in order of practicality:

1. **Isotropic Gaussian surrogate.** Pick a single scalar $c$ to
   represent the bulk scale of your prior — e.g. $c$ = trace of the
   empirical covariance of your training data divided by $d$, or the
   marginal variance along a dominant PC. Plug $c$ into the LG formula
   for $a(s) = [\beta\beta' c + s\gamma\gamma']/[\beta^2 c + s\gamma^2]$
   and compute $\Phi_{1,\tau} = \exp\!\int_\tau^1 a(s)\,ds$ by
   quadrature — exactly what `SDEConditionalLikelihood._phi_and_vcond`
   already does. All of this is **offline**, once per
   $(\tau$-grid, $c)$ pair.
2. **Moment matching from forward simulations.** Simulate $M$ prior
   trajectories from $x_0$ with the learned SDE. At each $\tau$ on a
   coarse grid, compute the empirical slope of the regression
   $x_1 - x_0$ on $x_\tau - x_0$ (scalar, via the ratio of sample
   covariance to sample variance in a dominant direction). Use that
   empirical slope directly as $\Phi_{1,\tau}$. This removes the
   "isotropic Gaussian" assumption and captures what the **actual
   learned drift** does on average. Cost: offline; $M$ of order a
   few hundred is plenty for a smooth 1-D function.
3. **Fit to a held-out validation set with known posteriors.**
   If you have *any* tractable-posterior cases (even a 1-D slice or
   a small test problem), fit $\lambda_\tau$ by minimizing
   $\|\text{sampled posterior} - \text{true posterior}\|$ in some
   metric. This is the most pragmatic choice for a paper: use the
   analytical LG test as the calibration set, extract
   $\lambda_\tau$ from it, and deploy the same schedule on the
   Navier–Stokes / weather experiments.

Option 1 is essentially free and gives a principled baseline.
Option 2 is also free (offline) and captures learned-drift effects.
I would recommend fitting **option 1 analytically and option 2 as
a sanity check**, then freezing the $\lambda_\tau(\cdot)$ schedule.

#### 2.2.4 Caveats

* **One scalar can't fix both mean and variance.** $\lambda_\tau$ is
  chosen to match the mean slope; the covariance still has the
  interpolation-joint bias. Combine with §3.1 / §3.4 to address
  that separately.
* **Bias in the direction, not just the magnitude.** Because
  $\lambda_\tau$ multiplies the vector $d_\tau$, it cannot rotate
  the direction of the score — only rescale its component along
  $d_\tau$. This is fine in the LG case where $d_\tau$ already
  points along the right direction, but in strongly non-Gaussian
  settings the directional error is what dominates, and $\lambda_\tau$
  won't fix it.
* **Sign.** Nothing prevents $\lambda_\tau$ from being negative. In
  the LG test it is positive and $<1$, but for other interpolation
  schedules or prior shapes you may find $\lambda_\tau > 1$ at some
  $\tau$ — that's fine, it just means the interpolation
  *under*-corrects there.

* **Cost:** zero at inference (a scalar per $\tau$, precomputed).
* **Can be combined** with the time-warp in §2.1 and the covariance
  tweaks in §3.

### 2.3 Tweedie / one-step ODE predictor  — **one extra drift call**

Replace the interpolation-implicit $\hat x_1^{\text{interp}}$ with a
deterministic Tweedie/probability-flow estimate:

$$
\hat x_1^{\text{ODE}}(x_\tau)
\;=\; x_\tau + (1-\tau)\,b_\theta(x_\tau,\tau,x_0).
$$

Then form the Gaussian likelihood score with mean $H\hat x_1^{\text{ODE}}$
and covariance $\Sigma_\tau$ (keep the interpolation covariance for
now — §3 handles it separately).

This is what FlowDAS does inside `_compute_one_step_prediction`, but
FlowDAS then wraps the prediction in a softmax particle-filter
reweighting that loses the Gaussian structure. The same one-step
prediction, plugged into a proper Gaussian log-likelihood and
differentiated, gives a better-behaved score at the same cost.

* **Cost:** one extra $b_\theta$ evaluation per reverse step. In
  neural settings this dominates, so total cost is ~$2\times$ the
  baseline reverse SDE.
* **Accuracy:** often better than the interpolation estimate at
  moderate nonlinearities, because it takes one step **forward under
  the real dynamics** instead of inverting the interpolation.
* **Can be upgraded** to a $K$-step probability-flow ODE rollout for
  extra cost $K\times$ drift.

### 2.4 Heun / RK2 refinement  — **two extra drift calls**

Upgrade 2.3 to a second-order predictor. Let $b_s := b_\theta(\cdot,s,x_0)$
and run one Heun step from $(x_\tau,\tau)$ to $s=1$:

$$
\hat x_1^{(1)} \;=\; x_\tau + (1-\tau)\,b_\tau(x_\tau),
\qquad
\hat x_1^{\text{Heun}} \;=\; x_\tau + \tfrac{1-\tau}{2}\bigl(b_\tau(x_\tau) + b_{1}(\hat x_1^{(1)})\bigr).
$$

Two drift evaluations per reverse step, second-order accurate in
$(1-\tau)$. For smooth $b_\theta$ this removes the leading-order
bias in the mean predictor without any Jacobian.

#### 2.4.1 Where does the bias term $d_\tau$ enter?

**Short answer: it doesn't.** Once you have a direct SDE-based
predictor $\hat x_1^{\text{Heun}}(x_\tau)$, the interpolation-joint
construction ($\bar y_\tau$, $d_\tau$, Lemma 4.2) becomes redundant
and is replaced in its entirety.

The $d_\tau$ term in `InterpolantLikelihood` exists for one reason:
the class treats the likelihood as $\bar y_\tau \mid x_\tau$ rather
than $y \mid x_\tau$. Under the interpolation joint,

$$
\bar y_\tau = \alpha_\tau H x_0 + \beta_\tau y = H x_\tau + \underbrace{\beta_\tau\,\eta - \gamma_\tau H W_\tau}_{\text{observation noise}},
$$

so $\mathbb E[\bar y_\tau\mid x_\tau, x_0] = H x_\tau + d_\tau$ with
$d_\tau = -\gamma_\tau H\,\mathbb E[W_\tau\mid x_\tau,x_0]$. That is,
$d_\tau$ is a "mean shift" that makes $\bar y_\tau$ an unbiased
estimator of $H x_\tau$ under the interpolation conditional — it
exists entirely to correct for the fact that $\bar y_\tau$ contains
the correlated Wiener contribution $\gamma_\tau H W_\tau$, which is
*not* independent of $x_\tau$.

In the Heun-based predictor you don't mix $y$ into a
time-interpolated surrogate $\bar y_\tau$ at all. You write the
likelihood directly on $y$:

$$
y \mid x_\tau, x_0 \;\approx\; \mathcal{N}\!\bigl(H\hat x_1^{\text{Heun}}(x_\tau),\; \Sigma_\tau^{\text{pred}}\bigr),
$$

where $\hat x_1^{\text{Heun}}$ is the SDE-roll-out mean and
$\Sigma_\tau^{\text{pred}} = H\,P_\tau\,H^{\!\top} + \sigma^{2} I$ for
some choice of predictive covariance $P_\tau$ (see §3 for options).
The corresponding residual is

$$
r^{\text{Heun}}(x_\tau) \;=\; y - H\,\hat x_1^{\text{Heun}}(x_\tau),
$$

and the score is
$\nabla_{x_\tau}\bigl[-\tfrac{1}{2}\,r^{\text{Heun}\!\top}\,\Sigma_\tau^{\text{pred}\,-1}\,r^{\text{Heun}}\bigr]$,
computed by a single `autograd.grad` pass through the Heun
predictor. Nothing from the interpolation joint — $\bar y_\tau$,
$d_\tau$, $\mathbb E[W_\tau\mid x_\tau,x_0]$ — survives.

#### 2.4.2 Why the Heun predictor subsumes $d_\tau$

Both $d_\tau$ and $\hat x_1^{\text{Heun}}$ are answers to the same
underlying question: *"given $x_\tau$, what is the best estimate of
$x_1$ under the prior dynamics?"* They just use different machinery:

* `InterpolantLikelihood` constructs an *implicit* predictor
  $\hat x_1^{\text{interp}}(x_\tau) = (x_\tau - \alpha_\tau x_0 + d_\tau)/\beta_\tau$,
  obtained by inverting the interpolation $x_\tau = \alpha_\tau x_0 + \beta_\tau x_1 + \gamma_\tau W_\tau$
  under its own joint law. The correction $d_\tau$ is precisely the
  term that converts the raw inverse $(x_\tau-\alpha_\tau x_0)/\beta_\tau$
  into the conditional mean of $x_1$ under the **interpolation**
  joint.
* Heun constructs an *explicit* predictor by integrating the prior
  SDE forward from $(x_\tau,\tau)$ to $s=1$ with two drift calls —
  i.e., it asks the real dynamics directly instead of inverting a
  surrogate joint. The result is the conditional mean of $x_1$
  under the **SDE** joint (to second order in $(1-\tau)$).

Both are trying to produce the same object, $\hat x_1\approx
\mathbb E[X_1 \mid x_\tau, x_0]$. Using both at the same time would
be double-counting. The Heun version replaces the interpolation
inverse **and** its bias correction $d_\tau$ with a single
dynamics-consistent estimate, and that is why $d_\tau$ simply
drops out of the §2.4 recipe.

#### 2.4.3 Hybrid: Heun mean with interpolation covariance

If you're unhappy about losing the Lemma 4.3 covariance structure
(which has some validity even under the SDE joint), you can use a
hybrid: **Heun mean, interpolation covariance**. Concretely,

$$
r^{\text{hybrid}}(x_\tau) \;=\; y - H\,\hat x_1^{\text{Heun}}(x_\tau),
\qquad
\Sigma_\tau^{\text{hybrid}} \;=\; \sigma^{2} I + \text{(existing $\Sigma_\tau$ from `sigma_fn`)}.
$$

This uses $\hat x_1^{\text{Heun}}$ only to fix the *mean* of the
Gaussian and leaves the covariance untouched. $d_\tau$ is still
gone — the mean shift is no longer needed — but the covariance
correction from `_compute_W_covariance` is retained. If you want
to kill the Jacobian cost in the covariance as well, pair this
with §3.1 or §3.2.

#### 2.4.4 Practical notes

* **Second drift call at $s=1$.** The Heun formula evaluates
  $b_\theta$ at $s=1$. If the learned drift blows up or is undefined
  at the endpoint, clamp to $s = 1 - \epsilon$.
* **Differentiability.** Both $\hat x_1^{(1)}$ and $\hat x_1^{\text{Heun}}$
  are smooth functions of $x_\tau$ built from two drift calls, so
  `autograd.grad` through the Gaussian log-density works without
  any special tricks.
* **Relationship to 2.3.** 2.3 is the first-order Euler version
  (one drift call); 2.4 is the second-order Heun version (two drift
  calls). If even one extra drift call is too expensive, fall back
  to 2.3; if you want third-order accuracy, use a 3-stage RK
  method — but in practice Heun is the sweet spot.

---

## 3. Cheap covariance corrections

These corrections keep the Jacobian out of the computation entirely,
or replace the full Jacobian with a projection.

### 3.1 Drop the correction term  — **free**

Simply use $C_\tau \approx \tau I$ (the unconditional Wiener variance).
This removes the $\gamma_\tau^{\,2}\tau^{\,2} A_\tau(\beta_\tau J_{b_\theta} - \dot\beta_\tau I)$
piece from `_compute_W_covariance` and eliminates all
`jacrev` calls.

* **Cost:** zero.
* **When it's fine:** the covariance correction is typically $O(\gamma_\tau^{\,2}\tau^{\,2})$
  relative to $\tau I$, i.e. small away from $\tau = 1$. For most of
  the reverse trajectory the uncorrected covariance is within a few
  percent of the corrected one.
* **Caveat:** near $\tau\to 1$ the correction becomes important; use
  one of the next options there.

### 3.2 Observation-space VJP instead of full Jacobian  — **$N_y$ backward passes**

The full $d\times d$ Jacobian $J_{b_\theta}$ is computed only to form
$H J_{b_\theta}$ (and then $H J_{b_\theta} H^{\!\top}$). Since $H\in\mathbb{R}^{N_y\times d}$
with $N_y\ll d$ in typical DA problems, compute $H J_{b_\theta}$
directly via $N_y$ **vector-Jacobian products**:

```python
# Conceptually: HJ[i, :] = H[i, :] @ J_b = vjp(b_theta, H[i, :])
_, vjp_fn = torch.func.vjp(lambda xx: b_theta(xx, t, x0), x)
HJ = torch.stack([vjp_fn(H[i])[0] for i in range(N_y)])  # (N_y, d)
HJH = HJ @ H.T                                            # (N_y, N_y)
```

This gives the *exact* same $H J_{b_\theta} H^{\!\top}$ used by the current
class, but replaces $d$ backward passes with $N_y$. For
$d=10^{5}, N_y=10$ this is **four orders of magnitude** cheaper.

* **Cost:** $N_y$ backward passes per reverse step.
* **Accuracy:** identical to the current implementation (no
  approximation).
* **When to use:** whenever $H$ is known and $N_y \ll d$. This is
  the default regime for data assimilation.
* **Gotcha:** for nonlinear observation operators, replace $H$ with
  the VJP of the observation map at $x_\tau$.

### 3.3 Hutchinson-style random projection  — **$M$ backward passes**

When $N_y$ is large *or* the observation operator is nonlinear and
the required quantity is $\operatorname{tr}(H J_{b_\theta} H^{\!\top}\Sigma^{-1})$-like, use a
random estimator:

$$
H J_{b_\theta} H^{\!\top} \;\approx\; \frac{1}{M}\sum_{m=1}^{M} (H J_{b_\theta} \xi_m)(H\xi_m)^{\!\top},
\qquad \xi_m\sim\mathcal{N}(0,I_d).
$$

Each sample requires one VJP (or JVP) on $b_\theta$, so cost is
$M$ backward passes. Even $M = 4$–$8$ gives a usable estimate of
the correction in the covariance, at a fraction of the full-Jacobian
cost.

* **Cost:** $M$ backward passes, $M$ chosen adaptively.
* **Accuracy:** unbiased, variance $O(1/M)$.
* **When to use:** nonlinear observation, or very large $N_y$.

### 3.4 Scalar covariance inflation $\kappa_\tau$  — **free**

In the same spirit as §2.2, multiply $\Sigma_\tau$ by a precomputed
scalar:

$$
\tilde\Sigma_\tau \;=\; \beta_\tau^{\,2}\sigma^{2} I + \kappa_\tau\,\gamma_\tau^{\,2}\,H\,\tau I\,H^{\!\top}.
$$

Fit $\kappa_\tau$ offline on a Gaussian surrogate so that
$\tilde\Sigma_\tau$ matches the SDE-conditional covariance at a few
pivot $\tau$ values. This is an extremely cheap replacement for
the full Jacobian correction and captures most of its effect when
the drift's nonlinearity is mild.

* **Cost:** zero at inference.
* **Use alongside** §2.1 or §2.2 on the mean.

### 3.5 Rank-1 directional correction  — **1 JVP**

Pick a single direction $v$ (e.g. the current residual, the score,
or a dominant eigenvector of $HH^{\!\top}$) and compute the scalar
$v^{\!\top} J_{b_\theta} v$ via one JVP. Use it to inflate the covariance
along $Hv$:

$$
\tilde\Sigma_\tau \;=\; \Sigma_\tau^{(0)} + c_\tau (v^{\!\top} J_{b_\theta} v)\,(Hv)(Hv)^{\!\top},
$$

where $\Sigma_\tau^{(0)}$ is the "no correction" covariance from §3.1
and $c_\tau$ is a geometry factor from Lemma 4.3. This is the
rank-1 approximation to the full correction, at the cost of a
single directional derivative of $b_\theta$.

* **Cost:** one extra JVP per reverse step.
* **When to use:** as a cheap middle ground between §3.1 (free,
  biased) and §3.2 ($N_y$-JVP, exact-but-projected).

---

## 4. Recommended recipes

Three combinations, ordered by compute budget:

### Recipe A — "essentially free" (no extra drift calls)

* Mean: **time-warp** $\tilde\tau(\tau)$ (§2.1) **or** scalar factor
  $\lambda_\tau$ (§2.2).
* Covariance: **drop the Jacobian term** (§3.1) and apply scalar
  inflation $\kappa_\tau$ (§3.4).

Fit $\tilde\tau, \lambda_\tau, \kappa_\tau$ once, offline, against an
isotropic Gaussian surrogate with the same interpolation schedule
and diffusion. At inference it's just arithmetic on the existing
score — **no new drift calls, no Jacobians.** This is what I would
try first on a new high-dimensional problem.

### Recipe B — "one extra drift call"

* Mean: **Tweedie/one-step ODE predictor** (§2.3).
* Covariance: **VJP in observation space** (§3.2) with a $\tau$-dependent
  gate that skips the VJP for $\tau$ well below 1.

Cost is ~$2\times$ baseline. Gives a sharp improvement whenever the
drift is strong but not too non-smooth.

### Recipe C — "Heun step + rank-1 covariance"

* Mean: **Heun / RK2 predictor** (§2.4).
* Covariance: **rank-1 directional correction** (§3.5) using the
  current residual as the probe direction.

Cost: 2 drift calls + 1 JVP per reverse step. This is the highest
accuracy you can get without anything Jacobian-like, and is where I
would stop if Recipe B still had a visible bias.

---

## 5. Things that don't help (and why)

* **Multiplying the final score by a scalar** (pure $\alpha\,\nabla\log \bar\Phi$).
  The gradient *direction* is unchanged, so it can't fix a mean
  error — it only rescales step size. Combine with a DPS-like
  schedule if you need step-size control, but don't expect it to
  correct the bias.
* **Inflating $\Sigma$ with a huge constant.** Softens the score
  indiscriminately. In the LG test this reduced the bias but at the
  price of underfitting the posterior — better to use the targeted
  $\kappa_\tau$ from §3.4.
* **Second-order Taylor of $\log\bar\Phi$.** Computing the Hessian
  of $\log\bar\Phi$ is no cheaper than the Jacobian route it was
  supposed to replace.

---

## 6. Open questions for the paper

1. Is there a principled offline procedure for fitting $\tilde\tau$,
   $\lambda_\tau$, $\kappa_\tau$ that doesn't require knowing the
   true posterior? E.g., matching a few low-order moments of the
   SDE conditional vs. the interpolation conditional analytically.
2. For learned $b_\theta$, does Recipe B's one-step ODE predictor
   inherit the bias of $b_\theta$ (i.e. do bad drifts lead to bad
   mean predictors)? An error-propagation analysis would be useful.
3. Recipe B's observation-space VJP trick is essentially free when
   $N_y$ is small. Is there any reason not to always include it in
   the paper's default recommendation?
4. Does §3.5's rank-1 covariance correction have a principled choice
   of probe direction $v$, or is any of {score, residual, dominant
   $H$ eigenvector} equally good in practice? Worth an ablation.

---

## 7. Relationship to `LinearizedDriftLikelihood` and `SDEConditionalLikelihood`

The linearized-drift classes give the *exact* Gaussian approximation
of the SDE conditional under the assumption that $b_\theta$ is
locally affine — at the cost of a full Jacobian. The cheap
corrections above are best thought of as **specific low-rank
projections of that exact computation**:

* §3.1 (drop term) = rank-0 projection.
* §3.5 (rank-1) = rank-1 projection.
* §3.2 (observation-space VJP) = projection onto the $N_y$-dim
  observation subspace — exact in that subspace, zero outside it.
* §3.3 (Hutchinson) = random rank-$M$ projection.

Viewed this way, the recipes are not heuristics but deliberate
low-rank compressions of the same underlying object. This gives a
clean story for the paper: the expensive method (§3.2/3.5) is a
drop-in exact computation, and the cheap method is a principled
projection of it.
