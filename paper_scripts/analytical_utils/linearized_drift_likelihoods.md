# Linearized-drift likelihood approximations

Two general-purpose classes that approximate the Doob h-transform
$h_{\text{SDE}}(x_\tau,\tau) = \mathbb{E}_{\text{SDE}}[p(y\mid X_1)\mid X_\tau, X_0]$
by linearizing the **learned drift** $b_\theta$ of the interpolant SDE.
Unlike `SDEConditionalLikelihood`, they do **not** require knowing
`target_mean` / `target_cov`. They only need $b_\theta$ and its
Jacobian $J = \partial b_\theta/\partial x$, both of which are available
for any (analytical or neural) drift model.

Both classes plug into `PosteriorModel` the same way as
`InterpolantLikelihood` — the returned score is multiplied by
$\gamma_\tau^{\,2}\,d\tau$ inside the Euler–Maruyama update.

---

## Shared construction

Given the (generally nonlinear) prior SDE

$$
dX_s = b_\theta(X_s, X_0, s)\,ds + \gamma_s\,dW_s,
$$

we want a Gaussian approximation of $X_1 \mid X_\tau, X_0$ so that
$\log p(y\mid X_1)$ can be integrated in closed form. Linearize the
drift around a reference point $(x^\star,\tau^\star)$,

$$
b_\theta(x, s, x_0) \;\approx\; b^\star + J^\star (x - x^\star),
\qquad b^\star = b_\theta(x^\star,\tau^\star,x_0),\; J^\star = \partial_x b_\theta(x^\star,\tau^\star,x_0),
$$

and treat the resulting **affine-linear** SDE as exact over some
interval $[s_0, s_1]$. The conditional mean $m(s)$ and covariance
$P(s)$ of $X_s$ then satisfy the ODEs

$$
\dot m \;=\; b^\star + J^\star(m - x^\star),\qquad
\dot P \;=\; J^\star P + P {J^\star}^{\!\top} + \gamma_s^{\,2}\,I,
$$

which are integrated forward by Euler steps. At $s = 1$ this gives

$$
X_1 \mid X_\tau, X_0 \;\approx\; \mathcal{N}\!\bigl(m(1),\; P(1)\bigr),
$$

and the observation likelihood becomes Gaussian with

$$
\mu_y = H\,m(1),\qquad \Sigma_y = H\,P(1)\,H^{\!\top} + \sigma^{2}\,I.
$$

The score $\nabla_{x_\tau}\log p(y\mid x_\tau,x_0)$ is obtained by
`torch.autograd.grad` on the Gaussian log-density.

The two classes differ only in **how the linearization point is
chosen and how often it is refreshed**.

---

## Option 1 — `LinearizedDriftLikelihood`

**Single linearization** at the current state $(x_\tau,\tau)$:

* Compute $b_0 = b_\theta(x_\tau,\tau,x_0)$ (one forward pass).
* Compute $J_0 = \partial_x b_\theta(x_\tau,\tau,x_0)$ using
  `torch.func.vmap(jacrev(...))` for batched autodiff.
* Set $x^\star := x_\tau$, $b^\star := b_0$, $J^\star := J_0$ and
  keep them **frozen** across the whole interval $[\tau, 1]$.
* Integrate $(m, P)$ with `num_quad` Euler steps; $\gamma_s$ is
  evaluated at each quadrature node.
* Initialize $m(\tau)=x_\tau$, $P(\tau)=0$.

**Cost per reverse-SDE step:** one drift call + one batched Jacobian
+ `num_quad` cheap matrix Euler updates.

**When it's accurate:** $b_\theta$ is approximately linear in $x$
over $[\tau,1]$ and $(1-\tau)$ is small (i.e. late in the reverse
run).

**When it fails:** strongly nonlinear drifts, or early reverse-time
steps where the interval $[\tau,1]$ is long — the constant-$J$
extrapolation drifts away from the true mean, biasing both $m(1)$
and $P(1)$.

**Empirical check (analytical LG test, isotropic prior $c=1$):**
on the test case where `SDEConditionalLikelihood` is exact,
`LinearizedDriftLikelihood` matches it closely at low observation
noise but degrades at higher $\sigma^{2}$ because the reverse-time
corrections happen over larger $[\tau,1]$ intervals.

---

## Option 3 — `MultiStepLinearizedDriftLikelihood`

**$K$-step re-linearization.** Splits $[\tau,1]$ into
`num_substeps` equal segments $[\tau_k,\tau_{k+1}]$ and, at the start
of each segment, re-computes the drift and Jacobian at the current
mean:

For $k = 0, 1, \dots, K-1$:

1. Set $x^\star_k := m(\tau_k)$, $\tau^\star_k := \tau_k$.
2. Compute $b^\star_k = b_\theta(x^\star_k, \tau^\star_k, x_0)$
   (batched forward pass).
3. Compute $J^\star_k = \partial_x b_\theta(x^\star_k, \tau^\star_k, x_0)$
   via `vmap(jacrev(...))`.
4. Advance $(m, P)$ across $[\tau_k,\tau_{k+1}]$ with
   `num_euler_per_substep` Euler steps using the frozen
   $(b^\star_k, J^\star_k)$.

`m` at the end of segment $k$ is used as the linearization point
for segment $k+1$, so the reference trajectory tracks the actual
posterior-mean path rather than extrapolating from a single frozen
point.

**Cost per reverse-SDE step:** `num_substeps` drift calls +
`num_substeps` batched Jacobians + `num_substeps * num_euler_per_substep`
Euler updates. ~$K\times$ option 1.

**When to use it:** whenever option 1 is not accurate enough —
nonlinear drifts, long $[\tau,1]$ intervals, or just for tighter
posteriors. A modest $K=3$–$5$ typically recovers most of the
improvement.

**Empirical check (same LG test):** with
`num_substeps=5, num_euler_per_substep=10`, the Wasserstein distance
to the true Kalman posterior is ~$0.02$–$0.06$ across
$\sigma^{2}\in\{0.5,1,2\}$, essentially matching
`SDEConditionalLikelihood` (which is analytically exact for that
test). Option 1 with a single linearization was markedly worse at
$\sigma^{2}=2$ ($W\approx 0.21$) — exactly the regime where the
reverse-time intervals are longest and a single frozen Jacobian is
least accurate.

---

## Implementation notes shared by both classes

* **Batching.** Jacobians are computed with
  `vmap(jacrev(single_drift, argnums=0))` so the full batch of
  samples is processed in one vectorized call.
* **Mean / covariance integration** is implemented as explicit Euler
  because the ODEs are linear and trivially batched via
  `torch.bmm`. A matrix-exponential solve would be marginally more
  accurate but not meaningfully cheaper at the step counts used here.
* **Score backprop.** Only the Gaussian log-density is
  differentiated w.r.t. $x_\tau$ via `autograd.grad`. The Jacobian
  from `jacrev` is used as a numerical quantity inside the forward
  pass (the same pattern as `InterpolantLikelihood`).
* **Covariance initialization.** $P(\tau)=0$ because we condition on
  $X_\tau = x_\tau$; all uncertainty in $X_1\mid X_\tau$ accrues from
  the diffusion term $\gamma_s\,dW_s$ on $(\tau,1]$.
* **`PosteriorModel` dispatch.** Both classes are included in the
  `isinstance` branch that applies the $\gamma_\tau^{\,2}\,d\tau$
  scaling — exactly as for `InterpolantLikelihood` and
  `SDEConditionalLikelihood`.

---

## How they relate to `SDEConditionalLikelihood`

`SDEConditionalLikelihood` is the **closed-form** limit of these
classes in the special case where

* the prior is Gaussian with isotropic `target_cov`,
* `target_mean(x_0) = x_0`,
* the analytical drift reduces to $b(x,\tau,x_0)=a(\tau)(x-x_0)$ with
  scalar $a(\tau)$.

In that setting $J$ is just $a(\tau)\,I$ and the ODEs can be
integrated analytically via $\Phi_{1,\tau} = \exp\!\int_\tau^1 a(s)\,ds$.
For any other prior or drift, $J$ is a full matrix that depends on
$x$, and one must resort to the Euler-based integration used by
options 1 and 3.

Options 1 and 3 therefore subsume `SDEConditionalLikelihood` and are
the intended path for learned drifts and non-Gaussian priors.
