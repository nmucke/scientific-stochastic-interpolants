# `SDEConditionalLikelihood`

A likelihood-score module that implements the **exact Doob h-transform** of
the prior interpolant SDE for the isotropic linear-Gaussian test case. It is
a drop-in replacement for `InterpolantLikelihood` inside `PosteriorModel`.

## Motivation

The posterior SDE (Theorem 3.1 in the paper) is obtained from the prior
SDE

$$
dX_\tau = b(X_\tau, X_0, \tau)\,d\tau + \gamma_\tau\,dW_\tau,
\qquad X_1 \sim p(\cdot \mid X_0),
$$

by a Doob h-transform with

$$
h(x_\tau, \tau) \;=\; \mathbb{E}_{\text{SDE}}\!\bigl[\,p(y \mid X_1)\,\bigm|\, X_\tau=x_\tau,\, X_0\bigr].
$$

The added drift is $\gamma_\tau^{2}\,\nabla_{x}\log h(x_\tau,\tau)$.

`InterpolantLikelihood` instead replaces $h$ with the Gaussian
$\bar{\Phi}_\tau = p(\bar{y}_\tau \mid x_\tau, x_0)$ derived from the
**interpolation joint** $(x_\tau, x_1)$ via Lemma 4.2 / 4.3. The
interpolation and the Markov SDE have matching *marginals* at every
$\tau$, but their *joint* laws for $(x_\tau, x_1)$ differ, so
$\bar{\Phi}_\tau \ne h_{\text{SDE}}$. This mismatch introduces
systematic bias in the posterior even in the linear-Gaussian case
(see the experiments in `main_analytical.py`).

`SDEConditionalLikelihood` computes $h_{\text{SDE}}$ directly.

## Setting

Assumptions baked into the class:

* Prior $p(x_0)$ is Gaussian.
* `target_mean(x_0) = x_0` and `target_cov(x_0) = c * I` (isotropic),
  with $c$ supplied via the `target_variance` argument.
* Observation model $y = H x_1 + \eta$ with $\eta\sim\mathcal{N}(0,\sigma^{2} I)$.
* The drift model is an `AnalyticalDriftModel` whose interpolation
  exposes `alpha, beta, gamma` and their time derivatives.

Under these assumptions the analytical prior drift from Proposition B.9
reduces to an **affine** function of $x$:

$$
b(x, \tau, x_0) \;=\; a(\tau)\,(x - x_0),
\qquad
a(\tau) \;=\; \frac{\beta_\tau\,\dot{\beta}_\tau\,c + \tau\,\gamma_\tau\,\dot{\gamma}_\tau}
                    {\beta_\tau^{\,2}\,c + \tau\,\gamma_\tau^{\,2}}.
$$

The SDE is therefore linear and its transition $X_\tau \to X_1$ is
closed-form Gaussian.

## What the class computes

For a given $\tau$ it computes two scalars:

1. **State-transition factor**
   $$
   \Phi_{1,\tau} \;=\; \exp\!\Bigl(\int_\tau^{1} a(s)\,ds\Bigr),
   $$
   computed by trapezoidal quadrature (`num_quad` nodes, default 200)
   over the interpolation coefficients.

2. **Conditional variance**
   $$
   V_{\text{cond}}(\tau) \;=\; c \;-\; \Phi_{1,\tau}^{\,2}\; V_\tau,
   \qquad V_\tau \;=\; \beta_\tau^{\,2}\,c + \tau\,\gamma_\tau^{\,2}.
   $$
   This is the unique variance consistent with the marginal
   $\operatorname{Var}(X_1\mid X_0)=c$ and the SDE transition above,
   obtained from the law-of-total-variance identity
   $c = \Phi^{2}\,V_\tau + V_{\text{cond}}$.

Together they give the **SDE conditional**

$$
X_1 \mid X_\tau, X_0 \;\sim\; \mathcal{N}\!\bigl(\,x_0 + \Phi_{1,\tau}\,(x_\tau - x_0),\; V_{\text{cond}}(\tau)\,I\,\bigr).
$$

Pushing through the linear observation $y = Hx_1 + \eta$ yields the
Gaussian
$y \mid x_\tau, x_0 \sim \mathcal{N}(\mu_y, \Sigma_y)$ with

$$
\mu_y \;=\; H\bigl(x_0 + \Phi_{1,\tau}(x_\tau - x_0)\bigr),
\qquad
\Sigma_y \;=\; V_{\text{cond}}\,H H^{\!\top} + \sigma^{2} I.
$$

The score

$$
\nabla_{x}\log p(y \mid x_\tau, x_0)
\;=\; \nabla_{x}\Bigl[-\tfrac{1}{2}(y - \mu_y)^{\!\top}\Sigma_y^{-1}(y - \mu_y)\Bigr]
$$

is returned via `torch.autograd.grad`, exactly as in the other
likelihood classes.

## Integration with `PosteriorModel`

`PosteriorModel.sample` applies the Euler–Maruyama update

$$
x_{i+1} \;=\; x_i \;+\; b(x_i, x_0, \tau_i)\,d\tau
\;+\; \gamma_{\tau_i}^{\,2}\,\nabla\log h\,d\tau
\;+\; \gamma_{\tau_i}\sqrt{d\tau}\,z_i,
$$

so the score returned here is multiplied by $\gamma_\tau^{\,2}\,d\tau$.
The dispatch in `posterior_model.py` therefore treats
`SDEConditionalLikelihood` the same way as `InterpolantLikelihood`
(see the `isinstance(..., (InterpolantLikelihood, SDEConditionalLikelihood))`
branch).

## Key implementation details

* `_phi_and_vcond(tau)` — discretises $[\tau, 1]$ with `num_quad`
  points, evaluates $a(s)$ via the interpolation coefficients, and
  integrates with `torch.trapz`. `V_cond` is clamped to be non-negative
  to guard against quadrature round-off near $\tau \to 1$.
* `score(...)` — stacks all batch elements through a single solve with
  $\Sigma_y$, which is shared across the batch because $\Phi$ and
  $V_{\text{cond}}$ depend only on $\tau$.
* The class needs `target_variance`; for the standard test case
  (`TARGET_COV = I`) the default `target_variance=1.0` is correct.

## Why this fixes the bias in the linear-Gaussian test

`main_analytical.py` reports Wasserstein distance between the sampled
posterior and the true Kalman-filter posterior. With
`SDEConditionalLikelihood` the distance drops to the Monte-Carlo /
KDE noise floor (~0.02 for all tested $\sigma^{2}$), whereas both
`InterpolantLikelihood` and `FlowdasLikelihood` leave a large residual
error. That is the empirical confirmation that replacing the
interpolation joint with the SDE transition recovers the exact Doob
h-transform, and hence the exact posterior, in the linear-Gaussian
setting.

## Limitations

* Requires an isotropic Gaussian prior with `target_mean(x_0) = x_0`.
  General non-Gaussian or anisotropic priors break the affine-drift
  assumption that makes $\Phi_{1,\tau}$ closed-form.
* Requires the analytical drift (so that $a(\tau)$ can be read off
  from the interpolation coefficients). For a learned drift one would
  need to estimate the linear coefficient differently, e.g. by
  differentiating the drift.
* The quadrature is 1D and cheap, but is recomputed per $\tau$ in the
  Euler–Maruyama loop. Caching across the time grid would be a trivial
  optimisation if needed.
