"""Closed-form posterior samplers for the analytical linear--Gaussian case.

Spec Section 4. The prior transition is the analytic Gaussian
``x1 | x0 ~ N(x0, I)`` and the observation model is ``y = H x1 + e``,
``e ~ N(0, R)`` with ``H = I``, ``R = I``; the exact posterior is
``N(x0 + K(y - H x0), I - K H)`` with ``K = 0.5 I``.

Because no network is trained here, every quantity the unified posterior drift
needs (the prior velocity/score, the source conditional moments, the
interpolant-likelihood score and the multiplicative gain) is available in closed
form. We implement the three samplers of the paper -- ``SI-SDE``, ``DM-SDE`` and
``FM-ODE`` -- directly as vector SDE/ODE integrators on this analytic prior,
parameterised by the three config-selectable likelihood modes
(``inflated`` / ``dps_full`` / ``dps_jacobian_free``).

All three samplers share one Euler-Maruyama loop (Algorithm 1); they differ only
in source / anchor / diffusion / guidance weight, exactly as in the spec table.

Closed forms used (target mean ``m1 = x0``, ``Cov1 = I``):

* **SI** interpolant ``alpha x0 + beta x1 + gamma W_tau`` with
  ``alpha = 1 - tau``, ``beta = tau``, ``gamma = g0 (1 - tau)`` and the Wiener
  source ``sigma_tau = gamma sqrt(tau)``. The marginal of ``x_tau`` is Gaussian
  with mean ``mbar = alpha x0 + beta x0 = x0`` and covariance
  ``Cbar = (beta^2 + tau gamma^2) I``; the prior SDE drift is the Foellmer drift
  (Prop. B.9 / ``analytical_utils``).
* **FM** rectified path ``alpha z + beta x1`` with ``alpha = 1 - tau``,
  ``beta = tau``, ``z ~ N(0, I)``. With target ``x1 ~ N(x0, I)`` the marginal of
  ``x_tau`` is ``N(beta x0, (alpha^2 + beta^2) I)``; the velocity and score are
  the affine-Gaussian closed forms.

The source conditional moments ``E[x1 | x_tau, x0]`` and
``Cov(x1 | x_tau, x0)`` are Gaussian-exact; the interpolant likelihood is then a
plain Gaussian whose score has the multiplicative-gain structure of the paper.
The three modes select which ``Sigma_s`` enters the gain.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class GaussianSystem:
    """The analytic linear--Gaussian system (spec Section 4).

    ``H = I``, ``R = obs_var I``; prior transition ``x1 | x0 ~ N(x0, I)``.
    """

    d: int
    obs_var: float = 1.0  # R = obs_var * I
    prior_var: float = 1.0  # Cov(x1 | x0) = prior_var * I  (=1 in the spec)

    def kalman_gain(self) -> float:
        """Scalar gain ``K`` (isotropic): ``K = c / (c + R)`` with ``c = prior_var``."""
        return self.prior_var / (self.prior_var + self.obs_var)

    def exact_posterior_moments(
        self, x0: Tensor, y: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Exact posterior mean and (isotropic) covariance scale.

        Returns ``(mean[d], cov_scale)`` with covariance ``cov_scale * I``.
        """
        K = self.kalman_gain()
        mean = x0 + K * (y - x0)  # H = I
        cov_scale = (1.0 - K) * self.prior_var
        return mean, torch.tensor(cov_scale, dtype=x0.dtype, device=x0.device)

    def exact_posterior_samples(
        self, x0: Tensor, y: Tensor, n: int, generator: torch.Generator
    ) -> Tensor:
        """Draw ``n`` exact-posterior samples ``[n, d]``."""
        mean, cov_scale = self.exact_posterior_moments(x0, y)
        z = torch.randn(n, self.d, generator=generator, dtype=x0.dtype)
        return mean.unsqueeze(0) + cov_scale.sqrt() * z


# --------------------------------------------------------------------------- #
# Schedules (rectified / linear: alpha = 1 - t, beta = t, gamma = g0 (1 - t)).
# --------------------------------------------------------------------------- #


def _alpha(t: Tensor) -> Tensor:
    return 1.0 - t


def _beta(t: Tensor) -> Tensor:
    return t


def _gamma(t: Tensor, g0: float) -> Tensor:
    return g0 * (1.0 - t)


# --------------------------------------------------------------------------- #
# Prior drift / velocity / score (closed form).
# --------------------------------------------------------------------------- #


def _si_prior_drift(x: Tensor, x0: Tensor, t: float, g0: float, prior_var: float) -> Tensor:
    """SI Foellmer prior SDE drift ``b(x, x0, tau)`` (Prop. B.9).

    ``b = adot x0 + bdot m1 + (beta bdot c + tau gamma gdot) Cbar^{-1}(x - mbar)``
    with ``m1 = x0``, ``mbar = x0``, ``Cbar = (beta^2 c + tau gamma^2) I``.
    """
    alpha_d = -1.0
    beta = t
    beta_d = 1.0
    gamma = g0 * (1.0 - t)
    gamma_d = -g0
    mbar = x0  # alpha x0 + beta m1 = (1-t)x0 + t x0 = x0
    cbar = beta**2 * prior_var + t * gamma**2
    coef = (beta * beta_d * prior_var + t * gamma * gamma_d) / cbar
    return alpha_d * x0 + beta_d * x0 + coef * (x - mbar)


def _si_prior_score(x: Tensor, x0: Tensor, t: float, g0: float, prior_var: float) -> Tensor:
    """SI prior score ``grad_x log p_tau(x_tau | x0) = -Cbar^{-1}(x - mbar)``."""
    beta = t
    gamma = g0 * (1.0 - t)
    cbar = beta**2 * prior_var + t * gamma**2
    return -(x - x0) / cbar


def _fm_prior_velocity(x: Tensor, x0: Tensor, t: float, prior_var: float) -> Tensor:
    """FM probability-flow velocity ``v(x, tau)`` for the Gaussian target.

    For ``x_tau = alpha z + beta x1``, ``v = E[xdot | x_tau]`` with
    ``xdot = alpha_d z + beta_d x1`` and ``(z, x1)`` jointly Gaussian given
    ``x_tau``. Closed form for the rectified path.
    """
    alpha = 1.0 - t
    alpha_d = -1.0
    beta = t
    beta_d = 1.0
    var_path = alpha**2 + beta**2 * prior_var
    mean_xtau = beta * x0
    # E[x1 | x_tau] and E[z | x_tau].
    slope_x1 = (beta * prior_var) / var_path
    mean_x1 = x0 + slope_x1 * (x - mean_xtau)
    slope_z = alpha / var_path
    mean_z = slope_z * (x - mean_xtau)
    return alpha_d * mean_z + beta_d * mean_x1


def _fm_prior_score(x: Tensor, x0: Tensor, t: float, prior_var: float) -> Tensor:
    """FM prior score ``grad_x log p_tau = -(x - beta x0) / var_path``."""
    alpha = 1.0 - t
    beta = t
    var_path = alpha**2 + beta**2 * prior_var
    return -(x - beta * x0) / var_path


# --------------------------------------------------------------------------- #
# Interpolant-likelihood guidance (the three config-selectable modes).
# --------------------------------------------------------------------------- #


def _likelihood_guidance(
    x: Tensor,
    x0: Tensor,
    y: Tensor,
    t: float,
    sys: GaussianSystem,
    *,
    model_class: str,
    g0: float,
    likelihood_mode: str,
) -> Tensor:
    """Interpolant-likelihood score ``Sbar`` (optionally gain-multiplied).

    With ``H = I``, ``R = obs_var I`` everything is scalar/isotropic. The
    interpolant likelihood in observation space is the Gaussian
    ``N(y; E[x1 | x_tau, x0], v1 + R)`` (convolving the source-conditional law of
    ``x1`` with the observation noise), so its score is

        Sbar = (y - mu_x1) / (v1 + R),
        mu_x1 = E[x1 | x_tau, x0],
        v1    = Var(x1 | x_tau, x0) = Sigma_s / beta^2.

    The three config-selectable modes differ only in which ``Sigma_s`` (hence
    which ``v1``) is used, and whether the multiplicative gain is applied:

      * ``inflated``          : exact source covariance Sigma_s -> exact v1; no gain
                                (PiGDM-style; exact for this Gaussian case).
      * ``dps_full``          : exact v1 AND the multiplicative gain
                                G_tau = 1 + Sigma_s/(beta^2 R) (faithful DPS+gain).
      * ``dps_jacobian_free`` : isotropic Sigma_s = sigma_tau^2 I -> v1 = sigma_tau^2/beta^2
                                (Corollary cheap_drift); no gain.

    ``mu_x1 = E[x1 | x_tau, x0]`` is the exact source-conditional mean in every
    mode (it is the prior's denoiser, not the inflation); the modes only change
    the likelihood covariance, which is the ablation the paper studies.
    """
    R = sys.obs_var
    beta = t
    alpha = 1.0 - t
    c = sys.prior_var

    # Source-conditional law of x1 (exact, Gaussian): mean ``mu_x1`` (the prior
    # denoiser) and variance ``v1_exact``. The interpolant likelihood in
    # observation space is N(ybar; H mu_bar, Sigma_bar), and Sbar = d/dx log p
    # carries the chain factor d(mu_bar)/dx = beta * d(mu_x1)/dx.
    if model_class == "si":
        gamma = g0 * (1.0 - t)
        sigma_tau_sq = gamma**2 * t                       # rho (isotropic source var)
        var_path = beta**2 * c + sigma_tau_sq             # Cbar scale
        mbar_x = x0                                        # E[x_tau | x0]
        ybar = alpha * x0 + beta * y                       # a0 = x0
    else:  # fm
        sigma_tau_sq = alpha**2
        var_path = alpha**2 + beta**2 * c
        mbar_x = beta * x0
        ybar = beta * y                                    # a0 = 0

    denoise_slope = beta * c / var_path                    # d(mu_x1)/dx
    mu_x1 = x0 + denoise_slope * (x - mbar_x)              # E[x1 | x_tau, x0]
    v1_exact = c - (beta * c) ** 2 / var_path             # Var(x1 | x_tau, x0)

    # Source covariance choice (the ablation): exact v1 for inflated / dps_full,
    # isotropic v1 = sigma_tau^2 / beta^2 for the Jacobian-free corollary.
    # ``inflated_shared`` (the ensemble-mean / shared-Jacobian variant) coincides
    # EXACTLY with ``inflated`` here: the source covariance v1_exact is
    # state-independent for this linear-Gaussian system, so sharing it across the
    # ensemble introduces no error (the shared/individual gap only appears for a
    # nonlinear prior, e.g. Navier--Stokes). It therefore also uses v1_exact.
    v1 = (
        v1_exact
        if likelihood_mode in ("inflated", "inflated_shared", "dps_full")
        else sigma_tau_sq / beta**2
    )

    # mu_bar = H E[ybar | x_tau, x0]; for SI = alpha x0 + beta mu_x1, FM = beta mu_x1.
    mu_bar = (alpha * x0 + beta * mu_x1) if model_class == "si" else beta * mu_x1
    sigma_bar = beta**2 * (v1 + R)                         # Var(ybar | x_tau)
    chain = beta * denoise_slope                          # d(mu_bar)/dx
    sbar = chain * (ybar - mu_bar) / sigma_bar            # d/dx log p(ybar | x_tau)

    if likelihood_mode == "dps_full":
        # Multiplicative gain G_tau = 1 + Sigma_s/(beta^2 R), Sigma_s = v1 beta^2.
        gain = 1.0 + (v1 * beta**2) / (beta**2 * R)
        sbar = gain * sbar
    return sbar


# --------------------------------------------------------------------------- #
# Guidance weight w_tau (Eq. posterior_drift; the velocity--score coefficient
# part). Closed form for the analytic Gaussian, verified against the exact
# posterior probability-flow velocity / Foellmer drift.
# --------------------------------------------------------------------------- #


def _guidance_weight_ode(t: float, model_class: str, g0: float, prior_var: float) -> float:
    """Probability-flow guidance weight pairing with ``Sbar = grad log p(y|x_tau)``.

    Derived by matching ``b_prior + w Sbar`` to the exact posterior velocity /
    SI Foellmer drift (target ``N(m_post, v_post I)``); since
    ``Sbar`` already carries the full ``1/(v1 + R)`` likelihood scaling, the
    weight is the source-to-state transfer factor and is bounded:

      * SI:  w = gamma_tau * g0 = g0^2 (1 - t)   (Foellmer SDE drift correction).
      * FM:  w = a_tau = alpha / beta            (velocity--score coefficient).
    """
    alpha = 1.0 - t
    beta = t
    if model_class == "si":
        return g0 * g0 * (1.0 - t)
    return alpha / beta


# --------------------------------------------------------------------------- #
# Diffusion-strength schedule g_tau for the SDE samplers.
# --------------------------------------------------------------------------- #


def _g_tau(t: float, model_class: str, g0: float) -> float:
    """Diffusion strength ``g_tau``.

    * SI-SDE: native ``g_tau = gamma_tau = g0 (1 - tau)``.
    * DM-SDE: endpoint-vanishing ``g_tau = g0 sqrt(alpha beta)`` (spec Alg. 3),
      keeping ``0.5 g^2 s`` finite as ``tau -> 1``.
    """
    if model_class == "si":
        return g0 * (1.0 - t)
    alpha = 1.0 - t
    beta = t
    return g0 * (alpha * beta) ** 0.5


# --------------------------------------------------------------------------- #
# The unified sampler loop.
# --------------------------------------------------------------------------- #


def sample_posterior(
    sys: GaussianSystem,
    x0: Tensor,
    y: Tensor,
    *,
    sampler: str,  # "si_sde" | "dm_sde" | "fm_ode"
    likelihood_mode: str = "inflated",
    ensemble_size: int = 64,
    num_steps: int = 50,
    g0: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Run one of the three samplers; return posterior ensemble ``[E, d]``.

    Shared Euler(-Maruyama) loop (Algorithm 1). The guidance is started at
    ``tau = dtau`` (not 0) to avoid the ``beta_0 = 0`` singularity.
    """
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    d = sys.d
    dt = 1.0 / num_steps
    x0b = x0.unsqueeze(0).expand(ensemble_size, d).contiguous()

    if sampler == "si_sde":
        model_class, use_noise = "si", True
        x = x0b.clone()  # SI init: point mass at x0
    elif sampler == "dm_sde":
        model_class, use_noise = "fm", True
        x = torch.randn(ensemble_size, d, generator=generator)  # exact N(0,I) init
    elif sampler == "fm_ode":
        model_class, use_noise = "fm", False
        x = torch.randn(ensemble_size, d, generator=generator)
    else:
        raise ValueError(f"unknown sampler {sampler!r}")

    for i in range(num_steps):
        t = (i + 1) * dt if model_class == "fm" else i * dt
        t = min(max(t, 1e-4), 1.0 - 1e-4)

        g = _g_tau(t, model_class, g0) if use_noise else 0.0
        w_ode = _guidance_weight_ode(t, model_class, g0, sys.prior_var)

        # Prior drift b_prior and guidance weight w_tau. For SI the Foellmer SDE
        # drift already carries the prior score, so w = w_ode (native). For FM the
        # prior probability-flow velocity is converted to an SDE via 0.5 g^2 s, and
        # since grad log p_post - grad log p_prior = Sbar exactly, the SDE adds
        # 0.5 g^2 Sbar -> w = w_ode + 0.5 g^2.
        if model_class == "si":
            b_prior = _si_prior_drift(x, x0, t, g0, sys.prior_var)
            w_tau = w_ode
        else:
            v = _fm_prior_velocity(x, x0, t, sys.prior_var)
            score = _fm_prior_score(x, x0, t, sys.prior_var)
            b_prior = v + 0.5 * g**2 * score
            w_tau = w_ode + 0.5 * g**2

        sbar = _likelihood_guidance(
            x, x0, y, t, sys,
            model_class=model_class, g0=g0, likelihood_mode=likelihood_mode,
        )

        drift = b_prior + w_tau * sbar
        x = x + drift * dt
        if use_noise and g > 0.0:
            x = x + g * (dt**0.5) * torch.randn(ensemble_size, d, generator=generator)

    return x


# --------------------------------------------------------------------------- #
# Baselines (classical filters are exact here: the forward model is known).
# --------------------------------------------------------------------------- #


def enkf_posterior(
    sys: GaussianSystem,
    x0: Tensor,
    y: Tensor,
    *,
    ensemble_size: int,
    generator: torch.Generator,
) -> Tensor:
    """Stochastic ensemble Kalman filter (Evensen) on the analytic system.

    Forecast ensemble ``x_f ~ N(x0, I)`` (the known prior transition), perturbed
    observations ``y + e``, ``e ~ N(0, R)``, sample-covariance gain. With ``H = I``
    this is the textbook stochastic EnKF and is exact up to ensemble noise.
    """
    d = sys.d
    R = sys.obs_var
    x_f = x0.unsqueeze(0) + sys.prior_var**0.5 * torch.randn(
        ensemble_size, d, generator=generator
    )
    # Sample forecast covariance.
    xf_mean = x_f.mean(0, keepdim=True)
    P = (x_f - xf_mean).T @ (x_f - xf_mean) / (ensemble_size - 1)
    # H = I: gain K = P (P + R I)^{-1}.
    K = P @ torch.linalg.inv(P + R * torch.eye(d))
    y_pert = y.unsqueeze(0) + (R**0.5) * torch.randn(
        ensemble_size, d, generator=generator
    )
    innov = y_pert - x_f  # H x_f = x_f
    return x_f + innov @ K.T


def particle_filter_posterior(
    sys: GaussianSystem,
    x0: Tensor,
    y: Tensor,
    *,
    ensemble_size: int,
    generator: torch.Generator,
) -> Tensor:
    """Bootstrap particle filter (Carrassi) on the analytic system.

    Propose ``x ~ N(x0, I)`` (prior transition), weight by the Gaussian
    likelihood ``N(y; H x, R)``, and resample (systematic). Exact in the
    ``E -> inf`` limit; reported with the same ``E`` as the other methods.
    """
    d = sys.d
    R = sys.obs_var
    x = x0.unsqueeze(0) + sys.prior_var**0.5 * torch.randn(
        ensemble_size, d, generator=generator
    )
    log_w = -0.5 * ((y.unsqueeze(0) - x) ** 2).sum(1) / R  # H = I
    w = torch.softmax(log_w, dim=0)
    # Systematic resampling.
    positions = (
        torch.arange(ensemble_size) + torch.rand(1, generator=generator)
    ) / ensemble_size
    cumsum = torch.cumsum(w, dim=0)
    idx = torch.searchsorted(cumsum, positions.clamp(max=1.0 - 1e-7))
    return x[idx.clamp(max=ensemble_size - 1)]


def flowdas_posterior(
    sys: GaussianSystem,
    x0: Tensor,
    y: Tensor,
    *,
    ensemble_size: int,
    num_steps: int,
    g0: float,
    n_mc: int,
    generator: torch.Generator,
) -> Tensor:
    """FlowDAS baseline (chen_flowdas_2025, Algorithm 2 line 10).

    Same analytical SI-SDE prior as our samplers (linear ``beta = t``); only the
    guidance differs. Per step, draw ``J = n_mc`` one-step predictions of ``x1``
    from ``X_s`` as ``X1_hat^(j) = mu_x1 + sqrt(v1) eps_j`` (the ``eps_j`` are
    DETACHED constants, so the ``X_s``-dependence is only through ``mu_x1``);
    compute DETACHED importance weights ``w_j = softmax_j(-||y - X1_hat^(j)||^2 /
    (2 sigma^2))`` and take the gradient of the weighted observation error THROUGH
    the predictor

        g = -grad_{X_s} sum_j w_j ||y - X1_hat^(j)||^2 .

    For the affine denoiser ``mu_x1`` with ``d mu_x1 / d X_s = denoise_slope``
    (scalar) and weights detached, this is the CLOSED FORM

        grad_{X_s} sum_j w_j ||y - X1_hat^(j)||^2
            = 2 denoise_slope (xhat1 - y),   xhat1 = sum_j w_j X1_hat^(j),

    so ``g = 2 denoise_slope (y - xhat1)``. Crucially the residual ``(y - xhat1)``
    is pulled THROUGH the denoiser Jacobian and points TOWARD the observation; it
    does NOT vanish as ``v1 -> 0`` (the old ``(xhat1 - mu_x1)/(v1+R)`` surrogate
    did). Step-normalize per member (DPS-style: the step SIZE was the original
    divergence, not the algorithm) and apply with the SI guidance weight ``w_tau``.
    ``J = n_mc`` is reported.
    """
    d = sys.d
    R = sys.obs_var
    c = sys.prior_var
    dt = 1.0 / num_steps
    x = x0.unsqueeze(0).expand(ensemble_size, d).contiguous().clone()
    for i in range(num_steps):
        t = min(max(i * dt, 1e-4), 1.0 - 1e-4)
        beta = t
        gamma = g0 * (1.0 - t)
        var_path = beta**2 * c + gamma**2 * t
        denoise_slope = beta * c / var_path  # d mu_x1 / d X_s
        mu_x1 = x0 + denoise_slope * (x - x0)
        v1 = c - (beta * c) ** 2 / var_path
        b_prior = _si_prior_drift(x, x0, t, g0, c)
        # J one-step predictions of x1 with DETACHED source noise; softmax weights.
        eps = torch.randn(n_mc, ensemble_size, d, generator=generator)
        preds = mu_x1.unsqueeze(0) + max(v1, 0.0) ** 0.5 * eps  # [J, E, d]
        log_w = -0.5 * ((y - preds) ** 2).sum(-1) / R  # [J, E]
        w = torch.softmax(log_w, dim=0)  # [J, E]
        x1_hat = (w.unsqueeze(-1) * preds).sum(0)  # weighted mean prediction
        # FlowDAS guidance: -grad of the weighted obs error through the predictor,
        # = denoise_slope (y - xhat1) (H = I); step-normalize per member.
        grad = denoise_slope * (x1_hat - y)  # propto grad_{X_s} (weighted err)
        gnorm = grad.norm(dim=1, keepdim=True)
        guidance = -grad / (gnorm + 1e-6)
        w_tau = _guidance_weight_ode(t, "si", g0, c)
        x = x + (b_prior + w_tau * guidance) * dt
        x = x + gamma * (dt**0.5) * torch.randn(ensemble_size, d, generator=generator)
    return x


# --------------------------------------------------------------------------- #
# New posterior-sampling baselines (closed-form / autograd on the Gaussian).
#
# Each mirrors its NS reference implementation, specialised to H = I, R = 1:
#   * Guided FM (FIG)      -> guidance.py FIGGaussianLikelihood
#   * Guided FM (OT-ODE)   -> guidance.py GuidanceGaussianLikelihood._score_ot_ode
#   * D-Flow SGLD          -> dflow_posterior.py DFlowPosterior
#   * SDA                  -> sda.py SDALikelihood (+ DiffusionPosterior weighting)
#   * SURGE                -> surge_posterior.py SurgePosterior
# --------------------------------------------------------------------------- #


def guided_fm_fig_posterior(
    sys: GaussianSystem,
    x0: Tensor,
    y: Tensor,
    *,
    ensemble_size: int,
    num_steps: int,
    guidance_steps: int = 1,
    guidance_scale: float = 80.0,
    interpolant_noise: float = 0.0,
    generator: torch.Generator,
) -> Tensor:
    """FIG measurement-interpolant guided FM-ODE (yan_fig_2025).

    FM-ODE prior (source ``N(0, I)``). After each prior Euler step
    ``x_next = x + v dt`` run ``k`` corrector iterations pulling toward the
    measurement interpolant ``y_t = t_next y`` with step ``c (1 - t) / t``,
    differentiating the residual NORM ``||y_t - x_next||`` (H = I): the gradient
    is ``-(y_t - x_next) / ||y_t - x_next||`` so the update is
    ``x_next += step (y_t - x_next) / ||y_t - x_next||``.

    Stability (FIX 1): (a) the unit-direction move magnitude is capped at the
    residual norm, ``effective_step = min(c (1 - t)/t, ||y_t - x_next||)``, so a
    single inner step lands at most ON ``y_t`` and never overshoots/collapses the
    ensemble (the M-independent fix for high-M variance collapse); (b) the
    corrector is skipped on the first and last integration steps (the official
    FIG ``1 <= i < N - 1`` guard).
    """
    d = sys.d
    c = sys.prior_var
    dt = 1.0 / num_steps
    x = torch.randn(ensemble_size, d, generator=generator)
    for i in range(num_steps):
        t = (i + 1) * dt
        t = min(max(t, 1e-4), 1.0 - 1e-4)
        t_next = min(t + dt, 1.0)
        v = _fm_prior_velocity(x, x0, t, c)
        x_next = x + v * dt
        # Official corrector guard: skip the first and last integration steps.
        if t <= dt + 1e-9 or t >= 1.0 - dt - 1e-9:
            x = x_next
            continue
        step = guidance_scale * (1.0 - t) / t
        y_t = t_next * y.unsqueeze(0)
        if interpolant_noise != 0.0:
            y_t = y_t + interpolant_noise * (1.0 - t) * torch.randn(
                ensemble_size, d, generator=generator
            )
        for _ in range(max(guidance_steps, 1)):
            residual = y_t - x_next  # H = I
            norm = residual.norm(dim=1, keepdim=True)
            # Cap the move magnitude at the residual norm so a single inner step
            # lands at most ON y_t (never past it -> no overshoot/collapse).
            eff_step = torch.clamp(norm, max=step)
            # grad_{x} ||y_t - x|| = -(y_t - x)/||y_t - x||; descent on the NORM.
            x_next = x_next + eff_step * residual / (norm + 1e-12)
        x = x_next
    return x


def guided_fm_otode_posterior(
    sys: GaussianSystem,
    x0: Tensor,
    y: Tensor,
    *,
    ensemble_size: int,
    num_steps: int,
    obs_variance: float = 0.0,
    guidance_scale: float = 4.0,
    generator: torch.Generator,
) -> Tensor:
    """OT-ODE covariance-preconditioned guided FM-ODE (pokle_training-free_2024).

    FM-ODE prior. One-step denoiser ``xhat1 = x + (1 - t) v``; posterior-variance
    estimate ``r_t^2 = (1-t)^2 / ((1-t)^2 + t^2)`` floored at 1e-2; preconditioned
    guidance ``g = J (y - xhat1) / (r_t^2 + sigma_y^2)`` where ``J = d xhat1/dx``
    (a scalar for the affine FM denoiser, derived in closed form). The posterior
    velocity gains ``a_tau guidance_scale g`` with ``a_tau = (1 - t)/t`` (the
    score->velocity coefficient applied by ``FlowMatchingPosterior``).
    """
    R2_FLOOR = 1e-2
    d = sys.d
    c = sys.prior_var
    dt = 1.0 / num_steps
    x = torch.randn(ensemble_size, d, generator=generator)
    for i in range(num_steps):
        t = (i + 1) * dt
        t = min(max(t, 1e-4), 1.0 - 1e-4)
        alpha = 1.0 - t
        beta = t
        v = _fm_prior_velocity(x, x0, t, c)
        xhat1 = x + (1.0 - t) * v  # E[x1 | x_t]
        # J = d xhat1 / dx (scalar for the affine FM denoiser).
        # xhat1 = x + (1-t) v, v = alpha_d mean_z + beta_d mean_x1; both means are
        # affine in x with slope d v/dx = (-alpha + beta c)/var_path, so
        # J = 1 + (1-t) dv/dx.
        var_path = alpha**2 + beta**2 * c
        dvdx = (-alpha + beta * c) / var_path
        J = 1.0 + (1.0 - t) * dvdx
        r2 = max((1.0 - t) ** 2 / ((1.0 - t) ** 2 + t**2), R2_FLOOR)
        g = J * (y.unsqueeze(0) - xhat1) / (r2 + obs_variance)  # H = I
        a_tau = (1.0 - t) / t
        x = x + (v + a_tau * guidance_scale * g) * dt
    return x


def _fm_ode_flow(
    z0: Tensor, x0: Tensor, num_steps: int, prior_var: float
) -> Tensor:
    """Differentiable FM-ODE rollout ``x1 = Phi(z0)`` (forward Euler, g = 0)."""
    dt = 1.0 / num_steps
    x = z0
    for i in range(num_steps):
        t = (i + 1) * dt
        t = min(max(t, 1e-4), 1.0 - 1e-4)
        v = _fm_prior_velocity(x, x0, t, prior_var)
        x = x + v * dt
    return x


def dflow_sgld_posterior(
    sys: GaussianSystem,
    x0: Tensor,
    y: Tensor,
    *,
    ensemble_size: int,
    num_steps: int,
    num_optim_steps: int = 20,
    step_size: float = 0.05,
    noise_scale: float = 1.0,
    guidance_scale: float = 1.0,
    precond_decay: float = 0.99,
    precond_eps: float = 1e-8,
    generator: torch.Generator,
) -> Tensor:
    """D-Flow SGLD (ben-hamu_d-flow_2024): pSGLD over the FM source latent z0.

    Optimise/sample ``z0`` so the differentiable FM-ODE flow ``Phi(z0) = x1``
    matches ``y``, targeting ``p(z0 | y) ~ N(y; Phi(z0), R) N(z0; 0, I)`` with a
    preconditioned-SGLD (RMSProp diagonal) update, then return ``Phi(z0)``. For
    this affine-Gaussian flow D-Flow SGLD targets the EXACT source posterior.
    """
    d = sys.d
    R = sys.obs_var
    c = sys.prior_var
    eta = step_size
    rho = precond_decay
    eps = precond_eps
    z = torch.randn(ensemble_size, d, generator=generator)
    V = torch.zeros_like(z)
    yb = y.unsqueeze(0)
    for k in range(1, max(num_optim_steps, 0) + 1):
        z_g = z.detach().requires_grad_(True)
        with torch.enable_grad():
            x1 = _fm_ode_flow(z_g, x0, num_steps, c)
            residual = yb - x1  # H = I
            data_term = 0.5 * (residual**2).sum(dim=1) / R
            prior_term = 0.5 * (z_g**2).sum(dim=1)
            loss = guidance_scale * data_term + prior_term
            grad = torch.autograd.grad(loss.sum(), z_g)[0]
        V = rho * V + (1.0 - rho) * grad * grad
        # FIX 2: Adam-style bias correction on the 2nd moment kills the RMSProp
        # cold-start P explosion (V ~ (1-rho) g^2 at k=1 -> P ~ 1/(|g| sqrt(1-rho))
        # ~ 10x too large). V_hat = V / (1 - rho^k) restores the published P form.
        V_hat = V / (1.0 - rho**k)
        P = 1.0 / (eps + V_hat.sqrt())
        if noise_scale != 0.0:
            noise = (
                noise_scale
                * (2.0 * eta) ** 0.5
                * P.sqrt()
                * torch.randn(ensemble_size, d, generator=generator)
            )
        else:
            noise = torch.zeros_like(z)
        z = (z_g - eta * P * grad).detach() + noise
    with torch.no_grad():
        x1 = _fm_ode_flow(z, x0, num_steps, c)
    return x1


def sda_posterior(
    sys: GaussianSystem,
    x0: Tensor,
    y: Tensor,
    *,
    ensemble_size: int,
    num_steps: int,
    g0: float,
    generator: torch.Generator,
) -> Tensor:
    """SDA single-window guidance on the diffusion-model (FM) prior (rozet_2023).

    DM prior = the FM prior reused: score ``_fm_prior_score``, velocity
    ``_fm_prior_velocity``, reverse-SDE drift ``v + 1/2 g^2 s`` with
    endpoint-vanishing ``g = g0 sqrt(alpha beta)``, source ``N(0, I)``. Guidance:
    Tweedie denoiser ``xhat1`` and ``Gamma_tau = (sigma^2 + rho) I`` with
    ``rho = alpha^2`` (the FM source variance).

    FIX 3: drop the ``1/||Gamma^-1 r||`` step-normalisation (it is exactly the DPS
    rescaling SDA warns against, and the old ``0.5 g^2 sqrt(t)`` weight then
    under-powered the guidance ~10-16x). Use the RAW likelihood score
    ``s_lik = J Gamma^-1 (y - xhat1)`` (J = denoise_slope) and apply it with the
    FM score->state coefficient ``a_tau + 0.5 g^2`` (``a_tau = alpha/beta``),
    dropping the ``sqrt(t)`` factor.
    """
    d = sys.d
    R = sys.obs_var
    c = sys.prior_var
    dt = 1.0 / num_steps
    x = torch.randn(ensemble_size, d, generator=generator)
    yb = y.unsqueeze(0)
    for i in range(num_steps):
        t = (i + 1) * dt
        t = min(max(t, 1e-4), 1.0 - 1e-4)
        alpha = 1.0 - t
        beta = t
        v = _fm_prior_velocity(x, x0, t, c)
        s = _fm_prior_score(x, x0, t, c)
        g = g0 * (alpha * beta) ** 0.5
        drift = v + 0.5 * g**2 * s

        # Tweedie denoiser xhat1 = E[x1 | x_t] = (x + sigma^2 s) / beta, a0 = 0.
        sigma_tau_sq = alpha**2
        xhat1 = (x + sigma_tau_sq * s) / beta
        # J = d xhat1 / dx for the affine FM denoiser.
        var_path = alpha**2 + beta**2 * c
        denoise_slope = beta * c / var_path  # d E[x1|x_t]/dx (exact)
        J = denoise_slope
        rho = alpha**2  # FM source variance
        gamma_scale = R + rho  # Gamma = (sigma^2 + rho) I  (H H^T = I)
        # Raw likelihood score (NO step-normalisation): s_lik = J Gamma^-1 r.
        s_lik = J * (yb - xhat1) / gamma_scale
        a_tau = alpha / beta

        x = x + drift * dt
        x = x + g * torch.randn(ensemble_size, d, generator=generator) * dt**0.5
        x = x + (a_tau + 0.5 * g**2) * s_lik * dt
    return x


def surge_posterior(
    sys: GaussianSystem,
    x0: Tensor,
    y: Tensor,
    *,
    ensemble_size: int,
    num_steps: int,
    g0: float,
    guidance_scale: float = 1.0,
    ess_threshold: float = 0.5,
    generator: torch.Generator,
) -> Tensor:
    """SURGE guided-SDE + Girsanov-corrected SMC particle filter (wei_surge).

    DM prior reverse-SDE proposal (``v + 1/2 g^2 s``, ``g = g0 sqrt(alpha beta)``,
    source ``N(0, I)``) with step-normalised DPS guidance. Reward
    ``R_t = -1/2 ||y - xhat1||^2 / sigma^2``. Girsanov SMC log-weight increments
    (reward delta + martingale ``- Sigma^{1/2} grad_G . sqrt(dt) xi`` and
    quadratic-variation ``- 1/2 Sigma ||grad_G||^2 dt``); ESS-resample across the
    ensemble when ESS/N < threshold, and a final resample.
    """
    d = sys.d
    R = sys.obs_var
    c = sys.prior_var
    dt = 1.0 / num_steps
    E = ensemble_size
    x = torch.randn(E, d, generator=generator)
    yb = y.unsqueeze(0)
    log_w = torch.zeros(E)

    def denoise_and_reward(xx: Tensor, tt: float) -> tuple[Tensor, Tensor, Tensor]:
        """Return (xhat1, reward, grad_G_normalised)."""
        alpha = 1.0 - tt
        beta = tt
        s = _fm_prior_score(xx, x0, tt, c)
        xhat1 = (xx + alpha**2 * s) / beta  # a0 = 0
        residual = yb - xhat1
        reward = -0.5 / R * (residual**2).sum(dim=1)  # [E]
        # grad_x reward = J^T H^T (y - xhat1)/sigma^2 (H = I); J = denoise slope.
        var_path = alpha**2 + beta**2 * c
        J = beta * c / var_path
        grad = J * residual / R  # [E, d]
        gnorm = grad.norm(dim=1, keepdim=True)
        grad_tilde = guidance_scale * grad / (gnorm + 1e-6)
        return xhat1, reward, grad_tilde

    def maybe_resample(
        xx: Tensor, lw: Tensor, force: bool = False
    ) -> tuple[Tensor, Tensor]:
        lw = torch.nan_to_num(lw, nan=-1e30, neginf=-1e30, posinf=1e30)
        w = torch.softmax(lw - lw.max(), dim=0)
        ess = 1.0 / torch.clamp((w**2).sum(), min=1e-30)
        if not force and float(ess) >= ess_threshold * E:
            return xx, lw
        u0 = torch.rand(1, generator=generator) / E
        positions = u0 + torch.arange(E) / E
        cumsum = torch.cumsum(w, dim=0)
        cumsum[-1] = 1.0
        idx = torch.searchsorted(cumsum, positions).clamp(max=E - 1)
        return xx[idx].clone(), torch.zeros(E)

    for i in range(num_steps):
        t = (i + 1) * dt
        t = min(max(t, 1e-4), 1.0 - 1e-4)
        t_next = min(t + dt, 1.0)
        alpha = 1.0 - t
        beta = t
        v = _fm_prior_velocity(x, x0, t, c)
        s = _fm_prior_score(x, x0, t, c)
        g = g0 * (alpha * beta) ** 0.5
        drift = v + 0.5 * g**2 * s
        sigma2 = g**2

        _, reward_k, grad_G = denoise_and_reward(x, t)

        noise = torch.randn(E, d, generator=generator)
        diffusion_incr = g * noise * dt**0.5
        x_next = x + (drift + sigma2 * grad_G) * dt + diffusion_incr

        # Reward at x_{k+1} for the reward-delta term.
        _, reward_next, _ = denoise_and_reward(x_next, t_next)
        reward_delta = t_next * reward_next - t * reward_k
        flat_grad = g * grad_G
        flat_dW = noise * dt**0.5
        martingale = -(flat_grad * flat_dW).sum(dim=1)
        quad_var = -0.5 * sigma2 * (grad_G**2).sum(dim=1) * dt
        log_incr = reward_delta + martingale + quad_var
        log_incr = torch.nan_to_num(log_incr, nan=-1e30, neginf=-1e30, posinf=1e30)
        log_w = log_w + log_incr

        x = x_next
        x, log_w = maybe_resample(x, log_w)

    x, _ = maybe_resample(x, log_w, force=True)
    return x


__all__ = [
    "GaussianSystem",
    "sample_posterior",
    "enkf_posterior",
    "particle_filter_posterior",
    "flowdas_posterior",
    "guided_fm_fig_posterior",
    "guided_fm_otode_posterior",
    "dflow_sgld_posterior",
    "sda_posterior",
    "surge_posterior",
]
