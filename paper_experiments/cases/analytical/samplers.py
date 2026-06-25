"""Closed-form posterior samplers for the analytical linear--Gaussian case.

Spec Section 4. The prior transition is the analytic Gaussian
``x1 | x0 ~ N(x0, I)`` and the observation model is ``y = H x1 + e``,
``e ~ N(0, R)`` with ``H = I``, ``R = I``; the exact posterior is
``N(x0 + K(y - H x0), I - K H)`` with ``K = 0.5 I``.

Because no network is trained here, every quantity the unified posterior drift
needs (the prior velocity/score, the source conditional moments, the
interpolant-likelihood score and the multiplicative gain) is available in closed
form. We implement the three samplers of the paper -- ``SI-SDE``, ``FM-SDE`` and
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
    v1 = v1_exact if likelihood_mode in ("inflated", "dps_full") else sigma_tau_sq / beta**2

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
    * FM-SDE: endpoint-vanishing ``g_tau = g0 sqrt(alpha beta)`` (spec Alg. 3),
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
    sampler: str,  # "si_sde" | "fm_sde" | "fm_ode"
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
    elif sampler == "fm_sde":
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
    """FlowDAS baseline (chen_flowdas_2025): SI-SDE prior with a Monte-Carlo
    likelihood instead of the observation-interpolant score.

    At each step, draw ``n_mc`` one-step predictions of ``x1`` (the analytic
    denoiser mean plus its conditional spread), softmax-weight by ``N(y; H x1, R)``
    and use the weighted score as the guidance. ``J = n_mc`` is reported.
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
        mu_x1 = x0 + (beta * c / var_path) * (x - x0)
        v1 = c - (beta * c) ** 2 / var_path
        b_prior = _si_prior_drift(x, x0, t, g0, c)
        # Monte-Carlo predictions of x1 and softmax weights.
        eps = torch.randn(n_mc, ensemble_size, d, generator=generator)
        preds = mu_x1.unsqueeze(0) + max(v1, 0.0) ** 0.5 * eps
        log_w = -0.5 * ((y - preds) ** 2).sum(-1) / R  # [n_mc, E]
        w = torch.softmax(log_w, dim=0)  # [n_mc, E]
        x1_hat = (w.unsqueeze(-1) * preds).sum(0)  # weighted mean prediction
        guidance = (x1_hat - mu_x1) / (v1 + R)  # MC likelihood pull
        w_tau = _guidance_weight_ode(t, "si", g0, c)
        x = x + (b_prior + w_tau * guidance) * dt
        x = x + gamma * (dt**0.5) * torch.randn(ensemble_size, d, generator=generator)
    return x


__all__ = [
    "GaussianSystem",
    "sample_posterior",
    "enkf_posterior",
    "particle_filter_posterior",
    "flowdas_posterior",
]
