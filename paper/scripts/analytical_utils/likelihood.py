from typing import Callable, Optional

import torch
import torch.nn as nn
import tqdm
from torch.func import jacrev, vmap


class InterpolantLikelihood(nn.Module):
    """Interpolant likelihood.

    Computes the score of the interpolated observation likelihood,
        nabla_x log p(ybar_tau | x_tau, x_0),
    using the mean correction d_tau and (optionally) the covariance
    correction C_tau from Lemma 4.2 and Lemma 4.3.
    """

    def __init__(
        self,
        obs_matrix: torch.Tensor,
        drift_model: nn.Module,
        original_variance: float,
        ensemble_size: int = 100,
        use_covariance_correction: bool = True,
        perturbation: Optional[str] = None,
        target_variance: float = 1.0,
        num_quad: int = 256,
    ) -> None:
        """Initialize interpolant likelihood.

        Args:
            obs_matrix: Observation matrix H, shape (N_y, N_u).
            drift_model: Trained (or analytical) drift model.
            original_variance: Scalar observation noise variance (R = original_variance * I).
            ensemble_size: Not used currently, kept for interface compatibility.
            use_covariance_correction: If True, include the C_tau correction
                from the conditional covariance of W_tau.
            perturbation: Which likelihood-score variant to use. One of:
                ``None``                -- raw interpolant score (no gain),
                ``"new_correction"``    -- Corollary 4.5 (Sigma_W ~= tau I),
                ``"expensive"``         -- Theorem 4.4 with Sigma_W computed
                                           via autograd Jacobian J_b. For H = I
                                           and R = sigma^2 I the gain cancels
                                           Sigma_bar^{-1} algebraically and
                                           "expensive" reduces to "new_correction"
                                           in that setting -- a property of
                                           Theorem 4.4, not a bug.
                ``"true"``              -- *Exact* analytical posterior
                                           likelihood score using the SDE's own
                                           transition expectation
                                                Phi^y_tau = E_SDE[p(y|X_1) | X_tau, X_0],
                                           rather than the interpolant joint.
                                           The SDE matches I_tau marginals but
                                           not joints, so Theorem 4.1 requires
                                           Phi to come from the SDE; closed
                                           form via :meth:`_sde_phi_and_vcond`.
            target_variance: Scalar prior variance c (used by ``"true"``).
            num_quad: Trapezoidal grid points for ``int_tau^1 K(s) ds`` in
                ``"true"``.
        """
        super(InterpolantLikelihood, self).__init__()
        self.drift_model = drift_model
        self.obs_matrix = obs_matrix
        self.original_variance = original_variance
        self.ensemble_size = ensemble_size
        self.use_covariance_correction = use_covariance_correction
        if perturbation not in (
            None,
            "new_correction",
            "true",
            "expensive",
        ):
            raise ValueError(
                f"perturbation must be None, 'new_correction', 'true', or "
                f"'expensive', got {perturbation!r}"
            )
        self.perturbation = perturbation
        self.target_variance = target_variance
        self.num_quad = num_quad

    def forward(self) -> None:
        """Forward pass."""
        pass

    def _compute_W_covariance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the  Wcovariance

        Returns:
            W_cov: shape (batch, N_y, N_y).
        """
        gamma = self.drift_model.interpolation.gamma(t)[0, 0].item()
        beta = self.drift_model.interpolation.beta(t)[0, 0].item()
        beta_diff = self.drift_model.interpolation.beta_diff(t)[0, 0].item()
        gamma_diff = self.drift_model.interpolation.gamma_diff(t)[0, 0].item()

        A_tau = (
            1.0
            / (t[0, 0] * gamma * (beta_diff * gamma - beta * gamma_diff) + 1e-8).item()
        )

        # --- Compute J_b via autograd on a detached copy of x ---

        x = x.unsqueeze(0)
        x0 = x0.unsqueeze(0)

        J_b = jacrev(
            lambda x: self.drift_model._compute_drift(x, t[0:1], x0).sum(dim=0),
        )(x)
        J_b = J_b.transpose(0, 1).squeeze(0)

        I = torch.eye(J_b.shape[-1], device=J_b.device, dtype=J_b.dtype)
        return t[0, 0] * I + gamma**2 * t[0, 0] ** 2 * A_tau * (
            beta * J_b - beta_diff * I
        )

    def mu_fn(self, x, x0, t, observations):
        """Compute the mean."""
        gamma = self.drift_model.interpolation.gamma(t)[0, 0]
        beta = self.drift_model.interpolation.beta(t)[0, 0]
        alpha = self.drift_model.interpolation.alpha(t)[0, 0]

        i_obs = alpha * torch.matmul(self.obs_matrix, x0) + beta * observations

        x_obs = torch.matmul(self.obs_matrix, x)

        drift = self.drift_model._compute_drift(x.unsqueeze(0), t, x0.unsqueeze(0))
        model_score = self.drift_model._compute_score_from_drift(
            x.unsqueeze(0), t[0:1], x0.unsqueeze(0), drift
        ).squeeze(0)

        E_W = -gamma * t[0, 0] * model_score  # E[W_tau | x_tau, x_0]
        d = -gamma * torch.matmul(self.obs_matrix, E_W)

        return i_obs - x_obs - d

    def sigma_fn(self, x, x0, t):
        """Compute the covariance."""

        gamma = self.drift_model.interpolation.gamma(t)[0, 0].item()
        beta = self.drift_model.interpolation.beta(t)[0, 0].item()

        R = beta**2 * self.original_variance

        cov_W = self._compute_W_covariance(x, t, x0)

        cov_W = self.obs_matrix @ cov_W @ self.obs_matrix.T
        cov_W = gamma**2 * cov_W

        sigma = R * torch.eye(self.obs_matrix.shape[0], device=x.device) + cov_W

        return sigma

    def _W_cov_analytical_scalar(self, t: torch.Tensor) -> float:
        """Closed-form scalar c such that Sigma_W(tau) = c * I.

        Valid in the linear analytical Gaussian setting where the drift
        Jacobian satisfies J_b = a(tau) I, with
            a(tau) = (beta beta_diff c + tau gamma gamma_diff) / V_tau,
            V_tau  = beta^2 c + tau gamma^2,
        so that
            Sigma_W = tau I + gamma^2 tau^2 A (beta a - beta_diff) I,
            A       = 1 / (tau gamma (beta_diff gamma - beta gamma_diff)).
        """
        interp = self.drift_model.interpolation
        beta = interp.beta(t)[0, 0].item()
        beta_diff = interp.beta_diff(t)[0, 0].item()
        gamma = interp.gamma(t)[0, 0].item()
        gamma_diff = interp.gamma_diff(t)[0, 0].item()
        tau = t[0, 0].item()
        c = self.target_variance

        V_tau = beta**2 * c + tau * gamma**2
        a_tau = (beta * beta_diff * c + tau * gamma * gamma_diff) / V_tau
        A_code = 1.0 / (tau * gamma * (beta_diff * gamma - beta * gamma_diff) + 1e-12)
        return tau + gamma**2 * tau**2 * A_code * (beta * a_tau - beta_diff)

    def sigma_fn_analytical(self, t: torch.Tensor) -> torch.Tensor:
        """Closed-form Sigma_bar(tau) for the analytical Gaussian case.

        Sigma_bar = beta^2 R + gamma^2 H Sigma_W H^T (Lemma 4.2), with
        Sigma_W = scalar * I from ``_W_cov_analytical_scalar``.
        """
        beta = self.drift_model.interpolation.beta(t)[0, 0].item()
        gamma = self.drift_model.interpolation.gamma(t)[0, 0].item()
        H = self.obs_matrix
        R = beta**2 * self.original_variance
        sigma_W_scalar = self._W_cov_analytical_scalar(t)
        cov_W = gamma**2 * sigma_W_scalar * (H @ H.T)
        return R * torch.eye(H.shape[0], device=H.device, dtype=H.dtype) + cov_W

    def gain_analytical(self, t: torch.Tensor) -> torch.Tensor:
        """Closed-form gain G(tau) = I + (gamma^2/beta^2) Sigma_W H^T R^{-1} H.

        Uses the scalar Sigma_W from the analytical Gaussian setting.
        Returns an (N_u, N_u) tensor independent of the batch.
        """
        beta = self.drift_model.interpolation.beta(t)[0, 0].item()
        gamma = self.drift_model.interpolation.gamma(t)[0, 0].item()
        sigma_W_scalar = self._W_cov_analytical_scalar(t)
        H = self.obs_matrix
        n = H.shape[1]
        I_n = torch.eye(n, device=H.device, dtype=H.dtype)
        HtRinvH = (H.T @ H) / self.original_variance
        return I_n + (gamma**2 / beta**2) * sigma_W_scalar * HtRinvH

    def gain_from_W_cov(
        self, sigma_W: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Gain G(tau) = I + (gamma^2/beta^2) Sigma_W H^T R^{-1} H (Theorem 4.4).

        Args:
            sigma_W: Conditional Wiener covariance, either (N_u, N_u) or
                (B, N_u, N_u).
            t: Pseudo-time tensor.
        """
        beta = self.drift_model.interpolation.beta(t)[0, 0].item()
        gamma = self.drift_model.interpolation.gamma(t)[0, 0].item()
        H = self.obs_matrix
        n = H.shape[1]
        I_n = torch.eye(n, device=H.device, dtype=H.dtype)
        HtRinvH = (H.T @ H) / self.original_variance
        return I_n + (gamma**2 / beta**2) * (sigma_W @ HtRinvH)

    def _sde_phi_and_vcond(self, tau: float) -> tuple[float, float]:
        """SDE transition coefficients for the linear analytical setting.

        Computes ``Phi(1, tau) = exp(int_tau^1 K(s) ds)`` and the conditional
        variance ``V_cond^SDE(tau) = c - Phi^2 V_tau`` for the linear SDE
        ``dX = b(X, x_0, tau) dtau + gamma_tau dW``, with drift Jacobian
            K(s) = (beta beta_diff c + s gamma gamma_diff) / V_s,
            V_s  = beta_s^2 c + s gamma_s^2.

        These determine the SDE's own conditional moments
            E_SDE[X_1 | X_tau, X_0] = m_0 + Phi(1, tau) (X_tau - alpha x_0 - beta m_0),
            Var_SDE(X_1 | X_tau, X_0) = V_cond^SDE I,
        which differ in general from the *interpolant* joint conditional
        because the analytical SDE matches only the marginals of I_tau, not
        the joint of (I_tau, I_1). Theorem 4.1 requires the SDE's own joint.
        """
        interp = self.drift_model.interpolation
        c = self.target_variance

        s = torch.linspace(tau, 1.0, self.num_quad).view(-1, 1)
        beta_s = interp.beta(s).view(-1)
        beta_diff = interp.beta_diff(s).view(-1)
        gamma_s = interp.gamma(s).view(-1)
        gamma_diff = interp.gamma_diff(s).view(-1)
        s_flat = s.view(-1)

        K = (beta_s * beta_diff * c + s_flat * gamma_s * gamma_diff) / (
            beta_s**2 * c + s_flat * gamma_s**2 + 1e-12
        )
        log_phi = torch.trapz(K, s_flat)
        phi = torch.exp(log_phi).item()

        t_tensor = torch.tensor([[tau]])
        beta_t = interp.beta(t_tensor)[0, 0].item()
        gamma_t = interp.gamma(t_tensor)[0, 0].item()
        V_tau = beta_t**2 * c + tau * gamma_t**2
        V_cond = max(c - phi**2 * V_tau, 0.0)
        return phi, V_cond

    def true_likelihood_score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Exact analytical score of the SDE's Phi^y_tau (Theorem 4.1).

        Theorem 4.1 requires Phi to be the *SDE's* transition expectation,
        ``Phi^y_tau(x_tau, x_0) = E_SDE[p(y | X_1) | X_tau = x_tau, X_0 = x_0]``,
        not the interpolant joint conditional. For the linear analytical SDE
        with isotropic Gaussian prior x_1 | x_0 ~ N(m_0, c I), this gives

            X_1 | X_tau, x_0 ~ N(mu_x1, V_cond^SDE I),
            mu_x1     = m_0 + Phi(1, tau) (x_tau - alpha x_0 - beta m_0),
            V_cond^SDE = c - Phi(1, tau)^2 V_tau,

        with ``Phi(1, tau) = exp(int_tau^1 K(s) ds)`` (computed by quadrature
        in :meth:`_sde_phi_and_vcond`). Convolving with the observation
        likelihood gives a Gaussian Phi:

            Phi^y_tau = N(y; H mu_x1, R + V_cond^SDE H H^T),

        and the closed-form score is

            S = Phi(1, tau) H^T (R + V_cond^SDE H H^T)^{-1} (y - H mu_x1).

        At tau = 1 we have Phi(1, 1) = 1 and V_cond^SDE = 0, recovering the
        raw observation score H^T R^{-1} (y - H x_1). Note this differs from
        the interpolant-joint score: the SDE has matching marginals but a
        different joint (X_tau, X_1) -- the empirical SDE slope of E[X_1 |
        X_tau] is Phi(1, tau), not the interpolant's beta c / V_tau.
        """
        interp = self.drift_model.interpolation
        alpha = interp.alpha(t)[0, 0].item()
        beta = interp.beta(t)[0, 0].item()
        tau = t[0, 0].item()

        phi, V_cond = self._sde_phi_and_vcond(tau)

        m0 = self.drift_model.target_mean(x0)  # (B, N_u)
        mu_x1 = m0 + phi * (x - alpha * x0 - beta * m0)

        H = self.obs_matrix
        R_eff = (
            self.original_variance
            * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
            + V_cond * (H @ H.T)
        )

        y = observations[0]  # (N_y,)
        mu_y = mu_x1 @ H.T  # (B, N_y)
        diff = y.unsqueeze(0) - mu_y  # (B, N_y)
        R_eff_inv_diff = torch.linalg.solve(
            R_eff, diff.unsqueeze(-1)
        ).squeeze(-1)  # (B, N_y)

        return phi * (R_eff_inv_diff @ H)  # (B, N_u)

    def new_correction(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        observations: torch.Tensor,
        dt: torch.Tensor,
        diffusion_term: Callable,
    ) -> torch.Tensor:
        """Corollary 4.5 (Jacobian-free) scalar gain.

        Under Sigma_W ~= tau I and H = I, R = sigma^2 I, the gain reduces to
            G(tau) = 1 + gamma^2 tau / (beta^2 sigma^2).
        """
        tau = t[0, 0].item()
        interp = self.drift_model.interpolation
        beta = interp.beta(t)[0, 0].item()
        gamma = interp.gamma(t)[0, 0].item()
        beta_safe = beta if abs(beta) > 1e-3 else 1e-3

        G = gamma * gamma * tau / (beta_safe * beta_safe * self.original_variance)
        return 1.0 + G

    def score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        observations: torch.Tensor,
        dt: torch.Tensor,
        diffusion_term: Callable,
    ) -> torch.Tensor:
        """Compute the likelihood score used in the SDE drift correction.

        For ``perturbation == "true"`` this returns the *exact* score of
        Phi^y_tau computed under the SDE's own transition kernel (Theorem
        4.1) -- bypassing the multiplicative-correction framework entirely.

        Otherwise (``None``, ``"new_correction"``, ``"expensive"``) it builds
        the Gaussian interpolant log-likelihood with mean mu_bar and
        covariance Sigma_bar (Lemma 4.2), differentiates w.r.t. x to obtain
        the interpolant score S_bar, and applies the requested multiplicative
        gain G(tau).

        Note: for H = I and R = sigma^2 I, G S_bar from Theorem 4.4 and
        Corollary 4.5 algebraically coincide -- the gain cancels Sigma_bar^{-1}
        independently of which Sigma_W approximation is used. ``"expensive"``
        and ``"new_correction"`` therefore produce identical scores there.
        Only ``"true"`` (which replaces the entire interpolant-likelihood
        construction with the SDE's exact Phi) escapes this.
        """

        if self.perturbation == "true":
            return self.true_likelihood_score(x, t, x0, observations)

        diff_fun = lambda x, x0: self.mu_fn(x, x0, t, observations[0])
        diff_fun_vmap = lambda x, x0: torch.vmap(diff_fun, in_dims=0, out_dims=0)(x, x0)
        diff = diff_fun_vmap(x, x0)

        if self.perturbation == "expensive":
            # Full Sigma_bar via autograd Jacobian J_b inside _compute_W_covariance.
            sigma_fn = lambda x, x0: self.sigma_fn(x, x0, t)
            Sigma = vmap(sigma_fn, in_dims=0, out_dims=0)(x, x0)  # (B, N_y, N_y)
            log_prb_fun = lambda diff, Sigma: (
                -0.5 * torch.dot(diff, torch.linalg.solve(Sigma, diff))
            )
            log_prb = torch.vmap(log_prb_fun, in_dims=0, out_dims=0)(diff, Sigma)
        else:
            tau = t[0, 0].item()
            interp = self.drift_model.interpolation
            beta = interp.beta(t)[0, 0].item()
            gamma = interp.gamma(t)[0, 0].item()
            beta_safe = beta if abs(beta) > 1e-3 else 1e-3
            sigma = beta_safe**2 * self.original_variance + gamma**2 * tau
            log_prb_fun = lambda diff: -0.5 * torch.dot(diff, diff) / sigma
            log_prb = torch.vmap(log_prb_fun, in_dims=0, out_dims=0)(diff)

        score = torch.autograd.grad(log_prb.sum(), x)[0]

        if self.perturbation == "new_correction":
            corr = self.new_correction(x, t, x0, observations, dt, diffusion_term)
            score = score * corr
        elif self.perturbation == "expensive":
            sigma_W_fn = lambda x_i, x0_i: self._compute_W_covariance(x_i, t, x0_i)
            sigma_W = torch.vmap(sigma_W_fn, in_dims=0, out_dims=0)(x, x0)
            G = self.gain_from_W_cov(sigma_W, t)  # (B, N_u, N_u)
            score = torch.einsum("bij,bj->bi", G, score)

        return score


class FlowdasLikelihood(nn.Module):
    """Interpolant likelihood."""

    def __init__(
        self,
        obs_matrix: torch.Tensor,
        drift_model: nn.Module,
        original_variance: float,
        ensemble_size: int = 25,
        multiplier: float = 4.0,
    ) -> None:
        """Initialize interpolant likelihood."""
        super(FlowdasLikelihood, self).__init__()
        self.drift_model = drift_model
        self.obs_matrix = obs_matrix
        self.original_variance = original_variance
        self.ensemble_size = ensemble_size
        self.multiplier = multiplier

    def forward(
        self,
    ) -> None:
        """Forward pass."""
        pass

    def _compute_one_step_prediction(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the one step prediction."""

        drift_milstein = self.drift_model(x, t, x0)
        pred = x + drift_milstein * (1.0 - t)
        # Add noise = integral of the diffusion term from t to 1
        pred = pred + torch.randn_like(x) * (
            2 / 3 - t.sqrt() + (1 / 3) * (t.sqrt()) ** 3
        )
        # RK step
        drift_rk = self.drift_model(pred, torch.ones_like(t), x0)
        pred = x + 0.5 * (drift_milstein + drift_rk) * (1 - t)

        # Expand the prediction to the ensemble size
        pred = pred.repeat(self.ensemble_size, 1, 1)

        # Add noise = integral of the diffusion term from t to 1
        return pred + torch.randn_like(pred) * (
            2 / 3 - t.sqrt() + (1 / 3) * (t.sqrt()) ** 3
        )

    def score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        observations: torch.Tensor,
        dt: torch.Tensor,
        diffusion_term: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Score function."""
        preds = self._compute_one_step_prediction(x, t, dt, x0)

        pred_obs = torch.matmul(preds, self.obs_matrix.T)

        diff = pred_obs - observations.unsqueeze(0)
        diff_norm = torch.linalg.norm(diff, dim=-1)
        diff_norm = -diff_norm / (2 * self.original_variance)

        weights = torch.softmax(diff_norm.detach() ** 2, dim=0)

        return (
            torch.autograd.grad((diff_norm * weights).sum(), x)[0]
            * dt
            * self.multiplier
        )
