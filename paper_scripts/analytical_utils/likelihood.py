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
        num_quad: int = 200,
    ) -> None:
        """Initialize interpolant likelihood.

        Args:
            obs_matrix: Observation matrix H, shape (N_y, N_u).
            drift_model: Trained (or analytical) drift model.
            original_variance: Scalar observation noise variance (R = original_variance * I).
            ensemble_size: Not used currently, kept for interface compatibility.
            use_covariance_correction: If True, include the C_tau correction
                from the conditional covariance of W_tau.  Requires computing
                the Jacobian of the drift, which costs O(N_u) backward passes.
            perturbation: Which LG likelihood correction to apply. One of
                ``None`` (no correction), ``"true"`` (exact analytical LG
                correction, requires forward integration of Phi and knowing
                the prior variance), ``"tangent"`` (cheap tangent-linear
                surrogate for Phi using ensemble variance -- Heuristic 1 of
                ``appendix_cheap_corrections.tex``), or ``"ensemble"``
                (ensemble-calibrated innovation variance, rescales the
                interpolant score magnitude -- Heuristic 3).
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
            "tangent",
            "ensemble",
            "residual",
            "deint",
            "endpoint",
            "hybrid",
            "adaptive",
            "blend",
            "M1",
            "M2",
            "M3",
            "M2M3",
        ):
            raise ValueError(
                f"perturbation must be None, 'true', 'tangent', 'ensemble', "
                f"'residual', 'deint', 'endpoint', or 'hybrid', "
                f"got {perturbation!r}"
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

    def _phi_and_vcond(self, tau: float) -> tuple[float, float]:
        """Compute Phi_{1,tau} and V_cond(tau) by quadrature (LG case)."""
        interp = self.drift_model.interpolation
        c = self.target_variance

        s = torch.linspace(tau, 1.0, self.num_quad).view(-1, 1)
        beta = interp.beta(s).view(-1)
        beta_diff = interp.beta_diff(s).view(-1)
        gamma = interp.gamma(s).view(-1)
        gamma_diff = interp.gamma_diff(s).view(-1)
        s_flat = s.view(-1)

        num = beta * beta_diff * c + s_flat * gamma * gamma_diff
        den = beta**2 * c + s_flat * gamma**2 + 1e-12
        a_vals = num / den

        log_phi = torch.trapz(a_vals, s_flat)
        phi = torch.exp(log_phi).item()

        t_tensor = torch.tensor([[tau]])
        beta_t = interp.beta(t_tensor)[0, 0].item()
        gamma_t = interp.gamma(t_tensor)[0, 0].item()
        V_tau = beta_t**2 * c + tau * gamma_t**2
        V_cond = max(c - phi**2 * V_tau, 0.0)
        return phi, V_cond

    def likelihood_perturbation(self, x, x0, t, observations):
        """Analytical LG correction to the interpolant likelihood score.

        Adds the term that turns the interpolant likelihood score into the
        exact posterior score for the linear-Gaussian test case of
        ``appendix_simple_test_case.tex``:

            perturbation = A_tau H^T (y - H x_0)  -  B_tau H^T H (x_tau - x_0)

        with

            A_tau = Phi / v_S  -  beta^3 c / (v_I V_tau)
            B_tau = Phi^2 / v_S  -  beta^4 c^2 / (v_I V_tau^2)

        and

            V_tau  = beta^2 c + tau gamma^2                       (SI marginal var)
            v_S    = V_cond + sigma^2,   V_cond = c - Phi^2 V_tau (SDE conditional)
            v_I    = beta^2 sigma^2 + gamma^2 tau
                     + gamma^4 tau^2 A_code (beta a - beta_diff)  (matches sigma_fn)

        The J_b-correction piece (last term in v_I) is essential: the score
        function in this class uses the full ``sigma_fn`` covariance, not the
        J_b-free surrogate -- so the analytical correction must be computed
        against the same v_I that the Gaussian log-likelihood uses, otherwise
        there is a residual bias and the perturbation fails to cancel.

        Assumptions: isotropic Gaussian prior N(x_0, c I), linear analytical
        drift (so J_b = a(tau) * I is a scalar), and observation operator H
        such that H^T H is isotropic (the test case uses H = I). Outside this
        setting the correction is only locally valid.
        """
        tau = t[0, 0].item()
        phi, V_cond = self._phi_and_vcond(tau)

        interp = self.drift_model.interpolation
        beta = interp.beta(t)[0, 0].item()
        beta_diff = interp.beta_diff(t)[0, 0].item()
        gamma = interp.gamma(t)[0, 0].item()
        gamma_diff = interp.gamma_diff(t)[0, 0].item()

        c = self.target_variance
        sigma2 = self.original_variance

        V_tau = beta**2 * c + tau * gamma**2
        v_S = V_cond + sigma2

        # Full scalar v_I matching sigma_fn: analytical drift Jacobian
        # J_b = a(tau) I enters the covariance correction as
        #   gamma^4 tau^2 A_code (beta a - beta_diff).
        a_tau = (beta * beta_diff * c + tau * gamma * gamma_diff) / V_tau
        A_code = 1.0 / (tau * gamma * (beta_diff * gamma - beta * gamma_diff) + 1e-12)
        jac_correction = gamma**4 * tau**2 * A_code * (beta * a_tau - beta_diff)
        v_I = beta**2 * sigma2 + gamma**2 * tau + jac_correction

        A_tau = phi / v_S - (beta**3 * c) / (v_I * V_tau)
        B_tau = phi**2 / v_S - (beta**4 * c**2) / (v_I * V_tau**2)

        H = self.obs_matrix
        y = observations[0]  # (N_y,)
        innovation = y.unsqueeze(0) - x0 @ H.T  # (B, N_y)
        displacement = (x - x0) @ (H.T @ H)  # (B, N_u)

        return A_tau * (innovation @ H) - B_tau * displacement

    def new_correction(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        observations: torch.Tensor,
        dt: torch.Tensor,
        diffusion_term: Callable,
    ) -> torch.Tensor:
        """Compute the new correction term for the likelihood score."""

        tau = t[0, 0].item()
        interp = self.drift_model.interpolation
        alpha = interp.alpha(t)[0, 0].item()
        beta = interp.beta(t)[0, 0].item()
        gamma = interp.gamma(t)[0, 0].item()
        beta_safe = beta if abs(beta) > 1e-3 else 1e-3

        # def compute_mu_t(x):
        #     drift = self.drift_model._compute_drift(x, t[0:1], x0)
        #     model_score = self.drift_model._compute_score_from_drift(
        #         x, t[0:1], x0, drift
        #     )

        #     E_W = -gamma * t[0, 0] * model_score  # E[W_tau | x_tau, x_0]

        #     mu_t = x - alpha * x0 - gamma * E_W
        #     mu_t = mu_t / beta_safe
        #     return mu_t.sum(dim=0)

        # # mu_t_jacobian = torch.autograd.functional.jacobian(compute_mu_t, x)
        # # mu_t_jacobian = mu_t_jacobian.transpose(0, 1)
        # H = self.obs_matrix

        # mu_t = compute_mu_t(x)
        # mu_t = mu_t @ H

        # HTR = H.T @ torch.eye(2) / self.original_variance

        # score = (observations - mu_t) @ HTR.T

        G = gamma * gamma * t[0, 0] / beta_safe / beta_safe / self.original_variance
        G = 1 + G

        return G

    def score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        observations: torch.Tensor,
        dt: torch.Tensor,
        diffusion_term: Callable,
    ) -> torch.Tensor:
        """Compute the likelihood score nabla_x log p(ybar_tau | x_tau, x_0).

        The score is obtained by constructing the Gaussian log-likelihood
        with the corrected mean (d_tau) and optionally the corrected
        covariance (C_tau), then differentiating w.r.t. x via autograd.
        """

        diff_fun = lambda x, x0: self.mu_fn(x, x0, t, observations[0])
        diff_fun_vmap = lambda x, x0: torch.vmap(diff_fun, in_dims=0, out_dims=0)(x, x0)
        diff = diff_fun_vmap(x, x0)

        if self.perturbation == "true":
            sigma_fn = lambda x, x0: self.sigma_fn(x, x0, t)
            Sigma = vmap(sigma_fn, in_dims=0, out_dims=0)(x, x0)  # (B, d_y, d_y)

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

        if self.perturbation == "true":
            score = score + self.likelihood_perturbation(x, x0, t, observations)
        elif self.perturbation == "new_correction":
            corr = self.new_correction(x, t, x0, observations, dt, diffusion_term)
            score = score * corr

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
