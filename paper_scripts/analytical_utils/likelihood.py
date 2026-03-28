# from typing import Callable, Optional

# import torch
# import torch.nn as nn
# import tqdm


# class InterpolantLikelihood(nn.Module):
#     """Interpolant likelihood."""

#     def __init__(
#         self,
#         obs_matrix: torch.Tensor,
#         drift_model: nn.Module,
#         original_variance: float,
#         ensemble_size: int = 100,
#     ) -> None:
#         """Initialize interpolant likelihood."""
#         super(InterpolantLikelihood, self).__init__()
#         self.drift_model = drift_model
#         self.obs_matrix = obs_matrix
#         self.original_variance = original_variance
#         self.ensemble_size = ensemble_size

#     def forward(
#         self,
#     ) -> None:
#         """Forward pass."""
#         pass

#     def _interpolate_observations(
#         self,
#         x: torch.Tensor,
#         t: torch.Tensor,
#         x0: torch.Tensor,
#         observations: torch.Tensor,
#     ) -> torch.Tensor:
#         """Interpolate the observations."""

#         gamma = self.drift_model.interpolation.gamma(t)
#         beta = self.drift_model.interpolation.beta(t)

#         x0_obs = torch.matmul(x0, self.obs_matrix.T)

#         interpolant_obs = self.drift_model.interpolation.forward(
#             x0_obs,
#             observations,
#             t,
#             0.0 * torch.randn_like(x0_obs),  # torch.zeros_like(x0_obs) #
#         )

#         # Compute the scale of the interpolant of the observation
#         interpolant_variance = beta ** 2 * self.original_variance
#         interpolant_variance += gamma ** 2 * t
        
#         # interpolant_variance -= (
#         #     (gamma ** 4 * t ** 2) / (beta ** 2 * self.original_variance + gamma ** 2 * t)
#         # )

#         return interpolant_obs, interpolant_variance

#     def score(
#         self,
#         x: torch.Tensor,
#         t: torch.Tensor,
#         x0: torch.Tensor,
#         observations: torch.Tensor,
#         dt: torch.Tensor,
#         diffusion_term: Callable,
#     ) -> torch.Tensor:
#         """Score function."""

#         gamma = self.drift_model.interpolation.gamma(t)
#         beta = self.drift_model.interpolation.beta(t)
#         alpha = self.drift_model.interpolation.alpha(t)

#         sigma = beta ** 2 * self.original_variance + gamma ** 2 * t
        
#         i_obs = alpha * torch.matmul(x0, self.obs_matrix.T) + beta * observations

#         model_score = self.drift_model._compute_score_from_drift(
#             x, t, x0, self.drift_model._compute_drift(x, t, x0)
#         )
#         conditional_noise_mean = -model_score * t * gamma
#         d = - gamma * torch.matmul(conditional_noise_mean, self.obs_matrix.T)

#         x_obs = torch.matmul(x, self.obs_matrix.T)

#         diff = i_obs - x_obs - d

#         score = torch.matmul(diff, self.obs_matrix) / sigma

#         multiplier = 1.0


#         diff_norm = torch.linalg.norm(diff, dim=-1)**2
#         diff_norm = -diff_norm / (2 * sigma[0,0])
#         _score = torch.autograd.grad(diff_norm.sum(), x)[0]

#         import pdb; pdb.set_trace()

#         return multiplier * score


# class FlowdasLikelihood(nn.Module):
#     """Interpolant likelihood."""

#     def __init__(
#         self,
#         obs_matrix: torch.Tensor,
#         drift_model: nn.Module,
#         original_variance: float,
#         ensemble_size: int = 25,
#         multiplier: float = 4.0,
#     ) -> None:
#         """Initialize interpolant likelihood."""
#         super(FlowdasLikelihood, self).__init__()
#         self.drift_model = drift_model
#         self.obs_matrix = obs_matrix
#         self.original_variance = original_variance
#         self.ensemble_size = ensemble_size
#         self.multiplier = multiplier

#     def forward(
#         self,
#     ) -> None:
#         """Forward pass."""
#         pass

#     def _compute_one_step_prediction(
#         self,
#         x: torch.Tensor,
#         t: torch.Tensor,
#         dt: torch.Tensor,
#         x0: torch.Tensor,
#     ) -> torch.Tensor:
#         """Compute the one step prediction."""

#         drift_milstein = self.drift_model(x, t, x0)
#         pred = x + drift_milstein * (1.0 - t)
#         # Add noise = integral of the diffusion term from t to 1
#         pred = pred + torch.randn_like(x) * (
#             2 / 3 - t.sqrt() + (1 / 3) * (t.sqrt()) ** 3
#         )
#         # RK step
#         drift_rk = self.drift_model(pred, torch.ones_like(t), x0)
#         pred = x + 0.5 * (drift_milstein + drift_rk) * (1 - t)

#         # Expand the prediction to the ensemble size
#         pred = pred.repeat(self.ensemble_size, 1, 1)

#         # Add noise = integral of the diffusion term from t to 1
#         return pred + torch.randn_like(pred) * (
#             2 / 3 - t.sqrt() + (1 / 3) * (t.sqrt()) ** 3
#         )

#     def score(
#         self,
#         x: torch.Tensor,
#         t: torch.Tensor,
#         x0: torch.Tensor,
#         observations: torch.Tensor,
#         dt: torch.Tensor,
#         diffusion_term: Optional[nn.Module] = None,
#     ) -> torch.Tensor:
#         """Score function."""
#         preds = self._compute_one_step_prediction(x, t, dt, x0)

#         pred_obs = torch.matmul(preds, self.obs_matrix.T)

#         diff = pred_obs - observations.unsqueeze(0)
#         diff_norm = torch.linalg.norm(diff, dim=-1)
#         diff_norm = -diff_norm / (2 * self.original_variance)

#         weights = torch.softmax(diff_norm.detach(), dim=0)

#         return (
#             torch.autograd.grad((diff_norm * weights).sum(), x)[0]
#             * dt
#             * self.multiplier
#         )


from typing import Callable, Optional

import torch
import torch.nn as nn
import tqdm


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
        use_covariance_correction: bool = False,
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
        """
        super(InterpolantLikelihood, self).__init__()
        self.drift_model = drift_model
        self.obs_matrix = obs_matrix
        self.original_variance = original_variance
        self.ensemble_size = ensemble_size
        self.use_covariance_correction = use_covariance_correction

    def forward(self) -> None:
        """Forward pass."""
        pass

    def _compute_covariance_correction(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the covariance correction C_tau (Lemma 4.3).

        C_tau = -gamma^4 * tau^2 * H @ [A_tau * (beta * J_b - beta_dot * I)] @ H^T

        where J_b = nabla_x b_theta(x, x0, tau) is the Jacobian of the drift.

        Note: C_tau is computed with detached x, so it is treated as constant
        w.r.t. x in the outer autograd call.  This is exact when the drift is
        affine in x (e.g. the analytical Gaussian test case).

        Returns:
            C_tau: shape (batch, N_y, N_y).
        """
        gamma = self.drift_model.interpolation.gamma(t)
        beta = self.drift_model.interpolation.beta(t)
        beta_diff = self.drift_model.interpolation.beta_diff(t)
        gamma_diff = self.drift_model.interpolation.gamma_diff(t)

        A_tau = 1.0 / (
            t * gamma * (beta_diff * gamma - beta * gamma_diff) + 1e-8
        )

        batch_size, dim = x.shape

        # --- Compute J_b via autograd on a detached copy of x ---
        x_d = x.detach().requires_grad_(True)
        drift_d = self.drift_model._compute_drift(x_d, t, x0)

        J_b = torch.zeros(batch_size, dim, dim, device=x.device)
        for i in range(dim):
            g = torch.autograd.grad(
                drift_d[:, i].sum(),
                x_d,
                retain_graph=(i < dim - 1),
            )[0]
            J_b[:, i, :] = g

        # --- Hessian of log p_tau:  A_tau * (beta * J_b - beta_dot * I) ---
        I_mat = torch.eye(dim, device=x.device).unsqueeze(0)
        hess = A_tau.unsqueeze(-1) * (
            beta.unsqueeze(-1) * J_b - beta_diff.unsqueeze(-1) * I_mat
        )

        # --- C_tau = -gamma^4 * tau^2 * H @ hess @ H^T ---
        H = self.obs_matrix  # (N_y, dim)
        H_hess = torch.matmul(H.unsqueeze(0), hess)  # (batch, N_y, dim)
        C_tau = -(gamma ** 4 * t ** 2).unsqueeze(-1) * torch.matmul(
            H_hess, H.T.unsqueeze(0)
        )

        return C_tau  # (batch, N_y, N_y)

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
        gamma = self.drift_model.interpolation.gamma(t)
        beta = self.drift_model.interpolation.beta(t)
        alpha = self.drift_model.interpolation.alpha(t)

        # --- Interpolated observation: ybar_tau = alpha * H(x0) + beta * y ---
        i_obs = alpha * torch.matmul(x0, self.obs_matrix.T) + beta * observations

        # --- Observation of current state: H @ x_tau ---
        x_obs = torch.matmul(x, self.obs_matrix.T)

        # --- Mean correction d_tau = -gamma * H @ E[W_tau | x_tau, x_0] ---
        drift = self.drift_model._compute_drift(x, t, x0)
        model_score = self.drift_model._compute_score_from_drift(
            x, t, x0, drift
        )
        E_W = -gamma * t * model_score  # E[W_tau | x_tau, x_0]
        d = -gamma * torch.matmul(E_W, self.obs_matrix.T)
        d = d

        # --- Innovation: ybar_tau - E[ybar_tau | x_tau, x_0] ---
        diff = i_obs - x_obs - d

        # import pdb; pdb.set_trace()

        if True: #self.use_covariance_correction:
            # Full matrix-form likelihood with corrected covariance
            N_y = self.obs_matrix.shape[0]

            # Marginal covariance: R_bar = beta^2 * R + gamma^2 * tau * H H^T
            HHT = torch.matmul(self.obs_matrix, self.obs_matrix.T)
            R_bar = (
                (beta ** 2 * self.original_variance).unsqueeze(-1)
                * torch.eye(N_y, device=x.device).unsqueeze(0)
                + (gamma ** 2 * t).unsqueeze(-1) * HHT.unsqueeze(0)
            )  # (batch, N_y, N_y)

            C_tau = self._compute_covariance_correction(x, t, x0)
            Sigma = R_bar - C_tau  # (batch, N_y, N_y)

            # log p = -0.5 * diff^T @ Sigma^{-1} @ diff  (up to const)
            # Note: log-det term is constant w.r.t. x since C_tau is detached,
            # so it does not contribute to the score.
            Sigma_inv_diff = torch.linalg.solve(Sigma, diff.unsqueeze(-1))
            log_lik = -0.5 * torch.sum(
                diff * Sigma_inv_diff.squeeze(-1), dim=-1
            )
        else:
            # Scalar variance (isotropic case, no covariance correction)
            sigma = beta ** 2 * self.original_variance + gamma ** 2 * t

            # log p = -||diff||^2 / (2 * sigma)
            log_lik = -torch.sum(diff ** 2, dim=-1) / (2.0 * sigma.squeeze(-1))

        score = torch.autograd.grad(log_lik.sum(), x)[0]

        # sigma = beta ** 2 * self.original_variance + gamma ** 2 * t
        # _score = torch.matmul(diff, self.obs_matrix) / sigma

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