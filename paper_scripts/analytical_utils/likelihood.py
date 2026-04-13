
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
        gamma = self.drift_model.interpolation.gamma(t)[0,0].item()
        beta = self.drift_model.interpolation.beta(t)[0,0].item()
        beta_diff = self.drift_model.interpolation.beta_diff(t)[0,0].item()
        gamma_diff = self.drift_model.interpolation.gamma_diff(t)[0,0].item()

        A_tau = 1.0 / (
            t[0, 0] * gamma * (beta_diff * gamma - beta * gamma_diff) + 1e-8
        ).item()

        # --- Compute J_b via autograd on a detached copy of x ---

        x = x.unsqueeze(0)
        x0 = x0.unsqueeze(0)

        J_b = jacrev(
            lambda x: self.drift_model._compute_drift(x, t[0:1], x0).sum(dim=0),
        )(x)
        J_b = J_b.transpose(0, 1).squeeze(0)

        return t[0,0] + gamma**2 * t[0,0]**2 * A_tau * (beta * J_b - beta_diff)
        # I = torch.eye(J_b.shape[-1], device=J_b.device, dtype=J_b.dtype)
        # return t[0,0] * I + gamma**2 * t[0,0]**2 * A_tau * (beta * J_b - beta_diff * I)

    def mu_fn(self, x, x0, t, observations):
        """Compute the mean."""
        gamma = self.drift_model.interpolation.gamma(t)[0,0]
        beta = self.drift_model.interpolation.beta(t)[0,0]
        alpha = self.drift_model.interpolation.alpha(t)[0,0]

        i_obs = alpha * torch.matmul(self.obs_matrix, x0) + beta * observations

        x_obs = torch.matmul(self.obs_matrix, x)

        drift = self.drift_model._compute_drift(x.unsqueeze(0), t, x0.unsqueeze(0))
        model_score = self.drift_model._compute_score_from_drift(
            x.unsqueeze(0), t[0:1], x0.unsqueeze(0), drift
        ).squeeze(0)

        E_W = -gamma * t[0,0] * model_score  # E[W_tau | x_tau, x_0]
        d = -gamma * torch.matmul(self.obs_matrix, E_W)

        return i_obs - x_obs - d
    
    def sigma_fn(self, x, x0, t):
        """Compute the covariance."""

        gamma = self.drift_model.interpolation.gamma(t)[0,0].item()
        beta = self.drift_model.interpolation.beta(t)[0,0].item()

        R = beta ** 2 * self.original_variance

        R = R #+ gamma**2 * t[0,0]

        cov_W = self._compute_W_covariance(x, t, x0)

        cov_W = self.obs_matrix@ cov_W @ self.obs_matrix.T
        cov_W = gamma**2 * cov_W

        lam = beta**2 * self.original_variance
        lam = lam / (beta**2*self.original_variance + gamma**2 * t[0,0])

        sigma = R * torch.eye(
            self.obs_matrix.shape[0], device=x.device
        ) + cov_W

        return sigma #* lam

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
        diff_fun_vmap = lambda x, x0: torch.vmap(
            diff_fun, in_dims=0, out_dims=0
        )(x, x0)
        diff = diff_fun_vmap(x, x0)

        sigma_fn = lambda x, x0: self.sigma_fn(x, x0, t)
        Sigma = vmap(sigma_fn, in_dims=0, out_dims=0)(x, x0)        # (B, d_y, d_y)

        log_prb_fun = lambda diff, Sigma: (
            -0.5 * torch.dot(diff, torch.linalg.solve(Sigma, diff))
        ) * 2

        log_prb = torch.vmap(log_prb_fun, in_dims=0, out_dims=0)(diff, Sigma)

        return torch.autograd.grad(log_prb.sum(), x)[0]


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