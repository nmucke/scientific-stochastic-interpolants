from typing import Callable, Optional

import torch
import torch.nn as nn
import tqdm


class InterpolantLikelihood(nn.Module):
    """Interpolant likelihood."""

    def __init__(
        self,
        obs_matrix: torch.Tensor,
        drift_model: nn.Module,
        original_variance: float,
        ensemble_size: int = 100,
    ) -> None:
        """Initialize interpolant likelihood."""
        super(InterpolantLikelihood, self).__init__()
        self.drift_model = drift_model
        self.obs_matrix = obs_matrix
        self.original_variance = original_variance
        self.ensemble_size = ensemble_size

    def forward(
        self,
    ) -> None:
        """Forward pass."""
        pass

    def _interpolate_observations(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate the observations."""

        gamma = self.drift_model.interpolation.gamma(t)
        beta = self.drift_model.interpolation.beta(t)

        x0_obs = torch.matmul(x0, self.obs_matrix.T)

        interpolant_obs = self.drift_model.interpolation.forward(
            x0_obs,
            observations,
            t,
            0.0 * torch.randn_like(x0_obs),  # torch.zeros_like(x0_obs) #
        )

        # Compute the scale of the interpolant of the observation
        interpolant_variance = beta ** 2 * self.original_variance
        interpolant_variance += gamma ** 2 * t
        
        # interpolant_variance -= (
        #     (gamma ** 4 * t ** 2) / (beta ** 2 * self.original_variance + gamma ** 2 * t)
        # )

        return interpolant_obs, interpolant_variance

    def score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        observations: torch.Tensor,
        dt: torch.Tensor,
        diffusion_term: Callable,
    ) -> torch.Tensor:
        """Score function."""

        gamma = self.drift_model.interpolation.gamma(t)
        beta = self.drift_model.interpolation.beta(t)
        alpha = self.drift_model.interpolation.alpha(t)

        sigma = beta ** 2 * self.original_variance + gamma ** 2 * t
        
        i_obs = alpha * torch.matmul(x0, self.obs_matrix.T) + beta * observations

        model_score = self.drift_model._compute_score_from_drift(
            x, t, x0, self.drift_model._compute_drift(x, t, x0)
        ).detach()
        conditional_noise_mean = -model_score * t * gamma

        x_obs = torch.matmul(x, self.obs_matrix.T)

        diff = i_obs - x_obs + gamma**2 * torch.matmul(conditional_noise_mean, self.obs_matrix.T)

        score = torch.matmul(diff, self.obs_matrix) / sigma

        multiplier = 2.0

        return multiplier * score


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

        weights = torch.softmax(diff_norm.detach(), dim=0)

        return (
            torch.autograd.grad((diff_norm * weights).sum(), x)[0]
            * dt
            * self.multiplier
        )
