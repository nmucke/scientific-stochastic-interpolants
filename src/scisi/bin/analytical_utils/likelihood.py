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

    def _interpolate_observations(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate the observations."""

        x0_obs = torch.matmul(x0, self.obs_matrix.T)

        interpolant_obs = self.drift_model.interpolation.forward(
            x0_obs,
            observations,
            t,
            torch.randn_like(x0_obs),  # torch.zeros_like(x0_obs) #
        )

        # Compute the scale of the interpolant of the observation
        interpolant_variance = (
            self.drift_model.interpolation.beta(t) ** 2 * self.original_variance
        )
        # interpolant_variance = (
        #     interpolant_variance + self.drift_model.interpolation.gamma(t) ** 2 * (t)
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

        # end_pred = self._compute_one_step_prediction(x, t, dt, x0)
        # diff = observations - torch.matmul(end_pred, self.obs_matrix.T)
        # diff_norm = (diff * diff).sum(dim=-1)
        # diff_norm = -0.5 * diff_norm / self.original_variance
        # weights = torch.softmax(diff_norm.detach(), dim=0)
        # end_score = torch.autograd.grad((diff_norm * weights).sum(), x)[0].detach()

        # x.requires_grad = True

        # preds = x + self.drift_model(x, t, x0) * dt
        # preds = preds.repeat(self.ensemble_size, 1, 1)
        # preds = preds + diffusion_term(t) * torch.randn_like(preds) * torch.sqrt(dt)

        interpolant_obs, interpolant_variance = self._interpolate_observations(
            x, t, x0, observations
        )

        diff = interpolant_obs - torch.matmul(x, self.obs_matrix.T)
        diff_norm = (diff * diff).sum(dim=-1)
        diff_norm = -0.5 * diff_norm / interpolant_variance[0, 0]
        score = torch.autograd.grad(diff_norm.sum(), x)[0].detach()
        # weights = torch.softmax(diff_norm.detach(), dim=0)
        # score = torch.autograd.grad((diff_norm * weights).sum(), x)[0]

        # inner_score = (end_score * score).sum(dim=-1)
        # inner_score = inner_score.mean(dim=0)

        # norm_score = (torch.abs(score * score)).sum(dim=-1)
        # norm_score = norm_score.mean(dim=0)

        # multiplier = inner_score / norm_score

        # print(multiplier)

        multiplier = 0.5  # / torch.sqrt(self.drift_model.interpolation.gamma(t))

        return multiplier * score


class FlowdasLikelihood(nn.Module):
    """Interpolant likelihood."""

    def __init__(
        self,
        obs_matrix: torch.Tensor,
        drift_model: nn.Module,
        original_variance: float,
        ensemble_size: int = 5,
    ) -> None:
        """Initialize interpolant likelihood."""
        super(FlowdasLikelihood, self).__init__()
        self.drift_model = drift_model
        self.obs_matrix = obs_matrix
        self.original_variance = original_variance
        self.ensemble_size = ensemble_size

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

        return torch.autograd.grad((diff_norm * weights).sum(), x)[0] * dt * 3
