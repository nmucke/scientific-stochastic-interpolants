from typing import Optional

import torch
import torch.nn as nn

from scisi.likelihood_models.observation_operators import LinearObservationOperator


class GuidanceGaussianLikelihood(nn.Module):
    """Multivariate Gaussian likelihood."""

    def __init__(
        self,
        model: nn.Module,
        obs_operator: nn.Module = LinearObservationOperator,
        variance: float = 0.05,
        ensemble_size: int = 1,
    ) -> None:
        """
        Initialize Multivariate Gaussian likelihood.

        Either covariance_matrix or precision_matrix must be provided.

        Args:
            obs_operator: Observation operator.
            model: Model.
            variance: Variance.
        """
        super(GuidanceGaussianLikelihood, self).__init__()
        self.obs_operator = obs_operator
        self.model = model
        self.original_variance = variance
        self.ensemble_size = ensemble_size

    def forward(
        self, x: torch.Tensor, observations: torch.Tensor, variance: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. [B, C, H, W]
            observations: Observations.
            variance: Variance.

        Returns:
            torch.Tensor: Log probability.
        """
        pass

    def _compute_one_step_prediction(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        drift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the one step predictions."""

        # alpha = self.model.interpolation.alpha(t)
        # alpha_diff = self.model.interpolation.alpha_diff(t)
        # beta = self.model.interpolation.beta(t)
        # beta_diff = self.model.interpolation.beta_diff(t)

        # x_coeff = alpha_diff / (beta_diff * alpha - alpha_diff * beta)
        # score_coeff = alpha / (beta_diff * alpha - alpha_diff * beta)

        # return x_coeff * x + score_coeff * drift

        return x + (1 - t) * drift

    def _schedule(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the schedule."""

        alpha = self.model.interpolation.alpha(t)
        alpha_diff = self.model.interpolation.alpha_diff(t)
        beta = self.model.interpolation.beta(t)
        beta_diff = self.model.interpolation.beta_diff(t)

        # return (beta_diff * alpha - alpha_diff * beta) / alpha
        # return alpha * (beta_diff * alpha - alpha_diff * beta) / beta
        return (1 - t) / t

    def score(
        self,
        observations: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        drift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the likelihood score."""

        preds = self._compute_one_step_prediction(
            x, t, dt, field_history, field_cond, pars_cond, drift
        )

        diff_norm = (
            torch.linalg.norm(observations - self.obs_operator(preds), dim=1) ** 2
        )
        diff_norm = 0.5 * diff_norm / self.original_variance

        # Compute weighted gradient
        score = torch.autograd.grad(
            outputs=diff_norm,
            inputs=x,
        )[0]

        # schedule = self._schedule(t)
        # if torch.isnan(schedule).any() or torch.isinf(schedule).any():
        #     schedule = 0.0

        # return -schedule * score * 0.1
        return -score * 0.01
