import pdb
from functools import partial
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.func as F

from scisi.likelihood_models.observation_operators import LinearObservationOperator
from scisi.models.interpolations import (
    LinearDeterministicInterpolation,
    QuadraticDeterministicInterpolation,
)

class InterpolantGaussianLikelihood(nn.Module):
    """Interpolant Gaussian likelihood."""

    def __init__(
        self,
        model: nn.Module,
        obs_operator: nn.Module = LinearObservationOperator,
        variance: float = 0.05,
        ensemble_size: int = 1,
        interpolant: Optional[nn.Module] = None,
    ) -> None:
        """
        Initialize Interpolant Gaussian likelihood.

        Either covariance_matrix or precision_matrix must be provided.

        Args:
            obs_operator: Observation operator.
            model: Model.
            variance: Variance.
        """
        super(InterpolantGaussianLikelihood, self).__init__()
        self.obs_operator = obs_operator
        self.model = model
        self.original_variance = variance
        self.ensemble_size = ensemble_size

        if interpolant is not None:
            self.interpolant = interpolant
        else:
            self.interpolant = self.model.interpolation

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

    def _interpolate_observations(
        self,
        observations: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        base_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate the observations."""

        interpolant_obs = (
            self.interpolant.alpha(t) * base_obs
            + self.interpolant.beta(t) * observations
            # + self.diffusion_term(t) * torch.randn_like(base_obs) * torch.sqrt(t)
        )

        # Compute the scale of the interpolant of the observation
        interpolant_variance = (
            self.interpolant.beta(t) ** 2 * self.original_variance
            + self.model.interpolation.gamma(t) ** 2 * t
        )

        # interpolant_variance -= self.model.interpolation.gamma(t) ** 2 * t

        # interpolant_variance += 1e-6
        
        return interpolant_obs, interpolant_variance

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
        diffusion_term: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Compute the likelihood score."""

        if diffusion_term is None:
            self.diffusion_term = self.model.interpolation.gamma
        else:
            self.diffusion_term = diffusion_term

        interpolant_obs, interpolant_variance = self._interpolate_observations(
            observations, x, t, self.obs_operator(field_history[..., -1])
        )

        model_score = self.model._prior_score(
            x, field_history[..., -1], drift, t
        )#.detach()
        conditional_noise_mean = -model_score * t * self.model.interpolation.gamma(t)

        d = - self.diffusion_term(t) * self.obs_operator(conditional_noise_mean)

        diff = interpolant_obs - self.obs_operator(x) - d

        diff_norm = torch.linalg.norm(diff, dim=1)

        diff_norm = - 0.5 * diff_norm / interpolant_variance

        likelihood_score = torch.autograd.grad(
            outputs=diff_norm.sum(),
            inputs=x,
        )[0]

        drift_norm = torch.linalg.norm(model_score[0])
        print(drift_norm)


        return drift_norm * likelihood_score * self.diffusion_term(t) ** 2, diff_norm

class FlowdasGaussianLikelihood(nn.Module):
    """Multivariate Gaussian likelihood."""

    def __init__(
        self,
        model: nn.Module,
        obs_operator: nn.Module = LinearObservationOperator,
        variance: float = 0.05,
        ensemble_size: int = 1,
        integration_order: int = 1,
    ) -> None:
        """
        Initialize Multivariate Gaussian likelihood.

        Either covariance_matrix or precision_matrix must be provided.

        Args:
            obs_operator: Observation operator.
            model: Model.
            variance: Variance.
        """
        super(FlowdasGaussianLikelihood, self).__init__()
        self.obs_operator = obs_operator
        self.model = model
        self.original_variance = variance
        self.ensemble_size = ensemble_size
        self.dist = torch.distributions.MultivariateNormal
        self.integration_order = integration_order

        self.integral_variance = lambda t: 2 / 3 - t.sqrt() + (1 / 3) * (t.sqrt()) ** 3

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

        # Milstein step
        drift_milstein = (
            drift
            if drift is not None
            else self.model.drift_model(x, t, field_history, field_cond, pars_cond)
        )
        pred = x + drift_milstein * (1.0 - t)

        # Add noise = integral of the diffusion term from t to 1
        pred = pred + torch.randn_like(x) * self.integral_variance(t)

        # RK step
        drift_rk = self.model.drift_model(
            pred, torch.ones_like(t), field_history, field_cond, pars_cond
        )
        pred = x + 0.5 * (drift_milstein + drift_rk) * (1 - t)

        # Expand the prediction to the ensemble size
        pred = pred.repeat(self.ensemble_size, 1, 1, 1)

        # Add noise = integral of the diffusion term from t to 1
        return pred + torch.randn_like(pred) * self.integral_variance(t)

    # def score(
    #     self,
    #     observations: torch.Tensor,
    #     x: torch.Tensor,
    #     t: torch.Tensor,
    #     field_history: torch.Tensor,
    #     field_cond: Optional[torch.Tensor] = None,
    #     pars_cond: Optional[torch.Tensor] = None,
    #     dt: Optional[torch.Tensor] = None,
    #     drift: Optional[torch.Tensor] = None,
    #     **kwargs: Any,
    # ) -> torch.Tensor:
    #     """Compute the likelihood score."""

    #     preds = self._compute_one_step_prediction(
    #         x, t, dt, field_history, field_cond, pars_cond, drift
    #     )

    #     pdb.set_trace()

    #     differences = torch.linalg.norm(observations - self.obs_operator(preds), dim=1)
    #     weights = - 0.5 * differences / self.original_variance

    #     # Detach the weights to avoid gradients
    #     weights_detached = weights.detach()

    #     # Compute weights
    #     softmax_weights = torch.softmax(weights_detached, dim=0)

    #     # Element-wise multiplication of the weights and the differences
    #     result = softmax_weights * differences
        
    #     # Sum the result to get the final result
    #     final_result = result.sum()

    #     # Compute weighted gradient
    #     score = torch.autograd.grad(
    #         outputs=final_result,
    #         inputs=x,
    #     )[0]

    #     # diffusion_term_squared = self.model.interpolation.gamma(t) ** 2 + 1e-6

    #     return (
    #         - score, #/ diffusion_term_squared, 
    #         weights # Is not used but required for compatibility
    #     )

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
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the likelihood score."""

        preds = self._compute_one_step_prediction(
            x, t, dt, field_history, field_cond, pars_cond, drift
        )

        diff_norm = torch.linalg.norm(observations - self.obs_operator(preds), dim=1)
        diff_norm = -diff_norm / (2 * self.original_variance)

        # Compute weights
        weights = torch.softmax(diff_norm.detach(), dim=0)

        # Compute weighted gradient
        score = torch.autograd.grad(
            outputs=(diff_norm * weights).sum(),
            inputs=x,
        )[0]

        return -score, weights
