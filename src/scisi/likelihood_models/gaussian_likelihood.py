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
        correct_likelihood_score: bool = True,
        correction_multiplier: float = 3.0,
        apply_multiplier_to_full_expression: bool = True,
    ) -> None:
        """
        Initialize Interpolant Gaussian likelihood.

        Either covariance_matrix or precision_matrix must be provided.

        Args:
            obs_operator: Observation operator.
            model: Model.
            variance: Variance.
            correction_multiplier: Scalar tuning knob on the corrected
                likelihood score. By default it scales only the (uncorrected)
                score term — see `apply_multiplier_to_full_expression`.
            apply_multiplier_to_full_expression: If True,
                `correction_multiplier` multiplies the entire corrected
                expression `likelihood_score + correction_factor * obs_mask *
                likelihood_score`. If False (default, original behavior), the
                multiplier scales only the leading score term and the rank-N_y
                correction is added on top unscaled.
        """
        super(InterpolantGaussianLikelihood, self).__init__()
        self.obs_operator = obs_operator
        self.model = model
        self.original_variance = variance
        self.ensemble_size = ensemble_size
        self.correct_likelihood_score = correct_likelihood_score
        self.correction_multiplier = correction_multiplier
        self.apply_multiplier_to_full_expression = apply_multiplier_to_full_expression

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
            + self.interpolant.gamma(t) ** 2 * t + 1e-4
        )
        
        return interpolant_obs, interpolant_variance

    def _compute_likelihood_score_correction(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        likelihood_score: torch.Tensor
    ):
        """Correct the likelihood score.

        For point-observation operators (grid/random selection), H is a selection
        matrix so H^T H is diagonal — 1 at observed locations, 0 elsewhere. The
        rank-N_y correction from eq. (26) therefore reduces to an elementwise
        multiply with the observation mask, avoiding the dense N_u x N_u product.
        """

        correction_factor = self.diffusion_term(t) ** 2 * t
        correction_factor /= self.interpolant.beta(t) ** 2 + 1e-3
        correction_factor /= self.original_variance

        if getattr(self, "_obs_mask", None) is None \
                or self._obs_mask.device != likelihood_score.device:
            self._obs_mask = self.obs_operator.obs_indices_on_grid.to(
                device=likelihood_score.device, dtype=likelihood_score.dtype
            )

        if self.apply_multiplier_to_full_expression:
            return self.correction_multiplier * (
                likelihood_score
                + correction_factor * self._obs_mask * likelihood_score
            )
        return (
            self.correction_multiplier * likelihood_score
            + correction_factor * self._obs_mask * likelihood_score
        )

    def _conditional_mean_wiener_process(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        drift: torch.Tensor,
    ):
        gamma = self.interpolant.gamma(t)
        gamma_diff = self.interpolant.gamma_diff(t)
        beta = self.interpolant.beta(t)
        beta_diff = self.interpolant.beta_diff(t)
        alpha = self.interpolant.alpha(t)
        alpha_diff = self.interpolant.alpha_diff(t)

        A = t * gamma * (beta_diff * gamma - beta * gamma_diff)
        A = 1 / (A + 1e-6)

        c = beta_diff * x + (beta * alpha_diff - beta_diff * alpha) * field_history[..., -1]

        return -gamma * t * A * (beta * drift - c)


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

        bias = self._conditional_mean_wiener_process(
            x=x, t=t,field_history=field_history, drift=drift,
        )
        bias = self.interpolant.gamma(t) * self.obs_operator(bias)

        diff = interpolant_obs - (self.obs_operator(x) - bias)

        diff_norm = torch.linalg.norm(diff, dim=1) ** 2

        diff_norm = - 0.5 * diff_norm / interpolant_variance

        likelihood_score = torch.autograd.grad(
            outputs=diff_norm.sum(),
            inputs=x,
        )[0]

        if self.correct_likelihood_score:
            likelihood_score = self._compute_likelihood_score_correction(
                x=x,
                t=t,
                likelihood_score=likelihood_score
            )

        return likelihood_score, diff_norm
        # return likelihood_score * self.diffusion_term(t) ** 2, diff_norm

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

        diff_norm = torch.linalg.norm(observations - self.obs_operator(preds), dim=1) ** 2
        diff_norm = -diff_norm / (2 * self.original_variance)

        # Compute weights
        weights = torch.softmax(diff_norm.detach(), dim=0)

        # Compute weighted gradient
        score = torch.autograd.grad(
            outputs=(diff_norm * weights).sum(),
            inputs=x,
        )[0]

        return -score, weights
