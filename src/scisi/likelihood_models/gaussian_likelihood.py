import pdb
from functools import partial
from re import T
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from scisi.likelihood_models.observation_operators import LinearObservationOperator
from scisi.models.interpolations import (
    LinearDeterministicInterpolation,
    QuadraticDeterministicInterpolation,
)


def diffuse_mask(
    value_ids: torch.Tensor,
    A: float = 1,
    sig: float = 0.44,
    search_dist: int = -1,
    N: int = 256,
    tol: float = 1e-6,
) -> np.ndarray:
    """Diffuse mask."""
    L = 2 * np.pi
    dx = dy = L / N
    grid = np.zeros((N, N))

    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    def gauss(x0: float, y0: float, x: float, y: float) -> Any:
        """Gaussian function."""
        return A * np.exp(-((x0 - x) ** 2 + (y0 - y) ** 2) / (2 * sig**2))

    if search_dist < 0:
        min_search_steps = 0
        while gauss(0, 0, dx * min_search_steps, 0) > tol:
            min_search_steps += 1
        search_dist = min_search_steps

    gaussian = np.zeros((search_dist * 2 + 1, search_dist * 2 + 1))
    x0 = y0 = search_dist * dx
    for i in range(len(gaussian)):
        for j in range(len(gaussian)):
            gaussian[i, j] = gauss(x0, y0, i * dx, j * dx)

    for sid in value_ids:
        i = sid // N
        j = sid % N

        ilb = max(0, i - search_dist)
        iub = min(N, i + search_dist + 1)
        jlb = max(0, j - search_dist)
        jub = min(N, j + search_dist + 1)

        S = search_dist * 2 + 1

        if i - search_dist < 0:
            gilb = search_dist - i
            giub = S
        else:
            gilb = 0
            if i + search_dist > N - 1:
                giub = N - i + search_dist
            else:
                giub = S

        if j - search_dist < 0:
            gjlb = search_dist - j
            gjub = S
        else:
            gjlb = 0
            if j + search_dist > N - 1:
                gjub = N - j + search_dist
            else:
                gjub = S

        grid[ilb:iub, jlb:jub] = np.fmax(
            gaussian[gilb:giub, gjlb:gjub], grid[ilb:iub, jlb:jub]
        )

        grid[:, 0] = 0
        grid[:, -1] = 0
        grid[0, :] = 0
        grid[-1, :] = 0

    return grid


class GaussianLikelihood(nn.Module):
    """Gaussian likelihood."""

    def __init__(
        self,
        obs_operator: nn.Module = LinearObservationOperator,
        loc: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ) -> None:
        """
        Initialize Gaussian likelihood.

        Args:
            obs_operator: Observation operator.
            loc: Location.
            scale: Scale.
        """
        super(GaussianLikelihood, self).__init__()
        self.obs_operator = obs_operator
        self.original_loc = loc
        self.original_scale = scale

        self.dist = torch.distributions.Normal(loc=loc, scale=scale)

    def update_obs(self, obs: torch.Tensor) -> None:
        """
        Update the observation.
        """
        self.dist.loc = obs

    def update_scale(self, scale: torch.Tensor) -> None:
        """
        Update the scale.
        """
        self.dist.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. [B, C, H, W]

        Returns:
            torch.Tensor: Log probability.
        """
        return self.dist.log_prob(self.obs_operator(x)).mean(dim=1)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Score function.

        Args:
            x: Input tensor. [B, C, H, W]
        """

        x.requires_grad = True

        return torch.autograd.grad(self.forward(x).sum(), x, create_graph=True)[0]


class InterpolantGaussianLikelihood(nn.Module):
    """Interpolant Gaussian likelihood."""

    def __init__(
        self,
        model: nn.Module,
        obs_operator: nn.Module = LinearObservationOperator,
        variance: float = 0.05,
        ensemble_size: int = 1,
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
        self.dist = torch.distributions.MultivariateNormal

        mask = diffuse_mask(
            self.obs_operator.obs_indices,
            A=1,
            sig=0.05,
            search_dist=-1,
            N=128,
            tol=1e-6,
        )
        mask = torch.from_numpy(mask).to("cuda")
        self.mask = mask

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
        # precision_matrix = (
        #     torch.eye(self.obs_operator.obs_indices.shape[0], device=variance.device)
        #     * 1
        #     / variance
        # )

        # dist = self.dist(loc=observations, precision_matrix=precision_matrix)

        # return dist.log_prob(self.obs_operator(x))

    def _interpolate_observations(
        self,
        observations: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Interpolate the observations."""

        base_obs = field_history[:, :, :, :, -1]
        # base_obs = self.obs_operator(field_history[:, :, :, :, -1])
        interpolant_obs = (
            self.model.interpolation.alpha(t) * base_obs
            + self.model.interpolation.beta(t) * observations
        )

        # interpolant_obs = self.model.interpolation.forward(
        #     base_obs,
        #     observations,
        #     t,
        #     # torch.randn_like(base_obs),  # torch.zeros_like(x0_obs) #
        #     torch.zeros_like(base_obs) #
        # )

        # Compute the scale of the interpolant of the observation
        # interpolant_variance = self.interpolation.beta(t) ** 2 * self.original_variance + 1e-2
        interpolant_variance = (
            self.model.interpolation.beta(t) ** 2
            * self.original_variance
            # + self.model.interpolation.gamma(t) ** 2 * t
        )

        return interpolant_obs, interpolant_variance

    def _compute_one_step_prediction(
        self,
        x: torch.Tensor,
        drift_model: nn.Module,
        diffusion_term: nn.Module,
        t: torch.Tensor,
        dt: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the one step predictions."""
        drift = drift_model(x, t, field_history, field_cond, pars_cond, diffusion_term)
        diffusion = diffusion_term(t)
        return x + drift * dt + diffusion * torch.randn_like(x) * torch.sqrt(dt)

    def _compute_likelihood(
        self,
        x: torch.Tensor,
        observations: torch.Tensor,
        variance: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the likelihood."""
        x_obs = self.obs_operator(x)
        obs_diff_inner = observations - x_obs
        obs_diff_inner = torch.bmm(
            obs_diff_inner.unsqueeze(1), obs_diff_inner.unsqueeze(2)
        ).squeeze()
        return torch.exp(-0.5 * obs_diff_inner / variance) / torch.sqrt(
            2 * torch.pi * variance
        )

    def _compute_log_likelihood(
        self,
        x: torch.Tensor,
        observations: torch.Tensor,
        variance: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the log likelihood."""
        x_obs = self.obs_operator(x)
        obs_diff_inner = observations - x_obs
        obs_diff_inner = torch.bmm(
            obs_diff_inner.unsqueeze(1), obs_diff_inner.unsqueeze(2)
        ).squeeze()
        # obs_diff_inner = torch.linalg.norm(observations - self.obs_operator(x), dim=1)**2
        return -0.5 * obs_diff_inner / variance

    def _compute_likelihood_score(
        self,
        x: torch.Tensor,
        observations: torch.Tensor,
        variance: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the likelihood score."""
        # log_likelihood = self._compute_log_likelihood(x, observations, variance)
        # return torch.autograd.grad(log_likelihood.sum(), x, create_graph=True)[0]

        # x_obs = self.obs_operator(x)
        # obs_diff_inner = observations - x_obs
        # return - 0.5 * obs_diff_inner / variance

        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        obs_diff = observations - self.obs_operator(x)

        out = obs_diff @ self.obs_operator.obs_matrix  # H.T * obs_diff in batched mode

        out = out / variance

        return torch.reshape(out, [b, c, h, w])

    def likelihood_weights(
        self,
        observations: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the likelihood."""

        interpolant_obs, _ = self._interpolate_observations(
            observations, x, t, field_history, field_cond, pars_cond
        )

        likelihood = self._compute_likelihood(
            x=x,
            observations=interpolant_obs,
            variance=torch.tensor(1.0),  # interpolant_variance,
        )

        weights = likelihood / likelihood.sum()

        return weights

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
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        num_obs = observations.shape[1]

        observations = torch.reshape(
            observations,
            [
                1,
                c,
                torch.tensor(np.sqrt(num_obs), dtype=torch.int32),
                torch.tensor(np.sqrt(num_obs), dtype=torch.int32),
            ],
        )
        observations = nn.functional.interpolate(
            observations, size=(h, w), mode="bilinear"
        )

        interpolant_obs, interpolant_variance = self._interpolate_observations(
            observations, x, t, field_history, field_cond, pars_cond
        )

        var_mask = 1 / (interpolant_variance + 1e-3)
        var_mask = var_mask * self.mask
        var_mask = var_mask.reshape(1, 1, 128, 128)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(var_mask[0, 0].cpu().detach())
        # plt.colorbar()
        # plt.show()
        # pdb.set_trace()

        # likelihood_score = self._compute_likelihood_score(
        #     x, interpolant_obs, interpolant_variance
        # )
        # diff = interpolant_obs - self.obs_operator(x)
        diff = interpolant_obs - x
        diff_norm = (diff * diff * var_mask).sum(dim=-1)
        # diff_norm = (diff * diff / (interpolant_variance + 1e-3)).sum(dim=-1)
        diff_norm = -0.5 * diff_norm

        # diff_norm = -0.5 * diff_norm / (interpolant_variance + 1e-3)
        likelihood_score = torch.autograd.grad(diff_norm.sum(), x)[0]

        if diffusion_term is None:
            diffusion_term = self.model.interpolation.gamma

        return likelihood_score * dt * diffusion_term(t) ** 2 * 0.1
        # drift = (
        #     drift
        #     if drift is not None
        #     else self.model.drift_model(x, t, field_history, field_cond, pars_cond)
        # )
        # preds = x + drift * dt

        # # preds = preds.repeat(self.ensemble_size, 1, 1, 1)
        # preds = (
        #     preds + diffusion_term(t) * torch.randn_like(preds)
        # ) * torch.sqrt(dt)

        # interpolant_obs, interpolant_variance = self._interpolate_observations(
        #     observations, preds.detach(), t + dt, field_history, field_cond, pars_cond
        # )
        # interpolant_obs = interpolant_obs
        # interpolant_variance = interpolant_variance

        # log_likelihood = self._compute_log_likelihood(
        #     preds, interpolant_obs, interpolant_variance
        # )

        # # weights = torch.softmax(log_likelihood.detach(), dim=0)
        # # log_likelihood_score = torch.autograd.grad(
        # #     (log_likelihood * weights).sum(), x
        # # )[0]
        # log_likelihood_score = torch.autograd.grad(
        #     log_likelihood.sum(), x
        # )[0]
        # return diffusion_term(t) ** 2 * log_likelihood_score * dt


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

    def _compute_lam(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the lam."""

        gamma = self.model.interpolation.gamma(t)
        beta = self.model.interpolation.beta(t)
        gamma_diff = self.model.interpolation.gamma_diff(t)
        beta_diff = self.model.interpolation.beta_diff(t)

        return torch.sqrt(t) * (beta * gamma_diff - beta_diff * gamma)

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

        return 0.01 * score
