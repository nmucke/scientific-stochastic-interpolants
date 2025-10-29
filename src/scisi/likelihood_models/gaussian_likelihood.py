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


def gaspari_cohn(r: np.ndarray) -> np.ndarray:
    """
    Gaspari-Cohn correlation function for covariance localization.

    This is a fifth-order piecewise polynomial function with compact support
    that smoothly tapers to zero, commonly used in ensemble data assimilation.

    Parameters:
    -----------
    r : float or np.ndarray
        Normalized distance(s), typically computed as d/R_l where:
        - d is the distance between two points
        - R_l is the localization radius (decorrelation length scale)

    Returns:
    --------
    float or np.ndarray
        Correlation value(s) in [0, 1]
        - Returns 1 at r=0 (perfect correlation)
        - Returns 0 for r >= 2 (zero correlation beyond cutoff)

    Reference:
    ----------
    Gaspari, G., and S. E. Cohn, 1999: Construction of correlation functions
    in two and three dimensions. Q. J. R. Meteorol. Soc., 125, 723–757.
    """
    r = np.asarray(r)
    rabs = np.abs(r)

    # Initialize output
    psi = np.zeros_like(rabs, dtype=float)

    # Region 1: 0 <= |r| <= 1
    mask1 = rabs <= 1
    r1 = rabs[mask1]
    psi[mask1] = (
        1.0
        - 5.0 / 3.0 * r1**2
        + 5.0 / 8.0 * r1**3
        + 1.0 / 2.0 * r1**4
        - 1.0 / 4.0 * r1**5
    )

    # Region 2: 1 < |r| <= 2
    mask2 = (rabs > 1) & (rabs <= 2)
    r2 = rabs[mask2]
    psi[mask2] = (
        4.0
        - 5.0 * r2
        + 5.0 / 3.0 * r2**2
        + 5.0 / 8.0 * r2**3
        - 1.0 / 2.0 * r2**4
        + 1.0 / 12.0 * r2**5
        - 2.0 / (3.0 * r2)
    )

    # Region 3: |r| > 2 (already initialized to zero)

    return psi if r.shape else float(psi)


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
            + self.diffusion_term(t) * torch.randn_like(base_obs) * torch.sqrt(t)
        )

        # Compute the scale of the interpolant of the observation
        interpolant_variance = (
            self.interpolant.beta(t) ** 2
            * self.original_variance
            # + self.diffusion_term(t) ** 2 * t
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
        x_obs: torch.Tensor,
        observations: torch.Tensor,
        variance: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the likelihood."""
        obs_diff_inner = observations - x_obs
        obs_diff_inner = torch.bmm(
            obs_diff_inner.unsqueeze(1), obs_diff_inner.unsqueeze(2)
        ).squeeze()
        return torch.exp(-0.5 * obs_diff_inner / variance) / torch.sqrt(
            2 * torch.pi * variance
        )

    def _compute_log_likelihood(
        self,
        x_obs: torch.Tensor,
        observations: torch.Tensor,
        variance: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the log likelihood."""
        obs_diff_inner = observations - x_obs
        obs_diff_inner = torch.bmm(
            obs_diff_inner.unsqueeze(1), obs_diff_inner.unsqueeze(2)
        ).squeeze()
        # obs_diff_inner = torch.linalg.norm(observations - self.obs_operator(x), dim=1)**2
        return -0.5 * obs_diff_inner / variance

    def _compute_likelihood_score(
        self,
        x_obs: torch.Tensor,
        observations: torch.Tensor,
        variance: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the likelihood score."""

        obs_diff = observations - x_obs

        out = obs_diff @ self.obs_operator.obs_matrix  # H.T * obs_diff in batched mode

        return out / (variance + 1e-3)

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

        if diffusion_term is None:
            self.diffusion_term = self.model.interpolation.gamma
        else:
            self.diffusion_term = diffusion_term

        interpolant_obs, interpolant_variance = self._interpolate_observations(
            observations, x, t, self.obs_operator(field_history[..., -1])
        )

        x_obs = self.obs_operator(x)
        likelihood_score = self._compute_likelihood_score(
            x_obs, interpolant_obs, interpolant_variance
        )
        likelihood_score = likelihood_score.reshape(b, c, h, w)

        return likelihood_score * dt * diffusion_term(t) ** 2  # type: ignore[misc]


class SpatialInterpolantGaussianLikelihood(InterpolantGaussianLikelihood):
    """Spatial Interpolant Gaussian likelihood."""

    def __init__(
        self,
        model: nn.Module,
        obs_operator: nn.Module = LinearObservationOperator,
        variance: float = 0.05,
        ensemble_size: int = 1,
        spatial_sigma: float = 0.05,
        interpolant: Optional[nn.Module] = None,
    ) -> None:
        """Initialize Spatial Interpolant Gaussian likelihood."""
        super(SpatialInterpolantGaussianLikelihood, self).__init__(
            model, obs_operator, variance, ensemble_size, interpolant
        )
        self.obs_operator = obs_operator
        self.model = model
        self.original_variance = variance

        self.mask = diffuse_mask(
            self.obs_operator.obs_indices,
            A=1,
            sig=spatial_sigma,
            search_dist=-1,
            N=self.obs_operator.H,
            tol=1e-6,
        )
        self.mask = torch.tensor(self.mask, dtype=torch.float32).to("cuda")
        self.mask = self.mask.unsqueeze(0).unsqueeze(0)

    def _compute_likelihood_score(
        self,
        x_obs: torch.Tensor,
        observations: torch.Tensor,
        variance: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the likelihood score."""

        likelihood_score = observations - x_obs
        likelihood_score = likelihood_score * self.mask

        return likelihood_score / (variance + 1e-3)

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

        if diffusion_term is None:
            self.diffusion_term = self.model.interpolation.gamma
        else:
            self.diffusion_term = diffusion_term

        sqrt_num_obs = torch.tensor(np.sqrt(observations.shape[1]), dtype=torch.int32)
        observations = torch.reshape(
            observations,
            [1, c, sqrt_num_obs, sqrt_num_obs],
        )
        observations = nn.functional.interpolate(
            observations, size=(h, w), mode="nearest"
        )

        interpolant_obs, interpolant_variance = self._interpolate_observations(
            observations, x, t, field_history[..., -1]
        )

        likelihood_score = self._compute_likelihood_score(
            x, interpolant_obs, interpolant_variance
        )

        return (
            likelihood_score * dt * diffusion_term(t) ** 2  # type: ignore[misc]
        )  # self.interpolant.beta(t)


class KalmanInterpolantGaussianLikelihood(nn.Module):
    """Spatial Interpolant Gaussian likelihood."""

    def __init__(
        self,
        model: nn.Module,
        obs_operator: nn.Module = LinearObservationOperator,
        variance: float = 0.05,
        ensemble_size: int = 1,
        interpolant: Optional[nn.Module] = None,
    ) -> None:
        """Initialize Spatial Interpolant Gaussian likelihood."""
        super(KalmanInterpolantGaussianLikelihood, self).__init__()
        self.obs_operator = obs_operator
        self.model = model
        self.original_variance = variance
        self.original_std = torch.sqrt(torch.tensor(variance, dtype=torch.float32))

        if interpolant is not None:
            self.interpolant = interpolant
        else:
            self.interpolant = self.model.interpolation

        # Create a 2D grid of points
        x = torch.linspace(0, 2 * torch.pi, 128)
        y = torch.linspace(0, 2 * torch.pi, 128)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        # Flatten the grid to a list of coordinates
        coords = torch.stack([X.flatten(), Y.flatten()], dim=-1)

        # Compute the pairwise distance matrix
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        self.dist_matrix = torch.sqrt(torch.sum(diff**2, dim=-1))
        self.dist_matrix = self.dist_matrix.to("cuda")

    def forward(
        self,
    ) -> torch.Tensor:
        """Forward pass."""
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
        )

        # Compute the scale of the interpolant of the observation
        interpolant_variance = (
            self.interpolant.beta(t) ** 2
            * self.original_variance
            # + self.model.interpolation.gamma(t) ** 2 * t
        )

        return interpolant_obs, interpolant_variance

    def _compute_kalman_gain(
        self,
        obs_diff: torch.Tensor,
        x_cov: torch.Tensor,
        gain_matrix: torch.Tensor,
        pivots: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the kalman gain."""

        # gain_matrix = torch.linalg.lu_solve(gain_matrix, pivots, obs_diff)
        gain_matrix = torch.linalg.solve(gain_matrix, obs_diff)

        return x_cov @ self.obs_operator.obs_matrix.t() @ gain_matrix

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

        interpolant_obs, interpolant_variance = self._interpolate_observations(
            observations, x, t, self.obs_operator(field_history[..., -1])
        )

        x = x.reshape(b, c * h * w)

        obs_diff = (
            interpolant_obs
            - self.obs_operator(x)
            + torch.randn_like(interpolant_obs) * torch.sqrt(interpolant_variance)
        )

        x_cov = torch.cov(x.detach().t())
        x_cov = x_cov * self.dist_matrix

        obs_cov = (
            torch.eye(interpolant_obs.shape[1], device=x.device) * interpolant_variance
        )
        kalman_gain = (
            self.obs_operator.obs_matrix @ x_cov @ self.obs_operator.obs_matrix.t()
        )
        kalman_gain = kalman_gain + obs_cov

        # kalman_gain, pivots = torch.linalg.lu_factor(kalman_gain)

        kalman_gain_vmap = torch.vmap(
            partial(
                self._compute_kalman_gain,
                x_cov=x_cov,
                gain_matrix=kalman_gain,
                pivots=0,
            )
        )

        kalman_gain = kalman_gain_vmap(obs_diff)

        return kalman_gain.reshape(b, c, h, w)


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
