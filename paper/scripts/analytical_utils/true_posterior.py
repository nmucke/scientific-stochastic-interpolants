from typing import Any, Callable, Optional

import torch


def kalman_gain(
    x0: torch.Tensor,
    target_cov: Callable,
    obs_matrix: torch.Tensor,
    obs_cov: torch.Tensor,
    observations: torch.Tensor,
) -> torch.Tensor:
    """Kalman gain."""
    target_cov = target_cov(x0).squeeze(0)
    K = target_cov @ obs_matrix.T
    return K @ torch.linalg.inv(obs_matrix @ target_cov @ obs_matrix.T + obs_cov)


def get_true_posterior(
    x0: torch.Tensor,
    target_mean: Callable,
    target_cov: Callable,
    obs_matrix: torch.Tensor,
    obs_cov: torch.Tensor,
    observations: torch.Tensor,
) -> torch.Tensor:
    """Get the true posterior."""
    K = kalman_gain(x0, target_cov, obs_matrix, obs_cov, observations)

    target_mean = target_mean(x0).squeeze(0)
    target_cov = target_cov(x0).squeeze(0)
    posterior_mean = target_mean + K @ (
        observations.squeeze(0) - obs_matrix @ target_mean
    )
    posterior_cov = (torch.eye(x0.shape[1]) - K @ obs_matrix) @ target_cov

    posterior_dist = torch.distributions.MultivariateNormal(
        posterior_mean, posterior_cov
    )
    return posterior_mean, posterior_cov, posterior_dist
