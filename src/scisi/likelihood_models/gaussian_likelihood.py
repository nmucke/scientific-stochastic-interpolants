import torch
import torch.nn as nn

from scisi.likelihood_models.observation_operators import LinearObservationOperator


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


class MultivariateGaussianLikelihood(nn.Module):
    """Multivariate Gaussian likelihood."""

    def __init__(
        self,
        obs_operator: nn.Module = LinearObservationOperator,
        variance: float = 0.05,
    ) -> None:
        """
        Initialize Multivariate Gaussian likelihood.

        Either covariance_matrix or precision_matrix must be provided.

        Args:
            obs_operator: Observation operator.
            variance: Variance.
        """
        super(MultivariateGaussianLikelihood, self).__init__()
        self.obs_operator = obs_operator
        self.original_variance = variance

        self.dist = torch.distributions.MultivariateNormal

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
        precision_matrix = (
            torch.eye(self.obs_operator.obs_indices.shape[0], device=variance.device)
            * 1
            / variance
        )

        dist = self.dist(loc=observations, precision_matrix=precision_matrix)

        return dist.log_prob(self.obs_operator(x))

    def score(
        self, x: torch.Tensor, observations: torch.Tensor, variance: torch.Tensor
    ) -> torch.Tensor:
        """
        Score function.

        Args:
            x: Input tensor. [B, C, H, W]
            observations: Observations.
            variance: Variance.
        """

        log_prob = self.forward(x, observations, variance)

        return torch.autograd.grad(log_prob.sum(), x, create_graph=True)[0]
