import torch
import torch.nn as nn

from scisi.likelihood_models.observation_operators import LinearObservationOperator


class Likelihood(nn.Module):
    """Gaussian likelihood."""

    def __init__(
        self,
        obs_operator: nn.Module = LinearObservationOperator,
        dist: torch.distributions.Distribution = torch.distributions.Normal,
        loc: torch.Tensor = None,
        scale: torch.Tensor = None,
    ) -> None:
        """
        Initialize likelihood.

        Args:
            dist: Distribution. Default is torch.distributions.Normal.
            dist_kwargs: Keyword arguments for the distribution.
            obs_operator: Observation operator.
        """
        super(Likelihood, self).__init__()
        self.obs_operator = obs_operator
        self.original_loc = loc
        self.original_scale = scale

        self.dist = dist(loc=loc, scale=scale)

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

        log_likelihood = self.forward(x)

        return torch.autograd.grad(log_likelihood.sum(), x, create_graph=True)[0]
