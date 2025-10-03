import pdb
from functools import partial
from typing import Optional
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


class InterpolantMultivariateGaussianLikelihood(nn.Module):
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
        super(InterpolantMultivariateGaussianLikelihood, self).__init__()
        self.obs_operator = obs_operator
        self.model = model
        self.original_variance = variance
        self.ensemble_size = ensemble_size
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


        # field_history_mean = field_history[:, :, :, :, -1].mean(dim=0, keepdim=True)
        base_obs = self.obs_operator(field_history[:, :, :, :, -1])
        interpolant_obs = self.model.interpolation.forward(
            base_obs, observations, t , torch.zeros_like(base_obs)
        )

        # Compute the scale of the interpolant of the observation
        interpolant_variance = (
            self.model.interpolation.beta(t) ** 2
            * self.original_variance
        )
        interpolant_variance = interpolant_variance \
            + self.model.interpolation.gamma(t) ** 2 * (t)

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
            obs_diff_inner.unsqueeze(1), 
            obs_diff_inner.unsqueeze(2)
        ).squeeze()
        return torch.exp(-0.5 * obs_diff_inner / variance)

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
            obs_diff_inner.unsqueeze(1), 
            obs_diff_inner.unsqueeze(2)
        ).squeeze()
        return - 0.5 * obs_diff_inner / variance

    def _compute_likelihood_score(
        self,
        x: torch.Tensor,
        observations: torch.Tensor,
        variance: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the likelihood score."""
        log_likelihood = self._compute_log_likelihood(x, observations, variance)
        return torch.autograd.grad(log_likelihood.sum(), x, create_graph=True)[0]

    def score(
        self,
        observations: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
        dt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the likelihood score."""

        interpolant_obs, interpolant_variance = self._interpolate_observations(
            observations, x, t, field_history, field_cond, pars_cond
        )

        x.requires_grad = True

        return self._compute_likelihood_score(x, interpolant_obs, interpolant_variance)

    # def compute_weights(
    #     self,
    #     x: torch.Tensor,
    #     drift_model: nn.Module,
    #     diffusion_term: nn.Module,
    #     t: torch.Tensor,
    #     dt: torch.Tensor,
    #     observations: torch.Tensor,
    #     variance: torch.Tensor,
    # ) -> torch.Tensor:
    #     """Compute the weights."""

    #     x.requires_grad = True

    #     func = partial(
    #         self.compute_obs_diff_norm, 
    #         drift_model=drift_model, 
    #         diffusion_term=diffusion_term, 
    #         t=t, 
    #         dt=dt,
    #         observations=observations,
    #         variance=variance
    #     )

    #     pred = [func(x) for _ in range(self.ensemble_size)]

    #     forward_grad = [torch.autograd.grad(
    #         pred[i].sum(), 
    #         x, 
    #         create_graph=True
    #     )[0] for i in range(self.ensemble_size)]
    #     x.detach()


    #     weights = torch.stack(pred, dim=0)
    #     weights = weights / weights.sum(dim=0, keepdim=True)


    #     forward_grad = [weights[i] * forward_grad[i] for i in range(self.ensemble_size)]

    #     forward_grad = torch.stack(forward_grad, dim=0)
    #     forward_grad = forward_grad.mean(dim=0, keepdim=False)

    #     return forward_grad

    # def score(
    #     self, 
    #     x: torch.Tensor, 
    #     observations: torch.Tensor, 
    #     variance: torch.Tensor,
    #     drift_model: nn.Module,
    #     diffusion_term: nn.Module,
    #     t: torch.Tensor,
    #     dt: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    #     Score function.

    #     Args:
    #         x: Input tensor. [B, C, H, W]
    #         observations: Observations.
    #         variance: Variance.
    #     """

        # return torch.autograd.grad(self.forward(x, observations, variance).sum(), x, create_graph=True)[0]

        # x = x.repeat(self.ensemble_size, 1, 1, 1)

        # if len(x.shape) > 4:
        #     x = x.mean(dim=0, keepdim=False)

        # b, c, h, w = x.shape


        # pred = x + drift_model(x) * dt
        # # pred = pred.repeat(self.ensemble_size, 1, 1, 1, 1)
        # pred = pred + torch.sqrt(dt) * diffusion_term(t) * torch.randn_like(pred)

        # func = partial(
        #     self.compute_obs_diff_norm, 
        #     drift_model=drift_model, 
        #     diffusion_term=diffusion_term, 
        #     t=t, 
        #     dt=dt,
        #     observations=observations,
        #     variance=variance
        # )


        # x.requires_grad = True
        # forward_grad = torch.autograd.grad(
        #     func(x).sum(), 
        #     x, 
        #     create_graph=True
        # )[0]
        # x.detach()

        # forward_grad = self.compute_weights(x, drift_model, diffusion_term, t, dt, observations, variance)
        # # pdb.set_trace()

        # forward_grad = 0.5 / variance * forward_grad
        # return forward_grad #.mean(dim=0, keepdim=True)

        # pdb.set_trace()

        # obs_pred = self.compute_obs_diff_norm(x, drift_model, diffusion_term, t, dt, observations)
        # I_obs_cov_inv = 1 / variance
        # out = I_obs_cov_inv[0, 0] * obs_diff

        # # out = torch.matmul(self.obs_operator.obs_matrix.T, out)
        # out = out @ self.obs_operator.obs_matrix  # H.T * out in batched mode

        # out = torch.reshape(out, [b, c, h, w])
        # out = torch.reshape(out, [self.ensemble_size, c, h, w])

        # return out.mean(dim=0, keepdim=True)
