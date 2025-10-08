import pdb
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
from scipy.stats import gaussian_kde

from scisi.models.interpolations import LinearStochasticInterpolation


class AnalyticalDriftModel(nn.Module):
    """Analytical drift model."""

    def __init__(
        self,
        interpolation: nn.Module,
        target_mean: Callable,
        target_cov: Callable,
    ) -> None:
        """Initialize analytical drift model."""
        super(AnalyticalDriftModel, self).__init__()
        self.interpolation = interpolation
        self.target_mean = target_mean
        self.target_cov = target_cov

    def _get_coefs(self, t: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Get the coefficients."""
        alpha = self.interpolation.alpha(t)
        alpha_diff = self.interpolation.alpha_diff(t)
        beta = self.interpolation.beta(t)
        beta_diff = self.interpolation.beta_diff(t)
        gamma = self.interpolation.gamma(t)
        gamma_diff = self.interpolation.gamma_diff(t)
        return alpha, alpha_diff, beta, beta_diff, gamma, gamma_diff

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""

        I = torch.eye(x0.shape[1]).expand(x0.shape[0], x0.shape[1], x0.shape[1])

        alpha, alpha_diff, beta, beta_diff, gamma, gamma_diff = self._get_coefs(t)

        out = alpha_diff * x0 + beta_diff * self.target_mean(x0)

        if t.any() < 1e-8:
            return out

        mean_bar = alpha * x0 + beta * self.target_mean(x0)

        cov_bar = (
            beta.unsqueeze(1) ** 2 * self.target_cov(x0)
            + (t * gamma**2).unsqueeze(1) * I
        )
        cov_bar_inv = torch.linalg.inv(cov_bar)

        out_2 = beta.unsqueeze(1) * beta_diff * self.target_cov(x0)
        out_2 = out_2 + (t * gamma * gamma_diff).unsqueeze(1) * I
        out_2 = torch.bmm(out_2, cov_bar_inv)
        out_2 = torch.bmm(out_2, (x - mean_bar).unsqueeze(-1)).squeeze(-1)

        return out + out_2


class AnalyticalStochasticInterpolant(nn.Module):
    """Analytical stochastic interpolant."""

    def __init__(
        self,
        interpolation: nn.Module,
        drift_model: nn.Module,
    ) -> None:
        """Initialize analytical stochastic interpolant."""
        super(AnalyticalStochasticInterpolant, self).__init__()
        self.interpolation = interpolation
        self.drift_model = drift_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pass

    def sample(
        self,
        x0: torch.Tensor,
        num_steps: int,
        diffusion_term: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Sample from the analytical stochastic interpolant."""

        if diffusion_term is None:
            diffusion_term = self.drift_model.interpolation.gamma

        t_vec = torch.linspace(0, 1, num_steps).unsqueeze(0).repeat(x0.shape[0], 1)
        dt = torch.tensor(1 / num_steps)

        x = x0.clone()

        pbar = tqdm.tqdm(range(num_steps))
        for i in pbar:
            t = t_vec[:, i : i + 1]
            x = x + self.drift_model(x, t, x0) * dt
            x = x + diffusion_term(t) * torch.randn_like(x) * torch.sqrt(dt)
        return x


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
        interpolant_variance = (
            interpolant_variance + self.drift_model.interpolation.gamma(t) ** 2 * (t)
        )

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

        end_pred = self._compute_one_step_prediction(x, t, dt, x0)
        diff = observations - torch.matmul(end_pred, self.obs_matrix.T)
        diff_norm = (diff * diff).sum(dim=-1)
        diff_norm = -0.5 * diff_norm / self.original_variance
        weights = torch.softmax(diff_norm.detach(), dim=0)
        end_score = torch.autograd.grad((diff_norm * weights).sum(), x)[0].detach()

        x.requires_grad = True

        preds = x + self.drift_model(x, t, x0) * dt
        preds = preds.repeat(self.ensemble_size, 1, 1)
        preds = preds + diffusion_term(t) * torch.randn_like(preds) * torch.sqrt(dt)

        interpolant_obs, interpolant_variance = self._interpolate_observations(
            x, t + dt, x0, observations
        )

        diff = interpolant_obs - torch.matmul(preds, self.obs_matrix.T)
        diff_norm = (diff * diff).sum(dim=-1)
        diff_norm = -0.5 * diff_norm / interpolant_variance[0, 0]
        # score = torch.autograd.grad(diff_norm .sum(), x)[0].detach()
        weights = torch.softmax(diff_norm.detach(), dim=0)
        score = torch.autograd.grad((diff_norm * weights).sum(), x)[0]

        inner_score = (end_score * score).sum(dim=-1)
        inner_score = inner_score.mean(dim=0)

        norm_score = (score * score).sum(dim=-1)
        norm_score = norm_score.mean(dim=0)

        multiplier = inner_score / norm_score

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

        return torch.autograd.grad((diff_norm * weights).sum(), x)[0]


class PosteriorModel(nn.Module):
    """Posterior model."""

    def __init__(
        self,
        drift_model: nn.Module,
        likelihood_model: nn.Module,
    ) -> None:
        """Initialize posterior model."""
        super(PosteriorModel, self).__init__()
        self.drift_model = drift_model
        self.likelihood_model = likelihood_model

    def forward(
        self,
    ) -> None:
        """Forward pass."""
        pass

    def sample(
        self,
        x0: torch.Tensor,
        num_steps: int,
        observations: torch.Tensor,
        diffusion_term: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Sample from the posterior model."""
        if diffusion_term is None:
            diffusion_term = self.drift_model.interpolation.gamma

        t_vec = torch.linspace(0, 1, num_steps).unsqueeze(0).repeat(x0.shape[0], 1)
        dt = torch.tensor(1 / num_steps)

        x = x0.clone()

        pbar = tqdm.tqdm(range(num_steps - 1))
        for i in pbar:
            t = t_vec[:, i : i + 1]
            new = x + self.drift_model(x, t, x0) * dt
            new = new + diffusion_term(t) * torch.randn_like(x) * torch.sqrt(dt)

            if t.any() < 1e-6:
                x = new
                continue
            x.requires_grad = True

            score = self.likelihood_model.score(
                x, t, x0, observations, dt, diffusion_term
            )

            if isinstance(self.likelihood_model, InterpolantLikelihood):
                x = new + score * dt * diffusion_term(t) ** 2
            else:
                x = new + score * dt * 2.5
            x = x.detach()
        return x


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


def get_2d_kde(
    samples: torch.Tensor,
    nbins: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> torch.Tensor:
    """Get the 2D KDE."""
    k = gaussian_kde(samples.T)
    xi, yi = np.mgrid[
        x_range[0] : x_range[1] : nbins * 1j, y_range[0] : y_range[1] : nbins * 1j  # type: ignore[misc]
    ]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi = zi.reshape(xi.shape)
    return xi, yi, zi


def main() -> None:
    """Main function."""
    # Create test tensors
    nbins = 100
    x_range = (-1, 7)
    y_range = (-1, 7)

    original_variance = 1.0
    batch_size = 2500
    dim = 2
    num_steps = 25
    x0_mean = 5
    diffusion_term = lambda t: 1 - t

    obs = torch.tensor([[1.0, 1.0]])
    obs_cov = torch.eye(dim) * original_variance

    obs_matrix = torch.eye(dim)
    likelihood_dist = torch.distributions.MultivariateNormal(obs[0], obs_cov)

    target_mean = lambda x: x
    target_cov = lambda x: torch.eye(x.shape[1]).expand(
        x.shape[0], x.shape[1], x.shape[1]
    )

    x0 = torch.ones(1, dim) * x0_mean
    true_prior_samples = torch.distributions.MultivariateNormal(
        target_mean(x0)[0], target_cov(x0)[0, :, :]
    ).sample((batch_size,))

    _, _, true_posterior_dist = get_true_posterior(
        x0, target_mean, target_cov, obs_matrix, obs_cov, obs
    )

    # Initialize components
    interpolation = LinearStochasticInterpolation()
    drift_model = AnalyticalDriftModel(interpolation, target_mean, target_cov)
    stochastic_interpolant = AnalyticalStochasticInterpolant(interpolation, drift_model)

    # likelihood_model = FlowdasLikelihood(obs_matrix, drift_model, original_variance=original_variance)
    likelihood_model = InterpolantLikelihood(
        obs_matrix, drift_model, original_variance=original_variance
    )
    posterior_model = PosteriorModel(drift_model, likelihood_model)

    x = x0.repeat(batch_size, 1)
    t = torch.ones(batch_size, 1)

    si_prior_samples = stochastic_interpolant.sample(x, num_steps=num_steps)
    si_prior_samples = si_prior_samples.numpy()
    si_prior_xi, si_prior_yi, si_prior_zi = get_2d_kde(
        si_prior_samples, nbins, x_range, y_range
    )

    so_posterior_samples = posterior_model.sample(
        x, num_steps=num_steps, observations=obs
    )
    so_posterior_samples = so_posterior_samples.numpy()
    so_posterior_xi, so_posterior_yi, so_posterior_zi = get_2d_kde(
        so_posterior_samples, nbins, x_range, y_range
    )

    true_prior_samples = true_prior_samples.numpy()
    true_prior_xi, true_prior_yi, true_prior_zi = get_2d_kde(
        true_prior_samples, nbins, x_range, y_range
    )

    likelihood_samples = likelihood_dist.sample((batch_size,))
    likelihood_xi, likelihood_yi, likelihood_zi = get_2d_kde(
        likelihood_samples, nbins, x_range, y_range
    )

    true_posterior_samples = true_posterior_dist.sample((batch_size,))
    true_posterior_samples = true_posterior_samples.numpy()
    true_posterior_xi, true_posterior_yi, true_posterior_zi = get_2d_kde(
        true_posterior_samples, nbins, x_range, y_range
    )

    # Get diagonal values from density matrices
    si_prior_diag = np.diag(si_prior_zi)
    true_prior_diag = np.diag(true_prior_zi)
    true_posterior_diag = np.diag(true_posterior_zi)
    likelihood_diag = np.diag(likelihood_zi)
    so_posterior_diag = np.diag(so_posterior_zi)

    # plot a density
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 3, 1)
    plt.pcolormesh(si_prior_xi, si_prior_yi, si_prior_zi, shading="gouraud")
    plt.title("SI Prior")
    plt.subplot(2, 3, 2)
    plt.pcolormesh(true_prior_xi, true_prior_yi, true_prior_zi, shading="gouraud")
    plt.title("True prior")
    plt.subplot(2, 3, 3)
    plt.pcolormesh(so_posterior_xi, so_posterior_yi, so_posterior_zi, shading="gouraud")
    plt.title("SO Posterior")
    plt.subplot(2, 3, 4)
    plt.pcolormesh(likelihood_xi, likelihood_yi, likelihood_zi, shading="gouraud")
    plt.title("Likelihood")
    plt.subplot(2, 3, 5)
    plt.pcolormesh(
        true_posterior_xi, true_posterior_yi, true_posterior_zi, shading="gouraud"
    )
    plt.title("True Posterior")
    plt.subplot(2, 3, 6)
    plt.plot(si_prior_diag, label="SI Prior")
    plt.plot(true_prior_diag, label="True prior")
    plt.plot(true_posterior_diag, label="True Posterior")
    plt.plot(likelihood_diag, label="Likelihood")
    plt.plot(so_posterior_diag, label="SI Posterior")
    plt.legend()
    plt.title("Diagonal values")

    plt.show()


if __name__ == "__main__":
    main()
