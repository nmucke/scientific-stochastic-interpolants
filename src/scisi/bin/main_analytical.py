import pdb
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
from scipy.stats import gaussian_kde

from scisi.bin.analytical_utils.kl_divergence import KLdivergence
from scisi.bin.analytical_utils.likelihood import (
    FlowdasLikelihood,
    InterpolantLikelihood,
)
from scisi.bin.analytical_utils.stochastic_interpolant import (
    AnalyticalDriftModel,
    AnalyticalStochasticInterpolant,
)
from scisi.bin.analytical_utils.true_posterior import get_true_posterior
from scisi.models.interpolations import LinearStochasticInterpolation


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
            diffusion_term = self.drift_model.diffusion_term

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
                x = new + score
            x = x.detach()
        return x


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
    batch_size = 25000
    dim = 2
    num_steps = 500
    x0_mean = 5
    diffusion_term = lambda t: 2.0 * torch.sqrt(1 - t)
    # diffusion_term = lambda t: 2.0 * (1 - t)
    # diffusion_term = lambda t: 1 * (1 - t)

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
    drift_model = AnalyticalDriftModel(
        interpolation, target_mean, target_cov, diffusion_term
    )
    stochastic_interpolant = AnalyticalStochasticInterpolant(
        interpolation, drift_model, diffusion_term
    )

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

    si_posterior_samples = posterior_model.sample(
        x, num_steps=num_steps, observations=obs
    )
    si_posterior_samples = si_posterior_samples.numpy()
    si_posterior_xi, si_posterior_yi, si_posterior_zi = get_2d_kde(
        si_posterior_samples, nbins, x_range, y_range
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

    si_posterior_div = KLdivergence(true_posterior_samples, si_posterior_samples)

    print(f"KL-Divergence: {si_posterior_div:0.4f}")

    # Get diagonal values from density matrices
    si_prior_diag = np.diag(si_prior_zi)
    true_prior_diag = np.diag(true_prior_zi)
    true_posterior_diag = np.diag(true_posterior_zi)
    likelihood_diag = np.diag(likelihood_zi)
    si_posterior_diag = np.diag(si_posterior_zi)

    # plot a density
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 3, 1)
    plt.pcolormesh(si_prior_xi, si_prior_yi, si_prior_zi, shading="gouraud")
    plt.title("SI Prior")
    plt.subplot(2, 3, 2)
    plt.pcolormesh(true_prior_xi, true_prior_yi, true_prior_zi, shading="gouraud")
    plt.title("True prior")
    plt.subplot(2, 3, 3)
    plt.pcolormesh(si_posterior_xi, si_posterior_yi, si_posterior_zi, shading="gouraud")
    plt.title("SI Posterior")
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
    plt.plot(si_posterior_diag, label="SI Posterior")
    plt.legend()
    plt.title("Diagonal values")

    plt.show()


if __name__ == "__main__":
    main()
