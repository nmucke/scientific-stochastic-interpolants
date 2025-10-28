import pdb
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
from scipy.stats import gaussian_kde

from scisi.bin.analytical_utils.kl_divergence import kl_divergence
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


def prepare_samples(
    samples: torch.Tensor,
    nbins: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Prepare samples for KL-div and plotting."""

    samples = samples.numpy()
    xi, yi, zi = get_2d_kde(samples, nbins, x_range, y_range)

    diag_samples = np.diag(zi)
    return samples, (xi, yi, zi), diag_samples


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
    diffusion_term = lambda t: 2.0 * (1 - t)
    flow_das_diffusion_term = lambda t: 1 - t
    interpolant_diffusion_term = lambda t: 2.0 * torch.sqrt(1 - t)

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
    true_drift_model = AnalyticalDriftModel(
        interpolation, target_mean, target_cov, diffusion_term
    )
    flow_das_drift_model = AnalyticalDriftModel(
        interpolation, target_mean, target_cov, flow_das_diffusion_term
    )
    interpolant_drift_model = AnalyticalDriftModel(
        interpolation, target_mean, target_cov, interpolant_diffusion_term
    )
    true_stochastic_interpolant = AnalyticalStochasticInterpolant(
        interpolation, true_drift_model, diffusion_term
    )

    flow_das_likelihood_model = FlowdasLikelihood(
        obs_matrix, flow_das_drift_model, original_variance
    )
    flow_das_posterior_model = PosteriorModel(
        true_drift_model, flow_das_likelihood_model
    )

    interpolant_likelihood_model = InterpolantLikelihood(
        obs_matrix, interpolant_drift_model, original_variance
    )
    interpolant_posterior_model = PosteriorModel(
        interpolant_drift_model, interpolant_likelihood_model
    )

    x = x0.repeat(batch_size, 1)
    t = torch.ones(batch_size, 1)

    si_prior_samples = true_stochastic_interpolant.sample(x, num_steps=num_steps)
    si_prior_samples, (si_prior_xi, si_prior_yi, si_prior_zi), si_prior_diag = (
        prepare_samples(si_prior_samples, nbins, x_range, y_range)
    )

    flow_das_si_posterior_samples = flow_das_posterior_model.sample(
        x, num_steps=num_steps, observations=obs
    )
    (
        flow_das_si_posterior_samples,
        (flow_das_si_posterior_xi, flow_das_si_posterior_yi, flow_das_si_posterior_zi),
        flow_das_si_posterior_diag,
    ) = prepare_samples(flow_das_si_posterior_samples, nbins, x_range, y_range)

    interpolant_si_posterior_samples = interpolant_posterior_model.sample(
        x, num_steps=num_steps, observations=obs
    )
    (
        interpolant_si_posterior_samples,
        (
            interpolant_si_posterior_xi,
            interpolant_si_posterior_yi,
            interpolant_si_posterior_zi,
        ),
        interpolant_si_posterior_diag,
    ) = prepare_samples(interpolant_si_posterior_samples, nbins, x_range, y_range)

    (
        true_prior_samples,
        (true_prior_xi, true_prior_yi, true_prior_zi),
        true_prior_diag,
    ) = prepare_samples(true_prior_samples, nbins, x_range, y_range)

    likelihood_samples = likelihood_dist.sample((batch_size,))
    (
        likelihood_samples,
        (likelihood_xi, likelihood_yi, likelihood_zi),
        likelihood_diag,
    ) = prepare_samples(likelihood_samples, nbins, x_range, y_range)

    true_posterior_samples = true_posterior_dist.sample((batch_size,))
    (
        true_posterior_samples,
        (true_posterior_xi, true_posterior_yi, true_posterior_zi),
        true_posterior_diag,
    ) = prepare_samples(true_posterior_samples, nbins, x_range, y_range)

    flow_das_si_posterior_div = kl_divergence(
        true_posterior_samples, flow_das_si_posterior_samples
    )
    interpolant_si_posterior_div = kl_divergence(
        true_posterior_samples, interpolant_si_posterior_samples
    )

    print(f"FlowDAS SI Posterior KL-Divergence: {flow_das_si_posterior_div:0.4f}")
    print(
        f"Interpolant SI Posterior KL-Divergence: {interpolant_si_posterior_div:0.4f}"
    )

    # plot a density
    plt.figure(figsize=(20, 10))
    plt.subplot(3, 3, 1)
    plt.pcolormesh(si_prior_xi, si_prior_yi, si_prior_zi, shading="gouraud")
    plt.title("SI Prior")
    plt.subplot(3, 3, 2)
    plt.pcolormesh(true_prior_xi, true_prior_yi, true_prior_zi, shading="gouraud")
    plt.title("True prior")
    plt.subplot(3, 3, 3)
    plt.pcolormesh(
        flow_das_si_posterior_xi,
        flow_das_si_posterior_yi,
        flow_das_si_posterior_zi,
        shading="gouraud",
    )
    plt.title("FlowDAS SI Posterior")
    plt.subplot(3, 3, 4)
    plt.pcolormesh(
        interpolant_si_posterior_xi,
        interpolant_si_posterior_yi,
        interpolant_si_posterior_zi,
        shading="gouraud",
    )
    plt.title("Interpolant SI Posterior")
    plt.subplot(3, 3, 5)
    plt.pcolormesh(likelihood_xi, likelihood_yi, likelihood_zi, shading="gouraud")
    plt.title("Likelihood")
    plt.subplot(3, 3, 6)
    plt.pcolormesh(
        true_posterior_xi, true_posterior_yi, true_posterior_zi, shading="gouraud"
    )
    plt.title("True Posterior")
    plt.subplot(3, 3, 7)
    plt.plot(flow_das_si_posterior_diag, label="FlowDAS SI Posterior")
    plt.plot(interpolant_si_posterior_diag, label="Interpolant SI Posterior")
    plt.plot(true_prior_diag, label="True prior")
    plt.plot(true_posterior_diag, label="True Posterior")
    plt.plot(likelihood_diag, label="Likelihood")
    plt.legend()
    plt.title("Diagonal values")

    plt.show()


if __name__ == "__main__":
    main()
