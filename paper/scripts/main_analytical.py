import sys
from pathlib import Path

# Add project root so `paper.scripts` is importable when run as script
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pdb
import matplotlib.pyplot as plt
import torch

from torch.distributions import MultivariateNormal 

from paper.scripts.analytical_utils.kde_utils import prepare_samples

from paper.scripts.analytical_utils.kl_divergence import kl_divergence
from paper.scripts.analytical_utils.likelihood import (
    FlowdasLikelihood,
    InterpolantLikelihood,
)
from paper.scripts.analytical_utils.stochastic_interpolant import (
    AnalyticalDriftModel,
    AnalyticalStochasticInterpolant,
)
from paper.scripts.analytical_utils.true_posterior import get_true_posterior
from scisi.models.interpolations import (
    LinearStochasticInterpolation,
    QuadraticStochasticInterpolation,
)
from paper.scripts.analytical_utils.posterior_model import PosteriorModel

# Domain
X_RANGE = (-1, 7)
Y_RANGE = (-1, 7)

# Grid
NBINS = 100
BATCH_SIZE = 5000
DIM = 2
SAMPLE_ARGS = {
    "nbins": NBINS,
    "x_range": X_RANGE,
    "y_range": Y_RANGE,
}

# Prior target
TARGET_MEAN = lambda x: x
TARGET_COV = lambda x: torch.eye(x.shape[1]).expand(
    x.shape[0], x.shape[1], x.shape[1]
)

# Stochastic Interpolant
NUM_STEPS = 1000
X0_MEAN = 5
DIFFUSION_TERM = lambda t: 1.0 * (1 - t)
INTERPOLATION = LinearStochasticInterpolation(wiener_process=True)
TRUE_DRIFT_MODEL = AnalyticalDriftModel(
    INTERPOLATION, TARGET_MEAN, TARGET_COV, DIFFUSION_TERM
)

TRUE_DRIFT_MODEL_1 = AnalyticalDriftModel(
    INTERPOLATION, TARGET_MEAN, TARGET_COV, lambda t: 1.0 * (1 - t)
)

# Likelihood
OBS = torch.tensor([[1.0, 1.0]])
ORIGINAL_VARIANCE = 1.0
OBS_COV = torch.eye(DIM) * ORIGINAL_VARIANCE
# OBS_MATRIX = torch.eye(DIM)
OBS_MATRIX = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

# Likelihood model
flowdas_args = {
    "obs_matrix": OBS_MATRIX,
    "drift_model": TRUE_DRIFT_MODEL,
    "original_variance": ORIGINAL_VARIANCE,
}
interpolant_args = {
    "obs_matrix": OBS_MATRIX,
    "drift_model": TRUE_DRIFT_MODEL,
    "original_variance": ORIGINAL_VARIANCE,
}


def main() -> None:
    """Main function."""

    likelihood_dist = MultivariateNormal(OBS[0], OBS_COV)

    x0 = torch.ones(1, DIM) * X0_MEAN
    true_prior_samples = MultivariateNormal(TARGET_MEAN(x0)[0], TARGET_COV(x0)[0, :, :]).sample((BATCH_SIZE,))

    _, _, true_posterior_dist = get_true_posterior(
        x0, TARGET_MEAN, TARGET_COV, OBS_MATRIX, OBS_COV, OBS
    )

    true_stochastic_interpolant = AnalyticalStochasticInterpolant(
        INTERPOLATION, TRUE_DRIFT_MODEL, DIFFUSION_TERM
    )

    flow_das_likelihood_model = FlowdasLikelihood( **flowdas_args)
    flow_das_posterior_model = PosteriorModel(
        TRUE_DRIFT_MODEL, flow_das_likelihood_model
    )

    interpolant_likelihood_model = InterpolantLikelihood(**interpolant_args)
    interpolant_posterior_model = PosteriorModel(
        TRUE_DRIFT_MODEL_1, interpolant_likelihood_model
    )

    x = x0.repeat(BATCH_SIZE, 1)

    si_prior = prepare_samples(
        true_stochastic_interpolant.sample(x, num_steps=NUM_STEPS),
        **SAMPLE_ARGS,
    )
    true_posterior = prepare_samples(
        true_posterior_dist.sample((BATCH_SIZE,)), **SAMPLE_ARGS
    )
    flow_das_si_posterior = prepare_samples(
        flow_das_posterior_model.sample(
            x, num_steps=NUM_STEPS, observations=OBS
        ), **SAMPLE_ARGS,
    )
    interpolant_si_posterior = prepare_samples(
        interpolant_posterior_model.sample(
            x, num_steps=NUM_STEPS, observations=OBS
        ), **SAMPLE_ARGS,
    )
    true_prior = prepare_samples(true_prior_samples, **SAMPLE_ARGS)
    likelihood = prepare_samples(
        likelihood_dist.sample((BATCH_SIZE,)), **SAMPLE_ARGS
    )

    flow_das_si_posterior_div = kl_divergence(
        true_posterior.samples, flow_das_si_posterior.samples
    )
    interpolant_si_posterior_div = kl_divergence(
        true_posterior.samples, interpolant_si_posterior.samples
    )

    print(f"FlowDAS SI Posterior KL-Divergence: {flow_das_si_posterior_div:0.4f}")
    print(
        f"Interpolant SI Posterior KL-Divergence: {interpolant_si_posterior_div:0.4f}"
    )

    # plot a density
    plt.figure(figsize=(20, 10))
    plt.subplot(3, 3, 1)
    plt.pcolormesh(si_prior.xi, si_prior.yi, si_prior.zi, shading="gouraud")
    plt.title("SI Prior")
    plt.subplot(3, 3, 2)
    plt.pcolormesh(true_prior.xi, true_prior.yi, true_prior.zi, shading="gouraud")
    plt.title("True prior")
    plt.subplot(3, 3, 3)
    plt.pcolormesh(
        flow_das_si_posterior.xi,
        flow_das_si_posterior.yi,
        flow_das_si_posterior.zi,
        shading="gouraud",
    )
    plt.title("FlowDAS SI Posterior")
    plt.subplot(3, 3, 4)
    plt.pcolormesh(
        interpolant_si_posterior.xi,
        interpolant_si_posterior.yi,
        interpolant_si_posterior.zi,
        shading="gouraud",
    )
    plt.title("Interpolant SI Posterior")
    plt.subplot(3, 3, 5)
    plt.pcolormesh(likelihood.xi, likelihood.yi, likelihood.zi, shading="gouraud")
    plt.title("Likelihood")
    plt.subplot(3, 3, 6)
    plt.pcolormesh(
        true_posterior.xi, true_posterior.yi, true_posterior.zi, shading="gouraud"
    )
    plt.title("True Posterior")
    plt.subplot(3, 3, 7)
    plt.plot(flow_das_si_posterior.diag, label="FlowDAS SI Posterior")
    plt.plot(interpolant_si_posterior.diag, label="Interpolant SI Posterior")
    plt.plot(true_prior.diag, label="True prior")
    plt.plot(true_posterior.diag, label="True Posterior")
    plt.plot(likelihood.diag, label="Likelihood")
    plt.legend()
    plt.title("Diagonal values")

    plt.show()


if __name__ == "__main__":
    main()
