"""Compare interpolant, true-pert, endpoint, and hybrid modes."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch

from paper_scripts.analytical_utils.kde_utils import prepare_samples
from paper_scripts.analytical_utils.kl_divergence import (
    kl_divergence,
    wasserstein_distance,
)
from paper_scripts.analytical_utils.likelihood import InterpolantLikelihood
from paper_scripts.analytical_utils.stochastic_interpolant import (
    AnalyticalDriftModel,
)
from paper_scripts.analytical_utils.true_posterior import get_true_posterior
from paper_scripts.analytical_utils.posterior_model import PosteriorModel
from scisi.models.interpolations import LinearStochasticInterpolation

torch.manual_seed(0)

DIM = 2
BATCH_SIZE = 1500
NUM_STEPS = 500
NBINS = 80
SAMPLE_ARGS = dict(nbins=NBINS, x_range=(-1, 7), y_range=(-1, 7))
ORIGINAL_VARIANCE_LIST = [0.5, 1.0, 2.0]
TARGET_MEAN = lambda x: x
TARGET_COV = lambda x: torch.eye(x.shape[1]).expand(
    x.shape[0], x.shape[1], x.shape[1]
)
DIFFUSION_TERM = lambda t: 1.0 * (1 - t)
INTERPOLATION = LinearStochasticInterpolation(wiener_process=True)
DRIFT = AnalyticalDriftModel(INTERPOLATION, TARGET_MEAN, TARGET_COV, DIFFUSION_TERM)

MODES = [None, "true", "endpoint", "M1", "M2", "M3", "M2M3"]


def main():
    OBS = torch.tensor([[1.0, 1.0]])
    OBS_MATRIX = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    x0 = torch.ones(1, DIM) * 5.0
    x_init = x0.repeat(BATCH_SIZE, 1)

    rows = []
    for sigma in ORIGINAL_VARIANCE_LIST:
        print(f"\n=== sigma^2 = {sigma} ===", flush=True)
        OBS_COV = torch.eye(DIM) * sigma
        _, _, true_dist = get_true_posterior(
            x0, TARGET_MEAN, TARGET_COV, OBS_MATRIX, OBS_COV, OBS
        )
        true_kde = prepare_samples(
            true_dist.sample((BATCH_SIZE,)), **SAMPLE_ARGS
        )
        true_s = true_kde.samples
        shared = dict(
            obs_matrix=OBS_MATRIX, drift_model=DRIFT, original_variance=sigma
        )
        for mode in MODES:
            lk = InterpolantLikelihood(
                **shared,
                perturbation=mode,
                target_variance=1.0,
                num_quad=500,
            )
            pm = PosteriorModel(DRIFT, lk)
            samps = pm.sample(x_init, num_steps=NUM_STEPS, observations=OBS)
            kde = prepare_samples(samps, **SAMPLE_ARGS)
            W = float(wasserstein_distance(true_s, kde.samples))
            KL = float(kl_divergence(true_s, kde.samples))
            rows.append((sigma, mode, W, KL))
            name = str(mode)
            print(f"  {name:10s}  W={W:.4f}  KL={KL:.4f}", flush=True)

    print("\nSummary")
    for sigma, mode, W, KL in rows:
        print(f"{sigma:<4g} {str(mode):10s} W={W:8.4f} KL={KL:8.4f}")


if __name__ == "__main__":
    main()
