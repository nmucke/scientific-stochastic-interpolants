import torch
import numpy as np
from scipy.stats import gaussian_kde

from dataclasses import dataclass
from typing import Callable, Optional


def get_2d_kde(
    samples: torch.Tensor,
    nbins: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> torch.Tensor:
    """Get the 2D KDE."""
    k = gaussian_kde(samples.T)
    xi, yi = np.mgrid[
        x_range[0] : x_range[1] : nbins * 1j, 
        y_range[0] : y_range[1] : nbins * 1j  # type: ignore[misc]
    ]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi = zi.reshape(xi.shape)
    return xi, yi, zi


def prepare_samples_raw(
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


@dataclass
class PreparedSamples:
    """Holds samples and KDE grid data for plotting and KL divergence."""

    samples: np.ndarray
    xi: np.ndarray
    yi: np.ndarray
    zi: np.ndarray
    diag: np.ndarray


def prepare_samples(
    samples: torch.Tensor,
    nbins: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> PreparedSamples:
    """Prepare samples for KL-div and plotting; returns PreparedSamples dataclass."""

    raw_samples, (xi, yi, zi), diag = prepare_samples_raw(
        samples, nbins, x_range, y_range
    )
    return PreparedSamples(samples=raw_samples, xi=xi, yi=yi, zi=zi, diag=diag)
