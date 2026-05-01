# https://mail.python.org/pipermail/scipy-user/2011-May/029521.html

from typing import Any

import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.special import kl_div
from scipy.stats import wasserstein_distance as scipy_wasserstein_distance

import torch


def kl_divergence(x: np.ndarray, y: np.ndarray) -> Any:
    """
    Compute the Kullback-Leibler divergence between two multivariate samples.

    Parameters
    ----------
    x : 2D array (n,d)
      Samples from distribution P, which typically represents the true
      distribution.
    y : 2D array (m,d)
      Samples from distribution Q, which typically represents the approximate
      distribution.
    Returns
    -------
    out : float
      The estimated Kullback-Leibler divergence D(P||Q).
    References
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of
    continuous distributions IEEE International Symposium on Information
    Theory, 2008.
    """

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n, d = x.shape
    m, dy = y.shape

    assert d == dy

    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=0.01, p=2)[0][:, 1]
    s = ytree.query(x, k=1, eps=0.01, p=2)[0]

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    # return -np.log(r / s).sum() * d / n + np.log(m / (n - 1.0))
    return  np.log(s/r).sum() * d / n + np.log(m / (n - 1.))


def wasserstein_distance(
    x: np.ndarray,
    y: np.ndarray,
    num_projections: int = 128,
    seed: int = 0,
) -> Any:
    """
    Compute a sliced Wasserstein distance between two multivariate samples.

    Parameters
    ----------
    x : 2D array (n,d)
      Samples from distribution P, which typically represents the true
      distribution.
    y : 2D array (m,d)
      Samples from distribution Q, which typically represents the approximate
      distribution.
    num_projections : int
      Number of random 1D projections used to approximate the multivariate
      Wasserstein distance.
    seed : int
      Seed for the random projection generator to keep the estimate
      deterministic.

    Returns
    -------
    out : float
      The estimated sliced Wasserstein distance between the two empirical
      distributions.
    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    _, d = x.shape
    _, dy = y.shape

    assert d == dy

    if d == 1:
        return scipy_wasserstein_distance(x[:, 0], y[:, 0])

    rng = np.random.default_rng(seed)
    projections = rng.normal(size=(num_projections, d))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    x_proj = x @ projections.T
    y_proj = y @ projections.T

    distances = [
        scipy_wasserstein_distance(x_proj[:, i], y_proj[:, i])
        for i in range(num_projections)
    ]
    return float(np.mean(distances))
