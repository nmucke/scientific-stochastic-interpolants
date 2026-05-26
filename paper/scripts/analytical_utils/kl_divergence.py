# https://mail.python.org/pipermail/scipy-user/2011-May/029521.html

from typing import Any

import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.special import kl_div
from scipy.stats import wasserstein_distance as scipy_wasserstein_distance

import torch


def kl_divergence(x: np.ndarray, y: np.ndarray, safe: bool = True) -> Any:
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
    safe : bool
      If True (default), clamp the result at 0. The Pérez-Cruz kNN estimator
      is only asymptotically non-negative; for finite samples, especially
      when P and Q are very close, the log-distance ratio averages can dip
      below zero. Clamping makes the estimator a valid divergence at the
      cost of a small positive bias near zero. Set to False to inspect the
      raw estimator (useful as a sanity-check / unbiasedness diagnostic).
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
    estimate = float(np.log(s / r).sum() * d / n + np.log(m / (n - 1.0)))
    return max(estimate, 1e-4) if safe else estimate


def gaussian_kl_divergence(x: np.ndarray, y: np.ndarray) -> Any:
    """
    KL divergence between two distributions assumed Gaussian, from samples.

    Estimates D(P||Q) where P ~ N(mu_p, Sigma_p), Q ~ N(mu_q, Sigma_q) and
    the moments are taken from the empirical mean/covariance of ``x``/``y``.
    Closed form:

        2 D(P||Q) = tr(Sigma_q^{-1} Sigma_p)
                  + (mu_q - mu_p)^T Sigma_q^{-1} (mu_q - mu_p)
                  - d
                  + log(det(Sigma_q) / det(Sigma_p)).

    Parameters
    ----------
    x : 2D array (n, d)
      Samples from P (typically the true distribution).
    y : 2D array (m, d)
      Samples from Q (typically the approximation).
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    _, d = x.shape
    _, dy = y.shape
    assert d == dy

    mu_p = x.mean(axis=0)
    mu_q = y.mean(axis=0)
    sigma_p = np.cov(x, rowvar=False)
    sigma_q = np.cov(y, rowvar=False)
    if d == 1:
        sigma_p = np.atleast_2d(sigma_p)
        sigma_q = np.atleast_2d(sigma_q)

    sign_p, logdet_p = np.linalg.slogdet(sigma_p)
    sign_q, logdet_q = np.linalg.slogdet(sigma_q)
    if sign_p <= 0 or sign_q <= 0:
        raise ValueError("Sample covariance is not positive-definite.")

    sigma_q_inv = np.linalg.inv(sigma_q)
    diff = mu_q - mu_p
    trace_term = np.trace(sigma_q_inv @ sigma_p)
    quad_term = diff @ sigma_q_inv @ diff
    return 0.5 * float(trace_term + quad_term - d + logdet_q - logdet_p)


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
