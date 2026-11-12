"""Sample-based KL and sliced-Wasserstein estimators of the analytical case.

Copied verbatim from the untracked ``archive/paper/scripts/analytical_utils/
kl_divergence.py`` (the copy the published single-step numbers used), so the case
no longer needs that directory on ``sys.path``. Only the two functions the case
calls are kept. ``tests/test_analytical_metrics.py`` pins their outputs to values
computed with the archived module, so moving them changes no number.
"""

from typing import Any

import numpy as np
from scipy.stats import wasserstein_distance as scipy_wasserstein_distance


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

    Each projection contributes a 1-D W1 distance (``scipy.stats``), so this is a
    sliced W1, not W2; see ``multistep.py``.

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
