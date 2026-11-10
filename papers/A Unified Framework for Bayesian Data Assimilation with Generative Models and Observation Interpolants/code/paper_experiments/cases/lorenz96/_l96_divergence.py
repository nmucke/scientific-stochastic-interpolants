"""Nonparametric distances of an ensemble to the particle-filter reference.

The posterior is not Gaussian, so the Gaussian KL of moments is not used here.

* :func:`knn_kl` -- the 1-nearest-neighbour KL estimator of Perez-Cruz (2008), as in
  the analytical case's archived ``analytical_utils.kl_divergence`` but with exact
  neighbours (no ``eps``) and without the clamp at ``1e-4``. Lorenz 96 reports
  ``KL(PF || ensemble)``: the 4096 stored PF draws are ``x``, the ensemble is ``y``.
  That direction stays finite when an ensemble holds duplicate members (SMC
  resampling), where ``KL(ensemble || PF)`` would divide by a zero distance. The PF
  draws themselves can hold duplicate rows (a filter that resamples after the kick,
  as the arctan reference and small references do): the estimator is computed on
  their UNIQUE rows, since a duplicate has a zero nearest-neighbour distance. With 64
  members in 16 dimensions the estimator is strongly biased: a perfect ensemble
  (``pf_subsample``) scores about 1.9, not 0. That floor is the same for every
  method at the same ``E``, so compare against it.
* :func:`energy_distance` -- ``2 E|X-Y| - E|X-X'| - E|Y-Y'|`` with within-sample
  U-statistics; zero in expectation for identical laws (it can dip below zero), and
  the multivariate generalisation of the CRPS.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial import cKDTree


def knn_kl(x: torch.Tensor, y: torch.Tensor) -> float:
    """``KL(P || Q)`` from ``x ~ P`` ``[n, d]`` and ``y ~ Q`` ``[m, d]`` (Perez-Cruz, 1-NN).

    Computed on the unique rows of ``x`` (first occurrences, in order, so draws
    without duplicates are used exactly as given; ``n`` counts them); NaN if fewer
    than two of them, or no point of ``y``.
    """
    x = np.asarray(x, dtype=np.float64)
    first = np.unique(x, axis=0, return_index=True)[1]
    if first.size < x.shape[0]:
        x = x[np.sort(first)]
    y = np.asarray(y, dtype=np.float64)
    n, d = x.shape
    m = y.shape[0]
    if n < 2 or m < 1:
        return math.nan
    r = cKDTree(x).query(x, k=2)[0][:, 1]
    if np.any(r <= 0):
        return math.nan
    s = cKDTree(y).query(x, k=1)[0]
    s = np.maximum(s, np.finfo(np.float64).tiny)
    return float(d * np.mean(np.log(s / r)) + np.log(m / (n - 1.0)))


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """Energy distance between samples ``x`` ``[n, d]`` and ``y`` ``[m, d]``."""
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    n, m = x.shape[0], y.shape[0]
    if n < 2 or m < 2:
        return math.nan
    xy = torch.cdist(x, y).mean()
    xx = torch.cdist(x, x).sum() / (n * (n - 1))
    yy = torch.cdist(y, y).sum() / (m * (m - 1))
    return float(2 * xy - xx - yy)
