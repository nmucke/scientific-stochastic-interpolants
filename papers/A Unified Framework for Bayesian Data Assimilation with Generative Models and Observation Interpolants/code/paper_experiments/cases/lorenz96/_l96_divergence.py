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


# --------------------------------------------------------------------------- #
# Torch versions (GPU scoring). Same estimators, float64; they agree with the
# numpy/scipy versions above (and the library's sliced_wasserstein_w2) to rounding,
# not bitwise. Every Lorenz 96 score goes through these, so rows stay comparable.
# --------------------------------------------------------------------------- #


def scoring_device() -> torch.device:
    """CUDA when available (``SCISI_SCORING_DEVICE`` overrides), else CPU."""
    import os

    name = os.environ.get("SCISI_SCORING_DEVICE")
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


_PROJECTIONS: dict[tuple[int, int, int], np.ndarray] = {}


def _projections(num_projections: int, d: int, seed: int) -> np.ndarray:
    """The library's directions: ``default_rng(seed).normal`` rows, unit-normalised."""
    key = (num_projections, d, seed)
    if key not in _PROJECTIONS:
        p = np.random.default_rng(seed).normal(size=(num_projections, d))
        _PROJECTIONS[key] = p / np.linalg.norm(p, axis=1, keepdims=True)
    return _PROJECTIONS[key]


def _sorted_quantiles(s: torch.Tensor, n: int) -> torch.Tensor:
    """``np.quantile(s, (arange(n) + .5) / n, method='linear')`` per column of sorted ``s`` [m, k]."""
    m = s.shape[0]
    q = (torch.arange(n, dtype=torch.float64, device=s.device) + 0.5) / n
    virtual = m * q + (1.0 - q) - 1.0  # numpy's _compute_virtual_index, alpha = beta = 1
    prev = torch.floor(virtual)
    gamma = (virtual - prev).unsqueeze(1)
    lo = prev.long().clamp(0, m - 1)
    hi = (lo + 1).clamp(max=m - 1)
    above = virtual >= m - 1
    lo = torch.where(above, torch.full_like(lo, m - 1), lo)
    a, b = s[lo], s[hi]
    diff = b - a
    # numpy's _lerp: the form near the closer end point.
    return torch.where(gamma >= 0.5, b - diff * (1 - gamma), a + diff * gamma)


def sliced_w2_torch(
    sampled: torch.Tensor, reference: torch.Tensor, num_projections: int = 128, seed: int = 0
) -> float:
    """Torch :func:`scisi.metrics.distributional.sliced_wasserstein_w2` (same directions)."""
    x = sampled.reshape(sampled.shape[0], -1).to(torch.float64)
    y = reference.reshape(reference.shape[0], -1).to(device=x.device, dtype=torch.float64)
    d = x.shape[1]
    n = max(x.shape[0], y.shape[0])
    if d == 1:
        xp, yp = x, y
    else:
        p = torch.as_tensor(_projections(num_projections, d, seed), device=x.device)
        xp, yp = x @ p.T, y @ p.T
    qa = _sorted_quantiles(torch.sort(xp, dim=0).values, n)
    qb = _sorted_quantiles(torch.sort(yp, dim=0).values, n)
    return float(((qa - qb) ** 2).mean(0).mean().sqrt())


def _exact_cdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cdist(a, b, compute_mode="donot_use_mm_for_euclid_dist")


def knn_kl_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """Torch :func:`knn_kl`: ``KL(P || Q)`` by 1-NN on the unique rows of ``x``."""
    x = torch.unique(x.to(torch.float64), dim=0)
    y = y.to(device=x.device, dtype=torch.float64)
    n, d = x.shape
    m = y.shape[0]
    if n < 2 or m < 1:
        return math.nan
    xx = _exact_cdist(x, x)
    xx.fill_diagonal_(math.inf)
    r = xx.min(1).values
    if bool((r <= 0).any()):
        return math.nan
    s = _exact_cdist(x, y).min(1).values.clamp_min(np.finfo(np.float64).tiny)
    return float(d * torch.log(s / r).mean() + math.log(m / (n - 1.0)))


def energy_distance_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """Torch :func:`energy_distance` on the device of ``x``."""
    x = x.to(torch.float64)
    y = y.to(device=x.device, dtype=torch.float64)
    n, m = x.shape[0], y.shape[0]
    if n < 2 or m < 2:
        return math.nan
    xy = _exact_cdist(x, y).mean()
    xx = _exact_cdist(x, x).sum() / (n * (n - 1))
    yy = _exact_cdist(y, y).sum() / (m * (m - 1))
    return float(2 * xy - xx - yy)
