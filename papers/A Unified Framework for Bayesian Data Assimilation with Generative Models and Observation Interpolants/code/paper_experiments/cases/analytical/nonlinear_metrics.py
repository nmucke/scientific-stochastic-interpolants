"""Distributional metrics against the EXACT (quadrature) nonlinear posterior.

The Gaussian moment-KL used by the linear case (``gaussian_kl_divergence``) is only
a moment comparison; for a non-Gaussian target (arctan saturation, bimodal
``square``) it is wrong -- an exact sampler of a bimodal posterior would score a
non-zero "KL" against a Gaussian fitted to the same moments, and a Gaussian
sampler with the right moments would score zero. The metrics below use the exact
target instead. ``p`` = exact posterior (marginals ``p_i``, mean ``m_i``, variance
``v_i``), ``q`` = the method's ``m`` samples, ``d`` = dimension. All are
coordinate-wise (the exact posterior factorises), so none degrades with ``d``.

``mean_err`` -- RMS mean error in posterior-sd units,
    sqrt(mean_i (mhat_i - m_i)^2 / v_i). MC floor ~ 1/sqrt(m) (0.016 at m = 4096).

``var_err`` -- RMS relative variance error, sqrt(mean_i (vhat_i / v_i - 1)^2).
    MC floor ~ sqrt(2/m) for Gaussian marginals (0.022), up to ~0.04 when kurtotic.

``marginal_w2`` -- standardised per-coordinate W2 between the method's empirical
    marginal and the EXACT marginal (exact quantile function and exact partial
    moments, so no reference-sample noise), divided by the exact marginal sd and
    RMS-averaged over coordinates: sqrt(mean_i W2(q_i, p_i)^2 / v_i). For
    Gaussians it is sqrt(dmu^2 + (s - 1)^2) in sd units (dmu = mean shift / sd,
    s = sd ratio). Floor at m = 4096 (exact samples as q, 10 replicates): ~0.03
    (max 0.056) for unimodal marginals; for BIMODAL marginals it is dominated by
    the mode-mass sampling noise delta ~ sqrt(w(1-w)/m) (W2 ~ gap * sqrt(delta)),
    floor ~0.13 (max 0.21) -- read ``mode_weight_err`` there. This is the tuning criterion of the baselines.

``mode_weight_err`` -- non-injective g (square, abs) only: mean over coordinates
    of |q(x_i > s) - p(x_i > s)| at the symmetry point s = 0 of g.
    MC floor ~ sqrt(w(1-w)/m): ~0.005 (max 0.011).

``kl_marginal_knn`` -- the ONLY KL: mean over coordinates of the 1-D forward
    KL(p_i || q_i) = H(p_i, q_i) - H(p_i), exact H(p_i) by quadrature and the
    cross-entropy by the 1-D k-NN estimator (Wang, Kulkarni & Verdu 2009),
        H(p_i, q_i) ~= log 2 + psi(m) - psi(k) + mean_{x ~ p_i} log nu_k(x),
    on clouds standardised by the exact marginal mean / sd. A PER-COORDINATE KL
    (multiply by d for a product-form joint KL); blind to spurious
    inter-coordinate dependence in q. ESTIMATOR CAVEAT: the k-NN density of q
    decays only like 1/nu in q's tails, so where q UNDER-covers p (under-dispersed
    or shifted q) the estimate is biased LOW: at m = n = 4096, k = 5 it gives
    0.56 for a true 0.81 (q sd x 0.5) and 3.8 for a true 4.5 (3-sd shift). It is
    accurate (within ~0.01) when q covers p (sd x 2: 0.325 vs 0.318; 1-sd shift:
    0.50 vs 0.50). Read it together with ``mean_err`` / ``var_err`` /
    ``marginal_w2``, which have no such blind spot. Floor |KL| <~ 0.025 at
    m = 4096 (mean ~0.004).

``kl_points`` -- IDENTITY cell only: the linear table's legacy Gaussian
    moment-KL (``gaussian_kl_divergence``) against the linear driver's own
    Gaussian reference draw, so the identity control is directly comparable with
    the linear table (emitted with the legacy ``sliced_w2`` -- a sliced *W1*,
    scipy's ``wasserstein_distance``). Not emitted for nonlinear cells.

``unique_frac`` -- DIAGNOSTIC: fraction of distinct sample rows. Resampling
    methods (particle filter, SURGE) duplicate particles; a value << 1 flags
    particle degeneracy that the coordinate metrics only see indirectly (for the
    bootstrap PF it is a lower bound on ESS / m).

Dropped (review 2026-11): the joint forward k-NN KL (badly biased low for
under-dispersed / shifted q: 0.23 for a true 8.07 at d = 10, sd x 0.5), the
reverse k-NN KL (had to drop duplicates, so it could not see degeneracy),
``marginal_w1`` and the true sliced W2 (redundant with ``marginal_w2``).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln

from .nonlinear_reference import PointwiseNonlinearPosterior

NON_INJECTIVE = ("square", "abs")
METRICS = ("mean_err", "var_err", "marginal_w2", "mode_weight_err", "kl_marginal_knn", "unique_frac")


def _log_unit_ball(d: int) -> float:
    return 0.5 * d * math.log(math.pi) - float(gammaln(0.5 * d + 1.0))


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return x[np.all(np.isfinite(x), axis=1)]


def knn_cross_entropy(ref: np.ndarray, samples: np.ndarray, k: int = 5) -> float:
    """k-NN estimate of H(p, q) = -E_p log q (ref ~ p, samples ~ q)."""
    ref, samples = _finite(ref), _finite(samples)
    n, d = ref.shape
    m = samples.shape[0]
    if m <= k:
        return float("inf")
    nu = cKDTree(samples).query(ref, k=k)[0]
    nu = nu[:, -1] if nu.ndim == 2 else nu
    nu = np.maximum(nu, 1e-300)
    return float(_log_unit_ball(d) + digamma(m) - digamma(k) + d * np.mean(np.log(nu)))


def _standardise(post: PointwiseNonlinearPosterior, x: np.ndarray) -> np.ndarray:
    return (x - post.mean[None, :]) / np.sqrt(post.var)[None, :]


def kl_marginal_knn(post: PointwiseNonlinearPosterior, ref: np.ndarray, samples: np.ndarray, k: int = 5) -> float:
    """Mean over coordinates of the 1-D forward KL(p_i || q_i) (exact H(p_i)).

    See the module docstring for the low bias when q under-covers p.
    """
    z_r, z_q = _standardise(post, _finite(ref)), _standardise(post, _finite(samples))
    vals = []
    for i, mg in enumerate(post.marginals):
        h = knn_cross_entropy(z_r[:, i : i + 1], z_q[:, i : i + 1], k=k)
        vals.append(h - (mg.entropy - 0.5 * math.log(mg.var)))
    return float(np.mean(vals))


def nonlinear_metrics(
    post: PointwiseNonlinearPosterior,
    ref: np.ndarray,
    samples: np.ndarray,
    *,
    k: int = 5,
    split: Optional[float] = None,
) -> dict[str, float]:
    """The exact-target metrics for one draw (keys = tidy metric names).

    ``mode_weight_err`` only when ``split`` is given (non-injective g). A draw
    with any non-finite row scores NaN on every accuracy metric (the harness
    also counts it as a failure); ``unique_frac`` is still reported.
    """
    x = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    keys = [m for m in METRICS if m != "mode_weight_err" or split is not None]
    uniq = float(np.unique(x, axis=0).shape[0] / max(x.shape[0], 1))
    if not np.all(np.isfinite(x)) or x.shape[0] < 2:
        return {**{m: float("nan") for m in keys}, "unique_frac": uniq}
    m, v = post.mean, post.var
    out: dict[str, float] = {
        "mean_err": float(np.sqrt(np.mean((x.mean(0) - m) ** 2 / v))),
        "var_err": float(np.sqrt(np.mean((x.var(0, ddof=1) / v - 1.0) ** 2))),
        "marginal_w2": float(np.sqrt(np.mean(
            [post.marginal_w2(x[:, i], i) ** 2 / v[i] for i in range(post.d)]))),
    }
    if split is not None:
        exact = np.array([post.mass_above(split, i) for i in range(post.d)])
        out["mode_weight_err"] = float(np.mean(np.abs((x > split).mean(0) - exact)))
    out["kl_marginal_knn"] = kl_marginal_knn(post, ref, x, k=k)
    out["unique_frac"] = uniq
    return out


__all__ = [
    "METRICS",
    "NON_INJECTIVE",
    "kl_marginal_knn",
    "knn_cross_entropy",
    "nonlinear_metrics",
]
