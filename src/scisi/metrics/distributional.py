"""Distributional-fidelity metrics (spec Section 3c).

Two estimators shared between the analytical case (Case 1) and the field cases
(Cases 2-3):

* :func:`kl_at_points` -- 1-D marginal KL ``KL(sampled || reference)`` at a set
  of grid points, with observed/unobserved points reported separately. Mirrors
  the Gaussian/KDE approach in
  ``paper/scripts/analytical_utils/kl_divergence.py`` so both cases use one
  estimator.
* :func:`sliced_wasserstein_w2` -- average 2-Wasserstein distance over random
  1-D projections (squared-cost optimal transport; the analytical util provides
  W1 only).
"""

from typing import Optional

import numpy as np
import torch

__all__ = [
    "gaussian_kl_1d",
    "kde_kl_1d",
    "kl_at_points",
    "sliced_wasserstein_w2",
]


def _to_numpy_1d(x: torch.Tensor) -> np.ndarray:
    return np.asarray(x.detach().cpu().reshape(-1), dtype=np.float64)


def gaussian_kl_1d(
    sampled: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Closed-form 1-D Gaussian ``KL(sampled || reference)`` from samples.

    Fits a Gaussian to each sample set (empirical mean/std) and returns

        KL = log(s_r / s_p) + (s_p^2 + (m_p - m_r)^2) / (2 s_r^2) - 0.5,

    where ``(m_p, s_p)`` and ``(m_r, s_r)`` are the sampled / reference moments.

    Args:
        sampled: 1-D sample tensor from the approximation P.
        reference: 1-D sample tensor from the reference Q.
        eps: Floor on the standard deviations.

    Returns:
        The KL divergence as a Python float.
    """
    p = _to_numpy_1d(sampled)
    q = _to_numpy_1d(reference)
    m_p, s_p = float(p.mean()), max(float(p.std()), eps)
    m_q, s_q = float(q.mean()), max(float(q.std()), eps)
    return float(
        np.log(s_q / s_p)
        + (s_p**2 + (m_p - m_q) ** 2) / (2.0 * s_q**2)
        - 0.5
    )


def kde_kl_1d(
    sampled: torch.Tensor,
    reference: torch.Tensor,
    n_grid: int = 512,
    eps: float = 1e-12,
) -> float:
    """1-D ``KL(sampled || reference)`` via Gaussian KDE, integrated on a grid.

    Both marginals are estimated with a Gaussian kernel-density estimate
    (Scott's rule), then ``KL = integral p log(p / q)`` is evaluated by the
    trapezoidal rule over a grid spanning both sample sets. Use this when the
    marginals are visibly non-Gaussian; otherwise :func:`gaussian_kl_1d` is
    cheaper and lower-variance.

    Args:
        sampled: 1-D sample tensor from the approximation P.
        reference: 1-D sample tensor from the reference Q.
        n_grid: Number of integration grid points.
        eps: Density floor to keep the logarithm finite.

    Returns:
        The KL divergence as a Python float.
    """
    from scipy.stats import gaussian_kde

    p = _to_numpy_1d(sampled)
    q = _to_numpy_1d(reference)

    lo = min(p.min(), q.min())
    hi = max(p.max(), q.max())
    pad = 0.1 * (hi - lo + eps)
    grid = np.linspace(lo - pad, hi + pad, n_grid)

    p_density = np.clip(gaussian_kde(p)(grid), eps, None)
    q_density = np.clip(gaussian_kde(q)(grid), eps, None)

    integrand = p_density * np.log(p_density / q_density)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(max(trapezoid(integrand, grid), 0.0))


def kl_at_points(
    sampled: torch.Tensor,
    reference: torch.Tensor,
    observed_mask: Optional[torch.Tensor] = None,
    method: str = "gaussian",
) -> dict[str, float]:
    """1-D marginal KL at a set of grid points (spec Section 3c).

    For each point the marginal ``KL(sampled || reference)`` is estimated from
    the two ensembles, then averaged over all points and, if ``observed_mask``
    is given, split into observed / unobserved sub-averages.

    Args:
        sampled: Sampled marginals ``[E_s, P]`` (``E_s`` samples at ``P``
            points), or ``[E_s]`` for a single point.
        reference: Reference marginals ``[E_r, P]`` (large-``E`` reference
            ensemble or analytic samples), or ``[E_r]`` for a single point.
        observed_mask: Optional boolean tensor ``[P]`` where ``True`` marks an
            observed point. If given, ``"observed"`` and ``"unobserved"``
            averages are added to the result.
        method: ``"gaussian"`` (default, closed-form moment-matched KL) or
            ``"kde"`` (Gaussian-KDE estimate).

    Returns:
        Dict with ``"mean"`` (average over all points), ``"per_point"`` (a list
        of per-point KLs) and, when ``observed_mask`` is given, ``"observed"``
        and ``"unobserved"`` averages. NaN is returned for an empty subset.
    """
    if sampled.dim() == 1:
        sampled = sampled.unsqueeze(1)
    if reference.dim() == 1:
        reference = reference.unsqueeze(1)
    if sampled.shape[1] != reference.shape[1]:
        raise ValueError("`sampled` and `reference` must share point count P.")

    estimator = {"gaussian": gaussian_kl_1d, "kde": kde_kl_1d}.get(method)
    if estimator is None:
        raise ValueError(f"Unknown method '{method}'; use 'gaussian' or 'kde'.")

    n_points = sampled.shape[1]
    per_point = [
        estimator(sampled[:, i], reference[:, i]) for i in range(n_points)
    ]
    per_point_t = np.asarray(per_point, dtype=np.float64)

    result: dict[str, float] = {
        "mean": float(per_point_t.mean()),
        "per_point": per_point,  # type: ignore[dict-item]
    }
    if observed_mask is not None:
        obs = observed_mask.to(torch.bool).detach().cpu().numpy()
        result["observed"] = (
            float(per_point_t[obs].mean()) if obs.any() else float("nan")
        )
        result["unobserved"] = (
            float(per_point_t[~obs].mean()) if (~obs).any() else float("nan")
        )
    return result


def sliced_wasserstein_w2(
    sampled: torch.Tensor,
    reference: torch.Tensor,
    num_projections: int = 128,
    seed: int = 0,
) -> float:
    """Sliced 2-Wasserstein distance ``W2`` between two sample sets (spec 3c).

    Averages the (squared-cost) 2-Wasserstein distance over random 1-D
    projections. For each unit direction ``theta`` the projected 1-D ``W2`` is
    computed in closed form by sorting the (possibly unequal-size) sample sets
    onto a common quantile grid:

        W2_theta^2 = mean_q (sorted_p(q) - sorted_q(q))^2,

    and the reported value is ``sqrt(mean_theta W2_theta^2)``. With identical
    sample sets this is 0.

    Args:
        sampled: Sample tensor ``[N, d]`` (or ``[N]``) from distribution P.
        reference: Sample tensor ``[M, d]`` (or ``[M]``) from distribution Q.
        num_projections: Number of random projection directions.
        seed: Seed for the projection generator (deterministic estimate).

    Returns:
        The sliced ``W2`` distance as a Python float.
    """
    x = _to_numpy_2d(sampled)
    y = _to_numpy_2d(reference)
    if x.shape[1] != y.shape[1]:
        raise ValueError("`sampled` and `reference` must share dimension d.")
    d = x.shape[1]

    def _w2_1d(a: np.ndarray, b: np.ndarray) -> float:
        n = max(len(a), len(b))
        qs = (np.arange(n) + 0.5) / n
        qa = np.quantile(np.sort(a), qs)
        qb = np.quantile(np.sort(b), qs)
        return float(np.mean((qa - qb) ** 2))

    if d == 1:
        return float(np.sqrt(_w2_1d(x[:, 0], y[:, 0])))

    rng = np.random.default_rng(seed)
    projections = rng.normal(size=(num_projections, d))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    # ``errstate`` silences spurious BLAS-path matmul warnings under numpy 2.x;
    # the inputs are finite so the projected values are well-defined.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        x_proj = x @ projections.T
        y_proj = y @ projections.T
    sq = [_w2_1d(x_proj[:, i], y_proj[:, i]) for i in range(num_projections)]
    return float(np.sqrt(np.mean(sq)))


def _to_numpy_2d(x: torch.Tensor) -> np.ndarray:
    """Coerce a sample tensor to a 2-D ``[N, d]`` array (N samples, d dims)."""
    arr = np.asarray(x.detach().cpu(), dtype=np.float64)
    if arr.ndim <= 1:
        return arr.reshape(-1, 1)
    return arr.reshape(arr.shape[0], -1)
