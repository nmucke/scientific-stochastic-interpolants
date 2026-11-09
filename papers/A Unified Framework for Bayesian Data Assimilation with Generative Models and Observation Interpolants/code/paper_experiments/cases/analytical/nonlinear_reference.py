"""Exact posterior for the one-step Gaussian case with a pointwise NONLINEAR observation.

System (per coordinate ``i = 1..d``, all independent):

    x1 = x0 + w,         w ~ N(0, prior_var I)
    y  = g(x1) + e,      e ~ N(0, obs_var I),     g elementwise

with ``g`` one of ``scisi.likelihood_models.observation_operators.NONLINEARITIES``
(the same ``_g_and_dg`` the ``PointwiseObservationOperator`` uses, so the reference
and the operator handed to the samplers can never disagree). The posterior
factorises over coordinates,

    p(x_i | y_i)  ∝  N(x_i; x0_i, prior_var) N(y_i; g(x_i), obs_var),

so each marginal is a 1-D density that we integrate EXACTLY (to quadrature
precision) on an adaptive fine grid:

1. A coarse grid over ``x0_i ± 15 sqrt(prior_var)`` (widened to include every
   preimage of ``y_i`` far in the prior tail, then doubled until the kept region
   is interior; ``RuntimeError`` after 20 doublings) locates every cell whose
   log-density is within ``LOG_CUT`` (=60 nats) of the global maximum -- this
   captures all modes, e.g. both branches ``±sqrt(y/c)`` of ``g = c z^2``.
2. A uniform fine grid (``n_fine`` = 200 001 points by default) over the
   bounding interval of those cells; trapezoidal quadrature gives ``Z``, the
   moments, the CDF and the partial moments ``int^x t^k p(t) dt`` (k = 1, 2).

The mass outside the fine interval is below ``exp(-LOG_CUT)`` relative to the
mode, and the fine spacing is ``<= 1e-4`` for every case in the sweep (posterior
widths are ``>= 3e-3``), so the trapezoidal error is far below anything a
sampler-metric can resolve. Validation (``tests/test_nonlinear_reference.py``):
identity ``g`` reproduces the closed-form Kalman posterior mean / variance /
entropy / log-density to the tested ``1e-8``; a 1e6-particle self-normalised
importance sampler (bootstrap PF) agrees for arctan / cubic / square within 5
Monte-Carlo standard errors.

Provides: :meth:`log_prob` (normalised joint log-density), :meth:`mean` /
:meth:`var` / :meth:`entropy`, exact sampling by inverse-CDF interpolation
(:meth:`sample`), quantile functions, and the per-marginal W2 between an
empirical sample and the exact marginal (:meth:`marginal_w2`) computed with the
exact quantile function (no reference-sample noise).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from scisi.likelihood_models.observation_operators import _g_and_dg

LOG_CUT = 60.0  # nats below the mode kept in the fine grid
_MAX_GROW = 20  # window doublings before _build gives up (raises)
_LOG_2PI = math.log(2.0 * math.pi)


def g_numpy(name: str, z: np.ndarray, strength: float = 1.0, eps: float = 1e-3) -> np.ndarray:
    """Elementwise ``g`` in float64, via the operator's own ``_g_and_dg``."""
    t = torch.as_tensor(np.asarray(z, dtype=np.float64))
    return _g_and_dg(name, t, float(strength), float(eps))[0].numpy()


def _cumtrapz(f: np.ndarray, h: float) -> np.ndarray:
    out = np.empty_like(f)
    out[0] = 0.0
    out[1:] = np.cumsum(0.5 * (f[1:] + f[:-1])) * h
    return out


@dataclass
class _Marginal:
    grid: np.ndarray  # [n] uniform
    h: float
    logZ: float  # log of int prior*lik (including Gaussian normalisers)
    pdf: np.ndarray  # normalised density on grid
    cdf: np.ndarray  # normalised CDF on grid (cdf[-1] == 1)
    m1: np.ndarray  # int_{lo}^{x} t p(t) dt
    m2: np.ndarray  # int_{lo}^{x} t^2 p(t) dt
    mean: float
    var: float
    entropy: float


class PointwiseNonlinearPosterior:
    """Exact (quadrature) posterior of ``x1 | x0, y`` for ``y = g(x1) + e``.

    Args:
        x0: Prior mean ``[d]`` (the previous state).
        y: Observation ``[d]``.
        prior_var: ``Cov(x1 | x0) = prior_var I``.
        obs_var: ``R = obs_var I``.
        nonlinearity, strength, eps: ``g`` (see ``PointwiseObservationOperator``).
        n_fine: fine-grid points per coordinate.
    """

    def __init__(
        self,
        x0,
        y,
        *,
        prior_var: float = 1.0,
        obs_var: float = 1.0,
        nonlinearity: str = "identity",
        strength: float = 1.0,
        eps: float = 1e-3,
        n_fine: int = 200_001,
        n_coarse: int = 40_001,
    ) -> None:
        self.x0 = np.asarray(torch.as_tensor(x0).detach().cpu().double().numpy()).reshape(-1)
        self.y = np.asarray(torch.as_tensor(y).detach().cpu().double().numpy()).reshape(-1)
        assert self.x0.shape == self.y.shape
        self.d = self.x0.size
        self.prior_var = float(prior_var)
        self.obs_var = float(obs_var)
        self.nonlinearity = nonlinearity
        self.strength = float(strength)
        self.eps = float(eps)
        self.n_fine = int(n_fine)
        self.n_coarse = int(n_coarse)
        self.marginals = [self._build(i) for i in range(self.d)]

    # ------------------------------------------------------------------ #
    def g(self, z: np.ndarray) -> np.ndarray:
        return g_numpy(self.nonlinearity, z, self.strength, self.eps)

    def _log_unnorm(self, x: np.ndarray, i: int) -> np.ndarray:
        """log N(x; x0_i, pv) + log N(y_i; g(x), R) (with normalisers)."""
        lp = -0.5 * (x - self.x0[i]) ** 2 / self.prior_var - 0.5 * (_LOG_2PI + math.log(self.prior_var))
        ll = -0.5 * (self.y[i] - self.g(x)) ** 2 / self.obs_var - 0.5 * (_LOG_2PI + math.log(self.obs_var))
        return lp + ll

    def _build(self, i: int) -> _Marginal:
        sp = math.sqrt(self.prior_var)
        lo, hi = self.x0[i] - 15.0 * sp, self.x0[i] + 15.0 * sp
        # Widen to any far-tail preimage of y (when y is far out, the posterior can
        # sit in the prior's tail). Bracket via a very wide coarse scan.
        wide = np.linspace(self.x0[i] - 1e3 * sp, self.x0[i] + 1e3 * sp, 200_001)
        with np.errstate(over="ignore", invalid="ignore"):
            lw = np.nan_to_num(self._log_unnorm(wide, i), nan=-np.inf, neginf=-np.inf)
        jw = int(np.argmax(lw))
        lo, hi = min(lo, wide[max(jw - 1, 0)]), max(hi, wide[min(jw + 1, wide.size - 1)])
        # Grow the window until the kept (>= max - LOG_CUT) region is interior.
        for _ in range(_MAX_GROW):
            coarse = np.linspace(lo, hi, self.n_coarse)
            with np.errstate(over="ignore", invalid="ignore"):
                lc = np.nan_to_num(self._log_unnorm(coarse, i), nan=-np.inf)
            keep = np.nonzero(lc > lc.max() - LOG_CUT)[0]
            grow_lo, grow_hi = keep[0] <= 2, keep[-1] >= coarse.size - 3
            if not (grow_lo or grow_hi):
                break
            w = hi - lo
            lo, hi = lo - w * grow_lo, hi + w * grow_hi
        else:
            raise RuntimeError(
                f"coordinate {i}: the kept (>= max - {LOG_CUT:g} nats) region still "
                f"touches the window edge after {_MAX_GROW} doublings ([{lo:.3g}, "
                f"{hi:.3g}]); the posterior is not resolvable on a finite grid "
                f"(nonlinearity={self.nonlinearity!r}, y={self.y[i]:.4g})."
            )
        hc = coarse[1] - coarse[0]
        a = coarse[max(keep[0] - 2, 0)] - 2 * hc
        b = coarse[min(keep[-1] + 2, coarse.size - 1)] + 2 * hc
        grid = np.linspace(a, b, self.n_fine)
        h = float(grid[1] - grid[0])
        lu = self._log_unnorm(grid, i)
        mx = lu.max()
        f = np.exp(lu - mx)
        Zs = float(np.trapezoid(f, dx=h)) if hasattr(np, "trapezoid") else float(np.trapz(f, dx=h))
        pdf = f / Zs
        cdf = _cumtrapz(pdf, h)
        tot = cdf[-1]
        pdf, cdf = pdf / tot, cdf / tot
        m1 = _cumtrapz(grid * pdf, h)
        m2 = _cumtrapz(grid**2 * pdf, h)
        mean = float(m1[-1])
        var = float(m2[-1] - mean**2)
        logp = lu - mx - math.log(Zs * tot)
        ent = float(-np.sum(0.5 * (pdf[1:] * logp[1:] + pdf[:-1] * logp[:-1])) * h)
        return _Marginal(grid, h, mx + math.log(Zs * tot), pdf, cdf, m1, m2, mean, var, ent)

    # ------------------------------------------------------------------ #
    @property
    def mean(self) -> np.ndarray:
        return np.array([m.mean for m in self.marginals])

    @property
    def var(self) -> np.ndarray:
        return np.array([m.var for m in self.marginals])

    @property
    def entropy(self) -> float:
        """Exact differential entropy of the joint posterior (sum of marginals)."""
        return float(sum(m.entropy for m in self.marginals))

    @property
    def log_evidence(self) -> float:
        return float(sum(m.logZ for m in self.marginals))

    def log_prob(self, x) -> np.ndarray:
        """Normalised joint posterior log-density at ``x [n, d]`` -> ``[n]``."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        out = np.zeros(x.shape[0])
        for i, m in enumerate(self.marginals):
            out += self._log_unnorm(x[:, i], i) - m.logZ
        return out

    def marginal_log_prob(self, x: np.ndarray, i: int) -> np.ndarray:
        return self._log_unnorm(np.asarray(x, dtype=np.float64), i) - self.marginals[i].logZ

    def cdf(self, x: np.ndarray, i: int) -> np.ndarray:
        m = self.marginals[i]
        return np.interp(x, m.grid, m.cdf, left=0.0, right=1.0)

    def quantile(self, u: np.ndarray, i: int) -> np.ndarray:
        """Exact quantile ``F_i^{-1}(u)`` (CDF is strictly increasing on the grid)."""
        m = self.marginals[i]
        return np.interp(u, m.cdf, m.grid)

    def mass_above(self, split: float, i: int) -> float:
        return float(1.0 - self.cdf(np.array([split]), i)[0])

    def sample(self, n: int, rng: Optional[np.random.Generator] = None, seed: int = 0) -> np.ndarray:
        """``n`` exact posterior samples ``[n, d]`` by inverse-CDF interpolation."""
        rng = rng if rng is not None else np.random.default_rng(seed)
        u = rng.random((n, self.d))
        return np.stack([self.quantile(u[:, i], i) for i in range(self.d)], axis=1)

    # ------------------------------------------------------------------ #
    # Exact-quantile W2 between an empirical sample and the exact marginal.
    # W_p^p = int_0^1 |F_n^{-1}(u) - F^{-1}(u)|^p du with the
    # empirical quantile piecewise constant (= s_(k) on ((k-1)/n, k/n]); the
    # integrals of F^{-1} and (F^{-1})^2 over each block are exact partial moments.
    # ------------------------------------------------------------------ #
    def _blocks(self, s: np.ndarray, i: int):
        m = self.marginals[i]
        s = np.sort(np.asarray(s, dtype=np.float64))
        n = s.size
        u = np.arange(n + 1) / n
        q = self.quantile(u, i)
        q[0], q[-1] = m.grid[0], m.grid[-1]
        M1 = np.interp(q, m.grid, m.m1)
        M2 = np.interp(q, m.grid, m.m2)
        return m, s, u, q, M1, M2

    def marginal_w2(self, s: np.ndarray, i: int) -> float:
        _, s, u, _, M1, M2 = self._blocks(s, i)
        du = np.diff(u)
        w2sq = np.sum(s**2 * du - 2.0 * s * np.diff(M1) + np.diff(M2))
        return float(math.sqrt(max(w2sq, 0.0)))


# ---------------------------------------------------------------------- #
# Independent cross-check: self-normalised importance sampling from the prior
# (= one bootstrap-particle-filter analysis step with N particles).
# ---------------------------------------------------------------------- #
def importance_sampling_reference(
    post: PointwiseNonlinearPosterior,
    n: int = 1_000_000,
    seed: int = 0,
    joint: bool = False,
    split: float = 0.0,
) -> dict:
    """SNIS / bootstrap PF estimate of the posterior moments.

    ``joint=False`` weights each coordinate independently (exact because the
    posterior factorises; keeps the ESS high in any ``d``). ``joint=True`` uses the
    joint likelihood weight (the literal bootstrap PF; ESS collapses with ``d``).

    Returns ``mean``, ``var``, ``mass_above`` (``P(x > split)``) and their
    delta-method standard errors, plus the ``ess`` per coordinate.
    """
    rng = np.random.default_rng(seed)
    d = post.d
    x = post.x0[None, :] + math.sqrt(post.prior_var) * rng.standard_normal((n, d))
    loglik = -0.5 * (post.y[None, :] - post.g(x)) ** 2 / post.obs_var
    if joint:
        lw = loglik.sum(1, keepdims=True).repeat(d, axis=1)
    else:
        lw = loglik
    w = np.exp(lw - lw.max(0, keepdims=True))
    w /= w.sum(0, keepdims=True)
    mean = (w * x).sum(0)
    var = (w * (x - mean) ** 2).sum(0)
    above = (w * (x > split)).sum(0)
    se_mean = np.sqrt((w**2 * (x - mean) ** 2).sum(0))
    se_var = np.sqrt((w**2 * ((x - mean) ** 2 - var) ** 2).sum(0))
    se_above = np.sqrt((w**2 * ((x > split) - above) ** 2).sum(0))
    ess = 1.0 / (w**2).sum(0)
    return dict(mean=mean, var=var, mass_above=above, se_mean=se_mean,
                se_var=se_var, se_above=se_above, ess=ess)


def kalman_posterior_moments(x0, y, prior_var: float, obs_var: float):
    """Closed-form posterior for ``g = identity`` (mean[d], scalar var)."""
    x0 = np.asarray(x0, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    K = prior_var / (prior_var + obs_var)
    return x0 + K * (y - x0), (1.0 - K) * prior_var


__all__ = [
    "PointwiseNonlinearPosterior",
    "importance_sampling_reference",
    "kalman_posterior_moments",
    "g_numpy",
]
