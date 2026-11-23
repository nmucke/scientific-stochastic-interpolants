"""Particle filters for additive-noise models with linear Gaussian observations.

The model must expose ``forecast_to_last_kick(x, generator, dim)``, which stops
just before the final additive kick, and ``stage_variance``, the variance ``q`` of
that kick. So

    x^{k+1} = z + sqrt(q) xi,      y^{k+1} = H x^{k+1} + eps,   eps ~ N(0, r I),

where ``z`` depends on ``x^k`` and on any earlier kicks. Two proposals are offered,
and both target the same filtering distribution ``p(x^k | y^{1:k})``:

* ``"bootstrap"``: draw ``xi``, weight by ``N(y; H x, r I)``.
* ``"optimal"``: integrate out the last kick. The weight
  ``N(y; H z, q H H^T + r I)`` depends on ``z`` only, so resampling happens BEFORE
  the kick is drawn, and the kick is then drawn from its exact conditional given
  ``z`` and ``y``. With a single noise stage this is the classical optimal proposal
  ``p(x^{k+1} | x^k, y^{k+1})`` in its fully adapted auxiliary form. With more stages
  it is SIR on the extended state with the last kick integrated out, which is valid
  for any stage count but is no longer fully adapted.

Observations may also be pointwise nonlinear, ``y = g(H x) + eps`` with ``g``
elementwise (``obs_nonlinearity``, returning ``(g(w), g'(w))``). The optimal
proposal needs linear-Gaussian observations and refuses ``g``; two proposals stay
exact for any ``g``:

* ``"bootstrap"``: as above, with weight ``N(y; g(H x), r I)``.
* ``"linearised"``: one-stage importance sampling with the optimal proposal of the
  observation linearised at ``z``, ``g(H x) ~ g(H z) + diag(g'(H z)) H (x - z)``.
  Each particle draws ``x ~ q(x | z, y) = N(m(z, y), P(z))`` and carries the exact
  weight ``N(y; g(H x), r I) N(x; z, q I) / q(x | z, y)``, so the filter targets the
  filtering distribution whatever ``g`` is; the linearisation only changes the
  variance of the weights. Resampling happens after the kick. ``H`` must be a
  selection (one-hot rows, distinct columns), so ``P`` is diagonal. For ``g`` the
  identity the weight is ``N(y; H z, (q + r) I)``, the optimal proposal's weight.
  The weights have finite variance when ``P > q / 2`` on every observed site, i.e.
  ``r > q g'^2``; otherwise a warning is issued (still exact, but possibly
  heavy-tailed weights).
* ``"square_mixture"`` (``g(w) = w^2`` only, checked; selection ``H``): exact
  importance sampling for the sign-blind square, whose one-site kick posterior
  ``p(x_j | z_j, y_j) ∝ N(x_j; z_j, q) N(y_j; x_j^2, r)`` can be bimodal
  (``x_j ≈ ±sqrt(y_j)``) and has ``g' = 2 x`` vanishing at 0, so one linearisation
  can miss a mode. Per observed site the proposal is the defensive mixture
  ``alpha N(z, q) + (1 - alpha) (pi_hi N(m_hi, P_hi) + pi_lo N(m_lo, P_lo))`` over
  the local minima of ``f(x) = (x - z)^2 / (2q) + (y - x^2)^2 / (2r)`` (the outer
  real roots of the cubic ``f'(x) = 0``, in closed form; both modes get a component
  whenever both exist), ``P = 1 / f''`` (Laplace), ``pi ∝ exp(-f(m)) sqrt(P)``.
  The weight is ``N(y; g(H x), r I) N(x; z, q I) / q(x | z, y)`` with the full
  mixture density, so the filter is exact whatever the components; the prior
  component bounds every weight by ``max N(y; x^2, r) / alpha`` (finite variance,
  no mode can be dropped). Resampling happens after the kick.
* ``"abs_optimal"`` (``g(w) = |w|`` only, checked; selection ``H``): the EXACT
  optimal proposal for the sign-blind absolute value. On each half-line
  ``N(x; z, q) N(y; |x|, r)`` is a Gaussian in ``x`` times a constant, so the one-site
  kick posterior is a two-component mixture of truncated Gaussians: with
  ``v = q + r``, ``s^2 = q r / v`` and ``m_± = (r z ± q y) / v``,
  ``p(x | z, y) = pi_+ N(x; m_+, s^2 | x > 0) + pi_- N(x; m_-, s^2 | x < 0)``,
  ``pi_± ∝ N(z; ±y, v) Phi(±m_± / s)``, and the per-site predictive is
  ``p(y | z) = N(z; y, v) Phi(m_+ / s) + N(z; -y, v) Phi(-m_- / s)``. The weight
  ``prod_j p(y_j | z_j)`` depends on ``z`` only, so (as for ``"optimal"``)
  resampling happens BEFORE the kick, which is then drawn exactly (branch, then
  inverse-CDF truncated normal; unobserved sites get the prior kick). Both sign
  branches are sampled with their exact masses, so no mode can be dropped.

Particles are held state-major, ``[d, N]``, and processed in chunks.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import torch

#: ``w -> (g(w), g'(w))``, elementwise on observation-space values.
Nonlinearity = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def selection_indices(obs_matrix: torch.Tensor) -> torch.Tensor:
    """Column of each row of a selection matrix (one-hot rows, distinct columns)."""
    H = obs_matrix
    one_hot = bool(((H == 0) | (H == 1)).all()) and bool((H.sum(1) == 1).all())
    index = H.argmax(1)
    if not one_hot or index.unique().numel() != index.numel():
        raise ValueError("this proposal needs a selection observation matrix")
    return index


def _identity(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return w, torch.ones_like(w)


def _check_square(g: Optional[Nonlinearity]) -> None:
    """Refuse a ``g`` other than ``w -> (w^2, 2 w)`` (``square_mixture`` solves the
    cubic of exactly that ``g``)."""
    w = torch.tensor([-3.0, -0.5, 0.0, 0.25, 2.0, 7.0], dtype=torch.float64)
    ok = g is not None
    if ok:
        gw, dg = g(w)
        ok = bool(torch.allclose(gw, w**2, rtol=1e-12, atol=0)) and bool(
            torch.allclose(dg, 2.0 * w, rtol=1e-12, atol=0)
        )
    if not ok:
        raise ValueError("the square_mixture proposal needs obs_nonlinearity g(w) = w^2")


def _check_abs(g: Optional[Nonlinearity]) -> None:
    """Refuse a ``g`` other than ``w -> |w|`` (``abs_optimal`` is exact for exactly
    that ``g``; ``g'`` is not used)."""
    w = torch.tensor([-3.0, -0.5, 0.0, 0.25, 2.0, 7.0], dtype=torch.float64)
    ok = g is not None and bool(torch.equal(g(w)[0], w.abs()))
    if not ok:
        raise ValueError("the abs_optimal proposal needs obs_nonlinearity g(w) = |w|")


def abs_kick_branches(z: torch.Tensor, y: torch.Tensor, q: float, r: float):
    """The exact one-site kick posterior of ``y = |x| + N(0, r)``, ``x ~ N(z, q)``.

    Elementwise, in float64: ``(m_pos, m_neg, s, log_pi_pos, log_pi_neg)`` with
    ``p(x | z, y) = pi_pos N(x; m_pos, s^2 | x > 0) + pi_neg N(x; m_neg, s^2 | x < 0)``
    where ``log_pi_pos + log_pi_neg`` are UNNORMALISED: their ``logaddexp`` is
    ``log p(y | z)``, the per-site predictive (the ``abs_optimal`` log weight).
    """
    z, y = z.double(), y.double()
    v = q + r
    s = math.sqrt(q * r / v)
    m_pos = (r * z + q * y) / v
    m_neg = (r * z - q * y) / v
    log_norm = -0.5 * math.log(2.0 * math.pi * v)
    log_pi_pos = log_norm - 0.5 * (z - y).pow(2) / v + torch.special.log_ndtr(m_pos / s)
    log_pi_neg = log_norm - 0.5 * (z + y).pow(2) / v + torch.special.log_ndtr(-m_neg / s)
    return m_pos, m_neg, s, log_pi_pos, log_pi_neg


def _standard_normal_above(a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """``Z ~ N(0, 1) | Z > a`` by inversion from uniforms ``u`` in ``[0, 1)``.

    ``a <= 0``: ``Phi(Z) = Phi(a) + u Phi(-a)``; ``a > 0`` (tail):
    ``Phi(-Z) = (1 - u) Phi(-a)``, in log space. Both are exact inversions, accurate
    to float64 rounding. Where ``Phi(-a)`` underflows (``a`` beyond ~37) the exact
    limit ``a (Z - a) -> Exp(1)`` is used.
    """
    tail = a > 0
    log_upper = torch.log1p(-u) + torch.special.log_ndtr(-a)
    z_tail = -torch.special.ndtri(log_upper.exp())
    bulk = torch.special.ndtr(a) + u * torch.special.ndtr(-a)
    z_bulk = torch.special.ndtri(bulk.clamp(max=1.0 - 2.0**-53))
    z = torch.where(tail, z_tail, z_bulk)
    deep = tail & (log_upper < -700.0)
    if bool(deep.any()):
        z = torch.where(deep, a - torch.log1p(-u) / a.clamp(min=1.0), z)
    return torch.maximum(z, a)


def square_kick_modes(z: torch.Tensor, y: torch.Tensor, q: float, r: float):
    """Local minima of ``f(x) = (x - z)^2 / (2q) + (y - x^2)^2 / (2r)``, elementwise.

    ``f'(x) = 0`` is the depressed cubic ``x^3 + p x + s = 0`` with
    ``p = r / (2q) - y``, ``s = -r z / (2q)``. With one real root it is the only
    minimum (returned twice); with three, the outer two are the minima (the middle
    one is a maximum). Returns ``(m_hi, m_lo, f''(m_hi), f''(m_lo))`` in float64.
    """
    z, y = z.double(), y.double()
    p = r / (2.0 * q) - y
    s = -r * z / (2.0 * q)
    disc = (s / 2.0) ** 2 + (p / 3.0) ** 3
    root = disc.clamp(min=0.0).sqrt()
    u, v = -s / 2.0 + root, -s / 2.0 - root
    single = u.sign() * u.abs().pow(1.0 / 3.0) + v.sign() * v.abs().pow(1.0 / 3.0)
    neg_p = (-p).clamp(min=1e-300)
    amp = 2.0 * (neg_p / 3.0).sqrt()
    arg = (-(3.0 * s) / (2.0 * neg_p) * (3.0 / neg_p).sqrt()).clamp(-1.0, 1.0)
    theta = torch.arccos(arg) / 3.0
    hi = amp * torch.cos(theta)  # largest root
    lo = amp * torch.cos(theta - 4.0 * math.pi / 3.0)  # smallest root
    three = disc <= 0
    m_hi = torch.where(three, hi, single)
    m_lo = torch.where(three, lo, single)

    def curvature(x):
        return 1.0 / q + (6.0 * x * x - 2.0 * y) / r

    return m_hi, m_lo, curvature(m_hi), curvature(m_lo)


def systematic_resample(
    log_weights: torch.Tensor, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Return systematic-resampling indices, computed in float64."""
    n = log_weights.numel()
    w = torch.exp(log_weights.double() - log_weights.max().double())
    cumulative = torch.cumsum(w / w.sum(), dim=0)
    cumulative[-1] = 1.0
    u = torch.rand((), generator=generator, dtype=torch.float64)
    positions = (torch.arange(n, dtype=torch.float64) + u) / n
    return torch.searchsorted(cumulative, positions).clamp_(max=n - 1)


def _check_finite(log_weights: torch.Tensor) -> None:
    """Refuse non-finite log weights before resampling.

    One NaN would otherwise poison the cumulative sum, and ``searchsorted`` would
    silently put every resampled particle on the last one.
    """
    if not torch.isfinite(log_weights).all():
        bad = int((~torch.isfinite(log_weights)).sum())
        raise FloatingPointError(
            f"{bad} of {log_weights.numel()} particle log weights are not finite."
        )


def effective_sample_size(log_weights: torch.Tensor) -> float:
    """Kish effective sample size of unnormalised log weights."""
    log_w = log_weights.double()
    log_w = log_w - torch.logsumexp(log_w, 0)
    return float(torch.exp(-torch.logsumexp(2.0 * log_w, 0)))


@dataclass
class ParticleFilterResult:
    """Per-step summaries of one filter run (``T`` assimilation steps)."""

    mean: torch.Tensor  # [T, d]
    variance: torch.Tensor  # [T, d]
    covariance: torch.Tensor  # [T, d, d]
    samples: torch.Tensor  # [T, S, d] equally weighted posterior draws
    ess: torch.Tensor  # [T] Kish ESS before resampling
    log_evidence: torch.Tensor  # [T] incremental log p(y^k | y^{1:k-1})
    num_particles: int
    # [T, d] fraction of the (equally weighted, resampled) particles with x > 0:
    # the mass of the positive mode of each site (sign-blind observations).
    positive_fraction: Optional[torch.Tensor] = None
    # [F, d] equally weighted draws of the LAST step (``final_samples`` > 0), the
    # first F of the same permutation as ``samples[-1]``; None otherwise.
    final_samples: Optional[torch.Tensor] = None

    @property
    def ess_fraction(self) -> torch.Tensor:
        return self.ess / self.num_particles


class BootstrapParticleFilter:
    """Sequential importance resampling with a bootstrap, optimal or linearised proposal.

    Args:
        obs_matrix: ``H`` of shape ``[d_y, d]``.
        obs_variance: Scalar ``r`` with ``R = r I``.
        proposal: ``"bootstrap"``, ``"optimal"`` (linear observations only),
            ``"linearised"`` (selection ``H``) or ``"square_mixture"`` (selection
            ``H``, ``g(w) = w^2``) or ``"abs_optimal"`` (selection ``H``,
            ``g(w) = |w|``); see the module docstring.
        chunk_size: Particles processed at once; bounds peak memory and keeps the
            RK4 working set in cache.
        num_saved_samples: Posterior draws kept per step for sample-based metrics.
        obs_nonlinearity: Optional elementwise ``g`` of ``y = g(H x) + eps``,
            returning ``(g(w), g'(w))``. ``None`` means linear observations and
            leaves every linear code path as it was.
        final_samples: Equally weighted draws of the last step to keep in
            ``final_samples`` (0: none). They reuse the permutation of
            ``samples``, so no random number is drawn for them.
        mixture_defensive: ``alpha``, the weight of the prior component of the
            ``square_mixture`` proposal.

    ``"abs_optimal"`` (selection ``H``, ``g(w) = |w|``) is the exact optimal
    proposal of the absolute value; see the module docstring.
    """

    def __init__(
        self,
        obs_matrix: torch.Tensor,
        obs_variance: float,
        proposal: str = "optimal",
        chunk_size: int = 1 << 15,
        num_saved_samples: int = 4096,
        obs_nonlinearity: Optional[Nonlinearity] = None,
        final_samples: int = 0,
        mixture_defensive: float = 0.1,
    ) -> None:
        if proposal not in (
            "bootstrap", "optimal", "linearised", "square_mixture", "abs_optimal"
        ):
            raise ValueError(f"unknown proposal {proposal!r}")
        if proposal == "optimal" and obs_nonlinearity is not None:
            raise ValueError(
                "the optimal proposal needs linear observations; use 'linearised' "
                "or 'bootstrap' with a nonlinear observation operator"
            )
        self.H = obs_matrix.double()
        self.obs_variance = float(obs_variance)
        self.proposal = proposal
        self.chunk_size = int(chunk_size)
        self.num_saved_samples = int(num_saved_samples)
        self.obs_nonlinearity = obs_nonlinearity
        self.final_samples = int(final_samples)
        if not 0.0 < float(mixture_defensive) < 1.0:
            raise ValueError(f"mixture_defensive must be in (0, 1), got {mixture_defensive!r}")
        self.mixture_defensive = float(mixture_defensive)
        if proposal == "square_mixture":
            _check_square(obs_nonlinearity)
        if proposal == "abs_optimal":
            _check_abs(obs_nonlinearity)
        selection = proposal in ("linearised", "square_mixture", "abs_optimal")
        self.obs_index = selection_indices(self.H) if selection else None

    def _forecast(self, model, x: torch.Tensor, generator) -> torch.Tensor:
        out = torch.empty_like(x)
        for start in range(0, x.shape[1], self.chunk_size):
            sl = slice(start, start + self.chunk_size)
            out[:, sl] = model.forecast_to_last_kick(
                x[:, sl].contiguous(), generator, dim=0
            )
        return out

    def _log_gaussian(self, residual: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """``log N(residual; 0, cov)`` for ``residual`` of shape ``[d_y, N]``."""
        chol = torch.linalg.cholesky(cov.to(residual))
        white = torch.linalg.solve_triangular(chol, residual, upper=False)
        log_det = 2.0 * torch.log(torch.diagonal(chol)).sum()
        return -0.5 * (
            (white * white).sum(0)
            + log_det
            + residual.shape[0] * math.log(2.0 * math.pi)
        )

    def _kick_conditional(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        q: float,
        S: torch.Tensor,
        generator,
    ) -> torch.Tensor:
        """Draw ``x ~ p(x | z, y)`` by Matheron's rule, in chunks and in place."""
        H = self.H.to(z)
        gain = (q * H.T @ torch.linalg.inv(S.to(z))).to(z)
        std_q, std_r = q**0.5, self.obs_variance**0.5
        d_y = H.shape[0]
        for start in range(0, z.shape[1], self.chunk_size):
            sl = slice(start, start + self.chunk_size)
            x = z[:, sl]
            x = x + std_q * torch.randn(x.shape, generator=generator, dtype=z.dtype)
            eps = std_r * torch.randn(
                d_y, x.shape[1], generator=generator, dtype=z.dtype
            )
            z[:, sl] = x + gain @ (y[:, None] - H @ x - eps)
        return z

    def run(
        self,
        model,
        initial_particles: torch.Tensor,
        observations: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> ParticleFilterResult:
        """Filter ``observations`` ``[T, d_y]`` from particles ``[d, N]``."""
        x = initial_particles
        num_particles = x.shape[1]
        q = float(model.stage_variance)
        H = self.H.to(x)
        d_y = self.H.shape[0]
        eye = torch.eye(d_y, dtype=torch.float64)
        S = q * (self.H @ self.H.T) + self.obs_variance * eye
        R = self.obs_variance * eye
        means, covariances, samples, ess, log_evidence = [], [], [], [], []
        positive, final = [], None

        for y in observations:
            y_x = y.to(x)
            z = self._forecast(model, x, generator)

            if self.proposal == "optimal":
                log_w = self._log_weights(z, y_x, H, S)
                _check_finite(log_w)
                idx = systematic_resample(log_w, generator)
                x = z[:, idx]
                x = self._kick_conditional(x, y_x, q, S, generator)
            elif self.proposal == "abs_optimal":
                log_w = self._abs_log_predictive(z, y_x, q)
                _check_finite(log_w)
                idx = systematic_resample(log_w, generator)
                x = z[:, idx]
                x = self._abs_kick(x, y_x, q, generator)
            elif self.proposal in ("linearised", "square_mixture"):
                kick = (
                    self._linearised_kick
                    if self.proposal == "linearised"
                    else self._square_mixture_kick
                )
                log_w = kick(z, y_x, q, generator)
                _check_finite(log_w)
                idx = systematic_resample(log_w, generator)
                x = z[:, idx]
            else:
                for start in range(0, z.shape[1], self.chunk_size):
                    sl = slice(start, start + self.chunk_size)
                    z[:, sl] = z[:, sl] + q**0.5 * torch.randn(
                        z[:, sl].shape, generator=generator, dtype=z.dtype
                    )
                log_w = self._log_weights(z, y_x, H, R)
                _check_finite(log_w)
                idx = systematic_resample(log_w, generator)
                x = z[:, idx]

            ess.append(effective_sample_size(log_w))
            log_evidence.append(
                float(torch.logsumexp(log_w, 0)) - math.log(num_particles)
            )
            mean, covariance = self._moments(x)
            means.append(mean)
            covariances.append(covariance)
            positive.append(self._positive_fraction(x))
            pick = torch.randperm(num_particles, generator=generator)
            samples.append(x[:, pick[: self.num_saved_samples]].T.clone())
        if self.final_samples > 0 and len(observations) > 0:
            final = x[:, pick[: self.final_samples]].T.clone()

        covariances = torch.stack(covariances)
        return ParticleFilterResult(
            mean=torch.stack(means),
            variance=torch.diagonal(covariances, dim1=1, dim2=2),
            covariance=covariances,
            samples=torch.stack(samples),
            ess=torch.tensor(ess, dtype=torch.float64),
            log_evidence=torch.tensor(log_evidence, dtype=torch.float64),
            num_particles=num_particles,
            positive_fraction=torch.stack(positive) if positive else None,
            final_samples=final,
        )

    def _positive_fraction(self, x: torch.Tensor) -> torch.Tensor:
        """Fraction of the particles ``[d, N]`` with ``x > 0``, per coordinate."""
        count = torch.zeros(x.shape[0], dtype=torch.float64)
        for start in range(0, x.shape[1], self.chunk_size):
            count += (x[:, start : start + self.chunk_size] > 0).sum(1).double()
        return count / x.shape[1]

    def _moments(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and unbiased covariance of ``[d, N]``, accumulated in float64."""
        d, n = x.shape
        total = torch.zeros(d, dtype=torch.float64)
        for start in range(0, n, self.chunk_size):
            total += x[:, start : start + self.chunk_size].double().sum(1)
        mean = total / n
        scatter = torch.zeros(d, d, dtype=torch.float64)
        for start in range(0, n, self.chunk_size):
            centred = x[:, start : start + self.chunk_size].double() - mean[:, None]
            scatter += centred @ centred.T
        return mean, scatter / (n - 1)

    def _log_weights(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        H: torch.Tensor,
        cov: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty(x.shape[1], dtype=torch.float64)
        g = self.obs_nonlinearity
        for start in range(0, x.shape[1], self.chunk_size):
            sl = slice(start, start + self.chunk_size)
            if g is None:
                residual = (y[:, None] - H @ x[:, sl]).double()
            else:
                residual = y[:, None].double() - g((H @ x[:, sl]).double())[0]
            out[sl] = self._log_gaussian(residual, cov)
        return out

    def _linearised_kick(
        self, z: torch.Tensor, y: torch.Tensor, q: float, generator
    ) -> torch.Tensor:
        """Kick ``z`` in place from the linearised optimal proposal; return log weights.

        Per observed site ``j`` (``w = z_j``, ``a = g'(w)``): ``s = q a^2 + r``,
        ``m = w + q a (y_j - g(w)) / s``, ``P = q r / s``; unobserved sites get the
        prior kick. The log weight is ``log N(y; g(H x), r I)`` plus, on the observed
        sites, ``log N(x_j; w, q) - log N(x_j; m, P)`` (the unobserved factors
        cancel exactly), evaluated in float64 at the stored value of ``x``.
        """
        g = self.obs_nonlinearity or _identity
        r = self.obs_variance
        index = self.obs_index
        out = torch.empty(z.shape[1], dtype=torch.float64)
        warned = False
        for start in range(0, z.shape[1], self.chunk_size):
            sl = slice(start, start + self.chunk_size)
            xi = torch.randn(z[:, sl].shape, generator=generator, dtype=z.dtype)
            w = z[index, sl].double()
            gw, slope = g(w)
            s = q * slope**2 + r
            m = w + q * slope * (y[:, None].double() - gw) / s
            P = q * r / s
            if not warned and bool((P <= 0.5 * q).any()):
                warnings.warn(
                    "linearised proposal: P <= q/2 at some site, so the importance "
                    "weights may have infinite variance (the filter stays exact).",
                    RuntimeWarning,
                    stacklevel=3,
                )
                warned = True
            kicked = (m + P.sqrt() * xi[index].double()).to(z.dtype)
            z[:, sl] = z[:, sl] + q**0.5 * xi
            z[index, sl] = kicked
            x_obs = kicked.double()
            residual = y[:, None].double() - g(x_obs)[0]
            log_lik = -0.5 * (
                residual.pow(2).sum(0) / r + w.shape[0] * math.log(2.0 * math.pi * r)
            )
            log_prior = -0.5 * ((x_obs - w).pow(2) / q + math.log(2.0 * math.pi * q)).sum(0)
            log_prop = -0.5 * ((x_obs - m).pow(2) / P + (2.0 * math.pi * P).log()).sum(0)
            out[sl] = log_lik + log_prior - log_prop
        return out

    def _square_mixture_kick(
        self, z: torch.Tensor, y: torch.Tensor, q: float, generator
    ) -> torch.Tensor:
        """Kick ``z`` in place from the ``square_mixture`` proposal; return log weights.

        Per observed site: with probability ``alpha`` the prior kick ``N(w, q)``,
        otherwise the Laplace component of the upper / lower minimum of ``f``
        (:func:`square_kick_modes`) with probability ``pi_hi`` / ``pi_lo``. The log
        weight is ``log N(y; x^2, r I)`` plus, on the observed sites,
        ``log N(x_j; w, q) - log q_mix(x_j)`` with the full mixture density (the
        unobserved factors cancel), evaluated in float64 at the stored ``x``.
        """
        r, alpha, index = self.obs_variance, self.mixture_defensive, self.obs_index
        out = torch.empty(z.shape[1], dtype=torch.float64)
        log_2pi = math.log(2.0 * math.pi)

        def log_normal(x, m, v):
            return -0.5 * ((x - m).pow(2) / v + v.log() + log_2pi)

        for start in range(0, z.shape[1], self.chunk_size):
            sl = slice(start, start + self.chunk_size)
            xi = torch.randn(z[:, sl].shape, generator=generator, dtype=z.dtype)
            u = torch.rand(index.numel(), xi.shape[1], generator=generator, dtype=torch.float64)
            w = z[index, sl].double()
            yy = y[:, None].double().expand_as(w)
            m_hi, m_lo, c_hi, c_lo = square_kick_modes(w, yy, q, r)
            # Laplace variances, capped at 4 q for a near-flat minimum (any P > 0 is
            # valid; this only shapes the proposal).
            P_hi = 1.0 / c_hi.clamp(min=0.25 / q)
            P_lo = 1.0 / c_lo.clamp(min=0.25 / q)

            def f(x):
                return (x - w).pow(2) / (2.0 * q) + (yy - x * x).pow(2) / (2.0 * r)

            logit = (-f(m_hi) + 0.5 * P_hi.log()) - (-f(m_lo) + 0.5 * P_lo.log())
            use_prior = u < alpha
            use_hi = (~use_prior) & (u < alpha + (1.0 - alpha) * torch.sigmoid(logit))
            q_t = torch.full_like(w, q)
            mean = torch.where(use_prior, w, torch.where(use_hi, m_hi, m_lo))
            var = torch.where(use_prior, q_t, torch.where(use_hi, P_hi, P_lo))
            kicked = (mean + var.sqrt() * xi[index].double()).to(z.dtype)
            z[:, sl] = z[:, sl] + q**0.5 * xi
            z[index, sl] = kicked
            x_obs = kicked.double()
            log_prior = log_normal(x_obs, w, q_t)
            log_mix = math.log1p(-alpha)
            log_prop = torch.logsumexp(
                torch.stack(
                    [
                        math.log(alpha) + log_prior,
                        log_mix + torch.nn.functional.logsigmoid(logit)
                        + log_normal(x_obs, m_hi, P_hi),
                        log_mix + torch.nn.functional.logsigmoid(-logit)
                        + log_normal(x_obs, m_lo, P_lo),
                    ]
                ),
                0,
            )
            residual = y[:, None].double() - x_obs * x_obs
            log_lik = -0.5 * (
                residual.pow(2).sum(0) / r + w.shape[0] * math.log(2.0 * math.pi * r)
            )
            out[sl] = log_lik + (log_prior - log_prop).sum(0)
        return out


    def _abs_log_predictive(self, z: torch.Tensor, y: torch.Tensor, q: float) -> torch.Tensor:
        """``sum_j log p(y_j | z_j)`` (:func:`abs_kick_branches`): the exact
        ``abs_optimal`` log weights of the forecasts ``z`` ``[d, N]``."""
        out = torch.empty(z.shape[1], dtype=torch.float64)
        for start in range(0, z.shape[1], self.chunk_size):
            sl = slice(start, start + self.chunk_size)
            w = z[self.obs_index, sl].double()
            yy = y[:, None].double().expand_as(w)
            *_, log_pos, log_neg = abs_kick_branches(w, yy, q, self.obs_variance)
            out[sl] = torch.logaddexp(log_pos, log_neg).sum(0)
        return out

    def _abs_kick(self, z: torch.Tensor, y: torch.Tensor, q: float, generator) -> torch.Tensor:
        """Draw ``x ~ p(x | z, y)`` exactly for ``g = |.|``, in chunks and in place.

        Unobserved sites: the prior kick ``z + sqrt(q) xi``. Observed sites: the sign
        branch with probability ``pi_pos`` / ``pi_neg``, then the truncated normal
        of that branch by inversion (:func:`_standard_normal_above`).
        """
        index, r = self.obs_index, self.obs_variance
        for start in range(0, z.shape[1], self.chunk_size):
            sl = slice(start, start + self.chunk_size)
            xi = torch.randn(z[:, sl].shape, generator=generator, dtype=z.dtype)
            shape = (index.numel(), xi.shape[1])
            u = torch.rand(shape, generator=generator, dtype=torch.float64)
            v = torch.rand(shape, generator=generator, dtype=torch.float64)
            w = z[index, sl].double()
            yy = y[:, None].double().expand_as(w)
            m_pos, m_neg, s, log_pos, log_neg = abs_kick_branches(w, yy, q, r)
            positive = u < torch.sigmoid(log_pos - log_neg)
            # x > 0: x = m_pos + s Z, Z > -m_pos / s;  x < 0: x = m_neg - s Z, Z > m_neg / s.
            a = torch.where(positive, -m_pos / s, m_neg / s)
            draw = _standard_normal_above(a, v)
            kicked = torch.where(positive, m_pos + s * draw, m_neg - s * draw)
            z[:, sl] = z[:, sl] + q**0.5 * xi
            z[index, sl] = kicked.to(z.dtype)
        return z


class OneStepPosteriorOracle:
    """Exact-model version of the paper's unweighted one-step assimilation.

    Each member independently draws ``x^n ~ p(x^n | x^{n-1}, y^n)`` from the TRUE
    model, with no reweighting across members. This is what a posterior sampler
    with a perfect prior and a perfect likelihood score would produce, so it
    separates the bias of the unweighted one-step scheme from network and
    closure error. Given a member, ``z`` is drawn from ``inner_samples`` candidates
    by SIR (exact in ``z`` when ``noise_stages == 1``, where all candidates
    coincide) and the last kick from its exact conditional.
    """

    def __init__(
        self,
        obs_matrix: torch.Tensor,
        obs_variance: float,
        inner_samples: int = 512,
    ) -> None:
        self.H = obs_matrix.double()
        self.obs_variance = float(obs_variance)
        self.inner_samples = int(inner_samples)

    def run(
        self,
        model,
        initial_ensemble: torch.Tensor,
        observations: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Return analysis ensembles ``[T, E, d]`` from an initial ``[E, d]``."""
        x = initial_ensemble
        members, d = x.shape
        q = float(model.stage_variance)
        H = self.H.to(x)
        d_y = H.shape[0]
        S = (q * self.H @ self.H.T + self.obs_variance * torch.eye(d_y).double()).to(x)
        S_inv = torch.linalg.inv(S)
        gain = q * H.T @ S_inv
        inner = 1 if model.noise_stages == 1 else self.inner_samples
        analysis = []
        for y in observations:
            y = y.to(x)
            z = model.forecast_to_last_kick(
                x.repeat_interleave(inner, dim=0), generator
            )
            if inner > 1:
                residual = y - z @ H.T
                log_w = -0.5 * ((residual @ S_inv) * residual).sum(1)
                pick = torch.multinomial(
                    torch.softmax(log_w.view(members, inner), dim=1),
                    1,
                    generator=generator,
                ).squeeze(1)
                z = z.view(members, inner, d)[torch.arange(members), pick]
            kicked = z + q**0.5 * torch.randn(z.shape, generator=generator, dtype=z.dtype)
            eps = self.obs_variance**0.5 * torch.randn(
                members, d_y, generator=generator, dtype=z.dtype
            )
            x = kicked + (y - kicked @ H.T - eps) @ gain.T
            analysis.append(x)
        return torch.stack(analysis)


@dataclass
class ImportanceOracleResult:
    """Analysis ensembles ``[T, E, d]`` and each member's Kish ESS ``[T, E]``."""

    analysis: torch.Tensor
    ess: torch.Tensor


class ImportanceOneStepOracle:
    """:class:`OneStepPosteriorOracle` for any observation likelihood, by importance sampling.

    Each member independently draws ``x^n ~ p(x^n | x^{n-1}_i, y^n)``: ``proposals``
    candidates from the TRUE transition ``p(. | x^{n-1}_i)`` (with one noise stage
    the forecast is integrated once and only the kick is redrawn), weighted by
    ``N(y; g(H x), r I)``, one of them picked by multinomial sampling. Exact as
    ``proposals -> infinity`` (sampling importance resampling with one draw); the
    per-member ESS of the candidate weights records how close it is. No
    reweighting across members, as for the closed-form oracle.

    Args:
        obs_matrix: ``H`` ``[d_y, d]``. obs_variance: ``r``.
        obs_nonlinearity: Elementwise ``g`` returning ``(g(w), g'(w))``; ``None``
            is linear.
        proposals: Candidates per member and step.
        max_candidates: Candidates held in memory at once (members are chunked).
    """

    def __init__(
        self,
        obs_matrix: torch.Tensor,
        obs_variance: float,
        obs_nonlinearity: Optional[Nonlinearity] = None,
        proposals: int = 4096,
        max_candidates: int = 1 << 20,
    ) -> None:
        self.H = obs_matrix.double()
        self.obs_variance = float(obs_variance)
        self.obs_nonlinearity = obs_nonlinearity or _identity
        self.proposals = int(proposals)
        self.max_candidates = int(max_candidates)

    def draw(
        self, model, x: torch.Tensor, y: torch.Tensor, generator=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One step from members ``[E, d]``: ``(new members [E, d], ESS [E])``."""
        members, d = x.shape
        P = self.proposals
        H = self.H.to(x)
        y = y.to(x)
        q = float(model.stage_variance)
        chunk = max(1, self.max_candidates // P)
        out, ess = [], []
        for start in range(0, members, chunk):
            xs = x[start : start + chunk]
            n = xs.shape[0]
            if model.noise_stages == 1:
                z = model.forecast_to_last_kick(xs, generator)
                cand = z[:, None, :] + q**0.5 * torch.randn(
                    n, P, d, generator=generator, dtype=x.dtype
                )
            else:
                cand = model.step(xs.repeat_interleave(P, dim=0), generator).view(n, P, d)
            residual = (y - self.obs_nonlinearity(cand @ H.T)[0]).double()
            log_w = -0.5 * residual.pow(2).sum(-1) / self.obs_variance  # [n, P]
            log_w = log_w - log_w.max(1, keepdim=True).values
            w = log_w.exp()
            w = w / w.sum(1, keepdim=True)
            ess.append(1.0 / w.pow(2).sum(1))
            pick = torch.multinomial(w, 1, generator=generator).squeeze(1)
            out.append(cand[torch.arange(n), pick])
        return torch.cat(out), torch.cat(ess)

    def run(
        self,
        model,
        initial_ensemble: torch.Tensor,
        observations: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> ImportanceOracleResult:
        """Analysis ensembles ``[T, E, d]`` from an initial ``[E, d]``."""
        x = initial_ensemble
        analysis, ess = [], []
        for y in observations:
            x, e = self.draw(model, x, y, generator)
            analysis.append(x)
            ess.append(e)
        return ImportanceOracleResult(torch.stack(analysis), torch.stack(ess))


__all__ = [
    "BootstrapParticleFilter",
    "ImportanceOneStepOracle",
    "ImportanceOracleResult",
    "OneStepPosteriorOracle",
    "ParticleFilterResult",
    "abs_kick_branches",
    "effective_sample_size",
    "selection_indices",
    "square_kick_modes",
    "systematic_resample",
]
