"""General linear--Gaussian state-space model for the multistep analytical case.

    x^{k+1} = A x^k + w_k,  w_k ~ N(0, Q)
    y^{k+1} = H x^{k+1} + e_k,  e_k ~ N(0, R)

Everything the sequential experiment needs that the single-step
``classical_baselines`` / ``prior_models`` cannot express, because those hard-code
``A = I``, ``Q = prior_var I``, ``H = I`` and a single shared ``x0``:

* :class:`LinearGaussianSSM` -- the system, with two closed-form recursions:

  - :meth:`~LinearGaussianSSM.kalman_step`, the exact filtering posterior
    ``p(x^k | y^{1:k})`` (the reference of the experiment);
  - :meth:`~LinearGaussianSSM.member_chain_step`, the law of an ensemble in which
    every member is pushed through the EXACT per-member conditional
    ``p(x^{k+1} | x^k_i, y^{k+1})`` with no interaction between members. That is
    what "feed each posterior sample back" computes, and it is NOT the filtering
    posterior once the input ensemble has spread: the filter also reweights
    members by ``p(y^{k+1} | x^k_i)``. See ``cases/analytical/README.md``.

* :func:`enkf_step`, :func:`particle_filter_step` -- the two classical filters on
  an ensemble input, same draw order as ``classical_baselines`` so that the paper
  system at step 1 reproduces them.
* :func:`adapted_resample` -- the missing ``p(y^{k+1} | x^k_i)`` reweighting.
* Closed-form SI / FM / DM priors for ``x^1 | x^0 ~ N(A x^0, Q)`` and a dense
  observation operator.

Matrices are held in float64 (the Kalman recursion); ensembles stay float32, the
precision of every other analytical run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from scisi.likelihood_models.observation_operators import LinearObservationOperator
from scisi.models.diffusion_model import DenoiseDiffusionModel
from scisi.models.flow_matching_model import FlowMatchingModel
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
from scisi.models.interpolations import (
    LinearDeterministicInterpolation,
    LinearStochasticInterpolation,
)
from scisi.posterior_models.surge_posterior import SurgeFlowMatchingPosterior

from .prior_models import _T_MAX, _T_MIN, _expand_t

Tensor = torch.Tensor
MatrixSpec = Union[None, float, int, Sequence]


def _as_matrix(value: MatrixSpec, rows: int, cols: int, name: str) -> Tensor:
    """``None`` -> identity, scalar ``s`` -> ``s I``, nested list -> that matrix."""
    if value is None:
        value = 1.0
    if isinstance(value, (int, float)):
        if rows != cols:
            raise ValueError(f"{name}: a scalar needs a square matrix, got {rows}x{cols}")
        return float(value) * torch.eye(rows, dtype=torch.float64)
    mat = torch.as_tensor(
        [list(row) for row in value], dtype=torch.float64
    )
    if mat.shape != (rows, cols):
        raise ValueError(f"{name}: expected shape {(rows, cols)}, got {tuple(mat.shape)}")
    return mat


def _check_spd(mat: Tensor, name: str) -> None:
    if not torch.allclose(mat, mat.T):
        raise ValueError(f"{name} must be symmetric")
    if float(torch.linalg.eigvalsh(mat)[0]) <= 0.0:
        raise ValueError(f"{name} must be positive definite")


@dataclass(frozen=True, eq=False)
class LinearGaussianSSM:
    """``x' = A x + w``, ``y = H x + e``. Build with :meth:`from_spec`."""

    A: Tensor
    Q: Tensor
    H: Tensor
    R: Tensor

    @classmethod
    def from_spec(
        cls,
        d: int,
        A: MatrixSpec = None,
        Q: MatrixSpec = None,
        H: MatrixSpec = None,
        R: MatrixSpec = None,
    ) -> "LinearGaussianSSM":
        """Defaults are the paper's system ``A = Q = H = R = I``."""
        m = d if H is None or isinstance(H, (int, float)) else len(H)
        system = cls(
            A=_as_matrix(A, d, d, "A"),
            Q=_as_matrix(Q, d, d, "Q"),
            H=_as_matrix(H, m, d, "H"),
            R=_as_matrix(R, m, m, "R"),
        )
        _check_spd(system.Q, "Q")
        _check_spd(system.R, "R")
        return system

    @property
    def d(self) -> int:
        return self.A.shape[0]

    @property
    def m(self) -> int:
        return self.H.shape[0]

    @property
    def isotropic_q(self) -> Optional[float]:
        """``q`` when ``Q = q I``, else ``None``."""
        q = float(self.Q[0, 0])
        return q if torch.equal(self.Q, q * torch.eye(self.d, dtype=self.Q.dtype)) else None

    @property
    def isotropic_r(self) -> Optional[float]:
        """``r`` when ``R = r I``, else ``None``."""
        r = float(self.R[0, 0])
        return r if torch.equal(self.R, r * torch.eye(self.m, dtype=self.R.dtype)) else None

    @property
    def is_paper_prior(self) -> bool:
        """``A = I`` and ``Q = q I``: the existing ``prior_models`` apply as they are."""
        eye = torch.eye(self.d, dtype=self.A.dtype)
        return torch.equal(self.A, eye) and self.isotropic_q is not None

    @property
    def is_identity_obs(self) -> bool:
        return self.m == self.d and torch.equal(
            self.H, torch.eye(self.d, dtype=self.H.dtype)
        )

    # -- truth ------------------------------------------------------------- #

    def propagate(self, x: Tensor, generator: torch.Generator) -> Tensor:
        """``A x + w`` for ``x`` of shape ``[..., d]`` (one ``w`` per row)."""
        noise = torch.randn(x.shape, generator=generator, dtype=x.dtype)
        return x @ self.A.to(x.dtype).T + noise @ _chol(self.Q).to(x.dtype).T

    def observe(self, x: Tensor, generator: torch.Generator) -> Tensor:
        """``H x + e`` for a single state ``x`` of shape ``[d]``."""
        noise = torch.randn(self.m, generator=generator, dtype=x.dtype)
        return x @ self.H.to(x.dtype).T + noise @ _chol(self.R).to(x.dtype).T

    # -- exact recursions (float64) ---------------------------------------- #

    def innovation_cov(self, P: Tensor) -> Tensor:
        """``H P H^T + R``."""
        return self.H @ P @ self.H.T + self.R

    def conditional_gain(self) -> Tensor:
        """Gain of ``p(x^{k+1} | x^k, y^{k+1})``: ``Q H^T (H Q H^T + R)^{-1}``."""
        return torch.linalg.solve(self.innovation_cov(self.Q), self.H @ self.Q).T

    def kalman_step(self, m: Tensor, P: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """One predict + update of the Kalman filter (Joseph-form covariance)."""
        y = y.to(torch.float64)
        m_f = self.A @ m
        P_f = self.A @ P @ self.A.T + self.Q
        K = torch.linalg.solve(self.innovation_cov(P_f), self.H @ P_f).T
        I_KH = torch.eye(self.d, dtype=P.dtype) - K @ self.H
        return (
            m_f + K @ (y - self.H @ m_f),
            I_KH @ P_f @ I_KH.T + K @ self.R @ K.T,
        )

    def member_chain_step(
        self, m: Tensor, P: Tensor, y: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Law of the ensemble after an EXACT per-member update, no reweighting.

        Each member becomes ``x' = (I - K H) A x + K y + eps`` with ``K`` the
        conditional gain and ``eps ~ N(0, (I - K H) Q)``, so an input ``N(m, P)``
        maps to ``N((I - K H) A m + K y, B P B^T + (I - K H) Q)`` with
        ``B = (I - K H) A``. Identical to :meth:`kalman_step` when ``P = 0``.
        """
        y = y.to(torch.float64)
        K = self.conditional_gain()
        I_KH = torch.eye(self.d, dtype=P.dtype) - K @ self.H
        B = I_KH @ self.A
        return B @ m + K @ y, B @ P @ B.T + I_KH @ self.Q


def _chol(mat: Tensor) -> Tensor:
    return torch.linalg.cholesky(mat)


def gaussian_samples(
    mean: Tensor, cov: Tensor, n: int, generator: torch.Generator
) -> Tensor:
    """``n`` draws from ``N(mean, cov)``, float32 like every ensemble here."""
    z = torch.randn(n, mean.shape[0], generator=generator)
    return (mean + z.to(mean.dtype) @ _chol(cov).T).to(torch.float32)


# --------------------------------------------------------------------------- #
# Classical filters on an ensemble input
# --------------------------------------------------------------------------- #


def enkf_step(
    system: LinearGaussianSSM, x: Tensor, y: Tensor, generator: torch.Generator
) -> Tensor:
    """Stochastic EnKF: forecast each member, perturbed-observation analysis.

    ``x`` is ``[n, d]``. Same draws, in the same order, as
    ``classical_baselines.enkf_posterior`` when every member equals ``x0`` and
    ``A = Q = H = R = I``.
    """
    n = x.shape[0]
    H, R = system.H.to(x.dtype), system.R.to(x.dtype)
    x_f = system.propagate(x, generator)
    anomalies = x_f - x_f.mean(0, keepdim=True)
    P = anomalies.T @ anomalies / (n - 1)
    K = P @ H.T @ torch.linalg.inv(H @ P @ H.T + R)
    noise = torch.randn(n, system.m, generator=generator, dtype=x.dtype)
    y_pert = y.unsqueeze(0) + noise @ _chol(system.R).to(x.dtype).T
    return x_f + (y_pert - x_f @ H.T) @ K.T


def _systematic_indices(log_w: Tensor, generator: torch.Generator) -> Tensor:
    """Systematic resampling indices (the ``classical_baselines`` scheme)."""
    n = log_w.shape[0]
    w = torch.softmax(log_w, dim=0)
    positions = (torch.arange(n) + torch.rand(1, generator=generator)) / n
    idx = torch.searchsorted(torch.cumsum(w, dim=0), positions.clamp(max=1.0 - 1e-7))
    return idx.clamp(max=n - 1)


def _gaussian_log_w(residual: Tensor, cov: Tensor) -> Tensor:
    """``-1/2 r^T cov^{-1} r`` per row of ``residual``."""
    whitened = torch.linalg.solve_triangular(
        _chol(cov).to(residual.dtype), residual.T, upper=False
    )
    return -0.5 * (whitened**2).sum(0)


def particle_filter_step(
    system: LinearGaussianSSM, x: Tensor, y: Tensor, generator: torch.Generator
) -> Tensor:
    """Bootstrap particle filter: propagate, weight by ``N(y; H x, R)``, resample."""
    x_f = system.propagate(x, generator)
    residual = y.unsqueeze(0) - x_f @ system.H.to(x.dtype).T
    return x_f[_systematic_indices(_gaussian_log_w(residual, system.R), generator)]


def adapted_resample(
    system: LinearGaussianSSM, x: Tensor, y: Tensor, generator: torch.Generator
) -> Tensor:
    """Resample members by the predictive likelihood ``p(y^{k+1} | x^k_i)``.

    ``p(y | x) = N(y; H A x, H Q H^T + R)``. Resampling with these weights and
    then applying the per-member conditional is the fully adapted auxiliary
    particle filter, whose target IS the filtering posterior. This is the term a
    per-member sampler leaves out.
    """
    A, H = system.A.to(x.dtype), system.H.to(x.dtype)
    residual = y.unsqueeze(0) - x @ A.T @ H.T
    log_w = _gaussian_log_w(residual, system.innovation_cov(system.Q))
    return x[_systematic_indices(log_w.to(torch.float64), generator)]


# --------------------------------------------------------------------------- #
# Closed-form priors for x^1 | x^0 ~ N(A x^0, Q)
# --------------------------------------------------------------------------- #


class _LinearGaussianPrior(nn.Module):
    """Shared state: ``A`` and the eigendecomposition ``Q = U diag(q) U^T``."""

    def __init__(self, A: Tensor, Q: Tensor) -> None:
        super().__init__()
        q, U = torch.linalg.eigh(Q)
        self.register_buffer("A", A.to(torch.float32))
        self.register_buffer("q", q.to(torch.float32))
        self.register_buffer("U", U.to(torch.float32))
        # BasePosterior.device reads next(self.parameters()).
        self._device_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _time(self, t: Tensor, x: Tensor) -> Tensor:
        return _expand_t(torch.clamp(t, _T_MIN, _T_MAX), x)

    def _apply_q_function(self, coef: Tensor, v: Tensor) -> Tensor:
        """``U diag(coef) U^T v``, with ``coef`` broadcast over the last axis."""
        return (coef * (v @ self.U)) @ self.U.T


class AnalyticalLinearSIDriftModel(_LinearGaussianPrior):
    """SI drift for ``x1 | x0 ~ N(A x0, Q)`` (Prop. B.9 of chen_probabilistic_2024).

    With ``alpha = 1 - t``, ``beta = t``, ``gamma = g0 (1 - t)`` and
    ``mu = A x0``, the interpolant is ``x_t | x0 ~ N(m_t, C_t)``,
    ``m_t = alpha x0 + beta mu``, ``C_t = beta^2 Q + t gamma^2 I``, and

        b = (mu - x0) + (beta beta' Q + t gamma gamma' I) C_t^{-1} (x - m_t).

    For ``A = I``, ``Q = q I`` it reduces to ``AnalyticalSIDriftModel``.
    """

    def __init__(self, A: Tensor, Q: Tensor, g0: float = 1.0) -> None:
        super().__init__(A, Q)
        self.g0 = float(g0)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        field_history: Optional[Tensor] = None,
        field_cond: Optional[Tensor] = None,
        pars_cond: Optional[Tensor] = None,
    ) -> Tensor:
        t4 = self._time(t, x)
        x0 = field_history[..., -1]
        mu = x0 @ self.A.T
        gamma = self.g0 * (1.0 - t4)
        cov = t4**2 * self.q + t4 * gamma**2
        coef = (t4 * self.q - t4 * gamma * self.g0) / cov
        return (mu - x0) + self._apply_q_function(
            coef, x - ((1.0 - t4) * x0 + t4 * mu)
        )


class AnalyticalLinearFMVelocityModel(_LinearGaussianPrior):
    """FM velocity for target ``N(A x0, Q)`` on ``x_t = (1 - t) z + t x1``.

    ``x_t ~ N(t mu, C_t)``, ``C_t = (1 - t)^2 I + t^2 Q``, and with
    ``r = x - t mu`` the conditional means are ``E[x1 | x_t] = mu + t Q C_t^{-1} r``
    and ``E[z | x_t] = (1 - t) C_t^{-1} r``; the velocity is their difference.
    For ``A = I``, ``Q = q I`` it reduces to ``AnalyticalFMVelocityModel``.
    """

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        field_history: Optional[Tensor] = None,
        field_cond: Optional[Tensor] = None,
        pars_cond: Optional[Tensor] = None,
    ) -> Tensor:
        t4 = self._time(t, x)
        mu = field_history[..., -1] @ self.A.T
        alpha = 1.0 - t4
        cov = alpha**2 + t4**2 * self.q
        return mu + self._apply_q_function(
            (t4 * self.q - alpha) / cov, x - t4 * mu
        )


def build_linear_si_model(
    system: LinearGaussianSSM, g0: float = 1.0
) -> FollmerStochasticInterpolant:
    return FollmerStochasticInterpolant(
        interpolation=LinearStochasticInterpolation(gamma_multiplier=g0),
        drift_model=AnalyticalLinearSIDriftModel(system.A, system.Q, g0=g0),
    )


def build_linear_fm_model(system: LinearGaussianSSM) -> FlowMatchingModel:
    return FlowMatchingModel(
        interpolation=LinearDeterministicInterpolation(),
        drift_model=AnalyticalLinearFMVelocityModel(system.A, system.Q),
    )


class FullFirstStepDiffusionModel(DenoiseDiffusionModel):
    """``DenoiseDiffusionModel`` whose first step keeps the velocity.

    The parent's ``_compute_first_step`` drifts by ``1/2 g^2 s`` only. With the
    endpoint-vanishing ``g`` that is the identity at ``tau = 0``, so the first
    Euler step loses ``v dt``, which contains ``mu dt``: the sampler becomes
    sensitive to the size of the state. Harmless near the origin, where the
    single-step case lives; on a random walk the state grows and SDA / SURGE (SDA)
    pick up an offset proportional to it. Used only from cycle 2 on (see
    ``MultistepRunner._dm_model``), so cycle 1 stays the single-step case.
    """

    def _compute_first_step(
        self,
        base: Tensor,
        t: Tensor,
        dt: Tensor,
        field_history: Tensor,
        field_cond: Optional[Tensor] = None,
        pars_cond: Optional[Tensor] = None,
        **_: object,
    ) -> Tensor:
        drift = self.drift(base, t, field_history, field_cond, pars_cond)
        noise = self.diffusion_term(t) * torch.randn_like(base)
        return base + drift * dt + noise * torch.sqrt(dt)


class NullGuidance(nn.Module):
    """No guidance: SURGE then proposes from the prior, and only its weights act.

    The ablation that shows what the wrapped sampler contributes to a SURGE row.
    """

    def score(self, x: Tensor, **_: object) -> Tensor:
        return torch.zeros_like(x)

    def sde_weight(self, t: Tensor, diffusion_term: object) -> float:
        return 0.0


class OursSurgeDiffusionPosterior(SurgeFlowMatchingPosterior):
    """SURGE around Ours (DM-SDE): the stock FM SURGE, fed the score Ours needs.

    ``InterpolantGaussianLikelihood`` in FM mode takes the score ``s`` as an input,
    which the stock class does not pass. It is recovered from the reverse-SDE drift
    the loop already holds, ``d = v + 1/2 g^2 s``, with the identity
    ``_denoise`` uses, so this costs no extra network evaluation.
    """

    def _lik_kwargs(self, x, t, field_history, field_cond, pars_cond, prior_drift) -> dict:
        t_grid = t.reshape(t.shape[0], *([1] * (x.dim() - 1)))
        beta = self.interpolation.beta(t_grid)
        gain = (
            self.interpolation.velocity_score_coeff(t_grid)
            + 0.5 * self.model.diffusion_term(t_grid) ** 2
        )
        beta_diff = self.interpolation.beta_diff(t_grid)
        return {"score": (beta * prior_drift - beta_diff * x) / (beta * gain)}


class DenseObservationOperator(LinearObservationOperator):
    """Arbitrary dense ``H`` on the ``[B, 1, 1, d]`` layout.

    ``LinearObservationOperator`` builds only selection / block-average matrices.
    A ``type`` outside ``grid`` / ``random`` routes ``forward`` and ``transpose``
    through the dense matmul, so only the matrix has to be swapped in.
    """

    def __init__(self, matrix: Tensor) -> None:
        num_obs, d = matrix.shape
        super().__init__(type="random", data_size=(1, 1, d), percent_obs=1.0, seed=0)
        self.type = "dense"
        self.obs_matrix = matrix.to(torch.float32)
        self.num_obs = num_obs


__all__ = [
    "LinearGaussianSSM",
    "gaussian_samples",
    "enkf_step",
    "particle_filter_step",
    "adapted_resample",
    "AnalyticalLinearSIDriftModel",
    "AnalyticalLinearFMVelocityModel",
    "build_linear_si_model",
    "build_linear_fm_model",
    "FullFirstStepDiffusionModel",
    "NullGuidance",
    "OursSurgeDiffusionPosterior",
    "DenseObservationOperator",
]
