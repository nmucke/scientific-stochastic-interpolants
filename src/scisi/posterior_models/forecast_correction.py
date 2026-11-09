"""Forecast-residual correction for sequential (multi-step) assimilation.

Background and derivation: ``FORECAST_CORRECTION.md`` in the paper-experiments
directory. In short, the per-member sampler draws ``x'_i ~ p(x^{k+1} | x^k_i, y)``
and so drops the predictive weight ``p(y | x^k_i)``. The correction restores the
missing across-member part of the forecast covariance with a Kalman-type step,

    x''_i = x'_i + gain_scale * (K_ens - K_q) (y_i + e_i - H f_i),

where

* ``f_i`` is an independent UNGUIDED forecast from the same previous state
  ``x^k_i`` (noise independent of ``x'_i``),
* ``K_ens = P_f H' (H P_f H' + R)^-1`` is the (localised) ensemble gain, ``P_f``
  the sample covariance of the ``f_i`` within one group of members,
* ``K_q = Q_hat H' (H Q_hat H' + R)^-1`` is the gain the sampler already applies,
  with ``Q_hat`` an estimate of the transition-noise covariance,
* ``e_i ~ N(0, R)``.

Do NOT use ``y - H x'_i`` as the innovation: it under-corrects and gets the
covariance wrong (see the note, section 2).

Main entry points
-----------------
* :class:`ForecastCorrectionConfig` -- all settings, default ``enabled=False``.
* :class:`ForecastCorrection` -- the correction. It runs inside
  ``BasePosterior.sample``, right after ``_post_sample`` and before the new state
  is appended to ``field_history`` (so the last slot is still each member's
  previous state ``x^k_i``). Enable it either per call,
  ``posterior.sample_trajectory(..., forecast_correction=corr)`` (set for that
  call only), or persistently, ``posterior.forecast_correction = corr``, which
  also covers callers that drive ``posterior.sample`` themselves -- including
  single-step runs: when enabled, EVERY ``sample`` call is corrected. Free-running
  steps of ``sample_trajectory`` call the prior directly and are never corrected.
  ``ForecastCorrection.correct(x, f, y, ...)`` applies the update alone, for
  callers with their own forecasts.

Supported posteriors: ``StochasticInterpolantPosterior`` (SI-SDE, without
``resample``), ``FlowMatchingPosterior`` (FM-ODE and FM-SDE / DM-SDE),
``DiffusionPosterior`` and ``DFlowPosterior`` (its own ``sample`` calls the same
hook; note its realised gain is not ``K_q``, calibrate ``gain_scale``). SURGE
posteriors and SI with ``resample=True`` raise (see
:meth:`ForecastCorrection.check_supported`).
* Localisation: :class:`CoordinateLocalisation`, :func:`gaspari_cohn_kernel`,
  :func:`periodic_1d_localisation`, :func:`periodic_2d_localisation`,
  :func:`grid_localisation`.
* Q_hat bank: :class:`ResidualBank`, :func:`build_residual_bank`.
* :func:`draw_unguided_forecast` -- the unguided forecast used by the hook.

Units
-----
Everything runs in the MODEL's internal (standardised) units: the states the
posterior passes around, the observation rows handed to ``sample_trajectory`` and
the observation-noise variance stored on the likelihood
(``likelihood_model.original_variance``, already in model units in every case of
this repository). ``H`` and ``R`` are read from the posterior's likelihood
(``likelihood_model.obs_operator`` / ``original_variance``) unless overridden in
the config, so they can change between runs without touching the correction.

Cost
----
No ``d x d`` matrix is ever formed. Per group of ``E`` members the correction
builds ``m x m`` innovation covariances (Cholesky) and applies ``d x m`` gains in
row chunks of ``row_chunk`` state entries. Without localisation the gain is
applied through the ``E`` anomalies (``O(E d + E m)`` memory). With localisation
the ``d x m`` taper is built from coordinates chunk by chunk (and cached when it
has at most ``cache_limit`` entries). For a 128^2 state and the 32^2 block-average
operator (m = 1024) that is 16.8 M entries; one group costs about
``E d m`` multiply-adds (~1e9 for E = 64), and the bank gain ``K_q`` is cached as
one ``d x m`` matrix per ``(H, R)``.
"""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import torch

from scisi.particle_filter.enkf import gaspari_cohn

logger = logging.getLogger(__name__)

Tensor = torch.Tensor

#: Forecast modes understood by :func:`draw_unguided_forecast`.
FORECAST_MODES = ("prior", "free_run", "ode")
#: Sources for Q_hat.
Q_SOURCES = ("online", "bank")


# =========================================================================== #
# Localisation
# =========================================================================== #


def chordal_distance(delta: Tensor, period: float) -> Tensor:
    """Chordal distance on a circle of circumference ``period``.

    ``(period / pi) * sin(pi |delta| / period)``: the Euclidean distance between
    the two points embedded on the circle. It equals the arc length for small
    gaps, and keeps Gaspari-Cohn positive definite on a 1D periodic lattice
    (``scisi.particle_filter.enkf.periodic_taper`` uses the same distance).
    """
    return (period / math.pi) * torch.sin(math.pi * delta.abs() / period)


def gaspari_cohn_kernel(
    radius: Union[float, Sequence[float]],
    periods: Optional[Sequence[Optional[float]]] = None,
    combine: str = "euclidean",
) -> Callable[[Tensor, Tensor], Tensor]:
    """Gaspari-Cohn taper between two coordinate sets.

    Args:
        radius: Half-width ``c`` of the taper (support ``2c``), in the units of
            the coordinates. A sequence gives one radius per coordinate axis; then
            each axis is divided by its own radius and the taper uses radius 1.
        periods: Per-axis period (``None`` for a non-periodic axis). Periodic axes
            use :func:`chordal_distance`.
        combine: ``"euclidean"`` (default): isotropic taper of the Euclidean norm of
            the per-axis distances. Positive semidefinite (PSD) is guaranteed when
            the coordinates embed in at most 3 dimensions (non-periodic 1D/2D/3D, or
            one periodic axis plus one non-periodic one); on a 2D periodic grid the
            embedding is 4D and PSD is not guaranteed, though it holds numerically
            on the grids and radii tested. ``"product"``: product of 1D tapers, one
            per axis. Always PSD (Schur product of PSD matrices), but anisotropic.

    Returns:
        ``kernel(a, b)`` mapping ``a: [n, k]`` and ``b: [p, k]`` to ``[n, p]``.
    """
    if combine not in ("euclidean", "product"):
        raise ValueError(f"combine must be 'euclidean' or 'product', got {combine!r}")

    def kernel(a: Tensor, b: Tensor) -> Tensor:
        k = a.shape[1]
        radii = (
            torch.as_tensor(radius, dtype=a.dtype, device=a.device).reshape(-1)
            if not isinstance(radius, (int, float))
            else torch.full((k,), float(radius), dtype=a.dtype, device=a.device)
        )
        if radii.numel() == 1:
            radii = radii.expand(k)
        pers = list(periods) if periods is not None else [None] * k
        if len(pers) != k or radii.numel() != k:
            raise ValueError("radius / periods must match the coordinate dimension")
        out = None
        sq = None
        for j in range(k):
            delta = a[:, j : j + 1] - b[None, :, j]
            dist = (
                chordal_distance(delta, float(pers[j]))
                if pers[j] is not None
                else delta.abs()
            ) / radii[j]
            if combine == "product":
                taper_j = gaspari_cohn(dist, 1.0)
                out = taper_j if out is None else out * taper_j
            else:
                sq = dist**2 if sq is None else sq + dist**2
        if combine == "product":
            return out
        return gaspari_cohn(sq.sqrt(), 1.0)

    return kernel


class Localisation:
    """Interface: tapers between state and observation locations.

    Subclasses return the ``[rows, m]`` block of the state-observation taper and
    the ``[m, m]`` observation-observation taper for a given observation operator.
    Both are Schur-multiplied onto ``P_f H'`` and ``H P_f H'`` (and the same for
    ``Q_hat``). Pass ``None`` instead of a ``Localisation`` for no localisation.
    """

    def state_obs(self, operator: Any, rows: slice, ref: Tensor) -> Tensor:
        """Taper between state entries ``rows`` (flat index) and all observations."""
        raise NotImplementedError

    def obs_obs(self, operator: Any, ref: Tensor) -> Tensor:
        """``[m, m]`` taper between observations."""
        raise NotImplementedError


def _operator_key(operator: Any) -> tuple:
    return (
        id(operator),
        id(getattr(operator, "obs_matrix", None)),
        id(getattr(operator, "obs_indices", None)),
        int(getattr(operator, "num_obs", -1)),
    )


def _state_shape(operator: Any) -> tuple[int, int, int]:
    try:
        return int(operator.C), int(operator.H), int(operator.W)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise ValueError(
            "The observation operator must expose the state layout C, H, W "
            "(LinearObservationOperator does)."
        ) from exc


class CoordinateLocalisation(Localisation):
    """Localisation from coordinates and a kernel.

    Args:
        state_coords: ``[d, k]`` coordinates of every flat state entry, in the
            ``C, H, W`` flattening of the model state.
        kernel: ``kernel(a [n, k], b [p, k]) -> [n, p]`` taper, e.g.
            :func:`gaspari_cohn_kernel`. Must be PSD for ``H P H'`` to stay PSD.
        obs_coords: Optional fixed ``[m, k]`` observation coordinates. By default
            they are derived from the operator as ``H c / H 1`` for each coordinate
            field ``c``: exact for selection operators (the sensor location) and
            the block centre for block-average operators. This keeps the
            observation operator free at test time.
        cache_limit: Cache the full ``[d, m]`` taper per operator when it has at
            most this many entries; otherwise recompute row chunks on every call.
    """

    def __init__(
        self,
        state_coords: Tensor,
        kernel: Callable[[Tensor, Tensor], Tensor],
        obs_coords: Optional[Tensor] = None,
        cache_limit: int = 2**24,
    ) -> None:
        self.state_coords = state_coords.detach().to(torch.float64)
        self.kernel = kernel
        self.fixed_obs_coords = (
            None if obs_coords is None else obs_coords.detach().to(torch.float64)
        )
        self.cache_limit = int(cache_limit)
        self._cache: dict = {}

    # -- observation coordinates ------------------------------------------- #

    def obs_coords(self, operator: Any) -> Tensor:
        if self.fixed_obs_coords is not None:
            return self.fixed_obs_coords
        key = ("coords",) + _operator_key(operator)
        if key not in self._cache:
            C, H, W = _state_shape(operator)
            d, k = self.state_coords.shape
            if d != C * H * W:
                raise ValueError(
                    f"state_coords has {d} rows but the operator state has {C * H * W}"
                )
            # float32 (the states' dtype) so the operator's cached matrix keeps
            # its dtype; grid coordinates are exact in float32.
            fields = self.state_coords.T.reshape(k, C, H, W).to(torch.float32)
            ones = torch.ones(1, C, H, W, dtype=torch.float32)
            weight = operator(ones)  # [1, m]
            if (weight.abs() < 1e-12).any():
                raise ValueError(
                    "Cannot place observations whose operator row sums to zero; "
                    "pass obs_coords explicitly."
                )
            self._cache[key] = (
                (operator(fields) / weight).T.contiguous().to(torch.float64).cpu()
            )  # [m, k]
        return self._cache[key]

    # -- tapers --------------------------------------------------------------- #

    def _full_state_obs(self, operator: Any) -> Optional[Tensor]:
        key = ("xy",) + _operator_key(operator)
        if key in self._cache:
            return self._cache[key]
        d = self.state_coords.shape[0]
        m = int(operator.num_obs)
        if d * m > self.cache_limit:
            return None
        taper = self.kernel(self.state_coords, self.obs_coords(operator)).to(
            torch.float32
        )
        self._cache[key] = taper
        return taper

    def state_obs(self, operator: Any, rows: slice, ref: Tensor) -> Tensor:
        full = self._full_state_obs(operator)
        if full is not None:
            block = full[rows]
        else:
            block = self.kernel(self.state_coords[rows], self.obs_coords(operator))
        return block.to(device=ref.device, dtype=ref.dtype)

    def obs_obs(self, operator: Any, ref: Tensor) -> Tensor:
        key = ("yy",) + _operator_key(operator)
        if key not in self._cache:
            oc = self.obs_coords(operator)
            self._cache[key] = self.kernel(oc, oc)
        return self._cache[key].to(device=ref.device, dtype=ref.dtype)

    def full_taper(self) -> Tensor:
        """``[d, d]`` state-state taper. For tests and small problems only."""
        return self.kernel(self.state_coords, self.state_coords)


class _ChannelTaperKernel:
    """Spatial kernel times a ``[C, C]`` cross-channel taper (first column = channel)."""

    def __init__(self, spatial: Callable, channel_taper: Tensor) -> None:
        self.spatial = spatial
        self.channel_taper = channel_taper.to(torch.float64)

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        ca = a[:, 0].round().long()
        cb = b[:, 0].round().long()
        if (b[:, 0] - cb).abs().max() > 1e-6:
            raise ValueError(
                "channel_taper needs every observation to read a single channel."
            )
        taper = self.channel_taper.to(device=a.device, dtype=a.dtype)
        return self.spatial(a[:, 1:], b[:, 1:]) * taper[ca][:, cb]


def grid_localisation(
    shape: tuple[int, int, int],
    radius: Union[float, Sequence[float]],
    periodic: tuple[bool, bool] = (False, False),
    spacing: tuple[float, float] = (1.0, 1.0),
    channel_coords: Optional[Tensor] = None,
    channel_taper: Optional[Tensor] = None,
    combine: str = "euclidean",
    cache_limit: int = 2**24,
) -> CoordinateLocalisation:
    """Gaspari-Cohn localisation on a ``(C, H, W)`` grid, several channels per site.

    The spatial coordinates of entry ``(c, h, w)`` are ``(h * spacing[0],
    w * spacing[1])``; with ``periodic[j]`` the axis wraps with period
    ``n_j * spacing[j]``. Channels at the same site are at distance zero unless
    ``channel_coords`` gives them extra coordinates.

    Args:
        shape: State layout ``(C, H, W)``.
        radius: Gaspari-Cohn half-width in spacing units (scalar, or one per
            coordinate axis: ``[*channel_coords axes, H, W]``).
        periodic: Whether the H and W axes are periodic.
        spacing: Grid spacing along H and W.
        channel_coords: Optional ``[C, kc]`` extra coordinates per channel,
            non-periodic, e.g. the height of each vertical level when a 3D field
            is stacked into channels (``[C, 1]``). This is how a 3D grid is
            described.
        channel_taper: Optional ``[C, C]`` PSD cross-channel taper (e.g. to damp
            cross-variable covariances: ones on the diagonal, a smaller value
            off it). ``None`` keeps all cross-channel covariances.
        combine: See :func:`gaspari_cohn_kernel`.
        cache_limit: See :class:`CoordinateLocalisation`.
    """
    C, H, W = shape
    c, h, w = torch.meshgrid(
        torch.arange(C, dtype=torch.float64),
        torch.arange(H, dtype=torch.float64) * spacing[0],
        torch.arange(W, dtype=torch.float64) * spacing[1],
        indexing="ij",
    )
    columns = [c.reshape(-1)]
    periods: list[Optional[float]] = []
    if channel_coords is not None:
        cc = torch.as_tensor(channel_coords, dtype=torch.float64).reshape(C, -1)
        for j in range(cc.shape[1]):
            columns.append(cc[:, j][c.reshape(-1).long()])
            periods.append(None)
    columns += [h.reshape(-1), w.reshape(-1)]
    periods += [
        H * spacing[0] if periodic[0] else None,
        W * spacing[1] if periodic[1] else None,
    ]
    coords = torch.stack(columns, dim=1)
    spatial = gaspari_cohn_kernel(radius, periods=periods, combine=combine)
    if channel_taper is None:
        kernel: Callable = lambda a, b: spatial(a[:, 1:], b[:, 1:])
    else:
        kernel = _ChannelTaperKernel(spatial, torch.as_tensor(channel_taper))
    return CoordinateLocalisation(coords, kernel, cache_limit=cache_limit)


def periodic_1d_localisation(
    dim_state: int, radius: float, cache_limit: int = 2**24
) -> CoordinateLocalisation:
    """Gaspari-Cohn on a periodic 1D lattice stored as ``[1, 1, d]`` (Lorenz 96).

    Uses the chordal distance, like ``enkf.periodic_taper``; the state-state taper
    is identical to it.
    """
    return grid_localisation(
        (1, 1, dim_state), radius, periodic=(False, True), cache_limit=cache_limit
    )


def periodic_2d_localisation(
    shape: tuple[int, int, int],
    radius: Union[float, Sequence[float]],
    channel_taper: Optional[Tensor] = None,
    combine: str = "euclidean",
    cache_limit: int = 2**24,
) -> CoordinateLocalisation:
    """Gaspari-Cohn on a doubly periodic ``(C, H, W)`` grid (Navier-Stokes)."""
    return grid_localisation(
        shape,
        radius,
        periodic=(True, True),
        channel_taper=channel_taper,
        combine=combine,
        cache_limit=cache_limit,
    )


# =========================================================================== #
# Unguided forecasts
# =========================================================================== #


def _zero_diffusion(t: Tensor) -> Tensor:
    return torch.zeros_like(t)


def _is_fm_sde(posterior: Any) -> bool:
    from scisi.posterior_models.flow_matching_posterior import FlowMatchingPosterior

    return isinstance(posterior, FlowMatchingPosterior) and not posterior.is_ode


def _lifted_fm_drift(model: Any, diffusion_term: Callable) -> Callable:
    """Prior reverse-SDE drift ``v + 1/2 g^2 s`` of ``FlowMatchingPosterior``."""

    def drift(x, t, field_history=None, field_cond=None, pars_cond=None):
        v = model.drift(x, t, field_history, field_cond, pars_cond)
        t_exp = t.reshape(t.shape[0], *([1] * (x.dim() - 1))) if t.dim() < x.dim() else t
        s = model.interpolation.score_from_velocity(
            x=x, v=v, t=t_exp, a0=torch.zeros_like(x)
        )
        return v + 0.5 * diffusion_term(t) ** 2 * s

    return drift


def draw_unguided_forecast(
    posterior: Any,
    field_history: Tensor,
    num_steps: int,
    stepper: Callable,
    mode: str = "prior",
    field_cond: Optional[Tensor] = None,
    pars_cond: Optional[Tensor] = None,
    batch_size: int = 128,
) -> Tensor:
    """One unguided forecast per member from ``field_history`` (fresh noise).

    Args:
        posterior: The ``BasePosterior`` whose prior ``posterior.model`` is used.
        field_history: ``[N, C, H, W, L]``; ``[..., -1]`` is the current state.
        num_steps: Pseudo-time steps of the forecast solve.
        stepper: SDE/ODE step function (``euler_maruyama_step``, ``euler_step``...).
        mode:
            * ``"prior"`` (default): the unguided version of the posterior's own
              SDE/ODE. SI: ``model.sample`` with the posterior's diffusion (exactly
              the free-running branch of ``sample_trajectory``). FM-ODE: the FM ODE.
              FM-SDE (DM-SDE): the lifted reverse SDE ``v + 1/2 g^2 s`` with the
              posterior's ``g``, exactly the posterior's drift without guidance.
              Other models (e.g. ``DenoiseDiffusionModel``): as ``"free_run"``.
            * ``"free_run"``: exactly the call of the free-running branch of
              ``sample_trajectory``. NOTE: for an FM-SDE posterior that call adds
              ``g dW`` to the bare velocity ``v`` (no score lift), which inflates
              the forecast spread; prefer ``"prior"`` there.
            * ``"ode"``: FM models only: the deterministic FM ODE from a fresh
              ``N(0, I)`` latent (an exact prior draw in the continuum, usually the
              most accurate at few steps). Raises for SI (a point-mass source).
        field_cond, pars_cond: Conditioning for this physical step.
        batch_size: Only used to broadcast a single-member history.

    Returns:
        ``[N, C, H, W]`` forecasts, on ``field_history``'s device.
    """
    from scisi.models.flow_matching_model import FlowMatchingModel
    from scisi.models.follmer_stochastic_interpolant import (
        FollmerStochasticInterpolant,
    )

    if mode not in FORECAST_MODES:
        raise ValueError(f"forecast mode must be one of {FORECAST_MODES}, got {mode!r}")
    model = posterior.model
    base = None if posterior.gaussian_base else field_history[..., -1].clone()
    common = dict(
        field_history=field_history,
        field_cond=field_cond,
        pars_cond=pars_cond,
        num_steps=int(num_steps),
        batch_size=batch_size,
        return_field_history=False,
        stepper=stepper,
    )
    is_fm = isinstance(model, FlowMatchingModel)
    if mode == "prior" and not (is_fm or isinstance(model, FollmerStochasticInterpolant)):
        mode = "free_run"
    with torch.no_grad():
        if mode == "ode":
            if not is_fm:
                raise ValueError("forecast mode 'ode' needs a flow-matching prior.")
            f = model.sample(base=None, diffusion_term=_zero_diffusion, **common)
        elif mode == "free_run" or not is_fm:
            extra = (
                {}
                if getattr(posterior, "default_diffusion_term", True)
                else {"diffusion_term": posterior.diffusion_term}
            )
            f = model.sample(base=base, **common, **extra)
        elif _is_fm_sde(posterior):
            g = posterior.diffusion_term
            common.pop("return_field_history")
            f = model._sample(
                base=None,
                diffusion_term=g,
                drift=_lifted_fm_drift(model, g),
                **common,
            )
        else:  # FM-ODE (and D-Flow, whose flow is the FM ODE)
            f = model.sample(base=None, diffusion_term=_zero_diffusion, **common)
    return f.to(field_history.device)


# =========================================================================== #
# Q_hat bank
# =========================================================================== #


@dataclass
class ResidualBank:
    """Stored transition-noise residuals ``r = (f1 - f2) / sqrt(2)``.

    ``f1``, ``f2`` are two unguided forecasts from the same state, so
    ``Cov(r) = Q(x)``; the bank estimates the state-averaged ``Q``. Stored in
    model units, ``[N_b, C, H, W]``. ``Q_hat H'`` and ``H Q_hat H'`` are formed at
    use time from ``H`` applied to the bank, so ``H`` and ``R`` stay free.

    ``meta`` records how the bank was built (forecast steps, stepper, mode); keep
    it consistent with the forecasts used at test time (see
    :class:`ForecastCorrectionConfig`).
    """

    residuals: Tensor
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.residuals.shape[0])

    def save(self, path: Union[str, Path]) -> None:
        torch.save({"residuals": self.residuals.cpu(), "meta": dict(self.meta)}, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ResidualBank":
        data = torch.load(path, map_location="cpu")
        return cls(residuals=data["residuals"], meta=dict(data.get("meta", {})))


def build_residual_bank(
    posterior: Any,
    field_history: Tensor,
    num_steps: int,
    stepper: Callable,
    mode: str = "prior",
    pairs_per_state: int = 1,
    field_cond: Optional[Tensor] = None,
    pars_cond: Optional[Tensor] = None,
    batch_size: int = 128,
    path: Optional[Union[str, Path]] = None,
    seed: Optional[int] = None,
) -> ResidualBank:
    """Build (and optionally save) a residual bank from a set of states.

    Args:
        posterior: The posterior whose prior is used (the same object, or the same
            kind, as at test time, so the forecast definition matches).
        field_history: ``[S, C, H, W, L]`` histories of validation or training
            states (model units).
        num_steps: Forecast pseudo-time steps. Use the same value as
            ``forecast_num_steps`` at test time so ``Q_hat`` matches the
            forecasts that enter ``K_ens``.
        stepper, mode: As :func:`draw_unguided_forecast`.
        pairs_per_state: Forecast pairs per state; ``N_b = S * pairs_per_state``.
        field_cond, pars_cond: Conditioning, one row per state (or ``None``).
        batch_size: States per forecast call.
        path: If given, save the bank there (``ResidualBank.load`` reads it).
        seed: Optional ``torch.manual_seed`` before drawing.
    """
    if seed is not None:
        torch.manual_seed(int(seed))
    S = field_history.shape[0]
    chunks = []
    for _ in range(int(pairs_per_state)):
        for start in range(0, S, batch_size):
            sl = slice(start, min(start + batch_size, S))
            kw = dict(
                num_steps=num_steps,
                stepper=stepper,
                mode=mode,
                field_cond=field_cond[sl] if field_cond is not None else None,
                pars_cond=pars_cond[sl] if pars_cond is not None else None,
                batch_size=batch_size,
            )
            f1 = draw_unguided_forecast(posterior, field_history[sl], **kw)
            f2 = draw_unguided_forecast(posterior, field_history[sl], **kw)
            chunks.append(((f1 - f2) / math.sqrt(2.0)).float().cpu())
    bank = ResidualBank(
        residuals=torch.cat(chunks, dim=0),
        meta={
            "num_steps": int(num_steps),
            "stepper": getattr(stepper, "__name__", repr(stepper)),
            "mode": mode,
            "num_states": int(S),
            "pairs_per_state": int(pairs_per_state),
        },
    )
    if path is not None:
        bank.save(path)
    return bank


# =========================================================================== #
# Config, diagnostics, correction
# =========================================================================== #


@dataclass
class ForecastCorrectionConfig:
    """Settings of the forecast-residual correction.

    Attributes:
        enabled: Master switch. ``False`` (default) leaves ``sample`` and
            ``sample_trajectory`` bit-for-bit unchanged (no extra random draws).
        forecast_num_steps: Pseudo-time steps of the unguided forecast ``f_i``
            (and of the online residual forecast). ``None`` (default): the
            sampler's ``num_steps``. Fewer steps make the correction cheap but
            the ``f_i`` become approximate prior draws (see below).
        forecast_stepper: Step function for the forecast; ``None``: the sampler's
            ``stepper``.
        forecast_mode: ``"prior"`` (default), ``"free_run"`` or ``"ode"``; see
            :func:`draw_unguided_forecast`.
        forecast_fn: Optional custom forecast,
            ``forecast_fn(posterior, field_history, field_cond, pars_cond,
            num_steps, stepper) -> [N, C, H, W]``. Overrides ``forecast_mode``.
        localisation: A :class:`Localisation` or ``None`` (no localisation).
        gain_scale: Multiplier on ``K_ens - K_q`` (default 1, the derivation). For
            calibration against the sampler's realised gain (note, 4.4); tune on
            validation data only.
        q_source: ``"online"`` (default) or ``"bank"``. See "Choosing Q_hat".
        residual_bank: The bank for ``q_source="bank"``: a :class:`ResidualBank`,
            a ``[N_b, C, H, W]`` tensor, or a path saved by ``ResidualBank.save``.
        group_size: Members per independent sequence when several sequences are
            batched together, members of one sequence contiguous. Ensemble
            statistics are computed per group only. ``None``: the whole batch is
            one ensemble. Must divide the batch size.
        update_cap: Optional cap on the update size in Mahalanobis units: with
            ``u_i = H delta_i`` and ``S = H P_f H' + R`` (localised), the size is
            ``sqrt(u_i' S^-1 u_i / m)`` (RMS per observation). Updates above the
            cap are scaled down to it; the fraction capped is logged. ``None``:
            no cap.
        post_update: Optional ``post_update(x_corrected, x_guided) -> Tensor``
            applied after the update, e.g. a divergence-free projection and the
            obstacle mask for uDALES. Shapes ``[N, C, H, W]``, model units.
        obs_operator: Override for ``H``. ``None``: the posterior's
            ``likelihood_model.obs_operator`` (or ``posterior.obs_operator``).
        obs_noise_var: Override for ``R`` in model units: scalar or ``[m]``
            (diagonal). ``None``: ``likelihood_model.original_variance``.
        forecast_seed: If set, the forecasts and the perturbations ``e_i`` are drawn
            from a forked RNG seeded with ``forecast_seed + call index``, so the
            global stream the guided sampler sees is the same as without the
            correction. ``None`` (default): drawn from the global stream after the
            guided sample (still independent of it).
        compute_dtype: dtype of the linear algebra (default ``float64``); ``None``
            keeps the state dtype.
        row_chunk: State rows per chunk when applying ``d x m`` gains.
        callback: Optional ``callback(record: dict)`` called with each step's
            diagnostics record.

    Choosing Q_hat. ``"online"`` draws a second unguided forecast per member each
    step and uses ``r_i = (f_i - f'_i) / sqrt(2)``: no offline artefact, ``Q_hat``
    follows the current state, and it is built from the SAME forecast solver as
    ``K_ens``, so a few-step solver's errors enter both gains alike. It costs one
    more prior run per member and step, and ``Q_hat`` then has rank ``E``
    (localise). ``"bank"`` averages ``Q(x)`` over the bank's states: cheaper per
    step and low-noise, but biased where ``Q`` is state-dependent, and it must be
    built with the same forecast steps/stepper used at test time. Default:
    ``"online"``, because it needs nothing beyond the trained prior and stays
    consistent with whatever ``forecast_num_steps`` is chosen.

    Few-step forecasts are approximate prior draws: their spread is not exactly
    ``Q`` and their mean not exactly ``mu_i``. That biases ``K_ens`` (through
    ``P_f``) and, for a bank built at a different step count, makes ``K_q``
    inconsistent with it. Check the forecast spread against a full-step run on
    validation data before trusting a very small ``forecast_num_steps``.
    """

    enabled: bool = False
    forecast_num_steps: Optional[int] = None
    forecast_stepper: Optional[Callable] = None
    forecast_mode: str = "prior"
    forecast_fn: Optional[Callable] = None
    localisation: Optional[Localisation] = None
    gain_scale: float = 1.0
    q_source: str = "online"
    residual_bank: Optional[Union[ResidualBank, Tensor, str, Path]] = None
    group_size: Optional[int] = None
    update_cap: Optional[float] = None
    post_update: Optional[Callable[[Tensor, Tensor], Tensor]] = None
    obs_operator: Optional[Any] = None
    obs_noise_var: Optional[Union[float, Tensor]] = None
    forecast_seed: Optional[int] = None
    compute_dtype: Optional[torch.dtype] = torch.float64
    row_chunk: int = 4096
    callback: Optional[Callable[[dict], None]] = None

    def __post_init__(self) -> None:
        if self.forecast_mode not in FORECAST_MODES:
            raise ValueError(f"forecast_mode must be one of {FORECAST_MODES}")
        if self.q_source not in Q_SOURCES:
            raise ValueError(f"q_source must be one of {Q_SOURCES}")
        if self.forecast_num_steps is not None and int(self.forecast_num_steps) < 1:
            raise ValueError("forecast_num_steps must be >= 1")
        if self.group_size is not None and int(self.group_size) < 2:
            raise ValueError("group_size must be >= 2")
        if self.update_cap is not None and float(self.update_cap) <= 0:
            raise ValueError("update_cap must be positive")


class CorrectionDiagnostics:
    """Per-step diagnostics of a :class:`ForecastCorrection`.

    ``records`` holds one dict per corrected step with keys (tensors are ``[G]``,
    one entry per group):

    * ``phys_idx``: physical step index (``None`` outside ``sample_trajectory``).
    * ``innovation_mean``: mean over observations of ``ybar - H fbar``, the
      forecast innovation of the group (model units).
    * ``innovation_msq``: mean over observations of the normalised innovation
      squared, ``(ybar - H fbar)_j^2 / S_jj`` with ``S = H P_f H' + R``.
    * ``innovation_lag1``: mean over observations of the product of this step's
      and the previous step's normalised innovations (``None`` at the first step
      or when ``m`` changed).
    * ``update_rms``: RMS of the update ``x'' - x'`` over members and entries.
    * ``spread_rms``: RMS forecast spread (std of ``f``), for scale.
    * ``cap_fraction``: fraction of members whose update was capped.

    :meth:`lag1_autocorrelation` aggregates the lag-1 autocorrelation of the
    normalised innovations, the usable symptom of the per-member bias (note,
    section 1); it should be near zero for a well-calibrated filter.
    """

    def __init__(self) -> None:
        self.records: list[dict] = []
        self._prev: Optional[Tensor] = None

    def reset(self) -> None:
        self.records = []
        self._prev = None

    def _lag(self, normalised: Tensor) -> Optional[Tensor]:
        prev = self._prev
        self._prev = normalised
        if prev is None or prev.shape != normalised.shape:
            return None
        return (normalised * prev).mean(dim=1)

    def lag1_autocorrelation(self) -> Optional[Tensor]:
        """``[G]`` lag-1 autocorrelation of the normalised innovations over steps."""
        num = 0.0
        a = 0.0
        b = 0.0
        prev_msq = None
        any_lag = False
        for rec in self.records:
            if rec["innovation_lag1"] is not None and prev_msq is not None:
                num = num + rec["innovation_lag1"]
                a = a + rec["innovation_msq"]
                b = b + prev_msq
                any_lag = True
            prev_msq = rec["innovation_msq"]
        if not any_lag:
            return None
        return num / torch.sqrt(a * b).clamp_min(1e-30)

    def as_dict(self) -> dict:
        """Stack the per-step records: ``{key: [steps, G]}`` (lag1 NaN-padded)."""
        if not self.records:
            return {}
        out: dict = {"phys_idx": [r["phys_idx"] for r in self.records]}
        for key in (
            "innovation_mean",
            "innovation_msq",
            "update_rms",
            "spread_rms",
            "cap_fraction",
        ):
            out[key] = torch.stack([r[key] for r in self.records])
        G = self.records[0]["innovation_mean"].shape[0]
        out["innovation_lag1"] = torch.stack(
            [
                r["innovation_lag1"]
                if r["innovation_lag1"] is not None
                else torch.full((G,), float("nan"))
                for r in self.records
            ]
        )
        return out


def _cholesky(S: Tensor) -> Tensor:
    L, info = torch.linalg.cholesky_ex(S)
    if int(info) == 0:
        return L
    # Indefinite through a non-PSD taper or round-off: add a small jitter.
    jitter = 1e-10 * S.diagonal().abs().mean().clamp_min(1e-30)
    for _ in range(8):
        L, info = torch.linalg.cholesky_ex(
            S + jitter * torch.eye(S.shape[0], dtype=S.dtype, device=S.device)
        )
        if int(info) == 0:
            logger.warning("forecast correction: jittered an indefinite S by %g", jitter)
            return L
        jitter = jitter * 100
    raise RuntimeError("forecast correction: innovation covariance is not PD")


class ForecastCorrection:
    """The forecast-residual correction (see module docstring).

    Use with ``sample_trajectory``::

        corr = ForecastCorrection(ForecastCorrectionConfig(enabled=True, ...))
        traj = posterior.sample_trajectory(..., forecast_correction=corr)
        corr.diagnostics.lag1_autocorrelation()

    or with ``sample`` (every call is corrected while it is set)::

        posterior.forecast_correction = corr
        x_new = posterior.sample(...)

    or directly, for a caller that runs the guided sampler itself::

        x_new = corr(x_guided, f, y, operator=H, obs_noise_var=r)

    Attributes:
        config: The :class:`ForecastCorrectionConfig`.
        diagnostics: :class:`CorrectionDiagnostics` of the last run (reset at the
            start of every ``sample_trajectory``).
    """

    def __init__(self, config: Optional[ForecastCorrectionConfig] = None, **kwargs: Any):
        if config is None:
            config = ForecastCorrectionConfig(**kwargs)
        elif kwargs:
            raise TypeError("pass either a config or keyword arguments, not both")
        self.config = config
        self.diagnostics = CorrectionDiagnostics()
        self._kq_cache: dict = {}
        self._bank: Optional[Tensor] = None
        self._calls = 0
        #: Physical step index recorded in the diagnostics; set by
        #: ``sample_trajectory`` before each assimilation step (else ``None``).
        self.current_phys_idx: Optional[int] = None

    # ------------------------------------------------------------------ #
    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def reset(self) -> None:
        """Clear diagnostics and the lag state (not the K_q cache)."""
        self.diagnostics.reset()
        self._calls = 0
        self.current_phys_idx = None

    @staticmethod
    def resolve(
        value: Union["ForecastCorrection", ForecastCorrectionConfig, None],
    ) -> Optional["ForecastCorrection"]:
        """Turn a ``sample_trajectory`` argument into an enabled correction or None."""
        if value is None:
            return None
        if isinstance(value, ForecastCorrectionConfig):
            value = ForecastCorrection(value)
        if not isinstance(value, ForecastCorrection):
            raise TypeError(
                "forecast_correction must be a ForecastCorrection, a "
                f"ForecastCorrectionConfig or None, got {type(value).__name__}"
            )
        return value if value.enabled else None

    def check_supported(self, posterior: Any) -> None:
        """Raise for posteriors the correction must not be combined with.

        SMC posteriors (SURGE; SI with ``resample=True``) reweight and resample
        members by the observation likelihood inside ``sample``. They resample
        ``base`` and ``field_history`` together, so the member <-> parent
        pairing survives, but the correction would then apply a second update
        for the same observation on top of weights that already (in SURGE's
        case, exactly) target the filtering posterior: the observation would be
        counted twice. Neither order (before or after their resampling) fixes
        that, so they are refused rather than ordered.
        """
        if not getattr(posterior, "supports_forecast_correction", True):
            raise NotImplementedError(
                f"{type(posterior).__name__} does not support the forecast "
                "correction: it is an SMC sampler whose weights already account "
                "for the observation; correcting on top would count it twice."
            )
        if bool(getattr(posterior, "resample", False)):
            raise NotImplementedError(
                f"{type(posterior).__name__} with resample=True does not support "
                "the forecast correction: its SMC reweighting already uses the "
                "observation (it would be counted twice)."
            )
        if self.config.q_source == "bank" and self.config.residual_bank is None:
            raise ValueError("q_source='bank' needs config.residual_bank")

    # ------------------------------------------------------------------ #
    def observation_model(self, posterior: Any = None) -> tuple[Any, Any]:
        """``(H, R)`` from the config, else from the posterior's likelihood."""
        op = self.config.obs_operator
        var = self.config.obs_noise_var
        if posterior is not None:
            lik = getattr(posterior, "likelihood_model", None)
            if op is None:
                op = getattr(lik, "obs_operator", None)
                if op is None:
                    op = getattr(posterior, "obs_operator", None)
            if var is None:
                var = getattr(lik, "original_variance", None)
                if var is None:
                    var = getattr(posterior, "original_variance", None)
                if var is None:
                    var = getattr(posterior, "variance", None)
        if op is None or var is None:
            raise ValueError(
                "forecast correction: no observation operator / noise variance; "
                "set config.obs_operator and config.obs_noise_var."
            )
        return op, var

    def _bank_tensor(self) -> Tensor:
        if self._bank is None:
            bank = self.config.residual_bank
            if isinstance(bank, (str, Path)):
                bank = ResidualBank.load(bank)
            if isinstance(bank, ResidualBank):
                bank = bank.residuals
            if not torch.is_tensor(bank):
                raise ValueError("residual_bank must be a ResidualBank, tensor or path")
            self._bank = bank
        return self._bank

    # ------------------------------------------------------------------ #
    # Forecast
    # ------------------------------------------------------------------ #

    def draw_forecast(
        self,
        posterior: Any,
        field_history: Tensor,
        num_steps: int,
        stepper: Callable,
        field_cond: Optional[Tensor] = None,
        pars_cond: Optional[Tensor] = None,
        batch_size: int = 128,
    ) -> Tensor:
        """One unguided forecast per member with the configured steps/stepper."""
        cfg = self.config
        steps = int(cfg.forecast_num_steps or num_steps)
        fstep = cfg.forecast_stepper or stepper
        if cfg.forecast_fn is not None:
            with torch.no_grad():
                return cfg.forecast_fn(
                    posterior, field_history, field_cond, pars_cond, steps, fstep
                )
        return draw_unguided_forecast(
            posterior,
            field_history,
            num_steps=steps,
            stepper=fstep,
            mode=cfg.forecast_mode,
            field_cond=field_cond,
            pars_cond=pars_cond,
            batch_size=batch_size,
        )

    @contextlib.contextmanager
    def _rng_context(self, ref: Tensor):
        """Forked, seeded RNG when ``forecast_seed`` is set; else a no-op."""
        seed = self.config.forecast_seed
        if seed is None:
            yield
            return
        devices = [ref.device.index or 0] if ref.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed) + self._calls)
            yield

    # ------------------------------------------------------------------ #
    # Hook used by BasePosterior.sample (and DFlowPosterior.sample)
    # ------------------------------------------------------------------ #

    def apply(
        self,
        posterior: Any,
        base: Tensor,
        prev_history: Tensor,
        observations: Tensor,
        num_steps: int,
        stepper: Callable,
        field_cond: Optional[Tensor] = None,
        pars_cond: Optional[Tensor] = None,
        batch_size: int = 128,
    ) -> Tensor:
        """Correct one guided sample; called by ``BasePosterior.sample``.

        Args:
            posterior: The posterior (its prior draws the forecasts; H and R come
                from its likelihood unless overridden).
            base: ``[N, C, H, W]`` guided samples ``x'_i``.
            prev_history: ``[N, C, H, W, L]`` history whose last slot is the
                previous state ``x^k_i`` of each member (the history BEFORE the
                new state is appended).
            observations, field_cond, pars_cond: This step's inputs to ``sample``.
            num_steps, stepper: The sampler's; ``forecast_num_steps`` /
                ``forecast_stepper`` override them for the forecast.

        Returns:
            ``[N, C, H, W]`` corrected samples, same dtype and device as ``base``.
            The phys index recorded in the diagnostics is
            ``self.current_phys_idx`` (set by ``sample_trajectory``).
        """
        operator, obs_var = self.observation_model(posterior)
        with torch.no_grad(), self._rng_context(base):
            fkw = dict(
                num_steps=num_steps,
                stepper=stepper,
                field_cond=field_cond,
                pars_cond=pars_cond,
                batch_size=batch_size,
            )
            f = self.draw_forecast(posterior, prev_history, **fkw).to(base.device)
            residuals = None
            if self.config.q_source == "online":
                f2 = self.draw_forecast(posterior, prev_history, **fkw)
                residuals = (f - f2.to(base.device)) / math.sqrt(2.0)
            new_base = self.correct(
                base,
                f,
                observations,
                operator=operator,
                obs_noise_var=obs_var,
                residuals=residuals,
                phys_idx=self.current_phys_idx,
            )
        self._calls += 1
        return new_base.to(dtype=base.dtype, device=base.device)

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        """Alias of :meth:`correct`."""
        return self.correct(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # The update itself
    # ------------------------------------------------------------------ #

    def correct(
        self,
        x: Tensor,
        f: Tensor,
        y: Tensor,
        operator: Any = None,
        obs_noise_var: Any = None,
        residuals: Optional[Tensor] = None,
        perturbations: Optional[Tensor] = None,
        phys_idx: Optional[int] = None,
    ) -> Tensor:
        """Apply ``x'' = x' + gain_scale (K_ens - K_q)(y + e - H f)``.

        Args:
            x: ``[N, C, H, W]`` guided samples ``x'_i`` (model units).
            f: ``[N, C, H, W]`` independent unguided forecasts ``f_i``.
            y: Observations, ``[1, m]`` (shared), ``[N, m]`` (one row per member)
                or ``[G, m]`` (one row per group).
            operator: ``H``; default from the config.
            obs_noise_var: ``R`` (scalar or ``[m]``); default from the config.
            residuals: ``[N, C, H, W]`` online residuals (``q_source="online"``).
            perturbations: Optional ``[N, m]`` ``e_i`` to use instead of fresh
                ``N(0, R)`` draws (for tests and paired comparisons).
            phys_idx: Recorded in the diagnostics.

        Returns:
            ``[N, C, H, W]`` corrected samples (``post_update`` applied).
        """
        cfg = self.config
        if operator is None or obs_noise_var is None:
            op, var = self.observation_model(None)
            operator = op if operator is None else operator
            obs_noise_var = var if obs_noise_var is None else obs_noise_var
        if cfg.q_source == "online" and residuals is None:
            raise ValueError("q_source='online' needs residuals (a second forecast)")

        N = x.shape[0]
        E = int(cfg.group_size or N)
        if N % E != 0:
            raise ValueError(f"batch {N} is not a multiple of group_size {E}")
        if E < 2:
            raise ValueError("the correction needs at least 2 members per group")
        G = N // E
        dtype = cfg.compute_dtype or x.dtype
        dev = x.device
        shape = x.shape
        d = x[0].numel()

        sdtype = x.dtype

        def apply_H(X: Tensor) -> Tensor:
            # H in the state dtype, so the operator's cached matrix is not
            # re-cast back and forth between the sampler and the correction.
            Xs = X.to(device=dev, dtype=sdtype).reshape((-1,) + tuple(shape[1:]))
            return operator(Xs).to(dtype)

        F = f.reshape(N, -1).to(device=dev, dtype=dtype)
        HF = apply_H(f)  # [N, m]
        m = HF.shape[1]
        R = torch.as_tensor(obs_noise_var, dtype=dtype, device=dev).reshape(-1)
        if R.numel() == 1:
            R = R.expand(m)
        if R.numel() != m:
            raise ValueError(f"obs_noise_var has {R.numel()} entries, expected {m}")

        Y = y.to(device=dev, dtype=dtype).reshape(-1, m)
        if Y.shape[0] == 1:
            Y = Y.expand(N, m)
        elif Y.shape[0] == G and G != N:
            Y = Y.repeat_interleave(E, dim=0)
        elif Y.shape[0] != N:
            raise ValueError(
                f"observations have {Y.shape[0]} rows; expected 1, {G} or {N}"
            )

        if perturbations is None:
            e = torch.randn(N, m, dtype=dtype, device=dev) * R.sqrt()
        else:
            e = perturbations.to(device=dev, dtype=dtype).reshape(N, m)
        V = Y + e - HF  # [N, m] perturbed innovations

        loc = cfg.localisation
        loc_yy = loc.obs_obs(operator, F) if loc is not None else None
        Kq_bank = (
            self._bank_gain(operator, R, loc, dtype, dev, shape, sdtype)
            if cfg.q_source == "bank"
            else None
        )
        if residuals is not None:
            Rres = residuals.reshape(N, -1).to(device=dev, dtype=dtype)
            HRres = apply_H(residuals)

        delta = torch.empty(N, d, dtype=dtype, device=dev)
        stats = {k: [] for k in ("mean", "msq", "norm", "upd", "spr", "cap")}
        for g in range(G):
            sl = slice(g * E, (g + 1) * E)
            Fg, HFg, Vg = F[sl], HF[sl], V[sl]
            A = Fg - Fg.mean(0, keepdim=True)
            AH = HFg - HFg.mean(0, keepdim=True)
            S_ens = self._obs_cov(AH, 1.0 / (E - 1), loc_yy) + torch.diag(R)
            L_ens = _cholesky(S_ens)
            Z_ens = torch.cholesky_solve(Vg.T, L_ens)  # [m, E]
            dg = self._apply_gain(A, AH, 1.0 / (E - 1), Z_ens, loc, operator)
            if Kq_bank is not None:
                dg = dg - Vg @ Kq_bank.T
            else:
                Rg, HRg = Rres[sl], HRres[sl]
                S_q = self._obs_cov(HRg, 1.0 / E, loc_yy) + torch.diag(R)
                Z_q = torch.cholesky_solve(Vg.T, _cholesky(S_q))
                dg = dg - self._apply_gain(Rg, HRg, 1.0 / E, Z_q, loc, operator)
            dg = float(cfg.gain_scale) * dg

            capped = torch.zeros(E, dtype=torch.bool, device=dev)
            if cfg.update_cap is not None:
                U = apply_H(dg)  # [E, m]
                W = torch.linalg.solve_triangular(L_ens, U.T, upper=False)
                size = (W**2).mean(0).sqrt()  # [E]
                capped = size > float(cfg.update_cap)
                scale = torch.where(
                    capped, float(cfg.update_cap) / size.clamp_min(1e-30), 1.0
                )
                dg = dg * scale[:, None]
            delta[sl] = dg

            # Diagnostics: forecast innovation of the group.
            innov = Y[sl].mean(0) - HFg.mean(0)
            normalised = innov / S_ens.diagonal().sqrt()
            stats["mean"].append(innov.mean())
            stats["msq"].append((normalised**2).mean())
            stats["norm"].append(normalised)
            stats["upd"].append(dg.pow(2).mean().sqrt())
            stats["spr"].append(A.pow(2).sum(0).div(E - 1).mean().sqrt())
            stats["cap"].append(capped.to(dtype).mean())

        x_new = (x.reshape(N, -1).to(dtype) + delta).to(x.dtype).reshape(shape)
        if cfg.post_update is not None:
            x_new = cfg.post_update(x_new, x)

        record = {
            "phys_idx": phys_idx,
            "innovation_mean": torch.stack(stats["mean"]).cpu(),
            "innovation_msq": torch.stack(stats["msq"]).cpu(),
            "innovation_lag1": None,
            "update_rms": torch.stack(stats["upd"]).cpu(),
            "spread_rms": torch.stack(stats["spr"]).cpu(),
            "cap_fraction": torch.stack(stats["cap"]).cpu(),
        }
        lag = self.diagnostics._lag(torch.stack(stats["norm"]).cpu())
        record["innovation_lag1"] = lag
        self.diagnostics.records.append(record)
        if cfg.callback is not None:
            cfg.callback(record)
        return x_new

    # ------------------------------------------------------------------ #
    # Linear-algebra helpers (never d x d)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _obs_cov(HX: Tensor, scale: float, loc_yy: Optional[Tensor]) -> Tensor:
        """``loc_yy o (HX' HX) * scale``: ``[m, m]``."""
        C = (HX.T @ HX) * scale
        return C * loc_yy if loc_yy is not None else C

    def _apply_gain(
        self,
        X: Tensor,
        HX: Tensor,
        scale: float,
        Z: Tensor,
        loc: Optional[Localisation],
        operator: Any,
    ) -> Tensor:
        """``[(loc_xy o (X' HX) * scale) Z]'``: ``[E_out, d]`` for ``Z: [m, E_out]``."""
        if loc is None:
            return ((HX @ Z).T @ X) * scale  # [E_out, d], no d x m matrix
        d = X.shape[1]
        out = torch.empty(Z.shape[1], d, dtype=X.dtype, device=X.device)
        chunk = max(1, int(self.config.row_chunk))
        for start in range(0, d, chunk):
            rows = slice(start, min(start + chunk, d))
            block = (X[:, rows].T @ HX) * scale  # [r, m]
            block = block * loc.state_obs(operator, rows, block)
            out[:, rows] = (block @ Z).T
        return out

    def _bank_gain(
        self,
        operator: Any,
        R: Tensor,
        loc: Optional[Localisation],
        dtype: torch.dtype,
        dev: torch.device,
        shape: torch.Size,
        sdtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """``K_q = (loc o Q_hat H') (loc o H Q_hat H' + R)^-1``, ``[d, m]``, cached."""
        key = (
            _operator_key(operator),
            tuple(R.detach().cpu().tolist()) if R.numel() <= 4096 else hash(R.cpu().numpy().tobytes()),
            id(loc),
            dtype,
            str(dev),
        )
        if key in self._kq_cache:
            return self._kq_cache[key]
        bank = self._bank_tensor()
        Nb = bank.shape[0]
        if tuple(bank.shape[1:]) != tuple(shape[1:]):
            raise ValueError(
                f"residual bank fields {tuple(bank.shape[1:])} do not match the "
                f"state {tuple(shape[1:])}"
            )
        Rb = bank.reshape(Nb, -1).to(device=dev, dtype=dtype)
        HRb = torch.cat(
            [
                operator(
                    Rb[i : i + 256].to(sdtype).reshape((-1,) + tuple(shape[1:]))
                ).to(dtype)
                for i in range(0, Nb, 256)
            ]
        )
        loc_yy = loc.obs_obs(operator, Rb) if loc is not None else None
        S_q = self._obs_cov(HRb, 1.0 / Nb, loc_yy) + torch.diag(R)
        L = _cholesky(S_q)
        d = Rb.shape[1]
        m = HRb.shape[1]
        K = torch.empty(d, m, dtype=dtype, device=dev)
        chunk = max(1, int(self.config.row_chunk))
        for start in range(0, d, chunk):
            rows = slice(start, min(start + chunk, d))
            block = (Rb[:, rows].T @ HRb) / Nb  # [r, m] = Q_hat H' rows
            if loc is not None:
                block = block * loc.state_obs(operator, rows, block)
            K[rows] = torch.cholesky_solve(block.T, L).T
        self._kq_cache[key] = K
        return K


__all__ = [
    "CoordinateLocalisation",
    "CorrectionDiagnostics",
    "ForecastCorrection",
    "ForecastCorrectionConfig",
    "Localisation",
    "ResidualBank",
    "build_residual_bank",
    "chordal_distance",
    "draw_unguided_forecast",
    "gaspari_cohn_kernel",
    "grid_localisation",
    "periodic_1d_localisation",
    "periodic_2d_localisation",
]
