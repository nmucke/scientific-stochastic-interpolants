"""Forecast-residual correction for sequential (multi-step) assimilation.

Background: ``FORECAST_CORRECTION.md`` (paper-experiments directory). The per-member
sampler draws ``x'_i ~ p(x^{k+1} | x^k_i, y)`` and drops the predictive weight
``p(y | x^k_i)``. The correction restores the across-member forecast covariance:

    x''_i = x'_i + gain_scale * (K_ens - K_q) (y_i + e_i - H f_i),

``f_i`` an independent UNGUIDED forecast from ``x^k_i``, ``K_ens`` the (localised)
gain of the sample covariance of the ``f_i`` in one group, ``K_q`` the gain of the
transition noise ``Q_hat`` (what the sampler already applies), ``e_i ~ N(0, R)``.
Do NOT use ``y - H x'_i`` as the innovation (note, section 2).

Use ``posterior.sample_trajectory(..., forecast_correction=corr)`` (that call only)
or ``posterior.forecast_correction = corr`` (every ``sample`` call): the hook runs in
``BasePosterior.sample`` after ``_post_sample``, before the new state is appended.
:meth:`ForecastCorrection.correct` applies the update alone.

Model (standardised) units throughout; ``H``, ``R`` are the likelihood's
``obs_operator`` / ``original_variance``. The online ``Q_hat`` reuses ``f_i``, which
also enters the innovation (an O(1/E) bias). No ``d x d`` matrix is formed; linear
algebra is float64, on the CPU when the state is on MPS.

Nonlinear ``H`` (``operator.is_linear`` False; a missing attribute means linear):
``K_ens`` and the online ``K_q`` are EnKF-style (anomalies of ``H(f_i)`` and
``(H(f_i) - H(f'_i)) / sqrt(2)``), the innovation uses ``H(f_i)``. ``q_source``
bank / fixed linearise ``H`` at each group's forecast mean (``operator.jvp``; gains
not cached). The cap measures ``H(x'_i + delta_i) - H(x'_i)``. Localisation places
observations with the linear sensor map ``operator.inner`` (or explicit
``obs_coords``). Linear operators take the linear code paths unchanged.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import torch

from scisi.particle_filter.enkf import gaspari_cohn

logger = logging.getLogger(__name__)
Tensor = torch.Tensor

Q_SOURCES = ("online", "bank", "fixed")
DTYPE = torch.float64  # linear-algebra dtype
ROW_CHUNK = 4096  # state rows per chunk when applying d x m gains
TAPER_CACHE_LIMIT = 2**26  # cache the [d, m] taper up to this many entries
H_CHUNK = 256  # states per call when applying H to a bank or unit vectors
_CACHE_SIZE = 16  # entries per operator cache (H may change every cycle)


def _is_linear(operator: Any) -> bool:
    """``operator.is_linear``; operators predating the attribute are linear."""
    return bool(getattr(operator, "is_linear", True))


def _sensor_map(operator: Any) -> Optional[Any]:
    """The linear map that places the observations: ``operator`` itself if linear,
    else its linear sensor map ``operator.inner`` (``SensorComposedOperator``), else
    ``None``."""
    if _is_linear(operator):
        return operator
    inner = getattr(operator, "inner", None)
    return inner if inner is not None and _is_linear(inner) else None


class _OperatorCache:
    """Values keyed by an operator. Entries hold the operator and its ``obs_matrix``,
    so a collected operator's ``id`` never gives a stale hit and reassigning
    ``obs_matrix`` invalidates them."""

    def __init__(self) -> None:
        self._entries: dict = {}

    def get(self, operator: Any, key: Any, build: Callable[[], Any]) -> Any:
        tag = (operator, getattr(operator, "obs_matrix", None))
        hit = self._entries.get((id(operator), key))
        if hit is not None and hit[0][0] is tag[0] and hit[0][1] is tag[1]:
            return hit[1]
        if len(self._entries) >= _CACHE_SIZE:
            self._entries.pop(next(iter(self._entries)))
        value = build()
        self._entries[(id(operator), key)] = (tag, value)
        return value


def gaspari_cohn_kernel(
    radius: Union[float, Sequence[float]], periods: Optional[Sequence[Optional[float]]] = None
) -> Callable[[Tensor, Tensor], Tensor]:
    """Isotropic Gaspari-Cohn taper ``kernel(a [n, k], b [p, k]) -> [n, p]``.

    ``radius``: half-width, scalar or per axis. Periodic axes (``periods[j]``) use the
    chordal distance ``(p / pi) sin(pi |delta| / p)``, as ``enkf.periodic_taper``.
    PSD is guaranteed in at most 3 embedding dimensions; on a 2D torus (4D) it held
    numerically for the grids and radii tested.
    """

    def kernel(a: Tensor, b: Tensor) -> Tensor:
        radii = torch.as_tensor(radius, dtype=a.dtype).reshape(-1).expand(a.shape[1])
        sq = torch.zeros(a.shape[0], b.shape[0], dtype=a.dtype, device=a.device)
        for j in range(a.shape[1]):
            delta = (a[:, j : j + 1] - b[None, :, j]).abs()
            p = None if periods is None else periods[j]
            if p:
                delta = (p / math.pi) * torch.sin(math.pi * delta / p)
            sq += (delta / float(radii[j])) ** 2
        return gaspari_cohn(sq.sqrt(), 1.0)

    return kernel


class CoordinateLocalisation:
    """Localisation from ``[d, k]`` state coordinates and a (PSD) kernel.

    ``state_shape`` is the per-member state shape (``prod = d``). ``mask``: optional
    bool (state-shaped or ``[d]``); ``False`` entries get a zero taper row, so the
    update never moves them. Observation coordinates are ``L c / L 1`` (the sensor
    for selections, the block centre for block averages), so ``H`` stays free. ``L``
    is ``H`` when linear, else its linear sensor map ``H.inner`` (``g(L x)``, sensor
    speed): a nonlinear observation sits at its sensor even when, like the speed,
    it depends on the state non-locally. ``obs_coords`` (``[m, k]``) fixes the
    coordinates instead (required for a nonlinear ``H`` without ``inner``). Tapers
    are cached per sensor map on the compute device and dtype.
    """

    def __init__(self, state_coords: Tensor, kernel: Callable, state_shape: Sequence[int],
                 mask: Optional[Tensor] = None, obs_coords: Optional[Tensor] = None) -> None:
        self.state_coords = state_coords.detach().to(DTYPE).cpu()
        self.kernel = kernel
        self.state_shape = tuple(int(s) for s in state_shape)
        if math.prod(self.state_shape) != self.state_coords.shape[0]:
            raise ValueError("state_coords rows must match prod(state_shape)")
        self.keep = None if mask is None else (
            torch.as_tensor(mask).to(torch.bool).expand(self.state_shape).reshape(-1))
        self.fixed_obs_coords = None if obs_coords is None else obs_coords.detach().to(DTYPE).cpu()
        if self.fixed_obs_coords is not None and (
                self.fixed_obs_coords.dim() != 2 or self.fixed_obs_coords.shape[1] != self.state_coords.shape[1]):
            raise ValueError("obs_coords must be [m, k] with the state coordinates' k")
        self._cache = _OperatorCache()

    def _placing(self, operator: Any) -> Any:
        """The operator whose observations the tapers are built for (cache key)."""
        if self.fixed_obs_coords is not None:
            return operator
        sensor = _sensor_map(operator)
        if sensor is None:
            raise ValueError(
                f"cannot place the observations of the nonlinear {type(operator).__name__} "
                "(no linear sensor map .inner); pass CoordinateLocalisation(obs_coords=...)")
        if sensor is not operator and math.prod(self.state_shape) != getattr(
                sensor, "num_dofs", math.prod(self.state_shape)):
            raise ValueError(f"the sensor map of {type(operator).__name__} acts on "
                             f"{sensor.num_dofs} entries, the state has {math.prod(self.state_shape)}; "
                             "pass CoordinateLocalisation(obs_coords=...)")
        return sensor

    def obs_coords(self, operator: Any) -> Tensor:
        """``[m, k]`` observation coordinates of ``operator`` (class docstring)."""
        if self.fixed_obs_coords is not None:
            m = getattr(operator, "num_obs", self.fixed_obs_coords.shape[0])
            if self.fixed_obs_coords.shape[0] != m:
                raise ValueError(f"obs_coords has {self.fixed_obs_coords.shape[0]} rows, H has {m}")
            return self.fixed_obs_coords
        operator = self._placing(operator)

        def build() -> Tensor:
            k = self.state_coords.shape[1]
            weight = operator(torch.ones(1, *self.state_shape))
            if (weight.abs() < 1e-12).any():
                raise ValueError("an observation row of H sums to zero; cannot place it")
            fields = self.state_coords.T.reshape(k, *self.state_shape).float()
            return (operator(fields) / weight).T.to(DTYPE).cpu()  # [m, k]

        return self._cache.get(operator, "coords", build)

    def _block(self, operator: Any, rows: slice, ref: Tensor) -> Tensor:
        oc = self.obs_coords(operator).to(ref.device)  # operator: already _placing's
        taper = self.kernel(self.state_coords[rows].to(ref.device), oc)
        if self.keep is not None:
            taper = taper * self.keep[rows, None].to(taper.device)
        return taper.to(ref.dtype)

    def state_obs(self, operator: Any, rows: slice, ref: Tensor) -> Tensor:
        """``[rows, m]`` state-observation taper on ``ref``'s device and dtype."""
        operator = self._placing(operator)
        d, m = self.state_coords.shape[0], self.obs_coords(operator).shape[0]
        if d * m > TAPER_CACHE_LIMIT:
            return self._block(operator, rows, ref)
        build = lambda: torch.cat([  # noqa: E731
            self._block(operator, slice(s, min(s + ROW_CHUNK, d)), ref) for s in range(0, d, ROW_CHUNK)])
        return self._cache.get(operator, ("xy", ref.device, ref.dtype), build)[rows]

    def obs_obs(self, operator: Any, ref: Tensor) -> Tensor:
        """``[m, m]`` observation-observation taper."""
        operator = self._placing(operator)

        def build() -> Tensor:
            oc = self.obs_coords(operator).to(ref.device)
            return self.kernel(oc, oc).to(ref.dtype)

        return self._cache.get(operator, ("yy", ref.device, ref.dtype), build)


def grid_localisation(shape: Sequence[int], radius: Union[float, Sequence[float]],
                      periodic: Union[bool, Sequence[bool]] = False, channel_axis: bool = True,
                      mask: Optional[Tensor] = None) -> CoordinateLocalisation:
    """Gaspari-Cohn on a unit-spaced grid of the state shape, e.g. ``(C, H, W)``.

    With ``channel_axis`` (default) ``shape[0]`` holds channels at one site (distance
    0) and the other 1-3 axes are spatial (1D lattice: ``(1, 1, d)``); otherwise all
    axes are spatial (a 3D ``(D, H, W)`` grid). ``radius`` in cells, scalar or per
    spatial axis; ``periodic`` one flag or one per spatial axis; ``mask`` as in
    :class:`CoordinateLocalisation` (state or spatial shape).
    """
    shape = tuple(int(s) for s in shape)
    grids = torch.meshgrid(*[torch.arange(n, dtype=DTYPE) for n in shape], indexing="ij")
    axes, spatial = (grids[1:], shape[1:]) if channel_axis else (grids, shape)
    flags = [bool(periodic)] * len(spatial) if isinstance(periodic, bool) else list(periodic)
    if len(flags) != len(spatial):
        raise ValueError(f"periodic needs {len(spatial)} flags, got {len(flags)}")
    coords = torch.stack([g.reshape(-1) for g in axes], dim=1)
    periods = [float(n) if p else None for n, p in zip(spatial, flags)]
    return CoordinateLocalisation(coords, gaspari_cohn_kernel(radius, periods), shape, mask)


def _zero_diffusion(t: Tensor) -> Tensor:
    return torch.zeros_like(t)


def _is_ode_stepper(stepper: Callable) -> bool:
    fn = getattr(stepper, "func", stepper)  # functools.partial
    return getattr(fn, "__module__", None) == "scisi.sampling.ode_solvers"


def _model_device(posterior: Any, fallback: torch.device) -> torch.device:
    try:
        return next(posterior.parameters()).device
    except (StopIteration, AttributeError):
        return fallback


def _lifted_fm_drift(model: Any, g: Callable) -> Callable:
    """Prior reverse-SDE drift ``v + 1/2 g^2 s`` of ``FlowMatchingPosterior``."""

    def drift(x, t, field_history=None, field_cond=None, pars_cond=None):
        v = model.drift(x, t, field_history, field_cond, pars_cond)
        t_exp = t.reshape(t.shape[0], *([1] * (x.dim() - 1))) if t.dim() < x.dim() else t
        s = model.interpolation.score_from_velocity(x=x, v=v, t=t_exp, a0=torch.zeros_like(x))
        return v + 0.5 * g(t) ** 2 * s

    return drift


def draw_unguided_forecast(posterior: Any, field_history: Tensor, num_steps: int, stepper: Callable,
                           field_cond: Optional[Tensor] = None, pars_cond: Optional[Tensor] = None,
                           batch_size: int = 128) -> Tensor:
    """One unguided forecast per member from ``field_history[..., -1]`` (fresh noise).

    The posterior's own SDE/ODE without guidance. SI: ``model.sample`` with its
    diffusion (the free-running branch of ``sample_trajectory``). FM-ODE: the FM ODE
    from ``N(0, I)``. FM-SDE (DM-SDE): the lifted SDE ``v + 1/2 g^2 s`` (the
    free-running branch drops the lift). D-Flow: its midpoint ODE ``_flow``
    (``stepper`` ignored). Others: ``model.sample``. FM honours ``drift_time_shift``.
    Members run in chunks of ``batch_size``; the FM source is drawn for all at once.
    Raises ``ValueError`` for an ODE stepper with an SI or FM-SDE posterior (the
    forecast would be deterministic or noiseless). Returns ``field_history``'s device.
    """
    from scisi.models.flow_matching_model import FlowMatchingModel
    from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
    from scisi.posterior_models.dflow_posterior import DFlowPosterior

    model = posterior.model
    is_fm = isinstance(model, FlowMatchingModel)
    is_dflow = isinstance(posterior, DFlowPosterior)
    is_sde = is_fm and not is_dflow and not getattr(posterior, "is_ode", True)
    if _is_ode_stepper(stepper) and (is_sde or isinstance(model, FollmerStochasticInterpolant)):
        raise ValueError(
            f"forecast stepper {getattr(stepper, '__name__', stepper)!r} is an ODE stepper but "
            f"the {type(posterior).__name__} forecast is an SDE; use an SDE stepper."
        )
    shift = float(getattr(posterior, "drift_time_shift", 0.0)) / int(num_steps)
    if shift and not (is_fm and not is_dflow):
        raise NotImplementedError("drift_time_shift is only supported for FM forecasts")

    N = field_history.shape[0]
    dev = _model_device(posterior, field_history.device)
    fh = field_history.to(dev)
    z = torch.randn_like(fh[..., -1]) if is_fm else None
    cut = lambda t, sl: None if t is None else t[sl].to(dev)  # noqa: E731

    def run(sl: slice) -> Tensor:
        kw = dict(field_history=fh[sl], field_cond=cut(field_cond, sl), pars_cond=cut(pars_cond, sl))
        if is_dflow:
            return posterior._flow(z[sl], int(num_steps), **kw)
        kw.update(num_steps=int(num_steps), batch_size=sl.stop - sl.start, stepper=stepper)
        if is_fm:
            g = posterior.diffusion_term if is_sde else _zero_diffusion
            drift = _lifted_fm_drift(model, g) if is_sde else model.drift
            if shift:
                drift = (lambda f: lambda x, t, *a: f(x, t + shift, *a))(drift)
                g = (lambda f: lambda t: f(t + shift))(g)
            return model._sample(base=z[sl], diffusion_term=g, drift=drift, **kw)
        if not getattr(posterior, "default_diffusion_term", True):
            kw["diffusion_term"] = posterior.diffusion_term
        base = None if posterior.gaussian_base else fh[sl][..., -1].clone()
        return model.sample(base=base, return_field_history=False, **kw)

    with torch.no_grad():
        chunks = [slice(s, min(s + int(batch_size), N)) for s in range(0, N, int(batch_size))]
        return torch.cat([run(sl).to(field_history.device) for sl in chunks])


def build_residual_bank(posterior: Any, field_history: Tensor, num_steps: int, stepper: Callable,
                        pairs_per_state: int = 1, field_cond: Optional[Tensor] = None,
                        pars_cond: Optional[Tensor] = None, batch_size: int = 128,
                        seed: Optional[int] = None) -> Tensor:
    """``[S * pairs, ...]`` residuals ``(f1 - f2) / sqrt(2)`` of forecast pairs (CPU).

    ``Cov(r) = Q``, averaged over the states, for ``q_source="bank"``. Build it with
    the test-time forecast steps/stepper. ``seed``: forked, seeded RNG (the global
    stream is untouched). ``torch.save`` it; ``residual_bank`` accepts the path.
    """
    devices = [field_history.device, _model_device(posterior, field_history.device)]
    out = []
    with _forked_rng(devices, seed) if seed is not None else contextlib.nullcontext():
        for _ in range(int(pairs_per_state)):
            args = (posterior, field_history, num_steps, stepper, field_cond, pars_cond, batch_size)
            f1, f2 = draw_unguided_forecast(*args), draw_unguided_forecast(*args)
            out.append(((f1 - f2) / math.sqrt(2.0)).float().cpu())
    return torch.cat(out)


@contextlib.contextmanager
def _forked_rng(devices: Sequence[torch.device], seed: int):
    """Fork the CPU generator and those of ``devices``; seed only those."""
    cuda = sorted({torch.cuda.current_device() if d.index is None else d.index
                   for d in devices if d.type == "cuda"})
    mps = any(d.type == "mps" for d in devices)
    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.random.fork_rng(devices=cuda, device_type="cuda"))
        if mps:
            stack.enter_context(torch.random.fork_rng(devices=[0], device_type="mps"))
            torch.mps.manual_seed(seed)
        torch.default_generator.manual_seed(seed)
        for i in cuda:
            torch.cuda.default_generators[i].manual_seed(seed)
        yield


def _mix_seed(seed: int, counter: int) -> int:
    """A seed decorrelated from ``seed`` itself (which may be the driver's seed)."""
    digest = hashlib.blake2b(f"forecast-correction:{int(seed)}:{int(counter)}".encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "little") >> 1


@dataclass
class ForecastCorrectionConfig:
    """Settings of the forecast-residual correction.

    Attributes:
        enabled: Master switch. ``False`` (default): bit-for-bit no-op.
        forecast_num_steps: Steps of the forecasts ``f_i`` (and ``f'_i``). ``None``:
            the sampler's ``num_steps`` (D-Flow: ``ode_steps``). Few steps give
            approximate prior draws; check their spread against a full-step run.
        forecast_stepper: Forecast step function; ``None``: the sampler's.
        localisation: :class:`CoordinateLocalisation` (anything with ``state_obs`` /
            ``obs_obs``) or ``None``.
        gain_scale: Multiplier on ``K_ens - K_q`` (note, 4.4; tune on validation).
        q_source: ``"online"`` (default): ``Q_hat`` from a second forecast,
            ``(f_i - f'_i) / sqrt(2)`` (same solver as ``K_ens``, rank ``E``).
            ``"bank"``: ``residual_bank`` (state-averaged; build it with the
            test-time forecast steps). ``"fixed"``: known diagonal ``Q`` from
            ``q_variance`` (closed-form ``K_q``, not localised). Nonlinear ``H``:
            bank / fixed use ``J_H`` at each group's forecast mean, recomputed
            every call (no cache); online needs no linearisation.
        residual_bank: ``[N_b, ...]`` residuals, or a ``torch.save`` path (tensor or
            dict with ``"residuals"``). ``Q = B' B`` is the bank ``sqrt(N_b) B``.
        q_variance: ``"fixed"``: scalar or per-state-entry variance.
        group_size: Members per independent sequence (contiguous); statistics are
            per group. The last group takes the remainder (one leftover member
            joins the previous group). ``None``: one ensemble.
        update_cap: Optional cap on ``sqrt(u' S^-1 u / m)``, ``u = H delta_i``
            (nonlinear ``H``: ``H(x'_i + delta_i) - H(x'_i)``), ``S = H P_f H' + R``
            (localised); larger updates are scaled down.
        post_update: Optional ``post_update(x_corrected, x_guided) -> Tensor``.
        forecast_seed: If set, forecasts and ``e_i`` come from forked generators
            seeded with a hash of ``(forecast_seed, call counter)``: the sampler's
            stream is unchanged. The counter is never reset (each call and
            trajectory gets fresh noise); use a new ``ForecastCorrection`` to repeat
            a run. ``None``: the global stream, after the guided sample.
    """

    enabled: bool = False
    forecast_num_steps: Optional[int] = None
    forecast_stepper: Optional[Callable] = None
    localisation: Optional[CoordinateLocalisation] = None
    gain_scale: float = 1.0
    q_source: str = "online"
    residual_bank: Optional[Union[Tensor, str, Path]] = None
    q_variance: Optional[Union[float, Tensor]] = None
    group_size: Optional[int] = None
    update_cap: Optional[float] = None
    post_update: Optional[Callable[[Tensor, Tensor], Tensor]] = None
    forecast_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.q_source not in Q_SOURCES:
            raise ValueError(f"q_source must be one of {Q_SOURCES}")
        if self.forecast_num_steps is not None and int(self.forecast_num_steps) < 1:
            raise ValueError("forecast_num_steps must be >= 1")
        if self.group_size is not None and int(self.group_size) < 2:
            raise ValueError("group_size must be >= 2")
        if self.update_cap is not None and float(self.update_cap) <= 0:
            raise ValueError("update_cap must be positive")


class CorrectionDiagnostics:
    """One record per corrected step; tensors are ``[G]`` (one entry per group).

    ``phys_idx``; ``innovation_mean`` (mean of ``ybar - H fbar``); ``innovation_msq``
    (mean of ``(ybar - H fbar)_j^2 / S_jj``); ``innovation_lag1`` (mean product with
    the previous step's normalised innovation; ``None`` at the first step);
    ``update_rms``; ``spread_rms`` (RMS forecast std); ``q_rms`` (RMS of
    ``sqrt(diag Q_hat)``: the Q share of the forecast variance is
    ``(q_rms / spread_rms)^2``); ``cap_fraction``.
    """

    KEYS = ("innovation_mean", "innovation_msq", "update_rms", "spread_rms", "q_rms", "cap_fraction")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.records: list[dict] = []
        self._prev: Optional[Tensor] = None

    def add(self, phys_idx: Optional[int], normalised: Tensor, **stats: Tensor) -> None:
        prev, self._prev = self._prev, normalised
        lag = None if prev is None or prev.shape != normalised.shape else (normalised * prev).mean(1)
        self.records.append({"phys_idx": phys_idx, "innovation_lag1": lag, **stats})

    def lag1_autocorrelation(self) -> Optional[Tensor]:
        """``[G]`` lag-1 autocorrelation of the normalised innovations (near 0 when
        calibrated; the symptom of the per-member bias, note section 1)."""
        num = a = b = 0.0
        found = False
        for prev, rec in zip(self.records, self.records[1:]):
            if rec["innovation_lag1"] is not None:
                num, found = num + rec["innovation_lag1"], True
                a, b = a + rec["innovation_msq"], b + prev["innovation_msq"]
        return num / torch.sqrt(a * b).clamp_min(1e-30) if found else None

    def as_dict(self) -> dict:
        """``{key: [steps, G]}`` (lag1 NaN-padded) plus ``phys_idx``."""
        if not self.records:
            return {}
        out: dict = {"phys_idx": [r["phys_idx"] for r in self.records]}
        out.update({key: torch.stack([r[key] for r in self.records]) for key in self.KEYS})
        nan = torch.full_like(self.records[0]["innovation_mean"], float("nan"))
        out["innovation_lag1"] = torch.stack(
            [nan if r["innovation_lag1"] is None else r["innovation_lag1"] for r in self.records])
        return out


def _cholesky(S: Tensor) -> Tensor:
    L, info = torch.linalg.cholesky_ex(S)
    jitter = 1e-10 * S.diagonal().abs().mean().clamp_min(1e-30)
    for _ in range(8):
        if int(info) == 0:
            return L
        # Indefinite through a non-PSD taper or round-off: add a small jitter.
        logger.warning("forecast correction: jittering an indefinite S by %g", float(jitter))
        L, info = torch.linalg.cholesky_ex(S + jitter * torch.eye(S.shape[0], dtype=S.dtype, device=S.device))
        jitter = jitter * 100
    if int(info) == 0:
        return L
    raise RuntimeError("forecast correction: innovation covariance is not PD")


def _obs_cov(HX: Tensor, scale: float, loc_yy: Optional[Tensor]) -> Tensor:
    C = (HX.T @ HX) * scale
    return C if loc_yy is None else C * loc_yy


def _unlocalised_gain(X: Tensor, HX: Tensor, Z: Tensor) -> Tensor:
    """``(X' HX Z)'``, ``[E_out, d]``, through the smaller intermediate."""
    if X.shape[0] * Z.shape[1] > HX.shape[1] * X.shape[1]:
        return Z.T @ (HX.T @ X)  # m x d intermediate
    return (HX @ Z).T @ X  # E_in x E_out intermediate


def _group_slices(n: int, group_size: Optional[int]) -> list[slice]:
    if n < 2:
        raise ValueError("the correction needs at least 2 members")
    if group_size is None or int(group_size) >= n:
        return [slice(0, n)]
    starts = list(range(0, n, int(group_size)))
    if n - starts[-1] < 2:
        starts.pop()
    return [slice(a, b) for a, b in zip(starts, starts[1:] + [n])]


class ForecastCorrection:
    """The forecast-residual correction (module docstring)::

        corr = ForecastCorrection(ForecastCorrectionConfig(enabled=True, ...))
        traj = posterior.sample_trajectory(..., forecast_correction=corr)
        corr.diagnostics.lag1_autocorrelation()

    ``diagnostics`` is reset by ``sample_trajectory``, which also sets
    ``current_phys_idx`` (recorded per step).
    """

    def __init__(self, config: Optional[ForecastCorrectionConfig] = None, **kwargs: Any):
        if config is None:
            config = ForecastCorrectionConfig(**kwargs)
        elif kwargs:
            raise TypeError("pass either a config or keyword arguments, not both")
        self.config = config
        self.diagnostics = CorrectionDiagnostics()
        self.current_phys_idx: Optional[int] = None
        self._calls = 0  # never reset: seeds the forecasts and guards the hook
        self._cache = _OperatorCache()
        self._bank: Optional[tuple[Any, Tensor]] = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def reset(self) -> None:
        """Clear the diagnostics (not the gain caches or the call counter)."""
        self.diagnostics.reset()
        self.current_phys_idx = None

    def check_supported(self, posterior: Any) -> None:
        """Refuse SMC posteriors (SURGE, SI with ``resample=True``): their weights use
        the observation already, so a correction would count it twice. A posterior
        overriding ``_post_step`` / ``_post_sample`` without a ``resample`` flag is
        refused too. ``DiffusionPosterior`` ignores its ``resample`` (no SMC)."""
        from scisi.posterior_models.base_posterior import BasePosterior

        cls, resample = type(posterior), getattr(posterior, "resample", None)
        custom = isinstance(posterior, BasePosterior) and any(
            getattr(cls, h) is not getattr(BasePosterior, h) for h in ("_post_step", "_post_sample"))
        if not getattr(posterior, "supports_forecast_correction", True) or bool(resample) or (
                custom and resample is None):
            raise NotImplementedError(
                f"{cls.__name__} does not support the forecast correction: it is (or may be) "
                "an SMC sampler whose weights already use the observation.")
        cfg = self.config
        if cfg.q_source == "bank" and cfg.residual_bank is None:
            raise ValueError("q_source='bank' needs config.residual_bank")
        if cfg.q_source == "fixed" and cfg.q_variance is None:
            raise ValueError("q_source='fixed' needs config.q_variance")

    def apply(self, posterior: Any, base: Tensor, prev_history: Tensor, observations: Tensor,
              num_steps: int, stepper: Callable, field_cond: Optional[Tensor] = None,
              pars_cond: Optional[Tensor] = None, batch_size: int = 128) -> Tensor:
        """Correct the guided sample ``base`` (hook of ``BasePosterior.sample``).

        ``prev_history`` is the history before the new state is appended.
        """
        cfg = self.config
        lik = posterior.likelihood_model
        operator, var = getattr(lik, "obs_operator", None), getattr(lik, "original_variance", None)
        if operator is None or var is None:
            raise ValueError("forecast correction needs likelihood_model.obs_operator / original_variance")
        steps = int(cfg.forecast_num_steps or getattr(posterior, "ode_steps", num_steps))
        fstep = cfg.forecast_stepper or stepper
        args = (posterior, prev_history, steps, fstep, field_cond, pars_cond, batch_size)
        devices = [base.device, prev_history.device, _model_device(posterior, base.device)]
        seed = cfg.forecast_seed
        with torch.no_grad(), (
            contextlib.nullcontext() if seed is None else _forked_rng(devices, _mix_seed(seed, self._calls))
        ):
            f = draw_unguided_forecast(*args)
            f2 = draw_unguided_forecast(*args) if cfg.q_source == "online" else None
            out = self.correct(base, f, observations, operator, var, f2=f2, phys_idx=self.current_phys_idx)
        self._calls += 1
        return out

    def correct(self, x: Tensor, f: Tensor, y: Tensor, operator: Any, obs_noise_var: Union[float, Tensor],
                f2: Optional[Tensor] = None, perturbations: Optional[Tensor] = None,
                phys_idx: Optional[int] = None) -> Tensor:
        """``x'' = x' + gain_scale (K_ens - K_q)(y + e - H f)``; returns ``x``'s shape,
        dtype and device (``post_update`` applied).

        ``x``, ``f``: ``[N, ...]`` guided samples and independent forecasts. ``y``:
        ``[1, m]``, one row per group or one per member. ``obs_noise_var``: ``R``,
        scalar or ``[m]``. ``f2``: second forecast (``q_source="online"``).
        ``perturbations``: optional ``[N, m]`` ``e_i`` instead of ``N(0, R)`` draws.
        A nonlinear ``operator`` (``is_linear`` False) needs ``jvp(x, v)`` for
        ``q_source`` bank / fixed (module docstring).
        """
        cfg = self.config
        online = cfg.q_source == "online"
        if online and f2 is None:
            raise ValueError("q_source='online' needs f2 (a second forecast)")
        linear = _is_linear(operator)
        if not (online or linear or callable(getattr(operator, "jvp", None))):
            raise NotImplementedError(
                f"q_source={cfg.q_source!r} with the nonlinear {type(operator).__name__} needs "
                "operator.jvp(x, v) to linearise H; use q_source='online'")
        N, shape, sdev, sdtype = x.shape[0], x.shape, x.device, x.dtype
        dev = torch.device("cpu") if sdev.type == "mps" else sdev  # MPS: no float64 / cholesky_solve
        d = x[0].numel()

        def flat(t: Tensor) -> Tensor:
            return t.reshape(t.shape[0], -1).to(dev).to(DTYPE)  # device first: MPS has no float64

        def to_state(t: Tensor) -> Tensor:
            return t.reshape(-1, *shape[1:]).to(sdtype).to(sdev)

        def apply_H(t: Tensor) -> Tensor:  # on the state's device/dtype (the operator's cache stays put)
            return flat(operator(to_state(t)))

        def linearised_H(x_ref: Tensor) -> Callable[[Tensor], Tensor]:
            """``v -> J_H(x_ref) v`` for a nonlinear ``H`` (bank / fixed ``K_q``)."""
            ref = to_state(x_ref)

            def apply(t: Tensor) -> Tensor:
                v = to_state(t)
                return flat(operator.jvp(ref.expand(v.shape[0], *shape[1:]).contiguous(), v))

            return apply

        F, HF = flat(f), apply_H(f)
        m = HF.shape[1]
        R = torch.as_tensor(obs_noise_var, dtype=DTYPE).reshape(-1).to(dev)
        if R.numel() not in (1, m):
            raise ValueError(f"obs_noise_var has {R.numel()} entries, expected 1 or {m}")
        R = R.expand(m)
        slices = _group_slices(N, cfg.group_size)
        Y = flat(y.reshape(-1, m))
        if Y.shape[0] == 1:
            Y = Y.expand(N, m)
        elif Y.shape[0] == len(slices) and Y.shape[0] != N:
            Y = Y.repeat_interleave(torch.tensor([s.stop - s.start for s in slices], device=dev), 0)
        elif Y.shape[0] != N:
            raise ValueError(f"observations have {Y.shape[0]} rows; expected 1, {len(slices)} or {N}")
        e = flat(perturbations) if perturbations is not None else (
            torch.randn(N, m, dtype=DTYPE, device=dev) * R.sqrt())
        V = Y + e - HF  # perturbed forecast innovations

        if online:  # H applied to f2 itself (exact for nonlinear H too)
            Rr, HRr = (F - flat(f2)) / math.sqrt(2.0), (HF - apply_H(f2)) / math.sqrt(2.0)
        loc = cfg.localisation
        loc_yy = None if loc is None else loc.obs_obs(operator, F)
        Kq, q_rms = (None, None) if online or not linear else self._fixed_gain(
            operator, R, loc_yy, shape, apply_H, F)
        Kq_groups = {}  # nonlinear bank / fixed: K_q linearised at each group's forecast mean
        if not (online or linear):
            for g, sl in enumerate(slices):
                Kq_groups[g] = self._fixed_gain(operator, R, loc_yy, shape,
                                                linearised_H(F[sl].mean(0, keepdim=True)), F, cache=False)
            q_rms = Kq_groups[0][1]

        groups = []  # (slice, L_ens, S_ens, [(X, HX, scale, Z)])
        for sl in slices:
            E = sl.stop - sl.start
            A, AH = F[sl] - F[sl].mean(0), HF[sl] - HF[sl].mean(0)
            S = _obs_cov(AH, 1.0 / (E - 1), loc_yy) + torch.diag(R)
            L = _cholesky(S)
            terms = [(A, AH, 1.0 / (E - 1), torch.cholesky_solve(V[sl].T, L))]
            if online:
                Sq = _obs_cov(HRr[sl], 1.0 / E, loc_yy) + torch.diag(R)
                terms.append((Rr[sl], HRr[sl], -1.0 / E, torch.cholesky_solve(V[sl].T, _cholesky(Sq))))
            groups.append((sl, L, S, terms))

        delta = torch.zeros(N, d, dtype=DTYPE, device=dev)
        if loc is None:
            for sl, _, _, terms in groups:
                for X, HX, scale, Z in terms:
                    delta[sl] += _unlocalised_gain(X, HX, Z) * scale
        else:
            for start in range(0, d, ROW_CHUNK):
                rows = slice(start, min(start + ROW_CHUNK, d))
                taper = loc.state_obs(operator, rows, F)  # one block for all groups and gains
                for sl, _, _, terms in groups:
                    for X, HX, scale, Z in terms:
                        delta[sl, rows] += (((X[:, rows].T @ HX) * (scale * taper)) @ Z).T
        if Kq is not None:
            delta -= V @ Kq.T
        for g, (Kg, _) in Kq_groups.items():
            delta[slices[g]] -= V[slices[g]] @ Kg.T
        delta *= float(cfg.gain_scale)

        norm, stats = [], {k: [] for k in CorrectionDiagnostics.KEYS}
        for sl, L, S, terms in groups:
            E = sl.stop - sl.start
            capped = torch.zeros(E, dtype=DTYPE, device=dev)
            if cfg.update_cap is not None:
                cap = float(cfg.update_cap)
                if linear:
                    u = apply_H(delta[sl])
                else:  # change of the predicted observation (no linearisation)
                    xs = flat(x[sl])
                    u = apply_H(xs + delta[sl]) - apply_H(xs)
                W = torch.linalg.solve_triangular(L, u.T, upper=False)
                size = W.pow(2).mean(0).sqrt()
                capped = (size > cap).to(DTYPE)
                delta[sl] *= torch.where(capped > 0, cap / size.clamp_min(1e-30), 1.0)[:, None]
            innov = Y[sl].mean(0) - HF[sl].mean(0)
            norm.append(innov / S.diagonal().sqrt())
            values = (innov.mean(), norm[-1].pow(2).mean(), delta[sl].pow(2).mean().sqrt(),
                      terms[0][0].pow(2).sum(0).div(E - 1).mean().sqrt(),
                      terms[1][0].pow(2).sum(0).div(E).mean().sqrt() if online else q_rms, capped.mean())
            for key, value in zip(CorrectionDiagnostics.KEYS, values):
                stats[key].append(value)

        x_new = (flat(x) + delta).to(sdtype).to(sdev).reshape(shape)
        if cfg.post_update is not None:
            x_new = cfg.post_update(x_new, x)
        stats = {k: torch.stack(v).cpu() for k, v in stats.items()}
        self.diagnostics.add(phys_idx, torch.stack(norm).cpu(), **stats)
        return x_new

    def _bank_tensor(self) -> Tensor:
        src = self.config.residual_bank
        if self._bank is None or self._bank[0] is not src:
            bank = torch.load(src, map_location="cpu") if isinstance(src, (str, Path)) else src
            bank = bank["residuals"] if isinstance(bank, dict) else bank
            if not torch.is_tensor(bank):
                raise ValueError("residual_bank must be a tensor or a path")
            self._bank = (src, bank)
        return self._bank[1]

    def _fixed_gain(self, operator: Any, R: Tensor, loc_yy: Optional[Tensor], shape: torch.Size,
                    apply_H: Callable[[Tensor], Tensor], ref: Tensor,
                    cache: bool = True) -> tuple[Tensor, Tensor]:
        """``(K_q [d, m], q_rms)`` for ``q_source`` bank or fixed, cached per operator.

        ``apply_H``: the linear map applied to residuals / unit vectors (``H``, or
        ``J_H(x_ref)`` with ``cache=False`` for a nonlinear ``H``); ``operator``
        places the observations for localisation.
        """
        cfg, loc = self.config, self.config.localisation
        d, dev = ref.shape[1], ref.device
        if cfg.q_source == "bank":
            bank = self._bank_tensor()  # held by self, so its id is stable
            if tuple(bank.shape[1:]) != tuple(shape[1:]):
                raise ValueError(f"residual bank fields {tuple(bank.shape[1:])} != state {tuple(shape[1:])}")
            ident: Any = id(bank)
        else:
            q = torch.as_tensor(cfg.q_variance, dtype=DTYPE).reshape(-1)
            if q.numel() not in (1, d):
                raise ValueError(f"q_variance has {q.numel()} entries, expected 1 or {d}")
            ident = q.numpy().tobytes()
        key = (cfg.q_source, ident, R.cpu().numpy().tobytes(), loc, str(dev))

        def build() -> tuple[Tensor, Tensor]:
            if cfg.q_source == "bank":
                Rb = bank.reshape(bank.shape[0], -1).to(dev).to(DTYPE)
                HRb = torch.cat([apply_H(Rb[i : i + H_CHUNK]) for i in range(0, Rb.shape[0], H_CHUNK)])
                S = _obs_cov(HRb, 1.0 / Rb.shape[0], loc_yy) + torch.diag(R)
                QHt = lambda rows: (Rb[:, rows].T @ HRb) / Rb.shape[0]  # noqa: E731
                q_rms = Rb.pow(2).mean(0).mean().sqrt()
            else:  # diagonal Q: K_q = Q H' (H Q H' + R)^-1, exact, not localised
                qd = q.to(dev).expand(d)
                Ht = []  # rows H e_j, i.e. H' [d, m]
                for i in range(0, d, H_CHUNK):
                    n = min(H_CHUNK, d - i)
                    unit = torch.zeros(n, d, dtype=DTYPE, device=dev)
                    unit[torch.arange(n), torch.arange(i, i + n)] = 1.0
                    Ht.append(apply_H(unit))
                Ht = torch.cat(Ht)
                QH = qd[:, None] * Ht
                S = Ht.T @ QH + torch.diag(R)
                QHt = lambda rows: QH[rows]  # noqa: E731
                q_rms = qd.mean().sqrt()
            L = _cholesky(S)
            K = torch.empty(d, S.shape[0], dtype=DTYPE, device=dev)
            for start in range(0, d, ROW_CHUNK):
                rows = slice(start, min(start + ROW_CHUNK, d))
                block = QHt(rows)
                if loc is not None and cfg.q_source == "bank":
                    block = block * loc.state_obs(operator, rows, ref)
                K[rows] = torch.cholesky_solve(block.T, L).T
            return K, q_rms

        return self._cache.get(operator, key, build) if cache else build()


__all__ = [
    "CoordinateLocalisation",
    "CorrectionDiagnostics",
    "ForecastCorrection",
    "ForecastCorrectionConfig",
    "build_residual_bank",
    "draw_unguided_forecast",
    "gaspari_cohn_kernel",
    "grid_localisation",
]
