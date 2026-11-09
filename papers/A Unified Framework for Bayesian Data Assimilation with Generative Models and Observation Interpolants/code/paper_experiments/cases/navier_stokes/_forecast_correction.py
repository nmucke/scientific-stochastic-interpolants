"""Forecast-residual correction wiring for the Navier--Stokes and urban cases.

The correction itself lives in ``scisi.posterior_models.forecast_correction``
(background: ``FORECAST_CORRECTION.md``). This module only turns the case config
block ``case.forecast_correction`` into a ``ForecastCorrection`` for one cell, adds
the case geometry (localisation), optional physical constraints (``post_update``)
and cheap optional diagnostics, and builds the tidy ``variant`` tag.

It is shared by both cases (the urban pipeline already imports the NS seams);
the urban-only projection is in ``cases/urban/_urban_constraints.py``.

Modes (``forecast_correction.mode``):

* ``correct``: apply the correction at every ASSIMILATED step (free-running steps
  of ``sample_trajectory`` are never corrected).
* ``measure``: "measure before fixing" (note, section 5 stage 3). Draws the
  unguided forecasts and records the innovation statistics (lag-1
  autocorrelation) and the share of the forecast variance that is Q, but leaves
  the trajectory unchanged: ``gain_scale = 0``, no localisation, and the
  forecasts come from a forked RNG (``forecast_seed``), so the guided samples
  are bit-for-bit those of the plain run. Only NFE / seconds differ (the
  forecasts are counted), which is why these rows get their own variant tag.

With ``enabled: false`` (the default) nothing here is constructed and the call to
``sample_trajectory`` is identical to before.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from scisi.posterior_models.forecast_correction import (
    ForecastCorrection,
    ForecastCorrectionConfig,
    Localisation,
    ResidualBank,
    build_residual_bank,
    draw_unguided_forecast,
    grid_localisation,
    periodic_2d_localisation,
)

logger = logging.getLogger(__name__)

#: The three "Ours" samplers: the only methods corrected by default.
DEFAULT_METHODS: tuple[str, ...] = ("Ours (SI-SDE)", "Ours (DM-SDE)", "Ours (FM-ODE)")

#: Defaults of the ``case.forecast_correction`` block. A key missing from the
#: YAML falls back to these, so an old config keeps working.
DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "mode": "correct",  # correct | measure
    "methods": list(DEFAULT_METHODS),
    # f_i forecast: pseudo-time steps (null = the sampler's M), stepper
    # (null = the sampler's; sde | ode), mode (prior | free_run | ode).
    "forecast_num_steps": None,
    "forecast_stepper": None,
    "forecast_mode": "prior",
    # Gaspari-Cohn half-width in grid cells; null = global (no localisation).
    # A mapping {scenario_key: r, default: r} gives per-scenario radii.
    "localisation_radius": 8.0,
    "localisation_combine": "euclidean",
    "localisation_cache_limit": 2**24,
    "gain_scale": 1.0,
    "update_cap": 3.0,
    "q_source": "online",  # online | bank
    "residual_bank": None,  # path (q_source=bank)
    "bank_indices": None,  # test_dataset rows to build a missing bank from
    "bank_max_states": 256,
    "bank_pairs_per_state": 1,
    "group_size": None,
    "forecast_seed": None,
    "compute_dtype": "float64",
    "row_chunk": 4096,
    "diagnostics": False,
    "diagnostics_root": None,
    "variant_suffix": None,
}

_STEPPER_NAMES = ("sde", "ode")


# --------------------------------------------------------------------------- #
# Settings and the variant tag
# --------------------------------------------------------------------------- #


def fc_settings(case_cfg: Any) -> dict:
    """Merge ``case_cfg.forecast_correction`` over :data:`DEFAULTS`."""
    block = None
    if case_cfg is not None:
        try:
            block = case_cfg.get("forecast_correction", None)
        except AttributeError:
            block = getattr(case_cfg, "forecast_correction", None)
    out = dict(DEFAULTS)
    if block is not None:
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(block):
                block = OmegaConf.to_container(block, resolve=True)
        except ImportError:  # pragma: no cover
            pass
        out.update(dict(block))
    if out["mode"] not in ("correct", "measure"):
        raise ValueError(
            f"forecast_correction.mode must be 'correct' or 'measure', got {out['mode']!r}"
        )
    if out["forecast_stepper"] is not None and out["forecast_stepper"] not in _STEPPER_NAMES:
        raise ValueError(
            f"forecast_correction.forecast_stepper must be null, 'sde' or 'ode', "
            f"got {out['forecast_stepper']!r}"
        )
    return out


def applies_to(settings: Mapping, method_name: str) -> bool:
    """Whether the correction is switched on for this method."""
    if not bool(settings.get("enabled", False)):
        return False
    methods = settings.get("methods", None) or DEFAULT_METHODS
    return str(method_name) in {str(m) for m in methods}


def resolve_radius(radius: Any, scenario_key: Optional[str]) -> Optional[float]:
    """Scalar, ``None`` or ``{scenario_key: r, "default": r}`` -> float or None."""
    if radius is None:
        return None
    if isinstance(radius, Mapping):
        value = radius.get(scenario_key, radius.get("default", None))
        return None if value is None else float(value)
    return float(radius)


def _fmt(x: Any) -> str:
    s = f"{float(x):g}" if isinstance(x, (int, float)) else str(x)
    return s.replace(".", "p").replace("-", "m")


def variant_suffix(settings: Mapping, scenario_key: Optional[str]) -> str:
    """Tag that keeps corrected (and measure-only) rows apart from plain ones.

    Built from every setting that changes the numbers, e.g.
    ``fc_r8_g1_nM_qon_cap3`` (radius 8, gain 1, forecast steps = M, online Q,
    cap 3) or ``fcdiag_nM`` for measure-only. ``variant_suffix`` overrides it.
    """
    if settings.get("variant_suffix"):
        return str(settings["variant_suffix"])
    steps = settings.get("forecast_num_steps")
    n = "M" if steps is None else str(int(steps))
    extra = []
    if settings.get("forecast_stepper"):
        extra.append(str(settings["forecast_stepper"]))
    if settings.get("forecast_mode", "prior") != "prior":
        extra.append(str(settings["forecast_mode"]))
    if settings.get("mode") == "measure":
        return "_".join(["fcdiag", f"n{n}", *extra])
    r = resolve_radius(settings.get("localisation_radius"), scenario_key)
    parts = [
        "fc",
        "rglob" if r is None else f"r{_fmt(r)}",
        f"g{_fmt(settings.get('gain_scale', 1.0))}",
        f"n{n}",
        *extra,
        "qon" if settings.get("q_source", "online") == "online" else "qbank",
    ]
    cap = settings.get("update_cap")
    parts.append("capnone" if cap is None else f"cap{_fmt(cap)}")
    return "_".join(parts)


def variant_with_correction(
    base: Optional[str], settings: Mapping, method_name: str, scenario_key: Optional[str]
) -> Optional[str]:
    """``base`` unchanged when the correction is off for this method, else
    ``base + "_" + suffix`` (or the suffix alone when ``base`` is empty)."""
    if not applies_to(settings, method_name):
        return base
    suffix = variant_suffix(settings, scenario_key)
    return f"{base}_{suffix}" if base else suffix


# --------------------------------------------------------------------------- #
# Localisation geometry
# --------------------------------------------------------------------------- #


class MaskedLocalisation(Localisation):
    """Zero the state-observation taper on masked-out (solid) state entries.

    The gain then has zero rows in obstacles, so the update never moves state
    inside a building. ``obs_obs`` is unchanged (sensors sit in fluid cells). The
    state-state taper is ``M T M`` (M the 0/1 mask), still PSD.
    """

    def __init__(self, base: Localisation, keep_flat: torch.Tensor) -> None:
        self.base = base
        self.keep = keep_flat.reshape(-1).to(torch.bool).cpu()

    def state_obs(self, operator: Any, rows: slice, ref: torch.Tensor) -> torch.Tensor:
        block = self.base.state_obs(operator, rows, ref)
        keep = self.keep[rows].to(device=block.device, dtype=block.dtype)
        return block * keep[:, None]

    def obs_obs(self, operator: Any, ref: torch.Tensor) -> torch.Tensor:
        return self.base.obs_obs(operator, ref)

    def obs_coords(self, operator: Any) -> torch.Tensor:
        return self.base.obs_coords(operator)  # type: ignore[attr-defined]


def ns_localisation(
    settings: Mapping, data_size: Sequence[int], scenario_key: Optional[str]
) -> Optional[Localisation]:
    """Doubly periodic Gaspari-Cohn on the NS grid (or ``None`` = global).

    Observation locations come from the operator itself (``H c / H 1`` in the
    library): the sensor cell for sparse masks and the block centre for the
    block-average super-resolution operator, so the same object serves every
    scenario and ``H`` stays free.
    """
    r = resolve_radius(settings.get("localisation_radius"), scenario_key)
    if r is None:
        return None
    return periodic_2d_localisation(
        tuple(int(s) for s in data_size),
        r,
        combine=str(settings.get("localisation_combine", "euclidean")),
        cache_limit=int(settings.get("localisation_cache_limit", 2**24)),
    )


def urban_localisation(
    settings: Mapping,
    data_size: Sequence[int],
    scenario_key: Optional[str],
    fluid_mask: Optional[torch.Tensor],
) -> Optional[Localisation]:
    """Gaspari-Cohn on the (by default non-periodic) urban grid, 4 channels per site.

    ``periodic`` ([H, W] flags), ``channel_taper_offdiag`` (cross-variable taper,
    1 = keep all cross-covariances) are read from ``settings``. Solid cells get a
    zero taper row (:class:`MaskedLocalisation`). Distances are straight-line;
    they do not go around buildings.
    """
    r = resolve_radius(settings.get("localisation_radius"), scenario_key)
    if r is None:
        return None
    C, H, W = (int(s) for s in data_size)
    periodic = tuple(bool(p) for p in (settings.get("periodic") or (False, False)))
    off = settings.get("channel_taper_offdiag", 1.0)
    channel_taper = None
    if off is not None and float(off) != 1.0:
        channel_taper = torch.full((C, C), float(off), dtype=torch.float64)
        channel_taper.fill_diagonal_(1.0)
        # (1-a) I + a 11' is PSD for 0 <= a <= 1.
        if not 0.0 <= float(off) <= 1.0:
            raise ValueError("channel_taper_offdiag must be in [0, 1]")
    loc = grid_localisation(
        (C, H, W),
        r,
        periodic=periodic,  # type: ignore[arg-type]
        channel_taper=channel_taper,
        combine=str(settings.get("localisation_combine", "euclidean")),
        cache_limit=int(settings.get("localisation_cache_limit", 2**26)),
    )
    if fluid_mask is not None and bool(settings.get("mask_obstacles", True)):
        keep = fluid_mask.to(torch.bool)
        if keep.dim() == 2:
            keep = keep.unsqueeze(0).expand(C, H, W)
        return MaskedLocalisation(loc, keep.reshape(-1))
    return loc


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

#: Integer-wavenumber band edges for the NS enstrophy spectra (128^2 grid).
NS_BAND_EDGES: tuple[float, ...] = (0.5, 2, 4, 8, 16, 32, 64, 91)


def enstrophy_band_spectra(
    omega: torch.Tensor, edges: Sequence[float] = NS_BAND_EDGES
) -> torch.Tensor:
    """Per-member enstrophy ``1/2 sum |omega_hat(k)|^2`` in radial bands.

    ``omega``: ``[N, C, H, W]`` (channel 0 is used) in any units; the ratio of two
    spectra is unit-free and the mean (k=0) is excluded by the first edge.
    Returns ``[N, n_bands]``.
    """
    w = omega[:, 0].to(torch.float64)
    N, H, W = w.shape
    wh = torch.fft.fft2(w) / (H * W)
    power = 0.5 * wh.abs() ** 2
    ky = torch.fft.fftfreq(H, d=1.0 / H).to(w.device)
    kx = torch.fft.fftfreq(W, d=1.0 / W).to(w.device)
    kk = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (kk >= lo) & (kk < hi)
        out.append(power[:, sel].sum(dim=1))
    return torch.stack(out, dim=1)


def ns_step_diagnostics(
    x_guided: torch.Tensor, x_new: torch.Tensor, x_out: torch.Tensor
) -> dict:
    """Enstrophy per band before (guided x') and after (x'') the correction."""
    before = enstrophy_band_spectra(x_guided)  # [N, B]
    after = enstrophy_band_spectra(x_out)
    member_ratio = after.sum(1) / before.sum(1).clamp_min(1e-300)
    inc = (x_out - x_guided).to(torch.float64)
    raw = (x_new - x_guided).to(torch.float64)
    return {
        "band_edges": list(NS_BAND_EDGES),
        "enstrophy_before": before.mean(0).tolist(),
        "enstrophy_after": after.mean(0).tolist(),
        "enstrophy_ratio_bands": (after.mean(0) / before.mean(0).clamp_min(1e-300)).tolist(),
        "member_enstrophy_ratio_min": float(member_ratio.min()),
        "member_enstrophy_ratio_median": float(member_ratio.median()),
        "member_enstrophy_ratio_max": float(member_ratio.max()),
        # Domain mean of the increment (vorticity on the torus has a fixed mean).
        # A localised gain is NOT a combination of whole anomaly fields, so the raw
        # update can shift the mean; `zero_mean_increment` removes that shift.
        "increment_domain_mean_abs_raw": float(raw.mean(dim=(2, 3)).abs().max()),
        "increment_domain_mean_abs": float(inc.mean(dim=(2, 3)).abs().max()),
        "increment_rms_raw": float(raw.pow(2).mean().sqrt()),
        "increment_rms": float(inc.pow(2).mean().sqrt()),
    }


def zero_mean_increment(x_new: torch.Tensor, x_guided: torch.Tensor) -> torch.Tensor:
    """NS ``post_update``: remove the domain mean of the increment, per channel.

    Orthogonal projection of ``x'' - x'`` onto zero-mean fields, so the guided
    sample's domain-mean vorticity is kept exactly (periodic domain).
    """
    delta = (x_new - x_guided).to(torch.float64)
    delta = delta - delta.mean(dim=(2, 3), keepdim=True)
    return (x_guided.to(torch.float64) + delta).to(x_new.dtype)


class CorrectionMonitor:
    """Optional constraint + diagnostics around one cell's correction.

    * ``forecast_fn`` (only installed when diagnostics are on): draws the
      forecasts exactly as the library would (same mode / steps / stepper) and
      keeps them, so the share of the forecast variance that is Q can be read off
      the pair ``(f, f')`` without extra network passes.
    * ``post_update(x_new, x_guided)``: applies the case constraint (urban
      projection; identity for NS) and records the per-step diagnostics.
    """

    def __init__(
        self,
        operator: Any,
        forecast_mode: str = "prior",
        constraint: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        step_diagnostics: Optional[Callable[..., dict]] = None,
        diagnostics: bool = False,
        bank: Optional[torch.Tensor] = None,
    ) -> None:
        self.operator = operator
        self.forecast_mode = forecast_mode
        self.constraint = constraint
        self.step_diagnostics = step_diagnostics
        self.diagnostics = bool(diagnostics)
        self.bank = bank
        self.records: list[dict] = []
        self._stash: list[torch.Tensor] = []
        self._bank_trq: Optional[tuple[float, float]] = None

    # -- forecasts ------------------------------------------------------- #
    def forecast_fn(self, posterior, field_history, field_cond, pars_cond, num_steps, stepper):  # type: ignore[no-untyped-def]
        f = draw_unguided_forecast(
            posterior,
            field_history,
            num_steps=num_steps,
            stepper=stepper,
            mode=self.forecast_mode,
            field_cond=field_cond,
            pars_cond=pars_cond,
            batch_size=int(field_history.shape[0]),
        )
        self._stash.append(f.detach())
        return f

    # -- Q share ----------------------------------------------------------- #
    def _q_share(self) -> dict:
        if not self._stash:
            return {}
        f = self._stash[0].to(torch.float64)
        N = f.shape[0]
        Hf = self.operator(self._stash[0]).to(torch.float64)
        trP = float(f.reshape(N, -1).var(0, unbiased=True).sum())
        trHP = float(Hf.var(0, unbiased=True).sum())
        if len(self._stash) >= 2:
            r = (f - self._stash[1].to(torch.float64)) / math.sqrt(2.0)
            Hr = self.operator(r.to(self._stash[0].dtype)).to(torch.float64)
            trQ = float(r.reshape(N, -1).pow(2).mean(0).sum())
            trHQ = float(Hr.pow(2).mean(0).sum())
            src = "online"
        elif self.bank is not None:
            if self._bank_trq is None:
                b = self.bank.to(torch.float64)
                Hb = self.operator(self.bank.to(f.device, self._stash[0].dtype)).to(torch.float64)
                self._bank_trq = (
                    float(b.reshape(b.shape[0], -1).pow(2).mean(0).sum()),
                    float(Hb.pow(2).mean(0).sum()),
                )
            trQ, trHQ = self._bank_trq
            src = "bank"
        else:
            return {"forecast_var_trace": trP, "forecast_var_trace_obs": trHP}
        return {
            "q_source": src,
            "forecast_var_trace": trP,
            "q_trace": trQ,
            "q_share_state": trQ / max(trP, 1e-300),
            "forecast_var_trace_obs": trHP,
            "q_trace_obs": trHQ,
            "q_share_obs": trHQ / max(trHP, 1e-300),
        }

    # -- hook ------------------------------------------------------------ #
    def post_update(self, x_new: torch.Tensor, x_guided: torch.Tensor) -> torch.Tensor:
        x_out = x_new if self.constraint is None else self.constraint(x_new, x_guided)
        if self.diagnostics:
            rec: dict = {}
            rec.update(self._q_share())
            if self.step_diagnostics is not None:
                rec.update(self.step_diagnostics(x_guided, x_new, x_out))
            self.records.append(rec)
        self._stash = []
        return x_out

    def reset(self) -> None:
        self.records = []
        self._stash = []


def _correction_summary(correction: ForecastCorrection) -> dict:
    """JSON-ready summary of the library's per-step diagnostics."""
    d = correction.diagnostics
    out: dict = {}
    lag = d.lag1_autocorrelation()
    out["lag1_autocorrelation"] = None if lag is None else [float(v) for v in lag]
    stacked = d.as_dict()
    for key, value in stacked.items():
        if torch.is_tensor(value):
            out[key] = value.tolist()
        else:
            out[key] = list(value)
    return out


def summary(correction: ForecastCorrection, monitor: Optional[CorrectionMonitor]) -> dict:
    """Everything worth keeping from one cell, JSON-ready."""
    out = _correction_summary(correction)
    if monitor is not None and monitor.records:
        out["steps"] = monitor.records
        shares = [r["q_share_obs"] for r in monitor.records if "q_share_obs" in r]
        if shares:
            out["q_share_obs_mean"] = float(sum(shares) / len(shares))
        shares = [r["q_share_state"] for r in monitor.records if "q_share_state" in r]
        if shares:
            out["q_share_state_mean"] = float(sum(shares) / len(shares))
    msq = out.get("innovation_msq")
    if msq:
        flat = [v for row in msq for v in (row if isinstance(row, list) else [row])]
        out["innovation_msq_mean"] = float(sum(flat) / len(flat))
    caps = out.get("cap_fraction")
    if caps:
        flat = [v for row in caps for v in (row if isinstance(row, list) else [row])]
        out["cap_fraction_mean"] = float(sum(flat) / len(flat))
    return out


def write_diagnostics(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    return path


# --------------------------------------------------------------------------- #
# Residual bank (q_source = bank)
# --------------------------------------------------------------------------- #


def build_bank_from_dataset(
    prior: Any,
    posterior: Any,
    indices: Sequence[int],
    num_steps: int,
    stepper: Callable,
    forecast_mode: str,
    max_states: int = 256,
    pairs_per_state: int = 1,
    path: Optional[Path] = None,
    seed: Optional[int] = 0,
) -> ResidualBank:
    """Build a Q_hat bank from sliding windows of ``prior.test_dataset[indices]``.

    Use rows that the evaluation does NOT score (NS: test rows 0-9; urban: 0, 7).
    States are normalised with the prior's preprocesser, exactly as the
    assimilation inputs are. ``num_steps`` / ``stepper`` / ``forecast_mode`` must
    be those used for the test-time forecasts.
    """
    pre = prior.preprocesser
    L = int(prior.len_field_history)
    device = next(posterior.parameters()).device
    histories, conds = [], []
    for idx in indices:
        sample = prior.test_dataset[int(idx)]
        traj = pre.transform(base=sample["x"].unsqueeze(0), is_batch=True, is_trajectory=True)[
            "base"
        ]  # [1, C, H, W, T]
        fc = None
        if "field_cond" in sample:
            fc = pre.transform(
                field_cond=sample["field_cond"].unsqueeze(0), is_batch=True, is_trajectory=True
            )["field_cond"]
        for t in range(L, traj.shape[-1]):
            histories.append(traj[..., t - L : t])
            conds.append(fc)
    if not histories:
        raise ValueError("bank_indices gave no states")
    stride = max(1, len(histories) // int(max_states))
    histories = histories[::stride][: int(max_states)]
    conds = conds[::stride][: int(max_states)]
    fh = torch.cat(histories, 0).to(device)
    field_cond = None if conds[0] is None else torch.cat(conds, 0).to(device)
    bank = build_residual_bank(
        posterior,
        fh,
        num_steps=int(num_steps),
        stepper=stepper,
        mode=forecast_mode,
        pairs_per_state=int(pairs_per_state),
        field_cond=field_cond,
        batch_size=64,
        path=None,
        seed=seed,
    )
    bank.meta["indices"] = [int(i) for i in indices]
    if path is not None:
        bank.save(path)
    logger.info("[FC] built residual bank: %d residuals -> %s", len(bank), path)
    return bank


# --------------------------------------------------------------------------- #
# One cell's correction
# --------------------------------------------------------------------------- #


def build_correction(
    settings: Mapping,
    *,
    posterior: Any,
    num_steps: int,
    stepper: Callable,
    localisation: Optional[Localisation],
    steppers: Mapping[str, Callable],
    constraint: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    step_diagnostics: Optional[Callable[..., dict]] = None,
    prior: Any = None,
) -> tuple[ForecastCorrection, Optional[CorrectionMonitor]]:
    """``ForecastCorrection`` (+ monitor) for one cell from the merged settings.

    ``stepper`` / ``num_steps`` are the sampler's (defaults for the forecast);
    ``steppers`` maps ``"sde"``/``"ode"`` to step functions.
    """
    measure = settings.get("mode") == "measure"
    fstep = steppers[settings["forecast_stepper"]] if settings.get("forecast_stepper") else None
    fsteps = settings.get("forecast_num_steps")
    fsteps = None if fsteps is None else int(fsteps)
    q_source = str(settings.get("q_source", "online"))
    forecast_seed = settings.get("forecast_seed")
    if measure:
        # Trajectory must stay that of the plain run: no update, forked RNG.
        localisation = None
        constraint = None
        q_source = "online"
        if forecast_seed is None:
            forecast_seed = 0

    bank = None
    if q_source == "bank":
        path = settings.get("residual_bank")
        if not path:
            raise ValueError("forecast_correction.q_source=bank needs residual_bank (a path)")
        path = Path(str(path))
        if path.is_file():
            bank_obj = ResidualBank.load(path)
            meta = bank_obj.meta
            want = fsteps or int(num_steps)
            if int(meta.get("num_steps", want)) != want or meta.get("mode", settings.get("forecast_mode")) != settings.get("forecast_mode"):
                logger.warning(
                    "[FC] residual bank %s was built with %s; test-time forecasts use "
                    "num_steps=%d mode=%s -- Q_hat and K_ens are then inconsistent.",
                    path, meta, want, settings.get("forecast_mode"),
                )
        elif settings.get("bank_indices"):
            if prior is None:
                raise ValueError("building a residual bank needs the prior")
            bank_obj = build_bank_from_dataset(
                prior,
                posterior,
                settings["bank_indices"],
                num_steps=fsteps or int(num_steps),
                stepper=fstep or stepper,
                forecast_mode=str(settings.get("forecast_mode", "prior")),
                max_states=int(settings.get("bank_max_states", 256)),
                pairs_per_state=int(settings.get("bank_pairs_per_state", 1)),
                path=path,
            )
        else:
            raise FileNotFoundError(
                f"residual bank {path} does not exist; set "
                "forecast_correction.bank_indices to build it"
            )
        bank = bank_obj.residuals

    diagnostics = bool(settings.get("diagnostics", False)) or measure
    monitor: Optional[CorrectionMonitor] = None
    if diagnostics or constraint is not None:
        monitor = CorrectionMonitor(
            operator=posterior.likelihood_model.obs_operator,
            forecast_mode=str(settings.get("forecast_mode", "prior")),
            constraint=constraint,
            step_diagnostics=step_diagnostics if diagnostics else None,
            diagnostics=diagnostics,
            bank=bank,
        )

    dtype_name = settings.get("compute_dtype", "float64")
    compute_dtype = None if dtype_name in (None, "none", "state") else getattr(torch, str(dtype_name))
    cfg = ForecastCorrectionConfig(
        enabled=True,
        forecast_num_steps=fsteps,
        forecast_stepper=fstep,
        forecast_mode=str(settings.get("forecast_mode", "prior")),
        forecast_fn=monitor.forecast_fn if (monitor is not None and diagnostics) else None,
        localisation=localisation,
        gain_scale=0.0 if measure else float(settings.get("gain_scale", 1.0)),
        q_source=q_source,
        residual_bank=bank,
        group_size=settings.get("group_size"),
        update_cap=None if measure else settings.get("update_cap"),
        post_update=monitor.post_update if monitor is not None else None,
        forecast_seed=None if forecast_seed is None else int(forecast_seed),
        compute_dtype=compute_dtype,
        row_chunk=int(settings.get("row_chunk", 4096)),
    )
    correction = ForecastCorrection(cfg)
    logger.info(
        "[FC] %s: forecast steps=%s stepper=%s mode=%s | loc=%s gain=%s cap=%s Q=%s seed=%s diag=%s",
        "MEASURE-ONLY" if measure else "CORRECT",
        fsteps or f"M={num_steps}",
        settings.get("forecast_stepper") or "sampler's",
        cfg.forecast_mode,
        type(localisation).__name__ if localisation is not None else "global",
        cfg.gain_scale,
        cfg.update_cap,
        q_source,
        cfg.forecast_seed,
        diagnostics,
    )
    return correction, monitor


def write_cell_diagnostics(
    root: Path,
    case_name: str,
    ctx: Any,
    variant: Optional[str],
    settings: Mapping,
    correction: ForecastCorrection,
    monitor: Optional[CorrectionMonitor],
) -> Path:
    """Write ``<root>/<case>__<method>__<scenario>__<variant>__traj..json`` and log it."""
    import re

    def _slug(x: object) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", str(x)).strip("_")

    payload = summary(correction, monitor)
    test_index = int(ctx.extra.get("test_index", -1))
    payload.update(
        case=case_name,
        method=ctx.method.value,
        scenario=ctx.scenario.value,
        variant=variant,
        seed=int(ctx.seed),
        test_index=test_index,
        E=int(ctx.ensemble_size),
        M=int(ctx.num_steps),
        settings={k: v for k, v in settings.items() if k != "methods"},
    )
    path = Path(root) / (
        f"{case_name}__{_slug(ctx.method.value)}__{_slug(ctx.scenario.value)}"
        f"__{variant}__traj{test_index}__seed{ctx.seed}"
        f"__E{ctx.ensemble_size}_M{ctx.num_steps}.json"
    )
    write_diagnostics(path, payload)
    log_summary(f"{ctx.method.value} | {ctx.scenario.value} | {variant}", payload)
    logger.info("[FC] diagnostics -> %s", path)
    return path


def log_summary(tag: str, payload: Mapping) -> None:
    lag = payload.get("lag1_autocorrelation")
    logger.info(
        "[FC] %s | lag1 innovation autocorr=%s | mean normalised innov^2=%s | "
        "Q share (obs)=%s (state)=%s | cap fraction=%s",
        tag,
        None if lag is None else [round(v, 3) for v in lag],
        _round(payload.get("innovation_msq_mean")),
        _round(payload.get("q_share_obs_mean")),
        _round(payload.get("q_share_state_mean")),
        _round(payload.get("cap_fraction_mean")),
    )


def _round(x: Any) -> Any:
    return None if x is None else round(float(x), 4)


__all__ = [
    "DEFAULTS",
    "CorrectionMonitor",
    "MaskedLocalisation",
    "applies_to",
    "build_bank_from_dataset",
    "build_correction",
    "enstrophy_band_spectra",
    "fc_settings",
    "log_summary",
    "ns_localisation",
    "ns_step_diagnostics",
    "resolve_radius",
    "summary",
    "urban_localisation",
    "variant_suffix",
    "variant_with_correction",
    "write_cell_diagnostics",
    "write_diagnostics",
    "zero_mean_increment",
]
