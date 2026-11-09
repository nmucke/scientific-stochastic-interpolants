"""Forecast-residual correction wiring for the gridded cases (Navier--Stokes, urban).

The correction is ``scisi.posterior_models.forecast_correction`` (background:
``FORECAST_CORRECTION.md``). This module turns the case block
``case.forecast_correction`` (the YAML is the only place defaults live) into a
``ForecastCorrection`` for one cell, builds the ``variant`` tag that keeps corrected
rows apart from plain ones, and writes one diagnostics JSON per cell.

With ``enabled: false`` (the default) nothing is built: ``sample_trajectory`` gets
``forecast_correction=None`` and the run, its variant tag and its files are exactly
those of a plain run.

Measuring without correcting ("measure before fixing", note section 5) is
``gain_scale=0 forecast_seed=0 diagnostics=true``: the forecasts then come from a
forked RNG, so the guided trajectory is bit-for-bit the plain one.

CLI (used by the grid scripts, so file names and the ``variant`` column share one
implementation)::

    python -m common.forecast_correction tag navier_stokes \
        case.forecast_correction.enabled=true case.forecast_correction.gain_scale=0.5
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

#: Keys that never change the numbers (they are kept out of the tag hash).
NON_NUMERIC_KEYS = frozenset({"enabled", "methods", "diagnostics", "diagnostics_root"})

Constraint = Callable[[Any, Any], Any]


# --------------------------------------------------------------------------- #
# Settings and the variant tag
# --------------------------------------------------------------------------- #


def fc_settings(case_cfg: Any) -> dict:
    """The ``case.forecast_correction`` block as a plain dict (``{}``: off)."""
    block = None if case_cfg is None else case_cfg.get("forecast_correction", None)
    if block is None:
        return {}
    from omegaconf import OmegaConf

    if OmegaConf.is_config(block):
        block = OmegaConf.to_container(block, resolve=True)
    return dict(block)


def applies_to(settings: Mapping, method_name: str) -> bool:
    """Whether the correction is switched on for this method."""
    return bool(settings.get("enabled", False)) and str(method_name) in {
        str(m) for m in settings.get("methods") or ()
    }


def _canon(value: Any) -> Any:
    """JSON-stable form: numbers as floats (``1`` == ``1.0``), sorted mappings."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping):
        return {str(k): _canon(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    return str(value)


def fc_tag(settings: Mapping) -> str:
    """``fc<steps>_<hash8>``: forecast steps (``M`` = the sampler's) and the first 8
    hex of the sha1 of every setting that changes the numbers. File-name safe."""
    payload = {k: v for k, v in settings.items() if k not in NON_NUMERIC_KEYS}
    digest = hashlib.sha1(json.dumps(_canon(payload), sort_keys=True).encode()).hexdigest()
    steps = settings.get("forecast_num_steps")
    return f"fc{'M' if steps is None else int(steps)}_{digest[:8]}"


def variant_with_correction(base: Optional[str], settings: Mapping, method_name: str) -> Optional[str]:
    """``base`` when the correction is off for this method, else ``<base>_<tag>``."""
    if not applies_to(settings, method_name):
        return base
    tag = fc_tag(settings)
    return f"{base}_{tag}" if base else tag


def resolve_radius(radius: Any, scenario_key: Optional[str]) -> Optional[float]:
    """Scalar, ``None`` or ``{scenario_key: r, default: r}`` -> float or None."""
    if isinstance(radius, Mapping):
        radius = radius.get(scenario_key, radius.get("default", None))
    return None if radius is None else float(radius)


# --------------------------------------------------------------------------- #
# Post-update (constraints + per-step diagnostics)
# --------------------------------------------------------------------------- #


class CorrectionMonitor:
    """``post_update(x_new, x_guided)``: keeps the guided values where ``mask`` is
    False (e.g. inside buildings), applies ``constraint``, and (with diagnostics)
    records ``step_diagnostics(x_guided, x_new, x_out)`` per corrected step."""

    def __init__(self, mask: Any = None, constraint: Optional[Constraint] = None,
                 step_diagnostics: Optional[Callable[..., dict]] = None) -> None:
        self.mask = mask
        self.constraint = constraint
        self.step_diagnostics = step_diagnostics
        self.records: list[dict] = []

    def post_update(self, x_new, x_guided):  # type: ignore[no-untyped-def]
        import torch

        x_out = x_new
        if self.mask is not None:
            x_out = torch.where(self.mask.to(x_new.device), x_out, x_guided)
        if self.constraint is not None:
            x_out = self.constraint(x_out, x_guided)
        if self.step_diagnostics is not None:
            self.records.append(self.step_diagnostics(x_guided, x_new, x_out))
        return x_out


# --------------------------------------------------------------------------- #
# One cell's correction
# --------------------------------------------------------------------------- #


def build_correction(
    settings: Mapping,
    *,
    case: str,
    ctx: Any,
    state_shape: Sequence[int],
    scenario_key: Optional[str],
    periodic: Any = False,
    mask: Any = None,
    constraint: Optional[Constraint] = None,
    step_diagnostics: Optional[Callable[..., dict]] = None,
) -> tuple[Any, Optional[CorrectionMonitor]]:
    """``(ForecastCorrection, monitor)`` for one cell from the case block.

    Gaspari-Cohn localisation on the state grid (``localisation_radius`` cells;
    ``None`` = global; a mapping gives per-scenario radii), observation sites from
    ``H`` itself (a nonlinear ``h``: its linear sensor map ``h.inner``; the gains
    are EnKF-style in ``h(f_i)``, see the library docstring). ``mask`` (bool, state shape): False entries get a zero taper and
    keep their guided values. ``forecast_seed``: when set, the forecasts use a
    forked RNG seeded per (case, scenario, test_index, seed, forecast_seed).
    """
    from common.seeding import derive_seed
    from scisi.posterior_models.forecast_correction import (
        ForecastCorrection,
        ForecastCorrectionConfig,
        grid_localisation,
    )
    from scisi.sampling.ode_solvers import euler_step
    from scisi.sampling.sde_solvers import euler_maruyama_step

    steppers = {"sde": euler_maruyama_step, "ode": euler_step}
    name = settings.get("forecast_stepper")
    if name is not None and name not in steppers:
        raise ValueError(f"forecast_correction.forecast_stepper must be null, sde or ode; got {name!r}")
    radius = resolve_radius(settings.get("localisation_radius"), scenario_key)
    loc = None if radius is None else grid_localisation(
        tuple(int(s) for s in state_shape), radius, periodic=periodic, mask=mask)
    diagnostics = bool(settings.get("diagnostics", False))
    monitor = None
    if mask is not None or constraint is not None or (diagnostics and step_diagnostics):
        monitor = CorrectionMonitor(mask, constraint, step_diagnostics if diagnostics else None)
    seed = settings.get("forecast_seed")
    if seed is not None:
        seed = derive_seed("forecast_correction", case, ctx.scenario.value,
                           ctx.extra.get("test_index", -1), ctx.seed, seed)
    steps = settings.get("forecast_num_steps")
    cfg = ForecastCorrectionConfig(
        enabled=True,
        forecast_num_steps=None if steps is None else int(steps),
        forecast_stepper=steppers[name] if name else None,
        localisation=loc,
        gain_scale=float(settings.get("gain_scale", 1.0)),
        q_source=str(settings.get("q_source", "online")),
        residual_bank=settings.get("residual_bank"),
        update_cap=settings.get("update_cap"),
        post_update=None if monitor is None else monitor.post_update,
        forecast_seed=seed,
    )
    logger.info("[FC] %s: steps=%s stepper=%s radius=%s gain=%s cap=%s Q=%s seed=%s",
                fc_tag(settings), steps or f"M={ctx.num_steps}", name or "sampler's", radius,
                cfg.gain_scale, cfg.update_cap, cfg.q_source, seed)
    return ForecastCorrection(cfg), monitor


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def summary(correction: Any, monitor: Optional[CorrectionMonitor]) -> dict:
    """JSON-ready: the library's per-step records ``[steps][groups]``, the lag-1
    innovation autocorrelation, the Q share ``(q_rms / spread_rms)^2`` and means."""
    import torch

    d = correction.diagnostics
    lag = d.lag1_autocorrelation()
    out: dict = {"lag1_autocorrelation": None if lag is None else lag.tolist()}
    for key, value in d.as_dict().items():
        out[key] = value.tolist() if torch.is_tensor(value) else list(value)
    if "q_rms" in out:
        q, s = torch.tensor(out["q_rms"]), torch.tensor(out["spread_rms"])
        out["q_share_state"] = (q / s.clamp_min(1e-300)).pow(2).tolist()
    for key in ("innovation_msq", "cap_fraction", "q_share_state"):
        if out.get(key):
            out[f"{key}_mean"] = float(torch.tensor(out[key]).mean())
    if monitor is not None and monitor.records:
        out["steps"] = monitor.records
    return out


def _slug(x: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(x)).strip("_")


def write_cell_diagnostics(root: Path, case_name: str, ctx: Any, variant: Optional[str],
                           settings: Mapping, correction: Any,
                           monitor: Optional[CorrectionMonitor], obs_operator: Any = None) -> Path:
    """``<root>/<case>__<method>__<scenario>__<variant>[__c<c>]__traj<i>__seed<s>__E.._M...json``.

    ``obs_operator``: adds the ``states_tag`` (``__c<c>`` for a strength-carrying
    nonlinear ``h``, as the saved states) so a c-sweep does not overwrite one file,
    and the operator metadata (``obs_linear``, ``obs_nonlinearity``, ``obs_strength``).
    """
    from common.obs_scenario import states_tag

    payload = summary(correction, monitor)
    test_index = int(ctx.extra.get("test_index", -1))
    payload.update(case=case_name, method=ctx.method.value, scenario=ctx.scenario.value,
                   variant=variant, seed=int(ctx.seed), test_index=test_index,
                   E=int(ctx.ensemble_size), M=int(ctx.num_steps),
                   settings={k: v for k, v in settings.items() if k != "methods"})
    obs_tag = ""
    if obs_operator is not None:
        obs_tag = states_tag(obs_operator)
        strength = getattr(obs_operator, "strength", None)
        linear = bool(getattr(obs_operator, "is_linear", True))
        payload.update(obs_linear=linear,
                       obs_nonlinearity=str(getattr(obs_operator, "nonlinearity", "") or ""),
                       obs_strength=None if linear or strength is None else float(strength))
    path = Path(root) / (
        f"{case_name}__{_slug(ctx.method.value)}__{_slug(ctx.scenario.value)}__{variant}{obs_tag}"
        f"__traj{test_index}__seed{ctx.seed}__E{ctx.ensemble_size}_M{ctx.num_steps}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=float))
    rnd = lambda x: None if x is None else round(float(x), 4)  # noqa: E731
    lag = payload["lag1_autocorrelation"]
    logger.info("[FC] %s | %s | %s | lag1 innov autocorr=%s | mean normalised innov^2=%s | "
                "Q share=%s | cap fraction=%s -> %s", ctx.method.value, ctx.scenario.value, variant,
                None if lag is None else [round(v, 3) for v in lag],
                rnd(payload.get("innovation_msq_mean")), rnd(payload.get("q_share_state_mean")),
                rnd(payload.get("cap_fraction_mean")), path)
    return path


# --------------------------------------------------------------------------- #
# Driver mixin (NS + urban)
# --------------------------------------------------------------------------- #


class ForecastCorrectionMixin:
    """``_fc_settings`` / ``_variant`` / ``_write_fc_diagnostics`` for a case runner
    with ``config``, ``case`` and ``_base_variant(method)``."""

    def _fc_settings(self) -> dict:
        if "_fc_cache" not in self.__dict__:
            try:
                case_cfg = self.config.case  # type: ignore[attr-defined]
            except Exception:
                case_cfg = None
            self.__dict__["_fc_cache"] = fc_settings(case_cfg)
        return self.__dict__["_fc_cache"]

    def _variant(self, method, scenario=None) -> Optional[str]:  # type: ignore[no-untyped-def]
        """Tidy ``variant``: ``_base_variant`` plus ``_fc<steps>_<hash>`` for a
        corrected cell (off -> exactly the plain tag). ``scenario`` is unused."""
        return variant_with_correction(
            self._base_variant(method), self._fc_settings(),  # type: ignore[attr-defined]
            getattr(method, "value", str(method)))

    def _write_fc_diagnostics(self, ctx, correction, monitor, obs_operator=None) -> Path:  # type: ignore[no-untyped-def]
        from common.paths import RESULTS_ROOT, output_path

        settings = self._fc_settings()
        case = self.case.value  # type: ignore[attr-defined]
        root = output_path(str(settings.get("diagnostics_root")
                               or RESULTS_ROOT / case / "forecast_correction"))
        return write_cell_diagnostics(root, case, ctx, self._variant(ctx.method), settings,
                                      correction, monitor, obs_operator=obs_operator)


# --------------------------------------------------------------------------- #
# CLI: the tag for a set of Hydra overrides (grid scripts)
# --------------------------------------------------------------------------- #


def _cli_tag(case: str, overrides: Sequence[str]) -> str:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from common.paths import EXPERIMENT_ROOT

    OmegaConf.register_new_resolver("experiment_root", lambda: str(EXPERIMENT_ROOT), replace=True)
    configs = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(configs), version_base=None):
        cfg = compose(config_name="benchmark", overrides=[f"case={case}", *overrides])
    return fc_tag(fc_settings(cfg.case))


if __name__ == "__main__":  # pragma: no cover - exercised by the grid scripts / tests
    if len(sys.argv) < 3 or sys.argv[1] != "tag":
        sys.exit("usage: python -m common.forecast_correction tag <case> [hydra overrides...]")
    print(_cli_tag(sys.argv[2], sys.argv[3:]))
