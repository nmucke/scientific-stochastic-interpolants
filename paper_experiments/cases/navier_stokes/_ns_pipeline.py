"""Navier--Stokes assimilation pipeline helpers (case-local; not in common/).

This module owns the heavy lifting the :class:`NavierStokesRunner` delegates to:
loading the trained prior + dataset, building the per-scenario observation
operator, running one autoregressive assimilation with a chosen sampler, and
computing the full metric set (spec Section 3) on the posterior ensemble.

It deliberately lives under ``cases/navier_stokes/`` (not ``common/``) so the
analytical driver, which owns ``common/``, is untouched. The runner plugs into
the rebuilt ``scisi`` samplers / metrics / observation operators through this
seam.

Loading note (GAP m7): the on-disk dataset is ``data/stochastic_navier_stokes/
data.npz`` with a single key ``state`` of shape ``[T, N, H, W]``;
``scisi.data.load_data.load_stochastic_navier_stokes`` already reads
``data["state"]`` and the training configs already point at ``data.npz``, so the
generator/loader key mismatch the audit flagged is *already reconciled* on the
loader side -- no fix is required here, only verification (done in the driver).

Weights note: the smoke configuration runs with whatever weights are present in
the checkpoint dir. When no ``model.pth`` exists (the case in this CPU/MPS env),
the model is built from the checkpoint's architecture config with **random**
weights so the pipeline can be exercised end-to-end; the numbers are then
smoke-scale only and a full-scale run must point at a dir that carries the
trained ``model.pth`` (see the driver's ``require_weights`` switch).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from scisi.likelihood_models.gaussian_likelihood import (
    FlowdasGaussianLikelihood,
    InterpolantGaussianLikelihood,
)
from scisi.likelihood_models.guidance import (
    DPSGaussianLikelihood,
    FIGGaussianLikelihood,
    GuidanceGaussianLikelihood,
)
from scisi.likelihood_models.observation_operators import LinearObservationOperator
from scisi.likelihood_models.sda import SDALikelihood
from scisi.metrics.accuracy import ensemble_mean_rmse
from scisi.metrics.calibration import crps, rank_histogram, spread_skill
from scisi.metrics.cost import NFECounter, StepTimer
from scisi.metrics.distributional import kl_at_points
from scisi.metrics.spectral import energy_spectrum_rmse
from scisi.models.diffusion_model import DenoiseDiffusionModel
from scisi.models.flow_matching_model import FlowMatchingModel
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
from scisi.models.interpolations import LinearDeterministicInterpolation
from scisi.posterior_models.flow_matching_posterior import (
    FlowMatchingPosterior,
    endpoint_vanishing_diffusion,
)
from scisi.posterior_models.stochastic_interpolant_posterior import (
    StochasticInterpolantPosterior,
)
from scisi.sampling.ode_solvers import euler_step
from scisi.sampling.sde_solvers import euler_maruyama_step

logger = logging.getLogger(__name__)

_STEPPERS = {"sde": euler_maruyama_step, "ode": euler_step}


# --------------------------------------------------------------------------- #
# Model + dataset loading
# --------------------------------------------------------------------------- #


@dataclass
class LoadedPrior:
    """The shared trained prior + the bits the loop needs around it."""

    si_model: torch.nn.Module  # FollmerStochasticInterpolant (SI / FlowDAS)
    fm_model: torch.nn.Module  # FlowMatchingModel (DM-SDE / FM-ODE)
    preprocesser: Any
    test_dataset: Any
    len_field_history: int
    train_cfg: DictConfig
    checkpoint_name: str
    has_trained_weights: bool
    diffusion_model: Optional[torch.nn.Module] = None  # DenoiseDiffusionModel (DPS)


def _checkpoint_dir(project: str, name: str) -> Path:
    return Path("checkpoints") / project / name


def _configured_run(case_cfg: DictConfig, role: str) -> Optional[str]:
    """Read the configured checkpoint run name for ``role`` ('si' or 'fm').

    Prefers the explicit ``checkpoints.si_run`` / ``checkpoints.fm_run`` keys
    (the GPU-box handoff keys); falls back to the legacy per-method map
    (``checkpoints["Ours (SI-SDE)"]`` etc.) for backward compatibility.
    """
    ckpts = case_cfg.get("checkpoints", {}) if case_cfg is not None else {}
    explicit_key = f"{role}_run"
    if explicit_key in ckpts and ckpts[explicit_key]:
        return str(ckpts[explicit_key])
    legacy_key = "Ours (SI-SDE)" if role == "si" else "Ours (DM-SDE)"
    if legacy_key in ckpts and ckpts[legacy_key]:
        return str(ckpts[legacy_key])
    return None


def _resolve_run_dir(
    case_cfg: DictConfig, role: str, require_weights: bool
) -> tuple[str, Path, bool]:
    """Resolve the checkpoint dir + whether it carries weights for ``role``.

    Returns ``(run_name, run_dir, has_weights)``. When ``require_weights`` is
    True the configured dir AND its ``model.pth`` must exist or this raises (no
    silent fallback to an arbitrary architecture). When False, a missing
    configured dir falls back to the first available dir under the project so the
    random-weights smoke can still run; a loud warning is logged by the caller.
    """
    project = case_cfg.project
    proj_dir = Path("checkpoints") / project
    configured = _configured_run(case_cfg, role)

    if configured and _checkpoint_dir(project, configured).is_dir():
        run_dir = _checkpoint_dir(project, configured)
        has_weights = (run_dir / "model.pth").is_file()
        if require_weights and not has_weights:
            raise FileNotFoundError(
                f"require_weights=True but no model.pth at {run_dir} "
                f"(configured {role}_run={configured!r}). Point checkpoints."
                f"{role}_run at a dir that holds the trained weights, or set "
                f"require_weights=false to run with random init."
            )
        return configured, run_dir, has_weights

    # Configured dir missing.
    if require_weights:
        raise FileNotFoundError(
            f"require_weights=True but the configured {role}_run "
            f"({configured!r}) is not a dir under {proj_dir}. Set checkpoints."
            f"{role}_run to a real run name on the GPU box, or set "
            f"require_weights=false to run with random init."
        )

    candidates = sorted(p.name for p in proj_dir.iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint dirs under {proj_dir}; cannot load a prior."
        )
    fallback = candidates[0]
    run_dir = _checkpoint_dir(project, fallback)
    has_weights = (run_dir / "model.pth").is_file()
    logger.warning(
        "Configured %s checkpoint %r not found; falling back to %r "
        "(architecture config only).",
        role,
        configured,
        fallback,
    )
    return fallback, run_dir, has_weights


def _build_fm_from_si(si_cfg: DictConfig) -> FlowMatchingModel:
    """Build an FM model reusing the SI checkpoint's drift architecture.

    The FM prior must share architecture + data with the SI prior (spec Section
    5 / reproducibility Section 9). We reuse the trained checkpoint's UNet drift
    config and pair it with a rectified-flow (linear deterministic) interpolation
    so ``FlowMatchingModel.score`` / ``velocity_score_coeff`` are well-defined.
    """
    drift_model = hydra.utils.instantiate(si_cfg.model.drift_model)
    interpolation = LinearDeterministicInterpolation()
    return FlowMatchingModel(interpolation=interpolation, drift_model=drift_model)


def load_prior(case_cfg: DictConfig, device: str) -> LoadedPrior:
    """Load (or build) the shared SI + FM priors and the test dataset.

    The SI prior is loaded from ``checkpoints.si_run``; the FM prior from
    ``checkpoints.fm_run`` when a dedicated FM checkpoint exists, else it reuses
    the SI architecture + (if present) the SI drift weights paired with a
    rectified-flow interpolation. ``case.require_weights`` (default False)
    controls fallback: when True a missing dir / ``model.pth`` is a hard error;
    when False the random-init path runs with a LOUD warning.
    """
    project = case_cfg.project
    require_weights = bool(case_cfg.get("require_weights", False))

    # --- SI prior (also the architecture source for the FM prior) ---------- #
    si_run, si_dir, si_has_weights = _resolve_run_dir(case_cfg, "si", require_weights)
    train_cfg = OmegaConf.load(si_dir / "config.yaml")

    try:
        len_field_history = train_cfg.model.drift_model.len_field_history
    except Exception:
        len_field_history = train_cfg.model.denoise_model.len_field_history

    si_model = hydra.utils.instantiate(train_cfg.model)
    if si_has_weights:
        si_model.load_state_dict(torch.load(si_dir / "model.pth", map_location="cpu"))
        logger.info("Loaded SI weights from %s", si_dir / "model.pth")

    # --- FM prior: dedicated fm_run if present, else reuse SI architecture -- #
    fm_run = _configured_run(case_cfg, "fm")
    fm_dir = _checkpoint_dir(project, fm_run) if fm_run else None
    fm_has_weights = False
    if fm_dir is not None and fm_dir.is_dir() and (fm_dir / "config.yaml").is_file():
        fm_train_cfg = OmegaConf.load(fm_dir / "config.yaml")
        fm_model = hydra.utils.instantiate(fm_train_cfg.model)
        if (fm_dir / "model.pth").is_file():
            fm_model.load_state_dict(
                torch.load(fm_dir / "model.pth", map_location="cpu")
            )
            fm_has_weights = True
            logger.info("Loaded FM weights from %s", fm_dir / "model.pth")
    else:
        # No dedicated FM checkpoint (the current repo state): reuse the SI UNet
        # drift architecture + a rectified-flow interpolation so .score is defined.
        fm_model = _build_fm_from_si(train_cfg)
        if si_has_weights:
            fm_model.drift_model.load_state_dict(si_model.drift_model.state_dict())
            fm_has_weights = True  # shares the trained SI drift weights

    has_weights = si_has_weights and fm_has_weights
    if not has_weights:
        # Loud, unmissable warning so smoke-scale numbers are never mistaken for
        # real results. require_weights=True would already have hard-failed above.
        missing = []
        if not si_has_weights:
            missing.append(f"SI ({si_dir})")
        if not fm_has_weights:
            missing.append("FM (no dedicated fm_run; SI drift weights also absent)")
        logger.warning(
            "=" * 78
            + "\nWARNING: running with RANDOM weights (no model.pth at: %s)."
            "\n         Results are SMOKE-SCALE ONLY -- not valid paper numbers."
            "\n         Set checkpoints.si_run / checkpoints.fm_run to real runs"
            "\n         and require_weights=true on the GPU box.\n" + "=" * 78,
            "; ".join(missing),
        )

    si_model.eval().to(device)
    fm_model.eval().to(device)

    # --- Diffusion prior (for the DPS / Guided-diffusion baseline) ---------- #
    # The Guided-diffusion baseline needs a diffusion prior. Two sources:
    #
    #  * ``checkpoints.diffusion_from_fm: true`` (DEFAULT) -- build it FROM the
    #    well-trained FM model via ``DenoiseDiffusionModel.from_flow_matching``
    #    (velocity mode: score / reverse-SDE drift reconstructed from the FM
    #    velocity). Preferred because the dedicated diffusion checkpoint is poorly
    #    trained; the FM prior is the same architecture, better optimised.
    #  * else -- a DEDICATED trained diffusion model loaded from
    #    ``checkpoints.diffusion_run`` (default "diffusion_model").
    #
    # None when neither a usable FM prior nor a diffusion checkpoint is available.
    diffusion_model = None
    diffusion_from_fm = bool(case_cfg.get("checkpoints", {}).get("diffusion_from_fm", True))
    if diffusion_from_fm:
        diffusion_model = DenoiseDiffusionModel.from_flow_matching(fm_model)
        diffusion_model.eval().to(device)
        logger.info("Built diffusion prior from the FM model (diffusion_from_fm=True).")
    else:
        diffusion_model = _load_diffusion_checkpoint(case_cfg, project, device, require_weights)

    preprocesser = hydra.utils.instantiate(train_cfg.preprocesser)
    test_dataset = hydra.utils.instantiate(train_cfg.test_data)

    return LoadedPrior(
        si_model=si_model,
        fm_model=fm_model,
        preprocesser=preprocesser,
        test_dataset=test_dataset,
        len_field_history=len_field_history,
        train_cfg=train_cfg,
        checkpoint_name=si_run,
        has_trained_weights=has_weights,
        diffusion_model=diffusion_model,
    )


def _load_diffusion_checkpoint(
    case_cfg: DictConfig, project: str, device: str, require_weights: bool
) -> Optional[torch.nn.Module]:
    """Load a dedicated trained diffusion model from ``checkpoints.diffusion_run``.

    Returns None when no diffusion checkpoint dir/config is configured. Used only
    when ``checkpoints.diffusion_from_fm`` is False.
    """
    diffusion_model = None
    diff_run = _configured_run(case_cfg, "diffusion")
    diff_dir = _checkpoint_dir(project, diff_run) if diff_run else None
    if diff_dir is not None and diff_dir.is_dir() and (diff_dir / "config.yaml").is_file():
        diff_train_cfg = OmegaConf.load(diff_dir / "config.yaml")
        diffusion_model = hydra.utils.instantiate(diff_train_cfg.model)
        if (diff_dir / "model.pth").is_file():
            diffusion_model.load_state_dict(
                torch.load(diff_dir / "model.pth", map_location="cpu")
            )
            logger.info("Loaded diffusion weights from %s", diff_dir / "model.pth")
        elif require_weights:
            raise FileNotFoundError(
                f"require_weights=True but no model.pth at {diff_dir} "
                f"(diffusion_run={diff_run!r})."
            )
        diffusion_model.eval().to(device)
    elif require_weights and diff_run:
        raise FileNotFoundError(
            f"require_weights=True but diffusion_run {diff_run!r} dir/config.yaml "
            f"is missing under checkpoints/{project}/."
        )

    return diffusion_model


# --------------------------------------------------------------------------- #
# NFE counting: wrap the drift network's forward with a counter.
# --------------------------------------------------------------------------- #


def attach_nfe_counter(model: torch.nn.Module) -> NFECounter:
    """Wrap ``model.drift_model.forward`` so each network eval bumps a counter.

    Counts every forward call of the drift network -- the prior velocity/drift
    eval, the FM score eval (which calls the drift net again), and the
    Jacobian-vector products inside the full-Sigma_s likelihood (each JVP runs
    one extra forward). This is the true NFE per assimilation step (spec 3d).
    """
    counter = NFECounter()
    # The network attribute is ``drift_model`` for SI/FM and ``denoise_model``
    # for the diffusion prior (DenoiseDiffusionModel); fall back to ``.model``.
    net = (
        getattr(model, "drift_model", None)
        or getattr(model, "denoise_model", None)
        or model.model
    )
    if getattr(net, "_nfe_wrapped", False):
        net._nfe_counter = counter  # type: ignore[attr-defined]
        return counter

    original_forward = net.forward

    def counting_forward(*args: Any, **kwargs: Any):  # noqa: ANN401
        net._nfe_counter.increment()  # type: ignore[attr-defined]
        return original_forward(*args, **kwargs)

    net._nfe_counter = counter  # type: ignore[attr-defined]
    net._nfe_wrapped = True  # type: ignore[attr-defined]
    net.forward = counting_forward  # type: ignore[assignment]
    return counter


# --------------------------------------------------------------------------- #
# Observation operator per scenario (seeded, shared across methods).
# --------------------------------------------------------------------------- #


def build_obs_operator(
    scenario_cfg: DictConfig,
    data_size: tuple[int, int, int],
    mask_seed: int,
) -> LinearObservationOperator:
    """Build the scenario's observation operator with a shared seeded mask.

    Sparse scenarios use a fixed seeded point mask (``mask_seed`` so the mask is
    identical across methods, spec Section 9); super-res scenarios use the
    deterministic block-average operator (no seed needed). ``data_size`` is the
    model-grid ``(C, H, W)``.
    """
    obs_cfg = OmegaConf.to_container(scenario_cfg.obs_operator, resolve=True)
    obs_cfg.pop("_target_", None)
    if obs_cfg.get("type") == "random":
        obs_cfg["seed"] = int(mask_seed)
    return LinearObservationOperator(data_size=tuple(data_size), **obs_cfg)


# --------------------------------------------------------------------------- #
# One trajectory's truth + observations (seeded, shared across methods).
# --------------------------------------------------------------------------- #


@dataclass
class TruthAndObs:
    """A single test trajectory's normalised truth, history and observations."""

    init_base: torch.Tensor  # [1, C, H, W]  x0 (normalised)
    field_history: torch.Tensor  # [1, C, H, W, L]
    field_cond: Optional[torch.Tensor]
    pars_cond: Optional[torch.Tensor]
    true_trajectory: torch.Tensor  # [1, C, H, W, T] (normalised)
    observations: torch.Tensor  # [1, N_y, T]
    obs_operator: LinearObservationOperator


def prepare_truth_and_obs(
    prior: LoadedPrior,
    test_index: int,
    obs_operator: LinearObservationOperator,
    variance: float,
    num_physical_steps: int,
    obs_noise_seed: int,
    device: str,
) -> TruthAndObs:
    """Pull one test trajectory, normalise it, and draw seeded observations."""
    pre = prior.preprocesser
    L = prior.len_field_history

    sample = prior.test_dataset[test_index]
    trajectory = sample["x"].unsqueeze(0)  # [1, C, H, W, T_full]
    field_cond = sample["field_cond"].unsqueeze(0) if "field_cond" in sample else None
    pars_cond = sample["pars_cond"].unsqueeze(0) if "pars_cond" in sample else None

    init_data = pre.transform(
        base=trajectory[..., L - 1],
        field_history=trajectory[..., 0:L],
        is_batch=True,
    )
    if field_cond is not None:
        init_data["field_cond"] = pre.transform(
            field_cond=field_cond, is_batch=True, is_trajectory=True
        )["field_cond"]
    else:
        init_data["field_cond"] = None
    if pars_cond is not None:
        init_data["pars_cond"] = pre.transform(
            pars_cond=pars_cond, is_batch=True, is_trajectory=True
        )["pars_cond"]
    else:
        init_data["pars_cond"] = None

    transformed_traj = pre.transform(
        base=trajectory, is_batch=True, is_trajectory=True
    )["base"]  # [1, C, H, W, T_full]
    transformed_traj = transformed_traj[..., :num_physical_steps]

    # Observations are drawn from the NORMALISED truth so they live in the same
    # space the sampler operates in (the operator + R are defined on that space).
    gen = torch.Generator(device="cpu").manual_seed(obs_noise_seed)
    sigma = float(variance) ** 0.5
    observations = torch.zeros(1, obs_operator.num_obs, num_physical_steps)
    for i in range(num_physical_steps):
        clean = obs_operator(transformed_traj[..., i].to(device)).cpu()
        observations[:, :, i] = clean + torch.randn(
            clean.shape, generator=gen
        ) * sigma

    return TruthAndObs(
        init_base=init_data["base"],
        field_history=init_data["field_history"],
        field_cond=init_data["field_cond"],
        pars_cond=init_data["pars_cond"],
        true_trajectory=transformed_traj,
        observations=observations,
        obs_operator=obs_operator,
    )


# --------------------------------------------------------------------------- #
# Sampler construction (SI-SDE / DM-SDE / FM-ODE / FlowDAS).
# --------------------------------------------------------------------------- #


def _is_schedule_table(value: Any) -> bool:
    """A hyperparameter is a per-``[scenario][M]`` table iff it is a mapping that
    carries a ``default:`` key (the explicit marker). Any other value -- scalar,
    ``null``, string, or a plain nested mapping -- is left untouched.
    """
    if isinstance(value, DictConfig):
        return "default" in value
    return isinstance(value, dict) and "default" in value


def resolve_scheduled_hparam(
    value: Any,
    scenario_key: Optional[str] = None,
    num_steps: Optional[int] = None,
    name: str = "hparam",
) -> Any:
    """Resolve ONE hyperparameter that may be a scalar OR a ``[scenario][M]`` table.

    Model-agnostic: works for any per-cell-tuned hyperparameter (FlowDAS's
    ``guidance_scale`` today; a future method's ``step_size``, ``K``,
    ``max_grad_norm``, ...). The value may be

    - anything that is not a schedule table (scalar, ``null``, ...) -> returned
      as-is; or
    - a table ``{default: x, <scenario>: {<M>: v, ...}, ...}`` -> looked up by
      ``scenario_key`` then ``num_steps`` (int/str keys both work).

    A missing scenario, missing ``M`` column, or ``null`` entry (an untuned cell)
    falls back to the table's ``default`` with a warning, so a single table can
    hold ``null`` placeholders for M values not yet tuned.
    """
    if not _is_schedule_table(value):
        return value

    table = (
        OmegaConf.to_container(value, resolve=True)
        if isinstance(value, DictConfig)
        else dict(value)
    )
    tbl_default = table.get("default")
    scen = table.get(scenario_key) if scenario_key is not None else None
    if scen is None:
        logger.warning(
            "%s: no entry for scenario=%r; using default %s.",
            name, scenario_key, tbl_default,
        )
        return tbl_default
    scen = {str(k): v for k, v in scen.items()}
    val = scen.get(str(num_steps))
    if val is None:
        logger.warning(
            "%s: scenario=%r M=%s not tuned yet (null); using default %s.",
            name, scenario_key, num_steps, tbl_default,
        )
        return tbl_default
    return val


def resolve_scheduled_hparams(
    block: Any,
    scenario_key: Optional[str] = None,
    num_steps: Optional[int] = None,
) -> dict:
    """Resolve EVERY scheduled hyperparameter in a config block to a scalar.

    Walks a method's config mapping and replaces each value that is a
    ``[scenario][M]`` schedule table (see :func:`resolve_scheduled_hparam`) with
    its resolved entry for ``(scenario_key, num_steps)``; scalars and other
    values pass through unchanged. Lets a new method opt any of its
    hyperparameters into per-cell tuning with zero bespoke wiring -- just write a
    table (with a ``default:``) in its config and read the resolved value here.
    """
    if block is None:
        return {}
    items = (
        OmegaConf.to_container(block, resolve=True).items()
        if isinstance(block, DictConfig)
        else dict(block).items()
    )
    return {
        k: resolve_scheduled_hparam(v, scenario_key, num_steps, name=str(k))
        for k, v in items
    }


def build_posterior(
    method_name: str,
    method_cfg: DictConfig,
    prior: LoadedPrior,
    obs_operator: LinearObservationOperator,
    variance: float,
    likelihood_ensemble_size: int,
    likelihood_mode: Optional[str] = None,
    g_multiplier: Optional[float] = None,
    scenario_key: Optional[str] = None,
    num_steps: Optional[int] = None,
) -> tuple[torch.nn.Module, torch.nn.Module, Callable]:
    """Instantiate (likelihood, posterior, stepper) for one method.

    Returns the trained model used, the posterior sampler, and the SDE/ODE
    stepper. ``likelihood_mode`` / ``g_multiplier`` override the config (used by
    the ablation sweep). ``scenario_key`` (e.g. ``"superres_16"``) and
    ``num_steps`` (M) select the per-cell guidance scale from a config table.
    """
    stepper = _STEPPERS[method_cfg.stepper]

    if method_name == "FlowDAS":
        import os

        model = prior.si_model
        lik_cfg = method_cfg.get("likelihood_model", {}) or {}
        # Resolve any per-(scenario, M) scheduled hyperparameters in the config
        # block to scalars (scalars pass through unchanged). Env vars still take
        # top precedence so a tuning sweep needs no edit to the tracked YAML.
        hp = resolve_scheduled_hparams(lik_cfg, scenario_key, num_steps)
        _z = hp.get("guidance_scale", 1.0)
        zeta = float(os.environ.get("FLOWDAS_ZETA", _z if _z is not None else 1.0))
        _mgn = os.environ.get("FLOWDAS_MAX_GRAD_NORM", hp.get("max_grad_norm", None))
        max_grad_norm = None if _mgn in (None, "", "null", "None") else float(_mgn)
        # J = MC x_1 draws per member (paper Eq. 10; 25 irrespective of E).
        # Decoupled from the DA ensemble size; falls back to the legacy
        # likelihood_ensemble_size when the config key is absent.
        n_mc = int(hp.get("num_mc_samples") or likelihood_ensemble_size)
        logger.info(
            "FlowDAS zeta=%s J=%d max_grad_norm=%s (scenario=%s, M=%s)",
            zeta, n_mc, max_grad_norm, scenario_key, num_steps,
        )
        likelihood = FlowdasGaussianLikelihood(
            model=model,
            obs_operator=obs_operator,
            variance=variance,
            num_mc_samples=n_mc,
            guidance_scale=zeta,
            max_grad_norm=max_grad_norm,
        )
        posterior = StochasticInterpolantPosterior(
            model=model, likelihood_model=likelihood, diffusion_term=None
        )
        return model, posterior, stepper

    if method_name == "Guided FM":
        # FIG / Guided-FM baseline (yan_fig_2024): the legacy one-step guidance
        # likelihood (GuidanceGaussianLikelihood) routed through the FM
        # probability-flow ODE (FM-ODE), reusing the trained FM prior. The config
        # targets FlowMatchingPosterior with stepper=ode (g=0, deterministic).
        model = prior.fm_model
        likelihood = GuidanceGaussianLikelihood(
            model=model,
            obs_operator=obs_operator,
            variance=variance,
            ensemble_size=likelihood_ensemble_size,
        )
        posterior = FlowMatchingPosterior(
            model=model, likelihood_model=likelihood, diffusion_term=None
        )
        return model, posterior, stepper

    if method_name == "Guided FM (FIG)":
        # FIG baseline (yan_fig_2025): the faithful measurement-interpolant
        # corrector (FIGGaussianLikelihood) routed through the FM probability-flow
        # ODE (FM-ODE), reusing the trained FM prior. The corrector pulls each
        # post-flow state toward y_t = t_next * y by k gradient-descent steps with
        # scale c * (1 - t) / t (paper Algorithm 1). Config targets
        # FlowMatchingPosterior with stepper=ode (g=0, deterministic).
        model = prior.fm_model
        lik_cfg = method_cfg.get("likelihood_model", {})
        likelihood = FIGGaussianLikelihood(
            model=model,
            obs_operator=obs_operator,
            variance=variance,
            ensemble_size=likelihood_ensemble_size,
            guidance_steps=int(lik_cfg.get("guidance_steps", 2)),
            guidance_scale=float(lik_cfg.get("guidance_scale", 10.0)),
            interpolant_noise=float(lik_cfg.get("interpolant_noise", 0.0)),
        )
        posterior = FlowMatchingPosterior(
            model=model, likelihood_model=likelihood, diffusion_term=None
        )
        return model, posterior, stepper

    if method_name == "Guided FM (OT-ODE)":
        # OT-ODE baseline (pokle_training-free_2024): "Training-free Linear Image
        # Inverses via Flows" (Pokle, Muckley, Chen & Karrer, TMLR 2024). The
        # GuidanceGaussianLikelihood in weighting='ot_ode' mode -- covariance-
        # preconditioned guidance g = (d xhat_1/d z_t)^T H^T
        # (r_t^2 H H^T + sigma_y^2 I)^{-1} (y - H xhat_1) with r_t^2 = (1-t)^2 /
        # ((1-t)^2 + t^2) (paper Eq. 16 / Alg. 1, mapped to our forward time).
        # Reuses the trained FM prior; routed through FlowMatchingPosterior with
        # stepper=ode (g=0, deterministic).
        model = prior.fm_model
        lik_cfg = method_cfg.get("likelihood_model", {})
        obs_var = lik_cfg.get("obs_variance", None)
        likelihood = GuidanceGaussianLikelihood(
            model=model,
            obs_operator=obs_operator,
            variance=variance,
            ensemble_size=likelihood_ensemble_size,
            weighting="ot_ode",
            obs_variance=float(obs_var) if obs_var is not None else None,
            guidance_scale=float(lik_cfg.get("guidance_scale", 1.0)),
        )
        posterior = FlowMatchingPosterior(
            model=model, likelihood_model=likelihood, diffusion_term=None
        )
        return model, posterior, stepper

    if method_name == "Guided diffusion":
        # DPS baseline (chung_diffusion_2023): faithful DPS likelihood (Tweedie
        # denoiser xhat_1 with autograd through the network, raw R = sigma^2 I)
        # on a diffusion prior (DenoiseDiffusionModel), routed through the
        # purpose-built reverse-SDE DiffusionPosterior (Gaussian init a0=0). The
        # prior is built FROM the FM model by default (case.checkpoints.
        # diffusion_from_fm=true) since the dedicated diffusion checkpoint is
        # poorly trained; set diffusion_from_fm=false to use checkpoints.
        # diffusion_run instead. Either way model.score / .drift behave as a
        # diffusion prior, so DPS is unchanged.
        from scisi.posterior_models.diffusion_posterior import DiffusionPosterior

        model = prior.diffusion_model
        if model is None:
            raise ValueError(
                "Guided diffusion requires a diffusion prior, but none was loaded. "
                "Either keep case.checkpoints.diffusion_from_fm=true (build it from "
                "the FM model) or set case.checkpoints.diffusion_run to a checkpoint "
                "under checkpoints/<project>/."
            )
        # model_class='fm' -> Tweedie denoiser from model.score with anchor a0=0,
        # which is exactly the diffusion model's parametrisation (source ~ N(0,I)).
        likelihood = DPSGaussianLikelihood(
            model=model,
            obs_operator=obs_operator,
            variance=variance,
            ensemble_size=likelihood_ensemble_size,
            model_class="fm",
        )
        posterior = DiffusionPosterior(
            model=model, likelihood_model=likelihood, diffusion_term=None
        )
        return model, posterior, stepper

    if method_name == "SDA":
        # SDA baseline (rozet_score-based_2023), single-window adaptation: SDA's
        # DPS-style guidance with the Gamma_tau = R + gamma_sda (sigma/mu)^2 H H^T
        # covariance (Rozet & Louppe Eq. 15; see scisi.likelihood_models.sda). SDA
        # is a score/diffusion method, so -- like Guided diffusion -- it runs on
        # the diffusion prior (built from the FM model by default,
        # case.checkpoints.diffusion_from_fm) through the reverse-SDE
        # DiffusionPosterior (Gaussian init a0=0). model_class='fm' -> Tweedie
        # denoiser from model.score with anchor a0=0; the likelihood score is
        # injected with the SDA weight g_tau^2. NOTE: this is the single-window
        # guidance, not SDA's full all-at-once trajectory score (documented in the
        # SDALikelihood module docstring).
        import os

        from scisi.posterior_models.diffusion_posterior import DiffusionPosterior

        model = prior.diffusion_model
        if model is None:
            raise ValueError(
                "SDA requires a diffusion prior, but none was loaded. Either keep "
                "case.checkpoints.diffusion_from_fm=true (build it from the FM "
                "model) or set case.checkpoints.diffusion_run to a checkpoint "
                "under checkpoints/<project>/."
            )
        lik_cfg = method_cfg.get("likelihood_model", {}) or {}
        # Resolve any per-(scenario, M) scheduled hyperparameters in the config
        # block to scalars (scalars pass through unchanged). Env SDA_GAMMA still
        # takes top precedence so a tuning sweep needs no edit to the tracked YAML.
        hp = resolve_scheduled_hparams(lik_cfg, scenario_key, num_steps)
        _g = hp.get("gamma_sda", 1e-2)
        gamma_sda = float(os.environ.get("SDA_GAMMA", _g if _g is not None else 1e-2))
        logger.info(
            "SDA gamma_sda=%s (scenario=%s, M=%s)", gamma_sda, scenario_key, num_steps
        )
        likelihood = SDALikelihood(
            model=model,
            obs_operator=obs_operator,
            variance=variance,
            ensemble_size=likelihood_ensemble_size,
            model_class="fm",
            gamma_sda=gamma_sda,
        )
        posterior = DiffusionPosterior(
            model=model, likelihood_model=likelihood, diffusion_term=None
        )
        return model, posterior, stepper

    if method_name == "D-Flow SGLD":
        # NEW baseline (flow matching + ODE): D-Flow (ben-hamu_d-flow_2024)
        # optimise-the-source-latent guidance, sampled with SGLD. Differentiates
        # the data cost through the WHOLE FM-ODE flow Phi(z_0) and runs SGLD over
        # the source latent z_0, targeting p(z_0 | y) propto N(y; H Phi(z_0), R)
        # N(z_0; 0, I). Because it backprops through the entire rollout it does
        # NOT fit the per-step _one_step + likelihood pattern, so it uses a
        # dedicated DFlowPosterior that overrides sample() with a differentiable
        # rollout; the likelihood is a thin holder for H and R. Reuses the trained
        # FM prior; forward map is the deterministic FM-ODE (stepper=ode, g=0).
        from scisi.likelihood_models.dflow import DFlowSGLDLikelihood
        from scisi.posterior_models.dflow_posterior import DFlowPosterior

        import os

        model = prior.fm_model
        lik_cfg = method_cfg.get("likelihood_model", {}) or {}
        # Resolve any per-(scenario, M) scheduled hyperparameters in the config
        # block to scalars (scalars pass through unchanged); env vars still take
        # top precedence so a tuning sweep needs no edit to the tracked YAML.
        # Algorithm 1 / Table D.3 hyperparameters (arXiv:2602.21469).
        hp = resolve_scheduled_hparams(lik_cfg, scenario_key, num_steps)

        def _env_float(name: str, val: float) -> float:
            v = os.environ.get(name)
            return float(v) if v not in (None, "", "null", "None") else float(val)

        def _env_int(name: str, val: int) -> int:
            v = os.environ.get(name)
            return int(float(v)) if v not in (None, "", "null", "None") else int(val)

        # PECULIAR TO D-FLOW SGLD: the global SDE/ODE step count `num_steps` (M)
        # drives the SGLD chain length `num_optim_steps` for THIS method only. Its
        # differentiable FM-ODE rollout is a FIXED `ode_steps`=6 midpoint
        # integration (Table D.3), so the global M is otherwise inert for D-Flow;
        # mapping M -> num_optim_steps makes the {50,100,250,500} sweep the
        # Langevin-steps axis. Env DFLOW_NSTEPS still overrides for tuning; the
        # config `num_optim_steps` table is the fallback only when M is unset.
        _dflow_m = int(num_steps) if num_steps is not None else hp.get("num_optim_steps", 300)
        num_optim_steps = _env_int("DFLOW_NSTEPS", _dflow_m)
        step_size = _env_float("DFLOW_STEP_SIZE", hp.get("step_size", 0.05))
        noise_scale = _env_float("DFLOW_NOISE_SCALE", hp.get("noise_scale", 1e-3))
        lambda_reg = _env_float("DFLOW_LAMBDA", hp.get("lambda_reg", 5e-6))
        precond_decay = float(hp.get("precond_decay", 0.99))        # omega
        precond_eps = float(hp.get("precond_eps", 1e-3))            # delta
        ode_steps = int(hp.get("ode_steps", 6))                     # midpoint steps
        burn = int(hp.get("burn", 100))
        logger.info(
            "D-Flow SGLD Nsteps=%s eta=%s s=%s lambda=%s omega=%s delta=%s "
            "ode_steps=%s (scenario=%s, M=%s)",
            num_optim_steps, step_size, noise_scale, lambda_reg,
            precond_decay, precond_eps, ode_steps, scenario_key, num_steps,
        )
        likelihood = DFlowSGLDLikelihood(
            model=model,
            obs_operator=obs_operator,
            variance=variance,
            ensemble_size=likelihood_ensemble_size,
            num_optim_steps=num_optim_steps,
            step_size=step_size,
            noise_scale=noise_scale,
            lambda_reg=lambda_reg,
        )
        posterior = DFlowPosterior(
            model=model,
            likelihood_model=likelihood,
            diffusion_term=None,
            num_optim_steps=num_optim_steps,
            step_size=step_size,
            noise_scale=noise_scale,
            lambda_reg=lambda_reg,
            precond_decay=precond_decay,
            precond_eps=precond_eps,
            ode_steps=ode_steps,
            burn=burn,
            variance=variance,
        )
        return model, posterior, stepper

    if method_name == "SURGE":
        # SURGE baseline (wei_surge_2026, arXiv:2605.18745), single-window: a
        # guided reverse-diffusion SDE proposal (DPS Gaussian observation
        # guidance) with Girsanov-corrected SMC reweighting + ESS-based
        # resampling across particles. Runs on the diffusion-model prior (built
        # from the FM model by default, case.checkpoints.diffusion_from_fm), like
        # SDA. Unlike the likelihood-only baselines it owns the full sampler
        # (needs the injected SDE noise + cross-particle resampling), so it is a
        # dedicated posterior (SurgePosterior), not a DiffusionPosterior+likelihood.
        from scisi.posterior_models.surge_posterior import SurgePosterior

        model = prior.diffusion_model
        if model is None:
            raise ValueError(
                "SURGE requires a diffusion prior, but none was loaded. Either "
                "keep case.checkpoints.diffusion_from_fm=true (build it from the "
                "FM model) or set case.checkpoints.diffusion_run to a checkpoint "
                "under checkpoints/<project>/."
            )
        surge_cfg = method_cfg.get("posterior_model", {}) or {}
        # Resolve any per-(scenario, M) table on the SURGE knobs (guidance_scale)
        # to a scalar; scalars pass through unchanged.
        surge_hp = resolve_scheduled_hparams(surge_cfg, scenario_key, num_steps)
        _gs = surge_hp.get("guidance_scale", 1.0)
        posterior = SurgePosterior(
            model=model,
            obs_operator=obs_operator,
            variance=variance,
            guidance_scale=float(_gs if _gs is not None else 1.0),
            ess_threshold=float(surge_hp.get("ess_threshold", 0.5)),
            diffusion_term=None,
        )
        return model, posterior, stepper

    if method_name == "SURGE (SDA)":
        # SURGE + SDA (wei_surge_2026, combined with rozet_score-based_2023): the
        # SURGE SMC particle filter with SDA's likelihood as the guidance G. SDA
        # is a score/diffusion method, so it runs on the FM-derived diffusion
        # prior (like the standalone SURGE / SDA baselines) through the FM SURGE
        # variant (source N(0, I), anchor a0 = 0). SDALikelihood(model_class='fm')
        # supplies grad_x G; SURGE owns the guided-SDE proposal + Girsanov SMC
        # reweighting and computes the true-likelihood reward itself.
        from scisi.posterior_models.surge_posterior import SurgeFlowMatchingPosterior

        model = prior.diffusion_model
        if model is None:
            raise ValueError(
                "SURGE (SDA) requires a diffusion prior, but none was loaded. "
                "Either keep case.checkpoints.diffusion_from_fm=true (build it "
                "from the FM model) or set case.checkpoints.diffusion_run to a "
                "checkpoint under checkpoints/<project>/."
            )
        import os

        lik_cfg = method_cfg.get("likelihood_model", {}) or {}
        surge_cfg = method_cfg.get("posterior_model", {}) or {}
        # SDA hyperparameters: use the standalone SDA config's settings
        # (configs/method/sda.yaml). Resolve any per-(scenario, M) table to a
        # scalar; env SDA_GAMMA overrides (top precedence), matching the SDA branch.
        hp = resolve_scheduled_hparams(lik_cfg, scenario_key, num_steps)
        _g = hp.get("gamma_sda", 1e-2)
        gamma_sda = float(os.environ.get("SDA_GAMMA", _g if _g is not None else 1e-2))
        logger.info(
            "SURGE (SDA) gamma_sda=%s (scenario=%s, M=%s); guidance weight = g^2",
            gamma_sda, scenario_key, num_steps,
        )
        likelihood = SDALikelihood(
            model=model,
            obs_operator=obs_operator,
            variance=variance,
            ensemble_size=likelihood_ensemble_size,
            model_class="fm",
            gamma_sda=gamma_sda,
        )
        # SURGE owns only the SMC layer; SDA's guidance enters with its OWN weight
        # g^2 (SurgePosterior._injection_weight reads guidance_weight=='g_squared'),
        # so there is no SURGE guidance_scale here -- only ess_threshold.
        surge_hp = resolve_scheduled_hparams(surge_cfg, scenario_key, num_steps)
        posterior = SurgeFlowMatchingPosterior(
            model=model,
            obs_operator=obs_operator,
            variance=variance,
            likelihood_model=likelihood,
            ess_threshold=float(surge_hp.get("ess_threshold", 0.5)),
            diffusion_term=None,
        )
        return model, posterior, stepper

    if method_name == "SURGE (FlowDAS)":
        # SURGE + FlowDAS (wei_surge_2026, combined with chen_flowdas_2025): the
        # SURGE SMC particle filter with FlowDAS's Monte-Carlo likelihood as the
        # guidance G. FlowDAS is SI-based (uses the SI drift + predictor spread),
        # so it runs on the SI prior through the SI SURGE variant (point-mass
        # delta_{x0} init). FlowdasGaussianLikelihood returns the RAW guidance
        # score. FlowDAS's guidance strength zeta is a LIKELIHOOD knob, applied by
        # SURGE via the likelihood's sde_weight (the combo proposal b + zeta*s_lik
        # IS FlowDAS's guided SDE, plus the SMC layer); max_grad_norm caps NS-scale
        # blow-ups. SURGE owns the guided-SDE proposal + Girsanov SMC and reward.
        from scisi.posterior_models.surge_posterior import (
            SurgeStochasticInterpolantPosterior,
        )

        import os

        model = prior.si_model
        lik_cfg = method_cfg.get("likelihood_model", {}) or {}
        surge_cfg = method_cfg.get("posterior_model", {}) or {}
        # FlowDAS hyperparameters: use the standalone FlowDAS config's settings
        # (configs/method/flowdas.yaml). guidance_scale (zeta) is a LIKELIHOOD knob;
        # resolve any per-(scenario, M) table to a scalar; env FLOWDAS_ZETA /
        # FLOWDAS_MAX_GRAD_NORM override, matching the FlowDAS branch.
        hp = resolve_scheduled_hparams(lik_cfg, scenario_key, num_steps)
        _z = hp.get("guidance_scale", 1.0)
        zeta = float(os.environ.get("FLOWDAS_ZETA", _z if _z is not None else 1.0))
        _mgn = os.environ.get("FLOWDAS_MAX_GRAD_NORM", hp.get("max_grad_norm", None))
        max_grad_norm = None if _mgn in (None, "", "null", "None") else float(_mgn)
        logger.info(
            "SURGE (FlowDAS) zeta=%s max_grad_norm=%s (scenario=%s, M=%s)",
            zeta, max_grad_norm, scenario_key, num_steps,
        )
        likelihood = FlowdasGaussianLikelihood(
            model=model,
            obs_operator=obs_operator,
            variance=variance,
            # J decoupled from E (paper Eq. 10); legacy fallback as above.
            num_mc_samples=int(hp.get("num_mc_samples") or likelihood_ensemble_size),
            guidance_scale=zeta,  # FlowDAS zeta; SURGE applies it via sde_weight.
            max_grad_norm=max_grad_norm,
        )
        # SURGE owns only the SMC layer -> no guidance_scale, only ess_threshold.
        surge_hp = resolve_scheduled_hparams(surge_cfg, scenario_key, num_steps)
        posterior = SurgeStochasticInterpolantPosterior(
            model=model,
            obs_operator=obs_operator,
            variance=variance,
            likelihood_model=likelihood,
            ess_threshold=float(surge_hp.get("ess_threshold", 0.5)),
            diffusion_term=None,
        )
        return model, posterior, stepper

    is_si = method_name == "Ours (SI-SDE)"
    model = prior.si_model if is_si else prior.fm_model
    mode = likelihood_mode or method_cfg.likelihood_model.get(
        "likelihood_mode", "inflated"
    )
    model_class = "si" if is_si else "fm"

    likelihood = InterpolantGaussianLikelihood(
        model=model,
        obs_operator=obs_operator,
        variance=variance,
        ensemble_size=likelihood_ensemble_size,
        likelihood_mode=mode,
        model_class=model_class,
    )

    if is_si:
        posterior = StochasticInterpolantPosterior(
            model=model, likelihood_model=likelihood, diffusion_term=None
        )
    else:
        # FM-ODE: diffusion_term=None -> g=0. DM-SDE: endpoint-vanishing schedule.
        if method_cfg.stepper == "ode":
            diffusion_term = None
        else:
            scale = g_multiplier if g_multiplier is not None else float(
                method_cfg.diffusion_term.get("multiplier", 1.0)
            )
            diffusion_term = endpoint_vanishing_diffusion(model.interpolation, scale)
        posterior = FlowMatchingPosterior(
            model=model, likelihood_model=likelihood, diffusion_term=diffusion_term
        )

    return model, posterior, stepper


# --------------------------------------------------------------------------- #
# One assimilation run.
# --------------------------------------------------------------------------- #


@dataclass
class AssimResult:
    """Output of one autoregressive assimilation."""

    posterior_trajectory: torch.Tensor  # [E, C, H, W, T] (normalised)
    true_trajectory: torch.Tensor  # [1, C, H, W, T] (normalised)
    nfe_per_step: float
    seconds_per_step: float


def run_assimilation(
    posterior: torch.nn.Module,
    model: torch.nn.Module,
    truth_obs: TruthAndObs,
    ensemble_size: int,
    num_steps: int,
    num_physical_steps: int,
    stepper: Callable,
    nfe_counter: NFECounter,
    gaussian_base: bool,
) -> AssimResult:
    """Run one autoregressive assimilation and record NFE + wall-clock.

    Feeds each posterior sample back as ``x^{n-1}`` via the base posterior's
    ``sample_trajectory`` (it already threads the field history). NFE is the
    counter delta divided by the number of assimilation steps; seconds is the
    wall-clock divided by the same.
    """
    L = truth_obs.field_history.shape[-1]
    n_assim = num_physical_steps - L

    common_input = {
        "base": None if gaussian_base else truth_obs.init_base,
        "field_history": truth_obs.field_history,
        "field_cond": truth_obs.field_cond,
        "pars_cond": truth_obs.pars_cond,
        "stepper": stepper,
        "num_physical_steps": num_physical_steps,
        "observations": truth_obs.observations[:, :, L:],
    }

    nfe_counter.reset()
    timer = StepTimer()
    with timer:
        posterior_trajectory = posterior.sample_trajectory(
            **common_input,
            ensemble_size=ensemble_size,
            num_steps=num_steps,
        )

    nfe_per_step = nfe_counter.count / max(n_assim, 1)
    seconds_per_step = timer.elapsed / max(n_assim, 1)

    return AssimResult(
        posterior_trajectory=posterior_trajectory,
        true_trajectory=truth_obs.true_trajectory,
        nfe_per_step=nfe_per_step,
        seconds_per_step=seconds_per_step,
    )


def build_reference_trajectory(
    prior: LoadedPrior,
    method_cfg: DictConfig,
    truth_obs: TruthAndObs,
    obs_operator: LinearObservationOperator,
    variance: float,
    likelihood_ensemble_size: int,
    reference_ensemble_size: int,
    num_steps: int,
    num_physical_steps: int,
    likelihood_mode: Optional[str] = None,
) -> torch.Tensor:
    """Draw ONE large-E reference posterior ensemble for KL-at-points.

    Uses the designated headline sampler (SI-SDE) on the SAME truth + obs + mask
    as the per-method runs (spec Section 9), at ``reference_ensemble_size``. The
    returned ``[E_ref, C, H, W, T]`` trajectory supplies the reference marginals
    every method's KL is measured against. Cached once per (scenario, test, seed)
    by the caller so it is not redrawn per method.

    ``likelihood_mode`` MUST match the run's mode (e.g. ``dps_jacobian_free`` for
    the CPU smoke) -- otherwise the reference silently falls back to the config
    default (``inflated``), whose per-column full-Sigma_s JVPs make the reference
    draw orders of magnitude slower than the per-method runs.
    """
    model, posterior, stepper = build_posterior(
        method_name="Ours (SI-SDE)",
        method_cfg=method_cfg,
        prior=prior,
        obs_operator=obs_operator,
        variance=variance,
        likelihood_ensemble_size=likelihood_ensemble_size,
        likelihood_mode=likelihood_mode,
    )
    nfe = attach_nfe_counter(model)
    result = run_assimilation(
        posterior=posterior,
        model=model,
        truth_obs=truth_obs,
        ensemble_size=reference_ensemble_size,
        num_steps=num_steps,
        num_physical_steps=num_physical_steps,
        stepper=stepper,
        nfe_counter=nfe,
        gaussian_base=False,  # SI-SDE point-mass init
    )
    return result.posterior_trajectory


# --------------------------------------------------------------------------- #
# Metrics on the posterior ensemble (spec Section 3).
# --------------------------------------------------------------------------- #


def compute_metrics(
    result: AssimResult,
    obs_operator: LinearObservationOperator,
    len_field_history: int,
    kl_num_points: int = 16,
    reference_trajectory: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Compute the full NS metric set, averaged over assimilation steps.

    Returns a dict keyed by the canonical metric strings: ``rmse``,
    ``energy_spec_rmse``, ``crps``, ``spread_skill`` (reported as
    ``|1-ratio|``), ``kl_points``, plus ``nfe`` / ``seconds``.

    * RMSE: ensemble-mean RMSE (spec 3a / fix m1).
    * energy_spec_rmse: log radially-averaged KE spectrum RMSE of xbar vs truth.
    * CRPS: unbiased pairwise estimator (spec 3b).
    * spread-skill: with sqrt((E+1)/E) correction, reported as |1-ratio|.
      Requires E >= 2; returns NaN for a degenerate single-member ensemble.
    * KL-at-points: 1-D marginal KL(posterior || reference) at observed +
      unobserved points. ``reference_trajectory`` is a large-E reference ensemble
      ``[E_ref, C, H, W, T]`` drawn from the SAME truth+obs+mask by the headline
      sampler (spec Section 9); when ``None`` the posterior ensemble is its own
      reference (degenerate ~0, used only when no reference is supplied).
    """
    post = result.posterior_trajectory  # [E, C, H, W, T]
    true = result.true_trajectory  # [1, C, H, W, T]
    T = post.shape[-1]
    E = post.shape[0]
    C, H, W = post.shape[1], post.shape[2], post.shape[3]

    # Observed-point mask on the flat grid (for the observed/unobserved split of
    # both KL-at-points and CRPS). For sparse (selection) operators the observed
    # set is the sensor locations; for the block-average super-res operator every
    # point feeds an observation, so the unobserved set is empty and the
    # unobserved CRPS is NaN (no unobserved points to score).
    obs_mask_grid = obs_operator.obs_indices_on_grid.reshape(-1).to(torch.bool)
    obs_mask_chw = obs_mask_grid.reshape(C, H, W)
    unobs_mask_chw = ~obs_mask_chw
    has_obs = bool(obs_mask_chw.any())
    has_unobs = bool(unobs_mask_chw.any())

    rmse_steps: list[float] = []
    espec_steps: list[float] = []
    crps_steps: list[float] = []
    crps_obs_steps: list[float] = []
    crps_unobs_steps: list[float] = []
    ss_steps: list[float] = []
    kl_steps: list[float] = []

    # Pick a fixed set of grid points: half observed, half unobserved if possible.
    flat_n = C * H * W
    obs_idx = torch.nonzero(obs_mask_grid).reshape(-1)
    unobs_idx = torch.nonzero(~obs_mask_grid).reshape(-1)
    half = max(kl_num_points // 2, 1)

    def _pick(idx: torch.Tensor, k: int) -> torch.Tensor:
        if idx.numel() == 0:
            return idx
        sel = torch.linspace(0, idx.numel() - 1, min(k, idx.numel())).long()
        return idx[sel]

    point_idx = torch.cat([_pick(obs_idx, half), _pick(unobs_idx, half)])
    point_is_obs = torch.cat(
        [
            torch.ones(_pick(obs_idx, half).numel(), dtype=torch.bool),
            torch.zeros(_pick(unobs_idx, half).numel(), dtype=torch.bool),
        ]
    )

    # Only score the assimilated steps (skip the seeded history prefix).
    for t in range(len_field_history, T):
        ens_t = post[..., t].reshape(E, C, H, W)  # [E, C, H, W]
        true_t = true[0, ..., t]  # [C, H, W]

        rmse_steps.append(float(ensemble_mean_rmse(ens_t, true_t)))
        # Spectrum on the single (C=1) vorticity channel.
        xbar = ens_t.mean(dim=0)  # [C, H, W]
        espec_steps.append(
            float(energy_spectrum_rmse(xbar[0], true_t[0]))
        )
        crps_steps.append(float(crps(ens_t, true_t)))
        # CRPS split into observed vs unobserved grid points (vs the truth).
        if has_obs:
            crps_obs_steps.append(float(crps(ens_t, true_t, mask=obs_mask_chw)))
        if has_unobs:
            crps_unobs_steps.append(float(crps(ens_t, true_t, mask=unobs_mask_chw)))
        # spread-skill is undefined for a single-member ensemble (needs E >= 2).
        if E >= 2:
            ss = spread_skill(ens_t, true_t)
            ss_steps.append(float(ss["deviation"]))

        # KL at points: sampled marginals [E, P] vs the large-E reference
        # marginals [E_ref, P] (separate reference ensemble; spec Section 9).
        ens_flat = ens_t.reshape(E, -1)[:, point_idx]  # [E, P]
        # Guard the time index against the reference's own length: an E=1000 EnKF
        # reference generated at a SMALLER num_physical_steps than the current run
        # (e.g. the np=15 refs reused under an np=20 grid) is shorter than the
        # posterior rollout, so steps past its end have no reference -> KL is NaN
        # for those steps rather than an out-of-bounds crash.
        if reference_trajectory is not None and t < reference_trajectory.shape[-1]:
            ref_t = reference_trajectory[..., t].reshape(
                reference_trajectory.shape[0], -1
            )[:, point_idx]  # [E_ref, P]
            kl = kl_at_points(
                sampled=ens_flat,
                reference=ref_t,
                observed_mask=point_is_obs,
                method="gaussian",
            )
            kl_steps.append(float(kl["mean"]))
        else:
            # No reference posterior supplied (or reference shorter than the
            # rollout) -> KL is undefined (NaN), NOT a degenerate self-reference
            # (which would falsely read as ~0).
            kl_steps.append(float("nan"))

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    return {
        "rmse": _mean(rmse_steps),
        "energy_spec_rmse": _mean(espec_steps),
        "crps": _mean(crps_steps),
        "crps_observed": _mean(crps_obs_steps),
        "crps_unobserved": _mean(crps_unobs_steps),
        "spread_skill": _mean(ss_steps),
        "kl_points": _mean(kl_steps),
        "nfe": result.nfe_per_step,
        "seconds": result.seconds_per_step,
        # Per-(assimilation-)step metric curves (one value per scored time step,
        # i.e. t = len_field_history .. T-1). Consumed by ``_save_states`` so the
        # metrics-vs-time plots can be made without recomputing. NOT a CSV row
        # (``_metric_rows`` reads only the scalar Metric keys above).
        "per_step": {
            "rmse": list(rmse_steps),
            "energy_spec_rmse": list(espec_steps),
            "crps": list(crps_steps),
            "crps_observed": list(crps_obs_steps),
            "crps_unobserved": list(crps_unobs_steps),
            "spread_skill": list(ss_steps),
            "kl_points": list(kl_steps),
        },
    }


__all__ = [
    "LoadedPrior",
    "load_prior",
    "attach_nfe_counter",
    "build_obs_operator",
    "prepare_truth_and_obs",
    "build_posterior",
    "resolve_scheduled_hparam",
    "resolve_scheduled_hparams",
    "run_assimilation",
    "build_reference_trajectory",
    "compute_metrics",
    "AssimResult",
    "TruthAndObs",
]
