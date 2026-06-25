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
from scisi.likelihood_models.observation_operators import LinearObservationOperator
from scisi.metrics.accuracy import ensemble_mean_rmse
from scisi.metrics.calibration import crps, rank_histogram, spread_skill
from scisi.metrics.cost import NFECounter, StepTimer
from scisi.metrics.distributional import kl_at_points
from scisi.metrics.spectral import energy_spectrum_rmse
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
    fm_model: torch.nn.Module  # FlowMatchingModel (FM-SDE / FM-ODE)
    preprocesser: Any
    test_dataset: Any
    len_field_history: int
    train_cfg: DictConfig
    checkpoint_name: str
    has_trained_weights: bool


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
    legacy_key = "Ours (SI-SDE)" if role == "si" else "Ours (FM-SDE)"
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
    )


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
    net = model.drift_model
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
# Sampler construction (SI-SDE / FM-SDE / FM-ODE / FlowDAS).
# --------------------------------------------------------------------------- #


def build_posterior(
    method_name: str,
    method_cfg: DictConfig,
    prior: LoadedPrior,
    obs_operator: LinearObservationOperator,
    variance: float,
    likelihood_ensemble_size: int,
    likelihood_mode: Optional[str] = None,
    g_multiplier: Optional[float] = None,
) -> tuple[torch.nn.Module, torch.nn.Module, Callable]:
    """Instantiate (likelihood, posterior, stepper) for one method.

    Returns the trained model used, the posterior sampler, and the SDE/ODE
    stepper. ``likelihood_mode`` / ``g_multiplier`` override the config (used by
    the ablation sweep).
    """
    stepper = _STEPPERS[method_cfg.stepper]

    if method_name == "FlowDAS":
        model = prior.si_model
        likelihood = FlowdasGaussianLikelihood(
            model=model,
            obs_operator=obs_operator,
            variance=variance,
            ensemble_size=likelihood_ensemble_size,
        )
        posterior = StochasticInterpolantPosterior(
            model=model, likelihood_model=likelihood, diffusion_term=None
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
        # FM-ODE: diffusion_term=None -> g=0. FM-SDE: endpoint-vanishing schedule.
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

    # Observed-point mask on the flat grid (for KL observed/unobserved split).
    obs_mask_grid = obs_operator.obs_indices_on_grid.reshape(-1).to(torch.bool)

    rmse_steps: list[float] = []
    espec_steps: list[float] = []
    crps_steps: list[float] = []
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
        # spread-skill is undefined for a single-member ensemble (needs E >= 2).
        if E >= 2:
            ss = spread_skill(ens_t, true_t)
            ss_steps.append(float(ss["deviation"]))

        # KL at points: sampled marginals [E, P] vs the large-E reference
        # marginals [E_ref, P] (separate reference ensemble; spec Section 9).
        ens_flat = ens_t.reshape(E, -1)[:, point_idx]  # [E, P]
        if reference_trajectory is not None:
            ref_t = reference_trajectory[..., t].reshape(
                reference_trajectory.shape[0], -1
            )[:, point_idx]  # [E_ref, P]
        else:
            ref_t = ens_flat  # degenerate self-reference (no reference supplied)
        kl = kl_at_points(
            sampled=ens_flat,
            reference=ref_t,
            observed_mask=point_is_obs,
            method="gaussian",
        )
        kl_steps.append(float(kl["mean"]))

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    return {
        "rmse": _mean(rmse_steps),
        "energy_spec_rmse": _mean(espec_steps),
        "crps": _mean(crps_steps),
        "spread_skill": _mean(ss_steps),
        "kl_points": _mean(kl_steps),
        "nfe": result.nfe_per_step,
        "seconds": result.seconds_per_step,
    }


__all__ = [
    "LoadedPrior",
    "load_prior",
    "attach_nfe_counter",
    "build_obs_operator",
    "prepare_truth_and_obs",
    "build_posterior",
    "run_assimilation",
    "build_reference_trajectory",
    "compute_metrics",
    "AssimResult",
    "TruthAndObs",
]
