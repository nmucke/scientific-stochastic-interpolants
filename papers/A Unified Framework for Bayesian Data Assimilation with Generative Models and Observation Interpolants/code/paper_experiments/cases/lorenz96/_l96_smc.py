"""SMC cells of the Lorenz 96 case: SURGE around Ours, SURGE + FlowDAS / SDA, and ESS logs.

Cells (``driver.CELLS`` takes :data:`SURGE_CELLS` as they are):

* ``surge_si_sde`` / ``surge_dm_sde``: Ours (SI-SDE / DM-SDE) inside the SURGE SMC
  layer, built as the multistep analytical case builds them
  (``cases/analytical/multistep.py``, ``MultistepRunner._ours_cycle`` with
  ``surge=True``): the plain cell's likelihood is the guidance of
  ``SurgeStochasticInterpolantPosterior`` (SI) or of ``OursSurgeDiffusionPosterior``
  on the ``FullFirstStepDiffusionModel`` lift of the FM prior (DM). SDE only: an
  ODE has no path measure for SURGE's Girsanov weight.
* ``surge_unguided_si`` / ``surge_unguided_dm``: the same SMC proposing from the
  unguided prior (``NullGuidance``), one control per wrapped prior, as in the
  multistep case (its variant ``surge_unguided`` of each Ours row).
* ``surge_flowdas`` / ``surge_sda``: the Navier--Stokes constructions, called
  through ``cases.navier_stokes._ns_pipeline.build_posterior`` with a method config
  whose every knob is an explicit scalar from this case's ``methods.<cell>`` block.

SURGE normalises its weights over every member it is given
(``SurgePosteriorBase._maybe_resample``), so each sampler call holds ONE sequence
(``E`` members); :func:`assimilate_one` and the recorder refuse anything else.

ESS logging (:class:`SMCRecorder`) wraps the posterior's ``_maybe_resample`` on
the instance and recomputes the ESS from the same weights with the library's own
formula; it draws no random numbers and changes no state, so outputs are bitwise
those of the unwrapped sampler. Per sequence and assimilation step:

* ``smc_ess_fraction``: ESS / N of the accumulated weights at the end of the step,
  before the final forced resample (1 when the last pseudo-step resampled);
* ``smc_ess_min_fraction``: its minimum over the step's pseudo-time;
* ``smc_resamples``: resampling events triggered by ESS / N < ``ess_threshold``
  during the step. The forced resample that ends every step is NOT counted.

The particle filters resample at every step whatever the ESS (systematic, before
the kick). For them ``smc_ess_fraction`` = ``smc_ess_min_fraction`` = the Kish ESS
/ N of the step's weights, and ``smc_resamples`` = 0: their one resample per step
is the step's unconditional (forced) one, which is not counted, as for SURGE.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable

import torch

from results_schema import Method

SMC_ESS = "smc_ess_fraction"
SMC_ESS_MIN = "smc_ess_min_fraction"
SMC_RESAMPLES = "smc_resamples"
SMC_KEYS = (SMC_ESS, SMC_ESS_MIN, SMC_RESAMPLES)

SURGE_VARIANT = "surge"
SURGE_UNGUIDED_VARIANT = "surge_unguided"  # the multistep case's label
# Ours inside SURGE: cell -> the plain cell whose likelihood is the guidance.
SURGE_OURS = {"surge_si_sde": "si_sde", "surge_dm_sde": "dm_sde"}
# The same SMC on the unguided prior, one per wrapped prior.
SURGE_UNGUIDED = {"surge_unguided_si": "si_sde", "surge_unguided_dm": "dm_sde"}
# The NS baselines: cell -> Method (its value is build_posterior's method_name).
SURGE_BASELINES = {"surge_flowdas": Method.SURGE_FLOWDAS, "surge_sda": Method.SURGE_SDA}
_PRIOR_METHOD = {"si_sde": Method.OURS_SI_SDE, "dm_sde": Method.OURS_DM_SDE}
# name -> (tidy method label, variant)
SURGE_CELLS = {
    **{c: (_PRIOR_METHOD[p], SURGE_VARIANT) for c, p in SURGE_OURS.items()},
    **{c: (_PRIOR_METHOD[p], SURGE_UNGUIDED_VARIANT) for c, p in SURGE_UNGUIDED.items()},
    **{c: (m, None) for c, m in SURGE_BASELINES.items()},
}
# Overrides of the NS method configs; the L96 block must set every knob explicitly.
NS_ENV_OVERRIDES = ("FLOWDAS_ZETA", "SDA_GAMMA", "FLOWDAS_MAX_GRAD_NORM", "FLOWDAS_MC_CHUNK")
# configs/method/<file> and which L96 key replaces which config knob.
BASELINE_KNOBS = {
    "surge_flowdas": (
        "surge_flowdas.yaml",
        {
            "likelihood_model": ("guidance_scale", "num_mc_samples", "max_grad_norm"),
            "posterior_model": ("ess_threshold",),
        },
    ),
    "surge_sda": (
        "surge_sda.yaml",
        {"likelihood_model": ("gamma_sda",), "posterior_model": ("ess_threshold",)},
    ),
}


# --------------------------------------------------------------------------- #
# ESS / resampling recorder
# --------------------------------------------------------------------------- #


def ess_fraction(normalised_weights: torch.Tensor) -> float:
    """``(1 / sum w_i^2) / N``, the formula of ``SurgePosteriorBase._maybe_resample``."""
    w = normalised_weights
    return float(1.0 / torch.clamp((w**2).sum(), min=1e-30)) / w.shape[0]


class SMCRecorder:
    """Records ESS and resampling of a SURGE posterior, one entry per ``sample`` call.

    Wraps ``posterior._maybe_resample`` on the instance. The ESS is recomputed from
    ``posterior._normalized_weights()`` exactly as the library does, before the
    library's call; the wrapped call itself is unchanged.
    """

    def __init__(self, posterior, members: int) -> None:
        self.posterior = posterior
        self.members = int(members)
        self.steps: list[dict[str, float]] = []
        self._values: list[float] = []
        self._resamples = 0
        original = posterior._maybe_resample

        def recorded(base, field_history, force=False):
            if base.shape[0] != self.members:
                raise ValueError(
                    f"SURGE got {base.shape[0]} members, expected {self.members}: its "
                    "weights are normalised over every member passed in, so a call "
                    "must hold exactly one sequence."
                )
            self._observe(base, force)
            return original(base, field_history, force=force)

        posterior._maybe_resample = recorded

    def _observe(self, base: torch.Tensor, force: bool) -> None:
        post = self.posterior
        if post._log_w is None or base.shape[0] < 2:
            return  # the library returns without resampling
        members = base.shape[0]
        # The library's own computation, so the resampling decision is the same.
        w = post._normalized_weights().to(base.device)
        ess = float(1.0 / torch.clamp((w**2).sum(), min=1e-30))
        self._values.append(ess / members)
        if force:
            self.steps.append(
                {
                    SMC_ESS: ess / members,
                    SMC_ESS_MIN: min(self._values),
                    SMC_RESAMPLES: float(self._resamples),
                }
            )
            self._values, self._resamples = [], 0
        elif not ess >= post.ess_threshold * members:
            self._resamples += 1

    def take(self, steps: int) -> dict[str, list[float]]:
        """The per-step lists of the last sequence, then reset."""
        if len(self.steps) != steps:
            raise RuntimeError(f"recorded {len(self.steps)} SMC steps, expected {steps}")
        out = {key: [entry[key] for entry in self.steps] for key in SMC_KEYS}
        self.steps = []
        return out


def pf_smc_record(ess: torch.Tensor, num_particles: int) -> dict[str, list[float]]:
    """Per-step record of a particle filter that resamples at every step.

    ``ess``: ``[T]`` Kish ESS of each step's weights (before its resample).
    """
    frac = (ess.double() / float(num_particles)).tolist()
    return {SMC_ESS: frac, SMC_ESS_MIN: list(frac), SMC_RESAMPLES: [0.0] * len(frac)}


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SurgeOursSettings:
    """``methods.surge_{si,dm}_sde``: the plain sampler's knobs plus SURGE's threshold.

    There is no particles-per-member knob: SURGE's SMC runs one particle per
    member (N = E), and neither the multistep classes nor the library expose one.
    """

    num_steps: int = 50
    likelihood_mode: str = "inflated"
    jacobian_damping: float = 1.0
    diffusion_scale: float = 1.0  # DM-SDE only
    ess_threshold: float = 0.5


@dataclass(frozen=True)
class SurgeUnguidedSettings:
    num_steps: int = 50
    diffusion_scale: float = 1.0  # DM prior only
    ess_threshold: float = 0.5


def _settings(cls, name: str, settings: dict, prior: str | None = None):
    known = {f.name for f in fields(cls)}
    unknown = set(settings) - known
    if prior == "si_sde" and "diffusion_scale" in settings:
        unknown.add("diffusion_scale")  # the SI-SDE has no diffusion knob here
    if unknown:
        raise ValueError(f"{name}: unknown settings {sorted(unknown)}")
    return cls(**settings)


def baseline_method_config(name: str, settings: dict, config_dir: Path):
    """``configs/method/<x>.yaml`` with every knob replaced by the L96 scalar.

    Every knob of :data:`BASELINE_KNOBS` must be in ``settings`` (plus
    ``num_steps``); other keys are refused, as is any schedule table left in the
    likelihood or posterior block, so nothing falls back to an NS value.
    """
    from omegaconf import OmegaConf

    from cases.navier_stokes._ns_pipeline import _is_schedule_table

    filename, knobs = BASELINE_KNOBS[name]
    wanted = {"num_steps", *(k for keys in knobs.values() for k in keys)}
    unknown, missing = set(settings) - wanted, wanted - set(settings)
    if unknown or missing:
        raise ValueError(
            f"{name}: settings must be exactly {sorted(wanted)}; unknown {sorted(unknown)}, "
            f"missing {sorted(missing)}"
        )
    cfg = OmegaConf.load(config_dir / filename)
    for block, keys in knobs.items():
        for key in keys:
            value = settings[key]
            if isinstance(value, (dict, list)):
                raise ValueError(f"{name}.{key} must be a scalar, got {value!r}")
            cfg[block][key] = value
        leftover = set(cfg[block]) - set(keys) - {"_target_"}
        if leftover:
            raise ValueError(f"{filename} {block} has knobs this case does not set: {sorted(leftover)}")
        for key, value in cfg[block].items():
            if _is_schedule_table(value):
                raise ValueError(f"{name}.{block}.{key} is still a schedule table")
    return cfg


def refuse_ns_env() -> None:
    present = [key for key in NS_ENV_OVERRIDES if os.environ.get(key) not in (None, "")]
    if present:
        raise RuntimeError(
            f"{present} set: the NS build_posterior would override this case's settings."
        )


# --------------------------------------------------------------------------- #
# Posteriors
# --------------------------------------------------------------------------- #


def build_surge_ours(prior: str, model, problem, stats, settings, guided: bool = True):
    """``(posterior, stepper)``: Ours ``prior`` inside SURGE, as in the multistep case.

    The guidance is the plain cell's own likelihood (built by
    ``_l96_generative.build_posterior`` from the same settings), or ``NullGuidance``
    when not ``guided``. Operator and variance come from ``observation_model``.
    """
    from cases.analytical.linear_system import (
        FullFirstStepDiffusionModel,
        NullGuidance,
        OursSurgeDiffusionPosterior,
    )
    from cases.lorenz96._l96_generative import (
        SamplerSettings,
        build_posterior,
        observation_model,
    )
    from scisi.posterior_models.flow_matching_posterior import (
        endpoint_vanishing_diffusion,
    )
    from scisi.posterior_models.surge_posterior import (
        SurgeStochasticInterpolantPosterior,
    )

    observed = observation_model(problem, stats)
    if guided:
        sampler = SamplerSettings(
            num_steps=settings.num_steps,
            likelihood_mode=settings.likelihood_mode,
            jacobian_damping=settings.jacobian_damping,
            diffusion_scale=settings.diffusion_scale,
        )
        plain, stepper = build_posterior(_PRIOR_METHOD[prior], model, problem, stats, sampler)
        likelihood = plain.likelihood_model
    else:
        from scisi.sampling.sde_solvers import euler_maruyama_step

        likelihood, stepper = NullGuidance(), euler_maruyama_step
    ess = float(settings.ess_threshold)
    if prior == "si_sde":
        posterior = SurgeStochasticInterpolantPosterior(
            model=model,
            obs_operator=observed.operator,
            variance=observed.variance,
            likelihood_model=likelihood,
            ess_threshold=ess,
            diffusion_term=None,
        )
        return posterior, stepper
    # DM-SDE: SURGE's prior is the reverse SDE, so the model is the diffusion lift
    # of the FM prior; its first step keeps the velocity, as Ours' first step does.
    diffusion = endpoint_vanishing_diffusion(model.interpolation, settings.diffusion_scale)
    lifted = FullFirstStepDiffusionModel.from_flow_matching(model, diffusion_term=diffusion)
    posterior = OursSurgeDiffusionPosterior(
        model=lifted,
        obs_operator=observed.operator,
        variance=observed.variance,
        likelihood_model=likelihood,
        ess_threshold=ess,
        diffusion_term=diffusion,
    )
    return posterior, stepper


def build_surge_baseline(name, settings, si_model, fm_model, problem, stats, members, config_dir):
    """``(network model, posterior, stepper)`` from the NS ``build_posterior``."""
    from omegaconf import OmegaConf

    from cases.lorenz96._l96_generative import observation_model
    from cases.navier_stokes._ns_pipeline import LoadedPrior, build_posterior
    from scisi.models.diffusion_model import DenoiseDiffusionModel

    refuse_ns_env()
    method_cfg = baseline_method_config(name, settings, config_dir)
    diffusion = DenoiseDiffusionModel.from_flow_matching(fm_model)
    diffusion.eval().to(next(fm_model.parameters()).device)
    prior = LoadedPrior(
        si_model=si_model,
        fm_model=fm_model,
        preprocesser=None,
        test_dataset=None,
        len_field_history=1,
        train_cfg=OmegaConf.create({}),
        checkpoint_name="lorenz96",
        has_trained_weights=True,
        diffusion_model=diffusion,
    )
    observed = observation_model(problem, stats)
    return build_posterior(
        method_name=SURGE_BASELINES[name].value,
        method_cfg=method_cfg,
        prior=prior,
        obs_operator=observed.operator,
        variance=observed.variance,
        likelihood_ensemble_size=members,
        case_key="lorenz96",
        scenario_key="lorenz96",
        num_steps=int(settings["num_steps"]),
    )


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def assimilate_one(posterior, stepper, stats, initial, observations, num_steps, seed, obs_to_model):
    """``_l96_generative.assimilate`` for ONE sequence (``initial`` ``[1, E, d]``)."""
    from cases.lorenz96._l96_generative import assimilate

    if initial.shape[0] != 1 or observations.shape[0] != 1:
        raise ValueError(
            "SURGE resamples across every member of a call: run one sequence per "
            f"call, got {initial.shape[0]}."
        )
    return assimilate(
        posterior,
        stepper,
        stats,
        initial,
        observations,
        num_steps,
        seed,
        obs_to_model=obs_to_model,
    )


@dataclass
class SMCRun:
    ensembles: torch.Tensor  # [S, T, E, d]
    seconds: float
    nfe: float | None  # network forward calls per assimilation step and sequence
    pseudo_steps: int
    smc: list[dict[str, list[float]]]


def run_surge(
    cfg,
    name: str,
    problem,
    sequences,
    members: int,
    settings: dict,
    initial: Callable[[int, object], torch.Tensor],
    seeds: list[int],
    data_path: Path,
    config_dir: Path,
) -> SMCRun:
    """Run a SURGE cell sequence by sequence.

    Args:
        initial: ``(index, sequence) -> [E, d]`` initial members (the shared
            ``driver.initial_ensemble``).
        seeds: One sampler seed per sequence.
    """
    from cases.lorenz96._l96_generative import (
        Standardisation,
        load_prior,
        observation_model,
    )
    from cases.lorenz96._l96_system import resolve_device
    from cases.navier_stokes._ns_pipeline import attach_nfe_counter

    device = resolve_device(cfg.evaluation.device)
    stats = Standardisation.from_data(data_path, problem)
    observed = observation_model(problem, stats)
    if name in SURGE_BASELINES:
        si_model = load_prior(cfg.checkpoints.si_run, device)
        fm_model = load_prior(cfg.checkpoints.fm_run, device)
        model, posterior, stepper = build_surge_baseline(
            name, settings, si_model, fm_model, problem, stats, members, config_dir
        )
        num_steps = int(settings["num_steps"])
    else:
        guided = name in SURGE_OURS
        prior = SURGE_OURS[name] if guided else SURGE_UNGUIDED[name]
        cls = SurgeOursSettings if guided else SurgeUnguidedSettings
        parsed = _settings(cls, name, settings, prior)
        if not guided:
            parsed = SurgeOursSettings(
                num_steps=parsed.num_steps,
                diffusion_scale=parsed.diffusion_scale,
                ess_threshold=parsed.ess_threshold,
            )
        run = cfg.checkpoints.si_run if prior == "si_sde" else cfg.checkpoints.fm_run
        model = load_prior(run, device)
        posterior, stepper = build_surge_ours(
            prior, model, problem, stats, parsed, guided=guided
        )
        num_steps = parsed.num_steps
    recorder = SMCRecorder(posterior, members)
    counter = attach_nfe_counter(model)
    out, smc = [], []
    start = time.time()
    for index, seq in enumerate(sequences):
        x0 = initial(index, seq)[None]
        ensembles = assimilate_one(
            posterior,
            stepper,
            stats,
            x0,
            seq.observations[None],
            num_steps,
            seeds[index],
            observed.to_model,
        )
        out.append(ensembles[0])
        smc.append(recorder.take(seq.observations.shape[0]))
    seconds = time.time() - start
    steps = sum(seq.observations.shape[0] for seq in sequences)
    nfe = counter.count / steps if steps else math.nan
    return SMCRun(torch.stack(out), seconds, nfe, num_steps, smc)
