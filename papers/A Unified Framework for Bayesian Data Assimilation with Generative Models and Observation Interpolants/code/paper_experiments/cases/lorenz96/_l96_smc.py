"""SMC cells of the Lorenz 96 case: SURGE around Ours, SURGE + FlowDAS / SDA, and ESS logs.

Cells (:data:`SURGE_CELLS`; labels in ``lorenz96_lineup.ROWS``):

* ``surge_si_sde`` / ``surge_dm_sde``: Ours (SI-SDE / DM-SDE) inside the SURGE SMC
  layer, built as the multistep analytical case builds them
  (``cases/analytical/multistep.py``, ``MultistepRunner._ours_cycle`` with
  ``surge=True``): the plain cell's likelihood is the guidance of
  ``SurgeStochasticInterpolantPosterior`` (SI) or of ``OursSurgeDiffusionPosterior``
  on the ``FullFirstStepDiffusionModel`` lift of the FM prior (DM). SDE only: an
  ODE has no path measure for SURGE's Girsanov weight. On the SI prior (here and in
  ``surge_unguided_si``, ``surge_flowdas``) the reward uses the exact Follmer-drift
  denoiser (``si_denoiser="follmer"``). Their sampler keys are the
  plain cell's (``driver.INHERITS``); their own block holds ``ess_threshold``.
* ``surge_unguided_si`` / ``surge_unguided_dm``: the same SMC proposing from the
  unguided prior (``NullGuidance``), one control per wrapped prior, as in the
  multistep case (its variant ``surge_unguided`` of each Ours row).
* ``surge_flowdas`` / ``surge_sda``: the Navier--Stokes constructions
  (``_l96_baselines.ns_posterior``).

SURGE normalises its weights over every member it is given
(``SurgePosteriorBase._maybe_resample``), so each sampler call holds ONE sequence
(``E`` members); the recorder refuses anything else.

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

import time
from dataclasses import dataclass

import torch

from lorenz96_lineup import (
    SMC_ESS_FRACTION as SMC_ESS,
    SMC_ESS_MIN_FRACTION as SMC_ESS_MIN,
    SMC_METRICS as SMC_KEYS,
    SMC_RESAMPLES,
)
from results_schema import Method

# Ours inside SURGE: cell -> the plain cell whose likelihood is the guidance.
SURGE_OURS = {"surge_si_sde": "si_sde", "surge_dm_sde": "dm_sde"}
# The same SMC on the unguided prior, one per wrapped prior.
SURGE_UNGUIDED = {"surge_unguided_si": "si_sde", "surge_unguided_dm": "dm_sde"}
# The Navier--Stokes constructions (_l96_baselines.NS_CELLS).
SURGE_NS = ("surge_flowdas", "surge_sda")
SURGE_CELLS = (*SURGE_OURS, *SURGE_UNGUIDED, *SURGE_NS)
_PRIOR_METHOD = {"si_sde": Method.OURS_SI_SDE, "dm_sde": Method.OURS_DM_SDE}


# --------------------------------------------------------------------------- #
# ESS / resampling recorder
# --------------------------------------------------------------------------- #


class SMCRecorder:
    """Records ESS and resampling of a SURGE posterior, one entry per ``sample`` call.

    Wraps ``posterior._maybe_resample`` on the instance. The ESS is recomputed from
    ``posterior._normalized_weights()`` exactly as the library does, before the
    library's call; the wrapped call itself is unchanged. A call holding more than
    one sequence's members is refused.
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
class SurgeSettings:
    """Settings of the four SURGE cells around Ours (allowed keys: :data:`SURGE_KEYS`).

    The plain sampler's knobs plus SURGE's threshold. There is no
    particles-per-member knob: SURGE's SMC runs one particle per member (N = E),
    and neither the multistep classes nor the library expose one.
    """

    num_steps: int = 50
    likelihood_mode: str = "inflated"
    jacobian_damping: float = 1.0
    diffusion_scale: float = 1.0  # DM prior only
    ess_threshold: float = 0.5


#: Keys each cell accepts; missing keys take the :class:`SurgeSettings` defaults.
SURGE_KEYS = {
    "surge_si_sde": ("num_steps", "likelihood_mode", "jacobian_damping", "ess_threshold"),
    "surge_dm_sde": (
        "num_steps", "likelihood_mode", "jacobian_damping", "diffusion_scale", "ess_threshold",
    ),
    "surge_unguided_si": ("num_steps", "ess_threshold"),
    "surge_unguided_dm": ("num_steps", "diffusion_scale", "ess_threshold"),
}


def surge_settings(name: str, settings: dict) -> SurgeSettings:
    unknown = set(settings) - set(SURGE_KEYS[name])
    if unknown:
        raise ValueError(f"{name}: unknown settings {sorted(unknown)}")
    return SurgeSettings(**settings)


# --------------------------------------------------------------------------- #
# Posteriors
# --------------------------------------------------------------------------- #


def build_surge_ours(prior: str, model, problem, stats, settings: SurgeSettings, guided: bool = True):
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
        # The reward's denoiser inverts the Follmer drift exactly (the library's
        # default `velocity` treats it as the path velocity).
        posterior = SurgeStochasticInterpolantPosterior(
            model=model,
            obs_operator=observed.operator,
            variance=observed.variance,
            likelihood_model=likelihood,
            ess_threshold=ess,
            diffusion_term=None,
            si_denoiser="follmer",
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


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def run_surge(cfg, name, problem, sequences, split, seed, members, settings):
    """Run a SURGE cell one sequence per sampler call; returns the driver's
    ``CellOutput`` with the per-sequence SMC records. Sequence ``i`` is sampled with
    the seed ``derive_seed(SEED_LABEL, name, split, i, seed)``."""
    import cases.lorenz96._l96_generative as generative
    from cases.lorenz96 import driver
    from cases.lorenz96._l96_baselines import ns_posterior
    from cases.navier_stokes._ns_pipeline import attach_nfe_counter
    from common.seeding import derive_seed

    stats, device, initial, observations = driver.sampler_inputs(
        cfg, problem, sequences, split, seed, members
    )
    if name in SURGE_NS:
        model, posterior, stepper, observed, num_steps = ns_posterior(
            cfg, name, problem, stats, members, settings, device
        )
    else:
        guided = name in SURGE_OURS
        prior = SURGE_OURS[name] if guided else SURGE_UNGUIDED[name]
        parsed = surge_settings(name, settings)
        run = cfg.checkpoints.si_run if prior == "si_sde" else cfg.checkpoints.fm_run
        model = generative.load_prior(run, device)
        posterior, stepper = build_surge_ours(prior, model, problem, stats, parsed, guided)
        observed = generative.observation_model(problem, stats)
        num_steps = parsed.num_steps
    recorder = SMCRecorder(posterior, members)
    counter = attach_nfe_counter(model)
    out, smc = [], []
    start = time.time()
    for index, seq in enumerate(sequences):
        ensembles = generative.assimilate(
            posterior,
            stepper,
            stats,
            initial[index : index + 1],
            observations[index : index + 1],
            num_steps,
            derive_seed(driver.SEED_LABEL, name, split, index, seed),
            obs_to_model=observed.to_model,
        )
        out.append(ensembles[0])
        smc.append(recorder.take(seq.observations.shape[0]))
    seconds = time.time() - start
    nfe = counter.count / sum(seq.observations.shape[0] for seq in sequences)
    return driver.CellOutput(torch.stack(out), seconds, nfe, members, num_steps, smc=smc)
