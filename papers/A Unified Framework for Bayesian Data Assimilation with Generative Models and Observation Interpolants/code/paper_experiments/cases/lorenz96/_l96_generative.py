"""Posterior sampling with the trained Lorenz 96 priors.

The observation-interpolant samplers of the paper run here exactly as in the other
cases. Models live in standardised units, so states, observations and the
observation variance are converted on the way in and states on the way out. Several
sequences run as one batch: every member carries its own observation row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from cases.lorenz96._l96_system import Problem
from common.paths import REPO_ROOT
from results_schema import Method
from scisi.likelihood_models.gaussian_likelihood import InterpolantGaussianLikelihood
from scisi.likelihood_models.observation_operators import (
    LinearObservationOperator,
    PointwiseObservationOperator,
)
from scisi.posterior_models.flow_matching_posterior import (
    FlowMatchingPosterior,
    endpoint_vanishing_diffusion,
)
from scisi.posterior_models.stochastic_interpolant_posterior import (
    StochasticInterpolantPosterior,
)
from scisi.sampling.ode_solvers import euler_step
from scisi.sampling.sde_solvers import euler_maruyama_step

CHECKPOINT_PROJECT = "lorenz96"


@dataclass(frozen=True)
class Standardisation:
    mean: float
    std: float

    @classmethod
    def from_data(cls, path: str | Path, problem: Problem) -> "Standardisation":
        """Read the constants, refusing data simulated from another system.

        The data hold states only, so they do not depend on the observation
        operator: they are compared with the problem's linear-observation version,
        which still refuses any other dynamics.
        """
        with np.load(path) as data:
            if json.loads(str(data["problem"])) != problem.dynamics().to_dict():
                raise ValueError(f"{path} was simulated from a different problem.")
            return cls(float(data["mean"]), float(data["std"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


@dataclass(frozen=True)
class SamplerSettings:
    """What is tuned on validation sequences, per method."""

    num_steps: int = 50
    likelihood_mode: str = "inflated"
    jacobian_damping: float = 1.0
    diffusion_scale: float = 1.0  # DM-SDE and TMPD
    covariance: str = "full"  # TMPD only: full | diagonal | row_sum (DTMPD)
    # TMPD only (defaults = configs/method/tmpd.yaml): score_substitution | g_squared
    guidance_weight: str = "score_substitution"
    vjp_chunk_size: int = 64  # TMPD only: VJP cotangents batched per autograd call


def build_obs_operator(problem: Problem) -> LinearObservationOperator:
    """Selection operator on the ``[1, 1, d]`` layout.

    The ``grid`` type strides both spatial axes, which leaves nothing on a lattice
    of height one, so the rows are set from the problem instead.
    """
    operator = LinearObservationOperator(
        type="grid", data_size=(1, 1, problem.dim_state), skip_grid=1
    )
    operator.obs_matrix = problem.obs_matrix.float()
    operator.obs_indices = problem.obs_indices
    operator.num_obs = problem.obs_indices.numel()
    return operator


@dataclass(frozen=True)
class ObservationModel:
    """How a sampler sees the observations, in the model's standardised units.

    ``operator`` acts on standardised states ``[B, 1, 1, d]``; ``to_model`` maps
    physical observations ``[..., d_y]`` into the units ``operator`` outputs, and
    ``variance`` is the observation-noise variance in those units. Every generative
    cell (ours, the baselines, the SURGE cells) takes its operator, variance and
    observation transform from :func:`observation_model` only, so that a new
    observation operator is wired in one place.
    """

    operator: object
    variance: float
    to_model: Callable[[torch.Tensor], torch.Tensor]


class StandardisedArctanOperator(PointwiseObservationOperator):
    """``h(z) = arctan(c (std L z + mean)) / c``: physical arctan observations of
    standardised states ``z = (x - mean) / std``.

    A :class:`PointwiseObservationOperator` ``g(L z)`` whose elementwise ``g`` folds
    in the de-standardisation, ``g(w) = arctan(c (std w + mean)) / c`` (the
    library's arctan of the physical state) and
    ``g'(w) = std / (1 + (c (std w + mean))^2)`` (written out: ``std`` times the
    library's ``g'`` would round differently). Only ``g`` / ``dg`` are
    overridden, so everything the likelihoods and the correction call (``forward``,
    ``jvp``, ``vjp``, ``gram``, ``jacobian``, ``linearize``, and ``g`` / ``dg`` /
    ``inner`` for ``inflated_shared`` and the statistical linearisation) is
    consistent, and ``inner`` stays the linear selection that places the
    observations for localisation. ``std = 1, mean = 0`` is the library's arctan.
    """

    def __init__(self, inner, strength: float, std: float = 1.0, mean: float = 0.0):
        super().__init__(inner, nonlinearity="arctan", strength=strength)
        self.std = float(std)
        self.mean = float(mean)

    def g(self, z: torch.Tensor) -> torch.Tensor:
        return super().g(self.std * z + self.mean)

    def dg(self, z: torch.Tensor) -> torch.Tensor:
        cw = self.strength * (self.std * z + self.mean)
        return self.std / (1.0 + cw**2)


def observation_model(problem: Problem, stats: "Standardisation") -> ObservationModel:
    """The observation model of ``problem`` for samplers in standardised units.

    Linear selection: ``H`` commutes with the affine standardisation, so the
    observations are standardised like the states and ``R`` is divided by
    ``std**2`` (the behaviour of every cell before this helper existed).

    Arctan: the operator maps standardised states to PHYSICAL observations,
    ``arctan(c (std L z + mean)) / c`` (:class:`StandardisedArctanOperator`), so the
    observations are used as they are (``to_model`` is the identity) and the
    variance is the physical ``R``.
    """
    if problem.linear_obs:
        return ObservationModel(
            operator=build_obs_operator(problem),
            variance=problem.obs_variance / stats.std**2,
            to_model=stats.forward,
        )
    return ObservationModel(
        operator=StandardisedArctanOperator(
            build_obs_operator(problem),
            problem.observation.strength,
            std=stats.std,
            mean=stats.mean,
        ),
        variance=float(problem.obs_variance),
        to_model=_identity,
    )


def physical_obs_operator(problem: Problem):
    """``h`` on PHYSICAL states ``[B, 1, 1, d]`` as an operator (the oracle's
    correction): the linear selection, or the arctan operator with ``std = 1``,
    ``mean = 0``."""
    if problem.linear_obs:
        return build_obs_operator(problem)
    return StandardisedArctanOperator(build_obs_operator(problem), problem.observation.strength)


def _identity(y: torch.Tensor) -> torch.Tensor:
    return y


def load_prior(run: str, device: str = "cpu") -> torch.nn.Module:
    """Rebuild a trained prior from its checkpoint directory."""
    directory = REPO_ROOT / "checkpoints" / CHECKPOINT_PROJECT / run
    cfg = OmegaConf.load(directory / "config.yaml")
    model = hydra.utils.instantiate(cfg.model)
    state = torch.load(directory / "model.pth", map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def build_posterior(
    method: Method,
    model: torch.nn.Module,
    problem: Problem,
    stats: Standardisation,
    settings: SamplerSettings,
):
    """Return ``(posterior, stepper)`` for one of the three samplers or TMPD."""
    if method is Method.TMPD:
        return build_tmpd_posterior(model, problem, stats, settings), euler_maruyama_step
    is_si = method is Method.OURS_SI_SDE
    observed = observation_model(problem, stats)
    likelihood = InterpolantGaussianLikelihood(
        model=model,
        obs_operator=observed.operator,
        variance=observed.variance,
        likelihood_mode=settings.likelihood_mode,
        model_class="si" if is_si else "fm",
        jacobian_damping=settings.jacobian_damping,
    )
    if is_si:
        posterior = StochasticInterpolantPosterior(
            model=model, likelihood_model=likelihood
        )
        return posterior, euler_maruyama_step
    if method is Method.OURS_FM_ODE:
        posterior = FlowMatchingPosterior(
            model=model, likelihood_model=likelihood, diffusion_term=None
        )
        return posterior, euler_step
    if method is Method.OURS_DM_SDE:
        diffusion = endpoint_vanishing_diffusion(
            model.interpolation, settings.diffusion_scale
        )
        posterior = FlowMatchingPosterior(
            model=model, likelihood_model=likelihood, diffusion_term=diffusion
        )
        return posterior, euler_maruyama_step
    raise ValueError(f"Not a generative method: {method}")


def build_tmpd_posterior(model, problem: Problem, stats: Standardisation, settings: SamplerSettings):
    """TMPD (boys_tweedie_2024) on the FM prior read as a diffusion model.

    The score comes from the trained velocity (score--velocity duality, anchor 0),
    as for the SDA baseline elsewhere. The reverse SDE uses the same
    endpoint-vanishing ``g = diffusion_scale * sqrt(alpha beta)`` as the DM-SDE cell.
    The FM model is passed as is, with that ``g`` given explicitly, so the
    ``tau = 0`` step keeps the velocity (drift ``v + 1/2 g^2 s = v`` there): the
    same full first step as ``FullFirstStepDiffusionModel`` in the multistep
    analytical case, which matters in a sequential case (the stock
    ``DenoiseDiffusionModel`` first step drops ``v dt``). Only the network forward
    is counted by the NFE counter; the per-step VJPs (``d_y`` per guided step for
    full / diagonal, 2 for row_sum) are counted by ``likelihood.vjp_count`` and
    reported by the driver as a separate ``vjp`` row.

    Linear ``h`` only: the TMPD likelihood raises ``NotImplementedError`` for the
    arctan operator (the cell is linear-only, ``lorenz96_lineup.Row.sweeps``).
    """
    from scisi.likelihood_models.tmpd import TMPDLikelihood
    from scisi.posterior_models.tmpd_posterior import TMPDPosterior

    observed = observation_model(problem, stats)
    likelihood = TMPDLikelihood(
        model=model,
        obs_operator=observed.operator,
        variance=observed.variance,
        model_class="fm",
        covariance=settings.covariance,
        vjp_chunk_size=settings.vjp_chunk_size,
    )
    diffusion = endpoint_vanishing_diffusion(model.interpolation, settings.diffusion_scale)
    return TMPDPosterior(
        model=model,
        likelihood_model=likelihood,
        diffusion_term=diffusion,
        guidance_weight=settings.guidance_weight,
    )


def assimilate(
    posterior,
    stepper,
    stats: Standardisation,
    initial: torch.Tensor,
    observations: torch.Tensor,
    num_steps: int,
    seed: int,
    forecast_correction=None,
    obs_to_model: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Run the sampler over a batch of sequences.

    Args:
        initial: ``[S, E, d]`` initial ensembles in physical units.
        observations: ``[S, T, d_y]`` in physical units.
        forecast_correction: Optional ``ForecastCorrection`` (see
            :func:`build_correction`); ``None`` leaves the call exactly as before.
        obs_to_model: ``ObservationModel.to_model`` of the cell; ``None`` means
            ``stats.forward`` (linear selection).

    Returns:
        ``[S, T, E, d]`` analysis ensembles in physical units.
    """
    num_sequences, members, d = initial.shape
    steps = observations.shape[1]
    # Baseline likelihoods (FlowDAS, SDA, ...) have no likelihood_mode.
    mode = getattr(posterior.likelihood_model, "likelihood_mode", None)
    if num_sequences > 1 and mode == "inflated_shared":
        raise ValueError("inflated_shared averages over the batch, mixing sequences.")
    torch.manual_seed(int(seed))
    x0 = stats.forward(initial.float()).reshape(-1, 1, 1, d)
    to_model = stats.forward if obs_to_model is None else obs_to_model
    y = to_model(observations.float())  # [S, T, d_y]
    y = y.permute(0, 2, 1).repeat_interleave(members, dim=0)  # [S E, d_y, T]
    # batch_size must cover every member: the sampler hands the unsliced observation
    # rows to each batch.
    extra = {} if forecast_correction is None else {"forecast_correction": forecast_correction}
    trajectory = posterior.sample_trajectory(
        base=x0.clone(),
        field_history=x0.unsqueeze(-1).clone(),
        observations=y,
        ensemble_size=x0.shape[0],
        batch_size=x0.shape[0],
        num_steps=num_steps,
        num_physical_steps=steps + 1,
        stepper=stepper,
        **extra,
    )
    states = trajectory[..., 1:].reshape(num_sequences, members, d, steps)
    return stats.inverse(states.permute(0, 3, 1, 2).double())


# --------------------------------------------------------------------------- #
# Forecast-residual correction and the "trained prior + EnKF" baseline
# --------------------------------------------------------------------------- #

#: Keys of a ``correction:`` block in ``configs/case/lorenz96.yaml``.
CORRECTION_KEYS = (
    "enabled",
    "forecast_num_steps",
    "localisation",
    "gain_scale",
    "q_source",
    "update_cap",
)
#: ``q_source`` values of a ``correction:`` block. ``exact`` is the library's
#: ``fixed`` source with the true transition variance, supplied by the case.
Q_SOURCES = ("online", "exact")


def correction_config(problem: Problem, settings: dict, members: int, q_variance: float):
    """``ForecastCorrectionConfig`` from a ``correction:`` block.

    Args:
        settings: Keys of :data:`CORRECTION_KEYS`; unknown keys are refused.
            ``localisation`` is a Gaspari-Cohn radius in sites (``None``: none).
            ``q_source`` is ``online`` (the library default: ``Q_hat`` from a second
            forecast per member) or ``exact``: the known transition variance
            ``q_variance`` (library ``q_source="fixed"``, closed-form
            ``K_q = Q H' (H Q H' + R)^-1``). The trained samplers are not told Q;
            only the oracle uses ``exact`` by default.
        members: Ensemble size ``E``; statistics are per sequence (``group_size``).
        q_variance: Transition-noise variance per site, in the units the states
            are corrected in.
    """
    from scisi.posterior_models.forecast_correction import (
        ForecastCorrectionConfig,
        grid_localisation,
    )

    unknown = set(settings) - set(CORRECTION_KEYS)
    if unknown:
        raise ValueError(f"Unknown correction settings: {sorted(unknown)}")
    q_source = settings.get("q_source", "online")
    if q_source not in Q_SOURCES:
        raise ValueError(f"correction.q_source must be one of {Q_SOURCES}, got {q_source!r}")
    exact = q_source == "exact"
    radius = settings.get("localisation")
    steps = settings.get("forecast_num_steps")
    cap = settings.get("update_cap")
    return ForecastCorrectionConfig(
        enabled=bool(settings.get("enabled", True)),
        forecast_num_steps=None if steps is None else int(steps),
        # On the [1, 1, d] layout: identical to ``enkf.periodic_taper(d, radius)``.
        localisation=None
        if radius is None
        else grid_localisation((1, 1, problem.dim_state), float(radius), periodic=True),
        gain_scale=float(settings.get("gain_scale", 1.0)),
        q_source="fixed" if exact else "online",
        q_variance=float(q_variance) if exact else None,
        group_size=int(members),
        update_cap=None if cap is None else float(cap),
    )


def build_correction(problem: Problem, stats: Standardisation, members: int, settings: dict):
    """The correction of a generative cell, in the model's standardised units.

    On MPS the library runs the update on the CPU in float64 by itself.
    """
    from scisi.posterior_models.forecast_correction import ForecastCorrection

    q_model = problem.noise_std**2 / stats.std**2
    return ForecastCorrection(correction_config(problem, settings, members, q_model))


def assimilate_prior_enkf(
    posterior,
    stepper,
    stats: Standardisation,
    initial: torch.Tensor,
    observations: torch.Tensor,
    num_steps: int,
    seed: int,
    enkf,
    generators: list[torch.Generator],
) -> torch.Tensor:
    """Unguided trained-prior forecast, then an EnKF analysis per sequence.

    The forecast is the unguided version of the posterior's own SDE/ODE
    (``draw_unguided_forecast``); the observation never enters the sampler. The
    analysis is ``enkf.analyse`` (the same update as the true-model EnKF cell) in
    physical units.

    Args:
        initial: ``[S, E, d]`` physical units. observations: ``[S, T, d_y]``.
        generators: One CPU generator per sequence for the EnKF perturbations.

    Returns:
        ``[S, T, E, d]`` analysis ensembles in physical units.
    """
    from scisi.posterior_models.forecast_correction import draw_unguided_forecast

    num_sequences, members, d = initial.shape
    device = posterior.device
    torch.manual_seed(int(seed))
    x = stats.forward(initial.float()).reshape(-1, 1, 1, d).to(device)
    out = []
    for t in range(observations.shape[1]):
        f = draw_unguided_forecast(
            posterior,
            x.unsqueeze(-1),
            num_steps=num_steps,
            stepper=stepper,
            batch_size=x.shape[0],
        )
        forecast = stats.inverse(f.reshape(num_sequences, members, d).cpu().double())
        analysis = torch.stack(
            [
                enkf.analyse(forecast[s], observations[s, t], generators[s])
                for s in range(num_sequences)
            ]
        )
        out.append(analysis)
        x = stats.forward(analysis).float().reshape(-1, 1, 1, d).to(device)
    return torch.stack(out, dim=1)
