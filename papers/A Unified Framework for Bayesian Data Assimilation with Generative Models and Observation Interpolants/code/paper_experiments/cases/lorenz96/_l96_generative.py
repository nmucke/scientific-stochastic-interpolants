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

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from cases.lorenz96._l96_system import Problem
from common.paths import REPO_ROOT
from results_schema import Method
from scisi.likelihood_models.gaussian_likelihood import InterpolantGaussianLikelihood
from scisi.likelihood_models.observation_operators import LinearObservationOperator
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
        """Read the constants, refusing data simulated from another system."""
        with np.load(path) as data:
            if json.loads(str(data["problem"])) != problem.to_dict():
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
    diffusion_scale: float = 1.0  # DM-SDE only


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
    """Return ``(posterior, stepper)`` for one of the three samplers."""
    is_si = method is Method.OURS_SI_SDE
    likelihood = InterpolantGaussianLikelihood(
        model=model,
        obs_operator=build_obs_operator(problem),
        variance=problem.obs_variance / stats.std**2,
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


def assimilate(
    posterior,
    stepper,
    stats: Standardisation,
    initial: torch.Tensor,
    observations: torch.Tensor,
    num_steps: int,
    seed: int,
) -> torch.Tensor:
    """Run the sampler over a batch of sequences.

    Args:
        initial: ``[S, E, d]`` initial ensembles in physical units.
        observations: ``[S, T, d_y]`` in physical units.

    Returns:
        ``[S, T, E, d]`` analysis ensembles in physical units.
    """
    num_sequences, members, d = initial.shape
    steps = observations.shape[1]
    mode = posterior.likelihood_model.likelihood_mode
    if num_sequences > 1 and mode == "inflated_shared":
        raise ValueError("inflated_shared averages over the batch, mixing sequences.")
    torch.manual_seed(int(seed))
    x0 = stats.forward(initial.float()).reshape(-1, 1, 1, d)
    y = stats.forward(observations.float())  # [S, T, d_y]
    y = y.permute(0, 2, 1).repeat_interleave(members, dim=0)  # [S E, d_y, T]
    # batch_size must cover every member: the sampler hands the unsliced observation
    # rows to each batch.
    trajectory = posterior.sample_trajectory(
        base=x0.clone(),
        field_history=x0.unsqueeze(-1).clone(),
        observations=y,
        ensemble_size=x0.shape[0],
        batch_size=x0.shape[0],
        num_steps=num_steps,
        num_physical_steps=steps + 1,
        stepper=stepper,
    )
    states = trajectory[..., 1:].reshape(num_sequences, members, d, steps)
    return stats.inverse(states.permute(0, 3, 1, 2).double())
