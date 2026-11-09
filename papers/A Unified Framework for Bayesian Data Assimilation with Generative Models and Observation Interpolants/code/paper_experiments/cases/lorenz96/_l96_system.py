"""The Lorenz 96 assimilation problem shared by every script of this case.

One :class:`Problem` fixes the model, the observation operator and the prior of
the initial state. :func:`make_sequence` draws a truth, its observations and the
prior mean from seeds alone, so every method sees identical inputs.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from omegaconf import OmegaConf

from scisi.dynamics.lorenz96 import StochasticLorenz96

CASE_CONFIG = Path(__file__).resolve().parents[2] / "configs/case/lorenz96.yaml"


@dataclass(frozen=True)
class Problem:
    """Defaults are those of ``configs/case/lorenz96.yaml``."""

    dim_state: int = 16
    forcing: float = 8.0
    dt: float = 0.01
    interval: float = 0.2
    noise_std: float = 1.0
    noise_stages: int = 1
    obs_stride: int = 4
    obs_variance: float = 4.0
    prior_std: float = 1.0
    num_steps: int = 50

    @property
    def model(self) -> StochasticLorenz96:
        return StochasticLorenz96(
            dim_state=self.dim_state,
            forcing=self.forcing,
            dt=self.dt,
            interval=self.interval,
            noise_std=self.noise_std,
            noise_stages=self.noise_stages,
        )

    @property
    def obs_indices(self) -> torch.Tensor:
        return torch.arange(0, self.dim_state, self.obs_stride)

    @property
    def obs_matrix(self) -> torch.Tensor:
        return torch.eye(self.dim_state, dtype=torch.float64)[self.obs_indices]

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_device(name: str) -> str:
    """``auto`` means CUDA, then MPS, then CPU; anything else passes through."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_problem(path: str | Path = CASE_CONFIG) -> Problem:
    return Problem(**OmegaConf.to_container(OmegaConf.load(path).problem))


@dataclass(frozen=True)
class Sequence:
    """One held-out assimilation problem."""

    prior_mean: torch.Tensor  # [d]
    truth: torch.Tensor  # [T + 1, d]; index 0 is the initial state
    observations: torch.Tensor  # [T, d_y]; observations[j] observes truth[j + 1]


def make_sequence(problem: Problem, seed: int, spinup_intervals: int = 200) -> Sequence:
    """Draw ``x^0 ~ N(prior_mean, prior_std^2 I)`` and roll out truth and data.

    ``prior_mean`` is an attractor state, so the prior is a genuine statement of
    uncertainty about ``x^0`` that all methods share.
    """
    gen = torch.Generator().manual_seed(int(seed))
    model = problem.model
    prior_mean = model.spin_up(1, spinup_intervals, gen)[0]
    x0 = prior_mean + problem.prior_std * torch.randn(
        problem.dim_state, generator=gen, dtype=torch.float64
    )
    truth = model.trajectory(x0, problem.num_steps, gen)
    noise = problem.obs_variance**0.5 * torch.randn(
        problem.num_steps,
        problem.obs_indices.numel(),
        generator=gen,
        dtype=torch.float64,
    )
    return Sequence(prior_mean, truth, truth[1:, problem.obs_indices] + noise)


def draw_prior_ensemble(
    problem: Problem,
    sequence: Sequence,
    members: int,
    seed: int,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Draw ``[members, d]`` from the shared prior of the initial state."""
    gen = torch.Generator().manual_seed(int(seed))
    noise = torch.randn(members, problem.dim_state, generator=gen, dtype=dtype)
    return sequence.prior_mean.to(dtype) + problem.prior_std * noise


def open_csv_for_append(path: Path, columns) -> tuple:
    """Open ``path`` to append rows with ``columns``; return ``(handle, writer)``.

    A new or empty file gets the header. An existing file must have exactly these
    columns, otherwise rows of different runs would mix silently, so it is refused.
    """
    columns = list(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="") as handle:
            header = next(csv.reader(handle), [])
        if header != columns:
            raise ValueError(
                f"{path} has columns {header}, not {columns}. Move it away or "
                "write to another file."
            )
    handle = open(path, "a", newline="")
    writer = csv.DictWriter(handle, columns)
    if handle.tell() == 0:
        writer.writeheader()
    return handle, writer
