"""The Lorenz 96 assimilation problem shared by every script of this case.

One :class:`Problem` fixes the model, the observation operator and the prior of
the initial state. :func:`make_sequence` draws a truth, its observations and the
prior mean from seeds alone, so every method sees identical inputs.
:func:`case_config` is the one path from ``configs/case/lorenz96.yaml`` to a
configured case (and :func:`problem_of` to its :class:`Problem`).
"""

from __future__ import annotations

import csv
import functools
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
from omegaconf import OmegaConf

from lorenz96_lineup import CONFIG_BLOCK, OBS_CHOICES
from scisi.dynamics.lorenz96 import StochasticLorenz96
from scisi.likelihood_models.observation_operators import NONLINEARITIES

CASE_CONFIG = Path(__file__).resolve().parents[2] / "configs/case/lorenz96.yaml"

#: Observation operators of this case: ``driver.py --obs <name>``.
OBS_OPERATORS = OBS_CHOICES


def arctan_nonlinearity(w: torch.Tensor, strength: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``(arctan(c w) / c, 1 / (1 + (c w)^2))``: the library's ``arctan``
    (``observation_operators.NONLINEARITIES``), as a picklable function."""
    return NONLINEARITIES["arctan"](w, strength, None)


@dataclass(frozen=True)
class Observation:
    """The observation operator on the selected sites ``L x`` (physical units).

    ``linear``: ``h(x) = L x``. ``arctan``: ``h(x) = arctan(c L x) / c`` with
    ``c = strength``. The noise ``N(0, obs_variance I)`` is the same for both.
    """

    operator: str = "linear"
    strength: float | None = None

    def __post_init__(self) -> None:
        if self.operator not in OBS_OPERATORS:
            raise ValueError(f"observation.operator must be one of {OBS_OPERATORS}, got {self.operator!r}")
        if self.operator == "linear" and self.strength is not None:
            raise ValueError("a linear observation takes no strength")
        if self.operator == "arctan":
            if self.strength is None or not float(self.strength) > 0:
                raise ValueError(f"arctan needs a strength > 0, got {self.strength!r}")
            object.__setattr__(self, "strength", float(self.strength))


@dataclass(frozen=True)
class Problem:
    """Defaults are those of ``configs/case/lorenz96.yaml``.

    ``observation`` (default linear) selects the observation operator. A linear
    problem's :meth:`to_dict` has no ``observation`` key, exactly the dictionary
    stored in the data and the linear references before the option existed.
    """

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
    observation: Observation = field(default_factory=Observation)

    def __post_init__(self) -> None:
        if isinstance(self.observation, dict):
            object.__setattr__(self, "observation", Observation(**self.observation))
        elif not isinstance(self.observation, Observation):
            raise TypeError(f"observation must be a dict or Observation, got {self.observation!r}")

    @property
    def obs_label(self) -> str:
        """``linear`` or ``arctan``: the results directory of this problem."""
        return self.observation.operator

    @property
    def linear_obs(self) -> bool:
        return self.observation.operator == "linear"

    @property
    def obs_nonlinearity(self):
        """Elementwise ``g`` of ``h(x) = g(L x)`` as ``w -> (g(w), g'(w))``; ``None``
        when linear (the filters then take their unchanged linear paths)."""
        if self.linear_obs:
            return None
        return functools.partial(arctan_nonlinearity, strength=self.observation.strength)

    def observe(self, x: torch.Tensor) -> torch.Tensor:
        """``h(x)`` of physical states ``[..., d]`` -> ``[..., d_y]``."""
        selected = x[..., self.obs_indices]
        if self.linear_obs:
            return selected
        return self.obs_nonlinearity(selected)[0]

    def dynamics(self) -> "Problem":
        """The same problem with linear observations: what the data depend on."""
        return replace(self, observation=Observation())

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
        out = asdict(self)
        if self.linear_obs:
            del out["observation"]  # unchanged layout for the linear problem
        return out


def resolve_device(name: str) -> str:
    """``auto`` means CUDA, then MPS, then CPU; anything else passes through."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def case_config(obs: str = "linear"):
    """The case config for observation operator ``obs`` (:func:`configure`)."""
    return configure(OmegaConf.load(CASE_CONFIG), obs)


def configure(cfg, obs: str = "linear"):
    """Select observation operator ``obs`` in a loaded case config, in place.

    ``linear`` leaves ``problem`` and ``methods`` as they are. Otherwise
    ``problem.observation`` is set from ``observations.<obs>`` and ``methods`` is
    replaced by ``methods_<obs>`` (:data:`CONFIG_BLOCK`), so every cell reads the
    settings of that problem. ``cfg.obs`` records the choice; configuring an
    already configured config for another operator is refused.
    """
    if obs not in OBS_OPERATORS:
        raise ValueError(f"obs must be one of {OBS_OPERATORS}, got {obs!r}")
    current = cfg.get("obs")
    if current is not None:
        if current != obs:
            raise ValueError(f"config is already set up for obs={current!r}, not {obs!r}")
        return cfg
    if obs != "linear":
        cfg.problem.observation = OmegaConf.to_container(cfg.observations[obs])
        cfg.methods = cfg[CONFIG_BLOCK[obs]]
    cfg.obs = obs
    return cfg


def problem_of(cfg) -> Problem:
    return Problem(**OmegaConf.to_container(cfg.problem))


def load_problem(obs: str = "linear") -> Problem:
    """The configured problem with observation operator ``obs``
    (``observations.<obs>`` of the case config; ``linear`` adds nothing)."""
    return problem_of(case_config(obs))


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

    Every draw (spin-up, ``x^0``, the model noise and the observation noise ``e``)
    comes before and independently of the observation operator, so a nonlinear
    problem has the same truth and the same ``e`` as the linear one:
    ``y = h(truth) + e``.
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
    return Sequence(prior_mean, truth, problem.observe(truth[1:]) + noise)


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
