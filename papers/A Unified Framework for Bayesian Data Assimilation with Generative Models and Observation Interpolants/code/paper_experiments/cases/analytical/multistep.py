"""Multistep analytical case: sequential filtering on a linear--Gaussian system.

Over ``K`` assimilation cycles, each method's posterior ensemble at cycle ``k`` is
its input at cycle ``k + 1``: member ``i`` is the anchor ``x^k_i`` of member ``i``'s
next draw, exactly as ``BasePosterior.sample_trajectory`` threads ``field_history``
on Navier--Stokes. The reference is the Kalman filter. With ``K = 1`` and the
paper's ``A = Q = H = R = I`` this is the single-step case: the same seeds, the
same truth / observation / reference draws, the same model and posterior classes.

``driver.py`` is deliberately untouched, so the numbers behind the paper cannot
move; the baselines are therefore re-assembled here around an ensemble input.

Two details a reader of the results needs:

* The per-member update targets the ``member_chain_step`` law, not the Kalman
  posterior. Hence the extra ``kl_chain`` metric and the two closed-form reference
  rows; see ``README.md``.
* From cycle 2 on, SDA and SURGE (SDA) use ``FullFirstStepDiffusionModel``, which
  corrects a first-step omission in ``DenoiseDiffusionModel`` that the single-step
  case does not expose. ``src/scisi`` is left as it is.
* With ``forecast_correction`` (off by default) the Ours rows also run with the
  forecast-residual correction of ``FORECAST_CORRECTION.md``, as variants
  ``<variant>_fc<S>[_bank][_g<E>]`` (:func:`fc_variant`). Off, every row is
  bit-for-bit what it was.
* ``sliced_w1`` is the single-step ``sliced_w2`` column under its correct name:
  ``analytical_utils.wasserstein_distance`` averages 1-D **W1** distances. In this
  case's output ``sliced_w2`` is a true sliced W2 (:func:`sliced_distances`).
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from common.aggregation import aggregate_over_seeds
from common.paths import output_path
from common.per_step_io import append_per_step, per_step_rows
from common.runner import ExperimentRunner, RunContext
from common.seeding import derive_seed
from results_schema import Method, Metric, ResultRecord, Scenario

from scisi.likelihood_models.dflow import DFlowSGLDLikelihood
from scisi.likelihood_models.gaussian_likelihood import (
    FlowdasGaussianLikelihood,
    InterpolantGaussianLikelihood,
)
from scisi.likelihood_models.guidance import FIGGaussianLikelihood
from scisi.likelihood_models.sda import SDALikelihood
from scisi.posterior_models.dflow_posterior import DFlowPosterior
from scisi.posterior_models.diffusion_posterior import DiffusionPosterior
from scisi.models.diffusion_model import DenoiseDiffusionModel
from scisi.posterior_models.flow_matching_posterior import (
    FlowMatchingPosterior,
    endpoint_vanishing_diffusion,
)
from scisi.posterior_models.forecast_correction import (
    ForecastCorrection,
    ForecastCorrectionConfig,
    build_residual_bank,
)
from scisi.sampling.sde_solvers import euler_maruyama_step
from scisi.posterior_models.stochastic_interpolant_posterior import (
    StochasticInterpolantPosterior,
)
from scisi.posterior_models.surge_posterior import (
    SurgeFlowMatchingPosterior,
    SurgeStochasticInterpolantPosterior,
)

from .driver import (
    _G0,
    _OURS_SAMPLER,
    _OURS_VARIANTS,
    ANALYTICAL_METHODS,
    AnalyticalRunner,
    _cfg_hparam,
    gaussian_kl_divergence,
)
from .linear_system import (
    DenseObservationOperator,
    FullFirstStepDiffusionModel,
    LinearGaussianSSM,
    NullGuidance,
    OursSurgeDiffusionPosterior,
    adapted_resample,
    build_linear_fm_model,
    build_linear_si_model,
    enkf_step,
    gaussian_samples,
    particle_filter_step,
)
from .prior_models import (
    build_fm_model,
    build_identity_obs_operator,
    build_si_model,
)

logger = logging.getLogger(__name__)

# The single-step lineup plus a diagnostic variant of the "shared" Ours rows that
# adds the predictive reweighting a per-member update leaves out.
ADAPTED_VARIANT = "adapted_shared"

# Ours inside the SURGE SMC layer, in the grid's separate ``surge_only`` cells. Only
# the two SDEs: SURGE's weight is a Girsanov ratio of path measures with the noise
# ``g dW``, which an ODE (g = 0) does not have. Giving FM-ODE noise makes it DM-SDE.
SURGE_PREFIX = "surge_"
SURGE_UNGUIDED = f"{SURGE_PREFIX}unguided"  # the same SMC, no sampler inside it
SURGE_WRAPPED: tuple[Method, ...] = (Method.OURS_SI_SDE, Method.OURS_DM_SDE)

# Forecast-residual correction (FORECAST_CORRECTION.md, variant A) on the Ours rows.
# A corrected row is ``<plain variant>_fc<forecast steps>``, plus ``_bank`` when Q_hat
# comes from a residual bank instead of the online second forecast, plus ``_g<E>``
# when the ensemble statistics are taken over independent groups of E members (the
# pooled N members are still what is scored). ``GROUP_ONLY`` cells hold the
# grouped comparators: EnKF and ``adapted_shared`` on the same groups.
FC_TAG = "_fc"
FC_Q_SOURCES = ("online", "bank")


def fc_variant(
    base: str, steps: int, q_source: str = "online", group: int | None = None
) -> str:
    """Variant tag of a corrected row, e.g. ``shared_fc5``, ``jacfree_fc2_bank_g64``."""
    if q_source not in FC_Q_SOURCES:
        raise ValueError(f"q_source must be one of {FC_Q_SOURCES}, got {q_source!r}")
    tag = f"{base}{FC_TAG}{int(steps)}"
    if q_source == "bank":
        tag += "_bank"
    if group is not None:
        tag += f"_g{int(group)}"
    return tag


def group_tag(group: int) -> str:
    return f"g{int(group)}"


def parse_fc_variant(variant: str | None) -> dict[str, object] | None:
    """``{base, steps, q_source, group}`` of a corrected tag, ``None`` otherwise."""
    match = re.fullmatch(r"(.+?)_fc(\d+)(_bank)?(?:_g(\d+))?", variant or "")
    if match is None:
        return None
    return {
        "base": match.group(1),
        "steps": int(match.group(2)),
        "q_source": "bank" if match.group(3) else "online",
        "group": int(match.group(4)) if match.group(4) else None,
    }


def _groups(n: int, group: int | None) -> list[slice]:
    """Contiguous member groups of size ``group``; the last one takes the remainder."""
    if group is None or group >= n:
        return [slice(0, n)]
    if group < 2:
        raise ValueError("a group needs at least 2 members")
    starts = list(range(0, n, group))
    if n - starts[-1] < 2:  # never leave a single member on its own
        starts.pop()
    return [slice(a, b) for a, b in zip(starts, [*starts[1:], n])]


def _per_group(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, group):
    return torch.cat([fn(x[sl]) for sl in _groups(x.shape[0], group)])


class GroupedForecastCorrection(ForecastCorrection):
    """``ForecastCorrection`` whose groups need not divide the batch.

    The library's ``group_size`` must divide ``N``; the paper protocol has
    ``N = 100000``, which 64 does not divide. Each contiguous group (the last one
    holds the remainder, see :func:`_groups`) is corrected as its own ensemble.

    It also reorders the unlocalised gain product. The library forms
    ``(HX Z)' X``, whose ``HX Z`` is ``E x E_out``: 80 GB at ``E = 10^5``. Here it is
    ``Z' (HX' X)``, with an ``m x d`` middle factor; the same matrix, in another
    floating-point order.
    """

    def _apply_gain(self, X, HX, scale, Z, loc, operator):
        if loc is not None:
            return super()._apply_gain(X, HX, scale, Z, loc, operator)
        return Z.T @ (HX.T @ X) * scale

    def __init__(self, config: ForecastCorrectionConfig, group: int | None) -> None:
        super().__init__(config)
        self.group = group

    def correct(self, x, f, y, operator=None, obs_noise_var=None, residuals=None,
                perturbations=None, phys_idx=None):
        slices = _groups(x.shape[0], self.group)
        if len(slices) == 1:
            return super().correct(
                x, f, y, operator=operator, obs_noise_var=obs_noise_var,
                residuals=residuals, perturbations=perturbations, phys_idx=phys_idx,
            )
        if y.reshape(-1, y.shape[-1]).shape[0] != 1:
            raise ValueError("grouped correction here takes one shared observation row")
        return torch.cat([
            super(GroupedForecastCorrection, self).correct(
                x[sl], f[sl], y, operator=operator, obs_noise_var=obs_noise_var,
                residuals=None if residuals is None else residuals[sl],
                perturbations=None if perturbations is None else perturbations[sl],
                phys_idx=phys_idx,
            )
            for sl in slices
        ])


# Closed-form rows, one ensemble drawn straight from the named Gaussian per cycle.
# The first is the Monte-Carlo floor of the metrics for iid members; the second is
# the score of a PERFECT per-member sampler. A biased sampler can happen to land
# closer to Kalman than this row, so it is a reference, not a lower bound.
REFERENCE_KALMAN = "Kalman filter (sampled)"
REFERENCE_MEMBER_CHAIN = "Exact per-member chain"

METRICS: tuple[str, ...] = (
    Metric.KL_POINTS.value,  # KL(filtering posterior || Gaussian fit), sampled reference
    "kl_exact",  # the same KL against the exact Kalman moments, no reference noise
    Metric.SLICED_W2.value,  # true sliced W2 to the filtering posterior
    "sliced_w1",  # the single-step estimator, see the module docstring
    "kl_chain",  # KL to the member-chain law: only meaningful for per-member rows
    "mean_err",  # || ensemble mean - Kalman mean ||
    "var_ratio",  # tr(ensemble cov) / tr(Kalman cov)
)


def sliced_distances(
    x: np.ndarray, y: np.ndarray, num_projections: int = 128, seed: int = 0
) -> tuple[float, float]:
    """Sliced ``(W1, W2)`` for equal-size samples.

    The 1-D Wasserstein distance between two equal-size empirical measures pairs
    their order statistics, so one sort per slice serves both. ``W1`` is
    ``mean_theta W1(theta.x, theta.y)``, the same number as
    ``analytical_utils.wasserstein_distance`` at a fraction of its cost at
    ``n = 10^5``; ``W2`` is ``sqrt(mean_theta W2^2)``.
    """
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(num_projections, x.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    gaps = np.sort(directions @ x.T, axis=1) - np.sort(directions @ y.T, axis=1)
    return float(np.abs(gaps).mean()), float(np.sqrt(np.mean(gaps**2)))


def _safe_kl(ref: np.ndarray, samples: np.ndarray) -> float:
    """Gaussian KL; a collapsed ensemble (singular covariance) reports NaN."""
    try:
        value = float(gaussian_kl_divergence(ref, samples))
    except (ValueError, np.linalg.LinAlgError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def kl_to_moments(mean: np.ndarray, cov: np.ndarray, samples: np.ndarray) -> float:
    """``KL(N(mean, cov) || N(sample mean, sample cov))``, NaN if collapsed.

    The direction of ``kl_points``, without the noise of a sampled reference.
    """
    fit_cov = np.atleast_2d(np.cov(samples, rowvar=False))
    sign, logdet_fit = np.linalg.slogdet(fit_cov)
    if sign <= 0:
        return float("nan")
    try:
        inv_fit = np.linalg.inv(fit_cov)
    except np.linalg.LinAlgError:
        return float("nan")
    diff = samples.mean(0) - mean
    value = 0.5 * float(
        np.trace(inv_fit @ cov) + diff @ inv_fit @ diff - mean.shape[0]
        + logdet_fit - np.linalg.slogdet(cov)[1]
    )
    return value if np.isfinite(value) else float("nan")


@dataclass(frozen=True)
class Cycle:
    """Everything shared by all methods at one assimilation cycle of one seed."""

    y: torch.Tensor  # [m]
    kalman_mean: torch.Tensor  # [d], float64
    kalman_cov: torch.Tensor  # [d, d], float64
    chain_mean: torch.Tensor
    chain_cov: torch.Tensor
    ref_kalman: np.ndarray  # [n, d] exact filtering-posterior samples
    ref_chain: np.ndarray  # [n, d] member-chain samples


def score_ensemble(cycle: Cycle, samples: np.ndarray) -> dict[str, float]:
    """All :data:`METRICS` for one ensemble."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        w1, w2 = sliced_distances(cycle.ref_kalman, samples)
        return {
            Metric.KL_POINTS.value: _safe_kl(cycle.ref_kalman, samples),
            "kl_exact": kl_to_moments(
                cycle.kalman_mean.numpy(), cycle.kalman_cov.numpy(),
                samples.astype(np.float64),
            ),
            Metric.SLICED_W2.value: w2,
            "sliced_w1": w1,
            "kl_chain": _safe_kl(cycle.ref_chain, samples),
            "mean_err": float(
                np.linalg.norm(samples.mean(0) - cycle.kalman_mean.numpy())
            ),
            "var_ratio": float(
                np.trace(np.atleast_2d(np.cov(samples, rowvar=False)))
                / np.trace(cycle.kalman_cov.numpy())
            ),
        }


class MultistepRunner(AnalyticalRunner):
    """Runner for the sequential linear--Gaussian case."""

    def __init__(self, config: object, *, seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> None:
        super().__init__(config, seeds=seeds)
        self._cycles: dict[tuple[int, int], tuple[torch.Tensor, list[Cycle]]] = {}
        self._per_step: list[dict[str, object]] = []

    # -- config ------------------------------------------------------------ #

    @property
    def surge_only(self) -> bool:
        """Run only the SURGE-wrapped Ours rows, for cells added to a finished grid."""
        return bool(self._case_get("surge_only", False))

    @property
    def fc_only(self) -> bool:
        """Run only the corrected Ours rows (a cell added to a finished grid)."""
        return bool(self._case_get("fc_only", False))

    @property
    def group_only(self) -> bool:
        """Run only the grouped comparators: EnKF and ``adapted_shared`` per group."""
        return bool(self._case_get("group_only", False))

    @property
    def added_cell(self) -> bool:
        """A cell joined to a finished grid: it holds no reference rows."""
        return self.surge_only or self.fc_only or self.group_only

    def methods(self) -> Sequence[Method]:
        if sum((self.surge_only, self.fc_only, self.group_only)) > 1:
            raise ValueError("surge_only, fc_only and group_only are separate cells")
        if self.surge_only:
            return SURGE_WRAPPED
        if self.fc_only:
            return tuple(_OURS_SAMPLER)
        if self.group_only:
            if self.fc_group_size is None:
                raise ValueError("group_only needs fc_group_size")
            return (*_OURS_SAMPLER, Method.ENKF)
        return super().methods()

    # -- forecast correction -------------------------------------------------- #

    @property
    def fc_enabled(self) -> bool:
        return self.fc_only or bool(self._case_get("forecast_correction", False))

    def fc_steps(self, num_steps: int) -> list[int]:
        """``forecast_num_steps`` values to run; default the sampler's ``M``."""
        value = self._case_get("fc_steps", None)
        if value is None:
            return [int(num_steps)]
        if isinstance(value, (int, float, str)):
            value = [value]
        return [int(v) for v in value]

    @property
    def fc_q_source(self) -> str:
        value = str(self._case_get("fc_q_source", "online"))
        if value not in FC_Q_SOURCES:
            raise ValueError(f"fc_q_source must be one of {FC_Q_SOURCES}")
        return value

    @property
    def fc_group_size(self) -> int | None:
        value = self._case_get("fc_group_size", None)
        return None if value in (None, "", "null") else int(value)

    def _fc_modes(self, num_steps: int) -> list[tuple[str, str, dict]]:
        """``(variant, likelihood_mode, spec)`` of the corrected Ours rows."""
        if not self.fc_enabled:
            return []
        q, group = self.fc_q_source, self.fc_group_size
        bases = self._case_get("fc_variants", None)
        bases = (
            [name for name, _ in _OURS_VARIANTS] if bases is None
            else [bases] if isinstance(bases, str) else [str(b) for b in bases]
        )
        unknown = set(bases) - {name for name, _ in _OURS_VARIANTS}
        if unknown:
            raise ValueError(f"fc_variants: unknown Ours variants {sorted(unknown)}")
        return [
            (fc_variant(name, steps, q, group), mode,
             {"base": name, "steps": steps, "q_source": q, "group": group})
            for steps in self.fc_steps(num_steps)
            for name, mode in _OURS_VARIANTS
            if name in bases
        ]

    def _make_correction(self, spec: dict) -> GroupedForecastCorrection:
        """One correction per rollout; H and R come from each cycle's likelihood."""
        return GroupedForecastCorrection(
            ForecastCorrectionConfig(
                enabled=True,
                forecast_num_steps=int(spec["steps"]),
                q_source=str(spec["q_source"]),
                # Filled at the first cycle (it needs the posterior); see _ours_cycle.
                residual_bank=None,
            ),
            group=spec["group"],
        )

    def _residual_bank(self, posterior, correction, num_steps, d, seed_key) -> None:
        """Build the ``q_source="bank"`` residuals once per rollout.

        Q does not depend on the state here, so the states are ``N(0, I)`` draws.
        The bank uses the forecast's own steps and stepper (the consistency the
        note asks for) and a forked RNG, so the global stream is untouched. Its
        cost is not in ``NFE``: it is an offline artefact.
        """
        cfg = correction.config
        if cfg.q_source != "bank" or cfg.residual_bank is not None:
            return
        size = int(self._case_get("fc_bank_size", 4096))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(derive_seed("multistep", "fc_bank", *seed_key))
            states = torch.randn(size, 1, 1, d, 1)
            cfg.residual_bank = build_residual_bank(
                posterior, states,
                num_steps=int(cfg.forecast_num_steps or num_steps),
                stepper=cfg.forecast_stepper or euler_maruyama_step,
                batch_size=size,
            )

    def _case_get(self, key: str, default: object) -> object:
        cli = self._cfg_get(key, None)
        if cli is not None:
            return cli
        case = getattr(self.config, "case", None)
        value = getattr(case, key, None) if case is not None else None
        return default if value is None else value

    @property
    def num_cycles(self) -> int:
        return int(self._case_get("num_cycles", 20))

    @property
    def scenario_label(self) -> str:
        return str(self._case_get("scenario_name", "multistep:paper"))

    def system(self) -> LinearGaussianSSM:
        from omegaconf import OmegaConf

        spec = self._case_get("system", None) or {}
        if OmegaConf.is_config(spec):
            spec = OmegaConf.to_container(spec, resolve=True)
        system = LinearGaussianSSM.from_spec(self._dim(), **spec)
        # The label is all that identifies a system in the results, so a custom
        # system must not pool with the paper's. New systems get their own YAML.
        if self.scenario_label == "multistep:paper" and not (
            system.is_paper_prior
            and system.isotropic_q == 1.0
            and system.is_identity_obs
            and system.isotropic_r == 1.0
        ):
            raise ValueError(
                "scenario 'multistep:paper' is reserved for A = Q = H = R = I; "
                "give this system its own scenario_name"
            )
        return system

    # -- truth, observations and references, shared across methods ---------- #

    def trajectory(self, seed: int, n: int) -> tuple[torch.Tensor, list[Cycle]]:
        """``(x0, cycles)`` for one seed.

        The truth stream continues the single-step generator (``x0``, ``w``, ``e``,
        then ``w``, ``e``, ... ) and the reference stream its ``ref`` generator, so
        cycle 1 is the single-step problem.
        """
        if (seed, n) in self._cycles:
            return self._cycles[seed, n]
        system = self.system()
        gx = torch.Generator().manual_seed(derive_seed("analytical", "truth", seed))
        gr = torch.Generator().manual_seed(derive_seed("analytical", "ref", seed))
        gc = torch.Generator().manual_seed(derive_seed("multistep", "ref_chain", seed))

        x0 = torch.randn(system.d, generator=gx)
        x_true = x0
        kalman_mean = chain_mean = x0.to(torch.float64)
        kalman_cov = chain_cov = torch.zeros(
            system.d, system.d, dtype=torch.float64
        )
        cycles: list[Cycle] = []
        for _ in range(self.num_cycles):
            x_true = system.propagate(x_true, gx)
            y = system.observe(x_true, gx)
            kalman_mean, kalman_cov = system.kalman_step(kalman_mean, kalman_cov, y)
            chain_mean, chain_cov = system.member_chain_step(chain_mean, chain_cov, y)
            cycles.append(
                Cycle(
                    y=y,
                    kalman_mean=kalman_mean,
                    kalman_cov=kalman_cov,
                    chain_mean=chain_mean,
                    chain_cov=chain_cov,
                    ref_kalman=gaussian_samples(kalman_mean, kalman_cov, n, gr).numpy(),
                    ref_chain=gaussian_samples(chain_mean, chain_cov, n, gc).numpy(),
                )
            )
        self._cycles[seed, n] = (x0, cycles)
        return self._cycles[seed, n]

    # -- model builders: the paper's classes whenever they apply ------------ #

    def _si_model(self, system: LinearGaussianSSM):
        if system.is_paper_prior:
            return build_si_model(d=system.d, g0=_G0, prior_var=system.isotropic_q)
        return build_linear_si_model(system, g0=_G0)

    def _fm_model(self, system: LinearGaussianSSM):
        if system.is_paper_prior:
            return build_fm_model(d=system.d, prior_var=system.isotropic_q)
        return build_linear_fm_model(system)

    def _dm_model(self, system: LinearGaussianSSM, cycle: int):
        """The driver's diffusion prior at cycle 1, the corrected one afterwards."""
        fm_model = self._fm_model(system)
        model_class = DenoiseDiffusionModel if cycle == 0 else FullFirstStepDiffusionModel
        return model_class.from_flow_matching(
            fm_model,
            diffusion_term=endpoint_vanishing_diffusion(fm_model.interpolation, scale=_G0),
        )

    @staticmethod
    def _obs(system: LinearGaussianSSM, y: torch.Tensor):
        """``(operator, variance, y)`` in the scalar-``R`` form the likelihoods take.

        Every likelihood accepts only ``R = r I``, so a general ``R`` is whitened:
        ``R^{-1/2} y = R^{-1/2} H x + e'``, ``e' ~ N(0, I)``. That is the same
        posterior for every method that uses ``R``. D-Flow and FIG never use it, so
        for them whitening changes the problem they solve instead.
        """
        r = system.isotropic_r
        if r is not None:
            operator = (
                build_identity_obs_operator(d=system.d)
                if system.is_identity_obs
                else DenseObservationOperator(system.H)
            )
            return operator, r, y.reshape(1, -1)
        chol = torch.linalg.cholesky(system.R)
        whitened_H = torch.linalg.solve_triangular(chol, system.H, upper=False)
        whitened_y = torch.linalg.solve_triangular(
            chol, y.to(chol.dtype).unsqueeze(-1), upper=False
        ).squeeze(-1)
        return (
            DenseObservationOperator(whitened_H),
            1.0,
            whitened_y.to(y.dtype).reshape(1, -1),
        )

    # -- one assimilation cycle --------------------------------------------- #

    def _ours_cycle(
        self,
        method: Method,
        system: LinearGaussianSSM,
        x: torch.Tensor,
        y: torch.Tensor,
        likelihood_mode: str | None,
        num_steps: int,
        surge: bool = False,
        correction: ForecastCorrection | None = None,
        bank_key: tuple = (),
    ) -> torch.Tensor:
        """``driver.draw_interpolant_posterior`` with one anchor per member.

        With ``surge`` the same likelihood is the guidance of a SURGE posterior, the
        way FlowDAS and SDA are wrapped: the proposal is this sampler's guided SDE,
        plus SMC weights and resampling across members. ``likelihood_mode=None``
        takes the guidance out, leaving the SMC on the prior. ``correction`` is set
        on the posterior, so ``BasePosterior.sample`` applies it after the draw.
        """
        sampler = _OURS_SAMPLER[method]
        block = self._method_cfg(method).get("likelihood_model", {})

        def hp(key: str, fallback: object) -> object:
            cli = self._cfg_get(key, None)
            return cli if cli is not None else _cfg_hparam(block, key, num_steps, fallback)

        cap = hp("residual_step_cap", None)
        stabilisation = dict(
            jacobian_damping=float(hp("jacobian_damping", 1.0)),
            sigma_bar_eig_floor=bool(hp("sigma_bar_eig_floor", False)),
            isotropic_front_factor=bool(hp("isotropic_front_factor", False)),
            residual_step_cap=float(cap) if cap is not None else None,
        )
        operator, variance, y_obs = self._obs(system, y)
        n, d = x.shape
        field_history = x.reshape(n, 1, 1, d, 1).clone()
        if sampler == "si_sde":
            model = self._si_model(system)
            likelihood = NullGuidance() if likelihood_mode is None else InterpolantGaussianLikelihood(
                model=model, obs_operator=operator, variance=variance,
                model_class="si", likelihood_mode=likelihood_mode, **stabilisation,
            )
            posterior = (
                SurgeStochasticInterpolantPosterior(
                    model=model, obs_operator=operator, variance=variance,
                    likelihood_model=likelihood,
                    ess_threshold=self._surge_ess_threshold(Method.SURGE_FLOWDAS, num_steps),
                    diffusion_term=None,
                )
                if surge
                else StochasticInterpolantPosterior(
                    model=model, likelihood_model=likelihood
                )
            )
            base = x.reshape(n, 1, 1, d).clone()
        else:
            model = self._fm_model(system)
            likelihood = NullGuidance() if likelihood_mode is None else InterpolantGaussianLikelihood(
                model=model, obs_operator=operator, variance=variance,
                model_class="fm", likelihood_mode=likelihood_mode, **stabilisation,
            )
            diffusion = (
                endpoint_vanishing_diffusion(model.interpolation, scale=_G0)
                if sampler == "dm_sde" else None
            )
            if surge:
                if diffusion is None:
                    raise ValueError("SURGE needs an SDE; FM-ODE has no path measure")
                # The prior in SURGE's weights is the reverse SDE, so the model is
                # the diffusion lift. Its first step must equal Ours' first step,
                # which keeps the velocity, at every cycle.
                posterior = OursSurgeDiffusionPosterior(
                    model=self._dm_model(system, 1), obs_operator=operator,
                    variance=variance, likelihood_model=likelihood,
                    ess_threshold=self._surge_ess_threshold(Method.SURGE_SDA, num_steps),
                    diffusion_term=diffusion,
                )
            else:
                posterior = FlowMatchingPosterior(
                    model=model, likelihood_model=likelihood, diffusion_term=diffusion,
                    drift_time_shift=float(hp("drift_time_shift", 0.0)),
                )
            base = None
        if correction is not None:
            if surge:
                raise ValueError("the correction is not combined with SURGE")
            self._residual_bank(posterior, correction, num_steps, d, bank_key)
            posterior.forecast_correction = correction
        with torch.no_grad():
            out = posterior.sample(
                base=base, batch_size=n, num_steps=num_steps,
                field_history=field_history, observations=y_obs,
            )
        return out.reshape(n, d)

    def _surge_ess_threshold(self, config: Method, num_steps: int) -> float:
        """The existing SURGE row's resampling threshold; nothing is tuned here."""
        block = self._method_cfg(config).get("posterior_model", {})
        return float(_cfg_hparam(block, "ess_threshold", num_steps, 0.5))

    def _baseline_cycle(
        self,
        method: Method,
        system: LinearGaussianSSM,
        x: torch.Tensor,
        y: torch.Tensor,
        num_steps: int,
        generator: torch.Generator,
        cycle: int,
    ) -> tuple[torch.Tensor, int]:
        """``driver._sample_and_count`` with one anchor per member."""
        if method == Method.ENKF:
            return enkf_step(system, x, y, generator), 1
        if method == Method.PARTICLE_FILTER:
            return particle_filter_step(system, x, y, generator), 1

        M = num_steps
        n, d = x.shape
        field_history = x.reshape(n, 1, 1, d, 1).clone()
        base_si = x.reshape(n, 1, 1, d).clone()
        operator, variance, y_obs = self._obs(system, y)
        config = self._method_cfg(method)
        lik = config.get("likelihood_model", {})
        post = config.get("posterior_model", {})

        def hp(block, key, fallback):
            return _cfg_hparam(block, key, M, fallback)

        if method in (Method.FLOWDAS, Method.SURGE_FLOWDAS):
            n_mc = int(hp(lik, "num_mc_samples", 25))
            model = self._si_model(system)
            likelihood = FlowdasGaussianLikelihood(
                model=model, obs_operator=operator, variance=variance,
                num_mc_samples=n_mc, guidance_scale=float(hp(lik, "guidance_scale", 1.0)),
            )
            posterior = (
                StochasticInterpolantPosterior(model=model, likelihood_model=likelihood)
                if method == Method.FLOWDAS
                else SurgeStochasticInterpolantPosterior(
                    model=model, obs_operator=operator, variance=variance,
                    likelihood_model=likelihood,
                    ess_threshold=float(hp(post, "ess_threshold", 0.5)),
                    diffusion_term=None,
                )
            )
            base, nfe = base_si, M * (1 + n_mc)
        elif method == Method.D_FLOW_SGLD:
            step_size = float(hp(lik, "step_size", 0.005))
            noise_scale = float(hp(lik, "noise_scale", 1e-3))
            lambda_reg = float(hp(lik, "lambda_reg", 1e-4))
            model = self._fm_model(system)
            likelihood = DFlowSGLDLikelihood(
                model=model, obs_operator=operator, variance=variance,
                num_optim_steps=M, step_size=step_size,
                noise_scale=noise_scale, lambda_reg=lambda_reg,
            )
            posterior = DFlowPosterior(
                model=model, likelihood_model=likelihood, num_optim_steps=M,
                step_size=step_size, noise_scale=noise_scale,
                lambda_reg=lambda_reg, variance=variance,
            )
            base, nfe = None, M * 6
        elif method in (Method.SDA, Method.SURGE_SDA):
            model = self._dm_model(system, cycle)
            likelihood = SDALikelihood(
                model=model, obs_operator=operator, variance=variance,
                model_class="fm", gamma_sda=float(hp(lik, "gamma_sda", 1e-2)),
            )
            posterior = (
                DiffusionPosterior(model=model, likelihood_model=likelihood)
                if method == Method.SDA
                else SurgeFlowMatchingPosterior(
                    model=model, obs_operator=operator, variance=variance,
                    likelihood_model=likelihood,
                    ess_threshold=float(hp(post, "ess_threshold", 0.5)),
                    diffusion_term=None,
                )
            )
            base, nfe = None, M
        elif method == Method.GUIDED_FM_FIG:
            fig_k = int(hp(lik, "guidance_steps", 1))
            # FIG has no MC-size field of its own; the driver reuses FlowDAS's J.
            n_mc = int(hp(
                self._method_cfg(Method.FLOWDAS).get("likelihood_model", {}),
                "num_mc_samples", 25,
            ))
            model = self._fm_model(system)
            likelihood = FIGGaussianLikelihood(
                model=model, obs_operator=operator, variance=variance,
                ensemble_size=n_mc, guidance_steps=fig_k,
                guidance_scale=float(hp(lik, "guidance_scale", 10.0)),
                interpolant_noise=float(hp(lik, "interpolant_noise", 0.0)),
            )
            posterior = FlowMatchingPosterior(
                model=model, likelihood_model=likelihood, diffusion_term=None
            )
            base, nfe = None, M * (1 + fig_k)
        else:
            raise ValueError(f"unhandled multistep method {method!r}")

        with torch.no_grad():
            out = posterior.sample(
                base=base, batch_size=n, num_steps=M,
                field_history=field_history, observations=y_obs,
            )
        return out.reshape(n, d), nfe

    # -- the cycle loop ------------------------------------------------------ #

    def _rollout(
        self,
        ctx: RunContext,
        variant: str | None,
        draw: Callable[[torch.Tensor, torch.Tensor, int], tuple[torch.Tensor, int]],
    ) -> Iterable[ResultRecord]:
        """Run the ``K`` cycles of one (method, variant, seed); one ensemble feeds back.

        A non-finite ensemble or a sampler failure ends the rollout: the remaining
        cycles report NaN and ``n_traj`` drops after aggregation. A NaN KL alone is
        not divergence; a singular but finite ensemble (FIG) keeps going.
        """
        n = self._n_eval()
        x0, cycles = self.trajectory(ctx.seed, n)
        x = x0.unsqueeze(0).repeat(n, 1)
        curves: dict[str, list[float]] = {metric: [] for metric in METRICS}
        seconds: list[float] = []
        nfe = 0
        diverged = False
        for k, cycle in enumerate(cycles):
            if not diverged:
                start = time.perf_counter()
                try:
                    x, nfe = draw(x, cycle.y, k)
                    diverged = not bool(torch.isfinite(x).all())
                except (torch.linalg.LinAlgError, AssertionError) as exc:
                    # The FM posterior asserts a finite drift, and a NaN state fails
                    # linear solves: one bad cell must not lose the whole seed.
                    logger.warning("[multistep] sampler failed: %s", exc)
                    diverged = True
                seconds.append(time.perf_counter() - start)
                if diverged:
                    logger.warning(
                        "[multistep] %s %s seed %d diverged at cycle %d",
                        ctx.method.value, variant, ctx.seed, k + 1,
                    )
            scores = (
                dict.fromkeys(METRICS, float("nan"))
                if diverged else score_ensemble(cycle, x.numpy())
            )
            for metric, value in scores.items():
                curves[metric].append(value)
        yield from self._emit(
            ctx.method.value, ctx, variant, curves, nfe, seconds, diverged=diverged
        )

    def _emit(
        self,
        method: str,
        ctx: RunContext,
        variant: str | None,
        curves: dict[str, list[float]],
        nfe: float,
        seconds: list[float],
        diverged: bool = False,
    ) -> Iterable[ResultRecord]:
        """Per-cycle rows to the per-step file; the mean over cycles as tidy rows.

        The cycle mean skips NaN cycles, so one singular cycle does not remove the
        seed from the across-seed mean; an all-NaN curve stays NaN. A diverged
        rollout's cycle mean is NaN: averaging only its cycles before divergence
        would count it as a full seed with a flattering score.
        """
        n = self._n_eval()
        mean_seconds = float(np.mean(seconds)) if seconds else None
        self._per_step.extend(
            per_step_rows(
                case=self.case.value, method=method, scenario=self.scenario_label,
                variant=variant, E=n, M=ctx.num_steps, seed=ctx.seed,
                test_index=ctx.seed, nfe=float(nfe), seconds=mean_seconds,
                per_step=curves,
            )
        )
        for metric, values in curves.items():
            yield ResultRecord(
                case=self.case.value, method=method, scenario=self.scenario_label,
                metric=metric,
                value=(
                    float(np.nanmean(values))
                    if not diverged and np.isfinite(values).any() else float("nan")
                ),
                E=n, M=ctx.num_steps,
                seed=ctx.seed, nfe=float(nfe), seconds=mean_seconds, variant=variant,
            )

    def _cycle_seed(self, method: Method, variant: str | None, seed: int, k: int) -> int:
        return derive_seed("multistep", method.value, variant, seed, k)

    def evaluate(self, ctx: RunContext) -> Iterable[ResultRecord]:
        system = self.system()
        M = ctx.num_steps
        group = self.fc_group_size
        if ctx.method in _OURS_SAMPLER:
            # (variant, likelihood mode, correction spec or None)
            if self.surge_only:
                modes = [(SURGE_PREFIX + name, mode, None) for name, mode in _OURS_VARIANTS]
                modes.append((SURGE_UNGUIDED, None, None))
            elif self.fc_only:
                modes = self._fc_modes(M)
            elif self.group_only:
                modes = [(f"{ADAPTED_VARIANT}_{group_tag(group)}", "inflated_shared", None)]
            else:
                modes = [(name, mode, None) for name, mode in _OURS_VARIANTS]
                if bool(self._case_get("adapted_variant", True)):
                    modes.append((ADAPTED_VARIANT, "inflated_shared", None))
                modes.extend(self._fc_modes(M))
            for variant, mode, spec in modes:
                # Cycle 1 shares the plain row's seed, so its noise is paired.
                first_label = (
                    "shared" if variant.startswith(ADAPTED_VARIANT)
                    else spec["base"] if spec is not None
                    else variant.removeprefix(SURGE_PREFIX)
                )
                resampler = torch.Generator().manual_seed(
                    derive_seed("multistep", "adapted", ctx.method.value, ctx.seed)
                )
                correction = self._make_correction(spec) if spec is not None else None
                # Steps per member per cycle: the guided draw, plus the forecast and
                # (online Q_hat) its twin, each one prior pass per step.
                nfe = M if spec is None else M + int(spec["steps"]) * (
                    2 if spec["q_source"] == "online" else 1
                )
                resample_group = group if variant != ADAPTED_VARIANT else None

                def draw(x, y, k, _variant=variant, _mode=mode, _label=first_label,
                         _corr=correction, _nfe=nfe, _group=resample_group):
                    if k == 0:  # the single-step seed
                        torch.manual_seed(
                            derive_seed(
                                "analytical", f"{ctx.method.value}:{_label}", ctx.seed
                            )
                        )
                    else:
                        torch.manual_seed(
                            self._cycle_seed(ctx.method, _variant, ctx.seed, k)
                        )
                        if _variant.startswith(ADAPTED_VARIANT):
                            x = _per_group(
                                lambda xs: adapted_resample(system, xs, y, resampler),
                                x, _group,
                            )
                    if _variant.startswith(SURGE_PREFIX):
                        # Per step: the prior at x_k, then the reward at x_{k+1}.
                        return self._ours_cycle(
                            ctx.method, system, x, y, _mode, M, surge=True
                        ), 2 * M
                    return self._ours_cycle(
                        ctx.method, system, x, y, _mode, M, correction=_corr,
                        bank_key=(ctx.method.value, _variant, ctx.seed),
                    ), _nfe

                yield from self._rollout(ctx, variant, draw)
            return

        generator = torch.Generator().manual_seed(
            derive_seed("analytical", ctx.method.value, ctx.seed)
        )
        variant = group_tag(group) if self.group_only else None

        def draw(x, y, k):
            # Cycle 1 runs on the state left by the runner's seed_everything(seed).
            if k > 0:
                torch.manual_seed(self._cycle_seed(ctx.method, variant, ctx.seed, k))
            if variant is not None:  # EnKF on independent groups of members
                return _per_group(
                    lambda xs: enkf_step(system, xs, y, generator), x, group
                ), 1
            return self._baseline_cycle(ctx.method, system, x, y, M, generator, k)

        yield from self._rollout(ctx, variant, draw)

    def _reference_rows(self, seed: int) -> Iterable[ResultRecord]:
        """Score ensembles drawn straight from the two closed-form laws."""
        n = self._n_eval()
        ctx = self.make_context(ANALYTICAL_METHODS[0], Scenario.ANALYTICAL, seed)
        _, cycles = self.trajectory(seed, n)
        for label, mean_key, cov_key in (
            (REFERENCE_KALMAN, "kalman_mean", "kalman_cov"),
            (REFERENCE_MEMBER_CHAIN, "chain_mean", "chain_cov"),
        ):
            generator = torch.Generator().manual_seed(
                derive_seed("multistep", label, seed)
            )
            curves: dict[str, list[float]] = {metric: [] for metric in METRICS}
            for cycle in cycles:
                samples = gaussian_samples(
                    getattr(cycle, mean_key), getattr(cycle, cov_key), n, generator
                )
                for metric, value in score_ensemble(cycle, samples.numpy()).items():
                    curves[metric].append(value)
            yield from self._emit(label, ctx, None, curves, 0.0, [])

    def run(self, *, aggregate: bool = True) -> list[ResultRecord]:
        # Seeds are told apart by the filename at aggregation (traj<seed>).
        if self._cfg_get("per_step_file", None) is not None and len(self.seeds) > 1:
            raise ValueError("per_step_file holds one seed; run one seed per file")
        raw = ExperimentRunner.run(self, aggregate=False)
        if not self.added_cell:  # the base cells already hold them
            for seed in self.seeds:
                raw.extend(self._reference_rows(seed))
        return aggregate_over_seeds(raw) if aggregate else raw

    def write(self, records: Iterable[ResultRecord], path: str | Path) -> Path:
        """The tidy file, plus the per-cycle curves when ``per_step_file`` is set."""
        out = super().write(records, path)
        per_step_file = self._cfg_get("per_step_file", None)
        if per_step_file is not None:
            append_per_step(output_path(str(per_step_file)), self._per_step)
            self._per_step = []
        return out

    def run_ablation(self, *, aggregate: bool = True) -> list[ResultRecord]:
        raise NotImplementedError(
            "the covariance ablation is a single-step study; use case=analytical"
        )


__all__ = [
    "MultistepRunner",
    "METRICS",
    "ADAPTED_VARIANT",
    "SURGE_PREFIX",
    "SURGE_UNGUIDED",
    "SURGE_WRAPPED",
    "REFERENCE_KALMAN",
    "REFERENCE_MEMBER_CHAIN",
    "FC_TAG",
    "GroupedForecastCorrection",
    "fc_variant",
    "group_tag",
    "parse_fc_variant",
    "sliced_distances",
    "kl_to_moments",
    "score_ensemble",
]
