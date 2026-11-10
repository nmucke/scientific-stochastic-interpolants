"""Lorenz 96 comparison against a particle-filter reference.

Every method is scored the same way on the same sequences. A sequence fixes truth,
observations and the prior of the initial state; a seed fixes the members drawn from
that prior and the method's own randomness.

    python cases/lorenz96/driver.py reference --split validation
    python cases/lorenz96/driver.py tune --method enkf --grid '{"inflation": [1.0, 1.1]}'
    python cases/lorenz96/driver.py reference --split test
    python cases/lorenz96/driver.py cell --method si_sde --seed 0 --num-steps 50
    python cases/lorenz96/driver.py cell --method si_sde_fc --seed 0 --tag g0.8 \
        --set correction.gain_scale=0.8

The cells and their tidy labels are ``lorenz96_lineup.ROWS``. Cells ending in
``_fc`` are the generative samplers (and the exact-kernel oracle) with the
forecast-residual correction of ``FORECAST_CORRECTION.md``; ``prior_enkf`` and
``prior_enkf_fm`` forecast with the unguided trained prior and update with the
localised EnKF. ``run_lorenz96.sh`` runs all of it. Test sequences are never used to
choose a setting: ``tune`` reads the validation split only, at M = ``HEADLINE_M``.

``cell --set`` changes a cell's settings and so needs ``--tag``: the tag goes into
every output path (``<cell>-<tag>``) and into the tidy ``variant``
(``<variant>_<tag>``), so a sweep never overwrites, or is aggregated as, the
default cell.

``--obs {linear,arctan}`` (before the subcommand, default ``linear``) selects the
observation operator (``observations:`` in ``configs/case/lorenz96.yaml``) and with
it the settings block (``methods:`` or ``methods_arctan:``). Every output of a
problem lives under ``results/lorenz96/<obs>/``, so the two sweeps never mix:

    python cases/lorenz96/driver.py --obs arctan reference --split validation
    python cases/lorenz96/driver.py --obs arctan cell --method enkf --seed 0
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import multiprocessing
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[1]))

from cases.lorenz96.pf_gate import reference_proposal, run_pf  # noqa: E402
from cases.lorenz96._l96_smc import SURGE_CELLS, pf_smc_record  # noqa: E402
from cases.lorenz96._l96_divergence import energy_distance, knn_kl  # noqa: E402
from cases.lorenz96._l96_system import (  # noqa: E402
    CASE_CONFIG,
    OBS_OPERATORS,
    Problem,
    Sequence,
    case_config,
    configure,
    draw_prior_ensemble,
    make_sequence,
    open_csv_for_append,
    problem_of,
    resolve_device,
)
from common.paths import REPO_ROOT, RESULTS_ROOT  # noqa: E402
from common.per_step_io import append_per_step, per_step_rows  # noqa: E402
from common.runner import ExperimentRunner, RunContext  # noqa: E402
from common.seeding import derive_seed  # noqa: E402
from lorenz96_lineup import (  # noqa: E402
    ENERGY_DISTANCE,
    HEADLINE_M,
    INHERITS,
    KL_KNN,
    M_VALUES,
    ROWS,
    SCENARIO_OF,
    SMC_METRICS,
    parse_assignments,
    set_dotted,
    settings_of,
)
from results_schema import (  # noqa: E402
    Case,
    Method,
    Metric,
    ResultRecord,
    ResultsWriter,
    Scenario,
)
from scisi.metrics.calibration import crps  # noqa: E402
from scisi.metrics.distributional import sliced_wasserstein_w2  # noqa: E402
from scisi.particle_filter.bootstrap import (  # noqa: E402
    BootstrapParticleFilter,
    ImportanceOneStepOracle,
    OneStepPosteriorOracle,
)
from scisi.particle_filter.enkf import (  # noqa: E402
    EnsembleKalmanFilter,
    periodic_taper,
)

RESULTS = RESULTS_ROOT / "lorenz96"  # the case; every output is under RESULTS / <obs>
SEED_LABEL = "lorenz96-evaluation"
DATA_PATH = REPO_ROOT / "data/lorenz96/data.npz"
DIVERGED_ABS = 100.0  # about 25 climatological standard deviations

# name -> (tidy method label, variant), from lorenz96_lineup.ROWS.
CELLS = {row.cell: (Method(row.method), row.variant or None) for row in ROWS}
# Corrected cell -> the plain cell whose sampler settings and seed stream it reuses.
CORRECTED = {name: INHERITS[name] for name in ("si_sde_fc", "fm_ode_fc", "dm_sde_fc")}
# Trained-prior forecast + EnKF update -> the sampler whose prior forecasts.
PRIOR_ENKF = {"prior_enkf": "si_sde", "prior_enkf_fm": "fm_ode"}
FC_VARIANT = "forecast_correction"
# The plain samplers; their corrected cells are in CORRECTED.
GENERATIVE = {"si_sde": Method.OURS_SI_SDE, "fm_ode": Method.OURS_FM_ODE, "dm_sde": Method.OURS_DM_SDE}
# Cells that take no settings: an override or a tuning grid would be ignored.
NO_SETTINGS = ("oracle", "pf_matched", "pf_subsample", "pf_reference")
# Cells whose step-0 members are not ``initial_ensemble``: their first innovation is
# left out (``pf_subsample`` draws from the reference posterior at steps 1..T only).
NO_INITIAL_ENSEMBLE = ("pf_subsample",)
TUNABLE = tuple(name for name in CELLS if name not in NO_SETTINGS)
# (sequence, step) entries with a non-finite core score; in practice fewer than two
# members left.
DIVERGED_STEPS = "diverged_steps"
# Every generative method runs exactly M = HEADLINE_M SDE/ODE steps per assimilation
# step (D-Flow: SGLD steps) in tuning, fixed by design and not tuned; the final grid
# runs them at every M of M_VALUES with those picks. Both are checked by check_m.
FIXED_NUM_STEPS = HEADLINE_M
# Per-sequence scalars (not per step) written next to the per-step metrics.
INNOVATION_LAG1 = "innovation_lag1"
INNOVATION_MSQ = "innovation_msq"
# Per-member ESS of the importance-sampling oracle (arctan `oracle` / `oracle_fc`):
# per step the mean and the minimum over members (and sequences).
ORACLE_ESS_MEAN = "oracle_ess_mean"
ORACLE_ESS_MIN = "oracle_ess_min"
# Tidy scenario label of each observation operator.
SCENARIOS = {obs: Scenario(value) for obs, value in SCENARIO_OF.items()}


def config_of(args):
    """The case config of a command's ``--obs``."""
    return case_config(getattr(args, "obs", "linear"))


def results_dir(obs: str) -> Path:
    """Root of every output of the ``obs`` problem: ``results/lorenz96/<obs>``."""
    return RESULTS / obs


def method_settings(cfg, name: str) -> dict:
    """Settings of a cell; a cell of ``lorenz96_lineup.INHERITS`` starts from its
    parent's (``lorenz96_lineup.settings_of``)."""
    return settings_of(OmegaConf.to_container(cfg.methods), name)


def apply_overrides(settings: dict, assignments) -> dict:
    """Apply ``key=value`` (dotted keys, JSON values) to a copy of ``settings``.

    ``correction.localisation=4`` sets ``settings["correction"]["localisation"]``.
    Values that are not JSON (``dps_jacobian_free``) are kept as strings.
    """
    return set_dotted(settings, parse_assignments(assignments))


def load_sequences(problem: Problem, split: str, count: int) -> list[Sequence]:
    return [
        make_sequence(problem, derive_seed(SEED_LABEL, "sequence", split, index))
        for index in range(count)
    ]


def split_size(cfg, split: str) -> int:
    return int(cfg.evaluation[f"{split}_sequences"])


def initial_ensemble(problem, sequence, members, split, index, seed) -> torch.Tensor:
    """Members depend on (sequence, seed, size) only, so methods share them."""
    return draw_prior_ensemble(
        problem,
        sequence,
        members,
        derive_seed(SEED_LABEL, "initial", split, index, seed),
    )


# --------------------------------------------------------------------------- #
# Particle-filter reference
# --------------------------------------------------------------------------- #


def reference_path(split: str, index: int, replicate: int, obs: str = "linear") -> Path:
    return (
        results_dir(obs) / "reference" / split / f"sequence{index}_replicate{replicate}.pt"
    )


def _reference_task(job) -> float:
    problem, sequence, split, index, replicate, particles = job
    torch.set_num_threads(1)
    # The seed does not depend on the operator: the arctan reference reuses the
    # linear one's random streams (its proposal, and so its draws, differ).
    seed = derive_seed(SEED_LABEL, "reference", split, index, replicate)
    start = time.time()
    # The gate's own filter, so its validity carries over by construction.
    run = run_pf(problem, sequence, particles, seed)
    seconds = time.time() - start
    path = reference_path(split, index, replicate, problem.obs_label)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            mean=run.mean,
            variance=run.variance,
            samples=run.samples,
            ess=run.ess,
            log_evidence=run.log_evidence,
            num_particles=run.num_particles,
            seconds=seconds,
            problem=problem.to_dict(),
        ),
        path,
    )
    return seconds


def cmd_reference(args) -> None:
    cfg = config_of(args)
    problem = problem_of(cfg)
    sequences = load_sequences(problem, args.split, split_size(cfg, args.split))
    replicates = int(cfg.evaluation.reference_replicates)
    if replicates != 2:
        raise ValueError(
            "evaluation.reference_replicates must be 2: pf_reference scores replicate "
            "1 against 0, and pf_subsample draws from replicate 1."
        )
    jobs = [
        (problem, seq, args.split, index, rep, int(cfg.evaluation.reference_particles))
        for index, seq in enumerate(sequences)
        for rep in range(replicates)
        if not reference_path(args.split, index, rep, problem.obs_label).exists()
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        for job, seconds in zip(jobs, pool.map(_reference_task, jobs)):
            print(f"{args.split} {job[3]} replicate {job[4]}: {seconds:.0f}s", flush=True)


def load_reference(split: str, index: int, replicate: int, problem: Problem) -> dict:
    """Load a stored reference, refusing one made for another system.

    The stored problem includes the observation operator (a linear one has no
    ``observation`` key), so a reference of the other operator is refused even
    if it were copied into this problem's directory.
    """
    path = reference_path(split, index, replicate, problem.obs_label)
    reference = torch.load(path, weights_only=False)
    if reference["problem"] != problem.to_dict():
        raise ValueError(f"Stale reference for {split} {index}: rerun `reference`.")
    return reference


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def score(mean, variance, samples, reference, truth) -> dict[str, list[float]]:
    """Per-step metrics of one posterior approximation.

    Args:
        mean, variance: ``[T, d]`` moments of the approximation.
        samples: ``T`` tensors ``[E_t, d]`` of draws from it.
        reference: Dictionary of the particle-filter reference.
        truth: ``[T, d]`` true states.
    """
    ref_mean = reference["mean"].double()
    ref_var = reference["variance"].double()
    mean, variance = mean.double(), variance.double()
    usable = [draws.double() if draws.shape[0] > 1 else None for draws in samples]
    nan = float("nan")
    return {
        Metric.MEAN_ERROR_VS_PF.value: ((mean - ref_mean) ** 2 / ref_var)
        .mean(-1)
        .sqrt()
        .tolist(),
        Metric.VAR_ERROR_VS_PF.value: (variance / ref_var)
        .log()
        .pow(2)
        .mean(-1)
        .sqrt()
        .tolist(),
        Metric.SLICED_W2.value: [
            nan
            if draws is None
            else sliced_wasserstein_w2(draws, reference["samples"][t].double())
            for t, draws in enumerate(usable)
        ],
        # Nonparametric distances to the PF draws (_l96_divergence): KL(PF || ensemble)
        # by 1-NN (biased at E = 64; pf_subsample gives the floor) and energy distance.
        KL_KNN: [
            nan if draws is None else knn_kl(reference["samples"][t].double(), draws)
            for t, draws in enumerate(usable)
        ],
        ENERGY_DISTANCE: [
            nan if draws is None else energy_distance(draws, reference["samples"][t].double())
            for t, draws in enumerate(usable)
        ],
        Metric.RMSE.value: (mean - truth).pow(2).mean(-1).sqrt().tolist(),
        Metric.SPREAD.value: variance.sqrt().mean(-1).tolist(),
        Metric.CRPS.value: [
            nan if draws is None else float(crps(draws, truth[t]))
            for t, draws in enumerate(usable)
        ],
    }


def score_ensemble(ensemble, reference, truth) -> dict[str, list[float]]:
    """Score an ensemble ``[T, E, d]``.

    The same rule holds for every method: a member that is non-finite or leaves any
    plausible range is dropped at that step and counted in ``diverged_fraction``, so
    failures stay visible and do not contaminate the other steps.
    """
    alive = (torch.isfinite(ensemble) & (ensemble.abs() < DIVERGED_ABS)).all(-1)
    samples = [members[keep] for members, keep in zip(ensemble, alive)]
    mean = torch.stack([draws.mean(0) for draws in samples])
    variance = torch.stack([draws.var(0) for draws in samples])
    out = score(mean, variance, samples, reference, truth)
    out[Metric.DIVERGED_FRACTION.value] = (1.0 - alive.double().mean(1)).tolist()
    return out


def score_output(ensembles, sequences, split, problem) -> list[dict[str, list[float]]]:
    """Score ensembles ``[S, T, E, d]`` of ``sequences`` against replicate 0."""
    return [
        score_ensemble(ens, load_reference(split, index, 0, problem), seq.truth[1:])
        for index, (ens, seq) in enumerate(zip(ensembles, sequences))
    ]


def summarise(scores) -> dict[str, list[float]]:
    """Per metric, the finite mean over steps of each sequence."""
    return {
        metric: [finite_mean(entry[metric]) for entry in scores] for metric in scores[0]
    }


#: Diagnostic distances: a NaN of theirs (e.g. too few distinct PF draws) is not a
#: divergence of the method, so it does not count in ``diverged_steps``.
DIAGNOSTIC_SCORES = (KL_KNN, ENERGY_DISTANCE)


def count_diverged_steps(scores) -> int:
    """(sequence, step) entries at which any core score is non-finite.

    ``finite_mean`` leaves such entries out, so a method that blows up early would
    otherwise be judged on its good steps only. This count keeps it visible. The
    :data:`DIAGNOSTIC_SCORES` are not core scores.
    """
    total = 0
    for entry in scores:
        core = [values for key, values in entry.items() if key not in DIAGNOSTIC_SCORES]
        for values in zip(*core):
            total += not all(math.isfinite(v) for v in values)
    return total


def normalised_innovations(problem, previous, observation) -> torch.Tensor:
    """``(y - H mean_i Phi(x_i)) / sqrt(diag(H P_f H') + q + r)`` for one step.

    ``Phi`` is the deterministic Lorenz 96 flow over one whole interval, so
    ``mean_i Phi(x_i)`` is the exact forecast mean of the member mixture under the
    true model and ``Cov_i(Phi(x_i)) + q I`` (``q = noise_std**2``) its forecast
    covariance. The same measure is used for every method, whatever its own
    forecast model.

    Both statements hold only for ``noise_stages == 1`` (one kick at the end of the
    interval). With several stages the earlier kicks pass through the nonlinear
    flow, so neither the mean nor the variance has this closed form; that case
    raises rather than report a wrong number.

    Nonlinear ``h = g(L x)`` (arctan): ``y - E[h(x^f)]`` over the same forecast
    mixture, divided by ``sqrt(Var[h(x^f)] + r)``, per site. The kick is independent
    per site and ``g`` elementwise, so per member ``E[g(a + sqrt(q) xi)]`` and its
    variance are 1D Gaussian integrals, computed by Gauss-Hermite quadrature
    (:data:`HERMITE_NODES` nodes, exact up to quadrature error): the predictive
    mean is ``mean_i m_i`` and the variance ``var_i(m_i) + mean_i v_i + r`` (the
    linear formula when ``g`` is the identity). These are the exact predictive mean
    and variance of the true-model forecast of the members, so for an exact filter
    the normalised innovations are still uncorrelated in time (martingale
    differences) with unit mean square, as in the linear case. What does not
    carry over is Gaussianity: the innovations are non-Gaussian, so their
    sampling spread and the PF-subsample floor differ from the linear problem's.

    Args:
        previous: ``[E, d]`` analysis members at the previous step (finite ones).
        observation: ``[d_y]``.
    """
    from scisi.dynamics.lorenz96 import integrate

    model = problem.model
    if model.noise_stages != 1:
        raise ValueError(
            "normalised_innovations is exact for noise_stages=1 only, "
            f"got {model.noise_stages}."
        )
    steps = model.steps_per_stage * model.noise_stages  # the whole interval
    flow = integrate(previous.double(), steps, problem.dt, problem.forcing)
    if not problem.linear_obs:
        return _nonlinear_innovations(problem, flow, observation)
    H = problem.obs_matrix
    projected = flow @ H.T
    variance = projected.var(0) + problem.noise_std**2 + problem.obs_variance
    return (observation.double() - projected.mean(0)) / variance.sqrt()


HERMITE_NODES = 40


def _nonlinear_innovations(problem, flow, observation) -> torch.Tensor:
    """Normalised innovations of a pointwise ``h`` (see :func:`normalised_innovations`)."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(HERMITE_NODES)
    nodes = torch.as_tensor(nodes, dtype=torch.float64)
    weights = torch.as_tensor(weights / weights.sum(), dtype=torch.float64)
    a = flow[:, problem.obs_indices]  # [E, d_y]
    values = problem.obs_nonlinearity(a[..., None] + problem.noise_std * nodes)[0]
    m = (values * weights).sum(-1)  # E[g | member], [E, d_y]
    v = (values.pow(2) * weights).sum(-1) - m.pow(2)
    variance = m.var(0) + v.mean(0) + problem.obs_variance
    return (observation.double() - m.mean(0)) / variance.sqrt()


def lag1_autocorrelation(z: torch.Tensor) -> float:
    """Pooled lag-1 autocorrelation of normalised innovations ``[T, d_y]``.

    ``sum_t z_t z_{t-1} / sqrt(sum_t z_t^2 sum_t z_{t-1}^2)`` over steps and sites:
    zero for an optimal filter (innovations are white), positive when the analysis
    under-uses the observations (FORECAST_CORRECTION.md, section 1).

    The mean is NOT removed, as in the library's
    ``CorrectionDiagnostics.lag1_autocorrelation``: a constant bias in the
    innovations counts as correlation (a constant sequence gives 1). Only pairs of
    adjacent steps that are both finite enter the sums, so a non-finite step drops
    the two pairs it belongs to and never joins the steps either side of it.
    """
    finite = torch.isfinite(z).all(-1)
    pair = finite[1:] & finite[:-1]
    if not bool(pair.any()):
        return float("nan")
    a, b = z[1:][pair], z[:-1][pair]
    return float((a * b).sum() / ((a**2).sum() * (b**2).sum()).sqrt())


def innovation_scores(problem, initial, ensemble, observations) -> dict[str, float]:
    """Lag-1 autocorrelation and mean square of the normalised innovations.

    Args:
        initial: ``[E, d]`` members before the first step, or None when the cell
            has none of its own: the first innovation is then NaN (left out).
        ensemble: ``[T, E, d]`` analysis members; observations: ``[T, d_y]``.
    """
    if initial is None:
        initial = torch.full_like(ensemble[0], float("nan"))
    previous = torch.cat([initial[None].double(), ensemble.double()[:-1]])
    rows = []
    for members, y in zip(previous, observations):
        alive = (torch.isfinite(members) & (members.abs() < DIVERGED_ABS)).all(-1)
        if alive.sum() < 2:
            rows.append(torch.full_like(y.double(), float("nan")))
        else:
            rows.append(normalised_innovations(problem, members[alive], y))
    z = torch.stack(rows)
    finite = z[torch.isfinite(z).all(-1)]
    msq = float(finite.pow(2).mean()) if finite.numel() else float("nan")
    return {INNOVATION_LAG1: lag1_autocorrelation(z), INNOVATION_MSQ: msq}


def innovation_output(
    ensembles, sequences, split, seed, problem, own_initial=True
) -> list[dict[str, float]]:
    """:func:`innovation_scores` of ensembles ``[S, T, E, d]``, one entry per sequence.

    The initial members are the cell's own (they depend on the ensemble size). A
    cell that does not start from :func:`initial_ensemble` passes
    ``own_initial=False`` and its first step is left out (:data:`NO_INITIAL_ENSEMBLE`).

    For ``noise_stages != 1`` the statistics have no exact form
    (:func:`normalised_innovations`): they are NaN, with a logged warning, so that
    the cell still runs.
    """
    if problem.model.noise_stages != 1:
        logging.getLogger(__name__).warning(
            "innovation statistics are exact for noise_stages=1 only (got %d); "
            "recording NaN.",
            problem.model.noise_stages,
        )
        nan = {INNOVATION_LAG1: float("nan"), INNOVATION_MSQ: float("nan")}
        return [dict(nan) for _ in sequences]
    members = ensembles.shape[2]
    return [
        innovation_scores(
            problem,
            initial_ensemble(problem, seq, members, split, index, seed)
            if own_initial
            else None,
            ens,
            seq.observations,
        )
        for index, (ens, seq) in enumerate(zip(ensembles, sequences))
    ]


# --------------------------------------------------------------------------- #
# Methods. Every runner takes (cfg, name, problem, sequences, split, seed,
# members, settings) and returns a CellOutput (RUNNERS, run_method).
# --------------------------------------------------------------------------- #


@dataclass
class CellOutput:
    ensembles: torch.Tensor  # [S, T, E, d]
    seconds: float  # wall clock of the whole cell
    nfe: float | None  # network evaluations per assimilation step
    members: int
    pseudo_steps: int | None
    # [S, T, E] per-member ESS of the importance-sampling oracle (arctan only).
    oracle_ess: torch.Tensor | None = None
    # SMC cells only: per sequence, per-step lists of the lorenz96_lineup.SMC_METRICS.
    smc: list[dict[str, list[float]]] | None = None


def sampler_inputs(cfg, problem, sequences, split, seed, members):
    """``(stats, device, initial [S, E, d], observations [S, T, d_y])`` of a cell
    that runs a trained network: the data standardisation, the sampler device, the
    shared initial members (:func:`initial_ensemble`) and the observations, all in
    physical units."""
    import cases.lorenz96._l96_generative as generative

    stats = generative.Standardisation.from_data(DATA_PATH, problem)
    device = resolve_device(cfg.evaluation.device)
    initial = torch.stack(
        [
            initial_ensemble(problem, seq, members, split, index, seed)
            for index, seq in enumerate(sequences)
        ]
    )
    observations = torch.stack([seq.observations for seq in sequences])
    return stats, device, initial, observations


def diverged_ensembles(problem, observations, members) -> torch.Tensor:
    """All-NaN ensembles ``[S, T, E, d]`` for ``observations`` ``[S, T, d_y]``: the
    output of a sampler that reported its own divergence (every step then counts
    in ``diverged_steps``)."""
    shape = (*observations.shape[:2], members, problem.dim_state)
    return torch.full(shape, float("nan"), dtype=torch.float64)


@dataclass(frozen=True)
class EnKFSettings:
    """What is tuned on validation sequences for the EnKF; unknown keys are refused."""

    inflation: float = 1.0
    localisation: float | None = None  # Gaspari--Cohn radius, None for none


def build_enkf(problem, settings: EnKFSettings) -> EnsembleKalmanFilter:
    """The stochastic EnKF of ``problem``: linear, or with predicted observations
    ``h(x_i)`` and ensemble cross-covariances for a nonlinear ``h`` (same
    localisation, observations placed at their sites)."""
    radius = settings.localisation
    taper = None if radius is None else periodic_taper(problem.dim_state, float(radius))
    return EnsembleKalmanFilter(
        problem.obs_matrix,
        problem.obs_variance,
        inflation=float(settings.inflation),
        localisation=taper,
        obs_nonlinearity=problem.obs_nonlinearity,
    )


def run_enkf(cfg, name, problem, sequences, split, seed, members, settings) -> CellOutput:
    enkf = build_enkf(problem, EnKFSettings(**settings))
    start = time.time()
    out = []
    for index, seq in enumerate(sequences):
        gen = torch.Generator().manual_seed(
            derive_seed(SEED_LABEL, "enkf", split, index, seed, members)
        )
        x0 = initial_ensemble(problem, seq, members, split, index, seed)
        out.append(enkf.run(problem.model, x0, seq.observations, gen).analysis)
    return CellOutput(torch.stack(out), time.time() - start, None, members, None)


def importance_oracle(problem, cfg) -> ImportanceOneStepOracle:
    """The per-member one-step oracle for a nonlinear ``h`` (no closed form), with
    ``oracle_arctan.proposals`` candidates per member and step."""
    return ImportanceOneStepOracle(
        problem.obs_matrix,
        problem.obs_variance,
        obs_nonlinearity=problem.obs_nonlinearity,
        proposals=int(cfg.oracle_arctan.proposals),
    )


def run_oracle(cfg, name, problem, sequences, split, seed, members, settings) -> CellOutput:
    """Exact one-step posterior per member: closed form when linear; for a
    nonlinear ``h`` importance sampling from the exact transition
    (:class:`ImportanceOneStepOracle`), whose per-member ESS is returned."""
    linear = problem.linear_obs
    if linear:
        oracle = OneStepPosteriorOracle(problem.obs_matrix, problem.obs_variance)
    else:
        oracle = importance_oracle(problem, cfg)
    start = time.time()
    out, ess = [], []
    for index, seq in enumerate(sequences):
        gen = torch.Generator().manual_seed(
            derive_seed(SEED_LABEL, "oracle", split, index, seed)
        )
        x0 = initial_ensemble(problem, seq, members, split, index, seed)
        result = oracle.run(problem.model, x0, seq.observations, gen)
        if linear:
            out.append(result)
        else:
            out.append(result.analysis)
            ess.append(result.ess)
    return CellOutput(
        torch.stack(out),
        time.time() - start,
        None,
        members,
        None,
        None if linear else torch.stack(ess),
    )


def run_oracle_fc(cfg, name, problem, sequences, split, seed, members, settings) -> CellOutput:
    """The exact one-step kernel per member, then the forecast-residual correction.

    The guided draw is ``OneStepPosteriorOracle``'s exact kick update (here
    ``noise_stages == 1``, so no inner SIR); for a nonlinear ``h`` it is the
    importance-sampling oracle (:class:`ImportanceOneStepOracle`, ESS returned). The
    correction uses true-model forecasts ``f_i ~ p(. | x^k_i)`` drawn independently,
    and by default the exact transition variance for ``K_q``
    (``correction.q_source: exact``; for a nonlinear ``h`` the library linearises
    it at each group's forecast mean), so what remains is the Gaussian closure on
    the across-member part. ``h`` and its Jacobian reach the correction through the
    physical-units operator (:func:`_l96_generative.physical_obs_operator`).
    """
    from cases.lorenz96._l96_generative import (
        build_obs_operator,
        correction_config,
        physical_obs_operator,
    )
    from scisi.posterior_models.forecast_correction import ForecastCorrection

    unknown = set(settings) - {"correction"}
    if unknown:
        raise ValueError(f"oracle_fc takes a correction block only, got {sorted(unknown)}")
    model = problem.model
    if model.noise_stages != 1:
        raise ValueError("oracle_fc implements the exact kernel for noise_stages=1 only.")
    q, r = model.stage_variance, problem.obs_variance
    H = problem.obs_matrix
    m, d = H.shape
    gain = q * H.T @ torch.linalg.inv(q * H @ H.T + r * torch.eye(m, dtype=H.dtype))
    block = {"q_source": "exact", **settings.get("correction", {})}
    corr = ForecastCorrection(correction_config(problem, block, members, q))
    online = corr.config.q_source == "online"
    linear = problem.linear_obs
    operator = build_obs_operator(problem) if linear else physical_obs_operator(problem)
    sampler = None if linear else importance_oracle(problem, cfg)
    start = time.time()
    out, ess = [], []
    for index, seq in enumerate(sequences):
        gen = torch.Generator().manual_seed(
            derive_seed(SEED_LABEL, "oracle-fc", split, index, seed)
        )
        x = initial_ensemble(problem, seq, members, split, index, seed)
        steps, steps_ess = [], []
        for y in seq.observations:
            if linear:
                z = model.forecast_to_last_kick(x, gen)
                kicked = z + q**0.5 * torch.randn(z.shape, generator=gen, dtype=z.dtype)
                eps = r**0.5 * torch.randn(members, m, generator=gen, dtype=z.dtype)
                guided = kicked + (y - kicked @ H.T - eps) @ gain.T
            else:
                guided, member_ess = sampler.draw(model, x, y, gen)
                steps_ess.append(member_ess)
            f = model.step(x, gen)
            f2 = model.step(x, gen).reshape(members, 1, 1, d) if online else None
            x = corr.correct(
                guided.reshape(members, 1, 1, d),
                f.reshape(members, 1, 1, d),
                y.reshape(1, m),
                operator,
                r,
                f2=f2,
                perturbations=r**0.5
                * torch.randn(members, m, generator=gen, dtype=guided.dtype),
            ).reshape(members, d)
            steps.append(x)
        out.append(torch.stack(steps))
        if not linear:
            ess.append(torch.stack(steps_ess))
    return CellOutput(
        torch.stack(out),
        time.time() - start,
        None,
        members,
        None,
        None if linear else torch.stack(ess),
    )


def run_prior_enkf(cfg, name, problem, sequences, split, seed, members, settings) -> CellOutput:
    """Unguided trained-prior forecast + EnKF update (the alternative of section 2).

    Same E, observations and initial ensembles as every other cell; the EnKF update
    is the one of the ``enkf`` cell (:class:`EnKFSettings`). The other keys are the
    prior's forecast solver (:class:`SamplerSettings`; only ``num_steps`` and, for
    a DM-SDE prior, ``diffusion_scale`` matter, as no likelihood is used).
    """
    import cases.lorenz96._l96_generative as generative
    from cases.navier_stokes._ns_pipeline import attach_nfe_counter

    settings = dict(settings)
    prior = settings.pop("prior", PRIOR_ENKF[name])
    if prior not in GENERATIVE:
        raise ValueError(f"{name}: prior must be one of {sorted(GENERATIVE)}, got {prior!r}")
    update = EnKFSettings(
        **{key: settings.pop(key) for key in ("inflation", "localisation") if key in settings}
    )
    sampler = generative.SamplerSettings(**settings)
    stats, device, initial, observations = sampler_inputs(
        cfg, problem, sequences, split, seed, members
    )
    run = cfg.checkpoints.si_run if prior == "si_sde" else cfg.checkpoints.fm_run
    model = generative.load_prior(run, device)
    posterior, stepper = generative.build_posterior(
        GENERATIVE[prior], model, problem, stats, sampler
    )
    enkf = build_enkf(problem, update)
    generators = [
        torch.Generator().manual_seed(
            derive_seed(SEED_LABEL, "prior-enkf", split, index, seed)
        )
        for index in range(len(sequences))
    ]
    counter = attach_nfe_counter(model)
    start = time.time()
    ensembles = generative.assimilate_prior_enkf(
        posterior,
        stepper,
        stats,
        initial,
        observations,
        sampler.num_steps,
        derive_seed(SEED_LABEL, name, split, seed),
        enkf,
        generators,
    )
    seconds = time.time() - start
    nfe = counter.count / observations.shape[1]
    return CellOutput(ensembles, seconds, nfe, members, sampler.num_steps)


def run_pf_matched(cfg, name, problem, sequences, split, seed, members, settings) -> CellOutput:
    """The same filter as the reference with only ``members`` particles."""
    pf = BootstrapParticleFilter(
        problem.obs_matrix,
        problem.obs_variance,
        reference_proposal(problem),
        num_saved_samples=members,
        obs_nonlinearity=problem.obs_nonlinearity,
    )
    start = time.time()
    out, smc = [], []
    for index, seq in enumerate(sequences):
        gen = torch.Generator().manual_seed(
            derive_seed(SEED_LABEL, "pf-matched", split, index, seed)
        )
        x0 = initial_ensemble(problem, seq, members, split, index, seed)
        run = pf.run(problem.model, x0.T.contiguous(), seq.observations, gen)
        out.append(run.samples)
        smc.append(pf_smc_record(run.ess, run.num_particles))
    return CellOutput(torch.stack(out), time.time() - start, None, members, None, smc=smc)


def run_subsample(cfg, name, problem, sequences, split, seed, members, settings) -> CellOutput:
    """``members`` reference draws: the floor any size-``members`` ensemble can reach."""
    # The permutation seed has no split (as before the split existed in it): the
    # validation and test subsamples share one stream. Left as it is, so that the
    # cell's numbers do not move.
    gen = torch.Generator().manual_seed(derive_seed(SEED_LABEL, "subsample", seed))
    out = []
    for index in range(len(sequences)):
        samples = load_reference(split, index, 1, problem)["samples"]
        keep = torch.randperm(samples.shape[1], generator=gen)[:members]
        out.append(samples[:, keep].double())
    return CellOutput(torch.stack(out), 0.0, None, members, None)


def run_generative(cfg, name, problem, sequences, split, seed, members, settings) -> CellOutput:
    """A plain sampler, or (``*_fc``) the same sampler with the correction."""
    import cases.lorenz96._l96_generative as generative
    from cases.navier_stokes._ns_pipeline import attach_nfe_counter

    settings = dict(settings)
    # Only a corrected cell reads a correction block; elsewhere SamplerSettings refuses it.
    block = settings.pop("correction", {}) if name in CORRECTED else None
    # A corrected cell runs its plain cell's sampler and seed stream.
    base_name = CORRECTED.get(name, name)
    settings = generative.SamplerSettings(**settings)
    stats, device, initial, observations = sampler_inputs(
        cfg, problem, sequences, split, seed, members
    )
    run = cfg.checkpoints.si_run if base_name == "si_sde" else cfg.checkpoints.fm_run
    model = generative.load_prior(run, device)
    posterior, stepper = generative.build_posterior(
        GENERATIVE[base_name], model, problem, stats, settings
    )
    correction = None
    if name in CORRECTED:
        correction = generative.build_correction(problem, stats, members, block)
    counter = attach_nfe_counter(model)
    start = time.time()
    nfe = None
    try:
        ensembles = generative.assimilate(
            posterior,
            stepper,
            stats,
            initial,
            observations,
            settings.num_steps,
            derive_seed(SEED_LABEL, base_name, split, seed),
            forecast_correction=correction,
            obs_to_model=generative.observation_model(problem, stats).to_model,
        )
        nfe = counter.count / observations.shape[1]
    except AssertionError as error:
        # The DM-SDE sampler asserts on a non-finite drift where the others carry
        # the NaN on. Count the whole cell as diverged rather than losing the run;
        # any other assertion is a bug and must crash. The partial count of network
        # calls says nothing about the cost of a full run, so nfe stays None.
        if base_name != "dm_sde" or "non-finite" not in str(error):
            raise
        print(f"{name} diverged: {error}", flush=True)
        ensembles = diverged_ensembles(problem, observations, members)
    seconds = time.time() - start
    return CellOutput(ensembles, seconds, nfe, members, settings.num_steps)


def run_baseline(*args) -> CellOutput:
    from cases.lorenz96._l96_baselines import run_baseline

    return run_baseline(*args)


def run_surge(*args) -> CellOutput:
    from cases.lorenz96._l96_smc import run_surge

    return run_surge(*args)


#: cell -> runner; every runner has the signature of :func:`run_method`'s call.
RUNNERS = {
    "enkf": run_enkf,
    "enkf_large": run_enkf,
    "oracle": run_oracle,
    "oracle_fc": run_oracle_fc,
    **{name: run_prior_enkf for name in PRIOR_ENKF},
    "pf_matched": run_pf_matched,
    "pf_subsample": run_subsample,
    **{name: run_generative for name in (*GENERATIVE, *CORRECTED)},
    **{name: run_baseline for name in ("flowdas", "sda", "dflow", "fig")},
    **{name: run_surge for name in SURGE_CELLS},
}


def run_method(cfg, name, problem, sequences, split, seed, settings=None) -> CellOutput:
    """Run cell ``name`` on ``sequences`` (``pf_reference`` is scored by
    :func:`evaluate_cell` from the stored references)."""
    size = "enkf_large_size" if name == "enkf_large" else "ensemble_size"
    members = int(cfg.evaluation[size])
    if settings is None:
        settings = method_settings(cfg, name)
    return RUNNERS[name](cfg, name, problem, sequences, split, seed, members, settings)


# --------------------------------------------------------------------------- #
# Cells
# --------------------------------------------------------------------------- #


def evaluate_cell(
    name: str,
    seed: int,
    cfg=None,
    split: str = "test",
    overrides=None,
    tag: str = "",
    num_steps: int | None = None,
    allow_other_m: bool = False,
    write_metrics: bool = False,
) -> list[ResultRecord]:
    """Run and score one (method, seed); writes per-step rows and the ensembles
    (and, with ``write_metrics``, the tidy metrics file ``cell`` writes).

    ``num_steps`` sets M of a generative cell (one with a ``num_steps`` setting);
    it is refused for the others. M comes from the cell's settings: it must be in
    :data:`M_VALUES` unless ``allow_other_m`` (:func:`check_m`), and a generative
    cell's output paths carry it (:func:`cell_file`), so the cells of the M sweep
    never overwrite each other.

    ``overrides`` are ``key=value`` strings applied to the cell's settings (see
    :func:`apply_overrides`); they need a ``tag``, which goes into every output
    path and into the tidy ``variant`` (:func:`tagged_variant`), so a non-default
    run never overwrites or poses as the default cell.
    """
    check_tag(name, tag, overrides)
    cfg = case_config() if cfg is None else cfg
    settings = cell_settings(cfg, name, overrides, num_steps)
    if not allow_other_m:
        check_m(name, settings, M_VALUES)
    check_correction_label(name, settings, tag)
    problem = problem_of(cfg)
    obs = problem.obs_label
    m = settings.get("num_steps")
    sequences = load_sequences(problem, split, split_size(cfg, split))
    method, variant = CELLS[name]
    variant = tagged_variant(variant, tag)
    nfe = None
    innovations = None
    oracle_ess = None
    smc = None

    if name == "pf_reference":
        # The second replicate against the first: the error of the reference itself.
        scores = []
        seconds = []
        smc = []
        for index, seq in enumerate(sequences):
            ref = load_reference(split, index, 0, problem)
            other = load_reference(split, index, 1, problem)
            seconds.append(other["seconds"])
            smc.append(pf_smc_record(other["ess"], other["num_particles"]))
            scores.append(
                score(
                    other["mean"],
                    other["variance"],
                    list(other["samples"]),
                    ref,
                    seq.truth[1:],
                )
            )
        per_step_seconds = finite_mean(seconds) / problem.num_steps
        members, pseudo_steps = int(cfg.evaluation.reference_particles), None
    else:
        output = run_method(cfg, name, problem, sequences, split, seed, settings)
        scores = score_output(output.ensembles, sequences, split, problem)
        innovations = innovation_output(
            output.ensembles,
            sequences,
            split,
            seed,
            problem,
            own_initial=name not in NO_INITIAL_ENSEMBLE,
        )
        members, pseudo_steps, nfe = output.members, output.pseudo_steps, output.nfe
        oracle_ess = output.oracle_ess
        smc = output.smc
        path = results_dir(obs) / "ensembles" / f"{cell_label(name, tag, m)}__seed{seed}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output.ensembles.float(), path)
        # Cost on equal terms: one sequence at this method's ensemble size, whereas
        # the run above batches sequences for the generative methods only.
        per_step_seconds = None
        if name != "pf_subsample":
            timed = run_method(cfg, name, problem, sequences[:1], split, seed, settings)
            per_step_seconds = timed.seconds / problem.num_steps

    common = dict(
        case=Case.LORENZ96.value,
        method=method.value,
        scenario=SCENARIOS[obs].value,
        E=members,
        M=pseudo_steps,
        seed=seed,
        nfe=nfe,
        seconds=per_step_seconds,
        variant=variant,
    )
    # Curves are averaged over sequences here: common.aggregate_lib then reduces
    # them over seeds, which are the buckets of this case.
    curves = {
        metric: [finite_mean(at_step) for at_step in zip(*(e[metric] for e in scores))]
        for metric in scores[0]
    }
    per_sequence = summarise(scores)
    if innovations is not None:
        for key in (INNOVATION_LAG1, INNOVATION_MSQ):
            per_sequence[key] = [entry[key] for entry in innovations]
    if smc is not None:
        # ESS / resampling of the SMC cells (_l96_smc): curves averaged over
        # sequences, per-sequence means over steps.
        for key in SMC_METRICS:
            curves[key] = [finite_mean(at_step) for at_step in zip(*(e[key] for e in smc))]
            per_sequence[key] = [finite_mean(entry[key]) for entry in smc]
    if oracle_ess is not None:
        # [S, T, E]: per step the mean / minimum over members and sequences.
        curves[ORACLE_ESS_MEAN] = oracle_ess.mean((0, 2)).tolist()
        curves[ORACLE_ESS_MIN] = oracle_ess.amin((0, 2)).tolist()
        per_sequence[ORACLE_ESS_MEAN] = oracle_ess.mean((1, 2)).tolist()
        per_sequence[ORACLE_ESS_MIN] = oracle_ess.amin((1, 2)).tolist()
    per_step_file = cell_path("per_step", name, seed, tag, obs, m)
    per_step_file.parent.mkdir(parents=True, exist_ok=True)
    per_step_file.unlink(missing_ok=True)
    append_per_step(
        per_step_file, per_step_rows(test_index=-1, per_step=curves, **common)
    )
    sequence_file = cell_path("per_sequence", name, seed, tag, obs, m).with_suffix(".json")
    sequence_file.parent.mkdir(parents=True, exist_ok=True)
    sequence_file.write_text(json.dumps(per_sequence))

    records = [
        ResultRecord(
            metric=metric,
            value=min(values) if metric == ORACLE_ESS_MIN else finite_mean(values),
            **common,
        )
        for metric, values in per_sequence.items()
    ]
    if per_step_seconds is not None:
        records.append(
            ResultRecord(metric=Metric.SECONDS.value, value=per_step_seconds, **common)
        )
    if nfe is not None:
        records.append(ResultRecord(metric=Metric.NFE.value, value=nfe, **common))
    diverged = float(count_diverged_steps(scores))
    records.append(ResultRecord(metric=DIVERGED_STEPS, value=diverged, **common))
    if write_metrics:
        path = cell_path("metrics", name, seed, tag, obs, m)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        with ResultsWriter(path) as writer:
            writer.extend(records)
    return records


def finite_mean(values) -> float:
    """Mean over finite entries.

    ``diverged_fraction`` records the members left out and ``diverged_steps`` the
    (sequence, step) entries left out.
    """
    kept = [value for value in values if math.isfinite(value)]
    return sum(kept) / len(kept) if kept else float("nan")


TAG_PATTERN = re.compile(r"[A-Za-z0-9.=-]+(_[A-Za-z0-9.=-]+)*")


def check_tag(name: str, tag: str, overrides) -> None:
    """Refuse overrides without a tag, a tag that would break file names, and
    overrides of a cell that ignores its settings."""
    if overrides and name in NO_SETTINGS:
        raise ValueError(f"{name} takes no settings; --set would be ignored.")
    if overrides and not tag:
        raise ValueError(
            "--set changes the cell, so it needs --tag: without one the run would "
            "overwrite the default cell's outputs and be aggregated as it."
        )
    if tag and (not TAG_PATTERN.fullmatch(tag) or re.search(r"traj\d", tag)):
        raise ValueError(
            f"Bad tag {tag!r}: use letters, digits, '.', '=', '-' and single '_' "
            "(no '__', no 'traj<N>')."
        )
    if tag:
        method, variant = CELLS[name]
        labelled = (method, tagged_variant(variant, tag))
        if labelled in CELLS.values():
            raise ValueError(
                f"Tag {tag!r} on {name} gives the label {labelled[0].value} / "
                f"{labelled[1]}, the label of an untagged cell: the two would be "
                "aggregated together. Choose another tag."
            )


def check_correction_label(name: str, settings: dict, tag: str) -> None:
    """Refuse an untagged correction cell whose correction is disabled.

    Its rows would be labelled ``forecast_correction`` though no correction ran
    (``enabled: false`` set in the YAML, where ``--set``'s tag rule does not reach).
    """
    if CELLS[name][1] != FC_VARIANT or tag:
        return
    if not bool(settings.get("correction", {}).get("enabled", True)):
        raise ValueError(
            f"{name} has correction.enabled=false, so its rows would read as the "
            "corrected cell: run it with a --tag (e.g. 'off')."
        )


def tagged_variant(variant: str | None, tag: str) -> str | None:
    """``forecast_correction`` + tag ``fs5`` -> ``forecast_correction_fs5``."""
    if not tag:
        return variant
    return f"{variant}_{tag}" if variant else tag


def cell_label(name: str, tag: str = "", num_steps: int | None = None) -> str:
    """``<cell>[-<tag>][__M<M>]``: the file-name stem of a cell's outputs."""
    label = f"{name}-{tag}" if tag else name
    return label if num_steps is None else f"{label}__M{int(num_steps)}"


def cell_path(
    kind: str, name: str, seed: int, tag: str = "", obs: str = "linear", num_steps: int | None = None
) -> Path:
    """``lorenz96__<cell>[-<tag>][__M<M>]__traj<seed>__.csv`` under the sweep's root.

    Generative cells pass their M; M-independent cells (EnKF, PFs, oracle) have none.
    Callers that have a config use :func:`cell_file`, which takes M from the settings.
    """
    # common.aggregate_lib buckets files by this token; here a bucket is a seed.
    label = cell_label(name, tag, num_steps)
    return results_dir(obs) / kind / f"lorenz96__{label}__traj{seed}__.csv"


def cell_settings(cfg, name: str, overrides=None, num_steps: int | None = None) -> dict:
    """The settings a cell runs with: its config block, ``overrides``, then M."""
    return with_num_steps(name, apply_overrides(method_settings(cfg, name), overrides), num_steps)


def cell_file(
    kind: str, name: str, seed: int, cfg, tag: str = "", overrides=None,
    num_steps: int | None = None,
) -> Path:
    """The output file of a cell run with these arguments (as :func:`evaluate_cell`
    writes it): M comes from the cell's settings, so no caller passes it by hand."""
    m = cell_settings(cfg, name, overrides, num_steps).get("num_steps")
    return cell_path(kind, name, seed, tag, problem_of(cfg).obs_label, m)


def with_num_steps(name: str, settings: dict, num_steps: int | None) -> dict:
    """``settings`` with M set to ``num_steps``; refused for a cell without an M."""
    if num_steps is None:
        return settings
    if "num_steps" not in settings:
        raise ValueError(f"{name} has no SDE/ODE steps; --num-steps does not apply.")
    return {**settings, "num_steps": int(num_steps)}


def check_m(name: str, settings: dict, allowed) -> None:
    """Refuse an M (``num_steps``) outside ``allowed`` and a decoupled forecast M.

    ``tune`` allows the tuning M only, ``cell`` every M of the sweep. The
    correction's ``forecast_num_steps`` must be null (the forecasts run at the
    sampler's M), so a stray override cannot mix step counts into the tables.
    """
    steps = settings.get("num_steps")
    if steps is not None and int(steps) not in tuple(allowed):
        raise ValueError(
            f"{name}: num_steps={steps}, but this run allows M in {tuple(allowed)}; "
            "pass --allow-other-m to run it anyway."
        )
    forecast = settings.get("correction", {}).get("forecast_num_steps")
    if forecast is not None:
        raise ValueError(
            f"{name}: correction.forecast_num_steps={forecast}, but the forecasts run at "
            "the sampler's M (null); pass --allow-other-m to run it anyway."
        )


def cmd_cell(args) -> None:
    torch.set_num_threads(args.threads)
    records = evaluate_cell(
        args.method,
        args.seed,
        config_of(args),
        overrides=getattr(args, "set", None),
        tag=getattr(args, "tag", ""),
        num_steps=getattr(args, "num_steps", None),
        allow_other_m=getattr(args, "allow_other_m", False),
        write_metrics=True,
    )
    for record in records:
        print(f"{record.metric:20s} {record.value:.4f}", flush=True)


def cmd_tune(args) -> None:
    """Grid over one method's settings on the validation split only."""
    torch.set_num_threads(args.threads)
    cfg = config_of(args)
    problem = problem_of(cfg)
    sequences = load_sequences(problem, "validation", split_size(cfg, "validation"))
    if args.method in NO_SETTINGS:
        raise ValueError(f"{args.method} takes no settings; there is nothing to tune.")
    base = apply_overrides(method_settings(cfg, args.method), getattr(args, "set", None))
    grid = json.loads(args.grid)
    tag = f"__{args.tag}" if getattr(args, "tag", "") else ""
    path = results_dir(problem.obs_label) / "tuning" / f"{args.method}__{'_'.join(grid)}{tag}.csv"
    # The base settings (after --set) go into every row, and a file with other
    # columns is refused, so rows of differently configured runs cannot mix
    # unnoticed.
    columns = [
        *grid,
        "base",
        "seconds",
        DIVERGED_STEPS,
        *(m.value for m in TUNING_METRICS),
    ]
    # Corrected and trained-prior cells add the innovation statistics and cost.
    innovations = args.method in CORRECTED or args.method in PRIOR_ENKF or args.method == "oracle_fc"
    if innovations:
        columns += [INNOVATION_LAG1, INNOVATION_MSQ, Metric.NFE.value]
    handle, writer = open_csv_for_append(path, columns)
    with handle:
        for values in itertools.product(*grid.values()):
            override = dict(zip(grid, values))
            # Dotted keys (correction.gain_scale) reach the nested block.
            settings = set_dotted(base, override)
            if not getattr(args, "allow_other_m", False):
                check_m(args.method, settings, (HEADLINE_M,))
            output = run_method(
                cfg,
                args.method,
                problem,
                sequences,
                "validation",
                0,
                settings,
            )
            scores = score_output(output.ensembles, sequences, "validation", problem)
            per_sequence = summarise(scores)
            row = {
                metric.value: finite_mean(per_sequence[metric.value])
                for metric in TUNING_METRICS
            }
            if innovations:
                stats = innovation_output(output.ensembles, sequences, "validation", 0, problem)
                for key in (INNOVATION_LAG1, INNOVATION_MSQ):
                    row[key] = finite_mean(entry[key] for entry in stats)
                row[Metric.NFE.value] = output.nfe
            diverged = count_diverged_steps(scores)
            if diverged:
                # Not eligible: a score over the surviving steps would flatter it.
                row.update({metric.value: math.inf for metric in TUNING_SCORES})
            row.update(
                override,
                base=json.dumps(base, sort_keys=True),
                seconds=output.seconds,
                **{DIVERGED_STEPS: diverged},
            )
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)


# Selection rule, fixed before results: lowest validation sliced_w2.
TUNING_SCORES = (
    Metric.MEAN_ERROR_VS_PF,
    Metric.VAR_ERROR_VS_PF,
    Metric.SLICED_W2,
    Metric.RMSE,
)
TUNING_METRICS = (*TUNING_SCORES, Metric.SPREAD, Metric.DIVERGED_FRACTION)


def cmd_prior_quality(args) -> None:
    """Compare each learned transition with the true kernel, without observations.

    States come from the validation trajectories. For the true kernel the conditional
    mean is the deterministic flow and the variance is ``noise_std**2`` at every site.
    """
    from cases.lorenz96._l96_generative import Standardisation, load_prior
    from scisi.dynamics.lorenz96 import integrate
    from scisi.sampling.ode_solvers import euler_step
    from scisi.sampling.sde_solvers import euler_maruyama_step

    torch.set_num_threads(args.threads)
    cfg = config_of(args)
    problem = problem_of(cfg)
    stats = Standardisation.from_data(DATA_PATH, problem)
    with np.load(DATA_PATH) as data:
        states = torch.from_numpy(data["states"][int(data["num_train"]) :, -1])
    states = states[: args.states].double()
    flow = integrate(
        states, problem.model.steps_per_stage, problem.dt, problem.forcing
    )
    device = resolve_device(cfg.evaluation.device)
    report = {}
    for label, run in (("si", cfg.checkpoints.si_run), ("fm", cfg.checkpoints.fm_run)):
        model = load_prior(run, device)
        torch.manual_seed(derive_seed(SEED_LABEL, "prior-quality", label))
        rows = []
        for x, target in zip(states, flow):
            x0 = stats.forward(x.float()).reshape(1, 1, 1, -1).repeat(args.draws, 1, 1, 1)
            x0 = x0.to(device)
            history = x0.unsqueeze(-1)
            with torch.no_grad():
                if label == "si":
                    draws = model.sample(
                        base=x0,
                        field_history=history,
                        batch_size=args.draws,
                        num_steps=args.num_steps,
                        stepper=euler_maruyama_step,
                    )
                else:
                    draws = model.sample(
                        field_history=history,
                        batch_size=args.draws,
                        num_steps=args.num_steps,
                        stepper=euler_step,
                    )
            draws = stats.inverse(draws.reshape(args.draws, -1).cpu().double())
            rows.append(
                (
                    (draws.mean(0) - target).pow(2).mean().sqrt().item(),
                    draws.var(0).mean().item(),
                )
            )
        errors, variances = zip(*rows)
        report[label] = dict(
            mean_error=sum(errors) / len(errors),
            mean_error_floor=problem.noise_std / args.draws**0.5,
            variance=sum(variances) / len(variances),
            true_variance=problem.noise_std**2,
            num_steps=args.num_steps,
        )
        print(label, json.dumps(report[label]), flush=True)
    # Observation-free, so the same for every operator; written under each
    # operator's root anyway, so that each results/lorenz96/<obs>/ tree is complete.
    path = results_dir(problem.obs_label) / "prior_quality.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


class Lorenz96Runner(ExperimentRunner):
    """``run.py`` entry point: one configured method over the case's seeds.

    Everything comes from the composed ``case`` config. References must exist
    (``driver.py reference --split test``). The large EnKF and the diagnostic rows
    have no ``configs/method`` file and run through ``driver.py cell``.
    """

    case = Case.LORENZ96
    _NAMES = {
        Method.OURS_SI_SDE: "si_sde",
        Method.OURS_FM_ODE: "fm_ode",
        Method.OURS_DM_SDE: "dm_sde",
        Method.ENKF: "enkf",
        Method.PARTICLE_FILTER: "pf_matched",
    }

    def __init__(self, config, *, seeds=None) -> None:
        super().__init__(config, seeds=list(config.case.evaluation.seeds))
        scenario = config.get("scenario", {}).get("name")
        by_label = {s.value: obs for obs, s in SCENARIOS.items()}
        if scenario not in by_label:
            raise ValueError(
                f"Use scenario=lorenz96 or scenario=lorenz96_arctan with this case, "
                f"not {scenario!r}."
            )
        self.obs = by_label[scenario]

    def methods(self):
        method = Method(self.config.method.name)
        if method not in self._NAMES:
            raise ValueError(f"{method.value} is not part of the Lorenz 96 case.")
        return [method]

    def scenarios(self):
        return [SCENARIOS[self.obs]]

    def evaluate(self, ctx: RunContext):
        # A plain copy: the composed Hydra node may be read-only / struct.
        case = OmegaConf.create(OmegaConf.to_container(self.config.case, resolve=True))
        cfg = configure(case, self.obs)
        return evaluate_cell(self._NAMES[ctx.method], ctx.seed, cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--obs",
        choices=OBS_OPERATORS,
        default="linear",
        help="observation operator (before the subcommand); outputs go to "
        "results/lorenz96/<obs>/",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    reference = sub.add_parser("reference")
    reference.add_argument("--split", choices=["validation", "test"], required=True)
    reference.add_argument("--workers", type=int, default=6)
    reference.set_defaults(func=cmd_reference)
    tune = sub.add_parser("tune")
    # Only cells that read settings: a grid over the others is a no-op.
    tune.add_argument("--method", choices=TUNABLE, required=True)
    tune.add_argument(
        "--grid",
        required=True,
        help="JSON: setting -> list of values; dotted keys reach the correction block",
    )
    tune.add_argument("--set", action="append", help="key=value base override (repeatable)")
    tune.add_argument("--tag", default="", help="suffix of the tuning file name")
    tune.add_argument("--threads", type=int, default=2)
    tune.add_argument(
        "--allow-other-m",
        action="store_true",
        help=f"allow num_steps other than the tuning M={HEADLINE_M}",
    )
    tune.set_defaults(func=cmd_tune)
    cell = sub.add_parser("cell")
    cell.add_argument("--method", choices=sorted(CELLS), required=True)
    cell.add_argument("--seed", type=int, required=True)
    cell.add_argument(
        "--set",
        action="append",
        help="key=value override of the cell's settings, e.g. "
        "correction.localisation=4, correction.gain_scale=0.8 "
        "(repeatable; JSON values); needs --tag",
    )
    cell.add_argument(
        "--tag",
        default="",
        help="label of a non-default run, required with --set: outputs go to "
        "<method>-<tag> and the tidy variant becomes <variant>_<tag>",
    )
    cell.add_argument("--threads", type=int, default=2)
    cell.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help=f"M of a generative cell, one of {M_VALUES} (default: the config's, "
        f"{HEADLINE_M}); refused for cells without SDE/ODE steps",
    )
    cell.add_argument(
        "--allow-other-m",
        action="store_true",
        help=f"allow num_steps outside {M_VALUES}",
    )
    cell.set_defaults(func=cmd_cell)
    quality = sub.add_parser("prior-quality")
    quality.add_argument("--states", type=int, default=64)
    quality.add_argument("--draws", type=int, default=512)
    quality.add_argument("--num-steps", type=int, default=100)
    quality.add_argument("--threads", type=int, default=2)
    quality.set_defaults(func=cmd_prior_quality)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
