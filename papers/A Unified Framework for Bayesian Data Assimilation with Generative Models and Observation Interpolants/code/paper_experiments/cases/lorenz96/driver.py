"""Lorenz 96 comparison against a particle-filter reference.

Every method is scored the same way on the same sequences. A sequence fixes truth,
observations and the prior of the initial state; a seed fixes the members drawn from
that prior and the method's own randomness.

    python cases/lorenz96/driver.py reference --split validation
    python cases/lorenz96/driver.py tune --method enkf --grid '{"inflation": [1.0, 1.1]}'
    python cases/lorenz96/driver.py reference --split test
    python cases/lorenz96/driver.py cell --method si_sde --seed 0

``run_lorenz96_grid.sh`` runs all of it. Test sequences are never used to choose a
setting: ``tune`` reads the validation split only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing
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

from cases.lorenz96.pf_gate import run_pf  # noqa: E402
from cases.lorenz96._l96_system import (  # noqa: E402
    CASE_CONFIG,
    Problem,
    Sequence,
    draw_prior_ensemble,
    make_sequence,
    open_csv_for_append,
    resolve_device,
)
from common.paths import REPO_ROOT, RESULTS_ROOT  # noqa: E402
from common.per_step_io import append_per_step, per_step_rows  # noqa: E402
from common.runner import ExperimentRunner, RunContext  # noqa: E402
from common.seeding import derive_seed  # noqa: E402
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
    OneStepPosteriorOracle,
)
from scisi.particle_filter.enkf import (  # noqa: E402
    EnsembleKalmanFilter,
    periodic_taper,
)

RESULTS = RESULTS_ROOT / "lorenz96"
SEED_LABEL = "lorenz96-evaluation"
DATA_PATH = REPO_ROOT / "data/lorenz96/data.npz"
DIVERGED_ABS = 100.0  # about 25 climatological standard deviations

# name -> (tidy method label, variant)
CELLS = {
    "si_sde": (Method.OURS_SI_SDE, None),
    "fm_ode": (Method.OURS_FM_ODE, None),
    "dm_sde": (Method.OURS_DM_SDE, None),
    "enkf": (Method.ENKF, "matched"),
    "enkf_large": (Method.ENKF, "large"),
    "oracle": (Method.ONE_STEP_ORACLE, None),
    "pf_matched": (Method.PARTICLE_FILTER, "matched"),
    "pf_subsample": (Method.PARTICLE_FILTER, "reference_subsample"),
    "pf_reference": (Method.PARTICLE_FILTER, "reference"),
}
GENERATIVE = {
    name: method for name, (method, _) in CELLS.items() if method.name.startswith("OURS_")
}
# (sequence, step) entries with a non-finite score: fewer than two members were left.
DIVERGED_STEPS = "diverged_steps"


def case_config():
    return OmegaConf.load(CASE_CONFIG)


def problem_of(cfg) -> Problem:
    return Problem(**OmegaConf.to_container(cfg.problem))


def method_settings(cfg, name: str) -> dict:
    node = cfg.methods.get(name)
    return {} if node is None else OmegaConf.to_container(node)


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


def reference_path(split: str, index: int, replicate: int) -> Path:
    return RESULTS / "reference" / split / f"sequence{index}_replicate{replicate}.pt"


def _reference_task(job) -> float:
    problem, sequence, split, index, replicate, particles = job
    torch.set_num_threads(1)
    seed = derive_seed(SEED_LABEL, "reference", split, index, replicate)
    start = time.time()
    # The gate's own filter, so its validity carries over by construction.
    run = run_pf(problem, sequence, particles, seed)
    seconds = time.time() - start
    path = reference_path(split, index, replicate)
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
    cfg = case_config()
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
        if not reference_path(args.split, index, rep).exists()
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        for job, seconds in zip(jobs, pool.map(_reference_task, jobs)):
            print(f"{args.split} {job[3]} replicate {job[4]}: {seconds:.0f}s", flush=True)


def load_reference(split: str, index: int, replicate: int, problem: Problem) -> dict:
    """Load a stored reference, refusing one made for another system."""
    reference = torch.load(reference_path(split, index, replicate), weights_only=False)
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


def count_diverged_steps(scores) -> int:
    """(sequence, step) entries at which any score is non-finite.

    ``finite_mean`` leaves such entries out, so a method that blows up early would
    otherwise be judged on its good steps only. This count keeps it visible.
    """
    total = 0
    for entry in scores:
        for values in zip(*entry.values()):
            total += not all(math.isfinite(v) for v in values)
    return total


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #


@dataclass
class CellOutput:
    ensembles: torch.Tensor  # [S, T, E, d]
    seconds: float  # wall clock of the whole cell
    nfe: float | None  # network evaluations per assimilation step
    members: int
    pseudo_steps: int | None


@dataclass(frozen=True)
class EnKFSettings:
    """What is tuned on validation sequences for the EnKF; unknown keys are refused."""

    inflation: float = 1.0
    localisation: float | None = None  # Gaspari--Cohn radius, None for none


def run_enkf(problem, sequences, split, seed, members, settings) -> CellOutput:
    settings = EnKFSettings(**settings)
    radius = settings.localisation
    taper = None if radius is None else periodic_taper(problem.dim_state, float(radius))
    enkf = EnsembleKalmanFilter(
        problem.obs_matrix,
        problem.obs_variance,
        inflation=float(settings.inflation),
        localisation=taper,
    )
    start = time.time()
    out = []
    for index, seq in enumerate(sequences):
        gen = torch.Generator().manual_seed(
            derive_seed(SEED_LABEL, "enkf", split, index, seed, members)
        )
        x0 = initial_ensemble(problem, seq, members, split, index, seed)
        out.append(enkf.run(problem.model, x0, seq.observations, gen).analysis)
    return CellOutput(torch.stack(out), time.time() - start, None, members, None)


def run_oracle(problem, sequences, split, seed, members) -> CellOutput:
    oracle = OneStepPosteriorOracle(problem.obs_matrix, problem.obs_variance)
    start = time.time()
    out = []
    for index, seq in enumerate(sequences):
        gen = torch.Generator().manual_seed(
            derive_seed(SEED_LABEL, "oracle", split, index, seed)
        )
        x0 = initial_ensemble(problem, seq, members, split, index, seed)
        out.append(oracle.run(problem.model, x0, seq.observations, gen))
    return CellOutput(torch.stack(out), time.time() - start, None, members, None)


def run_pf_matched(problem, sequences, split, seed, members) -> CellOutput:
    """The same filter as the reference with only ``members`` particles."""
    pf = BootstrapParticleFilter(
        problem.obs_matrix,
        problem.obs_variance,
        "optimal",
        num_saved_samples=members,
    )
    start = time.time()
    out = []
    for index, seq in enumerate(sequences):
        gen = torch.Generator().manual_seed(
            derive_seed(SEED_LABEL, "pf-matched", split, index, seed)
        )
        x0 = initial_ensemble(problem, seq, members, split, index, seed)
        run = pf.run(problem.model, x0.T.contiguous(), seq.observations, gen)
        out.append(run.samples)
    return CellOutput(torch.stack(out), time.time() - start, None, members, None)


def run_subsample(problem, sequences, split, seed, members) -> CellOutput:
    """``members`` reference draws: the floor any size-``members`` ensemble can reach."""
    gen = torch.Generator().manual_seed(derive_seed(SEED_LABEL, "subsample", seed))
    out = []
    for index in range(len(sequences)):
        samples = load_reference(split, index, 1, problem)["samples"]
        keep = torch.randperm(samples.shape[1], generator=gen)[:members]
        out.append(samples[:, keep].double())
    return CellOutput(torch.stack(out), 0.0, None, members, None)


def run_generative(cfg, name, problem, sequences, split, seed, members, settings):
    from cases.lorenz96._l96_generative import (
        SamplerSettings,
        Standardisation,
        assimilate,
        build_posterior,
        load_prior,
    )

    method = GENERATIVE[name]
    run = cfg.checkpoints.si_run if name == "si_sde" else cfg.checkpoints.fm_run
    settings = SamplerSettings(**settings)
    stats = Standardisation.from_data(DATA_PATH, problem)
    model = load_prior(run, resolve_device(cfg.evaluation.device))
    posterior, stepper = build_posterior(method, model, problem, stats, settings)
    initial = torch.stack(
        [
            initial_ensemble(problem, seq, members, split, index, seed)
            for index, seq in enumerate(sequences)
        ]
    )
    observations = torch.stack([seq.observations for seq in sequences])
    from cases.navier_stokes._ns_pipeline import attach_nfe_counter

    counter = attach_nfe_counter(model)
    start = time.time()
    nfe = None
    try:
        ensembles = assimilate(
            posterior,
            stepper,
            stats,
            initial,
            observations,
            settings.num_steps,
            derive_seed(SEED_LABEL, name, split, seed),
        )
        nfe = counter.count / observations.shape[1]
    except AssertionError as error:
        # The DM-SDE sampler asserts on a non-finite drift where the others carry
        # the NaN on. Count the whole cell as diverged rather than losing the run;
        # any other assertion is a bug and must crash. The partial count of network
        # calls says nothing about the cost of a full run, so nfe stays None.
        if name != "dm_sde" or "non-finite" not in str(error):
            raise
        print(f"{name} diverged: {error}", flush=True)
        shape = (len(sequences), observations.shape[1], members, problem.dim_state)
        ensembles = torch.full(shape, float("nan"), dtype=torch.float64)
    seconds = time.time() - start
    return CellOutput(ensembles, seconds, nfe, members, settings.num_steps)


def run_method(cfg, name, problem, sequences, split, seed, settings=None) -> CellOutput:
    members = int(cfg.evaluation.ensemble_size)
    if settings is None:
        settings = method_settings(cfg, name)
    if name == "enkf":
        return run_enkf(problem, sequences, split, seed, members, settings)
    if name == "enkf_large":
        size = int(cfg.evaluation.enkf_large_size)
        return run_enkf(problem, sequences, split, seed, size, settings)
    if name == "oracle":
        return run_oracle(problem, sequences, split, seed, members)
    if name == "pf_matched":
        return run_pf_matched(problem, sequences, split, seed, members)
    if name == "pf_subsample":
        return run_subsample(problem, sequences, split, seed, members)
    return run_generative(cfg, name, problem, sequences, split, seed, members, settings)


# --------------------------------------------------------------------------- #
# Cells
# --------------------------------------------------------------------------- #


def evaluate_cell(name: str, seed: int, cfg=None, split: str = "test") -> list[ResultRecord]:
    """Run and score one (method, seed); writes per-step rows and the ensembles."""
    cfg = case_config() if cfg is None else cfg
    problem = problem_of(cfg)
    sequences = load_sequences(problem, split, split_size(cfg, split))
    method, variant = CELLS[name]
    nfe = None

    if name == "pf_reference":
        # The second replicate against the first: the error of the reference itself.
        scores = []
        seconds = []
        for index, seq in enumerate(sequences):
            ref = load_reference(split, index, 0, problem)
            other = load_reference(split, index, 1, problem)
            seconds.append(other["seconds"])
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
        output = run_method(cfg, name, problem, sequences, split, seed)
        scores = score_output(output.ensembles, sequences, split, problem)
        members, pseudo_steps, nfe = output.members, output.pseudo_steps, output.nfe
        path = RESULTS / "ensembles" / f"{name}__seed{seed}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output.ensembles.float(), path)
        # Cost on equal terms: one sequence at this method's ensemble size, whereas
        # the run above batches sequences for the generative methods only.
        per_step_seconds = None
        if name != "pf_subsample":
            timed = run_method(cfg, name, problem, sequences[:1], split, seed)
            per_step_seconds = timed.seconds / problem.num_steps

    common = dict(
        case=Case.LORENZ96.value,
        method=method.value,
        scenario=Scenario.LORENZ96.value,
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
    per_step_file = cell_path("per_step", name, seed)
    per_step_file.parent.mkdir(parents=True, exist_ok=True)
    per_step_file.unlink(missing_ok=True)
    append_per_step(
        per_step_file, per_step_rows(test_index=-1, per_step=curves, **common)
    )
    sequence_file = cell_path("per_sequence", name, seed).with_suffix(".json")
    sequence_file.parent.mkdir(parents=True, exist_ok=True)
    sequence_file.write_text(json.dumps(per_sequence))

    records = [
        ResultRecord(metric=metric, value=finite_mean(values), **common)
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
    return records


def finite_mean(values) -> float:
    """Mean over finite entries.

    ``diverged_fraction`` records the members left out and ``diverged_steps`` the
    (sequence, step) entries left out.
    """
    kept = [value for value in values if math.isfinite(value)]
    return sum(kept) / len(kept) if kept else float("nan")


def cell_path(kind: str, name: str, seed: int) -> Path:
    # common.aggregate_lib buckets files by this token; here a bucket is a seed.
    return RESULTS / kind / f"lorenz96__{name}__traj{seed}__.csv"


def cmd_cell(args) -> None:
    torch.set_num_threads(args.threads)
    records = evaluate_cell(args.method, args.seed)
    path = cell_path("metrics", args.method, args.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with ResultsWriter(path) as writer:
        writer.extend(records)
    for record in records:
        print(f"{record.metric:20s} {record.value:.4f}", flush=True)


def cmd_tune(args) -> None:
    """Grid over one method's settings on the validation split only."""
    torch.set_num_threads(args.threads)
    cfg = case_config()
    problem = problem_of(cfg)
    sequences = load_sequences(problem, "validation", split_size(cfg, "validation"))
    base = method_settings(cfg, args.method)
    grid = json.loads(args.grid)
    path = RESULTS / "tuning" / f"{args.method}__{'_'.join(grid)}.csv"
    # The base settings go into every row, and a file with other columns is
    # refused, so rows of differently configured runs cannot mix unnoticed.
    columns = [
        *grid,
        "base",
        "seconds",
        DIVERGED_STEPS,
        *(m.value for m in TUNING_METRICS),
    ]
    handle, writer = open_csv_for_append(path, columns)
    with handle:
        for values in itertools.product(*grid.values()):
            override = dict(zip(grid, values))
            output = run_method(
                cfg,
                args.method,
                problem,
                sequences,
                "validation",
                0,
                {**base, **override},
            )
            scores = score_output(output.ensembles, sequences, "validation", problem)
            per_sequence = summarise(scores)
            row = {
                metric.value: finite_mean(per_sequence[metric.value])
                for metric in TUNING_METRICS
            }
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
    cfg = case_config()
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
    path = RESULTS / "prior_quality.json"
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
        if scenario != Scenario.LORENZ96.value:
            raise ValueError(f"Use scenario=lorenz96 with this case, not {scenario!r}.")

    def methods(self):
        method = Method(self.config.method.name)
        if method not in self._NAMES:
            raise ValueError(f"{method.value} is not part of the Lorenz 96 case.")
        return [method]

    def scenarios(self):
        return [Scenario.LORENZ96]

    def evaluate(self, ctx: RunContext):
        return evaluate_cell(self._NAMES[ctx.method], ctx.seed, self.config.case)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reference = sub.add_parser("reference")
    reference.add_argument("--split", choices=["validation", "test"], required=True)
    reference.add_argument("--workers", type=int, default=6)
    reference.set_defaults(func=cmd_reference)
    tune = sub.add_parser("tune")
    tune.add_argument("--method", choices=[*GENERATIVE, "enkf", "enkf_large"], required=True)
    tune.add_argument("--grid", required=True, help="JSON: setting -> list of values")
    tune.add_argument("--threads", type=int, default=2)
    tune.set_defaults(func=cmd_tune)
    cell = sub.add_parser("cell")
    cell.add_argument("--method", choices=sorted(CELLS), required=True)
    cell.add_argument("--seed", type=int, required=True)
    cell.add_argument("--threads", type=int, default=2)
    cell.set_defaults(func=cmd_cell)
    quality = sub.add_parser("prior-quality")
    quality.add_argument("--states", type=int, default=64)
    quality.add_argument("--draws", type=int, default=512)
    quality.add_argument("--num-steps", type=int, default=100)
    quality.add_argument("--threads", type=int, default=2)
    quality.set_defaults(func=cmd_prior_quality)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
