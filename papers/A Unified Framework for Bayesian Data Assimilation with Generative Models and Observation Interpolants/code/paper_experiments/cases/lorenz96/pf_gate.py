"""Design gate: can the particle filter serve as ground truth for this system?

``screen`` sweeps candidate systems at moderate N. For the chosen system:
``converge`` runs the ladder of particle counts with independent replicates against
coded pass criteria, ``calibrate`` checks the filter against the truth with wrong
filters as negative controls, and ``cross-proposal`` compares two different
proposals. Replicate agreement alone cannot detect a filter that is consistently
wrong, which is why all three exist. Sequences used here have their own seed range
and are never the held-out test sequences.

    python cases/lorenz96/pf_gate.py screen --grid d16 --out results/lorenz96/gate/screen_d16.csv
    python cases/lorenz96/pf_gate.py converge --out results/lorenz96/gate/convergence.json
    python cases/lorenz96/pf_gate.py calibrate --out results/lorenz96/gate/calibration.json
    python cases/lorenz96/pf_gate.py cross-proposal --out results/lorenz96/gate/cross.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import torch

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[1]))

from cases.lorenz96._l96_system import (  # noqa: E402
    Problem,
    draw_prior_ensemble,
    make_sequence,
    open_csv_for_append,
)
from common.seeding import derive_seed  # noqa: E402
from scisi.particle_filter.bootstrap import (  # noqa: E402
    BootstrapParticleFilter,
    OneStepPosteriorOracle,
)
from scisi.metrics.distributional import sliced_wasserstein_w2  # noqa: E402
from scisi.particle_filter.enkf import EnsembleKalmanFilter  # noqa: E402

GATE_SEED_LABEL = "gate"


# Replicates run in single-threaded worker processes: the filter is memory bound and
# gains nothing from torch threads, while separate processes scale. This chunk size
# keeps each worker's working set in cache.
PF_CHUNK = 8192


def run_pf(problem, sequence, n, seed, proposal="optimal"):
    """One float32 particle-filter run from the shared prior of ``sequence``.

    ``driver.py reference`` calls this same function, so what the gate validates is
    exactly what produces the references.
    """
    # Separate streams: reusing one seed makes the first kick equal the prior noise.
    gen = torch.Generator().manual_seed(derive_seed(seed, "filter"))
    particles = draw_prior_ensemble(
        problem, sequence, n, derive_seed(seed, "prior"), torch.float32
    ).T.contiguous()
    pf = BootstrapParticleFilter(
        problem.obs_matrix, problem.obs_variance, proposal, chunk_size=PF_CHUNK
    )
    return pf.run(problem.model, particles, sequence.observations, gen)


def _pf_task(job):
    torch.set_num_threads(1)
    return pickle.dumps(run_pf(*job))


def run_pf_many(jobs, workers):
    """Run ``run_pf`` argument tuples in parallel, preserving order."""
    if workers <= 1 or len(jobs) == 1:
        return [run_pf(*job) for job in jobs]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=context) as pool:
        return [pickle.loads(blob) for blob in pool.map(_pf_task, jobs)]


def moments_gap(mean, var, ref_mean, ref_var, skip):
    """Mean error in posterior-std units, and the ratio of mean variances."""
    scale = ref_var[skip:].mean().sqrt()
    gap = ((mean[skip:] - ref_mean[skip:]) ** 2).mean().sqrt() / scale
    return float(gap), float(var[skip:].mean() / ref_var[skip:].mean())


def screen_one(problem: Problem, n: int, sequences: int, enkf_members: int, skip: int):
    rows = []
    for s in range(sequences):
        seed = derive_seed(GATE_SEED_LABEL, "screen", s)
        seq = make_sequence(problem, seed)
        a = run_pf(problem, seq, n, seed + 1)
        b = run_pf(problem, seq, n, seed + 2)
        gen = torch.Generator().manual_seed(derive_seed(seed, "enkf", "filter"))
        prior = draw_prior_ensemble(
            problem, seq, enkf_members, derive_seed(seed, "enkf", "prior")
        )
        enkf = EnsembleKalmanFilter(problem.obs_matrix, problem.obs_variance).run(
            problem.model, prior, seq.observations, gen
        ).analysis
        oracle = OneStepPosteriorOracle(
            problem.obs_matrix, problem.obs_variance, inner_samples=128
        ).run(problem.model, prior[:1024], seq.observations, gen)

        mc_gap, mc_var = moments_gap(a.mean, a.variance, b.mean, b.variance, skip)
        enkf_gap, enkf_var = moments_gap(
            enkf.mean(1), enkf.var(1), a.mean, a.variance, skip
        )
        oracle_gap, oracle_var = moments_gap(
            oracle.mean(1), oracle.var(1), a.mean, a.variance, skip
        )
        x = a.samples[skip:].double()
        z = (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True)
        rmse = lambda m: float(  # noqa: E731
            ((m[skip:] - seq.truth[1:][skip:]) ** 2).mean().sqrt()
        )
        rows.append(
            dict(
                ess_mean=float(a.ess_fraction[skip:].mean()),
                ess_min=float(a.ess_fraction[skip:].min()),
                mc_gap=mc_gap,
                mc_var=mc_var,
                enkf_gap=enkf_gap,
                enkf_var=enkf_var,
                oracle_gap=oracle_gap,
                oracle_var=oracle_var,
                skew=float((z**3).mean(1).abs().mean()),
                ex_kurt=float(((z**4).mean(1) - 3).abs().mean()),
                post_std=float(a.variance[skip:].mean().sqrt()),
                rmse_pf=rmse(a.mean),
                rmse_enkf=rmse(enkf.mean(1)),
                rmse_oracle=rmse(oracle.mean(1)),
            )
        )
    summary = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    summary["ess_min"] = min(r["ess_min"] for r in rows)  # a minimum, not a mean of minima
    return summary


GRIDS = {
    # Effect of each knob around one d = 16 centre.
    "d16": dict(
        dim_state=[16],
        interval=[0.1, 0.2, 0.4],
        noise_std=[0.5, 1.0, 2.0],
        noise_stages=[1, 2],
        obs_stride=[1, 2, 4],
        obs_variance=[1.0],
    ),
}


# Columns of ``screen_one``'s summary, in order.
SCREEN_STATS = (
    "ess_mean",
    "ess_min",
    "mc_gap",
    "mc_var",
    "enkf_gap",
    "enkf_var",
    "oracle_gap",
    "oracle_var",
    "skew",
    "ex_kurt",
    "post_std",
    "rmse_pf",
    "rmse_enkf",
    "rmse_oracle",
)


def cmd_screen(args):
    grid = GRIDS[args.grid] if args.grid in GRIDS else json.loads(args.grid)
    keys = list(grid)
    columns = [*keys, "N", *SCREEN_STATS, "seconds"]
    # Refuses, before any run, a file written for another grid.
    fh, writer = open_csv_for_append(Path(args.out), columns)
    with fh:
        for values in itertools.product(*(grid[k] for k in keys)):
            problem = replace(Problem(num_steps=args.steps), **dict(zip(keys, values)))
            start = time.time()
            stats = screen_one(problem, args.n, args.sequences, args.enkf, args.skip)
            row = {**dict(zip(keys, values)), "N": args.n, **stats}
            row["seconds"] = time.time() - start
            writer.writerow(row)
            fh.flush()
            print(
                " ".join(
                    f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in row.items()
                ),
                flush=True,
            )


def entry_errors(run, ref_mean, ref_var):
    """Per-(step, coordinate) error of one run against a reference."""
    return (
        (run.mean - ref_mean).abs() / ref_var.sqrt(),
        (run.variance / ref_var).log().abs(),
    )


def quantiles(x):
    x = x.flatten().double()
    return dict(
        median=float(x.median()),
        q95=float(torch.quantile(x, 0.95)),
        max=float(x.max()),
    )


def rung_stats(runs, top, skip):
    """Statistics of one rung of the N ladder, over ALL steps unless named.

    Every error is that of a SINGLE run in units of the LOCAL posterior std, so
    neither averaging over replicates nor pooling over entries can hide it. The
    reference is the top rung, leaving the run out if it is itself a top run.

    The leave-one-out makes the top rung's reference noisier than a lower rung's:
    with ``R`` top replicates of ``N_top`` particles, a lower run's squared error
    is about ``s^2 (1/N + 1/(R N_top))`` but a top run's is
    ``s^2 (1 + 1/(R-1)) / N_top``. So the two-point rate of :func:`fit_rate` comes
    out slightly shallower than -1/2 for exact Monte Carlo error: about -0.48 for
    the ladder (1e5, 4e6) at ``R = 8``.
    """
    mean_err, var_err, cov_err, sw2 = [], [], [], []
    for i, run in enumerate(runs):
        pool = [r for r in top if r is not run]
        ref_mean = torch.stack([r.mean for r in pool]).mean(0)
        ref_var = torch.stack([r.variance for r in pool]).mean(0)
        m, v = entry_errors(run, ref_mean, ref_var)
        mean_err.append(m)
        var_err.append(v)
        ref_cov = torch.stack([r.covariance for r in pool]).mean(0)
        std = ref_var.sqrt()
        scale = std[:, :, None] * std[:, None, :]
        cov_err.append(((run.covariance - ref_cov) / scale).abs())
    mean_err, var_err = torch.stack(mean_err), torch.stack(var_err)
    cov_err = torch.stack(cov_err)
    ess = torch.stack([r.ess for r in runs])
    evidence = torch.stack([r.log_evidence.sum() for r in runs])
    a, b = runs[0], runs[1]
    last = a.samples.shape[0] - 1
    for step in (skip, last // 2 + skip // 2, last):
        sw2.append(
            sliced_wasserstein_w2(a.samples[step].double(), b.samples[step].double())
            / float(a.variance[step].mean().sqrt())
        )
    return dict(
        ess_fraction_mean=float((ess / runs[0].num_particles).mean()),
        ess_fraction_min=float((ess / runs[0].num_particles).min()),
        ess_abs_min=float(ess.min()),
        mean_error=quantiles(mean_err),
        mean_error_rms=float(mean_err.pow(2).mean().sqrt()),
        mean_error_after_skip=quantiles(mean_err[:, skip:]),
        log_var_error=quantiles(var_err),
        log_var_error_rms=float(var_err.pow(2).mean().sqrt()),
        correlation_error=quantiles(cov_err),
        sliced_w2_between_runs=sw2,
        log_evidence=float(evidence.mean()),
        log_evidence_sd=float(evidence.std()),
    )


def fit_rate(ladder, values):
    """Slope of log(value) against log(N); Monte Carlo error should give -1/2."""
    x = torch.tensor(ladder, dtype=torch.float64).log()
    y = torch.tensor(values, dtype=torch.float64).log()
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


# Coded pass criteria for the top rung. Errors are single-run and in units of the
# local posterior std; the log-variance error is a relative error.
PASS_CRITERIA = dict(
    mean_error_q95=0.10,
    log_var_error_q95=0.10,
    log_evidence_sd=0.50,
    ess_abs_min=500.0,
    rate_low=-0.70,
    rate_high=-0.30,
)


def verdict(top_stats, rate):
    checks = dict(
        mean_error_q95=top_stats["mean_error"]["q95"]
        <= PASS_CRITERIA["mean_error_q95"],
        log_var_error_q95=top_stats["log_var_error"]["q95"]
        <= PASS_CRITERIA["log_var_error_q95"],
        log_evidence_sd=top_stats["log_evidence_sd"] <= PASS_CRITERIA["log_evidence_sd"],
        ess_abs_min=top_stats["ess_abs_min"] >= PASS_CRITERIA["ess_abs_min"],
        rate=PASS_CRITERIA["rate_low"] <= rate <= PASS_CRITERIA["rate_high"],
    )
    return dict(checks=checks, passed=all(checks.values()))


def cmd_converge(args):
    problem = replace(Problem(), **json.loads(args.problem))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        problem=problem.to_dict(),
        proposal="optimal",
        criteria=PASS_CRITERIA,
        replicates=args.replicates,
        skip=args.skip,
        sequences=[],
    )
    ladder = [int(float(n)) for n in args.ladder.split(",")]
    for s in range(args.sequences):
        seed = derive_seed(GATE_SEED_LABEL, "converge", s)
        seq = make_sequence(problem, seed)
        runs = {}
        for n in ladder:
            start = time.time()
            runs[n] = run_pf_many(
                [
                    (problem, seq, n, derive_seed(seed, n, rep))
                    for rep in range(args.replicates)
                ],
                args.workers,
            )
            print(f"seq {s} N={n:.0e} {time.time() - start:.0f}s", flush=True)
        top = runs[ladder[-1]]
        entry = dict(seed=seed, ladder={})
        for n in ladder:
            entry["ladder"][str(n)] = rung_stats(runs[n], top, args.skip)
            print(f"  N={n:.0e}", json.dumps(entry["ladder"][str(n)]), flush=True)
        entry["rate_mean_error"] = fit_rate(
            ladder, [entry["ladder"][str(n)]["mean_error_rms"] for n in ladder]
        )
        entry["verdict"] = verdict(
            entry["ladder"][str(ladder[-1])], entry["rate_mean_error"]
        )
        print(
            f"  rate {entry['rate_mean_error']:.3f}", json.dumps(entry["verdict"]), flush=True
        )
        report["sequences"].append(entry)
        out.write_text(json.dumps(report, indent=2))
    report["passed"] = all(e["verdict"]["passed"] for e in report["sequences"])
    out.write_text(json.dumps(report, indent=2))
    print("GATE", "PASSED" if report["passed"] else "FAILED", flush=True)


def calibration_summary(problem, assumed, sequences, n, skip, label, workers=1):
    """Truth-based check that replicate agreement cannot provide.

    Data come from ``problem``; the filter assumes ``assumed``. If the filter is
    right, the truth is a draw from its posterior: E[(truth - mean)^2 / var] = 1 and
    the rank of the truth among posterior samples is uniform, on observed and on
    unobserved coordinates alike. Filtering with a deliberately wrong ``assumed``
    is a negative control showing that the test can actually fail.
    """
    observed = torch.zeros(problem.dim_state, dtype=torch.bool)
    observed[problem.obs_indices] = True
    seeds = [derive_seed(GATE_SEED_LABEL, "calibrate", s) for s in range(sequences)]
    seqs = [make_sequence(problem, seed) for seed in seeds]
    runs = run_pf_many(
        [(assumed, seq, n, derive_seed(seed, label)) for seq, seed in zip(seqs, seeds)],
        workers,
    )
    return calibration_stats([seq.truth[1:] for seq in seqs], runs, observed, skip)


def calibration_stats(truths, runs, observed, skip):
    """z^2 and truth ranks of filter runs against the truths ``[T, d]`` they track.

    ``runs`` need ``mean``, ``variance`` ``[T, d]`` and ``samples`` ``[T, S, d]``.
    """
    z2, ranks = [], []
    for truth, run in zip(truths, runs):
        z2.append(((truth - run.mean) ** 2 / run.variance)[skip:])
        ranks.append((run.samples.double() < truth[:, None, :]).double().mean(1)[skip:])
    z2 = torch.stack(z2)  # [S, T, d]
    ranks = torch.stack(ranks)
    sequences = len(runs)

    def block(mask):
        per_sequence = z2[..., mask].mean((1, 2))
        hist = torch.histc(ranks[..., mask].float(), bins=10, min=0.0, max=1.0)
        hist = hist / hist.sum()
        return dict(
            z2=float(per_sequence.mean()),
            z2_se=float(per_sequence.std() / sequences**0.5),
            rank_decile_fraction=[round(float(h), 4) for h in hist],
            rank_max_deviation=float((hist - 0.1).abs().max()),
        )

    return dict(all=block(torch.ones_like(observed)), observed=block(observed), unobserved=block(~observed))


def cmd_calibrate(args):
    problem = replace(Problem(), **json.loads(args.problem))
    report = dict(problem=problem.to_dict(), N=args.n, sequences=args.sequences)
    controls = dict(
        correct=problem,
        wrong_obs_variance_x2=replace(problem, obs_variance=2 * problem.obs_variance),
        wrong_model_noise_x0p5=replace(problem, noise_std=0.5 * problem.noise_std),
        wrong_forcing_6=replace(problem, forcing=6.0),
    )
    for label, assumed in controls.items():
        report[label] = calibration_summary(
            problem, assumed, args.sequences, args.n, args.skip, label, args.workers
        )
        print(label, json.dumps(report[label]), flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))


def cmd_cross_proposal(args):
    """Bootstrap against optimal on a system where BOTH converge, so that a
    disagreement would be evidence of an error rather than bootstrap noise."""
    problem = replace(Problem(), **json.loads(args.problem))
    report = dict(problem=problem.to_dict(), N=args.n, replicates=args.replicates, sequences=[])
    for s in range(args.sequences):
        seed = derive_seed(GATE_SEED_LABEL, "cross", s)
        seq = make_sequence(problem, seed)
        runs = {
            proposal: run_pf_many(
                [
                    (problem, seq, args.n, derive_seed(seed, proposal, rep), proposal)
                    for rep in range(args.replicates)
                ],
                args.workers,
            )
            for proposal in ("optimal", "bootstrap")
        }
        reps = args.replicates
        boot, opt = runs["bootstrap"], runs["optimal"]

        def t_stat(get):
            a = torch.stack([get(r) for r in boot])
            b = torch.stack([get(r) for r in opt])
            se = (a.var(0) / reps + b.var(0) / reps).sqrt()
            return ((a.mean(0) - b.mean(0)) / se).abs()

        t_mean, t_var = t_stat(lambda r: r.mean), t_stat(lambda r: r.variance)
        evidence_t = t_stat(lambda r: r.log_evidence.sum())
        entry = dict(
            seed=seed,
            ess_fraction_bootstrap=float(
                torch.stack([r.ess_fraction for r in boot]).mean()
            ),
            ess_abs_min_bootstrap=float(torch.stack([r.ess for r in boot]).min()),
            ess_fraction_optimal=float(torch.stack([r.ess_fraction for r in opt]).mean()),
            t_mean_rms=float(t_mean.pow(2).mean().sqrt()),
            t_var_rms=float(t_var.pow(2).mean().sqrt()),
            t_log_evidence=float(evidence_t),
            log_evidence_bootstrap=float(torch.stack([r.log_evidence.sum() for r in boot]).mean()),
            log_evidence_optimal=float(torch.stack([r.log_evidence.sum() for r in opt]).mean()),
        )
        print(json.dumps(entry), flush=True)
        report["sequences"].append(entry)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    # Accepted after the subcommand, as in GPU_HANDOFF.md.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--threads", type=int, default=7)
    screen = sub.add_parser("screen", parents=[common])
    screen.add_argument("--grid", default="d16")
    screen.add_argument("--out", required=True)
    screen.add_argument("--n", type=int, default=100_000)
    screen.add_argument("--sequences", type=int, default=2)
    screen.add_argument("--steps", type=int, default=30)
    screen.add_argument("--skip", type=int, default=10)
    screen.add_argument("--enkf", type=int, default=2000)
    screen.set_defaults(func=cmd_screen)
    converge = sub.add_parser("converge", parents=[common])
    converge.add_argument("--problem", default="{}", help="JSON overrides of Problem")
    converge.add_argument("--out", required=True)
    converge.add_argument("--ladder", default="1e4,1e5,1e6")
    converge.add_argument("--replicates", type=int, default=8)
    converge.add_argument("--workers", type=int, default=6)
    converge.add_argument("--sequences", type=int, default=3)
    converge.add_argument("--skip", type=int, default=10)
    converge.set_defaults(func=cmd_converge)
    calibrate = sub.add_parser("calibrate", parents=[common])
    calibrate.add_argument("--problem", default="{}", help="JSON overrides of Problem")
    calibrate.add_argument("--out", required=True)
    calibrate.add_argument("--n", type=int, default=100_000)
    calibrate.add_argument("--sequences", type=int, default=32)
    calibrate.add_argument("--skip", type=int, default=10)
    calibrate.add_argument("--workers", type=int, default=6)
    calibrate.set_defaults(func=cmd_calibrate)
    cross = sub.add_parser("cross-proposal", parents=[common])
    cross.add_argument("--problem", default="{}", help="JSON overrides of Problem")
    cross.add_argument("--out", required=True)
    cross.add_argument("--n", type=int, default=1_000_000)
    cross.add_argument("--replicates", type=int, default=8)
    cross.add_argument("--sequences", type=int, default=2)
    cross.add_argument("--workers", type=int, default=6)
    cross.set_defaults(func=cmd_cross_proposal)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.func(args)


if __name__ == "__main__":
    main()
