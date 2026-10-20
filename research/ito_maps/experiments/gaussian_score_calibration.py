#!/usr/bin/env python3
"""Illustrate expected proper score versus population-interval calibration.

Truth Y~N(0,1); forecast N(0,sigma**2). This is an elementary analytic
illustration, not a fit to Navier--Stokes and not finite-ensemble coverage.
The formulas are checked by independent Gaussian quadrature and paired Monte
Carlo using common normal draws. Outputs include JSON and a standalone figure.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np


SQRT_PI = math.sqrt(math.pi)
Z90 = NormalDist().inv_cdf(0.95)


def expected_crps(sigma):
    return (math.sqrt(2 * (1 + sigma * sigma)) - sigma) / SQRT_PI


def population_coverage90(sigma):
    return math.erf(Z90 * sigma / math.sqrt(2))


def quadrature_expected_crps(sigma):
    # Evaluate forecast CRPS(y) first, then independently integrate over truth Y.
    nodes, weights = np.polynomial.hermite.hermgauss(128)
    y = math.sqrt(2) * nodes
    absolute_error = np.array(
        [
            sigma
            * math.sqrt(2 / math.pi)
            * math.exp(-value * value / (2 * sigma * sigma))
            + value * math.erf(value / (math.sqrt(2) * sigma))
            for value in y
        ]
    )
    return float(np.sum(weights * (absolute_error - sigma / SQRT_PI)) / SQRT_PI)


def analytic_row(sigma):
    score = expected_crps(sigma)
    calibrated = expected_crps(1)
    numerical = quadrature_expected_crps(sigma)
    np.testing.assert_allclose(score, numerical, rtol=1e-11, atol=1e-12)
    coverage = population_coverage90(sigma)
    return {
        "sigma": sigma,
        "expected_crps": score,
        "expected_crps_excess": score - calibrated,
        "relative_expected_crps_penalty_percent": 100 * (score / calibrated - 1),
        "population_coverage90": coverage,
        "population_coverage_change_percentage_points": 100 * (coverage - 0.9),
        "quadrature_expected_crps": numerical,
        "quadratic_excess_approximation": (sigma - 1) ** 2 / (4 * SQRT_PI),
    }


def monte_carlo(rows, samples, seed):
    rng = np.random.default_rng(seed)
    truth, source, second_source = rng.normal(size=(3, samples))
    calibrated_score = np.abs(source - truth) - 0.5 * np.abs(source - second_source)
    for row in rows:
        sigma = row["sigma"]
        score = np.abs(sigma * source - truth) - 0.5 * sigma * np.abs(
            source - second_source
        )
        excess = score - calibrated_score
        row["mc_expected_crps"] = float(score.mean())
        row["mc_paired_excess"] = float(excess.mean())
        row["mc_paired_excess_standard_error"] = float(
            excess.std(ddof=1) / math.sqrt(samples)
        )
        row["mc_population_coverage90"] = float(np.mean(np.abs(truth) <= Z90 * sigma))


def plot(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigmas = np.linspace(0.5, 1.5, 401)
    penalty = [100 * (expected_crps(s) / expected_crps(1) - 1) for s in sigmas]
    coverage = [100 * population_coverage90(s) for s in sigmas]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    axes[0].plot(sigmas, penalty, color="#295e95", linewidth=2)
    axes[1].plot(sigmas, coverage, color="#a44e35", linewidth=2)
    for axis in axes:
        axis.axvline(1, color="0.5", linestyle="--", linewidth=1)
        axis.set_xlabel("Forecast standard deviation $\\sigma$")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Relative CRPS penalty (%)")
    axes[1].set_ylabel("Population 90% coverage (%)")
    axes[1].axhline(90, color="0.5", linestyle=":", linewidth=1)
    sigma = 0.84
    xpenalty = 100 * (expected_crps(sigma) / expected_crps(1) - 1)
    xcoverage = 100 * population_coverage90(sigma)
    axes[0].scatter([sigma], [xpenalty], color="#295e95", zorder=5)
    axes[0].annotate(
        f"$\\sigma=0.84$\n{xpenalty:.2f}% penalty",
        (sigma, xpenalty),
        xytext=(0.57, 4.7),
        arrowprops={"arrowstyle": "->", "color": "0.4"},
        fontsize=9,
    )
    axes[1].scatter([sigma], [xcoverage], color="#a44e35", zorder=5)
    axes[1].annotate(
        f"$\\sigma=0.84$\n{xcoverage:.1f}% coverage",
        (sigma, xcoverage),
        xytext=(1.03, 72),
        arrowprops={"arrowstyle": "->", "color": "0.4"},
        fontsize=9,
    )
    fig.savefig(output / "gaussian_score_calibration.pdf")
    fig.savefig(output / "gaussian_score_calibration.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results/gaussian_score_calibration",
    )
    parser.add_argument("--mc-samples", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=4631)
    args = parser.parse_args()
    if args.mc_samples < 2:
        parser.error("--mc-samples must be at least2")
    rows = [analytic_row(s) for s in (0.5, 0.7, 0.84, 0.9, 1.0, 1.1, 1.2)]
    monte_carlo(rows, args.mc_samples, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    coverage_derivative = 2 * Z90 * math.exp(-Z90 * Z90 / 2) / math.sqrt(2 * math.pi)
    record = {
        "truth": "N(0,1)",
        "forecast": "N(0,sigma^2)",
        "expected_crps_formula": "(sqrt(2*(1+sigma^2))-sigma)/sqrt(pi)",
        "quadratic_excess_near_sigma1": "(sigma-1)^2/(4*sqrt(pi)) + O((sigma-1)^3)",
        "population_coverage90_formula": "2*Phi(Phi^-1(0.95)*sigma)-1",
        "coverage_derivative_at_sigma1": coverage_derivative,
        "nominal_interval_gaussian_quantile": Z90,
        "mc_samples": args.mc_samples,
        "mc_seed": args.seed,
        "interpretation": "Illustrative analytic population-score/calibration tradeoff only. Not fitted to NS, not a novel theorem, and not the finite-E16 empirical interval metric.",
        "rows": rows,
    }
    (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    plot(args.output)
    for row in rows:
        print(
            f"sigma={row['sigma']:.2f}: relative CRPS penalty={row['relative_expected_crps_penalty_percent']:.6f}%; population coverage={100*row['population_coverage90']:.6f}%"
        )
    print(f"coverage derivative at sigma1={coverage_derivative:.8f}")


if __name__ == "__main__":
    main()
