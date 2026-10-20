#!/usr/bin/env python3
"""Analytic diagnostic of Brownian-feature compression for an additive linear SDE.

Consider dX_t = a X_t dt + (1-t)dW_t, X_0=0, on [0,1]. Its terminal
state is Gaussian with variance integral exp(2*a*(1-r))*(1-r)**2 dr.
Given K equal-time increments of M_t=integral(1-r)dW_r, Gaussian
conditioning gives an exactly calculable conditional-mean predictor and
unexplained variance. This diagnostic describes same-path squared-error
projection, not the optimum of every PSD objective or nonlinear forecast.

Run from the repository root with the project interpreter:
  .venv/bin/python research/ito_maps/experiments/brownian_compression.py
Optionally add --mc-samples 100000 to check the actual WeightedBrownianPath
features against the analytic projected variance. No SDE solver is used.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def exponential_second_moment(rate, left, right):
    """Exact integral of u**2 * exp(rate*u) from left to right.

    The closed-form primitive is stable for the nonpositive rates and interval
    sizes in this diagnostic; rate=0 is handled by its polynomial limit.
    """
    if rate == 0:
        return (right - left) * (right * right + right * left + left * left) / 3

    def primitive(u):
        return math.exp(rate * u) * (u * u / rate - 2 * u / rate**2 + 2 / rate**3)

    return primitive(right) - primitive(left)


def compression_result(drift, num_features):
    terminal_variance = exponential_second_moment(2 * drift, 0, 1)
    boundaries = np.linspace(0, 1, num_features + 1)
    variances = []
    covariances = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        # u=1-r reverses the bounds of each physical-time segment.
        variances.append(exponential_second_moment(0, 1 - right, 1 - left))
        covariances.append(exponential_second_moment(drift, 1 - right, 1 - left))
    variances, covariances = np.asarray(variances), np.asarray(covariances)
    projected_variance = float(np.sum(covariances**2 / variances))
    # Avoid reporting tiny negative roundoff as a negative conditional variance.
    residual_variance = max(0.0, terminal_variance - projected_variance)
    return {
        "drift": float(drift),
        "num_features": int(num_features),
        "terminal_variance": terminal_variance,
        "projected_variance": projected_variance,
        "residual_variance": residual_variance,
        "residual_variance_fraction": residual_variance / terminal_variance,
        "retained_std_fraction": math.sqrt(projected_variance / terminal_variance),
        "irreducible_rmse": math.sqrt(residual_variance),
        "segment_variances": variances.tolist(),
        "projection_coefficients": (covariances / variances).tolist(),
    }


def check_integrals():
    """Independent high-order quadrature checks the closed-form calculations."""
    points, weights = np.polynomial.legendre.leggauss(64)
    for rate in (0, -1, -2, -4, -8, -16, -32):
        for left, right in ((0, 1), (0, 1 / 16), (7 / 16, 0.5), (15 / 16, 1)):
            u = left + (points + 1) * (right - left) / 2
            quadrature = (right - left) / 2 * np.sum(weights * u * u * np.exp(rate * u))
            np.testing.assert_allclose(
                exponential_second_moment(rate, left, right),
                quadrature,
                rtol=1e-10,
                atol=1e-15,
            )


def monte_carlo(rows, samples, seed):
    import torch
    from scisi.sampling.brownian_path import WeightedBrownianPath

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(seed)
    for row in rows:
        count = row["num_features"]
        path = WeightedBrownianPath.sample_like(
            torch.zeros(1, samples, 1, 1, dtype=torch.float64),
            gamma_multiplier=1.0,
            generator=generator,
        )
        features = path.features(0, 1, count).reshape(count, samples)
        coefficients = torch.tensor(row["projection_coefficients"], dtype=torch.float64)
        projected = (coefficients[:, None] * features).sum(0)
        row["mc_projected_variance"] = projected.var(unbiased=True).item()
        row["mc_projected_mean"] = projected.mean().item()
        row["mc_variance_relative_error"] = (
            row["mc_projected_variance"] / row["projected_variance"] - 1
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results/brownian_compression",
    )
    parser.add_argument("--mc-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=719)
    args = parser.parse_args()
    if args.mc_samples < 0:
        parser.error("--mc-samples must be nonnegative")
    check_integrals()
    rows = [
        compression_result(drift, count)
        for drift in (0, -1, -4, -16)
        for count in (1, 2, 4, 8, 16)
    ]
    if args.mc_samples:
        monte_carlo(rows, args.mc_samples, args.seed)
    for drift in (0, -1, -4, -16):
        residuals = [row["residual_variance"] for row in rows if row["drift"] == drift]
        assert all(b <= a + 1e-14 for a, b in zip(residuals, residuals[1:]))
    args.output.mkdir(parents=True, exist_ok=True)
    record = {
        "sde": "dX_t = a*X_t dt + (1-t)dW_t; X_0=0; t in [0,1]",
        "features": "K equal-time increments of M_t = integral_0^t (1-r)dW_r",
        "method": "Closed-form Gaussian conditional projection; independently checked by 64-point Gauss-Legendre quadrature",
        "interpretation": "Same-path squared-error optimum under finite path conditioning. Not a nonlinear Navier-Stokes guarantee and not a PSD optimization guarantee.",
        "mc_samples": args.mc_samples,
        "mc_seed": args.seed if args.mc_samples else None,
        "results": rows,
    }
    (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    scalar_fields = [
        key for key, value in rows[0].items() if not isinstance(value, list)
    ]
    with (args.output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(" drift    K    omitted variance (%)   retained std (%)")
    for row in rows:
        print(
            f"{row['drift']:6.0f} {row['num_features']:4d} {100*row['residual_variance_fraction']:23.6f} {100*row['retained_std_fraction']:18.6f}"
        )
    print(f"Saved {args.output/'results.json'} and results.csv")


if __name__ == "__main__":
    main()
