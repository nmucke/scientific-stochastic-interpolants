import sys
from pathlib import Path

# Add this directory so `analytical_utils` is importable when run as script
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib.pyplot as plt
import torch
from torch.distributions import MultivariateNormal

from analytical_utils.kde_utils import prepare_samples
from analytical_utils.kl_divergence import (
    kl_divergence,
    wasserstein_distance,
)
from analytical_utils.likelihood import (
    FlowdasLikelihood,
    InterpolantLikelihood,
)
from analytical_utils.posterior_model import PosteriorModel
from analytical_utils.stochastic_interpolant import (
    AnalyticalDriftModel,
    AnalyticalStochasticInterpolant,
)
from analytical_utils.true_posterior import get_true_posterior
from scisi.models.interpolations import (
    LinearStochasticInterpolation,
    QuadraticStochasticInterpolation,
)

PLOT_ARGS = {
    "linewidth": 3,
    "markersize": 10,
    "linestyle": "-.",
}

# Domain
X_RANGE = (-1, 7)
Y_RANGE = (-1, 7)

# Grid
NBINS = 100
BATCH_SIZE = 2500
DIM = 2
SAMPLE_ARGS = {
    "nbins": NBINS,
    "x_range": X_RANGE,
    "y_range": Y_RANGE,
}

# Prior target
TARGET_MEAN = lambda x: x
TARGET_COV = lambda x: torch.eye(x.shape[1]).expand(x.shape[0], x.shape[1], x.shape[1])

# Stochastic Interpolant
NUM_STEPS = 250
X0_MEAN = 5
DIFFUSION_TERM = lambda t: 1.0 * (1 - t)
INTERPOLATION = LinearStochasticInterpolation(wiener_process=True)
TRUE_DRIFT_MODEL = AnalyticalDriftModel(
    INTERPOLATION, TARGET_MEAN, TARGET_COV, DIFFUSION_TERM
)
TRUE_DRIFT_MODEL_1 = AnalyticalDriftModel(
    INTERPOLATION, TARGET_MEAN, TARGET_COV, lambda t: 1.0 * (1 - t)
)

ORIGINAL_VARIANCE_LIST = [0.5, 1.0, 2.0]

METRICS = [
    ("Wasserstein", wasserstein_distance),
    ("KL-div", kl_divergence),
]


def build_models(obs_matrix: torch.Tensor, original_variance: float):
    """Return ``(name, posterior_model)`` pairs for every baseline."""
    shared = dict(
        obs_matrix=obs_matrix,
        drift_model=TRUE_DRIFT_MODEL,
        original_variance=original_variance,
    )
    return [
        (
            "New correction",
            PosteriorModel(
                TRUE_DRIFT_MODEL_1,
                InterpolantLikelihood(**shared, perturbation="new_correction"),
            ),
        ),
        (
            "FlowDAS",
            PosteriorModel(TRUE_DRIFT_MODEL, FlowdasLikelihood(**shared)),
        ),
        (
            "Interpolant",
            PosteriorModel(
                TRUE_DRIFT_MODEL_1,
                InterpolantLikelihood(**shared, perturbation=None),
            ),
        ),
        (
            "Interpolant (true pert.)",
            PosteriorModel(
                TRUE_DRIFT_MODEL_1,
                InterpolantLikelihood(
                    **shared,
                    perturbation="true",
                    num_quad=1000,
                    target_variance=1.0,
                ),
            ),
        ),
    ]


def format_results_table(
    results: dict,
    noise_values: list,
    model_names: list,
    metric_names: list,
) -> str:
    """Render ``results[model][sigma][metric]`` as an aligned text table.

    One row per model, one column per (sigma, metric), with an empty
    spacer column between consecutive sigma blocks.
    """
    col_w = 12
    name_w = max(len(n) for n in model_names) + 2
    spacer = "  "  # empty column between sigma blocks

    # --- Header line 1: sigma groups, centered over their metric columns ---
    group_w = col_w * len(metric_names)
    line1 = " " * name_w
    for j, sigma in enumerate(noise_values):
        if j > 0:
            line1 += spacer
        line1 += f"sigma^2 = {sigma:g}".center(group_w)

    # --- Header line 2: metric names ---
    line2 = " " * name_w
    for j in range(len(noise_values)):
        if j > 0:
            line2 += spacer
        for m in metric_names:
            line2 += m.rjust(col_w)

    # --- Separator ---
    sep = "-" * len(line2)

    # --- Body rows ---
    rows = []
    for name in model_names:
        row = name.ljust(name_w)
        for j, sigma in enumerate(noise_values):
            if j > 0:
                row += spacer
            for m in metric_names:
                v = results[name][sigma][m]
                row += f"{v:+{col_w}.4f}"
        rows.append(row)

    return "\n".join([line1, line2, sep, *rows])


def main() -> None:
    """Main function."""

    OBS = torch.tensor([[1.0, 1.0]])
    OBS_MATRIX = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    x0 = torch.ones(1, DIM) * X0_MEAN
    x = x0.repeat(BATCH_SIZE, 1)

    model_names = [
        "New correction",
        "FlowDAS",
        "Interpolant",
        "Interpolant (true pert.)",
    ]
    metric_names = [m[0] for m in METRICS]

    results = {name: {s: {} for s in ORIGINAL_VARIANCE_LIST} for name in model_names}
    kde_samples: dict = {name: {} for name in model_names}
    true_posterior_kde: dict = {}

    for sigma in ORIGINAL_VARIANCE_LIST:
        print(f"\n=== sigma^2 = {sigma:g} ===")

        OBS_COV = torch.eye(DIM) * sigma
        _, _, true_posterior_dist = get_true_posterior(
            x0, TARGET_MEAN, TARGET_COV, OBS_MATRIX, OBS_COV, OBS
        )
        true_posterior_kde[sigma] = prepare_samples(
            true_posterior_dist.sample((BATCH_SIZE,)), **SAMPLE_ARGS
        )
        true_posterior_samples = true_posterior_kde[sigma].samples

        for name, posterior_model in build_models(OBS_MATRIX, sigma):
            print(f"  sampling: {name}")
            kde = prepare_samples(
                posterior_model.sample(x, num_steps=NUM_STEPS, observations=OBS),
                **SAMPLE_ARGS,
            )
            kde_samples[name][sigma] = kde
            for metric_name, metric_fn in METRICS:
                results[name][sigma][metric_name] = float(
                    metric_fn(true_posterior_samples, kde.samples)
                )

    print()
    print(
        format_results_table(results, ORIGINAL_VARIANCE_LIST, model_names, metric_names)
    )
    print()

    # --- Convergence plots: one panel per metric, one curve per model ---
    fig, axes = plt.subplots(1, len(METRICS), figsize=(8 * len(METRICS), 6))
    if len(METRICS) == 1:
        axes = [axes]
    for ax, (metric_name, _) in zip(axes, METRICS):
        for name in model_names:
            ys = [results[name][s][metric_name] for s in ORIGINAL_VARIANCE_LIST]
            ax.loglog(ORIGINAL_VARIANCE_LIST, ys, label=name, **PLOT_ARGS)
        ax.set_xlabel("observation noise variance")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.show()

    # --- KDE heatmaps: rows = sigma, cols = [true posterior, each model] ---
    n_rows = len(ORIGINAL_VARIANCE_LIST)
    n_cols = 1 + len(model_names)
    _, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False
    )
    for i, sigma in enumerate(ORIGINAL_VARIANCE_LIST):
        tp = true_posterior_kde[sigma]
        axes[i, 0].pcolormesh(tp.xi, tp.yi, tp.zi, shading="gouraud")
        axes[i, 0].set_title(f"True posterior (sigma^2={sigma:g})")
        axes[i, 0].set_aspect("equal")
        for j, name in enumerate(model_names, start=1):
            kde = kde_samples[name][sigma]
            axes[i, j].pcolormesh(kde.xi, kde.yi, kde.zi, shading="gouraud")
            axes[i, j].set_title(f"{name} (sigma^2={sigma:g})")
            axes[i, j].set_aspect("equal")
    plt.tight_layout()
    plt.show()

    # --- Diagonal slice comparison, one subplot per sigma ---
    _, axes = plt.subplots(1, n_rows, figsize=(6 * n_rows, 5), squeeze=False)
    for i, sigma in enumerate(ORIGINAL_VARIANCE_LIST):
        ax = axes[0, i]
        ax.plot(true_posterior_kde[sigma].diag, label="True posterior", linewidth=2)
        for name in model_names:
            ax.plot(kde_samples[name][sigma].diag, label=name, **PLOT_ARGS)
        ax.set_title(f"Diagonal (sigma^2={sigma:g})")
        ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
