"""Visualizations for each test case.

Every plotter takes the true trajectory [C, H, W, T], the predicted ensemble
[B, C, H, W, T] (both in physical units), the test dataset and a figure
directory. Use ``get_case_plotter(project)`` to pick the one for a case.
"""

import logging
import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from scisi.metrics.spectral import compute_enstrophy_error
from scisi.plotting.animation import create_animation_from_tensors
from scisi.plotting.plot_fields import plot_fields
from scisi.plotting.plot_point_distributions import plot_point_distributions
from scisi.plotting.spectrum import plot_enstrophy_spectrum

logger = logging.getLogger(__name__)


def _plotting_times(num_steps: int) -> list[int]:
    return [min(5, num_steps - 1), num_steps // 2, num_steps - 1]


def _animate(
    true: torch.Tensor, predicted: torch.Tensor, file_name: str
) -> None:
    """Animate the true field [H, W, T] next to each ensemble member [B, H, W, T]."""
    create_animation_from_tensors(
        [true] + [member for member in predicted],
        fps=10,
        file_name=file_name,
        colormaps="viridis",
        titles=["True"] + [f"Ensemble member {i}" for i in range(len(predicted))],
        vmin=np.min(true.numpy()),
        vmax=np.max(true.numpy()),
        normalize=False,
    )


def _plot_snapshots(
    true: torch.Tensor,
    predicted: torch.Tensor,
    label: str,
    figure_path: str,
    show: bool,
) -> None:
    """Plot true [H, W, T] and first ensemble member [H, W, T] at a few times."""
    times = _plotting_times(true.shape[-1])
    plot_fields(
        fields=[[true[..., t] for t in times], [predicted[..., t] for t in times]],
        titles=[
            [f"True {label} at t={t}" for t in times],
            [f"Predicted {label} at t={t}" for t in times],
        ],
        vmin=np.min(true.numpy()),
        vmax=np.max(true.numpy()),
        figsize=(15, 10),
        figure_path=figure_path,
        show=show,
    )


def plot_generic(true, predicted, test_dataset, figure_path, show=False) -> None:
    """Animation and snapshots of every channel."""
    for c in range(true.shape[0]):
        _animate(true[c], predicted[:, c], f"{figure_path}/channel_{c}.mp4")
        _plot_snapshots(
            true[c], predicted[0, c], f"channel {c}", f"{figure_path}/channel_{c}.png", show
        )


def plot_stochastic_navier_stokes(
    true, predicted, test_dataset, figure_path, show=False
) -> None:
    """Vorticity animation, snapshots and enstrophy spectrum."""
    _animate(true[0], predicted[:, 0], f"{figure_path}/vorticity.mp4")
    _plot_snapshots(
        true[0], predicted[0, 0], "vorticity", f"{figure_path}/vorticity.png", show
    )
    plot_enstrophy_spectrum(
        trajectories=[true[0], predicted[0, 0]],
        titles=["True", "Predicted"],
        figure_path=figure_path,
        show=show,
    )
    error, errors = compute_enstrophy_error(true[0], predicted[0, 0], 2 * torch.pi / 128)
    logger.info(f"Enstrophy error: {error:.4f} ± {errors.std():.4f}")


def _plot_urban_flow(
    true, predicted, figure_path, variable_names, show=False
) -> None:
    """Per-variable animations plus velocity-magnitude snapshots and point statistics."""
    true_magnitude = true[:3].pow(2).sum(dim=0).sqrt()
    predicted_magnitude = predicted[:, :3].pow(2).sum(dim=1).sqrt()

    for c, variable in enumerate(variable_names):
        _animate(true[c], predicted[:, c], f"{figure_path}/{variable}.mp4")
    _animate(
        true_magnitude, predicted_magnitude, f"{figure_path}/velocity_magnitude.mp4"
    )

    _plot_snapshots(
        true_magnitude,
        predicted_magnitude[0],
        "velocity magnitude",
        f"{figure_path}/velocity_magnitude.png",
        show,
    )
    plot_point_distributions(
        true_field=true_magnitude,
        predicted_fields=predicted_magnitude,
        points=[(32, 32), (64, 64), (96, 96)],
        figure_path=f"{figure_path}/distribution_at_points.png",
        show=show,
    )


def plot_udales(true, predicted, test_dataset, figure_path, show=False) -> None:
    _plot_urban_flow(
        true,
        predicted,
        figure_path,
        ["velocity_x", "velocity_y", "velocity_z", "temperature"],
        show,
    )


def plot_xie_and_castro(true, predicted, test_dataset, figure_path, show=False) -> None:
    _plot_urban_flow(
        true,
        predicted,
        figure_path,
        ["velocity_x", "velocity_y", "velocity_z", "channel_3"],
        show,
    )


def plot_knmi(true, predicted, test_dataset, figure_path, show=False) -> None:
    """Temperature animation, snapshots with marked grid cells and their time series."""
    true, predicted = true[0], predicted[:, 0]
    lat, lon = test_dataset.lat, test_dataset.lon
    cells = [(32, 64), (42, 10), (2, 120)]
    times = _plotting_times(true.shape[-1])
    vmin, vmax = np.min(true.numpy()), np.max(true.numpy())

    _animate(true, predicted, f"{figure_path}/temperature.mp4")

    num_cols = max(len(times), len(cells))
    plt.figure(figsize=(25, 20))
    for i, t in enumerate(times):
        for row, (field, label) in enumerate(
            [(true, "True"), (predicted[0], "Predicted")]
        ):
            plt.subplot(3, num_cols, row * num_cols + i + 1)
            plt.imshow(field[..., t], vmin=vmin, vmax=vmax, origin="lower")
            plt.title(f"{label} at t={t}")
            for cell in cells:
                plt.scatter(cell[1], cell[0], color="tab:red", marker="x", s=100)
    for i, (y, x) in enumerate(cells):
        plt.subplot(3, num_cols, 2 * num_cols + i + 1)
        plt.plot(true[y, x], label="True", linewidth=4, color="tab:green")
        for j, member in enumerate(predicted):
            plt.plot(
                member[y, x],
                label="Ensemble" if j == 0 else None,
                linewidth=1,
                color="tab:blue",
                alpha=0.6,
            )
        plt.title(f"Lat, Lon = {float(lat[y]):.2f}, {float(lon[x]):.2f}")
        plt.legend()
    plt.savefig(f"{figure_path}/temperature.png")
    plt.show() if show else plt.close()


CASE_PLOTTERS: dict[str, Callable] = {
    "stochastic_navier_stokes": plot_stochastic_navier_stokes,
    "udales": plot_udales,
    "xie_and_castro": plot_xie_and_castro,
    "knmi": plot_knmi,
    "weather": plot_generic,
}


def get_case_plotter(project: str) -> Callable:
    """Return the plotter for a case, falling back to per-channel plots."""
    if project not in CASE_PLOTTERS:
        logger.warning(f"No plotter for case '{project}', using generic plots")
    return CASE_PLOTTERS.get(project, plot_generic)


def plot_case(project, true, predicted, test_dataset, figure_path, show=False) -> None:
    os.makedirs(figure_path, exist_ok=True)
    get_case_plotter(project)(true, predicted, test_dataset, figure_path, show)
