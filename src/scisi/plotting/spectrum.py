from typing import Optional

import matplotlib.pyplot as plt
import torch

from scisi.metrics.spectral import get_enstrophy_spectrum

COLORS = [
    "tab:green",
    "tab:blue",
    "tab:red",
    "tab:purple",
    "tab:orange",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]


def plot_enstrophy_spectrum(
    trajectories: list[torch.Tensor],
    titles: list[str],
    dx: float = 2 * torch.pi / 128,
    figure_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot the enstrophy spectrum of the trajectories."""

    num_steps = min(trajectory.shape[-1] for trajectory in trajectories)

    enstrophy: dict[str, list[torch.Tensor]] = {title: [] for title in titles}
    for title, trajectory in zip(titles, trajectories):
        for i in range(num_steps):
            ens, k = get_enstrophy_spectrum(trajectory[:, :, i], dx)
            enstrophy[title].append(ens)

    enstrophy_mean: dict[str, torch.Tensor] = {
        title: torch.stack(enstrophy[title]).mean(dim=0) for title in titles
    }
    enstrophy_std: dict[str, torch.Tensor] = {
        title: torch.stack(enstrophy[title]).std(dim=0) for title in titles
    }

    plt.figure()
    for i, title in enumerate(titles):
        plt.plot(
            k,
            enstrophy_mean[title],
            label=title,
            linewidth=2,
            color=COLORS[i],
        )
        plt.fill_between(
            k,
            enstrophy_mean[title] - enstrophy_std[title],
            enstrophy_mean[title] + enstrophy_std[title],
            alpha=0.2,
            color=COLORS[i],
        )
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.title("Enstrophy Spectrum")
    plt.xlabel("Wavenumber")
    plt.ylabel("Enstrophy")
    plt.grid(True)
    if figure_path is not None:
        plt.savefig(f"{figure_path}/enstrophy_spectrum.png")
    if show:
        plt.show()
    else:
        plt.close()