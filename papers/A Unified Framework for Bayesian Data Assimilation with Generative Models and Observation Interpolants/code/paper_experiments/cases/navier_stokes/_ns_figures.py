"""Figure helpers for the NS case (fig:ns_trajectories, fig:ns_diagnostics).

These produce the two figures the spec maps to ``sections/results.tex``
(Section 8). They take a single :class:`AssimResult` plus the obs operator and
save PNGs under the figures include path. For the smoke run a low-resolution
version is fine; the full-scale run reuses the same code on a real ensemble.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scisi.metrics.calibration import rank_histogram
from scisi.metrics.spectral import radial_kinetic_energy_spectrum


def save_ns_trajectories(
    result,  # AssimResult
    obs_operator,
    out_path: str | Path,
) -> Path:
    """fig:ns_trajectories -- truth / observed / posterior mean / spread.

    One scenario, the final assimilated step. Saved to ``out_path``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    post = result.posterior_trajectory  # [E, C, H, W, T]
    true = result.true_trajectory  # [1, C, H, W, T]
    t = post.shape[-1] - 1

    true_state = true[0, 0, :, :, t]
    post_mean = post[:, 0, :, :, t].mean(dim=0)
    post_std = post[:, 0, :, :, t].std(dim=0)
    obs_mask = obs_operator.obs_indices_on_grid[0]

    vmin, vmax = float(true_state.min()), float(true_state.max())
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(true_state, vmin=vmin, vmax=vmax)
    axes[0].set_title("Truth")
    axes[1].imshow(obs_mask * true_state, vmin=vmin, vmax=vmax)
    axes[1].set_title("Observed")
    axes[2].imshow(post_mean, vmin=vmin, vmax=vmax)
    axes[2].set_title("Posterior mean")
    axes[3].imshow(post_std)
    axes[3].set_title("Posterior spread")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def save_ns_diagnostics(
    result,  # AssimResult
    len_field_history: int,
    out_path: str | Path,
) -> Path:
    """fig:ns_diagnostics -- (a) RMSE vs step, (b) energy spectra, (c) rank hist."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    post = result.posterior_trajectory
    true = result.true_trajectory
    E, _, _, _, T = post.shape

    steps = list(range(len_field_history, T))
    rmse_curve = [
        float(((post[:, 0, :, :, t].mean(0) - true[0, 0, :, :, t]) ** 2).mean().sqrt())
        for t in steps
    ]

    t_last = T - 1
    k_true, ek_true = radial_kinetic_energy_spectrum(true[0, 0, :, :, t_last])
    k_post, ek_post = radial_kinetic_energy_spectrum(
        post[:, 0, :, :, t_last].mean(0)
    )

    ranks = rank_histogram(
        post[:, 0, :, :, t_last], true[0, 0, :, :, t_last]
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(steps, rmse_curve, marker="o")
    axes[0].set_xlabel("assimilation step")
    axes[0].set_ylabel("ensemble-mean RMSE")
    axes[0].set_title("(a) RMSE vs step")

    eps = 1e-12
    axes[1].loglog(k_true + eps, ek_true + eps, label="truth")
    axes[1].loglog(k_post + eps, ek_post + eps, label="posterior")
    axes[1].set_xlabel("wavenumber k")
    axes[1].set_ylabel("E(k)")
    axes[1].set_title("(b) energy spectra")
    axes[1].legend()

    counts = ranks.detach().cpu().numpy()
    axes[2].bar(range(len(counts)), counts, edgecolor="black")
    axes[2].axhline(counts.sum() / len(counts), color="r", linestyle="--")
    axes[2].set_xlabel("rank")
    axes[2].set_title("(c) rank histogram")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


__all__ = ["save_ns_trajectories", "save_ns_diagnostics"]
