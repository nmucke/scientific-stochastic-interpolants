import os
import pdb
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import jax_cfd.base.grids as grids
import matplotlib.pyplot as plt
import numpy as np
from jax import random

from jax_cfd_lib.ENKF import LocalizedSpectralEnKF, SpectralEnKF
from jax_cfd_lib.ETKF import LocalizedSpectralETKF, SpectralETKF
from jax_cfd_lib.navier_stokes_forward_model import set_up_forward_model


class ObservationOperator:
    def __init__(self, nx: int, ny: int, obs_indices: jnp.ndarray):
        self.nx = nx
        self.ny = ny
        self.obs_indices = obs_indices
        self.obs_coords = obs_indices

    def __call__(self, u_spectral: jnp.ndarray) -> jnp.ndarray:
        """
        Observation operator H: maps Fourier coefficients to physical space observations.
        """
        # Transform to physical space
        u_physical = jnp.fft.irfft2(u_spectral, s=(self.nx, self.ny))

        return u_physical[self.obs_indices[:, 0], self.obs_indices[:, 1]]

    def physical_observation_operator(self, u_physical: jnp.ndarray) -> jnp.ndarray:
        """
        Observation operator H: maps physical space to physical space observations.
        """

        return u_physical[self.obs_indices[:, 0], self.obs_indices[:, 1]]


def get_grid_observation_indices(
    data_size: tuple[int, int], skip_grid: int
) -> np.ndarray:
    """Get grid observation indices in (x, y) format.

    Returns:
        obs_indices: (n_obs, 2) array with [[x1, y1], [x2, y2], ...] format
    """
    H, W = data_size

    # Generate grid indices for every skip_grid-th point
    x_coords = np.arange(0, H, skip_grid)
    y_coords = np.arange(0, W, skip_grid)

    # Create meshgrid and stack into (n_obs, 2) format
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords, indexing="ij")
    obs_indices = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1)

    return obs_indices


def main() -> None:
    """Main function."""
    key = random.PRNGKey(123)

    skip_or_random = "skip"

    nx, ny = 256, 256
    grid = grids.Grid((nx, nx), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

    ensemble_size = 3000
    skip_grid = 16
    num_random_obs = 200

    obs_noise_std = 0.00005

    start_time = 5

    n_timesteps = 10

    Lx, Ly = 2 * jnp.pi, 2 * jnp.pi
    x = jnp.linspace(0, Lx, nx, endpoint=False)
    y = jnp.linspace(0, Ly, ny, endpoint=False)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    forward_model = set_up_forward_model(
        compile=True, use_true_model=False, stochastic=True, grid=grid
    )

    def pde(u_spectral: jnp.ndarray, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        u_final, _ = forward_model(u_spectral, rng_key)
        return u_final

    # Observation setup: sparse spatial observations

    if skip_or_random == "skip":
        obs_indices = get_grid_observation_indices((nx, ny), skip_grid)
        obs_indices = jnp.array(obs_indices)  # Convert to JAX array
        n_obs = obs_indices.shape[0]
        print(f"Number of observations: {n_obs}")
        save_str = f"paper_results/enkf_ns/obs_indices_{skip_or_random}_{skip_grid//2}"
        state_str = f"paper_results/enkf_ns/state_grid_{skip_or_random}_{skip_grid//2}"
    else:
        key, subkey = random.split(key)
        obs_x = random.randint(subkey, (num_random_obs,), 0, nx)
        key, subkey = random.split(key)
        obs_y = random.randint(subkey, (num_random_obs,), 0, ny)
        obs_indices = jnp.column_stack([obs_x, obs_y])
        save_str = (
            f"paper_results/enkf_ns/obs_indices_{skip_or_random}_{num_random_obs}"
        )
        state_str = f"paper_results/enkf_ns/state_{skip_or_random}_{num_random_obs}"

    n_obs = obs_indices.shape[0]
    observation_operator = ObservationOperator(nx, ny, obs_indices)

    if not os.path.exists(save_str):
        os.makedirs(os.path.dirname(save_str), exist_ok=True)

    jnp.savez(save_str, obs_indices // 2)

    # Initialize EnKF
    # enkf = LocalizedSpectralETKF(
    enkf = LocalizedSpectralEnKF(
        # enkf = LocalizedSpectralETKF(
        grid_shape=(nx, ny),
        ensemble_size=ensemble_size,
        model_noise_std=0,
        obs_noise_std=obs_noise_std,
        real_space=True,
        adaptive_localization=True,
        localization_radius=27,
        observation_operator=observation_operator,
    )

    # Load the trajectory
    true_trajectory_physical = np.load("trajectory.npz")["trajectory"]
    true_trajectory_physical = true_trajectory_physical[
        100 : 100 + start_time + n_timesteps
    ]

    jnp.savez(
        state_str + f"_field_history", true_trajectory_physical[100 : 100 + start_time]
    )

    true_trajectory_physical = true_trajectory_physical[start_time - 1 :]

    # Generate observations
    print("Generating sparse observations...")
    key, subkey = random.split(key)
    obs_noise = random.normal(subkey, (n_timesteps + 1, n_obs)) * obs_noise_std
    observations = (
        true_trajectory_physical[:, obs_indices[:, 0], obs_indices[:, 1]] + obs_noise
    )

    ensemble_physical = true_trajectory_physical[0].reshape(1, nx, ny)
    ensemble_physical = jnp.repeat(ensemble_physical, ensemble_size, axis=0)
    ensemble_spectral = jax.vmap(jnp.fft.rfft2)(ensemble_physical)

    print("Running 2D Spectral EnKF...")
    ensemble_means_physical: list[jnp.ndarray] = []
    for t in range(n_timesteps):
        key, subkey = random.split(key)

        ensemble_spectral, _ = enkf.assimilate(
            ensemble_spectral=ensemble_spectral,
            observations=observations[t + 1],
            dynamics=pde,
            key=subkey,
            inflation=1.0,
            # localization_radius=15
        )

        jnp.savez(
            state_str + f"_time_{t+start_time}",
            jax.vmap(jnp.fft.irfft2)(ensemble_spectral),
        )

        # Store mean in physical space
        mean_hat = jnp.mean(ensemble_spectral, axis=0)
        mean_phys = jnp.fft.irfft2(mean_hat, s=(nx, ny))
        ensemble_means_physical.append(mean_phys)

        if (t + 1) % 2 == 0:
            rmse = jnp.sqrt(
                jnp.mean((mean_phys - true_trajectory_physical[t + 1]) ** 2)
            )
            print(f"Step {t+1}/{n_timesteps}, RMSE: {rmse:.6f}")

    ensemble_means_physical = jnp.array(ensemble_means_physical)

    ensemble_physical = true_trajectory_physical[0].reshape(1, nx, ny)
    ensemble_physical = jnp.repeat(ensemble_physical, 5, axis=0)
    ensemble_spectral = jax.vmap(jnp.fft.rfft2)(ensemble_physical)
    rng_keys = jax.random.split(jax.random.PRNGKey(0), ensemble_spectral.shape[0])

    ensemble_means_physical_prior = []
    for t in range(n_timesteps):
        rng_keys = jax.random.split(rng_keys[0], ensemble_spectral.shape[0])
        ensemble_spectral = jax.vmap(pde)(ensemble_spectral, rng_keys)
        # mean_hat = jnp.mean(ensemble_spectral, axis=0)
        mean_hat = ensemble_spectral[0]
        mean_phys = jnp.fft.irfft2(mean_hat, s=(nx, ny))
        ensemble_means_physical_prior.append(mean_phys)

        if (t + 1) % 2 == 0:
            rmse = jnp.sqrt(
                jnp.mean((mean_phys - true_trajectory_physical[t + 1]) ** 2)
            )
            print(f"Step {t+1}/{n_timesteps}, RMSE: {rmse:.6f}")

    ensemble_means_physical_prior = jnp.array(ensemble_means_physical_prior)

    # Visualize results
    print("Creating 2D visualizations...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    times_to_plot = [0, n_timesteps // 2, n_timesteps]

    for i, t_idx in enumerate(times_to_plot):
        # True solution
        ax = axes[0, i]
        im = ax.imshow(
            true_trajectory_physical[t_idx].T,
            origin="lower",
            cmap="viridis",
            extent=[0, Lx, 0, Ly],
        )
        if t_idx == 0:
            ax.scatter(
                x[obs_indices[:, 0]],
                y[obs_indices[:, 1]],
                c="red",
                s=10,
                marker="x",
                alpha=0.5,
            )
        ax.set_title(f"True (t={t_idx})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax)

        # EnKF estimate
        ax = axes[1, i]
        if t_idx == 0:
            im = ax.imshow(
                true_trajectory_physical[0].T,
                origin="lower",
                cmap="viridis",
                extent=[0, Lx, 0, Ly],
            )
            ax.set_title(f"EnKF (t={t_idx})")
        else:
            im = ax.imshow(
                ensemble_means_physical[t_idx - 1].T,
                origin="lower",
                cmap="viridis",
                extent=[0, Lx, 0, Ly],
            )
            ax.set_title(f"EnKF (t={t_idx})")
        if t_idx == 0:
            ax.scatter(
                x[obs_indices[:, 0]],
                y[obs_indices[:, 1]],
                c="red",
                s=10,
                marker="x",
                alpha=0.5,
            )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax)

        # Prior estimate
        ax = axes[2, i]
        if t_idx == 0:
            im = ax.imshow(
                true_trajectory_physical[0].T,
                origin="lower",
                cmap="viridis",
                extent=[0, Lx, 0, Ly],
            )
            ax.set_title(f"Prior (t={t_idx})")
        else:
            im = ax.imshow(
                ensemble_means_physical_prior[t_idx - 1].T,
                origin="lower",
                cmap="viridis",
                extent=[0, Lx, 0, Ly],
            )
            ax.set_title(f"Prior (t={t_idx})")
        if t_idx == 0:
            ax.scatter(
                x[obs_indices[:, 0]],
                y[obs_indices[:, 1]],
                c="red",
                s=10,
                marker="x",
                alpha=0.5,
            )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(f"../../paper_results/enkf_ns/enkf_ns_results.png")
    plt.close()
    print("Plot saved!")

if __name__ == "__main__":
    main()
