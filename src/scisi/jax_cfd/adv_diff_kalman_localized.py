"""
Ensemble Kalman Filter for PDEs in Fourier Space

Handles systems where:
- State x is represented in Fourier (spectral) space
- Dynamics F operates in Fourier space
- Observations y are in physical space

State equation: x_{t+1} = F(x_t) + model_noise  (Fourier space)
Observation equation: y_{t+1} = H(x_{t+1}) + obs_noise  (Physical space)

where H is the transformation from Fourier to physical space at observation points.
"""

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random

from scisi.jax_cfd.ETKF import ETKF
from scisi.jax_cfd.kalman_filter import SpectralEnKF
from scisi.jax_cfd.letkf import LETKF
from scisi.jax_cfd.netf import SpectralNETF


class AdvectionDiffusion2D:
    """
    2D Advection-Diffusion equation in Fourier space:
    du/dt = -c_x * du/dx - c_y * du/dy + nu * (d²u/dx² + d²u/dy²)

    Solved using pseudo-spectral method.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        Lx: float,
        Ly: float,
        cx: float,
        cy: float,
        nu: float,
        dt: float,
    ):
        """
        Parameters
        ----------
        nx, ny : int
            Number of grid points
        Lx, Ly : float
            Domain size
        cx, cy : float
            Advection velocities
        nu : float
            Diffusion coefficient
        dt : float
            Time step
        """
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.cx, self.cy = cx, cy
        self.nu = nu
        self.dt = dt

        # Wavenumbers
        self.kx = 2 * jnp.pi * jnp.fft.fftfreq(nx, Lx / nx)
        self.ky = 2 * jnp.pi * jnp.fft.rfftfreq(ny, Ly / ny)
        self.KX, self.KY = jnp.meshgrid(self.kx, self.ky, indexing="ij")

        # Linear operator (diffusion)
        self.linear_op = -self.nu * (self.KX**2 + self.KY**2)

    def step(
        self, u_hat: jnp.ndarray, rng_key: jax.random.PRNGKey, n_steps: int = 1
    ) -> jnp.ndarray:
        """Time step using ETDRK4 (Exponential Time Differencing RK4)."""

        def nonlinear(u_hat_: jnp.ndarray) -> jnp.ndarray:
            """Nonlinear term of the advection-diffusion equation."""
            # Advection term: -c_x * du/dx - c_y * du/dy
            ux_hat = 1j * self.KX * u_hat_
            uy_hat = 1j * self.KY * u_hat_

            # Transform to physical space
            ux = jnp.fft.irfft2(ux_hat, s=(self.nx, self.ny))
            uy = jnp.fft.irfft2(uy_hat, s=(self.nx, self.ny))

            nonlinear_term = -self.cx * ux - self.cy * uy
            return jnp.fft.rfft2(nonlinear_term)

        # ETDRK4 coefficients
        E = jnp.exp(self.dt * self.linear_op)
        E2 = jnp.exp(self.dt * self.linear_op / 2)

        u_current = u_hat
        for _ in range(n_steps):
            Nu = nonlinear(u_current)
            a = E2 * u_current + self.dt * Nu / 2
            Na = nonlinear(a)
            b = E2 * u_current + self.dt * Na / 2
            Nb = nonlinear(b)
            c = E2 * a + self.dt * (2 * Nb - Nu) / 2
            Nc = nonlinear(c)

            u_current = E * u_current + self.dt * (Nu + 2 * Na + 2 * Nb + Nc) / 6

        return u_current


def run_2d_enkf_example() -> None:
    """Run ETKF with 2D advection-diffusion."""

    key = random.PRNGKey(123)

    # Domain parameters
    nx, ny = 64, 64
    Lx, Ly = 1.0, 1.0

    # Physics parameters
    cx, cy = 0.5, 0.3  # Advection velocities
    nu = 0.001  # Diffusion
    dt = 0.01
    n_timesteps = 100

    # Observation setup: sparse spatial observations
    n_obs = 25
    key, subkey = random.split(key)
    obs_x = random.randint(subkey, (n_obs,), 0, nx)
    key, subkey = random.split(key)
    obs_y = random.randint(subkey, (n_obs,), 0, ny)
    obs_indices = jnp.column_stack([obs_x, obs_y])

    # ETKF parameters
    ensemble_size = 5000
    model_noise_std = 0.05
    obs_noise_std = 0.02

    # Initialize PDE solver
    pde = AdvectionDiffusion2D(nx, ny, Lx, Ly, cx, cy, nu, dt)

    # Initialize EnKF
    etkf = ETKF(
        grid_shape=(nx, ny),
        ensemble_size=ensemble_size,
        model_noise_std=model_noise_std,
        obs_noise_std=obs_noise_std,
        real_space=True,
    )

    netf = SpectralNETF(
        grid_shape=(nx, ny),
        ensemble_size=ensemble_size,
        model_noise_std=model_noise_std,
        obs_noise_std=obs_noise_std,
        real_space=True,
    )

    # Create initial condition: Gaussian blob
    x = jnp.linspace(0, Lx, nx, endpoint=False)
    y = jnp.linspace(0, Ly, ny, endpoint=False)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    u0 = jnp.exp(-((X - 0.3) ** 2 + (Y - 0.3) ** 2) / 0.02)
    u0_hat = jnp.fft.rfft2(u0)

    # Generate true trajectory
    print("Generating true 2D trajectory...")
    true_state_hat = u0_hat
    true_trajectory_physical = [jnp.fft.irfft2(true_state_hat, s=(nx, ny))]

    for t in range(n_timesteps):
        key, subkey = random.split(key)
        noise_real = random.normal(subkey, true_state_hat.shape) * model_noise_std * 0.3
        key, subkey = random.split(key)
        noise_imag = random.normal(subkey, true_state_hat.shape) * model_noise_std * 0.3
        noise = noise_real + 1j * noise_imag

        true_state_hat = pde.step(true_state_hat, key) + noise
        true_trajectory_physical.append(jnp.fft.irfft2(true_state_hat, s=(nx, ny)))

    true_trajectory_physical = jnp.array(true_trajectory_physical)

    # Generate observations
    print("Generating sparse observations...")
    key, subkey = random.split(key)
    obs_noise = random.normal(subkey, (n_timesteps + 1, n_obs)) * obs_noise_std
    observations = (
        true_trajectory_physical[:, obs_indices[:, 0], obs_indices[:, 1]] + obs_noise  # type: ignore[call-overload]
    )

    # Initialize ensemble
    u00 = jnp.exp(-((X - 0.1) ** 2 + (Y - 0.1) ** 2) / 0.02)
    key, subkey = random.split(key)
    perturbations = random.normal(subkey, (ensemble_size, nx, ny)) * 0.1
    ensemble_physical = u00 + perturbations
    ensemble_spectral = jax.vmap(lambda u: jnp.fft.rfft2(u))(ensemble_physical)

    # Run EnKF
    print("Running 2D Spectral EnKF...")
    ensemble_means_physical = []

    for t in range(n_timesteps):
        key, subkey = random.split(key)

        ensemble_spectral, _ = etkf.assimilate(
            ensemble_spectral=ensemble_spectral,
            observations=observations[t + 1],
            obs_indices=obs_indices,
            dynamics=pde.step,
            key=subkey,
            inflation=1.02,
            localization_radius=50.5,
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

    # Simulate data without assimilation
    non_localized_means_physical = []

    # Initialize ensemble
    u00 = jnp.exp(-((X - 0.1) ** 2 + (Y - 0.1) ** 2) / 0.02)
    key, subkey = random.split(key)
    perturbations = random.normal(subkey, (ensemble_size, nx, ny)) * 0.1
    ensemble_physical = u00 + perturbations
    ensemble_spectral = jax.vmap(lambda u: jnp.fft.rfft2(u))(ensemble_physical)

    for t in range(n_timesteps):
        key, subkey = random.split(key)

        ensemble_spectral, _ = netf.assimilate(
            ensemble_spectral=ensemble_spectral,
            observations=observations[t + 1],
            obs_indices=obs_indices,
            dynamics=pde.step,
            key=subkey,
            inflation=1.02,
        )

        # Store mean in physical space
        mean_hat = jnp.mean(ensemble_spectral, axis=0)
        mean_phys = jnp.fft.irfft2(mean_hat, s=(nx, ny))
        non_localized_means_physical.append(mean_phys)

        if (t + 1) % 2 == 0:
            rmse = jnp.sqrt(
                jnp.mean((mean_phys - true_trajectory_physical[t + 1]) ** 2)
            )
            print(f"Step {t+1}/{n_timesteps}, RMSE: {rmse:.6f}")

    non_localized_means_physical = jnp.array(non_localized_means_physical)

    # Optionally, you can compare RMSE of non-assimilated data
    rmse_no_assim = jnp.sqrt(
        jnp.mean(
            (non_localized_means_physical - true_trajectory_physical[1:]) ** 2,  # type: ignore[operator]
            axis=(1, 2),
        )
    )
    print(f"\nFinal RMSE without assimilation: {rmse_no_assim[-1]:.6f}")
    print(f"Mean RMSE without assimilation: {jnp.mean(rmse_no_assim):.6f}")

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

        # Non-assimilated data
        ax = axes[2, i]
        if t_idx == 0:
            im = ax.imshow(
                true_trajectory_physical[0].T,
                origin="lower",
                cmap="viridis",
                extent=[0, Lx, 0, Ly],
            )
            ax.set_title(f"Non-assimilated (t={t_idx})")
        else:
            im = ax.imshow(
                non_localized_means_physical[t_idx - 1].T,
                origin="lower",
                cmap="viridis",
                extent=[0, Lx, 0, Ly],
            )
            ax.set_title(f"Non-assimilated (t={t_idx})")
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
    plt.show()
    print("Plot saved!")

    # Compute RMSE
    rmse_time = jnp.sqrt(
        jnp.mean(
            (ensemble_means_physical - true_trajectory_physical[1:]) ** 2, axis=(1, 2)  # type: ignore[operator]
        )
    )
    print(f"\nFinal RMSE: {rmse_time[-1]:.6f}")
    print(f"Mean RMSE: {jnp.mean(rmse_time):.6f}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING 2D ADVECTION-DIFFUSION EXAMPLE")
    print("=" * 70 + "\n")

    run_2d_enkf_example()

    print("\n✓ All examples completed successfully!\n")
