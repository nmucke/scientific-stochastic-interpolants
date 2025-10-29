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


class SpectralEnKF:
    """
    Ensemble Kalman Filter for PDEs in Fourier space.

    Parameters
    ----------
    grid_shape : tuple
        Shape of the physical space grid (nx,) for 1D or (nx, ny) for 2D
    ensemble_size : int
        Number of ensemble members
    model_noise_std : float
        Standard deviation of model noise in Fourier space
    obs_noise_std : float
        Standard deviation of observation noise in physical space
    real_space : bool
        If True, state is real-valued in Fourier space (e.g., using rfft)
        If False, state is complex-valued (e.g., using fft)
    """

    def __init__(
        self,
        grid_shape: Tuple[int, ...],
        ensemble_size: int,
        model_noise_std: float,
        obs_noise_std: float,
        real_space: bool = True,
    ):
        self.grid_shape = grid_shape
        self.ndim = len(grid_shape)
        self.ensemble_size = ensemble_size
        self.model_noise_std = model_noise_std
        self.obs_noise_std = obs_noise_std
        self.real_space = real_space

        # For real-valued fields, we use rfft which returns (N//2 + 1) coefficients
        if real_space:
            if self.ndim == 1:
                self.spectral_shape = (grid_shape[0] // 2 + 1,)
            elif self.ndim == 2:
                self.spectral_shape = (grid_shape[0], grid_shape[1] // 2 + 1)  # type: ignore[assignment]
        else:
            self.spectral_shape = grid_shape  # type: ignore[assignment]

    def physical_to_spectral(self, u_physical: jnp.ndarray) -> jnp.ndarray:
        """Transform from physical space to Fourier space."""
        if self.real_space:
            if self.ndim == 1:
                return jnp.fft.rfft(u_physical)
            elif self.ndim == 2:
                return jnp.fft.rfft2(u_physical)
        else:
            if self.ndim == 1:
                return jnp.fft.fft(u_physical)
            elif self.ndim == 2:
                return jnp.fft.fft2(u_physical)

    def spectral_to_physical(self, u_spectral: jnp.ndarray) -> jnp.ndarray:
        """Transform from Fourier space to physical space."""
        if self.real_space:
            if self.ndim == 1:
                return jnp.fft.irfft(u_spectral, n=self.grid_shape[0])
            elif self.ndim == 2:
                return jnp.fft.irfft2(u_spectral, s=self.grid_shape)
        else:
            if self.ndim == 1:
                return jnp.fft.ifft(u_spectral).real
            elif self.ndim == 2:
                return jnp.fft.ifft2(u_spectral).real

    def observation_operator(
        self, u_spectral: jnp.ndarray, obs_indices: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Observation operator H: maps Fourier coefficients to physical space observations.

        Parameters
        ----------
        u_spectral : jnp.ndarray
            State in Fourier space
        obs_indices : jnp.ndarray
            Indices of observation locations in physical space
            For 1D: shape (n_obs,)
            For 2D: shape (n_obs, 2)

        Returns
        -------
        jnp.ndarray
            Observations in physical space, shape (n_obs,)
        """
        # Transform to physical space
        u_physical = self.spectral_to_physical(u_spectral)

        # Extract observations at specified locations
        if self.ndim == 1:
            return u_physical[obs_indices]
        elif self.ndim == 2:
            return u_physical[obs_indices[:, 0], obs_indices[:, 1]]

    def forecast_step(
        self,
        ensemble_spectral: jnp.ndarray,
        dynamics: Callable,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """
        Forecast step in Fourier space.

        Parameters
        ----------
        ensemble_spectral : jnp.ndarray
            Ensemble in Fourier space (ensemble_size, *spectral_shape)
        dynamics : Callable
            Dynamics function operating on Fourier coefficients
        key : jax.random.PRNGKey
            Random key for model noise

        Returns
        -------
        jnp.ndarray
            Forecasted ensemble in Fourier space
        """
        rng_keys = jax.random.split(key, ensemble_spectral.shape[0])

        ensemble_spectral = jax.vmap(dynamics)(ensemble_spectral, rng_keys)

        return ensemble_spectral

    def gaspari_cohn(self, r: jnp.ndarray, c: float) -> jnp.ndarray:
        """
        Gaspari-Cohn localization function (5th order piecewise polynomial).

        Parameters
        ----------
        r : jnp.ndarray
            Distance(s)
        c : float
            Localization radius (half-width of compact support)

        Returns
        -------
        jnp.ndarray
            Localization weights in [0, 1]
        """
        # Normalize distance
        z = jnp.abs(r) / c

        # Gaspari-Cohn function (compact support at 2c)
        def piece1(z: jnp.ndarray) -> jnp.ndarray:  # 0 <= z < 1
            """Piece 1 of the Gaspari-Cohn function."""
            return 1 - 5 / 3 * z**2 + 5 / 8 * z**3 + 1 / 2 * z**4 - 1 / 4 * z**5

        def piece2(z: jnp.ndarray) -> jnp.ndarray:  # 1 <= z < 2
            """Piece 2 of the Gaspari-Cohn function."""
            return (
                4
                - 5 * z
                + 5 / 3 * z**2
                + 5 / 8 * z**3
                - 1 / 2 * z**4
                + 1 / 12 * z**5
                - 2 / (3 * z)
            )

        def piece3(z: jnp.ndarray) -> jnp.ndarray:  # z >= 2
            """Piece 3 of the Gaspari-Cohn function."""
            return jnp.zeros_like(z)

        # Piecewise evaluation
        result = jnp.where(z < 1, piece1(z), jnp.where(z < 2, piece2(z), piece3(z)))

        return result

    def compute_localization_matrix(
        self, obs_indices: jnp.ndarray, localization_radius: float
    ) -> jnp.ndarray:
        """
        Compute localization matrix for 2D fields.

        Parameters
        ----------
        obs_indices : jnp.ndarray
            Observation locations (n_obs, 2)
        localization_radius : float
            Localization radius in grid points

        Returns
        -------
        jnp.ndarray
            Localization matrix (n_grid_points, n_obs)
        """
        if self.ndim != 2:
            raise NotImplementedError("Localization only implemented for 2D")

        nx, ny = self.grid_shape

        # Create grid of all physical points
        x_grid = jnp.arange(nx)
        y_grid = jnp.arange(ny)
        X, Y = jnp.meshgrid(x_grid, y_grid, indexing="ij")

        # Flatten grid points: (nx*ny, 2)
        grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=1)

        # Compute all pairwise distances: (nx*ny, n_obs)
        # Need to handle periodic boundaries
        def periodic_distance(
            p1: jnp.ndarray, p2: jnp.ndarray, shape: Tuple[int, int]
        ) -> jnp.ndarray:
            """Compute periodic distance between points."""
            dx = jnp.abs(p1[0] - p2[0])
            dy = jnp.abs(p1[1] - p2[1])

            # Periodic wrapping
            dx = jnp.minimum(dx, shape[0] - dx)
            dy = jnp.minimum(dy, shape[1] - dy)

            return jnp.sqrt(dx**2 + dy**2)

        # Vectorized distance computation
        def compute_distances(grid_pt: jnp.ndarray) -> jnp.ndarray:
            """Compute distances from one grid point to all observations."""
            dists = jax.vmap(
                lambda obs: periodic_distance(grid_pt, obs, self.grid_shape)  # type: ignore[arg-type]
            )(obs_indices)
            return dists

        distances = jax.vmap(compute_distances)(grid_points)  # (nx*ny, n_obs)

        # Apply Gaspari-Cohn function
        localization_matrix = self.gaspari_cohn(distances, localization_radius)

        return localization_matrix  # Shape: (nx*ny, n_obs)

    def analysis_step(
        self,
        forecast_ensemble_spectral: jnp.ndarray,
        observations: jnp.ndarray,
        obs_indices: jnp.ndarray,
        key: jax.random.PRNGKey,
        inflation: float = 1.0,
        localization_radius: Optional[float] = None,
    ) -> jnp.ndarray:
        """
        Analysis step: update ensemble using physical space observations.

        Parameters
        ----------
        forecast_ensemble_spectral : jnp.ndarray
            Forecasted ensemble in Fourier space
        observations : jnp.ndarray
            Observations in physical space (n_obs,)
        obs_indices : jnp.ndarray
            Indices of observation locations
        key : jax.random.PRNGKey
            Random key
        inflation : float
            Covariance inflation factor
        localization_radius : float, optional
            Radius for covariance localization (in grid points).
            If None, no localization is applied.

        Returns
        -------
        jnp.ndarray
            Analysis ensemble in Fourier space
        """
        n_obs = len(observations)

        # Check if using localization
        use_localization = (localization_radius is not None) and (self.ndim == 2)

        if use_localization:
            # Localized analysis in physical space
            return self._localized_analysis(
                forecast_ensemble_spectral,
                observations,
                obs_indices,
                key,
                inflation,
                localization_radius,  # type: ignore[arg-type]
            )
        else:
            # Original non-localized analysis
            return self._standard_analysis(
                forecast_ensemble_spectral, observations, obs_indices, key, inflation
            )

    def _standard_analysis(
        self,
        forecast_ensemble_spectral: jnp.ndarray,
        observations: jnp.ndarray,
        obs_indices: jnp.ndarray,
        key: jax.random.PRNGKey,
        inflation: float,
    ) -> jnp.ndarray:
        """Standard non-localized analysis (original implementation)."""
        n_obs = len(observations)

        # Flatten spectral ensemble for easier manipulation
        # Shape: (ensemble_size, n_spectral_dofs)
        original_shape = forecast_ensemble_spectral.shape
        ensemble_flat = forecast_ensemble_spectral.reshape(self.ensemble_size, -1)

        # Handle complex arrays by treating real and imaginary parts separately
        if jnp.iscomplexobj(ensemble_flat):
            ensemble_real = ensemble_flat.real
            ensemble_imag = ensemble_flat.imag
            process_complex = True
        else:
            ensemble_real = ensemble_flat
            process_complex = False

        # Compute ensemble mean and perturbations for real part
        x_mean_real = jnp.mean(ensemble_real, axis=0)
        X_pert_real = inflation * (ensemble_real - x_mean_real)

        if process_complex:
            x_mean_imag = jnp.mean(ensemble_imag, axis=0)
            X_pert_imag = inflation * (ensemble_imag - x_mean_imag)

        # Map ensemble to observation space
        HX = jax.vmap(
            lambda u: self.observation_operator(
                u.reshape(original_shape[1:]), obs_indices
            )
        )(forecast_ensemble_spectral)

        HX_mean = jnp.mean(HX, axis=0)
        HX_pert = HX - HX_mean

        # Compute covariances
        # P_xy = X_pert^T @ HX_pert / (N - 1)
        Pxy_real = (X_pert_real.T @ HX_pert) / (self.ensemble_size - 1)

        if process_complex:
            Pxy_imag = (X_pert_imag.T @ HX_pert) / (self.ensemble_size - 1)

        # P_yy = HX_pert^T @ HX_pert / (N - 1) + R
        R = (self.obs_noise_std**2) * jnp.eye(n_obs)
        Pyy = (HX_pert.T @ HX_pert) / (self.ensemble_size - 1) + R

        # Kalman gain: K = P_xy @ (P_yy)^{-1}
        Pyy_inv = jnp.linalg.inv(Pyy)
        K_real = Pxy_real @ Pyy_inv

        if process_complex:
            K_imag = Pxy_imag @ Pyy_inv

        # Generate perturbed observations
        obs_noise = random.normal(key, (self.ensemble_size, n_obs)) * self.obs_noise_std
        perturbed_obs = observations + obs_noise

        # Update ensemble
        innovations = perturbed_obs - HX
        analysis_real = ensemble_real + innovations @ K_real.T

        if process_complex:
            analysis_imag = ensemble_imag + innovations @ K_imag.T
            analysis_flat = analysis_real + 1j * analysis_imag
        else:
            analysis_flat = analysis_real

        # Reshape back to original spectral shape
        analysis_ensemble = analysis_flat.reshape(original_shape)

        return analysis_ensemble

    def _localized_analysis(
        self,
        forecast_ensemble_spectral: jnp.ndarray,
        observations: jnp.ndarray,
        obs_indices: jnp.ndarray,
        key: jax.random.PRNGKey,
        inflation: float,
        localization_radius: float,
    ) -> jnp.ndarray:
        """
        Localized analysis for 2D fields.

        Uses Gaspari-Cohn localization to limit influence of observations
        to nearby grid points, reducing spurious correlations.
        """
        n_obs = len(observations)
        nx, ny = self.grid_shape
        n_grid = nx * ny

        original_shape = forecast_ensemble_spectral.shape

        # Convert ensemble to physical space for localized update
        # Shape: (ensemble_size, nx, ny)
        ensemble_physical = jax.vmap(self.spectral_to_physical)(
            forecast_ensemble_spectral
        )

        # Flatten physical ensemble: (ensemble_size, nx*ny)
        ensemble_flat = ensemble_physical.reshape(self.ensemble_size, -1)

        # Compute ensemble mean and perturbations
        x_mean = jnp.mean(ensemble_flat, axis=0)
        X_pert = inflation * (ensemble_flat - x_mean)

        # Map ensemble to observation space
        HX = jax.vmap(
            lambda u: self.observation_operator(
                u.reshape(original_shape[1:]), obs_indices
            )
        )(forecast_ensemble_spectral)

        # # Map ensemble to observation space
        # HX = jax.vmap(lambda u: self.observation_operator(
        #     u.reshape(nx, ny), obs_indices
        # ))(ensemble_physical)

        HX_mean = jnp.mean(HX, axis=0)
        HX_pert = HX - HX_mean

        # Compute localization matrix: (n_grid, n_obs)
        loc_matrix = self.compute_localization_matrix(obs_indices, localization_radius)

        # Compute localized covariances using Schur product
        # P_xy = X_pert^T @ HX_pert / (N - 1)
        Pxy = (X_pert.T @ HX_pert) / (self.ensemble_size - 1)  # (n_grid, n_obs)

        # Apply localization: element-wise multiplication
        Pxy_loc = Pxy * loc_matrix  # Schur product

        # P_yy = HX_pert^T @ HX_pert / (N - 1) + R
        R = (self.obs_noise_std**2) * jnp.eye(n_obs)
        Pyy = (HX_pert.T @ HX_pert) / (self.ensemble_size - 1) + R

        # For localized Pyy, we need to apply localization to observation-observation covariances
        # However, typically we only localize P_xy and keep Pyy as is
        # (since observations are usually independent)

        # Kalman gain: K = P_xy_loc @ (P_yy)^{-1}
        Pyy_inv = jnp.linalg.inv(Pyy)
        K_loc = Pxy_loc @ Pyy_inv  # (n_grid, n_obs)

        # Generate perturbed observations
        obs_noise = random.normal(key, (self.ensemble_size, n_obs)) * self.obs_noise_std
        perturbed_obs = observations + obs_noise

        # Update ensemble in physical space
        innovations = perturbed_obs - HX  # (ensemble_size, n_obs)
        analysis_flat = ensemble_flat + innovations @ K_loc.T  # (ensemble_size, n_grid)

        # Reshape to 2D: (ensemble_size, nx, ny)
        analysis_physical = analysis_flat.reshape(self.ensemble_size, nx, ny)

        # Transform back to spectral space
        analysis_spectral = jax.vmap(self.physical_to_spectral)(analysis_physical)

        return analysis_spectral

    def assimilate(
        self,
        ensemble_spectral: jnp.ndarray,
        observations: jnp.ndarray,
        obs_indices: jnp.ndarray,
        dynamics: Callable,
        key: jax.random.PRNGKey,
        inflation: float = 1.0,
        localization_radius: Optional[float] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Complete assimilation cycle."""
        key1, key2 = random.split(key)

        forecast_ensemble = self.forecast_step(ensemble_spectral, dynamics, key1)
        analysis_ensemble = self.analysis_step(
            forecast_ensemble,
            observations,
            obs_indices,
            key2,
            inflation,
            localization_radius,
        )

        return analysis_ensemble, forecast_ensemble
