import pdb
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random
from jax_cfd_lib.ENKF import SpectralEnKF
from jax.experimental.sparse.linalg import lobpcg_standard

# from jax_cfd_lib.ns_kalman import ObservationOperator


# If you have a SpectralEnKF base, subclass it; otherwise adapt imports/attributes accordingly.
class SpectralETKF(SpectralEnKF):
    """
    Ensemble Transform Kalman Filter (global transform, deterministic).
    Performs the analysis in ensemble space (N x N) and returns an analyzed ensemble
    in spectral space matching your existing code structure.
    """

    def __init__(  # type: ignore[no-untyped-def]
        self,
        grid_shape: Tuple[int, ...],
        ensemble_size: int,
        model_noise_std: float,
        obs_noise_std: float,
        observation_operator,
        real_space: bool = True,
        regularize_eps: float = 1e-6,
        rho: float = 0.5,
    ):
        """Initialize the ETKF.

        Parameters
        ----------
        grid_shape : Tuple[int, ...]
            Shape of the physical space grid
        ensemble_size : int
            Number of ensemble members
        model_noise_std : float
            Standard deviation of model noise
        obs_noise_std : float
            Standard deviation of observation noise
        real_space : bool
            If True, use real FFT (rfft), else complex FFT
        observation_operator : Optional[Any]
            ObservationOperator instance that maps state to observations
        regularize_eps : float
            Small numerical regularization for eigenvalues
        rho : float
            Inflation factor for ensemble transform
        """
        super().__init__(
            grid_shape=grid_shape,
            ensemble_size=ensemble_size,
            model_noise_std=model_noise_std,
            obs_noise_std=obs_noise_std,
            real_space=real_space,
            observation_operator=observation_operator,
        )
        self.regularize_eps = (
            regularize_eps  # small numerical regularization for eigenvalues
        )
        self.rho = rho

    def analysis_step(
        self,
        forecast_ensemble_spectral: jnp.ndarray,
        observations: jnp.ndarray,
        key: jax.random.PRNGKey,
        inflation: float = 1.0,
        localization_radius: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """
        ETKF Analysis step (deterministic ensemble update).

        The ETKF performs the analysis in ensemble space (N x N) rather than
        state space, making it more efficient and avoiding the need for
        perturbed observations.

        Parameters
        ----------
        forecast_ensemble_spectral : jnp.ndarray
            Forecasted ensemble in Fourier space (ensemble_size, *spectral_shape)
        observations : jnp.ndarray
            Observations in physical space (n_obs,)
        key : jax.random.PRNGKey
            Random key (unused in ETKF but kept for interface compatibility)
        inflation : float
            Covariance inflation factor
        localization_radius : float, optional
            Not implemented for ETKF yet (would require local ETKF variant)

        Returns
        -------
        jnp.ndarray
            Analysis ensemble in Fourier space
        """
        if localization_radius is not None:
            raise NotImplementedError(
                "Localization not yet implemented for ETKF. "
                "Use SpectralEnKF with localization or implement local ETKF."
            )

        n_obs = len(observations)
        N = self.ensemble_size

        # Flatten spectral ensemble for easier manipulation
        original_shape = forecast_ensemble_spectral.shape
        ensemble_flat = forecast_ensemble_spectral.reshape(N, -1)

        ensemble_real = ensemble_flat

        # Compute ensemble mean and perturbations (real part)
        x_mean_real = jnp.mean(ensemble_real, axis=0)
        X_pert_real = ensemble_real - x_mean_real  # (N, n_state)

        # Apply inflation to perturbations
        X_pert_real = inflation * X_pert_real

        # Map ensemble to observation space using the observation operator
        HX = jax.vmap(self.observation_operator)(
            forecast_ensemble_spectral
        )  # (N, n_obs)

        HX_mean = jnp.mean(HX, axis=0)  # (n_obs,)
        HX_pert = HX - HX_mean  # (N, n_obs)

        # Innovation (observation - forecast)
        innovation = observations - HX_mean  # (n_obs,)

        # Compute ensemble space matrices
        # P_yy = HX_pert^T @ HX_pert / (N - 1) + R
        R = (self.obs_noise_std**2) * jnp.eye(n_obs)

        # Work in ensemble space: (N-1) * I + HX_pert^T @ R^{-1} @ HX_pert
        # This is the key ETKF formulation
        R_inv = jnp.linalg.inv(R)

        # Ensemble space innovation covariance
        # C = (N-1)*I + Y^T @ R^{-1} @ Y, where Y = HX_pert^T
        # Shape: (N, N)
        C = (N - 1) * jnp.eye(N) + HX_pert @ R_inv @ HX_pert.T

        # Compute transform matrix T using eigendecomposition
        # T @ T^T = C^{-1} * rho^2 for inflation/regularization
        # Analysis perturbations: X_a' = X_f' @ T

        # Eigendecomposition for stability
        eigenvalues, eigenvectors = jnp.linalg.eigh(C)

        # Regularization to avoid numerical issues
        eigenvalues = jnp.maximum(eigenvalues, self.regularize_eps)

        # Transform matrix (square root of inverse with inflation)
        # T = U @ Lambda^{-1/2} @ U^T * sqrt((N-1)*rho)
        T = (
            eigenvectors
            @ jnp.diag(jnp.sqrt((N - 1) * self.rho / eigenvalues))
            @ eigenvectors.T
        )

        # Mean update weight: w = C^{-1} @ HX_pert^T @ R^{-1} @ innovation
        # Using eigendecomposition: C^{-1} = U @ Lambda^{-1} @ U^T
        C_inv = eigenvectors @ jnp.diag(1.0 / eigenvalues) @ eigenvectors.T
        w = C_inv @ HX_pert @ R_inv @ innovation  # (N,)

        # Update ensemble (real part)
        # Analysis mean: x_a = x_f + X_f' @ w
        # Analysis perturbations: X_a' = X_f' @ T
        x_mean_analysis_real = x_mean_real + X_pert_real.T @ w
        X_pert_analysis_real = X_pert_real.T @ T  # (n_state, N)

        # Reconstruct analysis ensemble
        analysis_real = (
            x_mean_analysis_real[:, None] + X_pert_analysis_real
        )  # (n_state, N)
        analysis_real = analysis_real.T  # (N, n_state)

        analysis_flat = analysis_real

        # Reshape back to original spectral shape
        analysis_ensemble = analysis_flat.reshape(original_shape)

        return analysis_ensemble


class LocalizedSpectralETKF(SpectralETKF):
    """
    Localized Ensemble Transform Kalman Filter (LETKF).

    Implements a local analysis version of the ETKF where the analysis is performed
    independently at each grid point using only nearby observations. This approach:

    1. Avoids global matrix operations, making it more efficient for large systems
    2. Naturally incorporates localization without modifying covariance matrices
    3. Is trivially parallelizable across grid points
    4. Works in spectral space with local analysis in physical space

    The LETKF performs analysis at each grid point using only observations within
    a localization radius, computing a local transform matrix for each point.

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
    localization_radius : float
        Radius of influence for localization (in grid points)
    real_space : bool
        If True, state is real-valued in Fourier space (e.g., using rfft)
    regularize_eps : float
        Small numerical regularization for eigenvalues
    rho : float
        Inflation factor for ensemble transform
    adaptive_localization : bool
        If True, automatically adjust localization radius based on ensemble statistics
    min_radius : float, optional
        Minimum allowed localization radius
    max_radius : float, optional
        Maximum allowed localization radius
    adaptation_method : str
        Method for adaptive localization ("correlation", "innovation", "hybrid")
    """

    def __init__(  # type: ignore[no-untyped-def]
        self,
        grid_shape: Tuple[int, ...],
        ensemble_size: int,
        model_noise_std: float,
        obs_noise_std: float,
        localization_radius: float,
        observation_operator,
        real_space: bool = True,
        regularize_eps: float = 1e-6,
        rho: float = 0.5,
        adaptive_localization: bool = False,
        min_radius: Optional[float] = None,
        max_radius: Optional[float] = None,
        adaptation_method: str = "hybrid",
    ):
        """Initialize the LocalizedSpectralETKF."""
        super().__init__(
            grid_shape=grid_shape,
            ensemble_size=ensemble_size,
            model_noise_std=model_noise_std,
            obs_noise_std=obs_noise_std,
            real_space=real_space,
            regularize_eps=regularize_eps,
            rho=rho,
            observation_operator=observation_operator,
        )
        self.localization_radius = localization_radius
        self.adaptive_localization = adaptive_localization
        self.adaptation_method = adaptation_method

        # Set default min/max radius for adaptive localization
        if min_radius is None:
            self.min_radius = max(2.0, localization_radius / 4.0)
        else:
            self.min_radius = min_radius

        if max_radius is None:
            self.max_radius = min(max(grid_shape) / 2.0, localization_radius * 4.0)
        else:
            self.max_radius = max_radius

        # History for adaptive adjustment
        self.radius_history: List[float] = []
        self.innovation_stats: List[float] = []

        # Precompute grid coordinates for distance calculations
        if self.ndim == 1:
            self.grid_coords = jnp.arange(grid_shape[0])
        elif self.ndim == 2:
            x = jnp.arange(grid_shape[0])
            y = jnp.arange(grid_shape[1])
            xx, yy = jnp.meshgrid(x, y, indexing="ij")
            self.grid_coords = jnp.stack([xx.flatten(), yy.flatten()], axis=1)

    def gaspari_cohn(self, distance: jnp.ndarray, radius: float) -> jnp.ndarray:
        """
        Gaspari-Cohn correlation function for localization.

        This is a fifth-order piecewise rational function with compact support.
        It smoothly tapers correlations to zero beyond 2*radius.

        Parameters
        ----------
        distance : jnp.ndarray
            Distances between points
        radius : float
            Localization radius (half-width of compact support)

        Returns
        -------
        jnp.ndarray
            Localization weights in [0, 1]
        """
        # Normalize by radius
        r = jnp.abs(distance) / radius

        # GC function has support on [0, 2]
        # For r in [0, 1]
        term1 = jnp.where(
            r <= 1, 1 - 5 / 3 * r**2 + 5 / 8 * r**3 + 1 / 2 * r**4 - 1 / 4 * r**5, 0.0
        )

        # For r in [1, 2]
        term2 = jnp.where(
            (r > 1) & (r <= 2),
            4
            - 5 * r
            + 5 / 3 * r**2
            + 5 / 8 * r**3
            - 1 / 2 * r**4
            + 1 / 12 * r**5
            - 2 / (3 * r),
            0.0,
        )

        return term1 + term2

    def compute_local_distances(
        self,
        state_idx: int,
        periodic: bool = True,
    ) -> jnp.ndarray:
        """
        Compute distances from a single state grid point to all observations.

        Parameters
        ----------
        state_idx : int
            Index of the state grid point (flattened)
        periodic : bool
            Whether to use periodic boundary conditions

        Returns
        -------
        jnp.ndarray
            Distances from state point to each observation
        """

        state_coord = self.grid_coords[state_idx]  # (2,)
        obs_coords = self.observation_operator.obs_coords  # (n_obs, 2)

        if periodic:
            Lx, Ly = self.grid_shape
            diff = state_coord[None, :] - obs_coords  # (n_obs, 2)
            diff_x = jnp.abs(diff[:, 0])
            diff_y = jnp.abs(diff[:, 1])
            diff_x = jnp.minimum(diff_x, Lx - diff_x)
            diff_y = jnp.minimum(diff_y, Ly - diff_y)
            distances = jnp.sqrt(diff_x**2 + diff_y**2)
        else:
            diff = state_coord[None, :] - obs_coords
            distances = jnp.linalg.norm(diff, axis=1)

        return distances

    def local_analysis_etkf(
        self,
        state_idx: int,
        X_pert_local: jnp.ndarray,
        x_mean_local: float,
        HX_pert_local: jnp.ndarray,
        HX_mean_local: jnp.ndarray,
        obs_local: jnp.ndarray,
        R_local: jnp.ndarray,
        localization_weights: jnp.ndarray,
    ) -> Tuple[float, jnp.ndarray]:
        """
        Perform ETKF analysis at a single grid point using local observations.

        Parameters
        ----------
        state_idx : int
            Index of the state grid point
        X_pert_local : jnp.ndarray
            Ensemble perturbations at this grid point (N,)
        x_mean_local : float
            Ensemble mean at this grid point
        HX_pert_local : jnp.ndarray
            Ensemble perturbations in observation space (N, n_obs_local)
        HX_mean_local : jnp.ndarray
            Ensemble mean in observation space (n_obs_local,)
        obs_local : jnp.ndarray
            Local observations (n_obs_local,)
        R_local : jnp.ndarray
            Local observation error covariance (n_obs_local, n_obs_local)
        localization_weights : jnp.ndarray
            Gaspari-Cohn weights for local observations (n_obs_local,)

        Returns
        -------
        tuple
            (analysis_mean, analysis_perturbations) at this grid point
        """
        N = self.ensemble_size

        # Check if localization weights are too small (effectively no local observations)
        # Use a smooth transition instead of hard cutoff to make it JAX-compatible
        total_weight = jnp.sum(localization_weights)
        weight_threshold = 1e-10

        # Scale factor that smoothly goes from 0 (no obs) to 1 (good obs)
        # Using tanh for smooth transition
        update_scale = jnp.tanh(total_weight / weight_threshold)

        # Apply localization weights to observation error covariance
        # R_local_weighted = R_local / localization_weights (diagonal scaling)
        # This effectively increases observation error for distant observations
        weight_sqrt = jnp.sqrt(localization_weights + 1e-10)

        # Weight the innovation and observation perturbations
        innovation_local = (obs_local - HX_mean_local) * weight_sqrt
        HX_pert_weighted = HX_pert_local * weight_sqrt[None, :]  # (N, n_obs_local)

        # Compute ensemble space covariance matrix
        # C = (N-1)*I + Y^T @ R^{-1} @ Y
        # With localization: Y is weighted by sqrt(localization)
        C = (N - 1) * jnp.eye(N) + HX_pert_weighted @ HX_pert_weighted.T

        # Eigendecomposition for numerical stability
        eigenvalues, eigenvectors = jnp.linalg.eigh(C)
        # eigenvalues, eigenvectors = lobpcg_standard(C)
        eigenvalues = jnp.maximum(eigenvalues, self.regularize_eps)

        # Transform matrix T = U @ Lambda^{-1/2} @ U^T * sqrt((N-1)*rho)
        T = (
            eigenvectors
            @ jnp.diag(jnp.sqrt((N - 1) * self.rho / eigenvalues))
            @ eigenvectors.T
        )

        # Mean update weight: w = C^{-1} @ Y^T @ R^{-1} @ innovation
        C_inv = eigenvectors @ jnp.diag(1.0 / eigenvalues) @ eigenvectors.T
        w = C_inv @ HX_pert_weighted @ innovation_local

        # Update at this grid point
        # Scale the update by the total weight to smoothly reduce updates when weights are small
        x_mean_analysis = x_mean_local + update_scale * jnp.dot(X_pert_local, w)
        # For perturbations, blend between forecast and analysis based on update_scale
        X_pert_analysis = (
            X_pert_local * (1 - update_scale) + (X_pert_local @ T.T) * update_scale
        )

        return x_mean_analysis, X_pert_analysis

    def analysis_step(
        self,
        forecast_ensemble_spectral: jnp.ndarray,
        observations: jnp.ndarray,
        key: jax.random.PRNGKey,
        inflation: float = 1.0,
        localization_radius: Optional[float] = None,
        periodic: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """
        Localized ETKF analysis step.

        Performs local analysis at each grid point using only nearby observations
        within the localization radius. This is the LETKF algorithm.

        Parameters
        ----------
        forecast_ensemble_spectral : jnp.ndarray
            Forecasted ensemble in Fourier space (ensemble_size, *spectral_shape)
        observations : jnp.ndarray
            Observations in physical space (n_obs,)
        key : jax.random.PRNGKey
            Random key (unused in ETKF but kept for interface compatibility)
        inflation : float
            Covariance inflation factor
        localization_radius : float, optional
            Override the instance localization radius
        periodic : bool
            Use periodic boundary conditions for distance calculations

        Returns
        -------
        jnp.ndarray
            Analysis ensemble in Fourier space
        """
        n_obs = len(observations)
        N = self.ensemble_size

        # Use provided radius or instance default
        loc_radius = (
            localization_radius
            if localization_radius is not None
            else self.localization_radius
        )

        # Transform ensemble to physical space for localized update
        forecast_ensemble_physical = jax.vmap(self.spectral_to_physical)(
            forecast_ensemble_spectral
        )

        # Flatten physical ensemble
        original_shape = forecast_ensemble_physical.shape
        ensemble_flat = forecast_ensemble_physical.reshape(N, -1)
        n_state = ensemble_flat.shape[1]

        # Compute ensemble mean and perturbations in physical space
        x_mean = jnp.mean(ensemble_flat, axis=0)  # (n_state,)
        X_pert = inflation * (ensemble_flat - x_mean)  # (N, n_state)

        # Map ensemble to observation space using the observation operator
        HX = jax.vmap(self.observation_operator.physical_observation_operator)(
            forecast_ensemble_physical
        )  # (N, n_obs)

        HX_mean = jnp.mean(HX, axis=0)  # (n_obs,)
        HX_pert = HX - HX_mean  # (N, n_obs)

        # Adaptive localization if enabled
        if self.adaptive_localization:
            innovations = observations - HX_mean
            # Use simplified adaptation based on innovation statistics
            innovation_norm = jnp.linalg.norm(innovations)
            expected_norm = jnp.sqrt(n_obs) * self.obs_noise_std
            normalized_innov = innovation_norm / (expected_norm + 1e-10)

            if normalized_innov > 1.5:
                loc_radius = min(loc_radius * 1.05, self.max_radius)
            elif normalized_innov < 0.7:
                loc_radius = max(loc_radius * 0.98, self.min_radius)

            self.radius_history.append(loc_radius)
            self.innovation_stats.append(float(innovation_norm))
            self.localization_radius = loc_radius

        # Perform local analysis at each grid point
        # This is the core LETKF loop - can be vectorized/vmapped
        def analyze_local_point(state_idx: int) -> Tuple[float, jnp.ndarray]:
            """Analyze a single grid point."""
            # Compute distances to all observations
            distances = self.compute_local_distances(state_idx, periodic)

            # Compute localization weights using Gaspari-Cohn
            loc_weights = self.gaspari_cohn(distances, loc_radius)

            # Get local ensemble perturbations and mean
            X_pert_local = X_pert[:, state_idx]  # (N,)
            x_mean_local = x_mean[state_idx]

            # Observation error covariance (diagonal)
            R_local = (self.obs_noise_std**2) * jnp.eye(n_obs)

            # Perform local ETKF update
            # The local_analysis_etkf handles the case where weights are very small
            mean_analysis, pert_analysis = self.local_analysis_etkf(
                state_idx,
                X_pert_local,
                x_mean_local,
                HX_pert,  # (N, n_obs)
                HX_mean,  # (n_obs,)
                observations,  # (n_obs,)
                R_local,
                loc_weights,
            )

            return mean_analysis, pert_analysis

        # Vectorize over all state grid points
        mean_analysis_all, pert_analysis_all = jax.vmap(
            lambda idx: analyze_local_point(idx)
        )(jnp.arange(n_state))

        # Reconstruct analysis ensemble in physical space
        # analysis_flat = mean + perturbations
        analysis_flat = mean_analysis_all[None, :] + pert_analysis_all.T  # (N, n_state)

        # Reshape back to physical grid
        analysis_physical = analysis_flat.reshape(original_shape)

        # Transform back to spectral space
        analysis_spectral = jax.vmap(self.physical_to_spectral)(analysis_physical)

        return analysis_spectral

    def get_adaptation_diagnostics(self) -> dict:
        """
        Get diagnostic information about adaptive localization.

        Returns
        -------
        dict
            Dictionary containing radius history and innovation statistics
        """
        return {
            "radius_history": self.radius_history,
            "innovation_stats": self.innovation_stats,
            "current_radius": self.localization_radius,
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
        }

    def reset_adaptation_history(self) -> None:
        """Reset the adaptation history."""
        self.radius_history = []
        self.innovation_stats = []
