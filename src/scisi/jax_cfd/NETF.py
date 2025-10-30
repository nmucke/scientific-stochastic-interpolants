"""
Nonlinear Ensemble Transform Filter (NETF) for PDEs in Fourier space.

The NETF is a particle filter variant that uses importance sampling based on
the observation likelihood, followed by deterministic resampling in ensemble space.
It is particularly effective for nonlinear observation operators and non-Gaussian
distributions.

References:
- Tödter and Ahrens (2015): "A Second-Order Exact Ensemble Square Root Filter
  for Nonlinear Data Assimilation"
- Reich (2013): "A Nonparametric Ensemble Transform Method for Bayesian Inference"
"""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random

from scisi.jax_cfd.ENKF import SpectralEnKF


class SpectralNETF(SpectralEnKF):
    """
    Nonlinear Ensemble Transform Filter for PDEs in Fourier space.

    The NETF updates the ensemble by:
    1. Computing importance weights based on observation likelihood
    2. Applying a deterministic transform in ensemble space
    3. Preserving the ensemble mean and covariance structure

    Unlike EnKF/ETKF which assume Gaussian distributions, NETF can handle
    non-Gaussian posteriors through its particle filter-like approach.

    Parameters
    ----------
    grid_shape : tuple
        Shape of the physical space grid
    ensemble_size : int
        Number of ensemble members
    model_noise_std : float
        Standard deviation of model noise
    obs_noise_std : float
        Standard deviation of observation noise
    real_space : bool
        If True, use real FFT (rfft), else complex FFT
    regularize_eps : float
        Small regularization for numerical stability
    adaptive_inflation : bool
        If True, adaptively adjust inflation based on weights
    """

    def __init__(
        self,
        grid_shape: Tuple[int, ...],
        ensemble_size: int,
        model_noise_std: float,
        obs_noise_std: float,
        real_space: bool = True,
        regularize_eps: float = 1e-8,
        adaptive_inflation: bool = False,
    ):
        super().__init__(
            grid_shape=grid_shape,
            ensemble_size=ensemble_size,
            model_noise_std=model_noise_std,
            obs_noise_std=obs_noise_std,
            real_space=real_space,
        )
        self.regularize_eps = regularize_eps
        self.adaptive_inflation = adaptive_inflation

    def compute_log_likelihood(
        self,
        innovations: jnp.ndarray,
        obs_noise_std: float,
    ) -> jnp.ndarray:
        """
        Compute log-likelihood of observations given ensemble members.

        Assumes Gaussian observation errors: p(y|x) ~ N(H(x), R)

        Parameters
        ----------
        innovations : jnp.ndarray
            Innovation vectors (observations - H(ensemble)), shape (N, n_obs)
        obs_noise_std : float
            Observation noise standard deviation

        Returns
        -------
        jnp.ndarray
            Log-likelihoods for each ensemble member, shape (N,)
        """
        # Gaussian log-likelihood: -0.5 * (d^T R^{-1} d + log|2πR|)
        n_obs = innovations.shape[1]

        # For diagonal R = σ^2 I:
        # log p(y|x_i) = -0.5 * (||d_i||^2 / σ^2 + n_obs * log(2π σ^2))
        log_likelihood = -0.5 * (
            jnp.sum(innovations**2, axis=1) / (obs_noise_std**2)
            + n_obs * jnp.log(2 * jnp.pi * obs_noise_std**2)
        )

        return log_likelihood

    def compute_importance_weights(
        self,
        log_likelihoods: jnp.ndarray,
        regularization: float = 1e-10,
    ) -> jnp.ndarray:
        """
        Compute normalized importance weights from log-likelihoods.

        Parameters
        ----------
        log_likelihoods : jnp.ndarray
            Log-likelihoods for each ensemble member, shape (N,)
        regularization : float
            Small value to prevent numerical issues

        Returns
        -------
        jnp.ndarray
            Normalized importance weights, shape (N,)
        """
        # Subtract maximum for numerical stability
        log_weights = log_likelihoods - jnp.max(log_likelihoods)

        # Convert to weights
        weights = jnp.exp(log_weights)

        # Add small regularization to prevent zeros
        weights = weights + regularization

        # Normalize
        weights = weights / jnp.sum(weights)

        return weights

    def compute_transform_matrix(
        self,
        weights: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """
        Compute the NETF transform matrix.

        The NETF transform redistributes ensemble members according to their
        importance weights while preserving the ensemble mean and spread.

        Following Tödter & Ahrens (2015), the transform is:
        T = sqrt(N) * (W - 1/N * 1 * 1^T)^{1/2}

        where W is a diagonal matrix of weights.

        Parameters
        ----------
        weights : jnp.ndarray
            Importance weights, shape (N,)
        key : jax.random.PRNGKey
            Random key (unused but kept for interface)

        Returns
        -------
        jnp.ndarray
            Transform matrix T, shape (N, N)
        """
        N = len(weights)

        # Create the centered weight matrix: W - 1/N * 1 * 1^T
        W = jnp.diag(weights)
        ones = jnp.ones((N, 1))
        W_centered = W - (1.0 / N) * (ones @ ones.T)

        # Compute matrix square root using eigendecomposition
        # For a symmetric matrix A: A^{1/2} = U Λ^{1/2} U^T
        eigenvalues, eigenvectors = jnp.linalg.eigh(W_centered)

        # Regularize to handle numerical issues (some eigenvalues may be negative due to numerics)
        eigenvalues = jnp.maximum(eigenvalues, 0.0) + self.regularize_eps

        # Compute square root
        W_centered_sqrt = (
            eigenvectors @ jnp.diag(jnp.sqrt(eigenvalues)) @ eigenvectors.T
        )

        # Scale by sqrt(N) for proper ensemble variance
        T = jnp.sqrt(N) * W_centered_sqrt

        return T

    def effective_sample_size(self, weights: jnp.ndarray) -> Any:
        """
        Compute the effective sample size (ESS) of the ensemble.

        ESS measures the degeneracy of the importance weights.
        ESS = 1 / sum(w_i^2), ranging from 1 (all weight on one member)
        to N (uniform weights).

        Parameters
        ----------
        weights : jnp.ndarray
            Normalized importance weights, shape (N,)

        Returns
        -------
        float
            Effective sample size
        """
        return 1.0 / jnp.sum(weights**2)

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
        NETF Analysis step using importance sampling and ensemble transform.

        Parameters
        ----------
        forecast_ensemble_spectral : jnp.ndarray
            Forecasted ensemble in Fourier space (ensemble_size, *spectral_shape)
        observations : jnp.ndarray
            Observations in physical space (n_obs,)
        obs_indices : jnp.ndarray
            Indices of observation locations
        key : jax.random.PRNGKey
            Random key for stochastic components
        inflation : float
            Covariance inflation factor (applied to perturbations)
        localization_radius : float, optional
            Not implemented for NETF (requires local NETF variant)

        Returns
        -------
        jnp.ndarray
            Analysis ensemble in Fourier space
        """
        if localization_radius is not None:
            raise NotImplementedError(
                "Localization not yet implemented for NETF. "
                "Consider implementing local NETF or use LocalizedSpectralEnKF."
            )

        N = self.ensemble_size
        original_shape = forecast_ensemble_spectral.shape
        # original_shape: (N, *spectral_shape)

        # Flatten spectral ensemble for linear algebra operations
        ensemble_flat = forecast_ensemble_spectral.reshape(N, -1)
        # ensemble_flat shape: (N, n_state) where n_state = prod(spectral_shape)

        # Compute ensemble mean and perturbations
        x_mean = jnp.mean(ensemble_flat, axis=0)  # (n_state,)
        X_pert = ensemble_flat - x_mean  # (N, n_state)

        # Apply inflation to forecast perturbations
        X_pert = inflation * X_pert  # (N, n_state)

        # Map ensemble to observation space
        HX = jax.vmap(
            lambda u: self.observation_operator(
                u.reshape(original_shape[1:]), obs_indices
            )
        )(
            forecast_ensemble_spectral
        )  # (N, n_obs)

        # Compute innovations (observation - forecast for each member)
        innovations = observations[None, :] - HX  # (N, n_obs)

        # Compute log-likelihoods and importance weights
        log_likelihoods = self.compute_log_likelihood(innovations, self.obs_noise_std)
        weights = self.compute_importance_weights(log_likelihoods)

        # Compute effective sample size for diagnostics
        ess = self.effective_sample_size(weights)

        # Adaptive inflation based on weight degeneracy
        if self.adaptive_inflation:
            # If ESS is low, increase inflation
            adaptive_factor = jnp.where(ess < N / 2.0, 1.0 + 0.1 * (1.0 - ess / N), 1.0)
            X_pert = adaptive_factor * X_pert

        # Compute NETF transform matrix
        T = self.compute_transform_matrix(weights, key)
        # T shape: (N, N) - transforms in ensemble space

        # NETF update following Tödter & Ahrens (2015):
        # The analysis ensemble is: X_a = mean + T @ X'_f
        # where X'_f are the forecast perturbations (already computed and inflated)

        # The analysis mean is the weighted mean of the forecast ensemble
        x_mean_analysis = x_mean + X_pert.T @ (weights - 1.0 / N)  # (n_state,)

        # Transform the perturbations in ensemble space
        # T @ X_pert: (N, N) @ (N, n_state) = (N, n_state)
        # Each new ensemble member is a linear combination of the forecast perturbations
        X_pert_transformed = T @ X_pert  # (N, n_state)

        # Reconstruct analysis ensemble
        analysis_flat = x_mean_analysis[None, :] + X_pert_transformed  # (N, n_state)

        # Reshape back to original spectral shape
        analysis_ensemble = analysis_flat.reshape(original_shape)

        return analysis_ensemble


class LocalizedSpectralNETF(SpectralNETF):
    """
    Localized Nonlinear Ensemble Transform Filter for PDEs in Fourier space.

    Combines the localization approach (Gaspari-Cohn function) with the
    nonlinear ensemble transform, providing:
    - Reduced spurious correlations through localization
    - Better handling of nonlinear observation operators
    - Non-Gaussian posterior representation

    The localization is applied by performing local analysis updates in physical
    space, with each grid point updated using only nearby observations.

    Parameters
    ----------
    grid_shape : tuple
        Shape of the physical space grid
    ensemble_size : int
        Number of ensemble members
    model_noise_std : float
        Standard deviation of model noise
    obs_noise_std : float
        Standard deviation of observation noise
    localization_radius : float
        Radius of influence for localization (in grid points)
    real_space : bool
        If True, use real FFT (rfft), else complex FFT
    regularize_eps : float
        Small regularization for numerical stability
    adaptive_inflation : bool
        If True, adaptively adjust inflation based on weights
    chunk_size : int
        Number of grid points to process at once (default: 512)
        Reduce this if running out of memory
    """

    def __init__(
        self,
        grid_shape: Tuple[int, ...],
        ensemble_size: int,
        model_noise_std: float,
        obs_noise_std: float,
        localization_radius: float,
        real_space: bool = True,
        regularize_eps: float = 1e-8,
        adaptive_inflation: bool = False,
        chunk_size: int = 512,
    ):
        super().__init__(
            grid_shape=grid_shape,
            ensemble_size=ensemble_size,
            model_noise_std=model_noise_std,
            obs_noise_std=obs_noise_std,
            real_space=real_space,
            regularize_eps=regularize_eps,
            adaptive_inflation=adaptive_inflation,
        )
        self.localization_radius = localization_radius
        self.chunk_size = chunk_size  # Process grid points in chunks to save memory

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
        r = jnp.abs(distance) / radius

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

    def compute_distance_to_obs(
        self,
        state_coords: jnp.ndarray,
        obs_coords: jnp.ndarray,
        periodic: bool = True,
    ) -> jnp.ndarray:
        """
        Compute distances from state points to observation points.

        Parameters
        ----------
        state_coords : jnp.ndarray
            State coordinates (n_state,) for 1D or (n_state, 2) for 2D
        obs_coords : jnp.ndarray
            Observation coordinates (n_obs,) for 1D or (n_obs, 2) for 2D
        periodic : bool
            Whether to use periodic boundary conditions

        Returns
        -------
        jnp.ndarray
            Distance matrix (n_state, n_obs)
        """
        if self.ndim == 1:
            if periodic:
                L = self.grid_shape[0]
                diff = jnp.abs(state_coords[:, None] - obs_coords[None, :])
                dist = jnp.minimum(diff, L - diff)
            else:
                dist = jnp.abs(state_coords[:, None] - obs_coords[None, :])

        elif self.ndim == 2:
            if periodic:
                Lx, Ly = self.grid_shape
                diff = state_coords[:, None, :] - obs_coords[None, :, :]
                diff_x = jnp.abs(diff[:, :, 0])
                diff_y = jnp.abs(diff[:, :, 1])
                diff_x = jnp.minimum(diff_x, Lx - diff_x)
                diff_y = jnp.minimum(diff_y, Ly - diff_y)
                dist = jnp.sqrt(diff_x**2 + diff_y**2)
            else:
                diff = state_coords[:, None, :] - obs_coords[None, :, :]
                dist = jnp.linalg.norm(diff, axis=2)

        return dist

    def compute_localized_weights(
        self,
        innovations: jnp.ndarray,
        distances: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute localized importance weights for each state point.

        For each grid point, compute weights using only nearby observations
        (within localization radius).

        Parameters
        ----------
        innovations : jnp.ndarray
            Innovation vectors (observations - H(ensemble)), shape (N, n_obs)
        distances : jnp.ndarray
            Distances from state points to observations, shape (n_state, n_obs)

        Returns
        -------
        jnp.ndarray
            Localized weights for each state point, shape (n_state, N)
        """
        n_state = distances.shape[0]
        N = innovations.shape[0]

        # Compute localization matrix
        loc_matrix = self.gaspari_cohn(distances, self.localization_radius)
        # loc_matrix shape: (n_state, n_obs)

        def compute_weights_for_point(loc_weights: jnp.ndarray) -> jnp.ndarray:
            """Compute importance weights for a single grid point."""
            # loc_weights shape: (n_obs,)

            # Apply localization by weighting innovations
            # Use sqrt of localization weights for proper covariance localization
            # Set very small weights to zero to avoid numerical issues
            loc_weights_sqrt = jnp.where(
                loc_weights > 1e-10, jnp.sqrt(loc_weights), 0.0
            )

            # Localized innovations: weight by sqrt of localization function
            loc_innov = innovations * loc_weights_sqrt[None, :]
            # Shape: (N, n_obs)

            # Compute localized log-likelihoods
            # Sum over observations (already weighted by localization)
            log_lik = -0.5 * jnp.sum(loc_innov**2, axis=1) / (self.obs_noise_std**2)
            # Shape: (N,)

            # Check if any observations influence this point
            has_influence = jnp.sum(loc_weights) > 1e-10

            # Convert to weights (with numerical stability)
            log_lik = log_lik - jnp.max(log_lik)
            weights = jnp.exp(log_lik)
            weights = weights + self.regularize_eps
            weights = weights / jnp.sum(weights)

            # If no observations influence this point, use uniform weights
            uniform = jnp.ones(N) / N
            weights = jnp.where(has_influence, weights, uniform)

            return weights

        # Compute weights for all state points
        weights_all = jax.vmap(compute_weights_for_point)(loc_matrix)
        # Shape: (n_state, N)

        return weights_all

    def observation_operator_physical(
        self, u_physical: jnp.ndarray, obs_indices: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Observation operator for physical space.

        Parameters
        ----------
        u_physical : jnp.ndarray
            State in physical space
        obs_indices : jnp.ndarray
            Observation locations

        Returns
        -------
        jnp.ndarray
            Observations, shape (n_obs,)
        """
        if self.ndim == 1:
            return u_physical[obs_indices]
        elif self.ndim == 2:
            return u_physical[obs_indices[:, 0], obs_indices[:, 1]]

    def analysis_step(
        self,
        forecast_ensemble_spectral: jnp.ndarray,
        observations: jnp.ndarray,
        obs_indices: jnp.ndarray,
        key: jax.random.PRNGKey,
        inflation: float = 1.0,
        localization_radius: Optional[float] = None,
        periodic: bool = True,
    ) -> jnp.ndarray:
        """
        Localized NETF analysis step.

        Performs local analysis updates using importance sampling with
        Gaspari-Cohn localization.

        Parameters
        ----------
        forecast_ensemble_spectral : jnp.ndarray
            Forecasted ensemble in Fourier space
        observations : jnp.ndarray
            Observations in physical space
        obs_indices : jnp.ndarray
            Indices of observation locations
        key : jax.random.PRNGKey
            Random key
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
        loc_radius = (
            localization_radius
            if localization_radius is not None
            else self.localization_radius
        )
        N = self.ensemble_size

        # Transform ensemble to physical space
        forecast_ensemble_physical = jax.vmap(self.spectral_to_physical)(
            forecast_ensemble_spectral
        )
        original_shape = forecast_ensemble_physical.shape

        # Flatten physical ensemble
        ensemble_flat = forecast_ensemble_physical.reshape(N, -1)
        n_state = ensemble_flat.shape[1]

        # Compute ensemble mean and perturbations
        x_mean = jnp.mean(ensemble_flat, axis=0)
        X_pert = inflation * (ensemble_flat - x_mean)

        # Map ensemble to observation space
        HX = jax.vmap(lambda u: self.observation_operator_physical(u, obs_indices))(
            forecast_ensemble_physical
        )

        # Compute innovations
        innovations = observations[None, :] - HX  # (N, n_obs)

        # Compute distances from all state points to observations
        distances = self.compute_distance_to_obs(
            self.grid_coords, obs_indices, periodic=periodic
        )
        # distances shape: (n_state, n_obs)

        # Compute localized weights for each state point
        weights_local = self.compute_localized_weights(innovations, distances)
        # weights_local shape: (n_state, N)

        # Process grid points one-by-one in Python loop to save memory
        # This is slower but necessary for large grids
        analysis_list = []

        for i in range(n_state):
            weights_i = weights_local[i, :]  # (N,)
            pert_i = X_pert[:, i]  # (N,)
            mean_i = x_mean[i]

            # Weighted mean update
            mean_shift = jnp.dot(pert_i, weights_i - 1.0 / N)
            mean_analysis_i = mean_i + mean_shift

            # Transform perturbations using NETF
            # T = sqrt(N) * (W - 1/N*11^T)^{1/2}
            W = jnp.diag(weights_i)
            ones = jnp.ones((N, 1))
            W_centered = W - (1.0 / N) * (ones @ ones.T)

            # Eigen decomposition for matrix square root
            eigenvalues, eigenvectors = jnp.linalg.eigh(W_centered)
            eigenvalues = jnp.maximum(eigenvalues, 0.0) + self.regularize_eps

            # Apply T @ pert_i = sqrt(N) * U @ sqrt(Lambda) @ U^T @ pert_i
            v_proj = eigenvectors.T @ pert_i
            v_scaled = jnp.sqrt(eigenvalues) * v_proj
            pert_transformed = jnp.sqrt(N) * (eigenvectors @ v_scaled)

            # Reconstruct
            analysis_i = mean_analysis_i + pert_transformed

            analysis_list.append(analysis_i)

        # Stack into array (n_state, N)
        analysis_flat = jnp.stack(analysis_list, axis=0)

        # Transpose to (N, n_state)
        analysis_flat = analysis_flat.T

        # Reshape back to physical grid
        analysis_physical = analysis_flat.reshape(original_shape)

        # Transform back to spectral space
        analysis_spectral = jax.vmap(self.physical_to_spectral)(analysis_physical)

        return analysis_spectral
