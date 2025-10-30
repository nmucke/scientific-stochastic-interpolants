import pdb
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import random

from scisi.jax_cfd.ENKF import SpectralEnKF


# If you have a SpectralEnKF base, subclass it; otherwise adapt imports/attributes accordingly.
class ETKF(SpectralEnKF):
    """
    Ensemble Transform Kalman Filter (global transform, deterministic).
    Performs the analysis in ensemble space (N x N) and returns an analyzed ensemble
    in spectral space matching your existing code structure.
    """

    def __init__(
        self, *args: Any, regularize_eps: float = 1e-6, rho: float = 0.5, **kwargs: Any
    ):
        """Initialize the ETKF."""
        super().__init__(*args, **kwargs)
        self.regularize_eps = (
            regularize_eps  # small numerical regularization for eigenvalues
        )
        self.rho = rho

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
        obs_indices : jnp.ndarray
            Indices of observation locations
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

        # Map ensemble to observation space
        HX = jax.vmap(
            lambda u: self.observation_operator(
                u.reshape(original_shape[1:]), obs_indices
            )
        )(
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
