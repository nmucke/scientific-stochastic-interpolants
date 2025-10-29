from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import random

from scisi.jax_cfd.kalman_filter import SpectralEnKF


# If you have a SpectralEnKF base, subclass it; otherwise adapt imports/attributes accordingly.
class ETKF(SpectralEnKF):
    """
    Ensemble Transform Kalman Filter (global transform, deterministic).
    Performs the analysis in ensemble space (N x N) and returns an analyzed ensemble
    in spectral space matching your existing code structure.
    """

    def __init__(self, *args: Any, regularize_eps: float = 1e-6, **kwargs: Any):
        """Initialize the ETKF."""
        super().__init__(*args, **kwargs)
        self.regularize_eps = (
            regularize_eps  # small numerical regularization for eigenvalues
        )

    # @jax.jit
    def analysis_step(  # type: ignore[override]
        self,
        forecast_ensemble_spectral: jnp.ndarray,
        observations: jnp.ndarray,
        obs_pos_flat: jnp.ndarray,
        key: jax.random.PRNGKey,
        inflation: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """
        Perform a single ETKF analysis and return the analyzed ensemble in spectral space.

        Args:
          forecast_ensemble_spectral: (N, ...) spectral representation per ensemble member
          observations: (n_obs,) observed values (matching obs_pos_flat)
          obs_pos_flat: (n_obs,) flattened indices in the physical grid where observations are located
          inflation: float multiplicative ensemble inflation applied to perturbations
        """
        N = self.ensemble_size
        eps = self.regularize_eps

        # Convert forecast to physical and flatten
        ensemble_physical = jax.vmap(self.spectral_to_physical)(
            forecast_ensemble_spectral
        )  # (N, ...) physical shape
        original_shape = ensemble_physical.shape

        # Flatten to (N, n_state) where n_state = prod(physical_shape[1:])
        ensemble_flat = ensemble_physical.reshape(N, -1)  # (N, n_state)

        # ---- Forecast stats in state space ----
        x_f_mean = jnp.mean(ensemble_flat, axis=0)  # (n_state,)
        X_f = (ensemble_flat - x_f_mean[None, :]).T  # (n_state, N)
        X_f = inflation * X_f  # apply inflation to perturbations

        # ---- Map ensemble into observation space ----
        # Build Y_f (n_obs, N) by selecting relevant positions from ensemble_flat
        # obs_pos_flat indexes into the flattened state
        # Ensure obs_pos_flat is 1D integer array (flatten if multi-dimensional)
        obs_pos_flat = jnp.atleast_1d(obs_pos_flat).astype(jnp.int32)
        obs_pos_flat = obs_pos_flat.ravel()  # Ensure truly 1D

        # Select observations: ensemble_flat is (N, n_state), obs_pos_flat is (n_obs,)
        # Use advanced indexing to select columns
        # Result should be (N, n_obs) or potentially higher-dimensional due to broadcasting
        Y_f_selected = ensemble_flat[:, obs_pos_flat]  # Could be (N, n_obs) or (N, ...)

        # Always reshape to 2D: flatten all dimensions except the first (ensemble axis)
        # This handles cases where indexing creates extra dimensions
        Y_f_selected = Y_f_selected.reshape(N, -1)  # (N, n_obs_total)

        # Transpose to (n_obs, N)
        Y_f = Y_f_selected.T  # (n_obs, N)

        y_f_mean = jnp.mean(Y_f, axis=1, keepdims=True)  # (n_obs, 1)
        Y_f_pert = Y_f - y_f_mean  # (n_obs, N)

        # If obs_noise_std is scalar or vector:
        obs_noise = self.obs_noise_std
        n_obs_actual = Y_f_pert.shape[0]  # Actual number of observations

        # Ensure R_sqrt_inv has shape (n_obs_actual,) for correct broadcasting
        if jnp.ndim(obs_noise) == 0:
            # Scalar: broadcast to all observations
            R_sqrt_inv = jnp.full(n_obs_actual, 1.0 / obs_noise)  # (n_obs,)
        else:
            # Vector: ensure it matches n_obs
            obs_noise = jnp.atleast_1d(obs_noise)
            obs_size = obs_noise.shape[0]  # type: ignore[attr-defined]
            if obs_size == 1:
                R_sqrt_inv = jnp.full(
                    n_obs_actual, 1.0 / obs_noise[0]  # type: ignore[index]
                )  # (n_obs,) # type: ignore[index]
            else:
                R_sqrt_inv = 1.0 / obs_noise  # (n_obs,)

        # Normalize observation perturbations by R^{1/2}
        # Broadcast: (n_obs, N) * (n_obs, 1) -> (n_obs, N)
        C = Y_f_pert * R_sqrt_inv[:, None]  # (n_obs, N)

        # ---- Ensemble-space computations (N x N) ----
        # S = C^T C  -> (N, N)
        S = C.T @ C

        # P_tilde = (N-1) I + S
        I_N = jnp.eye(N)
        P_tilde = (N - 1.0) * I_N + S

        # Innovation vector (y - y_f_mean), scaled by R^{-1/2}
        # Ensure observations are 1D and match Y_f shape
        observations_flat = jnp.atleast_1d(observations).ravel()  # Ensure 1D
        n_obs_from_y = Y_f.shape[0]
        # Reshape observations to match Y_f: if multi-dimensional, flatten; then take appropriate slice
        obs_total = observations_flat.shape[0]
        # Take first n_obs_from_y elements (assumes observations are in correct order)
        # If obs_total < n_obs_from_y, pad with zeros; if larger, take first n_obs_from_y
        pad_amount = jnp.maximum(0, n_obs_from_y - obs_total)
        observations_padded = jnp.pad(
            observations_flat, (0, pad_amount), mode="constant"
        )
        observations_matched = observations_padded[:n_obs_from_y]
        innov = (observations_matched.reshape(-1, 1) - y_f_mean) * R_sqrt_inv[
            :, None
        ]  # (n_obs, 1)
        r = C.T @ innov.ravel()  # (N,)

        # Solve for w_bar = P_tilde^{-1} r using eigendecomp (stable)
        eigvals, eigvecs = jnp.linalg.eigh(P_tilde)
        eigvals_safe = eigvals + eps
        eigvals_inv = 1.0 / eigvals_safe

        # Compute P_tilde^{-1} r
        w_bar = (eigvecs * eigvals_inv[None, :]) @ (eigvecs.T @ r)  # (N,)

        # Transform matrix T = sqrt((N-1) * P_tilde^{-1})
        eigvals_T = jnp.sqrt((N - 1.0) / eigvals_safe)
        T = (eigvecs * eigvals_T[None, :]) @ eigvecs.T  # (N, N)

        # Analysis perturbations (state space)
        X_a = X_f @ T  # (n_state, N)

        # Analysis mean in state space: x_a_mean = x_f_mean + X_f @ w_bar
        x_a_mean = x_f_mean + (X_f @ w_bar)  # (n_state,)

        # Build analyzed ensemble (n_state, N)
        analyzed_local = x_a_mean[:, None] + X_a  # (n_state, N)

        # Convert back to shape (N, n_state) then to physical shape and spectral
        analysis_flat = analyzed_local.T  # (N, n_state)
        nxny = int(analysis_flat.shape[1])
        # reshape to original physical dims
        physical_shape = ensemble_physical.shape[1:]  # e.g. (nx, ny)
        analysis_physical = analysis_flat.reshape((N,) + physical_shape)
        analysis_spectral = jax.vmap(self.physical_to_spectral)(analysis_physical)

        return analysis_spectral
