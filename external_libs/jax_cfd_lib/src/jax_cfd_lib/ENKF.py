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

from typing import Any, Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random

# from jax_cfd_lib.ns_kalman import ObservationOperator


# --------------------------------------------------------------------------- #
# Correlation-based localization (Vossepoel, Evensen & van Leeuwen, 2025, MWR,
# "Adaptive Correlation- and Distance-Based Localization for Iterative Ensemble
# Smoothers").
#
# Unlike Gaspari-Cohn DISTANCE localization (taper by physical distance), this
# selects and tapers observations by their ENSEMBLE CORRELATION to the updated
# state variable, which is expected to help for sparse sensors. The two helpers
# below are shared by the localized EnKF and (L)ETKF classes; they replace the
# distance taper with a correlation-based selection + observation-error inflation
# and are only invoked when ``localization_type == "correlation"`` (the default
# remains the existing distance-based Gaspari-Cohn path -- see the classes).
# --------------------------------------------------------------------------- #


def correlation_state_obs(
    X_pert: jnp.ndarray,
    HX_pert: jnp.ndarray,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Ensemble correlation rho(l, j) between every state var and measurement.

    Parameters
    ----------
    X_pert : jnp.ndarray
        State-anomaly matrix ``A`` (state perturbations about the ensemble mean),
        shape ``(N, n_state)``.
    HX_pert : jnp.ndarray
        Predicted-measurement anomaly matrix ``Y`` (``H`` applied to the forecast
        members, about their mean), shape ``(N, n_obs)``.
    eps : float
        Floor on the per-variable / per-measurement standard deviations to guard
        the normalisation against zero-spread directions.

    Returns
    -------
    jnp.ndarray
        Correlation matrix ``rho``, shape ``(n_state, n_obs)``, entries in
        ``[-1, 1]``. ``C_zy = A^T Y`` normalised by the per-row std of ``A`` and
        per-col std of ``Y`` (Pearson correlation across ensemble members).
    """
    # Cross-covariance C_zy = A^T Y / (N - 1); the (N-1) cancels in the
    # correlation normalisation, so we keep raw sums and normalise by stds.
    cov_zy = X_pert.T @ HX_pert  # (n_state, n_obs), == sum over members
    std_state = jnp.sqrt(jnp.sum(X_pert**2, axis=0))  # (n_state,)
    std_obs = jnp.sqrt(jnp.sum(HX_pert**2, axis=0))  # (n_obs,)
    denom = std_state[:, None] * std_obs[None, :] + eps
    return cov_zy / denom


def correlation_localization_weights(
    rho: jnp.ndarray,
    corr_threshold: float,
    inflation_max: float = 4.0,
    inflation_beta: float = 0.5,
) -> jnp.ndarray:
    """Correlation-based selection + obs-error-inflation taper (Eq. of the paper).

    For state variable ``l`` and measurement ``j`` with correlation distance
    ``d_c = 1 - |rho(l, j)|`` and truncation distance ``d_t = 1 - corr_threshold``:

    * **Truncation.** Retain ``j`` only if ``|rho| >= corr_threshold`` (i.e.
      ``d_c <= d_t``); truncated measurements get weight 0 so they do not enter
      the update of ``l``.
    * **Observation-error inflation.** A retained measurement's error variance is
      inflated by ``E_inf = 1`` if ``d_c <= beta * d_t`` else
      ``E_inf = exp( ((d_c - beta * d_t) / b)^2 )`` with
      ``b = (1 - beta) * d_t / sqrt(ln(E_max))``; ``E_inf`` rises smoothly from 1
      at ``d_c = beta * d_t`` toward ``E_max`` near ``d_c = d_t``.

    The returned weight ``w = 1 / E_inf`` (0 for truncated obs) is the localization
    weight in the SAME convention the distance-based path uses (it scales the
    observation precision; the existing classes fold it into ``R`` exactly like
    the Gaspari-Cohn weight, via ``sqrt(w)`` on the obs perturbations / innovation,
    or a Schur product on the covariance). This makes the correlation path a
    drop-in replacement for the GC taper inside the existing analysis math.

    Parameters
    ----------
    rho : jnp.ndarray
        Correlation matrix from :func:`correlation_state_obs`, shape
        ``(n_state, n_obs)``.
    corr_threshold : float
        Truncation threshold ``r_t`` on ``|rho|``. First-guess ``3 / sqrt(N)``
        (removes ~99.7% of spurious zero-correlation sampling noise); paper found
        tuned values 0.3--0.4 work well.
    inflation_max : float
        Maximum observation-error inflation ``E_max`` reached near ``d_c = d_t``
        (paper default 4--8).
    inflation_beta : float
        Fraction ``beta`` of ``d_t`` below which no inflation is applied
        (``E_inf = 1``); paper default 0.5.

    Returns
    -------
    jnp.ndarray
        Localization weights ``w(l, j) = 1 / E_inf`` in ``[0, 1]`` (0 = truncated),
        shape ``(n_state, n_obs)``.
    """
    d_t = 1.0 - corr_threshold
    d_c = 1.0 - jnp.abs(rho)  # correlation distance in [0, 1]

    # Inflation scale b so that E_inf -> E_max at d_c = d_t.
    b = (1.0 - inflation_beta) * d_t / jnp.sqrt(jnp.log(inflation_max))
    knee = inflation_beta * d_t

    # E_inf = 1 for d_c <= beta*d_t, else exp(((d_c - beta*d_t)/b)^2).
    excess = jnp.maximum(d_c - knee, 0.0)
    E_inf = jnp.where(d_c <= knee, 1.0, jnp.exp((excess / b) ** 2))

    weights = 1.0 / E_inf
    # Truncation: drop obs whose |rho| is below the threshold (d_c > d_t).
    retained = d_c <= d_t
    return jnp.where(retained, weights, 0.0)


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

    def __init__(  # type: ignore[no-untyped-def]
        self,
        grid_shape: Tuple[int, ...],
        ensemble_size: int,
        model_noise_std: float,
        obs_noise_std: float,
        observation_operator,
        real_space: bool = True,
    ):
        self.grid_shape = grid_shape
        self.ndim = len(grid_shape)
        self.ensemble_size = ensemble_size
        self.model_noise_std = model_noise_std
        self.obs_noise_std = obs_noise_std
        self.real_space = real_space
        self.observation_operator = observation_operator

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

    def analysis_step(
        self,
        forecast_ensemble_spectral: jnp.ndarray,
        observations: jnp.ndarray,
        key: jax.random.PRNGKey,
        inflation: float,
        *args: Any,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Standard non-localized analysis (original implementation)."""
        n_obs = len(observations)

        # Flatten spectral ensemble for easier manipulation
        # Shape: (ensemble_size, n_spectral_dofs)
        original_shape = forecast_ensemble_spectral.shape
        ensemble_flat = forecast_ensemble_spectral.reshape(self.ensemble_size, -1)

        ensemble_real = ensemble_flat

        # Compute ensemble mean and perturbations for real part
        x_mean_real = jnp.mean(ensemble_real, axis=0)
        X_pert_real = inflation * (ensemble_real - x_mean_real)

        # Map ensemble to observation space
        HX = jax.vmap(self.observation_operator)(forecast_ensemble_spectral)

        HX_mean = jnp.mean(HX, axis=0)
        HX_pert = HX - HX_mean

        # Compute covariances
        # P_xy = X_pert^T @ HX_pert / (N - 1)
        Pxy_real = (X_pert_real.T @ HX_pert) / (self.ensemble_size - 1)

        # P_yy = HX_pert^T @ HX_pert / (N - 1) + R
        R = (self.obs_noise_std**2) * jnp.eye(n_obs)
        Pyy = (HX_pert.T @ HX_pert) / (self.ensemble_size - 1) + R

        # Kalman gain: K = P_xy @ (P_yy)^{-1}
        Pyy_inv = jnp.linalg.inv(Pyy)
        K_real = Pxy_real @ Pyy_inv

        # Generate perturbed observations
        obs_noise = random.normal(key, (self.ensemble_size, n_obs)) * self.obs_noise_std
        perturbed_obs = observations + obs_noise

        # Update ensemble
        innovations = perturbed_obs - HX
        analysis_real = ensemble_real + innovations @ K_real.T

        analysis_flat = analysis_real

        # Reshape back to original spectral shape
        analysis_ensemble = analysis_flat.reshape(original_shape)

        return analysis_ensemble

    def assimilate(
        self,
        ensemble_spectral: jnp.ndarray,
        observations: jnp.ndarray,
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
            key2,
            inflation,
            localization_radius,
        )

        return analysis_ensemble, forecast_ensemble


class LocalizedSpectralEnKF(SpectralEnKF):
    """
    Localized Ensemble Kalman Filter for PDEs in Fourier space.

    Implements covariance localization using the Gaspari-Cohn (GC) function
    to reduce spurious correlations from limited ensemble sizes.

    The localization is applied in physical space, as correlations in spectral
    space are generally global. This means we:
    1. Transform ensemble to physical space for covariance computation
    2. Apply localization based on physical distance
    3. Update in physical space
    4. Transform back to spectral space

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
        Initial radius of influence for localization (in grid points)
        If adaptive_localization=True, this will be adjusted during assimilation
    real_space : bool
        If True, use real FFT (rfft), else complex FFT
    adaptive_localization : bool
        If True, automatically adjust localization radius based on ensemble statistics
    adaptation_method : str
        Method for adaptive localization ("correlation", "innovation", "hybrid")
    min_radius : float, optional
        Minimum allowed localization radius (default: localization_radius / 4)
    max_radius : float, optional
        Maximum allowed localization radius (default: min(domain_size/2, 4*localization_radius))
    localization_type : str
        Localization scheme. ``"distance"`` (default) uses the existing
        Gaspari-Cohn distance taper -- unchanged. ``"correlation"`` uses the
        OPT-IN ensemble-correlation selection + observation-error inflation of
        Vossepoel, Evensen & van Leeuwen (2025) (see
        :func:`correlation_localization_weights`); it ignores
        ``localization_radius`` and instead uses ``corr_threshold`` /
        ``corr_inflation_max`` / ``corr_inflation_beta``.
    corr_threshold : float, optional
        Truncation threshold ``r_t`` on ``|rho|`` for correlation localization.
        ``None`` (default) -> first-guess ``3 / sqrt(ensemble_size)``.
    corr_inflation_max : float
        Maximum obs-error inflation ``E_max`` for correlation localization
        (default 4.0; paper 4--8).
    corr_inflation_beta : float
        Fraction ``beta`` of ``d_t`` below which no inflation is applied
        (default 0.5).
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
        adaptive_localization: bool = False,
        min_radius: Optional[float] = None,
        max_radius: Optional[float] = None,
        adaptation_method: str = "hybrid",
        localization_type: str = "distance",
        corr_threshold: Optional[float] = None,
        corr_inflation_max: float = 4.0,
        corr_inflation_beta: float = 0.5,
    ):
        super().__init__(
            grid_shape=grid_shape,
            ensemble_size=ensemble_size,
            model_noise_std=model_noise_std,
            obs_noise_std=obs_noise_std,
            real_space=real_space,
            observation_operator=observation_operator,
        )
        self.localization_radius = localization_radius
        self.adaptive_localization = adaptive_localization
        self.adaptation_method = adaptation_method

        # Correlation-based localization (opt-in; default is distance/GC).
        if localization_type not in ("distance", "correlation"):
            raise ValueError(
                f"localization_type must be 'distance' or 'correlation', "
                f"got {localization_type!r}"
            )
        self.localization_type = localization_type
        # First-guess r_t = 3 / sqrt(N) removes ~99.7% of spurious sampling noise.
        self.corr_threshold = (
            float(corr_threshold)
            if corr_threshold is not None
            else 3.0 / (ensemble_size**0.5)
        )
        self.corr_inflation_max = corr_inflation_max
        self.corr_inflation_beta = corr_inflation_beta
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

    def estimate_correlation_length(
        self,
        ensemble_physical: jnp.ndarray,
        sample_points: int = 100,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> float:
        """
        Estimate the spatial correlation length scale from the ensemble.

        This provides a data-driven estimate of the localization radius based on
        the actual ensemble correlations.

        Parameters
        ----------
        ensemble_physical : jnp.ndarray
            Ensemble in physical space (N, *grid_shape)
        sample_points : int
            Number of random points to sample for correlation estimation
        key : jax.random.PRNGKey, optional
            Random key for sampling

        Returns
        -------
        float
            Estimated correlation length scale
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        N = ensemble_physical.shape[0]
        flat_ensemble = ensemble_physical.reshape(N, -1)
        n_state = flat_ensemble.shape[1]

        # Compute ensemble perturbations
        ens_mean = jnp.mean(flat_ensemble, axis=0)
        ens_pert = flat_ensemble - ens_mean

        # Sample random reference points
        key, subkey = random.split(key)
        ref_indices = random.choice(
            subkey, n_state, shape=(min(sample_points, n_state),), replace=False
        )

        # Compute correlations as a function of distance
        distances_list = []
        correlations_list = []

        for ref_idx in ref_indices[:sample_points]:
            # Reference point perturbations
            ref_pert = ens_pert[:, ref_idx]

            # Compute correlation with all other points
            correlations = jnp.mean(ens_pert * ref_pert[:, None], axis=0)
            ref_var = jnp.var(ref_pert)

            # Normalize
            correlations = jnp.where(ref_var > 1e-10, correlations / ref_var, 0.0)

            ref_coord = self.grid_coords[ref_idx]
            diff = self.grid_coords - ref_coord[None, :]
            diff_x = jnp.abs(diff[:, 0])
            diff_y = jnp.abs(diff[:, 1])
            Lx, Ly = self.grid_shape
            diff_x = jnp.minimum(diff_x, Lx - diff_x)
            diff_y = jnp.minimum(diff_y, Ly - diff_y)
            distances = jnp.sqrt(diff_x**2 + diff_y**2)

            distances_list.append(distances)
            correlations_list.append(correlations)

        # Concatenate all samples
        all_distances = jnp.concatenate(distances_list)
        all_correlations = jnp.concatenate(correlations_list)

        # Estimate length scale: find distance where correlation drops to e^{-1/2}
        # (half-width in Gaussian sense)
        target_corr = jnp.exp(-0.5)

        # Bin distances and compute average correlation
        max_dist = jnp.max(all_distances)
        n_bins = 50
        bin_edges = jnp.linspace(0, max_dist, n_bins + 1)

        bin_corrs = []
        bin_centers = []

        for i in range(n_bins):
            mask = (all_distances >= bin_edges[i]) & (all_distances < bin_edges[i + 1])
            if jnp.sum(mask) > 0:
                avg_corr = jnp.mean(jnp.abs(all_correlations[mask]))
                bin_corrs.append(avg_corr)
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2.0)

        if len(bin_corrs) == 0:
            return self.localization_radius

        bin_corrs = jnp.array(bin_corrs)
        bin_centers = jnp.array(bin_centers)

        # Find where correlation crosses the target
        below_target = bin_corrs < target_corr
        if jnp.any(below_target):
            first_below = jnp.argmax(below_target)
            length_scale = bin_centers[first_below]
        else:
            # If never drops below target, use maximum
            length_scale = bin_centers[-1]

        return float(length_scale)

    def adapt_localization_radius(
        self,
        ensemble_physical: jnp.ndarray,
        innovations: jnp.ndarray,
        method: str = "innovation",
        alpha: float = 0.1,
    ) -> float:
        """
        Adaptively adjust the localization radius based on filter performance.

        Parameters
        ----------
        ensemble_physical : jnp.ndarray
            Ensemble in physical space
        innovations : jnp.ndarray
            Innovation statistics (observations - H(ensemble_mean))
        method : str
            Method for adaptation:
            - "innovation": Based on innovation statistics
            - "correlation": Based on ensemble correlation length
            - "hybrid": Combination of both
        alpha : float
            Learning rate for adaptation (0 < alpha < 1)

        Returns
        -------
        float
            Updated localization radius
        """
        current_radius = self.localization_radius

        if method == "correlation":
            # Estimate from ensemble correlations
            estimated_length = self.estimate_correlation_length(ensemble_physical)
            # Use 2-3 times the correlation length for localization
            target_radius = 2.5 * estimated_length

        elif method == "innovation":
            # Based on innovation statistics
            # If innovations are large, ensemble spread might be too small
            # -> increase localization to allow more information from observations

            innovation_norm = jnp.linalg.norm(innovations)
            expected_norm = jnp.sqrt(len(innovations)) * self.obs_noise_std

            # Normalized innovation
            normalized_innov = innovation_norm / (expected_norm + 1e-10)

            # If innovations are larger than expected, increase radius
            # If innovations are smaller, decrease radius
            if normalized_innov > 1.5:
                # Filter might be overconfident, increase localization
                target_radius = current_radius * 1.1
            elif normalized_innov < 0.7:
                # Filter might be too uncertain, decrease localization
                target_radius = current_radius * 0.95
            else:
                target_radius = current_radius

        elif method == "hybrid":
            # Combine both approaches
            length_scale = self.estimate_correlation_length(ensemble_physical)
            corr_based = 2.5 * length_scale

            innovation_norm = jnp.linalg.norm(innovations)
            expected_norm = jnp.sqrt(len(innovations)) * self.obs_noise_std
            normalized_innov = innovation_norm / (expected_norm + 1e-10)

            # Adjust correlation-based estimate with innovation info
            if normalized_innov > 1.5:
                target_radius = corr_based * 1.2
            elif normalized_innov < 0.7:
                target_radius = corr_based * 0.9
            else:
                target_radius = corr_based

        else:
            raise ValueError(f"Unknown adaptation method: {method}")

        # Smooth update with learning rate
        new_radius = (1 - alpha) * current_radius + alpha * target_radius

        # Clip to bounds
        new_radius = jnp.clip(new_radius, self.min_radius, self.max_radius)

        return float(new_radius)

    def get_adaptation_diagnostics(self) -> dict:
        """
        Get diagnostic information about adaptive localization.

        Returns
        -------
        dict
            Dictionary containing:
            - 'radius_history': List of localization radii over time
            - 'innovation_stats': List of innovation norms over time
            - 'current_radius': Current localization radius
            - 'min_radius': Minimum allowed radius
            - 'max_radius': Maximum allowed radius
        """
        return {
            "radius_history": self.radius_history,
            "innovation_stats": self.innovation_stats,
            "current_radius": self.localization_radius,
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
        }

    def reset_adaptation_history(self) -> jnp.ndarray:
        """Reset the adaptation history."""
        self.radius_history = []
        self.innovation_stats = []

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

    def compute_distance_matrix(
        self, periodic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute distance matrices for localization.

        Parameters
        ----------
        periodic : bool
            Whether to use periodic boundary conditions for distance

        Returns
        -------
        tuple
            (distances_state_to_obs, distances_obs_to_obs)
            Both with localization applied via Gaspari-Cohn function
        """
        n_obs = len(self.observation_operator.obs_coords)

        # 2D case
        obs_coords = self.observation_operator.obs_coords  # shape (n_obs, 2)
        state_coords = self.grid_coords  # shape (n_state, 2)

        if periodic:
            # Periodic distance in 2D
            Lx, Ly = self.grid_shape

            diff_state_obs = state_coords[:, None, :] - obs_coords[None, :, :]
            diff_state_obs_x = jnp.abs(diff_state_obs[:, :, 0])
            diff_state_obs_y = jnp.abs(diff_state_obs[:, :, 1])
            diff_state_obs_x = jnp.minimum(diff_state_obs_x, Lx - diff_state_obs_x)
            diff_state_obs_y = jnp.minimum(diff_state_obs_y, Ly - diff_state_obs_y)
            dist_state_obs = jnp.sqrt(diff_state_obs_x**2 + diff_state_obs_y**2)

            diff_obs_obs = obs_coords[:, None, :] - obs_coords[None, :, :]
            diff_obs_obs_x = jnp.abs(diff_obs_obs[:, :, 0])
            diff_obs_obs_y = jnp.abs(diff_obs_obs[:, :, 1])
            diff_obs_obs_x = jnp.minimum(diff_obs_obs_x, Lx - diff_obs_obs_x)
            diff_obs_obs_y = jnp.minimum(diff_obs_obs_y, Ly - diff_obs_obs_y)
            dist_obs_obs = jnp.sqrt(diff_obs_obs_x**2 + diff_obs_obs_y**2)
        else:
            # Euclidean distance
            diff_state_obs = state_coords[:, None, :] - obs_coords[None, :, :]
            dist_state_obs = jnp.linalg.norm(diff_state_obs, axis=2)

            diff_obs_obs = obs_coords[:, None, :] - obs_coords[None, :, :]
            dist_obs_obs = jnp.linalg.norm(diff_obs_obs, axis=2)

        # Apply Gaspari-Cohn localization
        rho_state_obs = self.gaspari_cohn(dist_state_obs, self.localization_radius)
        rho_obs_obs = self.gaspari_cohn(dist_obs_obs, self.localization_radius)

        return rho_state_obs, rho_obs_obs

    def analysis_step(
        self,
        forecast_ensemble_spectral: jnp.ndarray,
        observations: jnp.ndarray,
        key: jax.random.PRNGKey,
        inflation: float = 1.0,
        localization_radius: Optional[float] = None,
        periodic: bool = True,
    ) -> jnp.ndarray:
        """
        Localized analysis step with covariance localization.

        The localization is applied in physical space using the Schur product
        (element-wise multiplication) of the sample covariance with a
        correlation matrix based on physical distance.

        Parameters
        ----------
        forecast_ensemble_spectral : jnp.ndarray
            Forecasted ensemble in Fourier space
        observations : jnp.ndarray
            Observations in physical space
        key : jax.random.PRNGKey
            Random key for perturbed observations
        inflation : float
            Covariance inflation factor
        localization_radius : float, optional
            Override the instance localization radius
        periodic : bool
            Use periodic boundary conditions for distance calculations
        adaptation_method : str, optional
            Method for adaptive localization ("correlation", "innovation", "hybrid")
            Only used if adaptive_localization is True

        Returns
        -------
        jnp.ndarray
            Analysis ensemble in Fourier space
        """
        n_obs = len(observations)

        # Transform ensemble to physical space for localized update
        forecast_ensemble_physical = jax.vmap(self.spectral_to_physical)(
            forecast_ensemble_spectral
        )

        # Adaptive localization if enabled
        if (
            self.adaptive_localization
            and self.localization_radius < self.max_radius
            and self.localization_radius > self.min_radius
        ):
            # Compute innovations for adaptation
            HX = jax.vmap(self.observation_operator.physical_observation_operator)(
                forecast_ensemble_physical
            )
            HX_mean = jnp.mean(HX, axis=0)
            innovations = observations - HX_mean

            # Adapt the radius
            new_radius = self.adapt_localization_radius(
                forecast_ensemble_physical,
                innovations,
                method=self.adaptation_method,
                alpha=0.1,
            )

            print(f"Adapted localization radius: {new_radius}")

            # Store history
            self.radius_history.append(new_radius)
            self.innovation_stats.append(float(jnp.linalg.norm(innovations)))

            # Update the radius
            self.localization_radius = new_radius
            loc_radius = new_radius
        else:
            loc_radius = (
                localization_radius
                if localization_radius is not None
                else self.localization_radius
            )

        # Flatten physical ensemble
        original_shape = forecast_ensemble_physical.shape
        ensemble_flat = forecast_ensemble_physical.reshape(self.ensemble_size, -1)
        n_state = ensemble_flat.shape[1]

        # Compute ensemble mean and perturbations
        x_mean = jnp.mean(ensemble_flat, axis=0)
        X_pert = inflation * (ensemble_flat - x_mean)

        # Map ensemble to observation space (use physical space ensemble)
        HX = jax.vmap(self.observation_operator.physical_observation_operator)(
            forecast_ensemble_physical
        )

        HX_mean = jnp.mean(HX, axis=0)
        HX_pert = HX - HX_mean

        # Compute localization matrices.
        if self.localization_type == "correlation":
            # OPT-IN correlation-based localization (default path is unchanged
            # distance/GC below). Taper P_xy by the ensemble-correlation weight
            # w(l, j) = 1 / E_inf (0 = truncated) in place of the GC distance
            # taper. The obs-obs covariance keeps a per-OBS error inflation
            # E_inf_obs that is the minimum inflation any retained state variable
            # applies to obs j (so a measurement that is far -- in correlation --
            # from the WHOLE state has its global precision down-weighted).
            #
            # TODO(author): in the paper the obs-error inflation is PER STATE
            # VARIABLE (local analysis), which the global-Schur EnKF cannot
            # express exactly; here it is approximated by (a) the per-(l, j)
            # Schur taper on P_xy and (b) a per-obs E_inf in P_yy reduced over l.
            # The LETKF path (LocalizedSpectralETKF, localization_type=
            # "correlation") implements the paper's per-variable local analysis
            # faithfully and should be preferred for the headline result.
            rho = correlation_state_obs(X_pert, HX_pert)  # (n_state, n_obs)
            w_state_obs = correlation_localization_weights(
                rho,
                self.corr_threshold,
                self.corr_inflation_max,
                self.corr_inflation_beta,
            )  # (n_state, n_obs) in [0, 1]
            rho_state_obs = w_state_obs
            # Per-obs inflation: smallest E_inf = 1 / max_l w(l, j) over retained
            # state vars (1.0 if some var sees obs j with no inflation).
            max_w_per_obs = jnp.max(w_state_obs, axis=0)  # (n_obs,)
            E_inf_obs = jnp.where(max_w_per_obs > 0.0, 1.0 / max_w_per_obs, 1.0)
            rho_obs_obs = jnp.eye(n_obs)
            R_diag = (self.obs_noise_std**2) * E_inf_obs  # (n_obs,)
            R = jnp.diag(R_diag)
        else:
            # DEFAULT: existing Gaspari-Cohn distance localization (unchanged).
            rho_state_obs, rho_obs_obs = self.compute_distance_matrix(
                periodic=periodic
            )
            R = (self.obs_noise_std**2) * jnp.eye(n_obs)

        # Compute localized covariances
        # P_xy = (X_pert^T @ HX_pert / (N-1)) ⊙ ρ_state_obs
        Pxy = (X_pert.T @ HX_pert) / (self.ensemble_size - 1)
        Pxy_localized = Pxy * rho_state_obs  # Schur product

        # P_yy = (HX_pert^T @ HX_pert / (N-1)) ⊙ ρ_obs_obs + R
        Pyy = (HX_pert.T @ HX_pert) / (self.ensemble_size - 1)
        Pyy_localized = Pyy * rho_obs_obs + R

        # Kalman gain with localization
        Pyy_inv = jnp.linalg.inv(Pyy_localized)
        K = Pxy_localized @ Pyy_inv

        # Generate perturbed observations (stochastic EnKF)
        obs_noise = random.normal(key, (self.ensemble_size, n_obs)) * self.obs_noise_std
        perturbed_obs = observations + obs_noise

        # Update ensemble in physical space
        innovations = perturbed_obs - HX
        analysis_flat = ensemble_flat + innovations @ K.T

        # Reshape back to physical grid
        analysis_physical = analysis_flat.reshape(original_shape)

        # Transform back to spectral space
        analysis_spectral = jax.vmap(self.physical_to_spectral)(analysis_physical)

        return analysis_spectral
