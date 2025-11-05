"""
Ensemble Kalman Filter for PDEs in Physical Space

Handles systems where:
- State x is represented in physical space (no Fourier transforms)
- Dynamics F operates in physical space
- Observations y are in physical space

State equation: x_{t+1} = F(x_t) + model_noise  (Physical space)
Observation equation: y_{t+1} = H(x_{t+1}) + obs_noise  (Physical space)

where H is the observation operator mapping physical space to observation points.
"""

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_cfd import grids
from torch_cfd.grids import GridArray, GridVariable


class ObservationOperator:
    """
    Observation operator for physical space velocity fields.

    Maps velocity fields (tuple of GridVariables) to observations at specific grid points.
    """

    def __init__(
        self, obs_indices: torch.Tensor, obs_components: Optional[List[str]] = None
    ):
        """
        Initialize observation operator.

        Parameters
        ----------
        obs_indices : torch.Tensor
            Observation indices in (x, y) format, shape (n_obs, 2)
        obs_components : List[str], optional
            Which velocity components to observe. Options: ['u', 'v'] or ['u', 'v'].
            If None, observes both components.
        """
        self.obs_indices = obs_indices  # shape (n_obs, 2)
        self.obs_coords = obs_indices  # For compatibility with localization
        self.obs_components = (
            obs_components if obs_components is not None else ["u", "v"]
        )

    def __call__(self, velocity: Tuple[Any, Any]) -> torch.Tensor:
        """
        Observation operator H: maps velocity fields to physical space observations.

        Parameters
        ----------
        velocity : Tuple[GridVariable, GridVariable]
            Velocity field as tuple of (u, v) GridVariables

        Returns
        -------
        torch.Tensor
            Observations at observation points, shape (n_obs,) or (n_obs, 2) if both components
        """
        u, v = velocity
        u_data = u.data.squeeze()  # Remove batch dimension if present
        v_data = v.data.squeeze()

        # Extract observations at specified indices
        obs_list = []
        if "u" in self.obs_components:
            obs_u = u_data[self.obs_indices[:, 0], self.obs_indices[:, 1]]
            obs_list.append(obs_u)
        if "v" in self.obs_components:
            obs_v = v_data[self.obs_indices[:, 0], self.obs_indices[:, 1]]
            obs_list.append(obs_v)

        if len(obs_list) == 1:
            return obs_list[0]
        else:
            return torch.stack(
                obs_list, dim=1
            ).flatten()  # Flatten to (n_obs * n_components,)


class PhysicalEnKF:
    """
    Ensemble Kalman Filter for PDEs in physical space.

    Parameters
    ----------
    grid_shape : tuple
        Shape of the physical space grid (nx, ny) for 2D
    ensemble_size : int
        Number of ensemble members
    model_noise_std : float
        Standard deviation of model noise in physical space
    obs_noise_std : float
        Standard deviation of observation noise in physical space
    observation_operator : ObservationOperator
        Operator that maps velocity fields to observations
    device : torch.device
        Device to run computations on
    dtype : torch.dtype
        Data type for computations
    """

    def __init__(
        self,
        grid_shape: Tuple[int, ...],
        ensemble_size: int,
        model_noise_std: float,
        obs_noise_std: float,
        observation_operator: ObservationOperator,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.grid_shape = grid_shape
        self.ndim = len(grid_shape)
        self.ensemble_size = ensemble_size
        self.model_noise_std = model_noise_std
        self.obs_noise_std = obs_noise_std
        self.observation_operator = observation_operator
        self.device = device
        self.dtype = dtype

        if self.ndim != 2:
            raise ValueError("Currently only supports 2D grids")

    def forecast_step(
        self,
        ensemble_velocity: List[Tuple[Any, Any]],
        dynamics: Callable,
        dt: float,
    ) -> List[Tuple[Any, Any]]:
        """
        Forecast step in physical space.

        Parameters
        ----------
        ensemble_velocity : List[Tuple[GridVariable, GridVariable]]
            Ensemble of velocity fields, each as tuple (u, v)
        dynamics : Callable
            Dynamics function operating on velocity fields: (v, dt) -> (v_new, p)
        dt : float
            Time step

        Returns
        -------
        List[Tuple[GridVariable, GridVariable]]
            Forecasted ensemble of velocity fields
        """
        forecast_ensemble = []
        for velocity in ensemble_velocity:
            # Dynamics function takes velocity and dt, returns (v_new, p)
            v_forecast, _ = dynamics(velocity, dt)
            forecast_ensemble.append(v_forecast)

        return forecast_ensemble

    def analysis_step(
        self,
        forecast_ensemble_velocity: List[Tuple[Any, Any]],
        observations: torch.Tensor,
        inflation: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ) -> List[Tuple[Any, Any]]:
        """
        Standard non-localized analysis step.

        Parameters
        ----------
        forecast_ensemble_velocity : List[Tuple[GridVariable, GridVariable]]
            Forecasted ensemble of velocity fields
        observations : torch.Tensor
            Observations in physical space, shape (n_obs,)
        inflation : float
            Covariance inflation factor

        Returns
        -------
        List[Tuple[GridVariable, GridVariable]]
            Analysis ensemble of velocity fields
        """
        n_obs = len(observations)

        # Extract physical space data from velocity fields
        # Shape: (ensemble_size, 2, nx, ny) - 2 for u and v components
        ensemble_physical = []
        for velocity in forecast_ensemble_velocity:
            u, v = velocity
            u_data = u.data.squeeze()  # Remove batch dimension if present
            v_data = v.data.squeeze()
            ensemble_physical.append(torch.stack([u_data, v_data], dim=0))

        ensemble_physical = torch.stack(
            ensemble_physical, dim=0
        )  # (ensemble_size, 2, nx, ny)

        # Flatten spatial dimensions: (ensemble_size, 2 * nx * ny)
        original_shape = ensemble_physical.shape  # type: ignore[attr-defined]
        ensemble_flat = ensemble_physical.reshape(self.ensemble_size, -1)  # type: ignore[attr-defined]

        # Compute ensemble mean and perturbations
        x_mean = torch.mean(ensemble_flat, dim=0)
        X_pert = inflation * (ensemble_flat - x_mean)

        # Map ensemble to observation space
        HX = []
        for velocity in forecast_ensemble_velocity:
            obs = self.observation_operator(velocity)
            HX.append(obs)
        HX = torch.stack(HX, dim=0)  # (ensemble_size, n_obs)

        HX_mean = torch.mean(HX, dim=0)
        HX_pert = HX - HX_mean

        # Compute covariances
        # P_xy = X_pert^T @ HX_pert / (N - 1)
        Pxy = (X_pert.T @ HX_pert) / (self.ensemble_size - 1)

        # P_yy = HX_pert^T @ HX_pert / (N - 1) + R
        R = (self.obs_noise_std**2) * torch.eye(
            n_obs, device=self.device, dtype=self.dtype
        )
        Pyy = (HX_pert.T @ HX_pert) / (self.ensemble_size - 1) + R

        # Kalman gain: K = P_xy @ (P_yy)^{-1}
        Pyy_inv = torch.linalg.inv(Pyy)
        K = Pxy @ Pyy_inv

        # Generate perturbed observations (stochastic EnKF)
        obs_noise = (
            torch.randn(self.ensemble_size, n_obs, device=self.device, dtype=self.dtype)
            * self.obs_noise_std
        )
        perturbed_obs = observations + obs_noise

        # Update ensemble
        innovations = perturbed_obs - HX
        analysis_flat = ensemble_flat + innovations @ K.T

        # Reshape back to original shape
        analysis_physical = analysis_flat.reshape(
            original_shape
        )  # (ensemble_size, 2, nx, ny)

        # Convert back to GridVariable format
        # Update the data attribute of existing GridVariables
        analysis_ensemble = []
        for i, forecast_velocity in enumerate(forecast_ensemble_velocity):
            u_forecast, v_forecast = forecast_velocity
            # Update the data attribute directly
            # Ensure the shape matches (add batch dimension if needed)
            u_data_new = analysis_physical[i, 0]
            v_data_new = analysis_physical[i, 1]

            # Match the original shape (might have batch dimension)
            if u_forecast.data.ndim == 3:  # Has batch dimension
                u_data_new = u_data_new.unsqueeze(0)
                v_data_new = v_data_new.unsqueeze(0)

            # Create new GridArray with updated data
            u_array_new = GridArray(
                u_data_new, u_forecast.array.offset, u_forecast.array.grid
            )
            v_array_new = GridArray(
                v_data_new, v_forecast.array.offset, v_forecast.array.grid
            )

            # Create new GridVariable
            u_new = GridVariable(u_array_new, u_forecast.bc)
            v_new = GridVariable(v_array_new, v_forecast.bc)

            analysis_ensemble.append((u_new, v_new))

        return analysis_ensemble

    def assimilate(
        self,
        ensemble_velocity: List[Tuple[Any, Any]],
        observations: torch.Tensor,
        dynamics: Callable,
        dt: float,
        inflation: float = 1.0,
        localization_radius: Optional[float] = None,
    ) -> Tuple[List[Tuple[Any, Any]], List[Tuple[Any, Any]]]:
        """
        Complete assimilation cycle.

        Parameters
        ----------
        ensemble_velocity : List[Tuple[GridVariable, GridVariable]]
            Current ensemble of velocity fields
        observations : torch.Tensor
            Observations in physical space
        dynamics : Callable
            Dynamics function: (v, dt) -> (v_new, p)
        dt : float
            Time step
        inflation : float
            Covariance inflation factor
        localization_radius : float, optional
            Localization radius (for compatibility, not used in base class)

        Returns
        -------
        Tuple[List[Tuple[GridVariable, GridVariable]], List[Tuple[GridVariable, GridVariable]]]
            (analysis_ensemble, forecast_ensemble)
        """
        forecast_ensemble = self.forecast_step(ensemble_velocity, dynamics, dt)
        analysis_ensemble = self.analysis_step(
            forecast_ensemble,
            observations,
            inflation,
            localization_radius,
        )

        return analysis_ensemble, forecast_ensemble


class LocalizedPhysicalEnKF(PhysicalEnKF):
    """
    Localized Ensemble Kalman Filter for PDEs in physical space.

    Implements covariance localization using the Gaspari-Cohn (GC) function
    to reduce spurious correlations from limited ensemble sizes.

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
    observation_operator : ObservationOperator
        Operator that maps velocity fields to observations
    device : torch.device
        Device to run computations on
    dtype : torch.dtype
        Data type for computations
    adaptive_localization : bool
        If True, automatically adjust localization radius based on ensemble statistics
    adaptation_method : str
        Method for adaptive localization ("correlation", "innovation", "hybrid")
    min_radius : float, optional
        Minimum allowed localization radius (default: localization_radius / 4)
    max_radius : float, optional
        Maximum allowed localization radius (default: min(domain_size/2, 4*localization_radius))
    periodic : bool
        Whether to use periodic boundary conditions for distance calculations
    """

    def __init__(
        self,
        grid_shape: Tuple[int, ...],
        ensemble_size: int,
        model_noise_std: float,
        obs_noise_std: float,
        localization_radius: float,
        observation_operator: ObservationOperator,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        adaptive_localization: bool = False,
        min_radius: Optional[float] = None,
        max_radius: Optional[float] = None,
        adaptation_method: str = "hybrid",
        periodic: bool = False,
    ):
        super().__init__(
            grid_shape=grid_shape,
            ensemble_size=ensemble_size,
            model_noise_std=model_noise_std,
            obs_noise_std=obs_noise_std,
            observation_operator=observation_operator,
            device=device,
            dtype=dtype,
        )
        self.localization_radius = localization_radius
        self.adaptive_localization = adaptive_localization
        self.adaptation_method = adaptation_method
        self.periodic = periodic

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
        x = torch.arange(grid_shape[0], device=device, dtype=dtype)
        y = torch.arange(grid_shape[1], device=device, dtype=dtype)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        self.grid_coords = torch.stack(
            [xx.flatten(), yy.flatten()], dim=1
        )  # (n_state, 2)

    def gaspari_cohn(self, distance: torch.Tensor, radius: float) -> torch.Tensor:
        """
        Gaspari-Cohn correlation function for localization.

        This is a fifth-order piecewise rational function with compact support.
        It smoothly tapers correlations to zero beyond 2*radius.

        Parameters
        ----------
        distance : torch.Tensor
            Distances between points
        radius : float
            Localization radius (half-width of compact support)

        Returns
        -------
        torch.Tensor
            Localization weights in [0, 1]
        """
        # Normalize by radius
        r = torch.abs(distance) / radius

        # GC function has support on [0, 2]
        # For r in [0, 1]
        term1 = torch.where(
            r <= 1,
            1 - 5 / 3 * r**2 + 5 / 8 * r**3 + 1 / 2 * r**4 - 1 / 4 * r**5,
            torch.tensor(0.0, device=distance.device, dtype=distance.dtype),
        )

        # For r in [1, 2]
        term2 = torch.where(
            (r > 1) & (r <= 2),
            4
            - 5 * r
            + 5 / 3 * r**2
            + 5 / 8 * r**3
            - 1 / 2 * r**4
            + 1 / 12 * r**5
            - 2 / (3 * r),
            torch.tensor(0.0, device=distance.device, dtype=distance.dtype),
        )

        return term1 + term2

    def compute_distance_matrix(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute distance matrices for localization.

        Returns
        -------
        tuple
            (distances_state_to_obs, distances_obs_to_obs)
            Both with localization applied via Gaspari-Cohn function
        """
        n_obs = len(self.observation_operator.obs_coords)

        # 2D case
        obs_coords = self.observation_operator.obs_coords.to(
            self.device
        )  # shape (n_obs, 2)
        state_coords = self.grid_coords  # shape (n_state, 2)

        if self.periodic:
            # Periodic distance in 2D
            Lx, Ly = self.grid_shape[0], self.grid_shape[1]

            diff_state_obs = state_coords[:, None, :] - obs_coords[None, :, :]
            diff_state_obs_x = torch.abs(diff_state_obs[:, :, 0])
            diff_state_obs_y = torch.abs(diff_state_obs[:, :, 1])
            diff_state_obs_x = torch.minimum(diff_state_obs_x, Lx - diff_state_obs_x)
            diff_state_obs_y = torch.minimum(diff_state_obs_y, Ly - diff_state_obs_y)
            dist_state_obs = torch.sqrt(diff_state_obs_x**2 + diff_state_obs_y**2)

            diff_obs_obs = obs_coords[:, None, :] - obs_coords[None, :, :]
            diff_obs_obs_x = torch.abs(diff_obs_obs[:, :, 0])
            diff_obs_obs_y = torch.abs(diff_obs_obs[:, :, 1])
            diff_obs_obs_x = torch.minimum(diff_obs_obs_x, Lx - diff_obs_obs_x)
            diff_obs_obs_y = torch.minimum(diff_obs_obs_y, Ly - diff_obs_obs_y)
            dist_obs_obs = torch.sqrt(diff_obs_obs_x**2 + diff_obs_obs_y**2)
        else:
            # Euclidean distance
            diff_state_obs = state_coords[:, None, :] - obs_coords[None, :, :]
            dist_state_obs = torch.norm(diff_state_obs, dim=2)

            diff_obs_obs = obs_coords[:, None, :] - obs_coords[None, :, :]
            dist_obs_obs = torch.norm(diff_obs_obs, dim=2)

        # Apply Gaspari-Cohn localization
        rho_state_obs = self.gaspari_cohn(dist_state_obs, self.localization_radius)
        rho_obs_obs = self.gaspari_cohn(dist_obs_obs, self.localization_radius)

        return rho_state_obs, rho_obs_obs

    def analysis_step(
        self,
        forecast_ensemble_velocity: List[Tuple[Any, Any]],
        observations: torch.Tensor,
        inflation: float = 1.0,
        localization_radius: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ) -> List[Tuple[Any, Any]]:
        """
        Localized analysis step with covariance localization.

        The localization is applied in physical space using the Schur product
        (element-wise multiplication) of the sample covariance with a
        correlation matrix based on physical distance.

        Parameters
        ----------
        forecast_ensemble_velocity : List[Tuple[GridVariable, GridVariable]]
            Forecasted ensemble of velocity fields
        observations : torch.Tensor
            Observations in physical space
        inflation : float
            Covariance inflation factor
        localization_radius : float, optional
            Override the instance localization radius

        Returns
        -------
        List[Tuple[GridVariable, GridVariable]]
            Analysis ensemble of velocity fields
        """
        n_obs = len(observations)

        # Extract physical space data from velocity fields
        ensemble_physical = []
        for velocity in forecast_ensemble_velocity:
            u, v = velocity
            u_data = u.data.squeeze()
            v_data = v.data.squeeze()
            ensemble_physical.append(torch.stack([u_data, v_data], dim=0))

        ensemble_physical = torch.stack(
            ensemble_physical, dim=0
        )  # (ensemble_size, 2, nx, ny)

        # Adaptive localization if enabled
        if (
            self.adaptive_localization
            and self.localization_radius < self.max_radius
            and self.localization_radius > self.min_radius
        ):
            # Compute innovations for adaptation
            HX = []
            for velocity in forecast_ensemble_velocity:
                obs = self.observation_operator(velocity)
                HX.append(obs)
            HX = torch.stack(HX, dim=0)
            HX_mean = torch.mean(HX, dim=0)
            innovations = observations - HX_mean

            # Adapt the radius (simplified version - can be enhanced)
            # For now, use a simple heuristic
            innovation_norm = torch.norm(innovations).item()
            expected_norm = np.sqrt(len(innovations)) * self.obs_noise_std
            normalized_innov = innovation_norm / (expected_norm + 1e-10)

            if normalized_innov > 1.5:
                new_radius = self.localization_radius * 1.1
            elif normalized_innov < 0.7:
                new_radius = self.localization_radius * 0.95
            else:
                new_radius = self.localization_radius

            new_radius = max(self.min_radius, min(self.max_radius, new_radius))

            print(f"Adapted localization radius: {new_radius:.2f}")

            # Store history
            self.radius_history.append(new_radius)
            self.innovation_stats.append(innovation_norm)

            # Update the radius
            self.localization_radius = new_radius
            loc_radius = new_radius
        else:
            loc_radius = (
                localization_radius
                if localization_radius is not None
                else self.localization_radius
            )

        # Flatten spatial dimensions
        original_shape = ensemble_physical.shape  # type: ignore[attr-defined]
        ensemble_flat = ensemble_physical.reshape(self.ensemble_size, -1)  # type: ignore[attr-defined]
        n_state = ensemble_flat.shape[1]

        # Compute ensemble mean and perturbations
        x_mean = torch.mean(ensemble_flat, dim=0)
        X_pert = inflation * (ensemble_flat - x_mean)

        # Map ensemble to observation space
        HX = []
        for velocity in forecast_ensemble_velocity:
            obs = self.observation_operator(velocity)
            HX.append(obs)
        HX = torch.stack(HX, dim=0)  # (ensemble_size, n_obs)

        HX_mean = torch.mean(HX, dim=0)
        HX_pert = HX - HX_mean

        # Compute localization matrices
        rho_state_obs, rho_obs_obs = self.compute_distance_matrix()

        # Compute localized covariances
        # P_xy = (X_pert^T @ HX_pert / (N-1)) ⊙ ρ_state_obs
        Pxy = (X_pert.T @ HX_pert) / (self.ensemble_size - 1)
        Pxy_localized = Pxy * rho_state_obs  # Schur product

        # P_yy = (HX_pert^T @ HX_pert / (N-1)) ⊙ ρ_obs_obs + R
        Pyy = (HX_pert.T @ HX_pert) / (self.ensemble_size - 1)
        Pyy_localized = Pyy * rho_obs_obs + (self.obs_noise_std**2) * torch.eye(
            n_obs, device=self.device, dtype=self.dtype
        )

        # Kalman gain with localization
        Pyy_inv = torch.linalg.inv(Pyy_localized)
        K = Pxy_localized @ Pyy_inv

        # Generate perturbed observations (stochastic EnKF)
        obs_noise = (
            torch.randn(self.ensemble_size, n_obs, device=self.device, dtype=self.dtype)
            * self.obs_noise_std
        )
        perturbed_obs = observations + obs_noise

        # Update ensemble in physical space
        innovations = perturbed_obs - HX
        analysis_flat = ensemble_flat + innovations @ K.T

        # Reshape back to original shape
        analysis_physical = analysis_flat.reshape(
            original_shape
        )  # (ensemble_size, 2, nx, ny)

        # Convert back to GridVariable format
        # Update the data attribute of existing GridVariables
        analysis_ensemble = []
        for i, forecast_velocity in enumerate(forecast_ensemble_velocity):
            u_forecast, v_forecast = forecast_velocity
            # Update the data attribute directly
            # Ensure the shape matches (add batch dimension if needed)
            u_data_new = analysis_physical[i, 0]
            v_data_new = analysis_physical[i, 1]

            # Match the original shape (might have batch dimension)
            if u_forecast.data.ndim == 3:  # Has batch dimension
                u_data_new = u_data_new.unsqueeze(0)
                v_data_new = v_data_new.unsqueeze(0)

            # Create new GridArray with updated data
            u_array_new = GridArray(
                u_data_new, u_forecast.array.offset, u_forecast.array.grid
            )
            v_array_new = GridArray(
                v_data_new, v_forecast.array.offset, v_forecast.array.grid
            )

            # Create new GridVariable
            u_new = GridVariable(u_array_new, u_forecast.bc)
            v_new = GridVariable(v_array_new, v_forecast.bc)

            analysis_ensemble.append((u_new, v_new))

        return analysis_ensemble

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

    def reset_adaptation_history(self) -> None:
        """Reset the adaptation history."""
        self.radius_history = []
        self.innovation_stats = []
