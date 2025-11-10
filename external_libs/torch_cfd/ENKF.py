"""
Ensemble Kalman Filter for PDEs in Physical Space with State and Parameter Estimation

Handles systems where:
- State x is represented in physical space (no Fourier transforms)
- Dynamics F operates in physical space
- Parameters θ can be estimated along with states
- Observations y are in physical space

State equation: x_{t+1} = F(x_t, θ_t) + model_noise  (Physical space)
Parameter equation: θ_{t+1} = θ_t + parameter_noise
Observation equation: y_{t+1} = H(x_{t+1}) + obs_noise  (Physical space)

where H is the observation operator mapping physical space to observation points.
"""

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_cfd import grids
from torch_cfd.grids import GridVariable

from scisi.external_libs.torch_cfd.forward_model import DynamicsModel


class ObservationOperator:
    """
    Observation operator for physical space velocity fields.

    Maps velocity fields (tuple of GridVariables) to observations at specific grid points.
    Can also handle tensor format [batch, channels, nx, ny, time].
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

    def __call__(self, velocity: Union[Tuple[Any, Any], torch.Tensor]) -> torch.Tensor:
        """
        Observation operator H: maps velocity fields to physical space observations.

        Parameters
        ----------
        velocity : Union[Tuple[GridVariable, GridVariable], torch.Tensor]
            Velocity field as tuple of (u, v) GridVariables or tensor [channels, nx, ny]

        Returns
        -------
        torch.Tensor
            Observations at observation points, shape (n_obs * n_components,)
        """
        if isinstance(velocity, tuple):
            # GridVariable format
            u, v = velocity
            u_data = u.data.squeeze()  # Remove batch dimension if present
            v_data = v.data.squeeze()
        else:
            # Tensor format [channels, nx, ny]
            u_data = velocity[0]
            v_data = velocity[1]

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
            # Flatten to (n_obs * n_components,)
            return torch.cat(obs_list, dim=0)


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
    ) -> List[Tuple[Any, Any]]:
        """
        Forecast step in physical space.

        Parameters
        ----------
        ensemble_velocity : List[Tuple[GridVariable, GridVariable]]
            Ensemble of velocity fields, each as tuple (u, v)
        dynamics : Callable
            Dynamics function operating on velocity fields: (v) -> (v_new, p)

        Returns
        -------
        List[Tuple[GridVariable, GridVariable]]
            Forecasted ensemble of velocity fields
        """
        forecast_ensemble = []
        for velocity in ensemble_velocity:
            # Dynamics function takes velocity, returns (v_new, p)
            v_forecast, _ = dynamics(velocity)
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

            # Create new GridVariable with updated data
            u_new = GridVariable(
                data=u_data_new,
                offset=u_forecast.offset,
                grid=u_forecast.grid,
                bc=u_forecast.bc,
            )
            v_new = GridVariable(
                data=v_data_new,
                offset=v_forecast.offset,
                grid=v_forecast.grid,
                bc=v_forecast.bc,
            )

            analysis_ensemble.append((u_new, v_new))

        return analysis_ensemble

    def assimilate(
        self,
        ensemble_velocity: List[Tuple[Any, Any]],
        observations: torch.Tensor,
        dynamics: Callable,
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
            Dynamics function: (v) -> (v_new, p)
        inflation : float
            Covariance inflation factor
        localization_radius : float, optional
            Localization radius (for compatibility, not used in base class)

        Returns
        -------
        Tuple[List[Tuple[GridVariable, GridVariable]], List[Tuple[GridVariable, GridVariable]]]
            (analysis_ensemble, forecast_ensemble)
        """
        forecast_ensemble = self.forecast_step(ensemble_velocity, dynamics)
        analysis_ensemble = self.analysis_step(
            forecast_ensemble,
            observations,
            inflation,
            localization_radius,
        )

        return analysis_ensemble, forecast_ensemble


class StateParameterEnKF(PhysicalEnKF):
    """
    Ensemble Kalman Filter for joint state and parameter estimation.

    This class extends PhysicalEnKF to simultaneously estimate both the velocity
    field (state) and model parameters (e.g., inlet_velocity_angle).

    The augmented state vector is: [x, θ] where x is the velocity field and θ are parameters.

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
    parameter_noise_std : float
        Standard deviation of parameter noise (for parameter random walk)
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
        parameter_noise_std: float,
        observation_operator: ObservationOperator,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
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
        self.parameter_noise_std = parameter_noise_std

    def forecast_step_with_parameters(
        self,
        ensemble_velocity: List[Tuple[Any, Any]],
        ensemble_parameters: torch.Tensor,
        dynamics_with_params: DynamicsModel,
    ) -> Tuple[List[Tuple[Any, Any]], torch.Tensor]:
        """
        Forecast step with parameter estimation.

        Parameters
        ----------
        ensemble_velocity : List[Tuple[GridVariable, GridVariable]]
            Ensemble of velocity fields
        ensemble_parameters : torch.Tensor
            Ensemble of parameters, shape (ensemble_size, n_params)
        dynamics_with_params : DynamicsModel
            Dynamics function: (v, params) -> (v_new, p)

        Returns
        -------
        Tuple[List[Tuple[GridVariable, GridVariable]], torch.Tensor]
            (forecasted_ensemble_velocity, forecasted_ensemble_parameters)
        """
        forecast_ensemble = torch.zeros(
            self.ensemble_size,
            2,
            self.grid_shape[0],
            self.grid_shape[1],
            device=self.device,
            dtype=self.dtype,
        )
        forecast_parameters = torch.zeros(
            self.ensemble_size, 1, device=self.device, dtype=self.dtype
        )

        for i, velocity in enumerate(ensemble_velocity):
            params = ensemble_parameters[i]

            # Dynamics function takes velocity and parameters, returns (v_new, p)
            v_forecast, _ = dynamics_with_params.forward_with_parameters(
                velocity, params[0].item()
            )

            u, v = v_forecast
            u_data = u.data.squeeze()
            v_data = v.data.squeeze()
            forecast_ensemble[i, 0] = u_data
            forecast_ensemble[i, 1] = v_data

            # Parameters follow random walk: θ_{t+1} = θ_t + noise
            param_noise = torch.randn_like(params) * self.parameter_noise_std
            params_forecast = params + param_noise
            forecast_parameters[i] = params_forecast

        return forecast_ensemble, forecast_parameters

    def analysis_step_with_parameters(
        self,
        forecast_ensemble_velocity: torch.Tensor,
        forecast_ensemble_parameters: torch.Tensor,
        observations: torch.Tensor,
        inflation: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Analysis step with joint state and parameter estimation.

        The augmented state vector [x, θ] is used for the Kalman update.

        Parameters
        ----------
        forecast_ensemble_velocity : List[Tuple[GridVariable, GridVariable]]
            Forecasted ensemble of velocity fields
        forecast_ensemble_parameters : torch.Tensor
            Forecasted ensemble of parameters, shape (ensemble_size, n_params)
        observations : torch.Tensor
            Observations in physical space
        inflation : float
            Covariance inflation factor

        Returns
        -------
        Tuple[List[Tuple[GridVariable, GridVariable]], torch.Tensor]
            (analysis_ensemble_velocity, analysis_ensemble_parameters)
        """
        n_obs = len(observations)

        original_shape = forecast_ensemble_velocity.shape

        # Flatten spatial dimensions: (ensemble_size, 2 * nx * ny)
        ensemble_augmented = forecast_ensemble_velocity.reshape(self.ensemble_size, -1)
        n_state = ensemble_augmented.shape[1]

        # Augment state with parameters: (ensemble_size, 2*nx*ny + n_params)
        ensemble_augmented = torch.cat(
            [ensemble_augmented, forecast_ensemble_parameters], dim=1
        )

        # Compute ensemble mean and perturbations for augmented state
        x_aug_mean = torch.mean(ensemble_augmented, dim=0)
        X_aug_pert = inflation * (ensemble_augmented - x_aug_mean)

        # Map ensemble to observation space
        HX = []
        for velocity in forecast_ensemble_velocity:
            obs = self.observation_operator(velocity)
            HX.append(obs)
        HX = torch.stack(HX, dim=0)  # (ensemble_size, n_obs)

        HX_mean = torch.mean(HX, dim=0)
        HX_pert = HX - HX_mean

        # Compute covariances for augmented state
        # P_xy = X_aug_pert^T @ HX_pert / (N - 1)
        Pxy = (X_aug_pert.T @ HX_pert) / (self.ensemble_size - 1)

        # P_yy = HX_pert^T @ HX_pert / (N - 1) + R
        R = (self.obs_noise_std**2) * torch.eye(
            n_obs, device=self.device, dtype=self.dtype
        )
        Pyy = (HX_pert.T @ HX_pert) / (self.ensemble_size - 1) + R

        # Kalman gain: K = P_xy @ (P_yy)^{-1}
        Pyy_inv = torch.linalg.inv(Pyy)
        K = Pxy @ Pyy_inv

        # Generate perturbed observations
        obs_noise = (
            torch.randn(self.ensemble_size, n_obs, device=self.device, dtype=self.dtype)
            * self.obs_noise_std
        )
        perturbed_obs = observations + obs_noise

        # Update augmented ensemble
        innovations = perturbed_obs - HX
        analysis_physical = ensemble_augmented + innovations @ K.T

        # Split back into state and parameters
        analysis_parameters = analysis_physical[:, n_state:]
        analysis_physical = analysis_physical[:, :n_state]

        # Reshape state back to physical space
        analysis_physical = analysis_physical.reshape(original_shape)

        # Convert back to GridVariable format
        analysis_ensemble = []
        for i, forecast_velocity in enumerate(forecast_ensemble_velocity):
            # u_forecast, v_forecast = forecast_velocity

            u_data_new = analysis_physical[i, 0]
            v_data_new = analysis_physical[i, 1]

            # Create new GridVariable with updated data
            u_new = GridVariable(
                data=u_data_new,
                offset=self.grid_info["u_offset"],
                grid=self.grid_info["u_grid"],
                bc=self.grid_info["u_bc"][i],
            )
            v_new = GridVariable(
                data=v_data_new,
                offset=self.grid_info["v_offset"],
                grid=self.grid_info["v_grid"],
                bc=self.grid_info["v_bc"][i],
            )

            analysis_ensemble.append((u_new, v_new))

        return analysis_ensemble, analysis_parameters

    def assimilate_with_parameters(
        self,
        ensemble_velocity: List[Tuple[Any, Any]],
        ensemble_parameters: torch.Tensor,
        observations: torch.Tensor,
        dynamics_model: DynamicsModel,
        inflation: float = 1.0,
    ) -> Tuple[List[Tuple[Any, Any]], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Complete assimilation cycle with parameter estimation.

        Parameters
        ----------
        ensemble_velocity : List[Tuple[GridVariable, GridVariable]]
            Current ensemble of velocity fields
        ensemble_parameters : torch.Tensor
            Current ensemble of parameters, shape (ensemble_size, n_params)
        observations : torch.Tensor
            Observations in physical space
        dynamics_model : DynamicsModel
            Dynamics function: (v, params) -> (v_new, p)
        inflation : float
            Covariance inflation factor

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            (analysis_ensemble_velocity, analysis_parameters,
             forecast_ensemble_velocity, forecast_parameters)
        """

        self.grid_info = {}
        self.grid_info["u_offset"] = ensemble_velocity[0][0].offset
        self.grid_info["v_offset"] = ensemble_velocity[0][1].offset

        self.grid_info["u_grid"] = ensemble_velocity[0][0].grid
        self.grid_info["v_grid"] = ensemble_velocity[0][1].grid
        self.grid_info["u_bc"] = [velocity[0].bc for velocity in ensemble_velocity]
        self.grid_info["v_bc"] = [velocity[1].bc for velocity in ensemble_velocity]

        # Forecast step
        forecast_ensemble, forecast_parameters = self.forecast_step_with_parameters(
            ensemble_velocity, ensemble_parameters, dynamics_model
        )

        # Analysis step
        analysis_ensemble, analysis_parameters = self.analysis_step_with_parameters(
            forecast_ensemble, forecast_parameters, observations, inflation
        )

        return (
            analysis_ensemble,
            analysis_parameters,
            forecast_ensemble,
            forecast_parameters,
        )
