from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import ray
import torch
from torch_cfd import advection, grids
from torch_cfd.fvm import NavierStokes2DFVMProjection, PressureProjection, RKStepper
from torch_cfd.grids import GridVariable
from torch_cfd.initial_conditions import velocity_field
from torch_cfd_lib.boundary_conditions import (
    get_inlet_velocities_from_angle,
    karman_vortex_multiple_squares_boundary_conditions,
)

dtype = torch.float32


@dataclass
class DynamicsModelConfig:
    inlet_velocity_angle: float
    nx: int
    ny: int
    density: float
    viscosity: float
    domain: tuple
    dt: float
    num_inner_steps: int
    dtype: torch.dtype
    obstacle_centers: List[Tuple[float, float]]
    obstacle_halfwidths: List[float]


# Default obstacle configuration
DEFAULT_CENTERS = [(1.2, 0.75), (0.3, 0.5), (1.2, 0.25)]
DEFAULT_HALFWIDTHS = [0.05, 0.08, 0.05]


class DynamicsModel:
    """Dynamics model for Navier-Stokes flow past obstacle."""

    def __init__(
        self,
        config: DynamicsModelConfig,
        device: torch.device = "cpu",
    ):
        """
        Initialize the dynamics model.

        Parameters
        ----------
        config : DynamicsModelConfig
            Configuration for the dynamics model
        device : torch.device, optional
            Device to run computations on. If None, uses CUDA if available.
        """
        self.device = device
        self.config = config

        # Set up grid
        self.grid = grids.Grid(
            (config.nx, config.ny),
            domain=config.domain,
            device=self.device,
        )

        # Set up time stepper
        self.step_fn = RKStepper.from_method(
            method="classic_rk4", requires_grad=False, dtype=self.config.dtype
        )

        # Set up boundary conditions
        self.velocity_bc, self.pressure_bc = self.get_bcs(config.inlet_velocity_angle)

        # Get velocity field offsets (need a sample to get offsets)
        x_velocity_fn = lambda x, y: torch.zeros_like(x)
        y_velocity_fn = lambda x, y: torch.zeros_like(x)

        v_sample = velocity_field(
            (x_velocity_fn, y_velocity_fn),
            self.grid,
            velocity_bc=self.velocity_bc,
            batch_size=1,
            random_state=0,
            noise=0.0,
            device=self.device,
        )

        self.offsets = (v_sample[0].offset, v_sample[1].offset)

        self.model = self.get_model()

    def get_bcs(self, inlet_velocity_angle: float) -> Any:
        self.x_inlet_velocity, self.y_inlet_velocity = get_inlet_velocities_from_angle(
            inlet_velocity_angle
        )
        return karman_vortex_multiple_squares_boundary_conditions(
            self.grid,
            inlet_velocity=(self.x_inlet_velocity, self.y_inlet_velocity),
            square_centers=self.config.obstacle_centers,
            square_halfwidths=self.config.obstacle_halfwidths,
            periodic_y=True,
        )

    def get_initial_condition(self) -> Any:
        """Get the initial condition for the dynamics model."""
        x_velocity_fn = lambda x, y: self.x_inlet_velocity * torch.ones_like(x)
        y_velocity_fn = lambda x, y: self.y_inlet_velocity * torch.ones_like(x)

        return velocity_field(
            (x_velocity_fn, y_velocity_fn),
            self.grid,
            velocity_bc=self.velocity_bc,
            batch_size=1,
            random_state=0,
            noise=0.1,
            device=self.device,
        )

    def get_model(self) -> NavierStokes2DFVMProjection:

        # Set up pressure projection
        pressure_proj = PressureProjection(
            grid=self.grid,
            bc=self.pressure_bc,
            dtype=self.config.dtype,
            implementation="rfft",
            solver="pseudoinverse",
        )

        # Set up convection
        convection = advection.ConvectionVector(
            grid=self.grid,
            offsets=self.offsets,
            bcs=self.velocity_bc,
            advect=advection.AdvectionVanLeer,
        )

        return NavierStokes2DFVMProjection(
            viscosity=self.config.viscosity,
            grid=self.grid,
            bcs=self.velocity_bc,
            density=self.config.density,
            step_fn=self.step_fn,
            pressure_proj=pressure_proj,
            convection=convection,
        ).to(self.device)

    def update_parameters(self, inlet_velocity_angle: float) -> None:
        """Update the parameters of the dynamics model."""

        self.config.inlet_velocity_angle = inlet_velocity_angle

        # Set up boundary conditions
        self.velocity_bc, self.pressure_bc = self.get_bcs(
            self.config.inlet_velocity_angle
        )

        self.model = self.get_model()

    def update_config(self, config: DynamicsModelConfig) -> None:
        """Update the configuration of the dynamics model."""
        self.config = config
        self.update_parameters(self.config.inlet_velocity_angle)

    def __call__(
        self,
        v: Tuple[GridVariable, GridVariable],
    ) -> Tuple[GridVariable, GridVariable]:
        """
        Apply dynamics for num_inner_steps iterations.

        Parameters
        ----------
        v : tuple
            Velocity field as tuple of (u, v) GridVariables

        Returns
        -------
        tuple
            Updated velocity field (v_new, p)
        """
        for _ in range(self.config.num_inner_steps - 1):
            v, p = self.model(v, self.config.dt)
        return v, p

    def forward_with_parameters(
        self,
        v: Tuple[GridVariable, GridVariable],
        inlet_velocity_angle: float = 0.0,
    ) -> Tuple[GridVariable, GridVariable]:
        """
        Apply dynamics for num_inner_steps iterations.

        Parameters
        ----------
        v : tuple
            Velocity field as tuple of (u, v) GridVariables
        inlet_velocity_angle : float
            Parameters for the dynamics model

        Returns
        -------
        tuple
            Updated velocity field (v_new, p)
        """
        self.update_parameters(inlet_velocity_angle)
        return self.__call__(v)


@ray.remote(num_gpus=0, num_cpus=1)  # type: ignore[misc]
def run_model(
    v: Tuple[GridVariable, GridVariable],
    config: DynamicsModelConfig,
    device: torch.device,
) -> Tuple[GridVariable, GridVariable]:
    """Run the dynamics model."""
    return DynamicsModel(config=config, device=device)(v)


class EnsembleDynamicsModel:
    """Ensemble dynamics model for running multiple simulations with different parameters in parallel."""

    def __init__(
        self,
        configs: List[DynamicsModelConfig],
        num_workers: int = 1,
        device: torch.device = "cpu",
    ):
        """
        Initialize ensemble of dynamics models.

        Parameters
        ----------
        config : DynamicsModelConfig
        num_workers : int, optional
            Number of workers to use for parallel execution.
            If None, runs sequentially. If > 1, uses parallel execution.
        device : torch.device
            Device to run computations on
        """
        self.configs = configs
        self.num_ensemble = len(configs)
        self.device = device
        self.num_workers = num_workers

        if self.num_workers > 1:

            ray.shutdown()

            self.use_ray = True
            ray.init(
                num_cpus=self.num_workers,
                num_gpus=0,
                runtime_env={
                    "excludes": ["*"],  # Exclude all files from being packaged
                },
            )

        else:
            self.use_ray = False

    def get_initial_conditions(self) -> List[Tuple[GridVariable, GridVariable]]:
        """Get the initial condition for the ensemble of dynamics models."""
        return [
            DynamicsModel(config=config, device=self.device).get_initial_condition()
            for config in self.configs
        ]

    def _run_parallel(
        self,
        v_list: List[Tuple[GridVariable, GridVariable]],
    ) -> Tuple[List[GridVariable], List[GridVariable]]:
        """Run the dynamics model in parallel using Ray."""
        futures = [
            run_model.remote(v, config, self.device)
            for config, v in zip(self.configs, v_list)
        ]
        futures = ray.get(futures)
        v_results = [future[0] for future in futures]
        p_results = [future[1] for future in futures]
        return v_results, p_results

    def _run_sequential(
        self,
        v_list: List[Tuple[GridVariable, GridVariable]],
    ) -> Tuple[List[GridVariable], List[GridVariable]]:
        """Run the dynamics model sequentially."""
        v_results = []
        p_results = []
        for config, v in zip(self.configs, v_list):
            model = DynamicsModel(config=config, device=self.device)
            v_new, p = model(v)
            v_results.append(v_new)
            p_results.append(p)
        return v_results, p_results

    def __call__(
        self,
        v_list: List[Tuple[GridVariable, GridVariable]],
    ) -> Tuple[List[GridVariable], List[GridVariable]]:
        """
        Apply dynamics to ensemble of velocity fields.

        Parameters
        ----------
        v_list : List[Tuple[GridVariable, GridVariable]]
            List of velocity fields, one for each ensemble member

        Returns
        -------
        List[Tuple[GridVariable, GridVariable]]
            List of updated velocity fields
        """
        if len(v_list) != self.num_ensemble:
            raise ValueError(
                f"Expected {self.num_ensemble} velocity fields, got {len(v_list)}"
            )

        if self.use_ray:
            # Run in parallel using Ray
            results = self._run_parallel(v_list)
        else:
            # Run sequentially
            results = self._run_sequential(v_list)

        return results

    def shutdown(self) -> None:
        """Shutdown Ray resources if they were initialized."""
        if self.use_ray:
            ray.shutdown()

    def __del__(self) -> None:
        """Cleanup Ray resources on deletion."""
        self.shutdown()
