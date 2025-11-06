from typing import Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import torch
import torch_cfd.finite_differences as fdm
import xarray
from torch_cfd import advection, boundaries, grids
from torch_cfd.fvm import NavierStokes2DFVMProjection, PressureProjection, RKStepper
from torch_cfd.grids import GridVariable
from torch_cfd.initial_conditions import velocity_field
from tqdm import tqdm

from scisi.external_libs.torch_cfd.boundary_conditions import (
    get_inlet_velocities_from_angle,
    karman_vortex_multiple_squares_boundary_conditions,
)

dtype = torch.float32

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("mps")


def vorticity(v: Tuple[GridVariable, GridVariable]) -> torch.Tensor:
    """Compute vorticity from velocity field."""

    nx = v[0].data.shape[1]
    ny = v[0].data.shape[2]
    batch_size = v[0].data.shape[0]

    def vorticity_fn(ds: xarray.Dataset) -> xarray.DataArray:
        """Compute vorticity from velocity field."""
        return (ds.v.differentiate("x") - ds.u.differentiate("y")).rename("vorticity")

    coords = {
        "batch": np.linspace(0, batch_size - 1, batch_size, dtype=np.int64),
        "x": np.linspace(0, 2, nx, dtype=np.float64),
        "y": np.linspace(0, 1, ny, dtype=np.float64),
    }

    u_data = xarray.DataArray(
        v[0].data.detach().cpu(), dims=["batch", "x", "y"], coords=coords
    ).to_dataset(name="u")

    v_data = xarray.DataArray(
        v[1].data.detach().cpu(), dims=["batch", "x", "y"], coords=coords
    ).to_dataset(name="v")

    data = xarray.merge([u_data, v_data]).assign(vorticity=vorticity_fn)

    vorticity_data = data["vorticity"].data
    vorticity_data = torch.from_numpy(vorticity_data)

    return vorticity_data


class DynamicsModel:
    """Dynamics model for Navier-Stokes flow past obstacle."""

    def __init__(
        self,
        inlet_velocity_angle: float,
        nx: int = 400,
        ny: int = 200,
        density: float = 1.0,
        viscosity: float = 1 / 500,
        domain: tuple = ((0, 2), (0, 1)),
        obstacle_centers: List[Tuple[float, float]] = [
            (1.2, 0.75),
            (0.3, 0.5),
            (1.2, 0.25),
        ],
        obstacle_halfwidths: List[float] = [0.05, 0.08, 0.05],
        dt: float = 1e-3,
        batch_size: int = 1,
        num_inner_steps: int = 100,
        device: torch.device = DEVICE,
        dtype: torch.dtype = dtype,
    ):
        """
        Initialize the dynamics model.

        Parameters
        ----------
        inlet_velocity_angle : float
            Inlet velocity angle in degrees (0° = right, 90° = up, -90° = down, 180° = left)
        nx : int
            Number of grid points in x direction
        ny : int
            Number of grid points in y direction
        density : float
            Fluid density
        viscosity : float
            Fluid viscosity
        domain : tuple
            Domain boundaries ((x_min, x_max), (y_min, y_max))
        num_inner_steps : int, optional
            Number of inner steps for the dynamics function. If None, computed from defaults.
        device : torch.device, optional
            Device to run computations on. If None, uses CUDA if available.
        dtype : torch.dtype
            Data type for computations
        """
        self.inlet_velocity_angle = inlet_velocity_angle
        self.nx = nx
        self.ny = ny
        self.density = density
        self.viscosity = viscosity
        self.domain = domain
        self.obstacle_centers = obstacle_centers
        self.obstacle_halfwidths = obstacle_halfwidths
        self.dtype = dtype
        self.num_inner_steps = num_inner_steps
        self.dt = dt
        self.batch_size = batch_size
        self.device = device

        # Set up grid
        self.grid = grids.Grid((nx, ny), domain=domain, device=self.device)

        # Set up time stepper
        self.step_fn = RKStepper.from_method(
            method="classic_rk4", requires_grad=False, dtype=self.dtype
        )

        # Set up boundary conditions
        self.velocity_bc, self.pressure_bc = self.get_bcs(inlet_velocity_angle)

        # Get velocity field offsets (need a sample to get offsets)
        x_velocity_fn = lambda x, y: torch.zeros_like(x)
        y_velocity_fn = lambda x, y: torch.zeros_like(x)

        v_sample = velocity_field(
            (x_velocity_fn, y_velocity_fn),
            self.grid,
            velocity_bc=self.velocity_bc,
            batch_size=self.batch_size,
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
            square_centers=self.obstacle_centers,
            square_halfwidths=self.obstacle_halfwidths,
            periodic_y=True,
        )

    def get_model(self) -> NavierStokes2DFVMProjection:

        # Set up pressure projection
        pressure_proj = PressureProjection(
            grid=self.grid,
            bc=self.pressure_bc,
            dtype=self.dtype,
            implementation="matmul",
        )

        # Set up convection
        convection = advection.ConvectionVector(
            grid=self.grid,
            offsets=self.offsets,
            bcs=self.velocity_bc,
            advect=advection.AdvectionVanLeer,
        )

        # Set up Navier-Stokes model
        model = NavierStokes2DFVMProjection(
            viscosity=self.viscosity,
            grid=self.grid,
            bcs=self.velocity_bc,
            density=self.density,
            step_fn=self.step_fn,
            pressure_proj=pressure_proj,
            convection=convection,
        ).to(self.device)

        return model

    def update_parameters(self, inlet_velocity_angle: float) -> None:

        # Set up boundary conditions
        self.velocity_bc, self.pressure_bc = self.get_bcs(inlet_velocity_angle)

        self.model = self.get_model()

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
        for _ in range(self.num_inner_steps - 1):
            v, p = self.model(v, self.dt)
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


def _run_single_model(args):
    """Helper function to run a single model (for multiprocessing)."""
    model, v = args
    return model(v)


class EnsembleDynamicsModel:
    """Ensemble dynamics model for running multiple simulations with different parameters in parallel."""

    def __init__(
        self,
        inlet_velocity_angles: List[float],
        nx: int = 400,
        ny: int = 200,
        density: float = 1.0,
        viscosity: float = 1 / 500,
        domain: tuple = ((0, 2), (0, 1)),
        obstacle_centers: List[List[Tuple[float, float]]] = None,
        obstacle_halfwidths: List[List[float]] = None,
        dt: float = 1e-3,
        batch_size: int = 1,
        num_inner_steps: int = 100,
        num_processes: int = None,
        device: torch.device = DEVICE,
        dtype: torch.dtype = dtype,
    ):
        """
        Initialize ensemble of dynamics models.

        Parameters
        ----------
        inlet_velocity_angles : List[float]
            List of inlet velocity angles for each ensemble member
        nx : int
            Number of grid points in x direction
        ny : int
            Number of grid points in y direction
        density : float
            Fluid density
        viscosity : float
            Fluid viscosity
        domain : tuple
            Domain boundaries ((x_min, x_max), (y_min, y_max))
        obstacle_centers : List[List[Tuple[float, float]]], optional
            List of obstacle center lists, one for each ensemble member.
            If None, uses default [(1.2, 0.75), (0.3, 0.5), (1.2, 0.25)] for all members.
            Can also be a single list to use same obstacles for all members.
        obstacle_halfwidths : List[List[float]], optional
            List of obstacle halfwidth lists, one for each ensemble member.
            If None, uses default [0.05, 0.08, 0.05] for all members.
            Can also be a single list to use same halfwidths for all members.
        dt : float
            Time step size
        batch_size : int
            Batch size for each model
        num_inner_steps : int
            Number of inner steps for the dynamics function
        num_processes : int, optional
            Number of processes to use for parallel execution.
            If None, runs sequentially. If > 1, uses ProcessPoolExecutor.
        device : torch.device
            Device to run computations on
        dtype : torch.dtype
            Data type for computations
        """
        self.inlet_velocity_angles = inlet_velocity_angles
        self.num_ensemble = len(inlet_velocity_angles)
        self.nx = nx
        self.ny = ny
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.num_processes = num_processes

        # Default obstacle configuration
        default_centers = [(1.2, 0.75), (0.3, 0.5), (1.2, 0.25)]
        default_halfwidths = [0.05, 0.08, 0.05]

        # Handle obstacle_centers input
        if obstacle_centers is None:
            # Use default for all ensemble members
            self.obstacle_centers = [default_centers] * self.num_ensemble
        elif len(obstacle_centers) > 0 and isinstance(obstacle_centers[0], tuple):
            # Single list of centers provided - use for all members
            self.obstacle_centers = [obstacle_centers] * self.num_ensemble
        else:
            # List of lists provided - one per ensemble member
            if len(obstacle_centers) != self.num_ensemble:
                raise ValueError(
                    f"Expected {self.num_ensemble} obstacle_centers lists, got {len(obstacle_centers)}"
                )
            self.obstacle_centers = obstacle_centers

        # Handle obstacle_halfwidths input
        if obstacle_halfwidths is None:
            # Use default for all ensemble members
            self.obstacle_halfwidths = [default_halfwidths] * self.num_ensemble
        elif len(obstacle_halfwidths) > 0 and isinstance(obstacle_halfwidths[0], (int, float)):
            # Single list of halfwidths provided - use for all members
            self.obstacle_halfwidths = [obstacle_halfwidths] * self.num_ensemble
        else:
            # List of lists provided - one per ensemble member
            if len(obstacle_halfwidths) != self.num_ensemble:
                raise ValueError(
                    f"Expected {self.num_ensemble} obstacle_halfwidths lists, got {len(obstacle_halfwidths)}"
                )
            self.obstacle_halfwidths = obstacle_halfwidths

        # Create a model for each ensemble member
        self.models = [
            DynamicsModel(
                inlet_velocity_angle=angle,
                nx=nx,
                ny=ny,
                density=density,
                viscosity=viscosity,
                domain=domain,
                obstacle_centers=centers,
                obstacle_halfwidths=halfwidths,
                dt=dt,
                batch_size=batch_size,
                num_inner_steps=num_inner_steps,
                device=device,
                dtype=dtype,
            )
            for angle, centers, halfwidths in zip(
                inlet_velocity_angles, self.obstacle_centers, self.obstacle_halfwidths
            )
        ]

    def __call__(
        self,
        v_list: List[Tuple[GridVariable, GridVariable]],
        parallel: bool = None,
    ) -> List[Tuple[GridVariable, GridVariable]]:
        """
        Apply dynamics to ensemble of velocity fields.

        Parameters
        ----------
        v_list : List[Tuple[GridVariable, GridVariable]]
            List of velocity fields, one for each ensemble member
        parallel : bool, optional
            If True, uses parallel execution with ProcessPoolExecutor.
            If False, runs sequentially. If None, uses num_processes setting.

        Returns
        -------
        List[Tuple[GridVariable, GridVariable]]
            List of updated velocity fields
        """
        if len(v_list) != self.num_ensemble:
            raise ValueError(
                f"Expected {self.num_ensemble} velocity fields, got {len(v_list)}"
            )

        # Determine whether to run in parallel
        use_parallel = parallel if parallel is not None else (self.num_processes is not None and self.num_processes > 1)

        if use_parallel:
            # Run in parallel using ProcessPoolExecutor
            num_workers = self.num_processes if self.num_processes is not None else self.num_ensemble
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(_run_single_model, zip(self.models, v_list)))
        else:
            # Run sequentially
            results = []
            for model, v in zip(self.models, v_list):
                v_new, p = model(v)
                results.append((v_new, p))

        return results

    def update_parameters(self, inlet_velocity_angles: List[float]) -> None:
        """
        Update inlet velocity angles for all ensemble members.

        Parameters
        ----------
        inlet_velocity_angles : List[float]
            New inlet velocity angles for each ensemble member
        """
        if len(inlet_velocity_angles) != self.num_ensemble:
            raise ValueError(
                f"Expected {self.num_ensemble} angles, got {len(inlet_velocity_angles)}"
            )

        self.inlet_velocity_angles = inlet_velocity_angles
        for model, angle in zip(self.models, inlet_velocity_angles):
            model.update_parameters(angle)

    def forward_with_parameters(
        self,
        v_list: List[Tuple[GridVariable, GridVariable]],
        inlet_velocity_angles: List[float],
        parallel: bool = None,
    ) -> List[Tuple[GridVariable, GridVariable]]:
        """
        Apply dynamics with updated parameters.

        Parameters
        ----------
        v_list : List[Tuple[GridVariable, GridVariable]]
            List of velocity fields, one for each ensemble member
        inlet_velocity_angles : List[float]
            Inlet velocity angles for each ensemble member
        parallel : bool, optional
            If True, uses parallel execution. If False, runs sequentially.
            If None, uses num_processes setting.

        Returns
        -------
        List[Tuple[GridVariable, GridVariable]]
            List of updated velocity fields
        """
        self.update_parameters(inlet_velocity_angles)
        return self.__call__(v_list, parallel=parallel)
