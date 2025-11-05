from typing import Any, List, Tuple

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
