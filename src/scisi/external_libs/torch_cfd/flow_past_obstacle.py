import math
from typing import Any, NoReturn, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
from scisi.plotting.animation import create_animation_from_tensors

velocity_bc_types = Tuple[
    boundaries.ImmersedBoundaryConditions, boundaries.ConstantBoundaryConditions
]
pressure_bc_types = boundaries.ConstantBoundaryConditions

dtype = torch.float32

NX = 400
NY = 200
DENSITY = 1.0
HF_DT = 1e-3
REDUCED_DT = 1e-1
BATCH_SIZE = 1
VISCOSITY = 1 / 500
FINAL_TIME = 5.0
OUTER_STEPS = int(FINAL_TIME // REDUCED_DT)
INNER_STEPS = int(FINAL_TIME // HF_DT) // OUTER_STEPS
DOMAIN = ((0, 2), (0, 1))


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def vorticity(v: Tuple[GridVariable, GridVariable]) -> torch.Tensor:
    """Compute vorticity from velocity field."""

    def vorticity_fn(ds: xarray.Dataset) -> xarray.DataArray:
        """Compute vorticity from velocity field."""
        return (ds.v.differentiate("x") - ds.u.differentiate("y")).rename("vorticity")

    coords = {
        "batch": np.linspace(0, BATCH_SIZE - 1, BATCH_SIZE, dtype=np.int64),
        "x": np.linspace(0, 2, NX, dtype=np.float64),
        "y": np.linspace(0, 1, NY, dtype=np.float64),
    }

    u_data = xarray.DataArray(
        v[0].data, dims=["batch", "x", "y"], coords=coords
    ).to_dataset(name="u")

    v_data = xarray.DataArray(
        v[1].data, dims=["batch", "x", "y"], coords=coords
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
        nx: int = NX,
        ny: int = NY,
        density: float = DENSITY,
        viscosity: float = VISCOSITY,
        domain: tuple = DOMAIN,
        dt: float = HF_DT,
        num_inner_steps: int = INNER_STEPS,
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
        self.dtype = dtype
        self.num_inner_steps = num_inner_steps
        self.dt = dt
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
        x_velocity_fn = lambda x, y: torch.ones_like(x)
        y_velocity_fn = lambda x, y: torch.ones_like(x)

        v_sample = velocity_field(
            (x_velocity_fn, y_velocity_fn),
            self.grid,
            velocity_bc=self.velocity_bc,
            batch_size=BATCH_SIZE,
            random_state=0,
            noise=0.0,
            device=self.device,
        )

        self.offsets = (v_sample[0].offset, v_sample[1].offset)

        self.model = self.get_model_from_bcs(self.velocity_bc, self.pressure_bc)

    def get_bcs(self, inlet_velocity_angle: float) -> Any:
        self.x_inlet_velocity, self.y_inlet_velocity = get_inlet_velocities_from_angle(
            inlet_velocity_angle
        )
        return karman_vortex_multiple_squares_boundary_conditions(
            self.grid,
            inlet_velocity=(self.x_inlet_velocity, self.y_inlet_velocity),
            square_centers=[(1.2, 0.75), (0.3, 0.5), (1.2, 0.25)],
            square_halfwidths=[0.05, 0.08, 0.05],
            periodic_y=True,
        )

    def get_model_from_bcs(
        self, velocity_bc: Any, pressure_bc: Any
    ) -> NavierStokes2DFVMProjection:

        # Set up pressure projection
        pressure_proj = PressureProjection(
            grid=self.grid, bc=pressure_bc, dtype=self.dtype, implementation="matmul"
        )

        # Set up convection
        convection = advection.ConvectionVector(
            grid=self.grid,
            offsets=self.offsets,
            bcs=velocity_bc,
            advect=advection.AdvectionVanLeer,
        )

        # Set up Navier-Stokes model
        model = NavierStokes2DFVMProjection(
            viscosity=self.viscosity,
            grid=self.grid,
            bcs=velocity_bc,
            density=self.density,
            step_fn=self.step_fn,
            pressure_proj=pressure_proj,
            convection=convection,
        ).to(self.device)

        return model

    def update_parameters(self, inlet_velocity_angle: float) -> None:

        # Set up boundary conditions
        self.velocity_bc, self.pressure_bc = self.get_bcs(inlet_velocity_angle)

        self.model = self.get_model_from_bcs(self.velocity_bc, self.pressure_bc)

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


def main() -> None:
    """Main function."""

    grid = grids.Grid((NX, NY), domain=DOMAIN, device=DEVICE)

    inflow_angle = 45.0  # degrees

    inflow_angle_vec = torch.linspace(-45, 45, OUTER_STEPS // 5 + 1)
    inflow_angle_vec = torch.cat(
        [inflow_angle_vec, inflow_angle_vec[-1] * torch.ones(4 * OUTER_STEPS // 5 + 1)]
    )

    inflow_angle_vec = [45 for _ in range(OUTER_STEPS)]

    model = DynamicsModel(inflow_angle_vec[0])

    x_velocity_fn = lambda x, y: model.x_inlet_velocity * torch.ones_like(x)
    y_velocity_fn = lambda x, y: model.y_inlet_velocity * torch.ones_like(x)

    v = velocity_field(
        (x_velocity_fn, y_velocity_fn),
        grid,
        velocity_bc=model.velocity_bc,
        batch_size=BATCH_SIZE,
        random_state=42,
        noise=0.1,
        device=DEVICE,
    )

    vorticity_data = vorticity(v)
    trajectory = []
    trajectory_plot = torch.zeros(BATCH_SIZE, 2, NX, NY, OUTER_STEPS)
    vorticity_plot = torch.zeros(BATCH_SIZE, NX, NY, OUTER_STEPS)
    pbar = tqdm(range(OUTER_STEPS))  # , desc=desc)
    with torch.no_grad():
        for i in pbar:
            # v, p = model(v, HF_DT,)
            v, p = model.forward_with_parameters(v, inflow_angle_vec[i])

            trajectory.append(v)

            trajectory_plot[:, 0, :, :, i] = v[0].data.detach().cpu()
            trajectory_plot[:, 1, :, :, i] = v[1].data.detach().cpu()
            vorticity_plot[:, :, :, i] = vorticity(v)

    vel_mag = torch.sqrt(
        trajectory_plot[0, 0, :, :, :] ** 2 + trajectory_plot[0, 1, :, :, :] ** 2
    )

    create_animation_from_tensors(
        [vel_mag],
        fps=10,
        file_name=f"figures/velocity_magnitude.mp4",
        colormaps="viridis",
        titles=["Velocity Magnitude"],
        vmin=-1.5,
        vmax=1.5,
        normalize=False,
    )
    create_animation_from_tensors(
        [vorticity_plot[0]],
        fps=10,
        file_name=f"figures/vorticity_trajectory.mp4",
        colormaps="viridis",
        titles=["Vorticity"],
        vmin=-20,
        vmax=20,
        normalize=False,
    )


if __name__ == "__main__":
    main()
