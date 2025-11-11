import os
import pdb
import sys
from pathlib import Path

import torch
from torch_cfd import grids
from torch_cfd.initial_conditions import velocity_field
from torch_cfd_lib.forward_model import DynamicsModel, DynamicsModelConfig
from torch_cfd_lib.utils import (
    get_tensor_from_grid_variables,
    get_vorticity_from_grid_variables,
)
from tqdm import tqdm

from scisi.plotting.animation import create_animation_from_tensors

dtype = torch.float32

NX = 400
NY = 200
DENSITY = 1.0
HF_DT = 1e-3
REDUCED_DT = 1e-1
BATCH_SIZE = 1
# VISCOSITY = 1 / 500
VISCOSITY = 1 / 500
FINAL_TIME = 1.0
OUTER_STEPS = int(FINAL_TIME // REDUCED_DT)
INNER_STEPS = int(FINAL_TIME // HF_DT) // OUTER_STEPS
DOMAIN = ((0, 2), (0, 1))

# Define obstacle as a grid of 3 by 3
OBSTACLE_CENTERS = []
OBSTACLE_HALFWIDTHS = []

x_positions = [0.5, 1.0, 1.5]
y_positions = [0.2, 0.5, 0.8]
halfwidth = 0.1
# x_positions = [0.5]
# y_positions = [0.5]
# halfwidth = 0.1

for x in x_positions:
    for y in y_positions:
        OBSTACLE_CENTERS.append((x, y))
        OBSTACLE_HALFWIDTHS.append(halfwidth)


# OBSTACLE_CENTERS = [(1.2, 0.75), (0.3, 0.5), (1.2, 0.25), (0.7, 0.75), (0.7, 0.25), (0.3, 0.75), (0.3, 0.25)]
# OBSTACLE_HALFWIDTHS = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("mps")


def main() -> None:
    """Main function."""

    config = DynamicsModelConfig(
        inlet_velocity_angle=0.0,
        nx=NX,
        ny=NY,
        density=DENSITY,
        viscosity=VISCOSITY,
        domain=DOMAIN,
        obstacle_centers=OBSTACLE_CENTERS,
        obstacle_halfwidths=OBSTACLE_HALFWIDTHS,
        dt=HF_DT,
        num_inner_steps=INNER_STEPS,
        dtype=dtype,
    )
    model = DynamicsModel(config=config, device=DEVICE)

    v = model.get_initial_condition()

    trajectory_plot = torch.zeros(2, NX, NY, OUTER_STEPS)
    vorticity_plot = torch.zeros(NX, NY, OUTER_STEPS)
    with torch.no_grad():
        for i in tqdm(range(OUTER_STEPS)):
            # v, _ = model.forward_with_parameters(v, inflow_angle_vec[i])
            v, _ = model(v)

            trajectory_plot[:, :, :, i] = get_tensor_from_grid_variables(v)
            vorticity_plot[:, :, i] = get_vorticity_from_grid_variables(v)

    trajectory_plot = trajectory_plot[:, ::2, ::2]
    # torch.save(trajectory_plot, "trajectory.pt")

    vel_mag = torch.sqrt(trajectory_plot[0] ** 2 + trajectory_plot[1] ** 2)

    figure_dir = "figures/torch_cfd"
    os.makedirs(figure_dir, exist_ok=True)

    create_animation_from_tensors(
        [vel_mag],
        fps=10,
        file_name=f"{figure_dir}/velocity_magnitude.mp4",
        colormaps="viridis",
        titles=["Velocity Magnitude"],
        # vmin=-1.5,
        # vmax=1.5,
        normalize=False,
    )
    create_animation_from_tensors(
        [vorticity_plot],
        fps=10,
        file_name=f"{figure_dir}/vorticity_trajectory.mp4",
        colormaps="viridis",
        titles=["Vorticity"],
        vmin=-30,
        vmax=30,
        normalize=False,
    )


if __name__ == "__main__":
    main()
