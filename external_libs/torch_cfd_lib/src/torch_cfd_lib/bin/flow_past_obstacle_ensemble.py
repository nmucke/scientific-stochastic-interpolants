import os
from pathlib import Path

import torch
from torch_cfd_lib.forward_model import DynamicsModelConfig, EnsembleDynamicsModel
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
VISCOSITY = 1 / 500
FINAL_TIME = 1.0
OUTER_STEPS = int(FINAL_TIME // REDUCED_DT)
INNER_STEPS = int(FINAL_TIME // HF_DT) // OUTER_STEPS
DOMAIN = ((0, 2), (0, 1))

# Ensemble configuration
NUM_ENSEMBLE = 4
INLET_ANGLES = [
    i for i in range(0, NUM_ENSEMBLE)
]  # Different inlet angles for each ensemble member
NUM_WORKERS = 2  # Number of parallel processes

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
print(DEVICE)


def main() -> None:
    """Main function."""

    configs = [
        DynamicsModelConfig(
            inlet_velocity_angle=angle,
            domain=DOMAIN,
            nx=NX,
            ny=NY,
            density=DENSITY,
            viscosity=VISCOSITY,
            obstacle_centers=OBSTACLE_CENTERS,
            obstacle_halfwidths=OBSTACLE_HALFWIDTHS,
            dt=HF_DT,
            num_inner_steps=INNER_STEPS,
            dtype=dtype,
        )
        for angle in INLET_ANGLES
    ]

    # Initialize ensemble model with different inlet angles
    ensemble_model = EnsembleDynamicsModel(
        configs=configs,
        num_workers=NUM_WORKERS,
        device=DEVICE,
    )

    # Initialize velocity fields for each ensemble member

    v_list = ensemble_model.get_initial_conditions()

    # Storage for trajectories
    trajectory_plots = torch.zeros(NUM_ENSEMBLE, 2, NX, NY, OUTER_STEPS)
    vorticity_plots = torch.zeros(NUM_ENSEMBLE, NX, NY, OUTER_STEPS)

    # Run ensemble simulation (sequentially to avoid memory issues)
    with torch.no_grad():
        for i in tqdm(range(OUTER_STEPS), desc="Simulating ensemble"):
            # Run all ensemble members sequentially
            v_list, p_list = ensemble_model(v_list)
            print(f"Completed step {i} of {OUTER_STEPS}")

            for j, v in enumerate(v_list):
                trajectory_plots[j, :, :, :, i] = get_tensor_from_grid_variables(v)
                vorticity_plots[j, :, :, i] = get_vorticity_from_grid_variables(v)

            # # Update velocity fields and store results
            # for j, (v_new, _) in enumerate(results):
            #     v_list[j] = v_new
            #     trajectory_plots[j, :, 0, :, :, i] = (
            #         v_new[0].data.detach().cpu().clone()
            #     )
            #     trajectory_plots[j, :, 1, :, :, i] = (
            #         v_new[1].data.detach().cpu().clone()
            #     )
            #     vorticity_plots[j, :, :, :, i] = vorticity(v_new)

    # torch.save(trajectory_plot, f"trajectory_ensemble_{j}_angle_{INLET_ANGLES[j]}.pt")

    ensemble_model.shutdown()

    vel_mag = [
        torch.sqrt(
            trajectory_plots[j, 0, :, :, :] ** 2 + trajectory_plots[j, 1, :, :, :] ** 2
        )
        for j in range(NUM_ENSEMBLE)
    ]

    figure_dir = "figures/torch_cfd"
    os.makedirs(figure_dir, exist_ok=True)

    create_animation_from_tensors(
        vel_mag,
        fps=10,
        file_name=f"{figure_dir}/velocity_magnitude_ensemble.mp4",
        colormaps="viridis",
        titles=[f"Angle: {INLET_ANGLES[j]}°" for j in range(NUM_ENSEMBLE)],
        normalize=False,
    )
    create_animation_from_tensors(
        [vorticity_plots[j, :, :, :] for j in range(NUM_ENSEMBLE)],
        fps=10,
        file_name=f"{figure_dir}/vorticity_trajectory_ensemble.mp4",
        colormaps="viridis",
        titles=[f"Angle: {INLET_ANGLES[j]}°" for j in range(NUM_ENSEMBLE)],
        vmin=-30,
        vmax=30,
        normalize=False,
    )


if __name__ == "__main__":
    main()
