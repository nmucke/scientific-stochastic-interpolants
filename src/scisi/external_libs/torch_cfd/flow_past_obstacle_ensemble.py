import torch
from torch_cfd import grids
from torch_cfd.grids import GridVariable
from torch_cfd.initial_conditions import velocity_field
from tqdm import tqdm

from scisi.external_libs.torch_cfd.forward_model import EnsembleDynamicsModel, vorticity
from scisi.plotting.animation import create_animation_from_tensors

dtype = torch.float32

NX = 400
NY = 200
DENSITY = 1.0
HF_DT = 5e-4
REDUCED_DT = 1e-2
BATCH_SIZE = 1
# VISCOSITY = 1 / 500
VISCOSITY = 1 / 1000
FINAL_TIME = 5.0
OUTER_STEPS = int(FINAL_TIME // REDUCED_DT)
INNER_STEPS = int(FINAL_TIME // HF_DT) // OUTER_STEPS
DOMAIN = ((0, 2), (0, 1))

# Ensemble configuration
NUM_ENSEMBLE = 2
INLET_ANGLES = [-60, 0]  # Different inlet angles for each ensemble member
NUM_PROCESSES = 2  # Number of parallel processes

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

    grid = grids.Grid((NX, NY), domain=DOMAIN, device=DEVICE)

    # Initialize ensemble model with different inlet angles
    ensemble_model = EnsembleDynamicsModel(
        inlet_velocity_angles=INLET_ANGLES,  # type: ignore[arg-type]
        nx=NX,
        ny=NY,
        density=DENSITY,
        viscosity=VISCOSITY,
        domain=DOMAIN,
        obstacle_centers=OBSTACLE_CENTERS,  # type: ignore[arg-type]
        obstacle_halfwidths=OBSTACLE_HALFWIDTHS,  # type: ignore[arg-type]
        dt=HF_DT,
        batch_size=BATCH_SIZE,
        num_inner_steps=INNER_STEPS,
        num_processes=NUM_PROCESSES,
        device=DEVICE,
    )

    # Initialize velocity fields for each ensemble member

    v_list = []
    for i in range(NUM_ENSEMBLE):
        x_velocity_fn = lambda x, y: ensemble_model.models[
            i
        ].x_inlet_velocity * torch.ones_like(x)
        y_velocity_fn = lambda x, y: ensemble_model.models[
            i
        ].y_inlet_velocity * torch.ones_like(x)
        v = velocity_field(
            (x_velocity_fn, y_velocity_fn),
            grid,
            velocity_bc=ensemble_model.models[i].velocity_bc,
            batch_size=BATCH_SIZE,
            random_state=42 + i,  # Different random seed for each member
            noise=0.1,
            device=DEVICE,
        )
        v_list.append(v)

    # Storage for trajectories
    trajectory_plots = torch.zeros(NUM_ENSEMBLE, BATCH_SIZE, 2, NX, NY, OUTER_STEPS)
    vorticity_plots = torch.zeros(NUM_ENSEMBLE, BATCH_SIZE, NX, NY, OUTER_STEPS)

    # Run ensemble simulation (sequentially to avoid memory issues)
    with torch.no_grad():
        for i in tqdm(range(OUTER_STEPS), desc="Simulating ensemble"):
            # Run all ensemble members sequentially
            results = ensemble_model(v_list, parallel=False)

            # Update velocity fields and store results
            for j, (v_new, _) in enumerate(results):
                v_list[j] = v_new
                trajectory_plots[j, :, 0, :, :, i] = (
                    v_new[0].data.detach().cpu().clone()
                )
                trajectory_plots[j, :, 1, :, :, i] = (
                    v_new[1].data.detach().cpu().clone()
                )
                vorticity_plots[j, :, :, :, i] = vorticity(v_new)

    # torch.save(trajectory_plot, f"trajectory_ensemble_{j}_angle_{INLET_ANGLES[j]}.pt")

    vel_mag = [
        torch.sqrt(
            trajectory_plots[j, 0, 0, :, :, :] ** 2
            + trajectory_plots[j, 0, 1, :, :, :] ** 2
        )
        for j in range(NUM_ENSEMBLE)
    ]

    create_animation_from_tensors(
        vel_mag,
        fps=10,
        file_name=f"figures/velocity_magnitude_ensemble.mp4",
        colormaps="viridis",
        titles=[f"Angle: {INLET_ANGLES[j]}°" for j in range(NUM_ENSEMBLE)],
        normalize=False,
    )
    create_animation_from_tensors(
        [vorticity_plots[j, 0, :, :, :] for j in range(NUM_ENSEMBLE)],
        fps=10,
        file_name=f"figures/vorticity_trajectory_ensemble.mp4",
        colormaps="viridis",
        titles=[f"Angle: {INLET_ANGLES[j]}°" for j in range(NUM_ENSEMBLE)],
        vmin=-30,
        vmax=30,
        normalize=False,
    )


if __name__ == "__main__":
    main()
