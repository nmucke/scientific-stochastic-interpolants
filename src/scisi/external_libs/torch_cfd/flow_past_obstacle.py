import torch
from torch_cfd import grids
from torch_cfd.initial_conditions import velocity_field
from tqdm import tqdm

from scisi.external_libs.torch_cfd.forward_model import DynamicsModel, vorticity
from scisi.plotting.animation import create_animation_from_tensors

dtype = torch.float32

NX = 512
NY = 256
DENSITY = 1.0
HF_DT = 1e-3
REDUCED_DT = 1e-1
BATCH_SIZE = 1
# VISCOSITY = 1 / 500
VISCOSITY = 1 / 500
FINAL_TIME = 25.0
OUTER_STEPS = int(FINAL_TIME // REDUCED_DT)
INNER_STEPS = int(FINAL_TIME // HF_DT) // OUTER_STEPS
DOMAIN = ((0, 2), (0, 1))

# Define obstacle as a grid of 3 by 3
OBSTACLE_CENTERS = []
OBSTACLE_HALFWIDTHS = []

x_positions = [0.5, 1.0, 1.5]
y_positions = [0.25, 0.5, 0.75]
halfwidth = 0.05

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

    # inflow_angle_vec = [-45] + [45 for _ in range(OUTER_STEPS)]
    # inflow_angle_vec = [45 for _ in range(OUTER_STEPS//5)] + [-45 for _ in range(4*OUTER_STEPS//5 + 1)]
    inflow_angle_vec = [-20 for _ in range(OUTER_STEPS + 1)]

    model = DynamicsModel(
        inlet_velocity_angle=-20,
        nx=NX,
        ny=NY,
        density=DENSITY,
        viscosity=VISCOSITY,
        domain=DOMAIN,
        obstacle_centers=OBSTACLE_CENTERS,
        obstacle_halfwidths=OBSTACLE_HALFWIDTHS,
        dt=HF_DT,
        batch_size=BATCH_SIZE,
        num_inner_steps=INNER_STEPS,
    )

    x_velocity_fn = lambda x, y: model.x_inlet_velocity * torch.ones_like(x)
    y_velocity_fn = lambda x, y: model.y_inlet_velocity * torch.ones_like(x)
    # x_velocity_fn = lambda x, y: torch.ones_like(x)
    # y_velocity_fn = lambda x, y: torch.zeros_like(x)

    v = velocity_field(
        (x_velocity_fn, y_velocity_fn),
        grid,
        velocity_bc=model.velocity_bc,
        batch_size=BATCH_SIZE,
        random_state=42,
        noise=0.1,
        device=DEVICE,
    )

    trajectory = []
    trajectory_plot = torch.zeros(BATCH_SIZE, 2, NX, NY, OUTER_STEPS)
    vorticity_plot = torch.zeros(BATCH_SIZE, NX, NY, OUTER_STEPS)
    with torch.no_grad():
        for i in tqdm(range(OUTER_STEPS)):
            v, _ = model.forward_with_parameters(v, inflow_angle_vec[i])

            trajectory.append(v)

            trajectory_plot[:, 0, :, :, i] = v[0].data.detach().cpu().clone()
            trajectory_plot[:, 1, :, :, i] = v[1].data.detach().cpu().clone()
            vorticity_plot[:, :, :, i] = vorticity(v.clone())

    trajectory_plot = trajectory_plot[:, :, ::2, ::2, :]
    torch.save(trajectory_plot, "trajectory.pt")

    vel_mag = torch.sqrt(
        trajectory_plot[0, 0, :, :, :] ** 2 + trajectory_plot[0, 1, :, :, :] ** 2
    )

    create_animation_from_tensors(
        [10 * vel_mag],
        fps=10,
        file_name=f"figures/velocity_magnitude.mp4",
        colormaps="viridis",
        titles=["Velocity Magnitude"],
        # vmin=-1.5,
        # vmax=1.5,
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
