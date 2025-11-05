# """
# Data assimilation script for flow past obstacle using Ensemble Kalman Filter.

# This script loads a true trajectory from trajectory.pt and performs data assimilation
# using the EnKF implementation for torch_cfd.
# """

# import os
# from ast import Tuple
# from typing import Any

# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from torch_cfd import grids
# from torch_cfd.boundaries import ConstantBoundaryConditions, ImmersedBoundaryConditions
# from torch_cfd.grids import GridArray, GridVariable
# from torch_cfd.initial_conditions import velocity_field
# from tqdm import tqdm

# from scisi.external_libs.torch_cfd.ENKF import (
#     LocalizedPhysicalEnKF,
#     ObservationOperator,
#     PhysicalEnKF,
# )
# from scisi.external_libs.torch_cfd.flow_past_obstacle import (
#     BATCH_SIZE,
#     DENSITY,
#     DOMAIN,
#     HF_DT,
#     NX,
#     NY,
#     VISCOSITY,
#     DynamicsModel,
#     device,
# )


# def get_grid_observation_indices(
#     data_size: tuple[int, int], skip_grid: int
# ) -> torch.Tensor:
#     """Get grid observation indices in (x, y) format.

#     Parameters
#     ----------
#     data_size : tuple
#         (nx, ny) grid dimensions
#     skip_grid : int
#         Skip every skip_grid-th point

#     Returns
#     -------
#     torch.Tensor
#         obs_indices: (n_obs, 2) array with [[x1, y1], [x2, y2], ...] format
#     """
#     nx, ny = data_size

#     # Generate grid indices for every skip_grid-th point
#     x_coords = torch.arange(0, nx, skip_grid, dtype=torch.long)
#     y_coords = torch.arange(0, ny, skip_grid, dtype=torch.long)

#     # Create meshgrid and stack into (n_obs, 2) format
#     X_grid, Y_grid = torch.meshgrid(x_coords, y_coords, indexing="ij")
#     obs_indices = torch.stack([X_grid.ravel(), Y_grid.ravel()], dim=1)

#     return obs_indices


# def velocity_from_array(
#     u_data: torch.Tensor,
#     v_data: torch.Tensor,
#     grid: grids.Grid,
#     velocity_bc: Any,
# ) -> Tuple[GridVariable, GridVariable]:
#     """Create velocity GridVariables from arrays.

#     Parameters
#     ----------
#     u_data : torch.Tensor
#         U velocity component data
#     v_data : torch.Tensor
#         V velocity component data
#     grid : Grid
#         Grid object
#     velocity_bc : BoundaryConditions
#         Velocity boundary conditions

#     Returns
#     -------
#     Tuple[GridVariable, GridVariable]
#         (u, v) velocity GridVariables
#     """
#     # Get offset from a sample velocity field
#     from torch_cfd.initial_conditions import velocity_field

#     sample_vel = velocity_field(
#         (lambda x, y: torch.ones_like(x), lambda x, y: torch.zeros_like(x)),
#         grid,
#         velocity_bc=velocity_bc,
#         batch_size=1,
#         random_state=0,
#         noise=0.0,
#         device=u_data.device,
#     )
#     u_offset = sample_vel[0].offset
#     v_offset = sample_vel[1].offset

#     u_array = GridArray(u_data, u_offset, grid)
#     v_array = GridArray(v_data, v_offset, grid)
#     u_gv = GridVariable(u_array, velocity_bc)
#     v_gv = GridVariable(v_array, velocity_bc)

#     return (u_gv, v_gv)


# START_TIME = 10
# END_TIME = 25


# def main() -> None:
#     """Main data assimilation function."""
#     # Set up parameters (should match flow_past_obstacle.py)
#     dtype = torch.float32

#     # EnKF parameters
#     ensemble_size = 50
#     skip_grid = 16  # Observation spacing
#     obs_noise_std = 0.01
#     start_time = 0  # Starting time index in trajectory
#     n_timesteps = 100  # Number of assimilation steps
#     inflation = 1.02
#     localization_radius = 20.0

#     # Set inflow angle (can be constant or time-varying)
#     inflow_angle = 0.0  # degrees

#     # Initialize dynamics model
#     dynamics_model = DynamicsModel(
#         inlet_velocity_angle=inflow_angle,
#         nx=NX,
#         ny=NY,
#         density=DENSITY,
#         viscosity=VISCOSITY,
#         domain=DOMAIN,
#         dt=HF_DT,
#         device=device,
#         dtype=dtype,
#     )

#     # Set up grid
#     grid = grids.Grid((NX, NY), domain=DOMAIN, device=device)

#     # Set up observations
#     obs_indices = get_grid_observation_indices((NX, NY), skip_grid)
#     n_obs = obs_indices.shape[0]
#     print(f"Number of observations: {n_obs}")

#     observation_operator = ObservationOperator(
#         obs_indices=obs_indices.to(device),
#         obs_components=["u", "v"],  # Observe both components
#     )

#     # Load true trajectory
#     trajectory = torch.load("trajectory.pt", map_location=device)

#     trajectory = trajectory[START_TIME:END_TIME]

#     # Now trajectory should be (batch, 2, n_time, nx, ny)
#     true_u_traj = trajectory[0, 0]  # (n_time, nx, ny)
#     true_v_traj = trajectory[0, 1]  # (n_time, nx, ny)

#     observations = []
#     for t in range(n_timesteps + 1):
#         obs = observation_operator(trajectory[i])
#         obs = obs + torch.randn(obs.shape)
#         observations.append(obs)

#     # Initialize ensemble from true initial condition with perturbations
#     print("Initializing ensemble...")
#     u_init_true = true_u_traj[0]
#     v_init_true = true_v_traj[0]

#     ensemble_velocity = []
#     for i in range(ensemble_size):
#         # Add small perturbations to initial condition
#         u_init = u_init_true + torch.randn_like(u_init_true) * 0.1
#         v_init = v_init_true + torch.randn_like(v_init_true) * 0.1

#         velocity = velocity_from_array(u_init, v_init, grid, dynamics_model.velocity_bc)
#         ensemble_velocity.append(velocity)

#     # Initialize EnKF
#     print("Initializing EnKF...")
#     enkf = PhysicalEnKF(
#         grid_shape=(NX, NY),
#         ensemble_size=ensemble_size,
#         model_noise_std=0.0,  # No model noise for now
#         obs_noise_std=obs_noise_std,
#         observation_operator=observation_operator,
#         device=device,
#         dtype=dtype,
#     )

#     # Run data assimilation
#     print("Running data assimilation...")
#     ensemble_means_u = []
#     ensemble_means_v = []
#     rmse_history = []

#     for t in tqdm(range(n_timesteps), desc="Assimilation"):
#         # Get observations for this time step
#         obs = observations[t + 1]

#         # Create a dynamics function that works with EnKF (no extra parameters)
#         def dynamics_fn(velocity: tuple, dt: float) -> tuple:
#             """Wrapper for dynamics function."""
#             return dynamics(velocity, dt, inlet_angle=None)  # Use constant angle

#         # Assimilate
#         ensemble_velocity, forecast_ensemble = enkf.assimilate(
#             ensemble_velocity=ensemble_velocity,
#             observations=obs,
#             dynamics=dynamics_fn,
#             dt=HF_DT,
#             inflation=inflation,
#         )

#         # Compute ensemble mean
#         u_mean_data = torch.stack(
#             [v[0].data.squeeze() for v in ensemble_velocity]
#         ).mean(dim=0)
#         v_mean_data = torch.stack(
#             [v[1].data.squeeze() for v in ensemble_velocity]
#         ).mean(dim=0)

#         ensemble_means_u.append(u_mean_data)
#         ensemble_means_v.append(v_mean_data)

#         # Compute RMSE
#         u_true = true_u_traj[t + 1]
#         v_true = true_v_traj[t + 1]

#         rmse_u = torch.sqrt(torch.mean((u_mean_data - u_true) ** 2)).item()
#         rmse_v = torch.sqrt(torch.mean((v_mean_data - v_true) ** 2)).item()
#         rmse = np.sqrt((rmse_u**2 + rmse_v**2) / 2)

#         rmse_history.append(rmse)

#         if (t + 1) % 10 == 0:
#             print(
#                 f"Step {t+1}/{n_timesteps}, RMSE: {rmse:.6f} (u: {rmse_u:.6f}, v: {rmse_v:.6f})"
#             )

#     # Save results
#     print("Saving results...")
#     ensemble_means_u = torch.stack(ensemble_means_u)  # (n_timesteps, nx, ny)
#     ensemble_means_v = torch.stack(ensemble_means_v)  # (n_timesteps, nx, ny)

#     results = {
#         "ensemble_mean_u": ensemble_means_u.cpu(),
#         "ensemble_mean_v": ensemble_means_v.cpu(),
#         "true_u": true_u_traj[1:].cpu(),
#         "true_v": true_v_traj[1:].cpu(),
#         "rmse_history": np.array(rmse_history),
#         "obs_indices": obs_indices.cpu(),
#     }

#     torch.save(results, "da_results.pt")
#     print("Results saved to da_results.pt")

#     # Plot RMSE
#     plt.figure(figsize=(10, 6))
#     plt.plot(rmse_history)
#     plt.xlabel("Time step")
#     plt.ylabel("RMSE")
#     plt.title("Data Assimilation RMSE")
#     plt.grid(True)
#     plt.savefig("da_rmse.png")
#     print("RMSE plot saved to da_rmse.png")

#     # Visualize some snapshots
#     print("Creating visualization snapshots...")
#     times_to_plot = [0, n_timesteps // 2, n_timesteps - 1]
#     fig, axes = plt.subplots(3, 3, figsize=(15, 15))

#     for i, t_idx in enumerate(times_to_plot):
#         if t_idx >= len(ensemble_means_u):
#             continue

#         # True solution
#         ax = axes[0, i]
#         im = ax.imshow(true_u_traj[t_idx + 1].cpu().T, origin="lower", cmap="viridis")
#         ax.set_title(f"True u (t={t_idx+1})")
#         plt.colorbar(im, ax=ax)

#         # EnKF estimate
#         ax = axes[1, i]
#         im = ax.imshow(ensemble_means_u[t_idx].cpu().T, origin="lower", cmap="viridis")
#         ax.set_title(f"EnKF u (t={t_idx+1})")
#         plt.colorbar(im, ax=ax)

#         # Error
#         ax = axes[2, i]
#         error = (ensemble_means_u[t_idx] - true_u_traj[t_idx + 1]).abs().cpu()
#         im = ax.imshow(error.T, origin="lower", cmap="Reds")
#         ax.set_title(f"Error u (t={t_idx+1})")
#         plt.colorbar(im, ax=ax)

#     plt.tight_layout()
#     plt.savefig("da_snapshots.png", dpi=150)
#     print("Snapshots saved to da_snapshots.png")

#     print("Data assimilation complete!")


# if __name__ == "__main__":
#     main()
