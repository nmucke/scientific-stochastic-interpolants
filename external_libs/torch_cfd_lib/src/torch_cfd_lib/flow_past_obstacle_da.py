"""
Data assimilation script for flow past obstacle using Ensemble Kalman Filter.

This script loads a true trajectory from trajectory.pt and performs data assimilation
using the EnKF implementation for torch_cfd.
"""

import os
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_cfd import grids
from torch_cfd.grids import GridVariable
from torch_cfd.initial_conditions import velocity_field
from tqdm import tqdm

from torch_cfd_lib.boundary_conditions import (
    get_inlet_velocities_from_angle,
    karman_vortex_multiple_squares_boundary_conditions,
)
from torch_cfd_lib.ENKF import ObservationOperator, StateParameterEnKF

# Import parameters from flow_past_obstacle
from torch_cfd_lib.flow_past_obstacle import (
    BATCH_SIZE,
    DENSITY,
    DEVICE,
    DOMAIN,
    HF_DT,
    NX,
    NY,
    VISCOSITY,
)
from torch_cfd_lib.forward_model import DynamicsModel


def get_grid_observation_indices(
    data_size: tuple[int, int], skip_grid: int
) -> torch.Tensor:
    """Get grid observation indices in (x, y) format.

    Parameters
    ----------
    data_size : tuple
        (nx, ny) grid dimensions
    skip_grid : int
        Skip every skip_grid-th point

    Returns
    -------
    torch.Tensor
        obs_indices: (n_obs, 2) array with [[x1, y1], [x2, y2], ...] format
    """
    nx, ny = data_size

    # Generate grid indices for every skip_grid-th point
    x_coords = torch.arange(0, nx, skip_grid, dtype=torch.long)
    y_coords = torch.arange(0, ny, skip_grid, dtype=torch.long)

    # Create meshgrid and stack into (n_obs, 2) format
    X_grid, Y_grid = torch.meshgrid(x_coords, y_coords, indexing="ij")
    obs_indices = torch.stack([X_grid.ravel(), Y_grid.ravel()], dim=1)

    return obs_indices


def velocity_from_array(
    u_data: torch.Tensor,
    v_data: torch.Tensor,
    grid: grids.Grid,
    velocity_bc: Any,
    batch_size: int = 1,
) -> Tuple[GridVariable, GridVariable]:
    """Create velocity GridVariables from arrays.

    Parameters
    ----------
    u_data : torch.Tensor
        U velocity component data (nx, ny)
    v_data : torch.Tensor
        V velocity component data (nx, ny)
    grid : Grid
        Grid object
    velocity_bc : Tuple[BoundaryConditions, BoundaryConditions]
        Velocity boundary conditions (u_bc, v_bc)
    batch_size : int
        Batch size (default: 1)

    Returns
    -------
    Tuple[GridVariable, GridVariable]
        (u, v) velocity GridVariables
    """
    # Get offset from a sample velocity field
    sample_vel = velocity_field(
        (lambda x, y: torch.ones_like(x), lambda x, y: torch.zeros_like(x)),
        grid,
        velocity_bc=velocity_bc,
        batch_size=batch_size,
        random_state=0,
        noise=0.0,
        device=u_data.device,
    )
    u_offset = sample_vel[0].offset
    v_offset = sample_vel[1].offset

    # Extract u and v boundary conditions from the tuple
    u_bc, v_bc = velocity_bc

    # Ensure data has batch dimension
    if u_data.ndim == 2:
        u_data = u_data.unsqueeze(0)
        v_data = v_data.unsqueeze(0)

    u_gv = GridVariable(data=u_data, offset=u_offset, grid=grid, bc=u_bc)
    v_gv = GridVariable(data=v_data, offset=v_offset, grid=grid, bc=v_bc)

    return (u_gv, v_gv)


# Time window for data assimilation
START_TIME = 100
END_TIME = 103


def main() -> None:
    """Main data assimilation function."""
    # Set up parameters (should match flow_past_obstacle.py)
    dtype = torch.float32

    # EnKF parameters
    ensemble_size = 15
    skip_grid = 40  # Observation spacing
    obs_noise_std = 0.05
    n_timesteps = END_TIME - START_TIME  # Number of assimilation steps
    inflation = 1.01
    parameter_noise_std = 0.1  # For parameter estimation

    # True inflow angle (what we're trying to estimate)
    true_inflow_angle = -20.0  # degrees

    # Initialize dynamics model with true parameters
    print("Initializing true dynamics model...")
    true_dynamics_model = DynamicsModel(
        inlet_velocity_angle=true_inflow_angle,
        nx=NX,
        ny=NY,
        density=DENSITY,
        viscosity=VISCOSITY,
        domain=DOMAIN,
        dt=HF_DT,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        dtype=dtype,
    )

    # Set up grid
    grid = grids.Grid((NX, NY), domain=DOMAIN, device=DEVICE)

    # Set up observations
    obs_indices = get_grid_observation_indices((NX, NY), skip_grid)
    n_obs = obs_indices.shape[0] * 2  # Both u and v components
    print(f"Number of observation points: {obs_indices.shape[0]}")
    print(f"Total observations (u + v): {n_obs}")

    observation_operator = ObservationOperator(
        obs_indices=obs_indices.to(DEVICE),
        obs_components=["u", "v"],  # Observe both components
    )

    # Load or generate true trajectory
    trajectory_file = "trajectory.pt"
    trajectory = torch.load(trajectory_file, map_location=DEVICE)
    # Extract time window
    trajectory = trajectory[:, :, :, :, START_TIME : END_TIME + 1]

    # Extract true trajectories (use batch 0)
    # trajectory shape: [batch, channels, nx, ny, time_steps]
    # We want: [time_steps, channels, nx, ny]
    true_traj = trajectory[0].permute(3, 0, 1, 2)  # [time, channels, nx, ny]
    n_timesteps = min(n_timesteps, true_traj.shape[0] - 1)

    print(f"True trajectory shape: {true_traj.shape}")
    print(f"Running {n_timesteps} assimilation steps")

    # Generate observations from true trajectory
    print("Generating observations...")
    observations = []
    for t in range(n_timesteps + 1):
        # Extract velocity at time t: [channels, nx, ny]
        velocity_t = true_traj[t]
        obs = observation_operator(velocity_t)
        # Add noise
        obs_noisy = obs + torch.randn_like(obs) * obs_noise_std
        observations.append(obs_noisy)

    # Initialize ensemble from true initial condition with perturbations
    print("Initializing ensemble...")
    u_init_true = true_traj[0, 0]  # [nx, ny]
    v_init_true = true_traj[0, 1]  # [nx, ny]

    ensemble_velocity = []
    ensemble_parameters = []

    # Initial parameter ensemble (centered around a biased initial guess)
    initial_param_guess = 0.0  # Wrong initial guess
    for i in range(ensemble_size):
        # Add perturbations to initial condition
        # u_init = u_init_true + torch.randn_like(u_init_true) * 0.1
        # v_init = v_init_true + torch.randn_like(v_init_true) * 0.1

        # Initialize parameters with spread around initial guess
        param = torch.tensor(
            [initial_param_guess + torch.randn(1).item() * 20.0],
            device=DEVICE,
            dtype=dtype,
        )
        ensemble_parameters.append(param)

        x_inlet_velocity, y_inlet_velocity = get_inlet_velocities_from_angle(param)

        velocity_bc, _ = karman_vortex_multiple_squares_boundary_conditions(
            grid,
            inlet_velocity=(
                true_dynamics_model.x_inlet_velocity,
                true_dynamics_model.y_inlet_velocity,
            ),
            square_centers=true_dynamics_model.obstacle_centers,
            square_halfwidths=true_dynamics_model.obstacle_halfwidths,
            periodic_y=True,
        )
        x_velocity_fn = lambda x, y: x_inlet_velocity * torch.ones_like(x)
        y_velocity_fn = lambda x, y: y_inlet_velocity * torch.ones_like(x)
        v = velocity_field(
            (x_velocity_fn, y_velocity_fn),
            grid,
            velocity_bc=velocity_bc,
            batch_size=BATCH_SIZE,
            random_state=42,
            noise=0.1,
            device=DEVICE,
        )

        # velocity = velocity_from_array(
        #     v[0].data.squeeze(), v[1].data.squeeze(), grid, velocity_bc, batch_size=1
        # )
        ensemble_velocity.append(v)

    ensemble_parameters = torch.stack(ensemble_parameters, dim=0)  # [ensemble_size, 1]

    initial_parameters = ensemble_parameters.clone()  # type: ignore[attr-defined]
    print(
        f"Initial parameter ensemble mean: {ensemble_parameters.mean().item():.2f} degrees"  # type: ignore[attr-defined]
    )
    print(f"True parameter: {true_inflow_angle:.2f} degrees")

    # Initialize EnKF with parameter estimation
    print("Initializing StateParameterEnKF...")
    enkf = StateParameterEnKF(
        grid_shape=(NX, NY),
        ensemble_size=ensemble_size,
        model_noise_std=0.0,  # No model noise for now
        obs_noise_std=obs_noise_std,
        parameter_noise_std=parameter_noise_std,
        observation_operator=observation_operator,
        device=DEVICE,
        dtype=dtype,
    )

    # Run data assimilation
    print("Running data assimilation with parameter estimation...")
    ensemble_means_u = []
    ensemble_means_v = []
    parameter_means = [ensemble_parameters.mean().item()]  # type: ignore[attr-defined]
    parameter_stds = [ensemble_parameters.std().item()]  # type: ignore[attr-defined]
    rmse_history = []
    param_error_history = [abs(ensemble_parameters.mean().item() - true_inflow_angle)]  # type: ignore[attr-defined]

    for t in tqdm(range(n_timesteps), desc="Assimilation"):
        # Get observations for this time step
        obs = observations[t + 1]

        # Assimilate with parameters
        (
            ensemble_velocity,
            ensemble_parameters,
            forecast_ensemble,
            forecast_parameters,
        ) = enkf.assimilate_with_parameters(
            ensemble_velocity=ensemble_velocity,
            ensemble_parameters=ensemble_parameters,
            observations=obs,
            dynamics_model=true_dynamics_model,
            inflation=inflation,
        )

        true_dynamics_model = DynamicsModel(
            inlet_velocity_angle=true_inflow_angle,
            nx=NX,
            ny=NY,
            density=DENSITY,
            viscosity=VISCOSITY,
            domain=DOMAIN,
            dt=HF_DT,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            dtype=dtype,
        )

        # Compute ensemble mean
        u_mean_data = torch.stack(
            [v[0].data.squeeze() for v in ensemble_velocity]
        ).mean(dim=0)
        v_mean_data = torch.stack(
            [v[1].data.squeeze() for v in ensemble_velocity]
        ).mean(dim=0)

        ensemble_means_u.append(u_mean_data.detach().cpu())
        ensemble_means_v.append(v_mean_data.detach().cpu())

        # Track parameter estimates
        param_mean = ensemble_parameters.mean().item()  # type: ignore[attr-defined]
        param_std = ensemble_parameters.std().item()  # type: ignore[attr-defined]
        parameter_means.append(param_mean)
        parameter_stds.append(param_std)

        # Compute RMSE for state
        u_true = true_traj[t + 1, 0]
        v_true = true_traj[t + 1, 1]

        rmse_u = torch.sqrt(torch.mean((u_mean_data - u_true) ** 2)).item()
        rmse_v = torch.sqrt(torch.mean((v_mean_data - v_true) ** 2)).item()
        rmse = np.sqrt((rmse_u**2 + rmse_v**2) / 2)

        rmse_history.append(rmse)

        # Parameter error
        param_error = abs(param_mean - true_inflow_angle)
        param_error_history.append(param_error)

        if (t + 1) % 1 == 0:
            print(
                f"Step {t+1}/{n_timesteps}, RMSE: {rmse:.6f}, "
                f"Param: {param_mean:.2f}±{param_std:.2f}° (True: {true_inflow_angle:.2f}°), "
                f"Param Error: {param_error:.2f}°"
            )

    # # Save results
    # print("Saving results...")
    # ensemble_means_u = torch.stack(ensemble_means_u)  # (n_timesteps, nx, ny)
    # ensemble_means_v = torch.stack(ensemble_means_v)  # (n_timesteps, nx, ny)

    # results = {
    #     "ensemble_mean_u": ensemble_means_u,
    #     "ensemble_mean_v": ensemble_means_v,
    #     "true_u": true_traj[1:, 0].cpu(),
    #     "true_v": true_traj[1:, 1].cpu(),
    #     "rmse_history": np.array(rmse_history),
    #     "parameter_means": np.array(parameter_means),
    #     "parameter_stds": np.array(parameter_stds),
    #     "parameter_error_history": np.array(param_error_history),
    #     "true_parameter": true_inflow_angle,
    #     "obs_indices": obs_indices.cpu(),
    # }

    # torch.save(results, "da_results.pt")
    # print("Results saved to da_results.pt")

    # Create output directory
    os.makedirs("figures", exist_ok=True)

    # Plot RMSE, parameter evolution, and parameter distribution
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # RMSE plot
    ax1.plot(rmse_history)
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("RMSE")
    ax1.set_title("State Estimation RMSE")
    ax1.grid(True)

    # Parameter evolution plot
    ax2.plot(parameter_means, label="Estimated", linewidth=2)
    ax2.fill_between(
        range(len(parameter_means)),
        np.array(parameter_means) - np.array(parameter_stds),
        np.array(parameter_means) + np.array(parameter_stds),
        alpha=0.3,
    )
    ax2.axhline(true_inflow_angle, color="r", linestyle="--", label="True", linewidth=2)
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Inlet angle (degrees)")
    ax2.set_title("Parameter Estimation Evolution")
    ax2.legend()
    ax2.grid(True)

    # Parameter distribution histogram (final time step)
    final_params = ensemble_parameters.detach().cpu().numpy().flatten()  # type: ignore[attr-defined]
    initial_params = initial_parameters.detach().cpu().numpy().flatten()
    ax3.hist(
        initial_params,
        bins=15,
        alpha=0.3,
        color="orange",
        edgecolor="black",
        density=True,
        label="Initial Ensemble",
    )
    ax3.hist(
        final_params,
        bins=15,
        alpha=0.5,
        color="blue",
        edgecolor="black",
        density=True,
        label="Final Ensemble",
    )
    ax3.axvline(
        true_inflow_angle,
        color="r",
        linestyle="--",
        linewidth=2.5,
        label=f"True ({true_inflow_angle:.1f}°)",
    )
    ax3.axvline(
        parameter_means[-1],
        color="green",
        linestyle="-",
        linewidth=2,
        label=f"Mean ({parameter_means[-1]:.1f}°)",
    )
    ax3.set_xlabel("Inlet angle (degrees)")
    ax3.set_ylabel("Probability density")
    ax3.set_title("Final Parameter Distribution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/da_diagnostics.png", dpi=150)
    print("Diagnostics plot saved to figures/da_diagnostics.png")

    # Visualize some snapshots
    print("Creating visualization snapshots...")
    times_to_plot = [0, n_timesteps // 2, n_timesteps - 1]
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Extract observation indices for plotting
    obs_x = obs_indices[:, 0].cpu().numpy()
    obs_y = obs_indices[:, 1].cpu().numpy()

    for i, t_idx in enumerate(times_to_plot):
        if t_idx >= len(ensemble_means_u):
            continue

        # True solution
        ax = axes[0, i]
        im = ax.imshow(
            true_traj[t_idx + 1, 0].detach().cpu().numpy().T,
            origin="lower",
            cmap="viridis",
        )
        if i == 0:
            ax.scatter(
                obs_x, obs_y, c="red", s=10, alpha=0.5, marker="o", label="Observations"
            )
        ax.set_title(f"True u (t={t_idx+1})")
        if i == 2:  # Add legend to the last plot in the row
            ax.legend(loc="upper right", fontsize=8)
        plt.colorbar(im, ax=ax)

        # EnKF estimate
        ax = axes[1, i]
        im = ax.imshow(
            ensemble_means_u[t_idx].detach().cpu().numpy().T,
            origin="lower",
            cmap="viridis",
        )
        if i == 0:
            ax.scatter(
                obs_x, obs_y, c="red", s=10, alpha=0.5, marker="o", label="Observations"
            )
        ax.set_title(f"EnKF u (t={t_idx+1})")
        if i == 2:  # Add legend to the last plot in the row
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"EnKF u (t={t_idx+1})")
        plt.colorbar(im, ax=ax)

        # Error
        ax = axes[2, i]
        error = (ensemble_means_u[t_idx] - true_traj[t_idx + 1, 0].cpu()).abs()
        im = ax.imshow(error.detach().cpu().numpy().T, origin="lower", cmap="Reds")
        ax.scatter(obs_x, obs_y, c="red", s=10, alpha=0.5, marker="o")
        ax.set_title(f"Error u (t={t_idx+1})")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("figures/da_snapshots.png", dpi=150)
    print("Snapshots saved to figures/da_snapshots.png")

    print("\n" + "=" * 60)
    print("Data assimilation complete!")
    print(
        f"Final parameter estimate: {parameter_means[-1]:.2f}±{parameter_stds[-1]:.2f}°"
    )
    print(f"True parameter: {true_inflow_angle:.2f}°")
    print(f"Final parameter error: {param_error_history[-1]:.2f}°")
    print(f"Final RMSE: {rmse_history[-1]:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
