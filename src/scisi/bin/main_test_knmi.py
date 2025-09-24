import contextlib
import logging
import os
import pdb

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from scisi.plotting.animation import create_animation_from_tensors
from scisi.sampling.sde_solvers import euler_maruyama_step, heun_step

torch.set_default_dtype(torch.float32)

logger = logging.getLogger(__name__)

MIXED_PRECISION = False
BATCH_SIZE = 3
NAME = "jolly-valley-7"
NUM_PHYSICAL_STEPS = 75
NUM_STEPS = 50
STARTING_TIME = 20000

END_TIME = STARTING_TIME + NUM_PHYSICAL_STEPS
# SDE_STEPPER = heun_step
SDE_STEPPER = euler_maruyama_step

mixed_precision_context = (
    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if MIXED_PRECISION
    else contextlib.nullcontext()
)


@hydra.main(  # type: ignore[misc]
    config_path="../../../checkpoints",
    config_name=f"knmi/{NAME}/config.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    project = list(cfg.keys())[0]
    name = list(cfg[project].keys())[0]
    cfg = OmegaConf.select(cfg, f"{project}.{name}")

    len_field_history = cfg.model.drift_model.len_field_history

    logger.info(f"Instantiating preprocesser...")
    preprocesser = hydra.utils.instantiate(cfg.preprocesser)

    logger.info(f"Instantiating test data...")
    test_dataset = hydra.utils.instantiate(cfg.test_data)
    lat, lon = test_dataset.lat, test_dataset.lon

    logger.info(f"Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)

    logger.info(f"Loading model from checkpoint...")
    model.load_state_dict(torch.load(f"checkpoints/{project}/{name}/model.pth"))
    model.eval()
    model.to("cuda")

    logger.info(f"Preparing trajectory...")
    sample = test_dataset[0:1]
    trajectory = sample["x"]
    field_cond = sample["field_cond"]
    pars_cond = sample["pars_cond"]
    del sample

    trajectory = trajectory[..., STARTING_TIME:END_TIME]

    logger.info(f"Preprocessing trajectory...")
    processed_data = preprocesser.transform(
        base=trajectory[..., len_field_history - 1 : len_field_history],
        field_history=trajectory[..., 0:len_field_history],
        field_cond=field_cond[..., STARTING_TIME + len_field_history : END_TIME],
        is_batch=True,
        is_trajectory=True,
    )

    input_dict = {
        "base": processed_data["base"].squeeze(-1).to("cuda"),
        "batch_size": BATCH_SIZE,
        "num_steps": NUM_STEPS,
        "field_history": processed_data["field_history"].to("cuda"),
        "field_cond": processed_data["field_cond"].to("cuda"),
        "pars_cond": pars_cond[:, STARTING_TIME + len_field_history : END_TIME].to(
            "cuda"
        ),
        "num_physical_steps": NUM_PHYSICAL_STEPS,
        "sde_stepper": SDE_STEPPER,
        # "diffusion_term": lambda t: 2.0 * model.interpolation.gamma(t),
    }

    # Use mixed precision if available
    logger.info(
        f"Sampling from the model using mixed precision..."
        if MIXED_PRECISION
        else f"Sampling from the model using full precision..."
    )
    with mixed_precision_context:
        predicted_trajectory = model.sample_trajectory(**input_dict)

    predicted_trajectory = predicted_trajectory.cpu()

    logger.info(f"Inverse transforming predicted trajectory...")
    predicted_trajectory = preprocesser.inverse_transform(
        base=predicted_trajectory, is_batch=True, is_trajectory=True
    )["base"]

    predicted_trajectory = predicted_trajectory[:, 0]
    true_trajectory = trajectory[0, 0].cpu()

    # Set indices for plotting
    lat_lon_idx_to_plot = [(32, 64), (42, 10), (2, 120)]
    lat_lon_to_plot = [
        (lat[lat_lon_idx[0]], lon[lat_lon_idx[1]])
        for lat_lon_idx in lat_lon_idx_to_plot
    ]

    true_grid_cells_to_plot = [
        true_trajectory[lat_lon_idx[0], lat_lon_idx[1], :]
        for lat_lon_idx in lat_lon_idx_to_plot
    ]

    predicted_grid_cells_to_plot = [
        predicted_trajectory[:, lat_lon_idx[0], lat_lon_idx[1], :]
        for lat_lon_idx in lat_lon_idx_to_plot
    ]

    figure_path = f"figures/{project}"
    os.makedirs(figure_path, exist_ok=True)

    logger.info(f"Creating animation...")
    create_animation_from_tensors(
        [true_trajectory] + [predicted_trajectory[i] for i in range(BATCH_SIZE)],
        fps=10,
        file_name=f"{figure_path}/predicted_trajectory.mp4",
        colormaps="viridis",
        titles=["True"] + [f"Emseble member {i}" for i in range(BATCH_SIZE)],
        vmin=np.min(true_trajectory.numpy()),
        vmax=np.max(true_trajectory.numpy()),
        normalize=False,
    )

    logger.info(f"Plotting trajectory...")
    plotting_times = [10, NUM_PHYSICAL_STEPS // 2, NUM_PHYSICAL_STEPS - 1]
    num_plot_times = len(plotting_times)
    plt.figure(figsize=(25, 20))
    for i, t in enumerate(plotting_times):
        plt.subplot(3, num_plot_times, i + 1)
        plt.imshow(
            true_trajectory[:, :, t],
            vmin=np.min(true_trajectory.numpy()),
            vmax=np.max(true_trajectory.numpy()),
        )
        plt.colorbar()
        for lat_lon in lat_lon_idx_to_plot:
            plt.scatter(lat_lon[1], lat_lon[0], color="red", marker="x", s=100)
        plt.title(f"True trajectory at t={t}")
        plt.subplot(3, num_plot_times, num_plot_times + 1 + i)
        plt.imshow(
            predicted_trajectory[0, :, :, t],
            vmin=np.min(true_trajectory.numpy()),
            vmax=np.max(true_trajectory.numpy()),
        )
        plt.colorbar()
        for lat_lon in lat_lon_idx_to_plot:
            plt.scatter(lat_lon[1], lat_lon[0], color="tab:red", marker="x", s=100)
        plt.title(f"Ensemble member 1 trajectory at t={t}")

    plt.subplot(3, num_plot_times, 2 * num_plot_times + 1)
    for i in range(len(lat_lon_idx_to_plot)):
        plt.subplot(3, num_plot_times, 2 * num_plot_times + i + 1)
        plt.plot(
            true_grid_cells_to_plot[i], label="True", linewidth=4, color="tab:green"
        )
        plt.plot(
            predicted_grid_cells_to_plot[i][0],
            label="Emseble predictions",
            linewidth=1,
            color="tab:blue",
            alpha=0.5,
        )
        for j in range(BATCH_SIZE):
            plt.plot(
                predicted_grid_cells_to_plot[i][j],
                linewidth=3,
                color="tab:blue",
                alpha=0.5,
            )

        y_min = 0.975 * np.min(true_grid_cells_to_plot[i].numpy())
        y_max = 1.025 * np.max(true_grid_cells_to_plot[i].numpy())
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.grid(True)
        plt.title(
            f"Lat, Lon = {lat_lon_to_plot[i][0]:.2f}, {lat_lon_to_plot[i][1]:.2f}"
        )
    plt.savefig(f"{figure_path}/predicted_trajectory.png")
    plt.show()


if __name__ == "__main__":
    main()
