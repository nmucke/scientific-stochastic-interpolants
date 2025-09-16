import logging
import pdb

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from scisi.preprocessing.preprocessor import Preprocesser
from scisi.sampling.sde_solvers import euler_maruyama_step, heun_step

torch.set_default_dtype(torch.float32)

logger = logging.getLogger(__name__)

VERBOSE = True
MIXED_PRECISION = True

DEFAULT_PROJECT = "knmi"
DEFAULT_NAME = "fancy-breeze-4"
NUM_PHYSICAL_STEPS = 50
NUM_STEPS = 50
STARTING_TIME = 20000


@hydra.main(  # type: ignore[misc]
    config_path="../../../checkpoints",
    config_name=f"{DEFAULT_PROJECT}/{DEFAULT_NAME}/config.yaml",
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
    sample = test_dataset[0]
    trajectory = sample["x"].unsqueeze(0)
    trajectory = trajectory[
        :, :, :, :, STARTING_TIME : STARTING_TIME + NUM_PHYSICAL_STEPS
    ]

    x = trajectory[:, :, :, :, len_field_history - 1]
    x_history = trajectory[:, :, :, :, 0:len_field_history]

    field_cond = sample["field_cond"].unsqueeze(0)
    field_cond = field_cond[
        :, :, :, :, STARTING_TIME : STARTING_TIME + NUM_PHYSICAL_STEPS
    ]
    pars_cond = sample["pars_cond"].unsqueeze(0)
    pars_cond = pars_cond[:, STARTING_TIME : STARTING_TIME + NUM_PHYSICAL_STEPS]

    field_cond = field_cond[:, :, :, :, len_field_history:NUM_PHYSICAL_STEPS]
    pars_cond = pars_cond[:, len_field_history:NUM_PHYSICAL_STEPS]

    del sample

    logger.info(f"Preprocessing trajectory...")
    x = preprocesser.transform(base=x, is_batch=True)["base"]
    x_history = preprocesser.transform(field_history=x_history, is_batch=True)[
        "field_history"
    ]
    field_cond = preprocesser.transform(
        field_cond=field_cond, is_batch=True, is_trajectory=True
    )["field_cond"]

    x = x.to("cuda")
    x_history = x_history.to("cuda")
    field_cond = field_cond.to("cuda")
    pars_cond = pars_cond.to("cuda")

    logger.info(f"Sampling from the model...")
    input_dict = {
        "base": x,
        "batch_size": 1,
        "num_steps": NUM_STEPS,
        "field_history": x_history,
        "field_cond": field_cond,
        "pars_cond": pars_cond,
        "num_physical_steps": NUM_PHYSICAL_STEPS,
        "sde_stepper": heun_step,
        "diffusion_term": lambda t: 2.0 * model.interpolation.gamma(t),
    }
    if MIXED_PRECISION:
        logger.info(f"Sampling from the model using mixed precision...")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predicted_trajectory = model.sample_trajectory(**input_dict)
    else:
        logger.info(f"Sampling from the model using full precision...")
        predicted_trajectory = model.sample_trajectory(**input_dict)

    predicted_trajectory = predicted_trajectory.cpu()

    logger.info(f"Inverse transforming predicted trajectory...")
    predicted_trajectory = preprocesser.inverse_transform(
        base=predicted_trajectory, is_batch=True, is_trajectory=True
    )["base"].numpy()

    predicted_trajectory = predicted_trajectory[0, 0]
    true_trajectory = trajectory[0, 0, :, :].cpu().numpy()

    # Set indices for plotting
    lat_1_idx, lon_1_idx = 32, 0
    lat_30_idx, lon_30_idx = 42, 10
    lat_lon_to_plot = [
        (lat[lat_1_idx].item(), lon[lon_1_idx].item()),
        (lat[lat_30_idx].item(), lon[lon_30_idx].item()),
    ]
    true_grid_cell_to_plot_1 = true_trajectory[lat_1_idx, lon_1_idx, :]
    predicted_grid_cell_to_plot_1 = predicted_trajectory[lat_1_idx, lon_1_idx, :]
    true_grid_cell_to_plot_30 = true_trajectory[lat_30_idx, lon_30_idx, :]
    predicted_grid_cell_to_plot_30 = predicted_trajectory[lat_30_idx, lon_30_idx, :]

    logger.info(f"Plotting trajectory...")
    plotting_times = [10, 25, 45]
    num_plot_times = len(plotting_times)
    plt.figure()
    for i, t in enumerate(plotting_times):
        plt.subplot(3, num_plot_times, i + 1)
        plt.imshow(
            true_trajectory[:, :, t],
            vmin=np.min(true_trajectory),
            vmax=np.max(true_trajectory),
        )
        plt.scatter(lat_1_idx, lon_1_idx, color="red", marker="x", s=100)
        plt.scatter(lat_30_idx, lon_30_idx, color="red", marker="x", s=100)
        plt.colorbar()
        plt.title(f"True Trajectory at t={t}")
        plt.subplot(3, num_plot_times, num_plot_times + 1 + i)
        plt.imshow(
            predicted_trajectory[:, :, t],
            vmin=np.min(true_trajectory),
            vmax=np.max(true_trajectory),
        )
        plt.scatter(lat_1_idx, lon_1_idx, color="red", marker="x", s=100)
        plt.scatter(lat_30_idx, lon_30_idx, color="red", marker="x", s=100)
        plt.colorbar()
        plt.title(f"Predicted Trajectory at t={t}")

    plt.subplot(3, num_plot_times, 2 * num_plot_times + 1)
    plt.plot(true_grid_cell_to_plot_1, label="True", linewidth=3)
    plt.plot(predicted_grid_cell_to_plot_1, label="Predicted", linewidth=3)
    plt.legend()
    plt.grid(True)
    plt.title(f"Lat, Lon = {lat_lon_to_plot[0][0]:.2f}, {lat_lon_to_plot[0][1]:.2f}")
    plt.subplot(3, num_plot_times, 2 * num_plot_times + 2)
    plt.plot(true_grid_cell_to_plot_30, label="True", linewidth=3)
    plt.plot(predicted_grid_cell_to_plot_30, label="Predicted", linewidth=3)
    plt.legend()
    plt.grid(True)
    plt.title(f"Lat, Lon = {lat_lon_to_plot[1][0]:.2f}, {lat_lon_to_plot[1][1]:.2f}")

    plt.show()


if __name__ == "__main__":
    main()
