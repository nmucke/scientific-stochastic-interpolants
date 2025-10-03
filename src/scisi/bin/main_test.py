import contextlib
import logging
import os
import pdb

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from scisi.plotting.animation import create_animation_from_tensors
from scisi.plotting.plot_fields import plot_fields
from scisi.sampling.sde_solvers import euler_maruyama_step, heun_step
from scisi.utils.device_utils import set_device

torch.set_default_dtype(torch.float32)

logger = logging.getLogger(__name__)

VERBOSE = True
MIXED_PRECISION = False

DEFAULT_PROJECT = "stochastic_navier_stokes"
DEFAULT_NAME = "brave-forest-1"  # PDE-Transformer Navier-Stokes
# DEFAULT_NAME = "warm-root-42"  # PDE-Transformer Navier-Stokes

# DEFAULT_PROJECT = "weather"
# DEFAULT_NAME = "dainty-sunset-0"  # PDE-Transformer Weather
NUM_PHYSICAL_STEPS = 25
NUM_STEPS = 100
BATCH_SIZE = 1
PLOTTING_TIMES = [5, NUM_PHYSICAL_STEPS // 2, NUM_PHYSICAL_STEPS - 1]
TEST_SAMPLE_INDEX = 5
SDE_STEPPER = euler_maruyama_step

mixed_precision_context = (
    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if MIXED_PRECISION
    else contextlib.nullcontext()
)


@hydra.main(  # type: ignore[misc]
    config_path="../../../checkpoints",
    config_name=f"{DEFAULT_PROJECT}/{DEFAULT_NAME}/config.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    project = list(cfg.keys())[0]
    name = list(cfg[project].keys())[0]
    cfg = OmegaConf.select(cfg, f"{project}.{name}")

    set_device(cfg)

    len_field_history = cfg.model.drift_model.len_field_history

    logger.info(f"Instantiating preprocesser...")
    preprocesser = hydra.utils.instantiate(cfg.preprocesser)

    logger.info(f"Instantiating test data...")
    test_dataset = hydra.utils.instantiate(cfg.test_data)

    logger.info(f"Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)

    logger.info(f"Loading model from checkpoint:")
    logger.info(f"Project: {project}")
    logger.info(f"Name: {name}")
    model.load_state_dict(torch.load(f"checkpoints/{project}/{name}/model.pth"))
    model.eval()
    model.to(cfg.trainer.device)

    logger.info(f"Preparing trajectory...")
    trajectory = test_dataset[TEST_SAMPLE_INDEX]["x"].unsqueeze(0)

    logger.info(f"Preprocessing trajectory...")
    init_data = preprocesser.transform(
        base=trajectory[:, :, :, :, len_field_history - 1],
        field_history=trajectory[:, :, :, :, 0:len_field_history],
        is_batch=True,
    )

    input_dict = {
        "base": init_data["base"].to(cfg.trainer.device),
        "batch_size": BATCH_SIZE,
        "num_steps": NUM_STEPS,
        "field_history": init_data["field_history"].to(cfg.trainer.device),
        "num_physical_steps": NUM_PHYSICAL_STEPS,
        "sde_stepper": SDE_STEPPER,
    }

    # Use mixed precision if available
    logger.info(
        f"Sampling from the model using mixed precision..."
        if MIXED_PRECISION
        else f"Sampling from the model using full precision..."
    )
    with mixed_precision_context:
        predicted_trajectory = model.sample_trajectory(**input_dict)

    true_trajectory = trajectory[0, 0].cpu()
    predicted_trajectory = predicted_trajectory.cpu()

    logger.info(f"Inverse transforming predicted trajectory...")
    predicted_trajectory = preprocesser.inverse_transform(
        base=predicted_trajectory, is_batch=True, is_trajectory=True
    )["base"]
    predicted_trajectory = predicted_trajectory[0, 0]

    figure_path = f"figures/{project}"
    os.makedirs(figure_path, exist_ok=True)

    logger.info(f"Creating animation...")
    create_animation_from_tensors(
        [true_trajectory[:, :, 0:NUM_PHYSICAL_STEPS], predicted_trajectory],
        fps=10,
        file_name=f"{figure_path}/predicted_trajectory.mp4",
        colormaps="viridis",
        titles=["True", "Predicted"],
        vmin=np.min(true_trajectory.numpy()),
        vmax=np.max(true_trajectory.numpy()),
        normalize=False,
    )

    logger.info(f"Plotting trajectory...")
    plot_fields(
        fields=[
            [true_trajectory[:, :, t] for t in PLOTTING_TIMES],
            [predicted_trajectory[:, :, t] for t in PLOTTING_TIMES],
        ],
        titles=[
            [f"True Trajectory at t={t}" for t in PLOTTING_TIMES],
            [f"Predicted Trajectory at t={t}" for t in PLOTTING_TIMES],
        ],
        vmin=np.min(true_trajectory.numpy()),
        vmax=np.max(true_trajectory.numpy()),
        figsize=(15, 10),
        figure_path=f"{figure_path}/predicted_trajectory.png",
    )


if __name__ == "__main__":
    main()
