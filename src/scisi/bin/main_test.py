import argparse
import contextlib
import logging
import os
import pdb

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from scisi.metrics.lsim import LSiM_distance
from scisi.metrics.spectral import compute_enstrophy_error, get_enstrophy_spectrum
from scisi.models.flow_matching_model import FlowMatchingModel
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
from scisi.plotting.animation import create_animation_from_tensors
from scisi.plotting.plot_fields import plot_fields
from scisi.plotting.spectrum import plot_enstrophy_spectrum
from scisi.sampling.ode_solvers import euler_step
from scisi.sampling.sde_solvers import euler_maruyama_step, heun_step
from scisi.utils.device_utils import set_device

torch.set_default_dtype(torch.float32)

logger = logging.getLogger(__name__)

VERBOSE = True
MIXED_PRECISION = True

DEFAULT_PROJECT = "stochastic_navier_stokes"
DEFAULT_NAME = "vivid-otter-65"  # SI UNet Navier-Stokes
# DEFAULT_NAME = "adventurous-acorn-45"
# DEFAULT_NAME = "brave-forest-1"  # SI PDE-Transformer Navier-Stokes
# DEFAULT_NAME = "warm-root-42"  # SI PDE-Transformer Navier-Stokes
# DEFAULT_NAME = "breezy-pine-46" # Flow Matching PDE-transformer Navier-Stokes
# DEFAULT_NAME = "cheerful-willow-47"  # Diffusion model PDE-transformer Navier-Stokes
# DEFAULT_NAME = "zany-lynx-67"  # Diffusion model UNet Navier-Stokes
# DEFAULT_NAME = "playful-fox-60"  # Flow Matching model UNet Navier-Stokes

# DEFAULT_PROJECT = "weather"
# DEFAULT_NAME = "dainty-sunset-0"  # PDE-Transformer Weather
# DEFAULT_NAME = "eager-mountain-3"  # PDE-Transformer Weather
NUM_PHYSICAL_STEPS = 20
NUM_STEPS = 500
BATCH_SIZE = 2
PLOTTING_TIMES = [5, NUM_PHYSICAL_STEPS // 2, NUM_PHYSICAL_STEPS - 1]
TEST_SAMPLE_INDEX = 3
SDE_STEPPER = euler_maruyama_step
ODE_STEPPER = euler_step

mixed_precision_context = (
    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if MIXED_PRECISION
    else contextlib.nullcontext()
)


def main(cfg: DictConfig, project: str, name: str) -> None:
    """Main function."""
    logger.info(f"Model is {cfg.model._target_}...")

    set_device(cfg)

    len_field_history = cfg.len_field_history

    logger.info(f"Instantiating preprocesser...")
    preprocesser = hydra.utils.instantiate(cfg.preprocesser)

    logger.info(f"Instantiating test data...")
    test_dataset = hydra.utils.instantiate(cfg.test_data)

    logger.info(f"Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)
    logger.info(f"Loading model from checkpoint...")
    model.load_state_dict(
        torch.load(f"checkpoints/{project}/{name}/model.pth", map_location="cpu")
    )
    model.eval()
    model.to(cfg.trainer.device)

    logger.info(f"Preparing and preprocessing trajectory...")
    trajectory = test_dataset[TEST_SAMPLE_INDEX]["x"].unsqueeze(0)
    init_data = preprocesser.transform(
        base=trajectory[:, :, :, :, len_field_history - 1],
        field_history=trajectory[:, :, :, :, 0:len_field_history],
        is_batch=True,
    )

    if not isinstance(model, FollmerStochasticInterpolant):
        logger.info(f"Model is a {type(model)}. Setting base to None...")
        init_data["base"] = None

    # Use mixed precision if available
    logger.info(
        f"Sampling from the model using mixed precision..."
        if MIXED_PRECISION
        else f"Sampling from the model using full precision..."
    )
    with mixed_precision_context:
        predicted_trajectory = model.sample_trajectory(
            base=(
                init_data["base"].to(cfg.trainer.device)
                if init_data["base"] is not None
                else None
            ),
            batch_size=BATCH_SIZE,
            num_steps=NUM_STEPS,
            field_history=init_data["field_history"].to(cfg.trainer.device),
            num_physical_steps=NUM_PHYSICAL_STEPS,
            stepper=(
                ODE_STEPPER if isinstance(model, FlowMatchingModel) else SDE_STEPPER
            ),
            # diffusion_term=lambda t: 2 * model.interpolation.gamma(t),
        )

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

    logger.info(f"Computing metrics...")
    lsim = [
        LSiM_distance(true_trajectory[:, :, i], predicted_trajectory[:, :, i])
        for i in range(NUM_PHYSICAL_STEPS)
    ]
    logger.info(f"LSiM: {np.mean(lsim):.4f} ± {np.std(lsim):.4f}")

    rmse = [
        torch.sqrt(
            nn.MSELoss()(true_trajectory[:, :, i], predicted_trajectory[:, :, i])
        )
        for i in range(NUM_PHYSICAL_STEPS)
    ]
    logger.info(f"RMSE: {np.mean(rmse):.4f} ± {np.std(rmse):.4f}")

    if project == "stochastic_navier_stokes":
        plot_enstrophy_spectrum(
            trajectories=[true_trajectory, predicted_trajectory],
            titles=["True", "Predicted"],
            figure_path=figure_path,
        )
        ens_error, ens_error_array = compute_enstrophy_error(
            true_trajectory, predicted_trajectory, 2 * torch.pi / 128
        )
        logger.info(f"Enstrophy error: {ens_error:.4f} ± {ens_error_array.std():.4f}")


if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Parse command line arguments for project and name
    parser = argparse.ArgumentParser(description="Test a trained model")
    parser.add_argument(
        "--project",
        type=str,
        default=DEFAULT_PROJECT,
        help="Project name (default: %(default)s)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_NAME,
        help="Model name (default: %(default)s)",
    )
    args, unknown = parser.parse_known_args()

    # Construct config directory path
    config_dir = os.path.abspath(f"checkpoints/{args.project}/{args.name}")

    logger.info(f"Loading config from: {config_dir}")
    logger.info(f"Project: {args.project}")
    logger.info(f"Name: {args.name}")

    # Initialize Hydra with the config directory
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # Compose the config, allowing overrides from command line
        cfg = compose(config_name="config", overrides=unknown)
        main(cfg, args.project, args.name)
