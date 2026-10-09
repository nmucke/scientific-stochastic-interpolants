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
from sympy.integrals.laplace import I

from scisi.metrics.lsim import LSiM_distance
from scisi.metrics.spectral import compute_enstrophy_error, get_enstrophy_spectrum
from scisi.plotting.animation import create_animation_from_tensors
from scisi.plotting.plot_fields import plot_fields
from scisi.plotting.plot_point_distributions import plot_point_distributions
from scisi.plotting.spectrum import plot_enstrophy_spectrum
from scisi.sampling.ode_solvers import euler_step
from scisi.sampling.sde_solvers import euler_maruyama_step, heun_step
from scisi.utils.device_utils import set_device
from scisi.models.follmer_ito_map_model import FollmerItoMapModel


torch.set_default_dtype(torch.float32)

logger = logging.getLogger(__name__)

SHOW_PLOTS = False
VERBOSE = True
MIXED_PRECISION = True

DEFAULT_PROJECT = "stochastic_navier_stokes"
DEFAULT_NAME = "follmer_ito_map"


NUM_PHYSICAL_STEPS = 100
NUM_STEPS = 5
BATCH_SIZE = 5
PLOTTING_TIMES = [5, NUM_PHYSICAL_STEPS // 2, NUM_PHYSICAL_STEPS - 1]
TEST_SAMPLE_INDEX = 0

mixed_precision_context = (
    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if MIXED_PRECISION
    else contextlib.nullcontext()
)


def main(cfg: DictConfig, project: str, name: str) -> None:
    """Main function."""
    logger.info(f"Model is {cfg.model._target_}...")

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
    try:
        field_cond = test_dataset[TEST_SAMPLE_INDEX]["field_cond"].unsqueeze(0)
    except:
        field_cond = None
    try:
        pars_cond = test_dataset[TEST_SAMPLE_INDEX]["pars_cond"].unsqueeze(0)
    except:
        pars_cond = None
    init_data = preprocesser.transform(
        base=trajectory[..., len_field_history - 1],
        field_history=trajectory[..., 0:len_field_history],
        is_batch=True,
    )
    init_data["field_cond"] = preprocesser.transform(
        field_cond=field_cond if field_cond is not None else None,
        is_batch=True,
        is_trajectory=True,
    )["field_cond"]
    init_data["pars_cond"] = preprocesser.transform(
        pars_cond=pars_cond if pars_cond is not None else None,
        is_batch=True,
        is_trajectory=True,
    )["pars_cond"]

    logger.info(f"Moving data to the correct device...")
    init_data = {
        k: v.to(cfg.trainer.device) for k, v in init_data.items() if v is not None
    }

    if not isinstance(model, FollmerItoMapModel):
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
            **init_data,
            batch_size=BATCH_SIZE,
            num_steps=NUM_STEPS,
            num_physical_steps=NUM_PHYSICAL_STEPS,
        )

    true_trajectory = trajectory.cpu()
    predicted_trajectory = predicted_trajectory.cpu()

    logger.info(f"Inverse transforming predicted trajectory...")
    predicted_trajectory = preprocesser.inverse_transform(
        base=predicted_trajectory,
        field_cond=field_cond,
        is_batch=True,
        is_trajectory=True,
    )["base"]

    figure_path = f"figures/{project}"
    os.makedirs(figure_path, exist_ok=True)

    logger.info(f"Creating animation...")
    
    create_animation_from_tensors(
        [true_trajectory[0, 0, :, :, 0:NUM_PHYSICAL_STEPS]]
        + [predicted_trajectory[i, 0] for i in range(BATCH_SIZE)],
        fps=10,
        file_name=f"{figure_path}/predicted_trajectory.mp4",
        colormaps="viridis",
        titles=["True"] + [f"Ensemble member {i}" for i in range(BATCH_SIZE)],
        vmin=np.min(true_trajectory.numpy()),
        vmax=np.max(true_trajectory.numpy()),
        normalize=False,
    )

    #### Compute metrics ####
    predicted_trajectory = predicted_trajectory[0, 0]
    true_trajectory = trajectory[0, 0].cpu()
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

    plot_enstrophy_spectrum(
        trajectories=[true_trajectory, predicted_trajectory],
        titles=["True", "Predicted"],
        figure_path=figure_path,
        show=SHOW_PLOTS,
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
