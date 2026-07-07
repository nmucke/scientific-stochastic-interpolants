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
from scisi.models.diffusion_model import DenoiseDiffusionModel
from scisi.models.flow_matching_model import FlowMatchingModel
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
from scisi.plotting.animation import create_animation_from_tensors
from scisi.plotting.plot_fields import plot_fields
from scisi.plotting.plot_point_distributions import plot_point_distributions
from scisi.plotting.spectrum import plot_enstrophy_spectrum
from scisi.sampling.ode_solvers import euler_step
from scisi.sampling.sde_solvers import euler_maruyama_step, heun_step
from scisi.utils.device_utils import set_device

torch.set_default_dtype(torch.float32)

logger = logging.getLogger(__name__)

SHOW_PLOTS = False
VERBOSE = True
MIXED_PRECISION = True

# DEFAULT_PROJECT = "stochastic_navier_stokes"
# DEFAULT_NAME = "diffusion_model"  # Diffusion model UNet Navier-Stokes
# DEFAULT_NAME = "flow_matching"  # Flow Matching model UNet Navier-Stokes
# DEFAULT_NAME = "stochastic_interpolant_small"  # Quadratic SI UNet Navier-Stokes
# DEFAULT_NAME = "stochastic_interpolant"  # Quadratic SI UNet Navier-Stokes

DEFAULT_PROJECT = "udales"
# DEFAULT_NAME = "kind-sky-8" # Udales
# DEFAULT_NAME = "flow_matching_big" # Udales
# DEFAULT_NAME = "flow_matching_small" # Udales
# DEFAULT_NAME = "flow_matching_big" # Udales
# DEFAULT_NAME = "stochastic_interpolant_small_gamma1"  # Udales
DEFAULT_NAME = "stochastic_interpolant_big_gamma1"  # Udales
# DEFAULT_NAME = "stochastic_interpolant_small_original"  # Udales


# When testing a diffusion model, build it from the (better-trained) flow-matching
# model via DenoiseDiffusionModel.from_flow_matching instead of loading the
# diffusion checkpoint's own weights. FM_NAME_FOR_DIFFUSION is the FM run name.
DIFFUSION_FROM_FM = True
FM_NAME_FOR_DIFFUSION = "flow_matching"

NUM_PHYSICAL_STEPS = 100
NUM_STEPS = 50
BATCH_SIZE = 5
PLOTTING_TIMES = [5, NUM_PHYSICAL_STEPS // 2, NUM_PHYSICAL_STEPS - 1]
TEST_SAMPLE_INDEX = 0
SDE_STEPPER = heun_step
# SDE_STEPPER = euler_maruyama_step
ODE_STEPPER = euler_step

mixed_precision_context = (
    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if MIXED_PRECISION
    else contextlib.nullcontext()
)


def _build_diffusion_from_fm(project: str, fm_name: str) -> DenoiseDiffusionModel:
    """Build a diffusion prior from a trained flow-matching checkpoint.

    Loads the FM run's own ``config.yaml`` / ``model.pth`` and wraps the trained
    velocity net via ``DenoiseDiffusionModel.from_flow_matching`` (velocity
    mode), so the score / reverse-SDE drift are reconstructed from the FM
    velocity rather than the diffusion checkpoint's weights.
    """
    fm_cfg = OmegaConf.load(f"checkpoints/{project}/{fm_name}/config.yaml")
    fm_model = hydra.utils.instantiate(fm_cfg.model)
    fm_model.load_state_dict(
        torch.load(f"checkpoints/{project}/{fm_name}/model.pth", map_location="cpu")
    )
    return DenoiseDiffusionModel.from_flow_matching(fm_model)


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
    if isinstance(model, DenoiseDiffusionModel) and DIFFUSION_FROM_FM:
        # Build the diffusion model from the well-trained FM model rather than the
        # diffusion checkpoint's own (worse) weights.
        logger.info(
            f"Building diffusion model from FM checkpoint "
            f"'{FM_NAME_FOR_DIFFUSION}' (velocity mode)..."
        )
        model = _build_diffusion_from_fm(project, FM_NAME_FOR_DIFFUSION)
    else:
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
            **init_data,
            batch_size=BATCH_SIZE,
            num_steps=NUM_STEPS,
            num_physical_steps=NUM_PHYSICAL_STEPS,
            stepper=(
                ODE_STEPPER if isinstance(model, FlowMatchingModel) else SDE_STEPPER
            ),
            # diffusion_term=lambda t: 2 * model.interpolation.gamma(t),
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

    if project == "udales":
        true_vel_magnitude = torch.sqrt(
            true_trajectory[0, 0] ** 2
            + true_trajectory[0, 1] ** 2
            + true_trajectory[0, 2] ** 2
        )
        predicted_vel_magnitude = torch.sqrt(
            predicted_trajectory[:, 0] ** 2
            + predicted_trajectory[:, 1] ** 2
            + predicted_trajectory[:, 2] ** 2
        )
    # else:
    #     true_vel_magnitude = torch.sqrt(
    #         true_trajectory[0, 0] ** 2 + true_trajectory[0, 1] ** 2
    #     )
    #     predicted_vel_magnitude = torch.sqrt(
    #         predicted_trajectory[:, 0] ** 2 + predicted_trajectory[:, 1] ** 2
    #     )

    logger.info(f"Creating animation...")
    if project == "udales":
        for i, file_name in enumerate(
            [
                "velocity_x",
                "velocity_y",
                "velocity_z",
                "temperature",
                "velocity_magnitude",
            ]
        ):
            if file_name != "velocity_magnitude":
                plot_list = [true_trajectory[0, i, :, :, 0:NUM_PHYSICAL_STEPS]] + [
                    predicted_trajectory[k, i] for k in range(BATCH_SIZE)
                ]
            else:
                plot_list = [true_vel_magnitude[:, :, 0:NUM_PHYSICAL_STEPS]] + [
                    predicted_vel_magnitude[k, :, :, 0:NUM_PHYSICAL_STEPS]
                    for k in range(BATCH_SIZE)
                ]
            create_animation_from_tensors(
                plot_list,
                fps=10,
                file_name=f"{figure_path}/{file_name}.mp4",
                colormaps="viridis",
                titles=["True"] + [f"Ensemble member {i}" for i in range(BATCH_SIZE)],
                vmin=np.min(plot_list[0].numpy()),
                vmax=np.max(plot_list[0].numpy()),
                normalize=False,
            )

    else:
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

    #### Plot velocity magnitude ####
    logger.info(f"Plotting trajectory...")
    if project == "udales":
        plot_fields(
            fields=[
                [true_vel_magnitude[:, :, t] for t in PLOTTING_TIMES],
                [predicted_vel_magnitude[0, :, :, t] for t in PLOTTING_TIMES],
            ],
            titles=[
                [f"True Velocity Magnitude at t={t}" for t in PLOTTING_TIMES],
                [f"Predicted Velocity Magnitude at t={t}" for t in PLOTTING_TIMES],
            ],
            vmin=np.min(true_vel_magnitude.numpy()),
            vmax=np.max(true_vel_magnitude.numpy()),
            figsize=(15, 10),
            figure_path=f"{figure_path}/predicted_trajectory.png",
            show=SHOW_PLOTS,
        )

        #### Plot distribution at points ####
        logger.info(f"Plotting velocity magnitude distribution at points...")
        points = [(32, 32), (64, 64), (96, 96)]
        plot_point_distributions(
            true_field=true_vel_magnitude,
            predicted_fields=predicted_vel_magnitude,
            points=points,
            figure_path=f"{figure_path}/distribution_at_points.png",
            show=SHOW_PLOTS,
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

    if project == "stochastic_navier_stokes":
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
