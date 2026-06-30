"""Quick smoke-test for the FM-derived diffusion prior.

Loads a trained *flow-matching* checkpoint, wraps it as a diffusion model via
``DenoiseDiffusionModel.from_flow_matching`` (velocity mode: the score and the
reverse-SDE drift are reconstructed from the FM velocity), samples a trajectory
with the SDE stepper, and writes the usual eyeball figures + RMSE / spectrum.

Run (from the repo root, with the venv python):

    .venv/bin/python -m scisi.bin.main_test_fm_as_diffusion
    .venv/bin/python -m scisi.bin.main_test_fm_as_diffusion --name flow_matching

It deliberately mirrors ``main_test.py`` but is trimmed to the Navier-Stokes
case and the single thing being checked: does the FM-as-diffusion sampler
produce sane trajectories?
"""

import argparse
import contextlib
import logging
import os

import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from scisi.metrics.spectral import compute_enstrophy_error
from scisi.models.diffusion_model import DenoiseDiffusionModel
from scisi.models.flow_matching_model import FlowMatchingModel
from scisi.plotting.animation import create_animation_from_tensors
from scisi.plotting.spectrum import plot_enstrophy_spectrum
from scisi.sampling.sde_solvers import euler_maruyama_step

torch.set_default_dtype(torch.float32)
logger = logging.getLogger(__name__)

SHOW_PLOTS = False
MIXED_PRECISION = True

DEFAULT_PROJECT = "stochastic_navier_stokes"
DEFAULT_NAME = "flow_matching"  # the well-trained FM prior

NUM_PHYSICAL_STEPS = 25
NUM_STEPS = 50
BATCH_SIZE = 5
TEST_SAMPLE_INDEX = 0
SDE_STEPPER = euler_maruyama_step

mixed_precision_context = (
    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if MIXED_PRECISION
    else contextlib.nullcontext()
)


def main(cfg: DictConfig, project: str, name: str) -> None:
    """Load the FM checkpoint, wrap as diffusion, sample, plot."""
    len_field_history = cfg.len_field_history

    logger.info("Instantiating preprocesser / test data...")
    preprocesser = hydra.utils.instantiate(cfg.preprocesser)
    test_dataset = hydra.utils.instantiate(cfg.test_data)

    logger.info("Instantiating flow-matching model...")
    fm_model = hydra.utils.instantiate(cfg.model)
    assert isinstance(fm_model, FlowMatchingModel), (
        f"Expected a FlowMatchingModel checkpoint, got {type(fm_model)}. "
        "Point --name at the trained flow-matching run."
    )
    fm_model.load_state_dict(
        torch.load(f"checkpoints/{project}/{name}/model.pth", map_location="cpu")
    )

    logger.info("Wrapping FM velocity as a diffusion prior (velocity mode)...")
    model = DenoiseDiffusionModel.from_flow_matching(fm_model)
    model.eval()
    model.to(cfg.trainer.device)

    logger.info("Preparing trajectory...")
    trajectory = test_dataset[TEST_SAMPLE_INDEX]["x"].unsqueeze(0)
    init_data = preprocesser.transform(
        base=trajectory[..., len_field_history - 1],
        field_history=trajectory[..., 0:len_field_history],
        is_batch=True,
    )
    init_data["field_cond"] = None
    init_data["pars_cond"] = None
    init_data = {
        k: v.to(cfg.trainer.device) for k, v in init_data.items() if v is not None
    }
    # Diffusion model samples from a Gaussian base (a0 = 0), so no SI base state.
    init_data["base"] = None

    logger.info("Sampling trajectory with the SDE stepper...")
    with mixed_precision_context:
        predicted_trajectory = model.sample_trajectory(
            **init_data,
            batch_size=BATCH_SIZE,
            num_steps=NUM_STEPS,
            num_physical_steps=NUM_PHYSICAL_STEPS,
            stepper=SDE_STEPPER,
        )

    true_trajectory = trajectory.cpu()
    predicted_trajectory = predicted_trajectory.cpu()
    predicted_trajectory = preprocesser.inverse_transform(
        base=predicted_trajectory,
        is_batch=True,
        is_trajectory=True,
    )["base"]

    figure_path = f"figures/{project}/fm_as_diffusion"
    os.makedirs(figure_path, exist_ok=True)

    logger.info("Creating animation...")
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

    #### Metrics + spectrum (member 0) ####
    pred0 = predicted_trajectory[0, 0]
    true0 = trajectory[0, 0].cpu()
    rmse = [
        torch.sqrt(nn.MSELoss()(true0[:, :, i], pred0[:, :, i]))
        for i in range(NUM_PHYSICAL_STEPS)
    ]
    logger.info(f"RMSE: {np.mean(rmse):.4f} ± {np.std(rmse):.4f}")

    plot_enstrophy_spectrum(
        trajectories=[true0, pred0],
        titles=["True", "FM-as-diffusion"],
        figure_path=figure_path,
        show=SHOW_PLOTS,
    )
    ens_error, ens_error_array = compute_enstrophy_error(
        true0, pred0, 2 * torch.pi / 128
    )
    logger.info(f"Enstrophy error: {ens_error:.4f} ± {ens_error_array.std():.4f}")
    logger.info(f"Figures written to {figure_path}/")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Test the FM-derived diffusion prior")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument("--name", type=str, default=DEFAULT_NAME)
    args, unknown = parser.parse_known_args()

    config_dir = os.path.abspath(f"checkpoints/{args.project}/{args.name}")
    logger.info(f"Loading config from: {config_dir}")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=unknown)
        main(cfg, args.project, args.name)
