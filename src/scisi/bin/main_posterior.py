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

logger = logging.getLogger(__name__)

torch.set_default_dtype(torch.float32)

torch.manual_seed(42)

NUM_PHYSICAL_STEPS = 40
NUM_STEPS = 500
MIXED_PRECISION = False
BATCH_SIZE = 1
SDE_STEPPER = euler_maruyama_step
TEST_SAMPLE_INDEX = 0
DIFFUSION_MULTIPLIER = 1.0

mixed_precision_context = (
    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if MIXED_PRECISION
    else contextlib.nullcontext()
)


@hydra.main(  # type: ignore[misc]
    config_path="../../../config",
    # config_name=f"weather_posterior.yaml",
    config_name=f"stochastic_navier_stokes_posterior.yaml",
    version_base=None,
)
def main(posterior_cfg: DictConfig) -> None:
    project = posterior_cfg.pre_trained_model.project
    name = posterior_cfg.pre_trained_model.name

    cfg = OmegaConf.load(f"checkpoints/{project}/{name}/config.yaml")
    logger.info(f"Loading config from checkpoint:")
    logger.info(f"Project: {project}")
    logger.info(f"Name: {name}")

    len_field_history = cfg.model.drift_model.len_field_history

    logger.info(f"Instantiating preprocesser...")
    preprocesser = hydra.utils.instantiate(cfg.preprocesser)

    logger.info(f"Instantiating test data...")
    test_dataset = hydra.utils.instantiate(cfg.test_data)
    trajectory = test_dataset[TEST_SAMPLE_INDEX]["x"].unsqueeze(0)

    logger.info(f"Preprocessing trajectory...")
    init_data = preprocesser.transform(
        base=trajectory,
        field_history=trajectory[:, :, :, :, 0:len_field_history],
        is_batch=True,
        is_trajectory=True,
    )
    trajectory = init_data["base"].to("cuda")
    base = init_data["base"][:, :, :, :, len_field_history - 1].to("cuda")
    field_history = init_data["field_history"].to("cuda")

    logger.info(f"Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)

    logger.info(f"Loading model from checkpoint...")
    model.load_state_dict(torch.load(f"checkpoints/{project}/{name}/model.pth"))
    model.eval()
    model.to("cuda")

    logger.info(f"Instantiating observation operator:")
    logger.info(f"Type: {posterior_cfg.obs_operator.type}")
    obs_operator = hydra.utils.instantiate(
        posterior_cfg.obs_operator,
        data_size=base[0].shape,
    )
    logger.info(f"Number of observations: {obs_operator.num_obs}")
    logger.info(
        f"{obs_operator.num_obs / obs_operator.num_dofs * 100}% of the data is observed"
    )

    logger.info(f"Instantiating likelihood model...")
    likelihood_model = hydra.utils.instantiate(
        posterior_cfg.likelihood_model,
        obs_operator=obs_operator,
        model=model,
    )

    logger.info(f"Instantiating posterior model...")
    posterior_model = hydra.utils.instantiate(
        posterior_cfg.posterior_model,
        model=model,
        likelihood_model=likelihood_model,
        diffusion_term=lambda t: DIFFUSION_MULTIPLIER * model.interpolation.gamma(t),
    )

    logger.info(f"Preparing observations...")
    observations = torch.zeros(1, obs_operator.num_obs, NUM_PHYSICAL_STEPS)
    for i in range(NUM_PHYSICAL_STEPS):
        observations[:, :, i] = obs_operator(trajectory[:, :, :, :, i])
        observations[:, :, i] += torch.randn_like(observations[:, :, i]) * torch.sqrt(
            torch.tensor(posterior_cfg.likelihood_model.variance)
        )

    input_dict = {
        "base": base,
        "batch_size": BATCH_SIZE,
        "num_steps": NUM_STEPS,
        "field_history": field_history,
        "sde_stepper": SDE_STEPPER,
        "num_physical_steps": NUM_PHYSICAL_STEPS,
        "observations": observations[:, :, len_field_history:].to("cuda"),
    }

    logger.info(
        f"Sampling using mixed precision..."
        if MIXED_PRECISION
        else f"Sampling using full precision..."
    )
    with mixed_precision_context:
        logger.info(f"Sampling from the posterior model...")
        posterior_trajectory = posterior_model.sample_trajectory(**input_dict)

        input_dict.pop("num_steps")
        input_dict.pop("observations")
        logger.info(f"Sampling from the prior model...")
        prior_trajectory = model.sample_trajectory(**input_dict, num_steps=50)

    true_trajectory = trajectory.to("cpu")

    logger.info(f"Inverse transforming predicted trajectory...")
    posterior_trajectory = preprocesser.inverse_transform(
        base=posterior_trajectory, is_batch=True, is_trajectory=True
    )["base"]
    prior_trajectory = preprocesser.inverse_transform(
        base=prior_trajectory, is_batch=True, is_trajectory=True
    )["base"]
    true_trajectory = preprocesser.inverse_transform(
        base=true_trajectory, is_batch=True, is_trajectory=True
    )["base"]

    posterior_trajectory = posterior_trajectory[0, 0]
    prior_trajectory = prior_trajectory[0, 0]
    true_trajectory = true_trajectory[0, 0]

    logger.info(f"Computing RMSE...")
    rmse_post = [
        torch.sqrt(
            nn.MSELoss()(posterior_trajectory[:, :, i], true_trajectory[:, :, i])
        ).item()
        for i in range(len_field_history, NUM_PHYSICAL_STEPS)
    ]
    rmse_prior = [
        torch.sqrt(
            nn.MSELoss()(prior_trajectory[:, :, i], true_trajectory[:, :, i])
        ).item()
        for i in range(len_field_history, NUM_PHYSICAL_STEPS)
    ]
    rmse_prior = np.array(rmse_prior)
    rmse_post = np.array(rmse_post)
    logger.info(f"RMSE of posterior: {np.mean(rmse_post):.6f}")
    logger.info(f"RMSE of prior: {np.mean(rmse_prior):.6f}")

    true_state = true_trajectory[:, :, NUM_PHYSICAL_STEPS - 1]
    posterior_state = posterior_trajectory[:, :, NUM_PHYSICAL_STEPS - 1]
    prior_state = prior_trajectory[:, :, NUM_PHYSICAL_STEPS - 1]

    figure_path = f"figures/{project}"
    os.makedirs(figure_path, exist_ok=True)

    vmin = np.min(true_state.numpy())
    vmax = np.max(true_state.numpy())

    logger.info(f"Creating animation...")
    create_animation_from_tensors(
        [
            true_trajectory[:, :, 0:NUM_PHYSICAL_STEPS],
            posterior_trajectory,
            prior_trajectory,
        ],
        fps=10,
        file_name=f"{figure_path}/posterior_trajectory.mp4",
        colormaps="viridis",
        titles=["True", "Posterior", "Prior"],
        normalize=False,
        vmin=vmin,
        vmax=vmax,
    )

    obs_indices = obs_operator.obs_indices_c_h_w
    obs_indices_on_grid = obs_operator.obs_indices_on_grid

    logger.info(f"Plotting results...")
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 4, 1)
    plt.imshow(true_state, vmin=vmin, vmax=vmax)
    # plt.scatter(obs_indices[:, 2], obs_indices[:, 1], color="red", marker=".", s=5)
    plt.colorbar()
    plt.title("True")
    plt.subplot(2, 4, 2)
    plt.imshow(obs_indices_on_grid[0] * true_state, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("True (Observed)")
    plt.subplot(2, 4, 3)
    plt.imshow(posterior_state, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("Posterior")
    plt.subplot(2, 4, 4)
    plt.imshow(prior_state, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("Prior")
    plt.subplot(2, 4, 5)
    plt.plot(
        range(len_field_history, NUM_PHYSICAL_STEPS),
        rmse_post,
        label="Posterior RMSE",
        linewidth=3,
        markersize=10,
    )
    plt.plot(
        range(len_field_history, NUM_PHYSICAL_STEPS),
        rmse_prior,
        label="Prior RMSE",
        linewidth=3,
        markersize=10,
    )
    plt.grid(True)
    plt.legend()
    plt.title("RMSE")
    plt.subplot(2, 4, 6)
    plt.imshow(np.abs(posterior_state.numpy() - true_state.numpy()))
    plt.colorbar()
    plt.title("Posterior Error")
    plt.subplot(2, 4, 7)
    plt.imshow(np.abs(prior_state.numpy() - true_state.numpy()))
    plt.colorbar()
    plt.title("Prior Error")
    plt.savefig(f"{figure_path}/posterior_trajectory.png")
    plt.show()


if __name__ == "__main__":
    main()
