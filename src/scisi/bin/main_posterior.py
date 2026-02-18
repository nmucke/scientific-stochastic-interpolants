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

from scisi.likelihood_models.observation_operators import LinearObservationOperator
from scisi.metrics.lsim import LSiM_distance
from scisi.metrics.spectral import compute_enstrophy_error
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
from scisi.plotting.animation import create_animation_from_tensors
from scisi.plotting.spectrum import plot_enstrophy_spectrum
from scisi.posterior_models.flow_matching_posterior import FlowMatchingPosterior
from scisi.posterior_models.stochastic_interpolant_posterior import (
    StochasticInterpolantPosterior,
)
from scisi.sampling.ode_solvers import euler_step
from scisi.sampling.sde_solvers import euler_maruyama_step, heun_step

COLORS = [
    "tab:green",
    "tab:blue",
    "tab:red",
    "tab:purple",
    "tab:orange",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]
LINESTYLES = ["--", ":", "-."]

logger = logging.getLogger(__name__)

torch.set_default_dtype(torch.float32)

torch.manual_seed(42)

NUM_PHYSICAL_STEPS = 25
NUM_STEPS = 150
MIXED_PRECISION = False
ENSEMBLE_SIZE = 2
SDE_STEPPER = euler_maruyama_step
ODE_STEPPER = euler_step
TEST_SAMPLE_INDEX = 0
DIFFUSION_MULTIPLIER = 2

mixed_precision_context = (
    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if MIXED_PRECISION
    else contextlib.nullcontext()
)


@hydra.main(  # type: ignore[misc]
    config_path="../../../config",
    # config_name=f"weather_posterior.yaml",
    # config_name=f"stochastic_navier_stokes_posterior.yaml",
    # config_name=f"udales_posterior.yaml",
    config_name=f"udales_flow_matching_posterior.yaml",
    # config_name=f"diffusion_stochastic_navier_stokes_posterior.yaml",
    # config_name=f"flow_matching_stochastic_navier_stokes_posterior.yaml",
    version_base=None,
)
def main(posterior_cfg: DictConfig) -> None:
    project = posterior_cfg.pre_trained_model.project
    name = posterior_cfg.pre_trained_model.name

    cfg = OmegaConf.load(f"checkpoints/{project}/{name}/config.yaml")
    logger.info(f"Loading config from checkpoint:")
    logger.info(f"Project: {project}")
    logger.info(f"Name: {name}")

    try:
        len_field_history = cfg.model.drift_model.len_field_history
    except:
        len_field_history = cfg.model.denoise_model.len_field_history

    logger.info(f"Instantiating preprocesser...")
    preprocesser = hydra.utils.instantiate(cfg.preprocesser)

    logger.info(f"Instantiating test data...")
    test_dataset = hydra.utils.instantiate(cfg.test_data)
    trajectory = test_dataset[TEST_SAMPLE_INDEX]["x"].unsqueeze(0)

    # trajectory = np.load("trajectory.npz")["trajectory"][100:, ::2, ::2]
    # trajectory = trajectory.transpose(1, 2, 0)
    # trajectory = trajectory.reshape(1, 1, 128, 128, 100)
    # trajectory = torch.from_numpy(trajectory).float()

    # logger.info(f"Preprocessing trajectory...")
    # init_data = preprocesser.transform(
    #     base=trajectory,
    #     field_history=trajectory[:, :, :, :, 0:len_field_history],
    #     is_batch=True,
    #     is_trajectory=True,
    # )
    # trajectory = init_data["base"]
    # base = init_data["base"][:, :, :, :, len_field_history - 1]
    # field_history = init_data["field_history"]

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

    trajectory = preprocesser.transform(
        base=trajectory,
        is_batch=True,
        is_trajectory=True,
    )["base"]

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
        data_size=init_data["base"][0].shape,
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
    if (
        posterior_cfg.likelihood_model._target_
        == "scisi.likelihood_models.gaussian_likelihood.InterpolantGaussianLikelihood"
    ):
        # diffusion_term = lambda t: DIFFUSION_MULTIPLIER * model.interpolation.gamma(t)
        # diffusion_term = lambda t: 2.0 * torch.sqrt(model.interpolation.gamma(t))
        diffusion_term = lambda t: 2.0 * model.interpolation.gamma(t)
    else:
        diffusion_term = None

    posterior_model = hydra.utils.instantiate(
        posterior_cfg.posterior_model,
        model=model,
        likelihood_model=likelihood_model,
        diffusion_term=diffusion_term,
    )

    logger.info(f"Preparing observations...")
    observations = torch.zeros(1, obs_operator.num_obs, NUM_PHYSICAL_STEPS)
    for i in range(NUM_PHYSICAL_STEPS):
        observations[:, :, i] = obs_operator(trajectory[:, :, :, :, i].to("cuda")).cpu()
        observations[:, :, i] += torch.randn_like(observations[:, :, i]) * torch.sqrt(
            torch.tensor(posterior_cfg.likelihood_model.variance)
        )

    input_dict = {
        "base": (
            init_data["base"]
            if isinstance(model, FollmerStochasticInterpolant)
            else None
        ),
        "ensemble_size": ENSEMBLE_SIZE,
        "num_steps": NUM_STEPS,
        "field_history": init_data["field_history"],
        "field_cond": init_data["field_cond"],
        "pars_cond": init_data["pars_cond"],
        "stepper": (
            ODE_STEPPER
            if isinstance(posterior_model, FlowMatchingPosterior)
            else SDE_STEPPER
        ),
        "num_physical_steps": NUM_PHYSICAL_STEPS,
        "observations": observations[:, :, len_field_history:],
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
        input_dict.pop("ensemble_size")
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

    logger.info(f"Computing metrics...")
    metrics: dict[str, dict[str, list[float]]] = {
        title: {
            "LSiM": [],
            "RMSE": [],
            "Enstrophy error": [],
        }
        for title in ["Posterior", "Prior"]
    }
    time_range_ids = range(len_field_history, NUM_PHYSICAL_STEPS)
    for pred_trajectory, title in zip(
        [posterior_trajectory, prior_trajectory], ["Posterior", "Prior"]
    ):
        logger.info(f"================================================")
        logger.info(f"{title} metrics:")
        get_true_and_pred = lambda i: (
            true_trajectory[:, :, i],
            pred_trajectory[:, :, i],
        )
        metrics[title]["LSiM"] = [
            LSiM_distance(*get_true_and_pred(i)).item() for i in time_range_ids
        ]
        metrics[title]["RMSE"] = [
            torch.sqrt(nn.MSELoss()(*get_true_and_pred(i))).item()
            for i in time_range_ids
        ]

        logger.info(
            f"LSiM: {np.mean(metrics[title]['LSiM']):.4f} ± {np.std(metrics[title]['LSiM']):.4f}"
        )
        logger.info(
            f"RMSE: {np.mean(metrics[title]['RMSE']):.4f} ± {np.std(metrics[title]['RMSE']):.4f}"
        )

        if project == "stochastic_navier_stokes":
            ens_error, ens_error_array = compute_enstrophy_error(
                true_trajectory[:, :, len_field_history:],
                pred_trajectory[:, :, len_field_history:],
                dx=2 * torch.pi / 128,
            )
            logger.info(
                f"Enstrophy error: {ens_error:.4f} ± {ens_error_array.std():.4f}"
            )
            metrics[title]["Enstrophy error"] = ens_error_array

    if project == "stochastic_navier_stokes":
        plot_enstrophy_spectrum(
            trajectories=[true_trajectory, posterior_trajectory, prior_trajectory],
            titles=["True", "Posterior", "Prior"],
            figure_path=figure_path,
        )

    logger.info(f"Plotting results...")
    plt.figure(figsize=(15, 10))
    for i, (title, state) in enumerate(
        zip(["True", "Posterior", "Prior"], [true_state, posterior_state, prior_state])
    ):
        plt.subplot(2, 4, i + 1)
        plt.imshow(state, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(title)

        plt.subplot(2, 4, i + 5)
        plt.imshow(np.abs(state.numpy() - true_state.numpy()))
        plt.colorbar()
        plt.title(f"{title} Error")

    plt.subplot(2, 4, 4)
    plt.imshow(obs_indices_on_grid[0] * true_state, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("True (Observed)")

    plt.subplot(2, 4, 5)
    metrics_str_list = ["RMSE", "LSiM"]
    if project == "stochastic_navier_stokes":
        metrics_str_list.append("Enstrophy error")
    plot_settings = {"linewidth": 3, "markersize": 10}
    for i, title in enumerate(["Posterior", "Prior"]):
        for j, metric in enumerate(metrics_str_list):
            plot_settings["color"] = COLORS[i]  # type: ignore[assignment]
            plot_settings["linestyle"] = LINESTYLES[j]  # type: ignore[assignment]
            plt.plot(
                time_range_ids,
                metrics[title][metric],
                label=f"{title} {metric}",
                **plot_settings,
            )

    plt.ylim(0, 2)
    plt.grid(True)
    plt.legend()
    plt.title("Metrics")
    plt.savefig(f"{figure_path}/posterior_trajectory.png")
    plt.show()


if __name__ == "__main__":
    main()
