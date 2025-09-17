import logging
import pdb

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import trackio
from omegaconf import DictConfig, OmegaConf

from scisi.likelihood_models.observation_operators import LinearObservationOperator
from scisi.preprocessing.preprocessor import Preprocesser
from scisi.sampling.sde_solvers import euler_maruyama_step, heun_step

logger = logging.getLogger(__name__)

torch.set_default_dtype(torch.float32)

torch.manual_seed(42)

VERBOSE = True
NUM_PHYSICAL_STEPS = 50
NUM_STEPS = 250


@hydra.main(  # type: ignore[misc]
    config_path="../../../config",
    config_name=f"stochastic_navier_stokes_posterior.yaml",
    version_base=None,
)
def main(posterior_cfg: DictConfig) -> None:
    project = posterior_cfg.project
    name = posterior_cfg.name

    cfg = OmegaConf.load(f"checkpoints/{project}/{name}/config.yaml")
    len_field_history = cfg.model.drift_model.len_field_history

    logger.info(f"Instantiating observation operator...")
    obs_operator = hydra.utils.instantiate(posterior_cfg.obs_operator)

    logger.info(f"Instantiating likelihood model...")
    likelihood_model = hydra.utils.instantiate(
        posterior_cfg.likelihood_model,
        obs_operator=obs_operator,
    )

    logger.info(f"Instantiating preprocesser...")
    preprocesser = hydra.utils.instantiate(cfg.preprocesser)

    logger.info(f"Instantiating test data...")
    test_dataset = hydra.utils.instantiate(cfg.test_data)
    trajectory = test_dataset[0]["x"].unsqueeze(0)

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

    logger.info(f"Instantiating posterior model...")
    posterior_model = hydra.utils.instantiate(
        posterior_cfg.posterior_model,
        model=model,
        likelihood_model=likelihood_model,
        diffusion_term=lambda t: 3.0 * model.interpolation.gamma(t),
    )

    logger.info(f"Preparing observations...")
    observations = torch.zeros(1, len(obs_operator.obs_indices), NUM_PHYSICAL_STEPS)
    for i in range(NUM_PHYSICAL_STEPS):
        observations[:, :, i] = obs_operator(trajectory[:, :, :, :, i])

    logger.info(f"Sampling from the posterior model...")
    x_post = posterior_model.sample(
        base=base,
        batch_size=1,
        num_steps=NUM_STEPS,
        field_history=field_history,
        observations=observations[:, :, len_field_history].to("cuda"),
        sde_stepper=heun_step,
    )

    logger.info(f"Sampling from the prior model...")
    x_prior = model.sample(
        base=base,
        batch_size=1,
        num_steps=NUM_STEPS,
        field_history=field_history,
        sde_stepper=euler_maruyama_step,
    )

    true_trajectory = trajectory[0, 0].cpu()
    predicted_trajectory = x_prior[0, 0].cpu()
    predicted_trajectory_post = x_post[0, 0].cpu()

    rmse_post = torch.sqrt(
        nn.MSELoss()(
            predicted_trajectory_post, true_trajectory[:, :, len_field_history]
        )
    )
    rmse_prior = torch.sqrt(
        nn.MSELoss()(predicted_trajectory, true_trajectory[:, :, len_field_history])
    )
    logger.info(f"RMSE of posterior: {rmse_post}")
    logger.info(f"RMSE of prior: {rmse_prior}")

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(true_trajectory[:, :, len_field_history])
    plt.title("True")
    plt.subplot(1, 3, 2)
    plt.imshow(predicted_trajectory)
    plt.title("Prior")
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_trajectory_post)
    plt.title("Posterior")
    plt.show()

    # logger.info(f"Inverse transforming predicted trajectory...")
    # predicted_trajectory = preprocesser.inverse_transform(
    #     base=predicted_trajectory, is_batch=True, is_trajectory=True
    # )["base"].numpy()
    # predicted_trajectory = predicted_trajectory[0, 0]

    # logger.info(f"Plotting trajectory...")
    # plotting_times = [2, 4, 6]
    # plt.figure()
    # for i, t in enumerate(plotting_times):
    #     plt.subplot(2, len(plotting_times), i + 1)
    #     plt.imshow(true_trajectory[:, :, t])
    #     plt.title(f"True Trajectory at t={t}")
    #     plt.subplot(2, len(plotting_times), len(plotting_times) + 1 + i)
    #     plt.imshow(predicted_trajectory[:, :, t])
    #     plt.title(f"Predicted Trajectory at t={t}")
    # plt.show()


if __name__ == "__main__":
    main()
