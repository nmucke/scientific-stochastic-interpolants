import logging
import pdb

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from scisi.likelihood_models.gaussian_likelihood import Likelihood
from scisi.likelihood_models.observation_operators import LinearObservationOperator
from scisi.preprocessing.preprocessor import Preprocesser
from scisi.sampling.sde_solvers import euler_maruyama_step, heun_step

logger = logging.getLogger(__name__)

VERBOSE = True

DEFAULT_PROJECT = "stochastic_navier_stokes"
DEFAULT_NAME = "zealous-wave-18"
NUM_PHYSICAL_STEPS = 50
SKIP_GRID = 4


@hydra.main(  # type: ignore[misc]
    config_path="../../../checkpoints",
    config_name=f"{DEFAULT_PROJECT}/{DEFAULT_NAME}/config.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    project = list(cfg.keys())[0]
    name = list(cfg[project].keys())[0]
    cfg = OmegaConf.select(cfg, f"{project}.{name}")

    logger.info(f"Instantiating preprocesser...")
    preprocesser = hydra.utils.instantiate(cfg.preprocesser)

    logger.info(f"Instantiating test data...")
    test_dataset = hydra.utils.instantiate(cfg.test_data)

    logger.info(f"Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)

    logger.info(f"Loading model from checkpoint...")
    model.load_state_dict(torch.load(f"checkpoints/{project}/{name}/model.pth"))
    model.eval()
    model.to("cuda")

    logger.info(f"Instantiating observation operator...")
    obs_indices = [
        (0, i, j) for i in range(0, 128, SKIP_GRID) for j in range(0, 128, SKIP_GRID)
    ]
    obs_indices = torch.tensor(obs_indices)

    logger.info(f"Preparing trajectory...")
    trajectory = test_dataset[0]["x"].unsqueeze(0)

    x = trajectory[:, :, :, :, 1]
    x_history = trajectory[:, :, :, :, 0:2]

    logger.info(f"Preparing observations...")
    obs_operator = LinearObservationOperator(
        obs_indices=obs_indices,
    )
    observations = torch.zeros(1, len(obs_indices), NUM_PHYSICAL_STEPS)
    for i in range(NUM_PHYSICAL_STEPS):
        y = preprocesser.transform(base=trajectory[:, :, :, :, i], is_batch=True)[
            "base"
        ]
        observations[:, :, i] = obs_operator(y)

    likelihood = Likelihood(
        obs_operator=obs_operator,
        dist=torch.distributions.Normal,
        loc=observations[:, :, 0],
        scale=0.05,
    )

    logger.info(f"Preprocessing trajectory...")
    x = preprocesser.transform(base=x, is_batch=True)["base"]
    x_history = preprocesser.transform(field_history=x_history, is_batch=True)[
        "field_history"
    ]

    x = x.to("cuda")
    x_history = x_history.to("cuda")

    logger.info(f"Sampling from the model...")
    num_steps = 500
    x_post = model.posterior_sample_trajectory(
        base=x,
        batch_size=1,
        num_steps=num_steps,
        field_history=x_history,
        num_physical_steps=NUM_PHYSICAL_STEPS,
        observations=observations[:, :, 2:],
        likelihood_model=likelihood,
        sde_stepper=euler_maruyama_step,
        diffusion_term=lambda t: 4.0 * model.interpolation.gamma(t),
    )

    x_prior = model.sample_trajectory(
        base=x,
        batch_size=1,
        num_steps=num_steps,
        field_history=x_history,
        num_physical_steps=NUM_PHYSICAL_STEPS,
    )

    true_trajectory = trajectory[0, 0].cpu().numpy()
    predicted_trajectory = x_prior[0, 0].cpu()
    predicted_trajectory_post = x_post[0, 0].cpu()

    rmse_post = torch.sqrt(
        torch.mean(
            (predicted_trajectory_post - true_trajectory[:, :, :NUM_PHYSICAL_STEPS])
            ** 2
        )
    )
    rmse_prior = torch.sqrt(
        torch.mean(
            (predicted_trajectory - true_trajectory[:, :, :NUM_PHYSICAL_STEPS]) ** 2
        )
    )

    logger.info(f"RMSE of posterior: {rmse_post}")
    logger.info(f"RMSE of prior: {rmse_prior}")

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
