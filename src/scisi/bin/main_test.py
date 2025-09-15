import logging
import pdb

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from scisi.preprocessing.preprocessor import Preprocesser
from scisi.sampling.sde_solvers import euler_maruyama_step, heun_step

torch.set_default_dtype(torch.float32)

logger = logging.getLogger(__name__)

VERBOSE = True

DEFAULT_PROJECT = "stochastic_navier_stokes"
DEFAULT_NAME = "elegant-pond-24"
NUM_PHYSICAL_STEPS = 50
NUM_STEPS = 50
BATCH_SIZE = 1

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

    logger.info(f"Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)

    logger.info(f"Loading model from checkpoint:")
    logger.info(f"Project: {project}")
    logger.info(f"Name: {name}")
    model.load_state_dict(torch.load(f"checkpoints/{project}/{name}/model.pth"))
    model.eval()
    model.to("cuda")

    logger.info(f"Preparing trajectory...")
    trajectory = test_dataset[0]["x"].unsqueeze(0)

    logger.info(f"Preprocessing trajectory...")
    init_data = preprocesser.transform(
        base=trajectory[:, :, :, :, len_field_history - 1], 
        field_history=trajectory[:, :, :, :, 0:len_field_history],
        is_batch=True
    )
    field_history = init_data["field_history"].to("cuda")
    base = init_data["base"].to("cuda")

    logger.info(f"Sampling from the model...")
    predicted_trajectory = model.sample_trajectory(
        base=base,
        batch_size=BATCH_SIZE,
        num_steps=NUM_STEPS,
        field_history=field_history,
        num_physical_steps=NUM_PHYSICAL_STEPS,
        sde_stepper=heun_step,
        # sde_stepper=euler_maruyama_step,
        # diffusion_term=lambda t: 2.0 * model.interpolation.gamma(t),
    )

    true_trajectory = trajectory[0, 0].cpu().numpy()
    predicted_trajectory = predicted_trajectory.cpu()

    logger.info(f"Inverse transforming predicted trajectory...")
    predicted_trajectory = preprocesser.inverse_transform(
        base=predicted_trajectory, is_batch=True, is_trajectory=True
    )["base"].numpy()
    predicted_trajectory = predicted_trajectory[0, 0]

    logger.info(f"Plotting trajectory...")
    plotting_times = [10, 30, 45]
    plt.figure()
    for i, t in enumerate(plotting_times):
        plt.subplot(2, len(plotting_times), i + 1)
        plt.imshow(true_trajectory[:, :, t])
        plt.title(f"True Trajectory at t={t}")
        plt.subplot(2, len(plotting_times), len(plotting_times) + 1 + i)
        plt.imshow(predicted_trajectory[:, :, t])
        plt.title(f"Predicted Trajectory at t={t}")
    plt.show()


if __name__ == "__main__":
    main()
