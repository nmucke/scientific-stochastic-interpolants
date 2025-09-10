import logging
import pdb

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from omegaconf import DictConfig

from scisi.preprocessing.preprocessor import Preprocesser
from scisi.sampling.sde_solvers import euler_maruyama_step, heun_step

logger = logging.getLogger(__name__)

VERBOSE = True


@hydra.main(  # type: ignore[misc]
    config_path="../../../config",
    config_name="stochastic_navier_stokes.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    logger.info(f"Instantiating preprocesser...")
    preprocesser = Preprocesser(
        base=cfg.preprocesser.base,
        target=cfg.preprocesser.target,
        field_cond=cfg.preprocesser.field_cond,
    )

    logger.info(f"Instantiating test data...")
    test_dataset = hydra.utils.instantiate(cfg.test_data)

    logger.info(f"Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)

    logger.info(f"Loading model from checkpoint...")
    model.load_state_dict(torch.load("checkpoints/model.pth"))
    model.eval()
    model.to("cuda")

    logger.info(f"Preparing trajectory...")
    trajectory = test_dataset[0]["x"].unsqueeze(0)

    x = trajectory[:, :, :, :, 1]
    x_history = trajectory[:, :, :, :, 0:2]

    logger.info(f"Preprocessing trajectory...")
    x = preprocesser.transform(base=x, is_batch=True)["base"]
    x_history = preprocesser.transform(field_history=x_history, is_batch=True)[
        "field_history"
    ]

    x = x.to("cuda")
    x_history = x_history.to("cuda")

    logger.info(f"Sampling from the model...")
    num_steps = 250
    x = model.sample(
        base=x,
        batch_size=1,
        num_steps=num_steps,
        field_history=x_history,
        sde_stepper=heun_step,
    )

    true_trajectory = trajectory[0, 0, :, :, 2].cpu().numpy()
    predicted_trajectory = x.cpu()

    logger.info(f"Inverse transforming predicted trajectory...")
    predicted_trajectory = preprocesser.inverse_transform(
        base=predicted_trajectory, is_batch=True
    )["base"].numpy()
    predicted_trajectory = predicted_trajectory[0, 0]

    logger.info(f"Plotting trajectory...")
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(true_trajectory)
    plt.title("True Trajectory")
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(predicted_trajectory)
    plt.title("Predicted Trajectory")
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(true_trajectory - predicted_trajectory)
    plt.title("Error")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
