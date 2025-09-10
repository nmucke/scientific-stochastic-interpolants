import logging
import pdb

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from omegaconf import DictConfig

from scisinterpolant.preprocessing.preprocessor import Preprocesser

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

    logger.info(f"Sampling from the model...")
    trajectory = test_dataset[0]["x"].unsqueeze(0)

    x = trajectory[:, :, :, :, 1]
    x_cond = trajectory[:, :, :, :, 0:2]
    x_cond = x_cond.squeeze(0)
    x_cond = torch.permute(x_cond, (0, 3, 1, 2))

    x = preprocesser.transform(base=x, is_batch=True)["base"]
    x_cond = preprocesser.transform(field_cond=x_cond, is_batch=True)["field_cond"]

    x = x.to("cuda")
    x_cond = x_cond.to("cuda")

    num_steps = 100

    x = model.sample_trajectory(
        base=x,
        batch_size=1,
        num_steps=num_steps,
        num_physical_steps=100,
        field_cond=x_cond,
        pars_cond=None,
    )

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(x[0, 0, :, :, 20].cpu().numpy())
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(x[0, 0, :, :, 30].cpu().numpy())

    plt.subplot(1, 3, 3)
    plt.imshow(x[0, 0, :, :, 40].cpu().numpy())
    plt.show()

    true_trajectory = trajectory[0, 0, :, :, 2].cpu().numpy()
    predicted_trajectory = x.cpu()
    predicted_trajectory = preprocesser.inverse_transform(
        base=predicted_trajectory, is_batch=True
    )["base"].numpy()
    predicted_trajectory = predicted_trajectory[0, 0]

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
