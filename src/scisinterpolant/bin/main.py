import logging
import pdb

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

from scisinterpolant.architectures.u_net import UNet

logger = logging.getLogger(__name__)


@hydra.main(  # type: ignore[misc]
    config_path="../../../config",
    config_name="stochastic_navier_stokes.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:

    logger.info(f"Preparing train dataloader...")
    train_dataloader = hydra.utils.instantiate(cfg.train_data)

    logger.info(f"Preparing val dataloader...")
    val_dataloader = hydra.utils.instantiate(cfg.val_data)

    logger.info(f"Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)

    logger.info(f"Instantiating optimizer...")
    optimizer = hydra.utils.instantiate(
        cfg.optimizer,
        params=model.drift_model.parameters(),
    )

    logger.info(f"Instantiating trainer...")
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    logger.info(f"Training...")
    trainer.train()

if __name__ == "__main__":
    main()
