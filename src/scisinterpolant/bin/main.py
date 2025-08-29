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

    logger.info(f"Preparing dataloader...")
    dataloader = hydra.utils.instantiate(cfg.data)

    logger.info(f"Instantiating model...")
    model = hydra.utils.instantiate(cfg.architecture)

    logger.info(f"Test forward pass...")
    batch = next(iter(dataloader))
    cond = torch.abs(torch.randn(cfg.data.batch_size, 1))
    out = model(batch["base"], cond, batch["field_cond"])

    pdb.set_trace()


if __name__ == "__main__":
    main()
