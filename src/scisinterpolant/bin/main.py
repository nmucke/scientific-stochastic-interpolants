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
    model = hydra.utils.instantiate(cfg.model)

    logger.info(f"Test forward pass...")
    batch = next(iter(dataloader))

    t = torch.abs(torch.randn(cfg.data.batch_size, 1))
    noise = torch.randn(batch["base"].shape)

    out = model(
        base=batch["base"],
        target=batch["target"],
        t=t,
        noise=noise,
        field_cond=batch.get("field_cond", None),
        pars_cond=batch.get("pars_cond", None),
    )
    pdb.set_trace()


if __name__ == "__main__":
    main()
