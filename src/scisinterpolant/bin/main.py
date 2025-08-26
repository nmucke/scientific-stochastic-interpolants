import logging
import pdb

import hydra
import torch
from omegaconf import DictConfig

from scisinterpolant.models.conv_next import MultipleConvNextBlocks
from scisinterpolant.models.embeddings import FourierScalarEncoder

logger = logging.getLogger(__name__)


@hydra.main(  # type: ignore[misc]
    config_path="../../../config",
    config_name="stochastic_navier_stokes.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:

    logger.info(f"Preparing dataloader...")
    dataloader = hydra.utils.instantiate(cfg.data)

    batch = next(iter(dataloader))
    cond = torch.abs(torch.randn(cfg.data.batch_size, 1))

    cond = FourierScalarEncoder(embedding_dim=4)(cond)

    model = MultipleConvNextBlocks(
        in_channels=1,
        out_channels=5,
        cond_dim=4,
        multiplier=2,
        num_blocks=2,
    )

    out = model(batch["base"], cond)

    pdb.set_trace()


if __name__ == "__main__":
    main()
