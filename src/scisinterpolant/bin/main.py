import logging
import pdb

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(  # type: ignore[misc]
    config_path="../../../config",
    config_name="stochastic_navier_stokes.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:

    logger.info(f"Preparing dataloader...")
    dataloader = hydra.utils.instantiate(cfg.data)


if __name__ == "__main__":
    main()
