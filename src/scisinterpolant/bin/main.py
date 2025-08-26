import hydra
import pdb
import logging

logger = logging.getLogger(__name__)

@hydra.main(
    config_path="../../../config",
    config_name="stochastic_navier_stokes.yaml",
    version_base=None,
)
def main(cfg):

    logger.info(f"Preparing dataloader...")
    dataloader = hydra.utils.instantiate(cfg.data)
    


if __name__ == "__main__":
    main()