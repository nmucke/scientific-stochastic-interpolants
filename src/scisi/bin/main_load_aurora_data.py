import logging

# Disable slow attention warnings
import warnings

import hydra
import torch

# Enable flash attention
import torch.backends.cuda
from omegaconf import DictConfig
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*The operator.*is not optimized.*")

logger = logging.getLogger(__name__)

# Suppress httpx logs to avoid cluttering the logs from Trackio initialization
logging.getLogger("httpx").setLevel(logging.WARNING)


@hydra.main(  # type: ignore[misc]
    config_path="../../../config",
    config_name="aurora.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Load Aurora data to cache."""

    logger.info(f"Preparing train dataloader...")
    train_dataloader = hydra.utils.instantiate(
        cfg.train_data,
        dataset={"preprocesser": None},
    )

    logger.info(f"Preparing val dataloader...")
    val_dataloader = hydra.utils.instantiate(
        cfg.val_data,
        dataset={"preprocesser": None},
    )

    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        pass

    for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        pass


if __name__ == "__main__":
    main()
