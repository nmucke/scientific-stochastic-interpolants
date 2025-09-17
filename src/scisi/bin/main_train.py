import logging
import pdb

import hydra
import torch

# Enable flash attention
import torch.backends.cuda
import torch.nn as nn
import trackio
from omegaconf import DictConfig, OmegaConf

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

torch.set_default_dtype(torch.float32)

# Disable slow attention warnings
import warnings

warnings.filterwarnings("ignore", message=".*The operator.*is not optimized.*")

logger = logging.getLogger(__name__)

# Suppress httpx logs to avoid cluttering the logs from Trackio initialization
logging.getLogger("httpx").setLevel(logging.WARNING)

VERBOSE = True
CONTINUE_FROM_CHECKPOINT = True
CHECKPOINT_PROJECT = "stochastic_navier_stokes"
CHECKPOINT_NAME = "quirky-cave-36"
CHECKPOINT_PATH = f"checkpoints/{CHECKPOINT_PROJECT}/{CHECKPOINT_NAME}/model.pth"


@hydra.main(  # type: ignore[misc]
    config_path="../../../config",
    config_name="stochastic_navier_stokes.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:

    if CONTINUE_FROM_CHECKPOINT:
        logger.info(f"Loading config from checkpoint:")
        logger.info(f"Project: {CHECKPOINT_PROJECT}")
        logger.info(f"Name: {CHECKPOINT_NAME}")
        cfg = OmegaConf.load(
            f"checkpoints/{CHECKPOINT_PROJECT}/{CHECKPOINT_NAME}/config.yaml"
        )
    

    logger.info(f"Instantiating experiment tracking...")
    tracker = trackio.init(
        project=cfg.experiment_tracking.project,
        config=cfg,
    )
    logger.info(f"Tracker instantiated with properties:")
    logger.info(f"Project: {tracker.project}")
    logger.info(f"URL: {tracker.url}")
    logger.info(f"Name: {tracker.name}")

    logger.info(f"Instantiating preprocesser...")
    preprocesser = hydra.utils.instantiate(cfg.preprocesser)

    logger.info(f"Preparing train dataloader...")
    train_dataloader = hydra.utils.instantiate(
        cfg.train_data,
        dataset={"preprocesser": preprocesser},
    )

    logger.info(f"Preparing val dataloader...")
    val_dataloader = hydra.utils.instantiate(
        cfg.val_data,
        dataset={"preprocesser": preprocesser},
    )

    logger.info(f"Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)
    if CONTINUE_FROM_CHECKPOINT:
        logger.info(f"Loading model from checkpoint:")
        logger.info(f"{CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH))

    logger.info(f"Instantiating optimizer...")
    optimizer = hydra.utils.instantiate(
        cfg.optimizer,
        params=model.drift_model.parameters(),
    )

    logger.info(f"Instantiating scheduler...")
    scheduler = hydra.utils.instantiate(
        cfg.scheduler,
        optimizer=optimizer,
    )

    logger.info(f"Instantiating trainer...")
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tracker=tracker,
    )

    logger.info(f"Training...")
    trainer.train()

    logger.info(f"Closing tracker...")
    tracker.finish()


if __name__ == "__main__":
    main()
