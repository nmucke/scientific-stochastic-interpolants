import logging

import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def get_device() -> str:
    """Get the device of the model."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_device(cfg: DictConfig) -> None:
    """Set the device of the model."""
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using GPU.")

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        
        cfg.trainer.device = "cuda"
    else:
        logger.info(f"CUDA is not available. Using CPU.")
        cfg.trainer.device = "cpu"
