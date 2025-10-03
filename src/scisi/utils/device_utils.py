import torch
from omegaconf import DictConfig


def get_device() -> str:
    """Get the device of the model."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_device(cfg: DictConfig) -> None:
    """Set the device of the model."""
    if torch.cuda.is_available():
        cfg.trainer.device = "cuda"
    else:
        cfg.trainer.device = "cpu"
