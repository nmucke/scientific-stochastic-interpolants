"""Shared builders for the Ito-map and trainer-refactor tests."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from scisi.architectures.u_net import UNet
from scisi.training.base_trainer import EarlyStopping
from scisi.training.gradient_clipping import EmaGradientClipper

NUM_CHANNELS = 1
HEIGHT = 8
WIDTH = 8
LEN_FIELD_HISTORY = 2


class ToyDataset(Dataset):
    """Tiny synthetic dataset with the repo's train-batch layout."""

    def __init__(self, num_samples: int = 8, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.field_history = torch.randn(
            num_samples,
            NUM_CHANNELS,
            HEIGHT,
            WIDTH,
            LEN_FIELD_HISTORY,
            generator=generator,
        )
        self.target = torch.randn(
            num_samples, NUM_CHANNELS, HEIGHT, WIDTH, generator=generator
        )

    def __len__(self) -> int:
        return self.field_history.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "field_history": self.field_history[idx],
            "base": self.field_history[idx, :, :, :, -1],
            "target": self.target[idx],
        }


def make_tiny_unet(
    cond_dim: int = 1,
    two_time_cond: bool = False,
    field_cond_channels: int | None = None,
) -> UNet:
    """Attention-free two-level UNet small enough for CPU tests."""
    return UNet(
        in_channels=NUM_CHANNELS,
        out_channels=NUM_CHANNELS,
        hidden_channels=[4, 8],
        cond_dim=cond_dim,
        cond_embedding_dim=16,
        len_field_history=LEN_FIELD_HISTORY,
        field_cond_channels=field_cond_channels,
        multiplier=2,
        num_blocks=1,
        dropout_rate=0.0,
        attention_in_layers=[False, False],
        attention={"target": "torch.nn.Identity"},
        two_time_cond=two_time_cond,
    )


def make_tiny_attention_unet(
    cond_dim: int = 2,
    two_time_cond: bool = True,
    field_cond_channels: int | None = None,
) -> UNet:
    """Tiny UNet with bottleneck attention (exercises the SDPA/jvp path)."""
    return UNet(
        in_channels=NUM_CHANNELS,
        out_channels=NUM_CHANNELS,
        hidden_channels=[4, 8],
        cond_dim=cond_dim,
        cond_embedding_dim=16,
        len_field_history=LEN_FIELD_HISTORY,
        field_cond_channels=field_cond_channels,
        multiplier=2,
        num_blocks=1,
        dropout_rate=0.0,
        attention_in_layers=[False, True],
        attention={
            "target": "scisi.architectures.attention.SpatialAttention",
            "pos_embedding_type": "rotary",
            "heads": 2,
            "patch_size": 1,
            "dropout_rate": 0.0,
            "embedding_dim": 16,
        },
        two_time_cond=two_time_cond,
    )


def make_dataloaders(
    num_samples: int = 8, batch_size: int = 4
) -> tuple[DataLoader, DataLoader]:
    """Deterministic (unshuffled) train/val dataloaders over the toy data."""
    train = DataLoader(ToyDataset(num_samples, seed=0), batch_size=batch_size)
    val = DataLoader(ToyDataset(num_samples, seed=1), batch_size=batch_size)
    return train, val


def make_trainer_kwargs(model: nn.Module, loss_fn: nn.Module | None = None) -> dict:
    """Common BaseTrainer constructor kwargs for a 2-epoch CPU run."""
    train_dataloader, val_dataloader = make_dataloaders()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    return {
        "num_epochs": 2,
        "model": model,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "gradient_clipper": EmaGradientClipper(),
        "early_stopping": EarlyStopping(patience=10),
        "loss_fn": loss_fn if loss_fn is not None else nn.MSELoss(),
        "device": "cpu",
        "tracker": None,
        "mixed_precision_warmup": 0,
    }


class RecordingMSELoss(nn.Module):
    """MSE loss that records every value it computes (train and val)."""

    def __init__(self) -> None:
        super().__init__()
        self.values: list[float] = []

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.nn.functional.mse_loss(pred, target)
        self.values.append(loss.item())
        return loss
