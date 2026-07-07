"""Tests for the trainer hierarchy: DeterministicTrainer end-to-end training,
the StochasticInterpolantTrainer batch hook, and the back-compat aliases."""

import torch
import torch.nn as nn
from ito_map_test_helpers import ToyDataset, make_tiny_unet
from torch.utils.data import DataLoader

from scisi.deterministic_models import DeterministicModel
from scisi.training.base_trainer import BaseTrainer
from scisi.training.deterministic_trainer import DeterministicTrainer
from scisi.training.gradient_clipping import EmaGradientClipper
from scisi.training.trainer import (
    EarlyStopping,
    StochasticInterpolantTrainer,
    Trainer,
)


def _make_deterministic_trainer(num_epochs: int = 2) -> DeterministicTrainer:
    torch.manual_seed(42)
    model = DeterministicModel(network=make_tiny_unet())

    train_dataloader = DataLoader(ToyDataset(num_samples=8, seed=0), batch_size=2)
    val_dataloader = DataLoader(ToyDataset(num_samples=8, seed=1), batch_size=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    return DeterministicTrainer(
        num_epochs=num_epochs,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_clipper=EmaGradientClipper(),
        early_stopping=EarlyStopping(patience=2),
        loss_fn=nn.MSELoss(),
        device="cpu",
        tracker=None,
        mixed_precision_warmup=0,
    )


def _loss_on_fixed_batch(trainer: DeterministicTrainer) -> float:
    batch = next(iter(trainer.train_dataloader))
    trainer.model.eval()
    with torch.no_grad():
        loss = trainer._compute_loss(trainer._prepare_batch(batch))
    return loss.item()


def test_deterministic_trainer_trains_and_loss_decreases():
    trainer = _make_deterministic_trainer(num_epochs=2)

    loss_before = _loss_on_fixed_batch(trainer)
    trainer.train()
    loss_after = _loss_on_fixed_batch(trainer)

    assert loss_after < loss_before


def test_stochastic_interpolant_trainer_prepare_batch_adds_time_and_noise():
    torch.manual_seed(0)
    model = DeterministicModel(network=make_tiny_unet())
    dataloader = DataLoader(ToyDataset(num_samples=4, seed=0), batch_size=2)

    trainer = StochasticInterpolantTrainer(
        num_epochs=1,
        model=model,
        train_dataloader=dataloader,
        val_dataloader=dataloader,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        gradient_clipper=EmaGradientClipper(),
        early_stopping=EarlyStopping(patience=2),
        device="cpu",
        tracker=None,
    )

    batch = trainer._prepare_batch(next(iter(dataloader)))

    assert batch["t"].shape == (batch["base"].shape[0], 1)
    assert batch["noise"].shape == batch["base"].shape


def test_backwards_compat_aliases():
    assert Trainer is StochasticInterpolantTrainer
    assert issubclass(DeterministicTrainer, BaseTrainer)

    # Checkpointed config.yamls reference scisi.training.trainer.EarlyStopping.
    from scisi.training.base_trainer import EarlyStopping as BaseEarlyStopping
    from scisi.training.trainer import EarlyStopping as ReexportedEarlyStopping

    assert ReexportedEarlyStopping is BaseEarlyStopping
