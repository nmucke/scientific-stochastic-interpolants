"""Refactor guard for the trainer split (plan Phase 1).

Asserts that the new ``StochasticInterpolantTrainer`` produces a bit-identical
loss trajectory to the pre-refactor ``Trainer`` (copied verbatim below as
``_LegacyTrainer``), and that the back-compat aliases hold so the 25+ existing
configs and checkpointed config.yamls keep working.
"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn
from ito_map_test_helpers import (
    RecordingMSELoss,
    make_tiny_unet,
    make_trainer_kwargs,
)
from tqdm import tqdm

from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
from scisi.models.interpolations import QuadraticStochasticInterpolation
from scisi.training.base_trainer import BaseTrainer
from scisi.training.trainer import (
    SCHEDULERS_THAT_REQUIRE_LOSS,
    EarlyStopping,
    StochasticInterpolantTrainer,
    Trainer,
)

logger = logging.getLogger(__name__)


class _LegacyTrainer:
    """The pre-refactor scisi.training.trainer.Trainer, copied verbatim.

    Only deviation: ``_print_info`` guards against ``tracker is None`` (the
    original crashed without a tracker), which does not affect the loss
    trajectory.
    """

    def __init__(
        self,
        num_epochs,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        gradient_clipper,
        early_stopping,
        loss_fn=nn.MSELoss(),
        scheduler=None,
        device="cpu",
        tracker=None,
        mixed_precision_warmup=0,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.gradient_clipper = gradient_clipper

        self.scheduler = scheduler
        if self.scheduler is not None:
            self.scheduler_requires_loss = (
                scheduler.__class__.__name__ in SCHEDULERS_THAT_REQUIRE_LOSS
            )
        else:
            self.scheduler_requires_loss = False

        self.mixed_precision_warmup = mixed_precision_warmup
        if (self.mixed_precision_warmup > 0) and (self.device != "cpu"):
            self.full_precision = False
            self.scaler = torch.amp.GradScaler(self.device)
            self._train_step = self._train_step_mixed_precision
        else:
            self.full_precision = True
            self._train_step = self._train_step_full_precision

        self.model.train()
        self.model.to(self.device)

        self.current_lr = self.scheduler.get_last_lr()[0]

        self.tracker = tracker

    def _prepare_batch(self, batch):
        for key, value in batch.items():
            batch[key] = value.to(self.device)

        batch["t"] = torch.rand(batch["base"].shape[0], 1, device=self.device)
        batch["noise"] = torch.randn(batch["base"].shape, device=self.device)

        return batch

    def _compute_loss(self, batch):
        pred_drift, true_diff = self.model(**batch)
        return self.loss_fn(pred_drift, true_diff)

    def _train_step_mixed_precision(self, batch):
        self.optimizer.zero_grad()

        with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
            loss = self._compute_loss(batch)

        self.scaler.scale(loss).backward()
        self.gradient_clipper.clip_grads(self.model.parameters())
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss

    def _train_step_full_precision(self, batch):
        self.optimizer.zero_grad()
        loss = self._compute_loss(batch)

        loss.backward()
        self.gradient_clipper.clip_grads(self.model.parameters())

        self.optimizer.step()
        return loss

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()

            train_loss = 0.0
            pbar = tqdm(self.train_dataloader)
            for batch in pbar:
                batch = self._prepare_batch(batch)
                loss = self._train_step(batch)
                train_loss += loss.item()
                pbar.set_description(f"Epoch {epoch}, Loss: {loss:.4f}")

            if (epoch >= self.mixed_precision_warmup) and (not self.full_precision):
                self._train_step = self._train_step_full_precision
                self.full_precision = True

            train_loss /= len(self.train_dataloader)

            val_loss = self._compute_val_loss()

            if self._check_early_stopping(val_loss, epoch):
                break

            self._update_scheduler(val_loss)

    def _check_early_stopping(self, val_loss, epoch):
        if self.early_stopping(val_loss):
            return True
        return False

    def _update_scheduler(self, val_loss):
        self.scheduler.step(val_loss if self.scheduler_requires_loss else None)
        if np.abs(self.current_lr - self.scheduler.get_last_lr()[0]) > 1e-6:
            self.current_lr = self.scheduler.get_last_lr()[0]

    def _compute_val_loss(self):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in self.val_dataloader:
                batch = self._prepare_batch(batch)
                loss = self._compute_loss(batch)
                val_loss += loss.item()
            val_loss /= len(self.val_dataloader)
        return val_loss


def _build_model(seed: int) -> FollmerStochasticInterpolant:
    torch.manual_seed(seed)
    return FollmerStochasticInterpolant(
        interpolation=QuadraticStochasticInterpolation(gamma_multiplier=1.0),
        drift_model=make_tiny_unet(),
    )


def _run(trainer_cls) -> list[float]:
    """Build everything from a fixed seed, train 2 epochs, return all losses."""
    model = _build_model(seed=42)
    loss_fn = RecordingMSELoss()
    kwargs = make_trainer_kwargs(model, loss_fn=loss_fn)

    torch.manual_seed(123)
    trainer = trainer_cls(**kwargs)
    trainer.train()
    return loss_fn.values


def test_golden_loss_trajectory_matches_legacy_trainer():
    legacy_losses = _run(_LegacyTrainer)
    new_losses = _run(StochasticInterpolantTrainer)

    assert len(legacy_losses) > 0
    assert legacy_losses == new_losses  # bit-identical


def test_trainer_alias_and_reexports():
    assert Trainer is StochasticInterpolantTrainer
    assert issubclass(StochasticInterpolantTrainer, BaseTrainer)

    # Checkpointed config.yamls reference these paths.
    from scisi.training.trainer import EarlyStopping as ReexportedEarlyStopping
    from scisi.training.base_trainer import EarlyStopping as BaseEarlyStopping

    assert ReexportedEarlyStopping is BaseEarlyStopping
    assert "ReduceLROnPlateau" in SCHEDULERS_THAT_REQUIRE_LOSS


def test_checkpoint_saving_without_tracker(tmp_path):
    """BaseTrainer saves checkpoints without a tracker when given a path,
    and checkpoints the EMA weights alongside the model when EMA is on."""
    model = _build_model(seed=7)
    kwargs = make_trainer_kwargs(model)
    kwargs["checkpoint_path"] = str(tmp_path / "run")
    kwargs["ema_decay"] = 0.9

    trainer = StochasticInterpolantTrainer(**kwargs)
    trainer.train()

    assert os.path.exists(str(tmp_path / "run" / "model.pth"))
    assert os.path.exists(str(tmp_path / "run" / "ema_model.pth"))
    ema_state = torch.load(str(tmp_path / "run" / "ema_model.pth"))
    assert set(ema_state.keys()) == set(trainer.model.state_dict().keys())


def test_ema_weights_track_model():
    """The optional weight EMA starts at the model weights and stays finite."""
    model = _build_model(seed=8)
    kwargs = make_trainer_kwargs(model)
    kwargs["ema_decay"] = 0.9

    trainer = StochasticInterpolantTrainer(**kwargs)
    trainer.train()

    for ema_param, param in zip(
        trainer.ema_model.parameters(), trainer.model.parameters()
    ):
        assert torch.isfinite(ema_param).all()
        assert ema_param.shape == param.shape
