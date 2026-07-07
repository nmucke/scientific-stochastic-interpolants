"""Base trainer.

Owns everything generic about training (epoch loop, validation, logging,
checkpointing, early stopping, scheduler handling, mixed precision, gradient
clipping, optional weight EMA). Model-specific trainers subclass it and only
override the two hooks ``_prepare_batch`` and ``_compute_loss``.
"""

import logging
import os
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import trackio
from omegaconf import OmegaConf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from scisi.training.gradient_clipping import EmaGradientClipper

logger = logging.getLogger(__name__)

SCHEDULERS_THAT_REQUIRE_LOSS = [
    "ReduceLROnPlateau",
]


@dataclass
class EarlyStopping:
    """Early stopping for the trainer."""

    patience: int = 10
    delta: float = 1e-6
    best_loss: float = float("inf")
    counter: int = 0
    early_stop: bool = False
    save_checkpoint: bool = False

    def __call__(self, val_loss: float) -> bool:
        """Check if the early stopping condition is met."""
        if val_loss < self.best_loss + self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint = True

            logger.info(f"New best Val Loss: {self.best_loss:.4f}")
        else:
            self.counter += 1
            self.save_checkpoint = False
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop


class BaseTrainer:
    """Generic trainer.

    Subclasses override:
        - ``_prepare_batch(batch)``: base implementation moves tensors to the
          device; subclasses inject sampled quantities (pseudo-times, noise,
          Brownian paths, ...).
        - ``_compute_loss(batch)``: default is the ``(pred, target) =
          model(**batch)`` contract followed by ``loss_fn(pred, target)``.
    """

    def __init__(
        self,
        num_epochs: int,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: Optimizer,
        gradient_clipper: EmaGradientClipper,
        early_stopping: EarlyStopping,
        loss_fn: nn.Module = nn.MSELoss(),
        scheduler: LRScheduler = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tracker: trackio.Run | None = None,
        mixed_precision_warmup: int = 0,
        checkpoint_path: str | None = None,
        ema_decay: float | None = None,
        val_seed: int | None = None,
    ):
        """Initialize the trainer.

        Args:
            num_epochs: Number of epochs to train for.
            model: Model to train.
            train_dataloader: Train dataloader.
            val_dataloader: Validation dataloader.
            optimizer: Optimizer.
            gradient_clipper: EMA gradient clipper.
            early_stopping: Early stopping handler.
            loss_fn: Loss function applied to ``(pred, target)``.
            scheduler: Optional learning-rate scheduler.
            device: Device to train on.
            tracker: Optional trackio run for logging + checkpoint naming.
            mixed_precision_warmup: Number of epochs to train in bf16 before
                switching to full precision. 0 disables mixed precision.
            checkpoint_path: Directory to save checkpoints to when no tracker
                is provided. With a tracker, checkpoints go to
                ``checkpoints/<project>/<name>`` as before.
            ema_decay: Optional decay for an exponential moving average of the
                model weights (e.g. 0.999). ``None`` disables the EMA. The EMA
                model is exposed as ``self.ema_model`` for subclasses that use
                it as a stop-gradient target (self-distillation losses), and
                its weights are checkpointed alongside the model
                (``ema_model.pth``) - for self-distillation training the EMA
                weights are typically the ones to evaluate.
            val_seed: Optional seed for the validation loop's stochastic batch
                preparation (pseudo-times, noise, Brownian paths). When set,
                every validation pass samples identically, so val losses are
                comparable across epochs and best-model selection is not
                partly noise. ``None`` (default) keeps the legacy freshly
                randomized validation. The training RNG stream is unaffected
                either way (the RNG state is forked around the loop).
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.gradient_clipper = gradient_clipper

        # Initialize scheduler
        self.scheduler = scheduler
        if self.scheduler is not None:
            self.scheduler_requires_loss = (
                scheduler.__class__.__name__ in SCHEDULERS_THAT_REQUIRE_LOSS
            )
        else:
            self.scheduler_requires_loss = False

        # Initialize mixed precision
        self.mixed_precision_warmup = mixed_precision_warmup
        if (self.mixed_precision_warmup > 0) and (self.device != "cpu"):
            logger.info(f"Mixed precision warmup: {self.mixed_precision_warmup}")
            self.full_precision = False
            self.scaler = torch.amp.GradScaler(self.device)
            self._train_step = self._train_step_mixed_precision
        else:
            logger.info(f"Mixed precision warmup not set, using full precision")
            self.full_precision = True
            self._train_step = self._train_step_full_precision

        # Initialize model
        self.model.train()
        self.model.to(self.device)

        if self.scheduler is not None:
            self.current_lr = self.scheduler.get_last_lr()[0]
        else:
            self.current_lr = self.optimizer.param_groups[0]["lr"]

        # Optional EMA of the model weights (off by default)
        self.ema_decay = ema_decay
        if ema_decay is not None:
            self.ema_model = deepcopy(self.model)
            self.ema_model.to(self.device)
            for param in self.ema_model.parameters():
                param.requires_grad_(False)
            self.ema_model.eval()
        else:
            self.ema_model = None

        self.val_seed = val_seed

        # Initialize tracker
        self.tracker = tracker

        # Initialize checkpointing. With a tracker the path is derived from the
        # trackio run; without one an explicit checkpoint_path can be supplied.
        self.checkpoint_model_path = None
        if self.tracker is not None:
            checkpoint_path = f"checkpoints/{self.tracker.project}/{self.tracker.name}"

        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
            os.makedirs(self.checkpoint_path, exist_ok=True)

            # Dump config to file (the config lives on the tracker):
            if self.tracker is not None:
                config_path = f"{self.checkpoint_path}/config.yaml"
                with open(config_path, "w") as f:
                    OmegaConf.save(self.tracker.config, f)

            self.checkpoint_model_path = f"{self.checkpoint_path}/model.pth"

    def _prepare_batch(self, batch: dict) -> dict[str, torch.Tensor]:
        """Prepare the batch for the model: device transfer only."""
        for key, value in batch.items():
            batch[key] = value.to(self.device)

        return batch

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        """Compute the loss for the model."""

        # Compute pred and true drift
        pred_drift, true_diff = self.model(**batch)

        # Compute and return the loss
        return self.loss_fn(pred_drift, true_diff)

    def _on_epoch_start(self, epoch: int) -> None:
        """Hook called at the top of every epoch. No-op by default."""

    def _update_ema(self) -> None:
        """Update the EMA of the model weights after an optimizer step."""
        if self.ema_model is None:
            return
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                ema_param.mul_(self.ema_decay).add_(param, alpha=1 - self.ema_decay)
            for ema_buffer, buffer in zip(
                self.ema_model.buffers(), self.model.buffers()
            ):
                ema_buffer.copy_(buffer)

    def _train_step_mixed_precision(self, batch: dict) -> torch.Tensor:
        """Train the model for one step using mixed precision."""
        self.optimizer.zero_grad()

        with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
            loss = self._compute_loss(batch)

        self.scaler.scale(loss).backward()

        # Clip gradients to prevent exploding gradients
        self.gradient_clipper.clip_grads(self.model.parameters())

        self.scaler.step(self.optimizer)
        self.scaler.update()

        self._update_ema()
        return loss

    def _train_step_full_precision(self, batch: dict) -> torch.Tensor:
        """Train the model for one step."""
        self.optimizer.zero_grad()
        loss = self._compute_loss(batch)

        loss.backward()

        # Clip gradients to prevent exploding gradients
        self.gradient_clipper.clip_grads(self.model.parameters())

        self.optimizer.step()

        self._update_ema()
        return loss

    def train(self) -> None:
        """Train the model."""

        for epoch in range(self.num_epochs):
            self._on_epoch_start(epoch)

            self.model.train()

            # Loop over batches
            train_loss = 0.0
            pbar = tqdm(self.train_dataloader)
            for batch in pbar:

                batch = self._prepare_batch(batch)

                loss = self._train_step(batch)
                train_loss += loss.item()

                pbar.set_description(f"Epoch {epoch}, Loss: {loss:.4f}")

            # Switch to full precision if warmup is finished
            if (epoch >= self.mixed_precision_warmup) and (not self.full_precision):
                logger.info(
                    f"Mixed precision warmup finished, switching to full precision"
                )
                self._train_step = self._train_step_full_precision
                self.full_precision = True

            # Compute average train loss
            train_loss /= len(self.train_dataloader)

            val_loss = self._compute_val_loss()

            self._log_with_tracker(epoch, train_loss, val_loss)

            if self._check_early_stopping(val_loss, epoch):
                break

            self._save_checkpoint(epoch)

            self._update_scheduler(val_loss)

            self._print_info(epoch, train_loss, val_loss)

        logger.info(f"Training finished")
        logger.info(f"Best Val Loss: {self.early_stopping.best_loss:.4f}")
        (
            logger.info(f"Tracking: {self.tracker.name} - {self.tracker.url}")
            if self.tracker is not None
            else None
        )

    def _log_with_tracker(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """Log with tracker."""
        if self.tracker is not None:
            self.tracker.log(
                {
                    "epoch": epoch,
                    "log_train_loss": np.log(train_loss),
                    "log_val_loss": np.log(val_loss),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )

    def _check_early_stopping(self, val_loss: float, epoch: int) -> bool:
        """Early stopping."""
        if self.early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            if (self.checkpoint_model_path is not None) and os.path.exists(
                self.checkpoint_model_path
            ):
                self.model.load_state_dict(torch.load(self.checkpoint_model_path))
            return True
        return False

    def _update_scheduler(self, val_loss: float) -> None:
        """Update the scheduler."""
        if self.scheduler is None:
            return
        self.scheduler.step(val_loss if self.scheduler_requires_loss else None)
        if np.abs(self.current_lr - self.scheduler.get_last_lr()[0]) > 1e-6:
            self.current_lr = self.scheduler.get_last_lr()[0]
            logger.info(f"Scheduler updated learning rate to {self.current_lr:.6f}")

    def _save_checkpoint(self, epoch: int) -> None:
        """Save the checkpoint if the early stopping condition is met."""
        if (self.early_stopping.save_checkpoint) and (
            self.checkpoint_model_path is not None
        ):
            logger.info(f"Saving checkpoint at epoch {epoch}")
            torch.save(self.model.state_dict(), self.checkpoint_model_path)
            if self.ema_model is not None:
                torch.save(
                    self.ema_model.state_dict(),
                    f"{self.checkpoint_path}/ema_model.pth",
                )

    def _print_info(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """Print the information about the training and the tracking."""
        tracking_info = (
            f"Tracking: {self.tracker.name} - {self.tracker.url}"
            if self.tracker is not None
            else "Tracking: disabled"
        )
        logger.info(
            f"Epoch {epoch}, Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Best Val Loss: {self.early_stopping.best_loss:.4f}, "
            f"{tracking_info}"
        )

    def _compute_val_loss(self) -> float:
        """Validate the model.

        With ``val_seed`` set, the loop's stochastic batch preparation is
        seeded inside a forked RNG scope so every validation pass samples
        identically and the training RNG stream is untouched.
        """
        if self.val_seed is None:
            return self._run_validation_loop()

        devices = [self.device] if str(self.device).startswith("cuda") else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.val_seed)
            return self._run_validation_loop()

    def _run_validation_loop(self) -> float:
        """Run one pass over the validation dataloader."""
        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in self.val_dataloader:
                batch = self._prepare_batch(batch)
                loss = self._compute_loss(batch)
                val_loss += loss.item()
            val_loss /= len(self.val_dataloader)
        return val_loss
