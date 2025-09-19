import logging
import os
import pdb
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


class Trainer:
    """Trainer for the stochastic interpolant."""

    def __init__(
        self,
        num_epochs: int,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: Optimizer,
        early_stopping: EarlyStopping,
        loss_fn: nn.Module = nn.MSELoss(),
        scheduler: LRScheduler = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_grad_norm: float = 1.0,
        tracker: trackio.Run | None = None,
        mixed_precision_warmup: int = 0,
    ):
        """Initialize the trainer."""
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.max_grad_norm = max_grad_norm

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
        if self.mixed_precision_warmup > 0:
            logger.info(f"Mixed precision warmup: {self.mixed_precision_warmup}")
            self.full_precision = False
            self.scaler = torch.amp.GradScaler()
            self._train_step = self._train_step_mixed_precision
        else:
            logger.info(f"Mixed precision warmup not set, using full precision")
            self.full_precision = True
            self._train_step = self._train_step_full_precision

        # Initialize model
        self.model.train()
        self.model.to(self.device)

        self.current_lr = self.scheduler.get_last_lr()[0]

        # Initialize tracker
        self.tracker = tracker

        # Initialize checkpointing with Trackio tracker
        if self.tracker is not None:

            self.checkpoint_path = (
                f"checkpoints/{self.tracker.project}/{self.tracker.name}"
            )
            os.makedirs(self.checkpoint_path, exist_ok=True)

            # Dump config to file:
            config_path = f"{self.checkpoint_path}/config.yaml"
            with open(config_path, "w") as f:
                OmegaConf.save(self.tracker.config, f)

            self.checkpoint_model_path = f"{self.checkpoint_path}/model.pth"

    def _prepare_batch(self, batch: dict) -> dict[str, torch.Tensor]:
        """Prepare the batch for the model."""
        for key, value in batch.items():
            batch[key] = value.to(self.device)

        # Sample pseudo-time
        batch["t"] = torch.abs(
            torch.randn(batch["base"].shape[0], 1, device=self.device)
        )

        # Sample noise
        batch["noise"] = torch.randn(batch["base"].shape, device=self.device)

        return batch

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        """Compute the loss for the model."""

        # Compute pred and true drift
        pred_drift, true_diff = self.model(**batch)

        # Compute and return the loss
        return self.loss_fn(pred_drift, true_diff)

    def _train_step_mixed_precision(self, batch: dict) -> torch.Tensor:
        """Train the model for one step using mixed precision."""
        self.optimizer.zero_grad()

        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            loss = self._compute_loss(batch)

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.max_grad_norm
        )

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss

    def _train_step_full_precision(self, batch: dict) -> torch.Tensor:
        """Train the model for one step."""
        self.optimizer.zero_grad()
        loss = self._compute_loss(batch)

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.max_grad_norm
        )
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self) -> None:
        """Train the model."""

        for epoch in range(self.num_epochs):
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
            if self.tracker is not None:
                self.model.load_state_dict(torch.load(self.checkpoint_model_path))
            return True
        return False

    def _update_scheduler(self, val_loss: float) -> None:
        """Update the scheduler."""
        self.scheduler.step(val_loss if self.scheduler_requires_loss else None)
        if np.abs(self.current_lr - self.scheduler.get_last_lr()[0]) > 1e-6:
            self.current_lr = self.scheduler.get_last_lr()[0]
            logger.info(f"Scheduler updated learning rate to {self.current_lr:.6f}")

    def _save_checkpoint(self, epoch: int) -> None:
        """Save the checkpoint if the early stopping condition is met."""
        if (self.early_stopping.save_checkpoint) and (self.tracker is not None):
            logger.info(f"Saving checkpoint at epoch {epoch}")
            torch.save(self.model.state_dict(), self.checkpoint_model_path)

    def _print_info(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """Print the information about the training and the tracking."""
        logger.info(
            f"Epoch {epoch}, Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Best Val Loss: {self.early_stopping.best_loss:.4f}, "
            f"Tracking: {self.tracker.name} - {self.tracker.url}"  # type: ignore[union-attr]
        )

    def _compute_val_loss(self) -> float:
        """Validate the model."""
        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in self.val_dataloader:
                batch = self._prepare_batch(batch)
                loss = self._compute_loss(batch)
                val_loss += loss.item()
            val_loss /= len(self.val_dataloader)
        return val_loss
