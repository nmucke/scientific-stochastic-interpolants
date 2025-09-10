import logging
import pdb
from dataclasses import dataclass

import torch
import torch.nn as nn
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
        checkpoint_path: str | None = None,
        loss_fn: nn.Module = nn.MSELoss(),
        scheduler: LRScheduler = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_grad_norm: float = 1.0,
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
        self.checkpoint_path = checkpoint_path
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        if self.scheduler is not None:
            self.scheduler_requires_loss = (
                scheduler.__class__.__name__ in SCHEDULERS_THAT_REQUIRE_LOSS
            )
        else:
            self.scheduler_requires_loss = False

        self.model.train()
        self.model.to(self.device)

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        """Compute the loss for the model."""

        # Move data to device
        for key, value in batch.items():
            batch[key] = value.to(self.device)

        # Sample pseudo-time
        t = torch.abs(torch.randn(batch["base"].shape[0], 1, device=self.device))

        # Sample noise
        noise = torch.randn(batch["base"].shape, device=self.device)

        # Compute pred and true drift
        pred_drift, true_diff = self.model(
            **batch,
            t=t,
            noise=noise,
        )

        # Compute and return the loss
        return self.loss_fn(pred_drift, true_diff)

    def _train_step(self, batch: dict) -> torch.Tensor:
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

    def train(self, verbose: bool = True) -> None:
        """Train the model."""

        for epoch in range(self.num_epochs):
            total_loss = 0
            self.model.train()

            # Loop over batches
            pbar = tqdm(self.train_dataloader) if verbose else self.train_dataloader
            for batch in pbar:
                loss = self._train_step(batch)
                total_loss += loss.item()
                if verbose:
                    pbar.set_description(f"Epoch {epoch}, Loss: {loss:.4f}")

            # Compute validation loss
            val_loss = self._compute_val_loss()

            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                self.model.load_state_dict(torch.load(self.checkpoint_path))
                break

            # Save checkpoint
            if self.early_stopping.save_checkpoint:
                logger.info(f"Saving checkpoint at epoch {epoch}")
                torch.save(self.model.state_dict(), self.checkpoint_path)

            # Update scheduler
            if self.scheduler_requires_loss:
                self.scheduler.step(val_loss)
            if self.scheduler:
                self.scheduler.step()

            total_loss /= len(self.train_dataloader)  # type: ignore[assignment]
            logger.info(
                f"Epoch {epoch}, Train Loss: {total_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Best Val Loss: {self.early_stopping.best_loss:.4f}"
            )

    def _compute_val_loss(self) -> float:
        """Validate the model."""
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch in self.val_dataloader:
                loss = self._compute_loss(batch)
                total_loss += loss.item()
            total_loss /= len(self.val_dataloader)  # type: ignore[assignment]
        return total_loss
