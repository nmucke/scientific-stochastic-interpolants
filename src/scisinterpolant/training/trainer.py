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

        for key, value in batch.items():
            batch[key] = value.to(self.device)

        t = torch.abs(torch.randn(batch["base"].shape[0], 1, device=self.device))
        noise = torch.randn(batch["base"].shape, device=self.device).to(self.device)

        drift, x_diff = self.model(
            base=batch["base"],
            target=batch["target"],
            t=t,
            noise=noise,
            field_cond=batch.get("field_cond", None),
            pars_cond=batch.get("pars_cond", None),
        )
        return self.loss_fn(drift, x_diff)

    def _train_step(self, batch: dict) -> torch.Tensor:
        """Train the model for one step."""
        self.optimizer.zero_grad()
        loss = self._compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self, verbose: bool = True) -> None:
        """Train the model."""
        self.model.train()

        for epoch in range(self.num_epochs):
            total_loss = 0
            self.model.train()
            pbar = tqdm(self.train_dataloader) if verbose else self.train_dataloader
            for batch in pbar:
                loss = self._train_step(batch)
                total_loss += loss.item()
                if verbose:
                    pbar.set_description(f"Epoch {epoch}, Loss: {loss:.4f}")

            val_loss = self._compute_val_loss()

            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

            if self.early_stopping.save_checkpoint:
                logger.info(f"Saving checkpoint at epoch {epoch}")
                torch.save(self.model.state_dict(), self.checkpoint_path)

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
