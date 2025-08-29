import logging

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for the stochastic interpolant."""

    def __init__(
        self,
        num_epochs: int,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: Optimizer,
        loss_fn: nn.Module = nn.MSELoss(),
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

            val_loss = self._val()

            total_loss /= len(self.train_dataloader)  # type: ignore[assignment]
            logger.info(
                f"Epoch {epoch}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    def _val(self) -> float:
        """Validate the model."""
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch in self.val_dataloader:
                loss = self._compute_loss(batch)
                total_loss += loss.item()
            total_loss /= len(self.val_dataloader)  # type: ignore[assignment]
        return total_loss
