import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer for the stochastic interpolant."""

    def __init__(
        self,
        num_epochs: int,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: Optimizer,
        loss_fn: nn.Module = nn.MSELoss(),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the trainer."""
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs

        self.model.train()
        self.model.to(self.device)

    def _train_step(self, batch: dict):
        """Train the model for one step."""
        self.model.train()
        self.optimizer.zero_grad()

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
        loss = self.loss_fn(drift, x_diff)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, verbose: bool = True):
        """Train the model."""
        self.model.train()

        for epoch in range(self.num_epochs):
            total_loss = 0
            pbar = tqdm(self.dataloader) if verbose else self.dataloader
            for batch in pbar:
                loss = self._train_step(batch)
                total_loss += loss
                if verbose:
                    pbar.set_description(f"Epoch {epoch}, Loss: {loss:.4f}")

            total_loss /= len(self.dataloader)
            logger.info(f"Epoch {epoch}, Loss: {total_loss:.4f}")

