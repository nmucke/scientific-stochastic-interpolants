"""Base trainer shared by the flow and map trainers.

Subclasses implement ``_compute_loss``; everything else (mixed precision,
gradient clipping, validation, early stopping, logging, checkpointing and
resuming) lives here.
"""

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class EarlyStopping:
    """Stop when the validation loss has not improved for `patience` epochs."""

    patience: int = 10
    delta: float = 1e-6
    best_loss: float = float("inf")
    counter: int = 0
    early_stop: bool = False
    save_checkpoint: bool = False

    def __call__(self, val_loss: float) -> bool:
        # Strict improvement; small regressions do not reset patience.
        self.save_checkpoint = val_loss < self.best_loss - self.delta
        if self.save_checkpoint:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        self.early_stop = self.counter >= self.patience
        return self.early_stop


class Trainer:
    """Training loop, validation and checkpointing; subclasses define the loss.

    Checkpoints in ``checkpoint_dir``:
      * ``model.pth``: best model weights (by validation loss),
      * ``last.pth``: full state after the last epoch, for ``resume_from``,
      * ``history.json``: per-epoch losses.
    """

    def __init__(
        self,
        num_epochs,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        gradient_clipper=None,
        early_stopping=None,
        loss_fn=None,
        scheduler=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tracker=None,
        mixed_precision_warmup=0,
        *,
        checkpoint_dir=None,
        max_train_batches=None,
        max_val_batches=None,
        resume_from=None,
    ):
        self.num_epochs = num_epochs
        self.device = device
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.gradient_clipper = gradient_clipper
        self.early_stopping = early_stopping or EarlyStopping()
        self.loss_fn = nn.MSELoss() if loss_fn is None else loss_fn
        # A constant schedule keeps the loop free of special cases.
        self.scheduler = scheduler or torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda _: 1.0
        )
        self.tracker = tracker
        self.max_train_batches = max_train_batches
        self.max_val_batches = max_val_batches
        self.resume_from = resume_from

        # bf16 autocast for the first `mixed_precision_warmup` epochs.
        self.mixed_precision_warmup = mixed_precision_warmup
        self.full_precision = (
            mixed_precision_warmup == 0 or torch.device(device).type != "cuda"
        )

        if checkpoint_dir is None:
            checkpoint_dir = (
                f"checkpoints/{tracker.project}/{tracker.name}"
                if tracker is not None
                else "checkpoints/model"
            )
        self.checkpoint_path = Path(checkpoint_dir)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_model_path = self.checkpoint_path / "model.pth"
        if tracker is not None:
            OmegaConf.save(tracker.config, self.checkpoint_path / "config.yaml")

        self.epoch = 0  # next epoch to run
        self.global_step = 0
        self.skipped_steps = 0  # updates skipped for a nonfinite gradient norm
        self.history = []
        self.components = {}  # named loss terms of the last _compute_loss call

    # ------------------------------------------------------------------ loss

    def _prepare_batch(self, batch):
        """Move to device and sample pseudo-time and noise (without mutating batch)."""
        batch = {k: v.to(self.device) for k, v in batch.items() if v is not None}
        batch["t"] = torch.rand(batch["base"].shape[0], 1, device=self.device)
        batch["noise"] = torch.randn_like(batch["base"])
        return batch

    def _compute_loss(self, batch):
        """Return the scalar loss; may store named terms in self.components."""
        raise NotImplementedError

    # ------------------------------------------------------------- training

    def _after_step(self):
        """Hook called after every optimizer step (e.g. EMA updates)."""

    def _train_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=torch.bfloat16,
            enabled=not self.full_precision,
        ):
            loss = self._compute_loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {self.global_step}")
        loss.backward()
        if self.gradient_clipper is not None:
            grad_norm = self.gradient_clipper.clip_grads(self.model.parameters())
            if grad_norm is not None and not torch.isfinite(torch.as_tensor(grad_norm)):
                # Skip the update (and the EMA) rather than apply a broken step.
                self.skipped_steps += 1
                logger.warning(
                    "Nonfinite gradient norm at step %d; update skipped",
                    self.global_step,
                )
                self.optimizer.zero_grad(set_to_none=True)
                return loss
        self.optimizer.step()
        self.global_step += 1
        self._after_step()
        return loss

    def _run_epoch(self, loader, max_batches, step_fn, description):
        """Average loss and its components over (at most `max_batches` of) a loader."""
        totals = {}
        count = 0
        pbar = tqdm(loader, total=max_batches, desc=description, leave=False)
        for index, batch in enumerate(pbar):
            if max_batches is not None and index >= max_batches:
                break
            batch = self._prepare_batch(batch)
            n = batch["base"].shape[0]
            loss = step_fn(batch).item()
            for key, value in {"loss": loss, **self.components}.items():
                totals[key] = totals.get(key, 0.0) + value * n
            count += n
            pbar.set_postfix(loss=f"{loss:.4f}")
        if not count:
            raise ValueError(f"{description} loader must yield at least one batch")
        return {key: value / count for key, value in totals.items()}

    def _validate(self):
        """Validation losses with fixed noise, so epochs are comparable."""
        self.model.eval()
        cuda_devices = (
            [torch.device(self.device).index or torch.cuda.current_device()]
            if torch.device(self.device).type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            torch.manual_seed(0)
            result = self._run_epoch(
                self.val_dataloader, self.max_val_batches, self._compute_loss, "Validation"
            )
        return result

    def _update_scheduler(self, val_loss):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def _log(self, row):
        if self.tracker is not None:
            # "step" is reserved by trackio; the epoch is its x-axis instead.
            metrics = {k: v for k, v in row.items() if k != "step"}
            self.tracker.log({**metrics, "global_step": row["step"]}, step=row["epoch"])
        logger.info(
            "Epoch %d | %s | best val %.4g",
            row["epoch"],
            ", ".join(
                f"{k} {v:.4g}" for k, v in row.items() if k not in ("epoch", "step")
            ),
            self.early_stopping.best_loss,
        )

    def train(self):
        if self.resume_from is not None:
            self.load_checkpoint(self.resume_from)

        for epoch in range(self.epoch, self.num_epochs):
            self.model.train()
            train = self._run_epoch(
                self.train_dataloader,
                self.max_train_batches,
                self._train_step,
                f"Epoch {epoch}",
            )
            if epoch + 1 >= self.mixed_precision_warmup:
                self.full_precision = True
            val = self._validate()

            row = {
                "epoch": epoch,
                "step": self.global_step,
                "lr": self.scheduler.get_last_lr()[0],
                **{f"train_{k}": v for k, v in train.items()},
                **{f"val_{k}": v for k, v in val.items()},
            }
            self.history.append(row)
            stop = self.early_stopping(row["val_loss"])
            self._update_scheduler(row["val_loss"])
            self._log(row)

            self.epoch = epoch + 1
            self.save_checkpoint()
            if stop:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        # Return with the best weights loaded.
        if self.checkpoint_model_path.exists():
            self.model.load_state_dict(
                torch.load(
                    self.checkpoint_model_path, map_location=self.device, weights_only=True
                )
            )
        logger.info("Training finished, best val loss %.4g", self.early_stopping.best_loss)
        return self.history

    # ---------------------------------------------------------- checkpoints

    def checkpoint_state(self):
        """Everything needed to resume training exactly where it stopped."""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "early_stopping": asdict(self.early_stopping),
            "gradient_clipper": (
                dict(vars(self.gradient_clipper))
                if self.gradient_clipper is not None
                else None
            ),
            "full_precision": self.full_precision,
            "history": self.history,
            "torch_rng_state": torch.random.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
            ),
        }

    def load_checkpoint_state(self, state):
        self.epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        for key, value in state["early_stopping"].items():
            setattr(self.early_stopping, key, value)
        if state["gradient_clipper"] is not None and self.gradient_clipper is not None:
            vars(self.gradient_clipper).update(state["gradient_clipper"])
        self.full_precision = state["full_precision"]
        self.history = state["history"]
        torch.random.set_rng_state(state["torch_rng_state"])
        if state["cuda_rng_states"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng_states"])

    def save_checkpoint(self):
        """Save the resumable state, the history and (if improved) the best model."""
        torch.save(self.checkpoint_state(), self.checkpoint_path / "last.pth")
        with open(self.checkpoint_path / "history.json", "w") as handle:
            json.dump(self.history, handle, indent=2)
        if self.early_stopping.save_checkpoint:
            torch.save(self.model.state_dict(), self.checkpoint_model_path)

    def load_checkpoint(self, path):
        """Resume from a ``last.pth``; `num_epochs` stays the total budget."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_checkpoint_state(state)
        # Keep the best model of the earlier run if we write somewhere new.
        best = Path(path).parent / "model.pth"
        if best.exists() and not self.checkpoint_model_path.exists():
            shutil.copy2(best, self.checkpoint_model_path)
        logger.info("Resumed from %s at epoch %d, step %d", path, self.epoch, self.global_step)
