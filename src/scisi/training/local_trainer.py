"""Offline training of existing drift models with reproducible validation."""

import json
import logging
import time
from pathlib import Path

import torch
from torch import nn

from scisi.training.trainer import Trainer

logger = logging.getLogger(__name__)


class LocalTrainer(Trainer):
    """Keep Trainer's regression objective and add local experiment artifacts.

    Used for new video teachers without requiring an external tracking account.
    Validation uses a fixed random stream and restores the training RNG. Limits
    count optimizer updates through bounded epochs; they do not alter the data
    objective. The original Trainer and existing scientific runs are unchanged.
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
        loss_fn=None,
        scheduler=None,
        device="cpu",
        tracker=None,
        mixed_precision_warmup=0,
        *,
        checkpoint_dir="checkpoints/model",
        max_train_batches=None,
        max_val_batches=None,
        validation_seed=1937,
        resume_from=None,
    ):
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        super().__init__(
            num_epochs,
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            gradient_clipper,
            early_stopping,
            nn.MSELoss() if loss_fn is None else loss_fn,
            scheduler,
            device,
            tracker,
            mixed_precision_warmup,
        )
        self.checkpoint_path = Path(checkpoint_dir)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_model_path = self.checkpoint_path / "model.pth"
        for name, limit in [
            ("max_train_batches", max_train_batches),
            ("max_val_batches", max_val_batches),
        ]:
            if limit is not None and limit < 1:
                raise ValueError(f"{name} must be positive or None")
        self.max_train_batches = max_train_batches
        self.max_val_batches = max_val_batches
        self.validation_seed = int(validation_seed)
        self.global_step = 0
        self.start_epoch = 0
        self.history = []
        self.start_time = time.perf_counter()
        if resume_from is not None:
            self._restore_checkpoint(resume_from)

    def _restore_checkpoint(self, path):
        # Restart state is an explicitly selected local training artifact.
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        self.start_epoch = state["epoch"] + 1
        self.history = state["history"]
        self.early_stopping.__dict__.update(state["early_stopping"])
        self.full_precision = state["full_precision"]
        if self.gradient_clipper is not None and state["gradient_clipper"] is not None:
            self.gradient_clipper.__dict__.update(state["gradient_clipper"])
        if hasattr(self, "scaler") and state["scaler"] is not None:
            self.scaler.load_state_dict(state["scaler"])
        torch.random.set_rng_state(state["torch_rng_state"].cpu())
        if state["cuda_rng_states"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng_states"]])
        self.start_time -= state["elapsed_seconds"]
        # Best weights must survive resuming into a new directory.
        source_best = Path(path).parent / "model.pth"
        if (
            source_best.exists()
            and source_best.resolve() != self.checkpoint_model_path.resolve()
        ):
            import shutil

            shutil.copy2(source_best, self.checkpoint_model_path)

    def _local_train_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        use_amp = not self.full_precision
        with torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            loss = self._compute_loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite drift loss at step {self.global_step}")
        if use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        if self.gradient_clipper is not None:
            self.gradient_clipper.clip_grads(self.model.parameters())
        if use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.global_step += 1
        return loss

    def _compute_val_loss(self):
        self.model.eval()
        device = torch.device(self.device)
        devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        total, count = 0.0, 0
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(self.validation_seed)
            for index, batch in enumerate(self.val_dataloader):
                if self.max_val_batches is not None and index >= self.max_val_batches:
                    break
                batch = self._prepare_batch(batch)
                loss = self._compute_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite validation loss")
                n = batch["base"].shape[0]
                total += float(loss) * n
                count += n
        if not count:
            raise ValueError("Validation loader must yield at least one batch")
        return total / count

    def _save_checkpoint(self, epoch):
        elapsed = time.perf_counter() - self.start_time
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": epoch,
                "global_step": self.global_step,
                "history": self.history,
                "early_stopping": vars(self.early_stopping).copy(),
                "gradient_clipper": (
                    vars(self.gradient_clipper).copy()
                    if self.gradient_clipper is not None
                    else None
                ),
                "full_precision": self.full_precision,
                "scaler": self.scaler.state_dict() if hasattr(self, "scaler") else None,
                "torch_rng_state": torch.random.get_rng_state(),
                "cuda_rng_states": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_initialized()
                    else None
                ),
                "elapsed_seconds": elapsed,
            },
            self.checkpoint_path / "last.pth",
        )
        if self.early_stopping.save_checkpoint:
            torch.save(self.model.state_dict(), self.checkpoint_model_path)

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.model.train()
            total, count = 0.0, 0
            for index, batch in enumerate(self.train_dataloader):
                if (
                    self.max_train_batches is not None
                    and index >= self.max_train_batches
                ):
                    break
                batch = self._prepare_batch(batch)
                loss = self._local_train_step(batch)
                n = batch["base"].shape[0]
                total += float(loss.detach()) * n
                count += n
            if not count:
                raise ValueError("Training loader must yield at least one batch")
            if epoch >= self.mixed_precision_warmup:
                self.full_precision = True
            train_loss, val_loss = total / count, self._compute_val_loss()
            stop = self.early_stopping
            stop.save_checkpoint = val_loss < stop.best_loss - stop.delta
            if stop.save_checkpoint:
                stop.best_loss, stop.counter = val_loss, 0
            else:
                stop.counter += 1
            stop.early_stop = stop.counter >= stop.patience
            self._update_scheduler(val_loss)
            self._log_with_tracker(epoch, train_loss, val_loss)
            self.history.append(
                {
                    "epoch": epoch,
                    "step": self.global_step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "elapsed_seconds": time.perf_counter() - self.start_time,
                }
            )
            self._save_checkpoint(epoch)
            (self.checkpoint_path / "history.json").write_text(
                json.dumps(self.history, indent=2)
            )
            logger.info(
                "Epoch %d step %d train %.6g validation %.6g best %.6g",
                epoch,
                self.global_step,
                train_loss,
                val_loss,
                stop.best_loss,
            )
            if stop.early_stop:
                break
        if self.checkpoint_model_path.exists():
            self.model.load_state_dict(
                torch.load(
                    self.checkpoint_model_path,
                    map_location=self.device,
                    weights_only=True,
                )
            )
        return self.history
