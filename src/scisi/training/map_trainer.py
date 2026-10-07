"""Training for conditional deterministic and stochastic finite-time maps.

The default objective is the diagonal regression plus velocity-form progressive
self-distillation (Pan et al., arXiv:2606.11156, Algorithm 3).  ``teacher_rollout``
is an explicitly simulation-based alternative, useful for distillation and for
checking that consistency alone has not converged to incorrect dynamics.
"""

import copy
import json
import logging
import shutil
import time
from pathlib import Path

import torch
from torch import nn

from scisi.training.trainer import Trainer

logger = logging.getLogger(__name__)


class MapTrainer(Trainer):
    """Reuse Trainer's optimizer, clipping and epoch lifecycle for map losses."""

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
        device="cuda" if torch.cuda.is_available() else "cpu",
        tracker=None,
        mixed_precision_warmup=0,
        *,
        teacher=None,
        objective="progressive",
        diagonal_weight=1.0,
        consistency_weight=1.0,
        ema_decay=0.99,
        diagonal_warmup_steps=0,
        consistency_ramp_steps=500,
        interval_curriculum_steps=1000,
        initial_max_interval=0.125,
        endpoint_probability=0.25,
        teacher_steps=16,
        checkpoint_dir="checkpoints/map_model",
        max_train_batches=None,
        max_val_batches=None,
        validation_seed=1937,
        resume_from=None,
    ):
        if objective not in {"progressive", "teacher_rollout"}:
            raise ValueError("objective must be progressive or teacher_rollout")
        if not 0 <= ema_decay < 1 or not 0 <= endpoint_probability <= 1:
            raise ValueError("Invalid EMA decay or endpoint probability")
        if not 0 < initial_max_interval <= 1 or teacher_steps < 1:
            raise ValueError("Invalid interval curriculum or teacher step count")
        if (
            min(
                diagonal_weight,
                consistency_weight,
                diagonal_warmup_steps,
                consistency_ramp_steps,
                interval_curriculum_steps,
            )
            < 0
        ):
            raise ValueError("Loss weights and warmup durations must be nonnegative")
        # Trainer currently assumes a scheduler; a constant one preserves that
        # lifecycle when the caller deliberately requests no scheduling.
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        super().__init__(
            num_epochs=num_epochs,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            gradient_clipper=gradient_clipper,
            early_stopping=early_stopping,
            loss_fn=nn.MSELoss() if loss_fn is None else loss_fn,
            scheduler=scheduler,
            device=device,
            tracker=tracker,
            mixed_precision_warmup=mixed_precision_warmup,
        )
        self.teacher = teacher
        if teacher is not None:
            teacher.to(self.device).eval().requires_grad_(False)
        if objective == "teacher_rollout" and teacher is None:
            raise ValueError("teacher_rollout requires a frozen teacher")
        self.objective = objective
        self.diagonal_weight = float(diagonal_weight)
        self.consistency_weight = float(consistency_weight)
        self.ema_decay = float(ema_decay)
        self.diagonal_warmup_steps = int(diagonal_warmup_steps)
        self.consistency_ramp_steps = int(consistency_ramp_steps)
        self.interval_curriculum_steps = int(interval_curriculum_steps)
        self.initial_max_interval = float(initial_max_interval)
        self.endpoint_probability = float(endpoint_probability)
        self.teacher_steps = int(teacher_steps)
        self.max_train_batches = max_train_batches
        self.max_val_batches = max_val_batches
        self.validation_seed = int(validation_seed)
        self.global_step = 0
        self.start_epoch = 0
        self._validating = False
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.checkpoint_path = str(Path(checkpoint_dir))
        Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)
        self.checkpoint_model_path = str(Path(checkpoint_dir) / "model.pth")
        self.components = {}
        self.history = []
        self.start_time = time.perf_counter()
        self._train_step = self._map_train_step
        if resume_from is not None:
            self._restore_checkpoint(resume_from)

    def _loader_generators(self):
        """Explicit loader generators are independent of the global torch RNG."""
        generators = {}
        for name, loader in (
            ("train", self.train_dataloader),
            ("val", self.val_dataloader),
        ):
            for part, obj in (
                ("loader", loader),
                ("sampler", getattr(loader, "sampler", None)),
            ):
                generator = getattr(obj, "generator", None)
                if isinstance(generator, torch.Generator):
                    generators[f"{name}_{part}"] = generator
        return generators

    def _restore_checkpoint(self, filename):
        """Resume an epoch-boundary last.pth; num_epochs remains the total budget.

        Worker-local states in persistent DataLoader workers are not restorable
        here. Exact stochastic replay assumes deterministic datasets or workers
        that are recreated and seeded at each epoch.
        """
        source = Path(filename)
        state = torch.load(source, map_location="cpu", weights_only=True)
        required = {
            "model",
            "ema_model",
            "optimizer",
            "scheduler",
            "global_step",
            "epoch",
        }
        if not required.issubset(state):
            raise ValueError(
                "resume_from must refer to a full MapTrainer last.pth checkpoint"
            )
        if state.get("objective", self.objective) != self.objective:
            raise ValueError("Cannot resume with a different map training objective")
        self.model.load_state_dict(state["model"])
        self.ema_model.load_state_dict(state["ema_model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.current_lr = self.scheduler.get_last_lr()[0]
        self.global_step = int(state["global_step"])
        self.start_epoch = int(state["epoch"]) + 1
        early_state = state.get(
            "early_stopping", {"best_loss": state.get("best_loss", float("inf"))}
        )
        for name, value in early_state.items():
            if hasattr(self.early_stopping, name):
                setattr(self.early_stopping, name, value)
        clipping = state.get("gradient_clipper")
        if clipping is not None:
            if self.gradient_clipper is None:
                raise ValueError(
                    "Checkpoint contains clipping state but gradient_clipper is None"
                )
            clip_class = f"{type(self.gradient_clipper).__module__}.{type(self.gradient_clipper).__qualname__}"
            if clipping["class"] != clip_class:
                raise ValueError(
                    "Checkpoint gradient clipper does not match configured clipper"
                )
            values = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in clipping["state"].items()
            }
            if clipping["format"] == "state_dict":
                self.gradient_clipper.load_state_dict(values)
            else:
                self.gradient_clipper.__dict__.update(values)
        self.full_precision = state.get("full_precision", self.full_precision)
        if state.get("scaler") is not None:
            if not hasattr(self, "scaler"):
                self.scaler = torch.amp.GradScaler(torch.device(self.device).type)
            self.scaler.load_state_dict(state["scaler"])
        self.history = state.get("history", [])
        if not self.history and (source.parent / "history.json").exists():
            old_history = json.loads((source.parent / "history.json").read_text())
            self.history = [
                row for row in old_history if row["epoch"] < self.start_epoch
            ]
        self.start_time = time.perf_counter() - float(state.get("elapsed_seconds", 0.0))
        # Retain the previous best checkpoint even if no subsequent epoch improves.
        destination = Path(self.checkpoint_path)
        if source.parent.resolve() != destination.resolve():
            for name in ("model.pth", "ema.pth"):
                if (source.parent / name).exists():
                    shutil.copy2(source.parent / name, destination / name)
        if self.history:
            (destination / "history.json").write_text(
                json.dumps(self.history, indent=2)
            )
        for name, generator in self._loader_generators().items():
            if name in state.get("loader_rng_states", {}):
                generator.set_state(state["loader_rng_states"][name])
        if "torch_rng_state" in state:
            torch.random.set_rng_state(state["torch_rng_state"])
        if (
            state.get("cuda_rng_states") is not None
            and torch.device(self.device).type == "cuda"
        ):
            torch.cuda.set_rng_state_all(state["cuda_rng_states"])
        logger.info(
            "Resumed %s at epoch %d, step %d",
            source,
            self.start_epoch,
            self.global_step,
        )

    def _prepare_batch(self, batch):
        # Preserve caller-owned dictionaries and the existing dataset contract.
        batch = {k: v.to(self.device) for k, v in batch.items() if v is not None}
        batch["t"] = torch.rand(batch["base"].shape[0], 1, device=self.device)
        batch["noise"] = torch.randn_like(batch["base"])
        return batch

    def _intervals(self, batch_size):
        step = max(0, self.global_step - self.diagonal_warmup_steps)
        progress = min(1.0, step / max(1, self.interval_curriculum_steps))
        if self.interval_curriculum_steps == 0 or self._validating:
            progress = 1.0
        max_interval = self.initial_max_interval + progress * (
            1 - self.initial_max_interval
        )
        # Uniform length within the curriculum, with an explicit endpoint atom.
        length = max_interval * (
            0.01 + 0.99 * torch.rand(batch_size, 1, device=self.device)
        )
        s = torch.rand_like(length) * (1 - length)
        t = s + length
        endpoint = torch.rand_like(length) < self.endpoint_probability
        s = torch.where(endpoint, torch.zeros_like(s), s)
        t = torch.where(endpoint, torch.full_like(t, max_interval), t)
        u = s + (t - s) * (0.25 + 0.5 * torch.rand_like(s))
        return s, u, t

    def _compute_loss(self, batch):
        conditions = {
            k: batch[k]
            for k in ("field_history", "field_cond", "pars_cond")
            if k in batch
        }
        x_diag, target_diag = self.model.training_pair(
            batch["base"], batch["target"], batch["t"], batch["noise"]
        )
        if self.teacher is not None:
            with torch.no_grad():
                target_diag = self.teacher.drift(x_diag, batch["t"], **conditions)
        diagonal = self.loss_fn(
            self.model.average_velocity(x_diag, batch["t"], batch["t"], **conditions),
            target_diag,
        )
        weight = self.consistency_weight
        if self._validating:
            pass
        elif self.global_step < self.diagonal_warmup_steps:
            weight = 0.0
        elif self.consistency_ramp_steps:
            weight *= min(
                1.0,
                (self.global_step - self.diagonal_warmup_steps + 1)
                / self.consistency_ramp_steps,
            )
        consistency = diagonal.new_zeros(())
        if weight > 0:
            s, u, t = self._intervals(batch["base"].shape[0])
            x, _ = self.model.training_pair(
                batch["base"], batch["target"], s, torch.randn_like(batch["base"])
            )
            path = self.model.make_brownian_path(x)
            with torch.no_grad():
                if self.objective == "progressive" and not (
                    self._validating and self.teacher is not None
                ):
                    g1 = self.ema_model.average_velocity(
                        x, s, u, brownian_path=path, **conditions
                    )
                    mid = x + (u - s).reshape(-1, 1, 1, 1) * g1
                    if path is not None:
                        mid = mid + path.increment(s, u)
                    g2 = self.ema_model.average_velocity(
                        mid, u, t, brownian_path=path, **conditions
                    )
                    fraction = ((u - s) / (t - s)).reshape(-1, 1, 1, 1)
                    target = fraction * g1 + (1 - fraction) * g2
                else:
                    endpoint = x.clone()
                    dt = (t - s) / self.teacher_steps
                    for j in range(self.teacher_steps):
                        left = s + j * dt
                        right = t if j == self.teacher_steps - 1 else s + (j + 1) * dt
                        endpoint = endpoint + dt.reshape(
                            -1, 1, 1, 1
                        ) * self.teacher.drift(endpoint, left, **conditions)
                        if path is not None:
                            endpoint = endpoint + path.increment(left, right)
                    displacement = endpoint - x
                    if path is not None:
                        displacement = displacement - path.increment(s, t)
                    target = displacement / (t - s).reshape(-1, 1, 1, 1)
            prediction = self.model.average_velocity(
                x, s, t, brownian_path=path, **conditions
            )
            consistency = self.loss_fn(prediction, target)
        loss = self.diagonal_weight * diagonal + weight * consistency
        self.components = {
            "diagonal_loss": float(diagonal.detach()),
            "consistency_loss": float(consistency.detach()),
            "consistency_weight": weight,
        }
        return loss

    def _map_train_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        # Full precision by default; preserve Trainer's warmup interpretation.
        use_amp = not self.full_precision
        with torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            loss = self._compute_loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite map loss at step {self.global_step}")
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
        with torch.no_grad():
            for average, current in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                average.lerp_(current, 1 - self.ema_decay)
            for average, current in zip(self.ema_model.buffers(), self.model.buffers()):
                average.copy_(current)
        return loss

    def _compute_val_loss(self):
        self.model.eval()
        self._validating = True
        total = 0.0
        count = 0
        # Fixed validation noise makes successive checkpoint comparisons useful.
        cuda_devices = (
            [torch.device(self.device).index or torch.cuda.current_device()]
            if torch.device(self.device).type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            torch.manual_seed(self.validation_seed)
            for index, batch in enumerate(self.val_dataloader):
                if self.max_val_batches is not None and index >= self.max_val_batches:
                    break
                batch = self._prepare_batch(batch)
                n = batch["base"].shape[0]
                total += float(self._compute_loss(batch)) * n
                count += n
        if not count:
            raise ValueError("Validation loader must yield at least one batch")
        self._validating = False
        return total / count

    def _check_early_stopping(self, val_loss, epoch):
        # Correct minimization criterion; do not count small regressions as gains.
        state = self.early_stopping
        state.save_checkpoint = val_loss < state.best_loss - state.delta
        if state.save_checkpoint:
            state.best_loss = val_loss
            state.counter = 0
        else:
            state.counter += 1
        state.early_stop = state.counter >= state.patience
        return state.early_stop

    def _save_checkpoint(self, epoch):
        clipping = None
        if self.gradient_clipper is not None:
            has_state_dict = callable(
                getattr(self.gradient_clipper, "state_dict", None)
            )
            clipping = {
                "class": f"{type(self.gradient_clipper).__module__}.{type(self.gradient_clipper).__qualname__}",
                "format": "state_dict" if has_state_dict else "attributes",
                "state": (
                    self.gradient_clipper.state_dict()
                    if has_state_dict
                    else vars(self.gradient_clipper).copy()
                ),
            }
        payload = {
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": epoch,
            "best_loss": self.early_stopping.best_loss,
            "objective": self.objective,
            "elapsed_seconds": time.perf_counter() - self.start_time,
            "early_stopping": vars(self.early_stopping).copy(),
            "gradient_clipper": clipping,
            "full_precision": self.full_precision,
            "scaler": self.scaler.state_dict() if hasattr(self, "scaler") else None,
            "history": self.history,
            "torch_rng_state": torch.random.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
            ),
            "loader_rng_states": {
                name: generator.get_state()
                for name, generator in self._loader_generators().items()
            },
        }
        torch.save(payload, Path(self.checkpoint_path) / "last.pth")
        if self.early_stopping.save_checkpoint:
            torch.save(self.model.state_dict(), self.checkpoint_model_path)
            torch.save(
                self.ema_model.state_dict(), Path(self.checkpoint_path) / "ema.pth"
            )

    def _print_info(self, epoch, train_loss, val_loss):
        logger.info(
            "Epoch %d step %d train %.6g validation %.6g best %.6g",
            epoch,
            self.global_step,
            train_loss,
            val_loss,
            self.early_stopping.best_loss,
        )

    def train(self):
        # Same Trainer lifecycle, with bounded pilot runs and offline checkpoints.
        for epoch in range(self.start_epoch, self.num_epochs):
            self.model.train()
            total = 0.0
            count = 0
            for index, batch in enumerate(self.train_dataloader):
                if (
                    self.max_train_batches is not None
                    and index >= self.max_train_batches
                ):
                    break
                batch = self._prepare_batch(batch)
                loss = self._map_train_step(batch)
                n = batch["base"].shape[0]
                total += float(loss.detach()) * n
                count += n
            if not count:
                raise ValueError("Training loader must yield at least one batch")
            if epoch >= self.mixed_precision_warmup:
                self.full_precision = True
            train_loss = total / count
            val_loss = self._compute_val_loss()
            self._log_with_tracker(epoch, train_loss, val_loss)
            stop = self._check_early_stopping(val_loss, epoch)
            self._update_scheduler(val_loss)
            self._print_info(epoch, train_loss, val_loss)
            row = {
                "epoch": epoch,
                "step": self.global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "elapsed_seconds": time.perf_counter() - self.start_time,
                **self.components,
            }
            self.history.append(row)
            self._save_checkpoint(epoch)
            with open(Path(self.checkpoint_path) / "history.json", "w") as handle:
                json.dump(self.history, handle, indent=2)
            if stop:
                break
        if Path(self.checkpoint_model_path).exists():
            self.model.load_state_dict(
                torch.load(
                    self.checkpoint_model_path,
                    map_location=self.device,
                    weights_only=True,
                )
            )
        return self.history
