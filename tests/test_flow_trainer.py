"""Lifecycle of the base Trainer through FlowTrainer: checkpoints and resuming."""

import json

import torch
from torch import nn

from scisi.training.flow_trainer import FlowTrainer
from scisi.training.gradient_clipping import EmaGradientClipper
from scisi.training.trainer import EarlyStopping


class ConstantRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0))

    def forward(self, base, target, t, noise, **kwargs):
        return self.value.expand_as(target), target + 0.1 * noise + 0.1 * t.reshape(
            -1, 1, 1, 1
        )


def build_trainer(directory, epochs=2, resume=None):
    model = ConstantRegression()
    batch = {"base": torch.zeros(2, 3, 4, 4), "target": torch.ones(2, 3, 4, 4)}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    return FlowTrainer(
        num_epochs=epochs,
        model=model,
        optimizer=optimizer,
        train_dataloader=[batch.copy() for _ in range(3)],
        val_dataloader=[batch.copy()],
        gradient_clipper=EmaGradientClipper(),
        early_stopping=EarlyStopping(patience=10),
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5),
        checkpoint_dir=directory,
        max_train_batches=2,
        device="cpu",
        resume_from=resume,
    )


def test_validation_is_fixed_and_leaves_rng_untouched(tmp_path):
    trainer = build_trainer(tmp_path)
    rng = torch.random.get_rng_state()
    assert trainer._validate() == trainer._validate()
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_checkpoint_contains_full_training_state(tmp_path):
    trainer = build_trainer(tmp_path)
    trainer.train()
    assert trainer.global_step == 4
    assert {"model.pth", "last.pth", "history.json"}.issubset(
        p.name for p in tmp_path.iterdir()
    )
    state = torch.load(tmp_path / "last.pth", weights_only=True)
    assert state["epoch"] == 2 and state["global_step"] == 4
    for key in ("model", "optimizer", "scheduler", "early_stopping",
                "gradient_clipper", "history", "torch_rng_state"):
        assert key in state
    assert json.loads((tmp_path / "history.json").read_text()) == trainer.history


def test_resume_matches_uninterrupted_training(tmp_path):
    torch.manual_seed(42)
    uninterrupted = build_trainer(tmp_path / "all")
    uninterrupted.train()

    torch.manual_seed(42)
    build_trainer(tmp_path / "first", epochs=1).train()
    torch.rand(13)  # unrelated draws must not change the continuation
    resumed = build_trainer(tmp_path / "resume", resume=tmp_path / "first" / "last.pth")
    resumed.train()

    torch.testing.assert_close(
        uninterrupted.model.value, resumed.model.value, rtol=0, atol=0
    )
    assert uninterrupted.history == resumed.history
    assert (
        uninterrupted.gradient_clipper.ema_index == resumed.gradient_clipper.ema_index
    )
    assert uninterrupted.scheduler.get_last_lr() == resumed.scheduler.get_last_lr()
    assert (tmp_path / "resume" / "model.pth").exists()


def test_regressed_validation_is_not_an_improvement():
    stopping = EarlyStopping(patience=3)
    stopping(0.5)
    assert stopping.save_checkpoint
    stopping(0.5 + 1e-7)
    assert not stopping.save_checkpoint and stopping.best_loss == 0.5
