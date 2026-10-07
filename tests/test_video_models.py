"""Pixel-space model contracts and offline teacher lifecycle."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch import nn

from scisi.training.local_trainer import LocalTrainer
from scisi.training.trainer import EarlyStopping

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        return compose(config_name=name)


@pytest.mark.parametrize("dataset,channels", [("kth", 1), ("clevrer", 3)])
@pytest.mark.parametrize("stochastic", [False, True])
def test_video_teacher_map_rgb_history_and_initialization(
    dataset, channels, stochastic
):
    torch.set_num_threads(1)
    family = "follmer_stochastic_interpolant" if stochastic else "flow_matching"
    student_family = "follmer_ito_map" if stochastic else "flow_map"
    teacher_cfg, student_cfg = config(f"{family}_{dataset}"), config(
        f"{student_family}_{dataset}"
    )
    teacher, student = instantiate(teacher_cfg.model), instantiate(student_cfg.model)
    student.map_model.load_from_teacher(teacher)
    history = torch.randn(2, channels, 16, 16, teacher_cfg.len_field_history)
    x, target, t = (
        history[..., -1],
        torch.randn_like(history[..., -1]),
        torch.full((2, 1), 0.4),
    )
    drift = teacher.drift(x, t, field_history=history)
    diagonal = student.average_velocity(x, t, t, field_history=history)
    torch.testing.assert_close(diagonal, drift, rtol=0, atol=0)
    prediction, truth = teacher(
        base=x, target=target, t=t, noise=torch.randn_like(x), field_history=history
    )
    assert prediction.shape == truth.shape == (2, channels, 16, 16)
    ((prediction - truth) ** 2).mean().backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in teacher.parameters()
    )
    with torch.no_grad():
        for model in (teacher, student):
            kwargs = {"base": x[:1]} if stochastic else {}
            sample, new_history = model.sample(
                field_history=history[:1],
                batch_size=2,
                num_steps=2,
                return_field_history=True,
                **kwargs,
            )
            assert sample.shape == (2, channels, 16, 16)
            assert new_history.shape == history.shape
            torch.testing.assert_close(new_history[..., -1], sample)
            torch.testing.assert_close(
                new_history[..., :-1],
                history[:1, ..., 1:].expand_as(new_history[..., :-1]),
            )
            assert torch.isfinite(sample).all()


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
    return LocalTrainer(
        num_epochs=epochs,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
        train_dataloader=[batch.copy() for _ in range(3)],
        val_dataloader=[batch.copy()],
        gradient_clipper=None,
        early_stopping=EarlyStopping(patience=10),
        scheduler=None,
        checkpoint_dir=directory,
        max_train_batches=2,
        device="cpu",
        resume_from=resume,
    )


def test_local_teacher_checkpoints_validation_rng_and_resume(tmp_path):
    torch.manual_seed(42)
    uninterrupted = build_trainer(tmp_path / "all")
    rng = torch.random.get_rng_state()
    first, second = uninterrupted._compute_val_loss(), uninterrupted._compute_val_loss()
    assert first == second
    assert torch.equal(torch.random.get_rng_state(), rng)
    uninterrupted.train()
    assert uninterrupted.global_step == 4
    assert {"model.pth", "last.pth", "history.json"}.issubset(
        p.name for p in (tmp_path / "all").iterdir()
    )
    torch.manual_seed(42)
    first_half = build_trainer(tmp_path / "first", epochs=1)
    first_half.train()
    resumed = build_trainer(tmp_path / "resume", resume=tmp_path / "first" / "last.pth")
    resumed.train()
    torch.testing.assert_close(
        uninterrupted.model.value, resumed.model.value, rtol=0, atol=0
    )
    assert [r["val_loss"] for r in uninterrupted.history] == [
        r["val_loss"] for r in resumed.history
    ]
