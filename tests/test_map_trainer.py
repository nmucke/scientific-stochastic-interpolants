"""Analytic dynamics and offline lifecycle checks for MapTrainer."""

import json

import pytest
import torch
from torch import nn

from scisi.models.flow_map_model import FlowMapModel
from scisi.models.follmer_ito_map_model import FollmerItoMapModel
from scisi.models.interpolations import (
    LinearDeterministicInterpolation,
    QuadraticStochasticInterpolation,
)
from scisi.training.map_trainer import MapTrainer
from scisi.training.trainer import EarlyStopping


class ConstantMapNetwork(nn.Module):
    def __init__(self, value=0.4):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def forward(self, x, cond, **kwargs):
        return self.value.expand_as(x)


class ConstantTeacher(nn.Module):
    def __init__(self, value=0.4):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def drift(self, x, t, **kwargs):
        return self.value.expand_as(x)


def batch():
    generator = torch.Generator().manual_seed(11)
    base = torch.randn(3, 1, 2, 2, generator=generator)
    return {
        "base": base,
        "target": base + torch.randn(base.shape, generator=generator),
        "field_history": base.unsqueeze(-1),
    }


def trainer(tmp_path, stochastic, objective="progressive", value=0.4, **kwargs):
    network = ConstantMapNetwork(value)
    if stochastic:
        model = FollmerItoMapModel(
            QuadraticStochasticInterpolation(gamma_multiplier=0.3), network
        )
    else:
        model = FlowMapModel(LinearDeterministicInterpolation(), network)
    return MapTrainer(
        num_epochs=kwargs.pop("num_epochs", 1),
        model=model,
        teacher=ConstantTeacher(),
        train_dataloader=[batch(), batch()],
        val_dataloader=[batch()],
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9),
        scheduler=None,
        tracker=None,
        gradient_clipper=kwargs.pop("gradient_clipper", None),
        early_stopping=EarlyStopping(patience=3),
        device="cpu",
        objective=objective,
        checkpoint_dir=str(tmp_path),
        consistency_ramp_steps=0,
        interval_curriculum_steps=0,
        teacher_steps=5,
        **kwargs,
    )


@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("objective", ["progressive", "teacher_rollout"])
def test_constant_drift_is_exact_for_shared_path_and_teacher_rollout(
    tmp_path, stochastic, objective
):
    torch.manual_seed(7)
    instance = trainer(tmp_path, stochastic, objective)
    loss = instance._compute_loss(instance._prepare_batch(batch()))
    assert loss.item() < 1e-10
    assert instance.components["consistency_loss"] < 1e-10


@pytest.mark.parametrize("stochastic", [False, True])
def test_optimizer_updates_student_without_teacher_or_ema_gradients(
    tmp_path, stochastic
):
    instance = trainer(tmp_path, stochastic, "teacher_rollout", value=0.0)
    original = instance.model.map_model.value.detach().clone()
    loss = instance._map_train_step(instance._prepare_batch(batch()))
    assert torch.isfinite(loss)
    assert instance.model.map_model.value.item() > original.item()
    assert instance.global_step == 1
    assert instance.model.map_model.value.grad is not None
    assert all(
        not p.requires_grad and p.grad is None for p in instance.teacher.parameters()
    )
    assert all(
        not p.requires_grad and p.grad is None for p in instance.ema_model.parameters()
    )
    assert instance.ema_model.map_model.value.item() > original.item()


def test_offline_training_saves_student_ema_and_restart_state(tmp_path):
    instance = trainer(tmp_path, True, "teacher_rollout", value=0.0)
    original_batch = batch()
    instance._prepare_batch(original_batch)
    assert "t" not in original_batch and "noise" not in original_batch
    history = instance.train()
    assert len(history) == 1 and history[0]["step"] == 2
    assert all(
        (tmp_path / name).exists()
        for name in ("model.pth", "ema.pth", "last.pth", "history.json")
    )
    state = torch.load(tmp_path / "last.pth", map_location="cpu", weights_only=True)
    assert state["global_step"] == 2
    assert "optimizer" in state and "scheduler" in state and "ema_model" in state
    assert json.loads((tmp_path / "history.json").read_text())[0]["step"] == 2
    # Fixed validation draws give identical loss without perturbing caller RNG.
    rng = torch.random.get_rng_state().clone()
    a, b = instance._compute_val_loss(), instance._compute_val_loss()
    assert a == b
    assert torch.equal(rng, torch.random.get_rng_state())


def test_regressed_validation_is_not_checkpoint_improvement(tmp_path):
    instance = trainer(tmp_path, False)
    instance._check_early_stopping(0.5, 0)
    assert instance.early_stopping.save_checkpoint
    instance._check_early_stopping(0.5 + 1e-7, 1)
    assert not instance.early_stopping.save_checkpoint
    assert instance.early_stopping.best_loss == 0.5


@pytest.mark.parametrize("stochastic", [False, True])
def test_teacher_rollout_matches_analytic_left_endpoint_quadrature(
    tmp_path, stochastic
):
    """For b(t,x)=t, Euler's finite-step mean drift is known exactly."""
    instance = trainer(tmp_path, stochastic, "teacher_rollout")

    class TimeTeacher(ConstantTeacher):
        def drift(self, x, t, **kwargs):
            return t.reshape(-1, 1, 1, 1).expand_as(x)

    class TimeMap(ConstantMapNetwork):
        def forward(self, x, cond, **kwargs):
            s, t = cond[:, :1], cond[:, 1:]
            mean = s + (t - s) * (instance.teacher_steps - 1) / (
                2 * instance.teacher_steps
            )
            return mean.reshape(-1, 1, 1, 1).expand_as(x) + 0 * self.value

    instance.teacher = TimeTeacher().requires_grad_(False)
    instance.model.map_model = TimeMap()
    loss = instance._compute_loss(instance._prepare_batch(batch()))
    assert loss.item() < 1e-10


def test_constructor_accepts_trainer_positional_arguments(tmp_path):
    model = FlowMapModel(LinearDeterministicInterpolation(), ConstantMapNetwork())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    instance = MapTrainer(
        1,
        model,
        [batch()],
        [batch()],
        optimizer,
        None,
        EarlyStopping(),
        nn.MSELoss(),
        None,
        "cpu",
        None,
        0,
        checkpoint_dir=str(tmp_path),
        consistency_weight=0,
    )
    instance.train()
    assert instance.global_step == 1


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_resume_matches_uninterrupted_training_and_restores_full_state(tmp_path):
    from scisi.training.gradient_clipping import EmaGradientClipper

    torch.manual_seed(871)
    complete = trainer(
        tmp_path / "complete",
        True,
        "teacher_rollout",
        value=0.0,
        num_epochs=2,
        gradient_clipper=EmaGradientClipper(),
    )
    complete.train()
    reference = torch.load(tmp_path / "complete/last.pth", weights_only=True)

    torch.manual_seed(871)
    first = trainer(
        tmp_path / "first",
        True,
        "teacher_rollout",
        value=0.0,
        gradient_clipper=EmaGradientClipper(),
    )
    first.train()
    saved = torch.load(tmp_path / "first/last.pth", weights_only=True)
    # Initialization after unrelated random draws must not change the continuation.
    torch.rand(43)
    continued = trainer(
        tmp_path / "continued",
        True,
        "teacher_rollout",
        value=9.0,
        num_epochs=2,
        gradient_clipper=EmaGradientClipper(),
        resume_from=tmp_path / "first/last.pth",
    )
    assert continued.start_epoch == 1 and continued.global_step == 2
    assert torch.equal(torch.random.get_rng_state(), saved["torch_rng_state"])
    assert continued.history == saved["history"]
    assert (
        continued.gradient_clipper.ema_index
        == saved["gradient_clipper"]["state"]["ema_index"]
    )
    assert continued.early_stopping.best_loss == saved["best_loss"]
    assert (tmp_path / "continued/model.pth").exists()
    continued.train()
    actual = torch.load(tmp_path / "continued/last.pth", weights_only=True)
    for key in (
        "model",
        "ema_model",
        "optimizer",
        "scheduler",
        "global_step",
        "epoch",
        "best_loss",
        "early_stopping",
        "gradient_clipper",
        "torch_rng_state",
    ):
        assert_nested_equal(actual[key], reference[key])
    assert [row["epoch"] for row in actual["history"]] == [0, 1]
    assert [row["val_loss"] for row in actual["history"]] == [
        row["val_loss"] for row in reference["history"]
    ]
    assert actual["elapsed_seconds"] >= saved["elapsed_seconds"]


def test_resume_restores_explicit_loader_generators_and_rejects_wrong_objective(
    tmp_path,
):
    instance = trainer(tmp_path / "first", False, "teacher_rollout", value=0.0)

    class Batches(list):
        pass

    loader = Batches(instance.train_dataloader)
    loader.generator = torch.Generator().manual_seed(41)
    torch.rand(13, generator=loader.generator)
    instance.train_dataloader = loader
    instance.train()
    expected = loader.generator.get_state().clone()
    resumed = trainer(tmp_path / "resumed", False, "teacher_rollout", value=0.0)
    resumed.train_dataloader = Batches(resumed.train_dataloader)
    resumed.train_dataloader.generator = torch.Generator().manual_seed(999)
    resumed._restore_checkpoint(tmp_path / "first/last.pth")
    assert torch.equal(resumed.train_dataloader.generator.get_state(), expected)
    with pytest.raises(ValueError, match="different map training objective"):
        trainer(
            tmp_path / "bad",
            False,
            "progressive",
            resume_from=tmp_path / "first/last.pth",
        )
