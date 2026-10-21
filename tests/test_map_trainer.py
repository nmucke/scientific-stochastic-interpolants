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
from scisi.training.map_trainer import (
    MapTrainer,
    lagrangian_self_distillation_loss,
    progressive_self_distillation_loss,
)
from scisi.training.trainer import EarlyStopping


class ConstantMapNetwork(nn.Module):
    def __init__(self, value=0.4):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def forward(self, x, cond, **kwargs):
        return self.value.expand_as(x)


def batch():
    generator = torch.Generator().manual_seed(11)
    base = torch.randn(3, 1, 2, 2, generator=generator)
    return {
        "base": base,
        "target": base + torch.randn(base.shape, generator=generator),
        "field_history": base.unsqueeze(-1),
    }


def trainer(tmp_path, stochastic, value=0.4, **kwargs):
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
        train_dataloader=[batch(), batch()],
        val_dataloader=[batch()],
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9),
        gradient_clipper=None,
        early_stopping=EarlyStopping(patience=3),
        device="cpu",
        checkpoint_dir=str(tmp_path),
        warmup_epochs=0,
        **kwargs,
    )


OFF_DIAGONAL = ["progressive_self_distillation", "lagrangian_self_distillation"]


class LinearFlowMapNetwork(nn.Module):
    """Average velocity of dx/dt = a x: exact, or the one-step Euler map."""

    def __init__(self, a=0.7, exact=True):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a))
        self.exact = exact

    def forward(self, x, cond, **kwargs):
        gap = (cond[:, 1:] - cond[:, :1]).reshape(-1, 1, 1, 1)
        if not self.exact:
            return self.a * x
        # (e^{a h} - 1) / h, continued to a at h = 0.
        safe = torch.where(gap > 0, gap, torch.ones_like(gap))
        ratio = torch.where(gap > 0, torch.expm1(self.a * safe) / safe, self.a)
        return ratio * x


@pytest.mark.parametrize(
    "loss", [progressive_self_distillation_loss, lagrangian_self_distillation_loss]
)
@pytest.mark.parametrize("exact", [True, False])
def test_off_diagonal_losses_vanish_exactly_for_the_true_flow_map(loss, exact):
    model = FlowMapModel(
        LinearDeterministicInterpolation(), LinearFlowMapNetwork(exact=exact)
    )
    x = torch.randn(4, 1, 2, 2)
    s = torch.tensor([[0.0], [0.1], [0.3], [0.5]])
    t = torch.tensor([[1.0], [0.6], [0.4], [0.9]])
    u = (s + t) / 2
    value = loss(model, model, nn.MSELoss(), x, s, u, t, None, {})
    if exact:
        assert value.item() < 1e-10
    else:
        assert value.item() > 1e-3


def test_lagrangian_loss_backpropagates_through_the_time_derivative():
    model = FlowMapModel(
        LinearDeterministicInterpolation(), LinearFlowMapNetwork(exact=False)
    )
    target = FlowMapModel(LinearDeterministicInterpolation(), LinearFlowMapNetwork())
    x = torch.randn(4, 1, 2, 2)
    s, t = torch.zeros(4, 1), torch.ones(4, 1)
    lagrangian_self_distillation_loss(
        model, target.requires_grad_(False), nn.MSELoss(), x, s, None, t, None, {}
    ).backward()
    assert model.map_model.a.grad is not None and model.map_model.a.grad != 0


class TimeAndPathNetwork(nn.Module):
    """A smooth map that depends on time, state and the Brownian features."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, cond, field_history=None, field_cond=None, pars_cond=None, path_features=None):
        t = cond[:, 1:].reshape(-1, 1, 1, 1)
        return self.a * t * x + t**2 * path_features.sum(1, keepdim=True)


@pytest.mark.parametrize("features", ["karhunen_loeve", "local"])
def test_lagrangian_time_derivative_holds_brownian_features_fixed(features):
    """KL features do not move with t (exact); local ones are frozen at [s, t]."""
    torch.manual_seed(3)
    model = FollmerItoMapModel(
        QuadraticStochasticInterpolation(gamma_multiplier=0.5),
        TimeAndPathNetwork(),
        brownian_features=features,
    )
    x = torch.randn(3, 1, 2, 2)
    s, t = torch.full((3, 1), 0.2), torch.full((3, 1), 0.7)
    path = model.make_brownian_path(x)
    loss = lagrangian_self_distillation_loss(
        model, model, nn.MSELoss(), x, s, None, t, path, {}
    )

    def displacement(time, frozen):
        if frozen:  # features of [s, t], used at a nearby end time
            features = path.features(s, t, model.num_path_features)
            cond = torch.cat([s, time], -1)
            velocity = model.map_model(x, cond, path_features=features)
        else:  # features recomputed for [s, time]
            velocity = model.average_velocity(x, s, time, brownian_path=path)
        return (time - s).reshape(-1, 1, 1, 1) * velocity

    h = 1e-3
    with torch.no_grad():
        mapped = x + displacement(t, True) + path.increment(s, t)
        target = model.average_velocity(mapped, t, t)
        frozen = (displacement(t + h, True) - displacement(t - h, True)) / (2 * h)
        moving = (displacement(t + h, False) - displacement(t - h, False)) / (2 * h)
    torch.testing.assert_close(loss, nn.MSELoss()(frozen, target), rtol=1e-4, atol=1e-6)
    if features == "karhunen_loeve":
        torch.testing.assert_close(frozen, moving, rtol=1e-4, atol=1e-6)
    else:
        assert (frozen - moving).abs().max() > 1e-2


def test_lagrangian_with_local_features_warns(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        instance = trainer(tmp_path, True, off_diagonal="lagrangian_self_distillation")
    assert instance.model.brownian_features == "local"
    assert "Karhunen-Loeve" in caplog.text


def test_unknown_off_diagonal_loss_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown off_diagonal"):
        trainer(tmp_path, False, off_diagonal="euler")


@pytest.mark.parametrize("off_diagonal", OFF_DIAGONAL)
@pytest.mark.parametrize("stochastic", [False, True])
def test_constant_drift_has_zero_off_diagonal_loss(tmp_path, stochastic, off_diagonal):
    torch.manual_seed(7)
    instance = trainer(tmp_path, stochastic, off_diagonal=off_diagonal)
    instance._compute_loss(instance._prepare_batch(batch()))
    assert instance.components["off_diagonal_loss"] < 1e-10


@pytest.mark.parametrize("off_diagonal", OFF_DIAGONAL)
@pytest.mark.parametrize("stochastic", [False, True])
def test_optimizer_updates_student_without_ema_gradients(
    tmp_path, stochastic, off_diagonal
):
    instance = trainer(tmp_path, stochastic, value=0.0, off_diagonal=off_diagonal)
    original = instance.model.map_model.value.detach().clone()
    loss = instance._train_step(instance._prepare_batch(batch()))
    assert torch.isfinite(loss)
    assert instance.global_step == 1
    assert instance.model.map_model.value.grad is not None
    assert instance.model.map_model.value.item() != original.item()
    assert all(
        not p.requires_grad and p.grad is None for p in instance.ema_model.parameters()
    )
    assert instance.ema_model.map_model.value.item() != original.item()


@pytest.mark.parametrize("off_diagonal", OFF_DIAGONAL)
def test_training_saves_student_ema_and_history(tmp_path, off_diagonal):
    instance = trainer(tmp_path, True, value=0.0, off_diagonal=off_diagonal)
    original_batch = batch()
    instance._prepare_batch(original_batch)
    assert "t" not in original_batch and "noise" not in original_batch
    history = instance.train()
    assert len(history) == 1 and history[0]["step"] == 2
    for key in ("train_loss", "val_loss", "train_diagonal_loss", "val_off_diagonal_loss"):
        assert key in history[0]
    assert all(
        (tmp_path / name).exists()
        for name in ("model.pth", "ema.pth", "last.pth", "history.json")
    )
    assert json.loads((tmp_path / "history.json").read_text())[0]["step"] == 2
    # Fixed validation draws give identical loss without perturbing caller RNG.
    rng = torch.random.get_rng_state().clone()
    assert instance._validate() == instance._validate()
    assert torch.equal(rng, torch.random.get_rng_state())


def test_diagonal_epochs_then_linear_warmup_in_epochs(tmp_path):
    instance = trainer(tmp_path, False, diagonal_epochs=2)
    instance.warmup_epochs = 4
    batch_ = instance._prepare_batch(batch())
    for epoch, weight in [(0, 0.0), (1, 0.0), (2, 0.25), (3, 0.5), (5, 1.0), (9, 1.0)]:
        instance.epoch = epoch
        instance._compute_loss(batch_)
        assert instance._current_off_diagonal_weight() == weight
        if weight == 0.0:
            assert instance.components["off_diagonal_loss"] == 0.0


def test_nontrivial_mask_is_rejected_before_training_initialization(tmp_path):
    model = FlowMapModel(LinearDeterministicInterpolation(), ConstantMapNetwork())
    model.mask = torch.tensor([[[[1.0, 0.0]]]])
    with pytest.raises(ValueError, match="identically-one mask"):
        MapTrainer(
            1,
            model,
            [batch()],
            [batch()],
            torch.optim.SGD(model.parameters(), lr=0.01),
            None,
            EarlyStopping(),
            device="cpu",
            checkpoint_dir=tmp_path / "masked",
        )
    assert not (tmp_path / "masked").exists()


def test_resume_restores_ema_and_matches_uninterrupted_training(tmp_path):
    torch.manual_seed(3)
    uninterrupted = trainer(tmp_path / "all", True, value=0.0, num_epochs=2)
    uninterrupted.train()

    torch.manual_seed(3)
    trainer(tmp_path / "first", True, value=0.0, num_epochs=1).train()
    resumed = trainer(
        tmp_path / "resume",
        True,
        value=5.0,
        num_epochs=2,
        resume_from=tmp_path / "first" / "last.pth",
    )
    resumed.train()

    for a, b in (
        (uninterrupted.model, resumed.model),
        (uninterrupted.ema_model, resumed.ema_model),
    ):
        torch.testing.assert_close(a.map_model.value, b.map_model.value, rtol=0, atol=0)
    assert uninterrupted.history == resumed.history
