"""Tests for the deterministic time-stepping model."""

import pytest
import torch

from scisi.architectures.u_net import UNet
from scisi.deterministic_models import DeterministicModel

BATCH_SIZE = 2
NUM_CHANNELS = 1
HEIGHT = 16
WIDTH = 16
LEN_FIELD_HISTORY = 3


def _make_network(dropout_rate: float = 0.0) -> UNet:
    """Attention-free two-level UNet small enough for CPU tests."""
    return UNet(
        in_channels=NUM_CHANNELS,
        out_channels=NUM_CHANNELS,
        hidden_channels=[4, 8],
        cond_dim=1,
        cond_embedding_dim=16,
        len_field_history=LEN_FIELD_HISTORY,
        multiplier=2,
        num_blocks=1,
        dropout_rate=dropout_rate,
        attention_in_layers=[False, False],
        attention={"target": "torch.nn.Identity"},
    )


@pytest.fixture
def network() -> UNet:
    torch.manual_seed(0)
    return _make_network()


@pytest.fixture
def batch() -> dict:
    generator = torch.Generator().manual_seed(1)
    field_history = torch.randn(
        BATCH_SIZE,
        NUM_CHANNELS,
        HEIGHT,
        WIDTH,
        LEN_FIELD_HISTORY,
        generator=generator,
    )
    return {
        "field_history": field_history,
        "base": field_history[:, :, :, :, -1],
        "target": torch.randn(
            BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH, generator=generator
        ),
    }


def test_forward_returns_pred_target_and_ignores_extra_kwargs(network, batch):
    model = DeterministicModel(network=network).eval()

    with torch.no_grad():
        pred, target = model(
            base=batch["base"],
            target=batch["target"],
            field_history=batch["field_history"],
            t=torch.rand(BATCH_SIZE, 1),
            noise=torch.randn_like(batch["base"]),
        )

    assert pred.shape == (BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH)
    assert target is batch["target"]


def test_residual_flag(network, batch):
    """residual=True returns base + network output; residual=False the output."""
    direct = DeterministicModel(network=network, residual=False).eval()
    residual = DeterministicModel(network=network, residual=True).eval()

    with torch.no_grad():
        direct_pred, _ = direct(**batch)
        residual_pred, _ = residual(**batch)

    assert not torch.allclose(direct_pred, residual_pred)
    assert torch.allclose(residual_pred, batch["base"] + direct_pred)


def test_sample_single_step(network, batch):
    model = DeterministicModel(network=network).eval()

    pred = model.sample(field_history=batch["field_history"], base=None)

    assert pred.shape == (BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH)

    # base=None falls back to the last history slice.
    pred_from_base = model.sample(
        field_history=batch["field_history"], base=batch["base"]
    )
    assert torch.allclose(pred, pred_from_base)


def test_sample_returns_rolled_field_history(network, batch):
    model = DeterministicModel(network=network).eval()

    pred, rolled = model.sample(
        field_history=batch["field_history"], base=None, return_field_history=True
    )

    assert rolled.shape == batch["field_history"].shape
    assert torch.allclose(rolled[:, :, :, :, :-1], batch["field_history"][:, :, :, :, 1:])
    assert torch.allclose(rolled[:, :, :, :, -1], pred)


def test_sample_trajectory_rollout(network, batch):
    """Rollout has the right shape, keeps the seeded history, and ignores
    stochastic-sampler kwargs (num_steps, stepper)."""
    model = DeterministicModel(network=network).eval()
    num_physical_steps = 6

    trajectory = model.sample_trajectory(
        field_history=batch["field_history"],
        base=None,
        num_physical_steps=num_physical_steps,
        num_steps=50,
        stepper=object(),
    )

    assert trajectory.shape == (
        BATCH_SIZE,
        NUM_CHANNELS,
        HEIGHT,
        WIDTH,
        num_physical_steps,
    )
    assert torch.allclose(
        trajectory[:, :, :, :, :LEN_FIELD_HISTORY], batch["field_history"]
    )


def test_sample_is_deterministic_despite_dropout_and_train_mode(batch):
    """sample() forces eval mode internally: dropout must not perturb the
    prediction, and the caller's training mode must be restored afterwards."""
    torch.manual_seed(0)
    model = DeterministicModel(network=_make_network(dropout_rate=0.5))
    model.train()

    pred_first = model.sample(field_history=batch["field_history"], base=None)
    pred_second = model.sample(field_history=batch["field_history"], base=None)

    assert torch.allclose(pred_first, pred_second)
    assert model.training


def test_sample_trajectory_rejects_too_short_rollout(network, batch):
    model = DeterministicModel(network=network).eval()

    with pytest.raises(ValueError, match="num_physical_steps"):
        model.sample_trajectory(
            field_history=batch["field_history"],
            num_physical_steps=LEN_FIELD_HISTORY,
        )


def test_drift_raises(network):
    model = DeterministicModel(network=network)

    with pytest.raises(NotImplementedError):
        model.drift(
            x=torch.randn(BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH),
            t=torch.rand(BATCH_SIZE, 1),
        )
