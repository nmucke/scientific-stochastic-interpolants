"""Invariants that distinguish conditional flow/Itô maps from numerical solvers."""

import pytest
import torch
from torch import nn

from scisi.architectures.map_conditioning import MapConditionedNetwork
from scisi.architectures.u_net import UNet
from scisi.models.flow_map_model import FlowMapModel
from scisi.models.follmer_ito_map_model import FollmerItoMapModel
from scisi.models.interpolations import (
    LinearDeterministicInterpolation,
    QuadraticStochasticInterpolation,
)


class ConstantVelocity(nn.Module):
    def __init__(self, value=0.5):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def forward(
        self,
        x,
        cond,
        field_history=None,
        field_cond=None,
        pars_cond=None,
        path_features=None,
    ):
        return torch.ones_like(x) * self.value


def flow(value=0.5):
    return FlowMapModel(LinearDeterministicInterpolation(), ConstantVelocity(value))


def ito(value=0.5, gamma=1.0):
    return FollmerItoMapModel(
        QuadraticStochasticInterpolation(gamma_multiplier=gamma, wiener_process=True),
        ConstantVelocity(value),
    )


def test_flow_identity_composition_and_backprop():
    model = flow()
    x = torch.randn(3, 1, 4, 4)
    torch.testing.assert_close(model.map(x, 0.4, 0.4), x, rtol=0, atol=0)
    direct = model.map(x, 0.0, 1.0)
    composed = model.map(model.map(x, 0.0, 0.3), 0.3, 1.0)
    torch.testing.assert_close(direct, composed)
    direct.mean().backward()
    torch.testing.assert_close(model.model.value.grad, torch.tensor(1.0))


def test_ito_same_path_composition_and_diagonal():
    model = ito()
    x = torch.randn(3, 1, 4, 4)
    path = model.make_brownian_path(x, torch.Generator().manual_seed(10))
    torch.testing.assert_close(model.map(x, 0.4, 0.4), x, rtol=0, atol=0)
    direct = model.map(x, 0.0, 1.0, brownian_path=path)
    composed = model.map(
        model.map(x, 0.0, 0.3, brownian_path=path), 0.3, 1.0, brownian_path=path
    )
    torch.testing.assert_close(direct, composed)
    with pytest.raises(ValueError, match="shared"):
        model.map(x, 0.0, 1.0)
    with pytest.raises(ValueError, match="s <= t"):
        model.map(x, 1.0, 0.0, brownian_path=path)


def test_stochastic_initialization_and_zero_diffusion_limit():
    history = torch.full((1, 1, 4, 4, 2), 3.0)
    deterministic_ito = ito(gamma=0.0)
    out, next_history = deterministic_ito.sample(
        history, batch_size=3, return_field_history=True
    )
    torch.testing.assert_close(out, torch.full((3, 1, 4, 4), 3.5))
    torch.testing.assert_close(next_history[..., 0], torch.full_like(out, 3.0))
    torch.testing.assert_close(next_history[..., -1], out)
    fm = flow()
    noise_out = fm.sample(
        history, batch_size=3, generator=torch.Generator().manual_seed(5)
    )
    assert not torch.allclose(noise_out[0], noise_out[1])
    explicit = fm.sample(history, base=torch.zeros(3, 1, 4, 4), num_steps=3)
    torch.testing.assert_close(explicit, torch.full_like(explicit, 0.5))
    with pytest.raises(ValueError, match="positive integer"):
        fm.sample(history, num_steps=0)


def test_trajectory_shifts_history_and_uses_last_observed_state():
    model = ito(gamma=0.0)
    history = torch.stack([torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4)], -1)
    result = model.sample_trajectory(history, batch_size=2, num_physical_steps=5)
    assert result.shape == (2, 1, 4, 4, 5)
    expected = torch.tensor([0.0, 1.0, 1.5, 2.0, 2.5]).expand_as(result)
    torch.testing.assert_close(result, expected)


def test_training_pair_preserves_repository_follmer_drift():
    model = ito()
    base, target, noise = [torch.randn(2, 1, 4, 4) for _ in range(3)]
    t = torch.tensor([[0.2], [0.7]])
    x, drift = model.training_pair(base, target, t, noise)
    expanded = t[:, :, None, None]
    expected_x = (
        (1 - expanded) * base
        + expanded.square() * target
        + (1 - expanded) * expanded.sqrt() * noise
    )
    expected_drift = -base + 2 * expanded * target - expanded.sqrt() * noise
    torch.testing.assert_close(x, expected_x)
    torch.testing.assert_close(drift, expected_drift)
    fm_x, fm_drift = flow().training_pair(base, target, t, noise)
    torch.testing.assert_close(fm_x, (1 - expanded) * noise + expanded * target)
    torch.testing.assert_close(fm_drift, target - noise)


def small_unet():
    return UNet(
        in_channels=1,
        out_channels=1,
        hidden_channels=[4, 8],
        cond_dim=1,
        cond_embedding_dim=8,
        len_field_history=2,
        multiplier=1,
        num_blocks=1,
        dropout_rate=0.0,
        padding="torch.nn.CircularPad2d",
        attention_in_layers=[False, False],
        attention={"target": "torch.nn.Identity"},
    )


@pytest.mark.parametrize("path_channels", [0, 4])
def test_actual_unet_teacher_transplant_and_new_condition_gradients(path_channels):
    torch.set_num_threads(1)
    teacher = small_unet().eval()
    network = MapConditionedNetwork(small_unet(), path_channels).eval()
    network.load_from_teacher(teacher)
    x = torch.randn(2, 1, 8, 8)
    history = torch.randn(2, 1, 8, 8, 2)
    s = torch.tensor([[0.2], [0.7]])
    times = torch.cat([s, s + 0.1], -1)
    features = torch.randn(2, path_channels, 8, 8) if path_channels else None
    expected = teacher(x, s, history)
    actual = network(x, times, history, path_features=features)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    assert network.backbone.cond_encoder.gap[-1].weight.grad.abs().sum() > 0
    if path_channels:
        assert network.backbone.init_conv.projection.weight.grad.abs().sum() > 0
    restored = MapConditionedNetwork(small_unet(), path_channels).eval()
    restored.load_state_dict(network.state_dict())
    torch.testing.assert_close(
        restored(x, times, history, path_features=features), actual
    )


@pytest.mark.parametrize("time", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_pseudo_time_rejected(time):
    with pytest.raises(ValueError, match="finite"):
        flow().map(torch.zeros(2, 1, 4, 4), 0.0, time)


def test_navier_stokes_sample_without_optional_preprocessor():
    from scisi.data.datasets import StochasticNavierStokesDataset

    dataset = StochasticNavierStokesDataset.__new__(StochasticNavierStokesDataset)
    dataset.preprocesser = None
    window = torch.randn(1, 4, 4, 3)
    sample = dataset._prepare_train_sample(window)
    torch.testing.assert_close(sample["base"], window[..., -2])
    torch.testing.assert_close(sample["target"], window[..., -1])
    torch.testing.assert_close(sample["field_history"], window[..., :-1])


@pytest.mark.parametrize(
    "config_name",
    ["flow_map_stochastic_navier_stokes", "follmer_ito_map_stochastic_navier_stokes"],
)
def test_configured_map_unet_preserves_teacher_on_diagonal(config_name):
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    torch.set_num_threads(1)
    config_dir = str(Path(__file__).resolve().parents[1] / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    teacher = instantiate(cfg.model.map_model.backbone).eval()
    model = instantiate(cfg.model).eval()
    model.map_model.load_from_teacher(teacher)
    x = torch.randn(1, 1, 32, 32)
    history = torch.randn(1, 1, 32, 32, cfg.len_field_history)
    t = torch.tensor([[0.4]])
    with torch.no_grad():
        torch.testing.assert_close(
            model.average_velocity(x, t, t, history),
            teacher(x, t, history),
            rtol=0,
            atol=0,
        )
