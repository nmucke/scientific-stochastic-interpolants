"""Periodic 1D drift network: symmetry, differentiability and library integration."""

import pytest
import torch
import torch.nn as nn

from scisi.architectures.periodic_1d import PeriodicResNet1d
from scisi.dynamics.lorenz96 import integrate
from scisi.likelihood_models.gaussian_likelihood import InterpolantGaussianLikelihood
from scisi.likelihood_models.observation_operators import LinearObservationOperator
from scisi.models.flow_matching_model import FlowMatchingModel
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
from scisi.models.interpolations import (
    LinearDeterministicInterpolation,
    QuadraticStochasticInterpolation,
)

BATCH = 6

BUILDERS = {
    "resnet": lambda d, L: PeriodicResNet1d(
        len_field_history=L, hidden_channels=32, num_blocks=4, kernel_size=7
    ),
}
VARIANTS = list(BUILDERS)


@pytest.fixture(autouse=True)
def _seed_and_single_thread():
    torch.set_num_threads(1)
    torch.manual_seed(0)


def build(name, d=24, L=1, dtype=torch.float32):
    return BUILDERS[name](d, L).to(dtype)


def inputs(batch=BATCH, d=24, L=1, dtype=torch.float32):
    return (
        torch.randn(batch, 1, 1, d, dtype=dtype),
        torch.rand(batch, 1, dtype=dtype),
        torch.randn(batch, 1, 1, d, L, dtype=dtype),
    )


# --- shapes and broadcasting -------------------------------------------------
@pytest.mark.parametrize("name", VARIANTS)
@pytest.mark.parametrize("d", [16, 24])
@pytest.mark.parametrize("L", [1, 3])
def test_output_shape(name, d, L):
    x, t, history = inputs(d=d, L=L)
    assert build(name, d, L)(x, t, history).shape == x.shape


@pytest.mark.parametrize("name", VARIANTS)
def test_keyword_call_matches_positional_call(name):
    net = build(name)
    x, t, history = inputs()
    keyword = net(x=x, cond=t, field_history=history, field_cond=None, pars_cond=None)
    assert torch.equal(keyword, net(x, t, history, None, None))


@pytest.mark.parametrize("name", VARIANTS)
@pytest.mark.parametrize("shape", [(BATCH, 1), (1, 1), (1,)])
def test_cond_broadcasts_to_batch(name, shape):
    net = build(name)
    x, _, history = inputs()
    expected = net(x, torch.full((BATCH, 1), 0.3), history)
    out = net(x, torch.full(shape, 0.3), history)
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", VARIANTS)
def test_history_of_batch_one_broadcasts_to_x(name):
    net = build(name)
    x, t, history = inputs()
    expected = net(x, t, history[:1].expand(BATCH, -1, -1, -1, -1).contiguous())
    torch.testing.assert_close(net(x, t, history[:1]), expected)


@pytest.mark.parametrize("name", VARIANTS)
def test_output_depends_on_time_and_previous_state(name):
    net = build(name)
    x, t, history = inputs()
    out = net(x, t, history)
    assert not torch.allclose(out, net(x, 1.0 - t, history), atol=1e-4)
    changed = history.clone()
    changed[..., -1] += 1.0
    assert not torch.allclose(out, net(x, t, changed), atol=1e-4)


def test_default_size():
    net = PeriodicResNet1d()
    assert 1e5 <= sum(p.numel() for p in net.parameters()) <= 1e6
    x, t, history = inputs()
    assert net(x, t, history).shape == x.shape


# --- rejected input ----------------------------------------------------------
def test_resnet_rejects_lattice_beyond_receptive_field():
    net = PeriodicResNet1d(hidden_channels=8, num_blocks=1, kernel_size=3)
    x, t, history = inputs(d=16)
    with pytest.raises(ValueError, match="Receptive field"):
        net(x, t, history)


@pytest.mark.parametrize("name", VARIANTS)
def test_rejects_unsupported_input(name):
    net = build(name)
    x, t, history = inputs()
    with pytest.raises(ValueError, match="field_history"):
        net(x, t)
    with pytest.raises(ValueError, match="not supported"):
        net(x, t, history, pars_cond=t)
    with pytest.raises(ValueError, match="history states"):
        net(x, t, history.expand(-1, -1, -1, -1, 2))
    with pytest.raises(ValueError, match="broadcast"):
        net(x, t[:2], history)


# --- translation equivariance ------------------------------------------------
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.float64, 1e-12)])
@pytest.mark.parametrize("d", [16, 24])
@pytest.mark.parametrize("shift", [1, 2, 3, 4, 5, 8, 11])
@pytest.mark.parametrize("name", VARIANTS)
def test_translation_equivariance(name, shift, d, dtype, tol):
    net = build(name, d, L=2, dtype=dtype)
    x, t, history = inputs(d=d, L=2, dtype=dtype)
    with torch.no_grad():
        rolled_then_net = net(x.roll(shift, -1), t, history.roll(shift, -2))
        net_then_rolled = net(x, t, history).roll(shift, -1)
    assert (rolled_then_net - net_then_rolled).abs().max() < tol


# --- receptive field ---------------------------------------------------------
@pytest.mark.parametrize("name", VARIANTS)
@pytest.mark.parametrize("d", [16, 24])
def test_output_site_depends_on_all_sites(name, d):
    net = build(name, d, dtype=torch.float64)
    x, t, history = inputs(batch=1, d=d, dtype=torch.float64)

    def site_0(state, previous):
        return net(state, t, previous)[0, 0, 0, 0]

    grad_x, grad_history = torch.func.grad(site_0, argnums=(0, 1))(x, history)
    assert (grad_x.abs() > 0).all()
    assert (grad_history.abs() > 0).all()


# --- forward-mode differentiation --------------------------------------------
def finite_difference(fn, x, v, eps=1e-5):
    return (fn(x + eps * v) - fn(x - eps * v)) / (2 * eps)


@pytest.mark.parametrize("name", VARIANTS)
@pytest.mark.parametrize("shape", [(BATCH, 1), (1, 1), (1,)])
def test_jvp_matches_finite_difference(name, shape):
    net = build(name, dtype=torch.float64)
    x, _, history = inputs(dtype=torch.float64)
    t = torch.full(shape, 0.4, dtype=torch.float64)
    v = torch.randn_like(x)

    def fn(z):
        return net(z, t, history)

    with torch.no_grad():
        _, jv = torch.func.jvp(fn, (x,), (v,))
        reference = finite_difference(fn, x, v)
    torch.testing.assert_close(jv, reference, rtol=1e-6, atol=1e-7)


def make_si(net):
    return FollmerStochasticInterpolant(
        interpolation=QuadraticStochasticInterpolation(), drift_model=net
    )


def make_fm(net):
    return FlowMatchingModel(
        interpolation=LinearDeterministicInterpolation(), drift_model=net
    )


MODELS = {"si": (make_si, "drift"), "fm": (make_fm, "score")}


@pytest.mark.parametrize("name", VARIANTS)
@pytest.mark.parametrize("model_class", ["si", "fm"])
@pytest.mark.parametrize("state_batch", [1, BATCH])
def test_library_jvp_with_batch_tangent(name, model_class, state_batch):
    """The JVP as built by the likelihood, with the state broadcast to the tangent."""
    make, net_fn = MODELS[model_class]
    model = make(build(name, dtype=torch.float64))
    likelihood = InterpolantGaussianLikelihood(
        model=model, obs_operator=nn.Identity(), model_class=model_class
    )
    x, _, history = inputs(batch=state_batch, dtype=torch.float64)
    t = torch.full((1, 1), 0.4, dtype=torch.float64)
    v = torch.randn(BATCH, 1, 1, 24, dtype=torch.float64)

    jv = likelihood._build_jvp_fn(
        x=x, t_net=t, field_history=history, field_cond=None, pars_cond=None
    )(v)

    def fn(z):
        wide_t = t.expand(BATCH, 1)
        wide_history = history.expand(BATCH, -1, -1, -1, -1)
        return getattr(model, net_fn)(z, wide_t, wide_history)

    with torch.no_grad():
        reference = finite_difference(fn, x.expand(BATCH, -1, -1, -1), v)
    assert jv.shape == v.shape
    torch.testing.assert_close(jv, reference, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", VARIANTS)
@pytest.mark.parametrize("model_class", ["si", "fm"])
@pytest.mark.parametrize("mode", ["inflated", "inflated_shared"])
def test_likelihood_score_runs(name, model_class, mode):
    make, _ = MODELS[model_class]
    model = make(build(name, d=16))
    likelihood = InterpolantGaussianLikelihood(
        model=model,
        obs_operator=LinearObservationOperator(
            type="random", data_size=(1, 1, 16), percent_obs=0.25, seed=0
        ),
        likelihood_mode=mode,
        model_class=model_class,
    )
    x, _, history = inputs(d=16)
    t = torch.full((1, 1), 0.4)
    with torch.no_grad():
        drift = model.drift(x, t, history)
        score = model.score(x, t, history) if model_class == "fm" else None
        sbar, _ = likelihood.score(
            observations=torch.randn(BATCH, 4),
            x=x,
            t=t,
            field_history=history,
            drift=drift,
            score=score,
        )
    assert sbar.shape == x.shape and torch.isfinite(sbar).all()


# --- integration with the generative models ----------------------------------
def assert_gradients_reach_all_parameters(net):
    for parameter_name, parameter in net.named_parameters():
        assert parameter.grad is not None, parameter_name
        assert torch.isfinite(parameter.grad).all(), parameter_name
        assert parameter.grad.abs().sum() > 0, parameter_name


@pytest.mark.parametrize("name", VARIANTS)
@pytest.mark.parametrize("model_class", ["si", "fm"])
def test_model_forward_and_sample(name, model_class):
    net = build(name)
    model = MODELS[model_class][0](net)
    target, t, history = inputs()
    base = history[..., -1]

    pred, true = model(
        base=base,
        target=target,
        t=t,
        noise=torch.randn_like(target),
        field_history=history,
    )
    loss = (pred - true).pow(2).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert_gradients_reach_all_parameters(net)

    with torch.no_grad():
        sample = model.sample(base=base, field_history=history, num_steps=4)
        ensemble = model.sample(
            base=base[:1], field_history=history[:1], batch_size=BATCH, num_steps=4
        )
    assert sample.shape == target.shape and torch.isfinite(sample).all()
    assert ensemble.shape == target.shape and torch.isfinite(ensemble).all()


# --- smoke fit ---------------------------------------------------------------
@pytest.fixture(scope="module")
def lorenz96_pairs():
    """Standardised pairs (x, x after 0.4 time units) from the attractor."""
    generator = torch.Generator().manual_seed(0)
    x = 8.0 + torch.randn(256, 24, generator=generator, dtype=torch.float64)
    x = integrate(x, 1000, 0.01, 8.0)
    mean, std = x.mean(), x.std()
    z = (x - mean) / std
    target = (integrate(mean + std * z, 40, 0.01, 8.0) - mean) / std
    return z.float()[:, None, None], target.float()[:, None, None]


@pytest.mark.parametrize("name", VARIANTS)
def test_smoke_fit_reduces_regression_loss(name, lorenz96_pairs):
    state, target = lorenz96_pairs
    net = build(name)
    history = state.unsqueeze(-1)
    x = torch.randn_like(state)
    t = torch.full((state.shape[0], 1), 0.5)
    optimiser = torch.optim.Adam(net.parameters(), lr=2e-3)

    losses = []
    for _ in range(300):
        loss = (net(x, t, history) - target).pow(2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    assert min(losses[-10:]) < 0.2 * losses[0]
