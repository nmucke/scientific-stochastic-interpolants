"""Tests for the NS velocity-magnitude (speed) observation operator."""

import math

import pytest
import torch

from scisi.likelihood_models.physics_observation_operators import (
    VelocityMagnitudeOperator,
)

N = 32


def _grid(n: int = N):
    x = torch.arange(n, dtype=torch.float64) * (2 * math.pi / n)
    X, Y = torch.meshgrid(x, x, indexing="ij")  # axis -2 = x, axis -1 = y
    return X, Y


def _op(percent_obs: float = 0.1, **kw) -> VelocityMagnitudeOperator:
    return VelocityMagnitudeOperator(
        inner={"type": "random", "percent_obs": percent_obs, "seed": 0},
        data_size=(1, N, N),
        **kw,
    )


def test_velocity_of_analytic_vorticity():
    """omega = 2 sin x sin y -> psi = sin x sin y, u = sin x cos y, v = -cos x sin y."""
    X, Y = _grid()
    omega = (2 * torch.sin(X) * torch.sin(Y))[None, None]
    op = _op(norm_std=1.0, norm_mean=0.0)
    u, v = op.velocity(omega)
    torch.testing.assert_close(u[0, 0], torch.sin(X) * torch.cos(Y), atol=1e-12, rtol=0)
    torch.testing.assert_close(v[0, 0], -torch.cos(X) * torch.sin(Y), atol=1e-12, rtol=0)
    # Speed at the sensors.
    speed = torch.sqrt(u**2 + v**2 + op.eps**2).reshape(1, -1)[:, op.inner.obs_indices]
    torch.testing.assert_close(op(omega), speed, atol=1e-12, rtol=0)


def test_normalisation_and_scale():
    X, Y = _grid()
    omega_raw = (2 * torch.sin(X) * torch.sin(Y))[None, None]
    op = _op(norm_std=4.0, norm_mean=0.0, velocity_scale=0.5, eps=0.0)
    y = op(omega_raw / 4.0)
    ref = _op(norm_std=1.0, eps=0.0)(omega_raw) / 0.5
    torch.testing.assert_close(y, ref, atol=1e-12, rtol=0)


def test_divergence_free_and_mean_mode():
    torch.manual_seed(0)
    omega = torch.randn(2, 1, N, N, dtype=torch.float64)
    op = _op(norm_std=1.0)
    u, v = op.velocity(omega)
    # Spectral divergence du/dx + dv/dy = 0 (exact up to round-off).
    k = torch.fft.fftfreq(N, 2 * math.pi / N, dtype=torch.float64) * 2 * math.pi
    div = torch.fft.ifft2(
        1j * k.reshape(N, 1) * torch.fft.fft2(u) + 1j * k.reshape(1, N) * torch.fft.fft2(v)
    ).real
    assert div.abs().max() < 1e-10
    # Constant vorticity offset produces no velocity (psi mean mode dropped).
    u2, v2 = op.velocity(omega + 3.0)
    torch.testing.assert_close(u2, u, atol=1e-12, rtol=0)
    torch.testing.assert_close(v2, v, atol=1e-12, rtol=0)


@pytest.mark.parametrize("velocity_scale", [1.0, 0.25])
def test_jvp_vjp_vs_finite_differences_and_autodiff(velocity_scale):
    torch.manual_seed(1)
    op = _op(norm_std=3.09969, velocity_scale=velocity_scale)
    x = torch.randn(3, 1, N, N, dtype=torch.float64)
    dx = torch.randn_like(x)
    w = torch.randn(3, op.num_obs, dtype=torch.float64)

    # Analytic JVP vs central finite differences.
    h = 1e-6
    fd = (op(x + h * dx) - op(x - h * dx)) / (2 * h)
    jv = op.jvp(x, dx)
    torch.testing.assert_close(jv, fd, atol=1e-6, rtol=1e-6)

    # Analytic JVP / VJP vs torch.func (autodiff through the FFTs).
    _, jv_ad = torch.func.jvp(op.forward, (x,), (dx,))
    torch.testing.assert_close(jv, jv_ad, atol=1e-10, rtol=1e-10)
    _, pullback = torch.func.vjp(op.forward, x)
    torch.testing.assert_close(op.vjp(x, w), pullback(w)[0], atol=1e-10, rtol=1e-10)

    # Adjoint identity <J dx, w> == <dx, J^T w>.
    lhs = (jv * w).sum()
    rhs = (dx * op.vjp(x, w)).sum()
    torch.testing.assert_close(lhs, rhs, atol=1e-10, rtol=1e-10)


def test_dense_jacobian_vs_finite_differences():
    torch.manual_seed(2)
    op = _op(percent_obs=0.02, norm_std=1.0)
    x = torch.randn(1, 1, N, N, dtype=torch.float64)
    J = op.jacobian(x)
    assert J.shape == (op.num_obs, N * N)
    h = 1e-6
    cols = torch.randint(0, N * N, (8,))
    for c in cols:
        e = torch.zeros(1, N * N, dtype=torch.float64)
        e[0, c] = 1.0
        e = e.reshape(1, 1, N, N)
        fd = (op(x + h * e) - op(x - h * e)) / (2 * h)
        torch.testing.assert_close(J[:, c], fd[0], atol=1e-6, rtol=1e-6)
    lin = op.linearize(x)
    assert lin.is_linear and lin.obs_matrix is not None
    dx = torch.randn_like(x)
    torch.testing.assert_close(lin(dx), op.jvp(x, dx), atol=1e-10, rtol=1e-10)
    # Default gram (chunked VJP -> JVP) == J J^T.
    torch.testing.assert_close(op.gram(x)[0], J @ J.T, atol=1e-10, rtol=1e-10)


def test_nonlinear_contract():
    op = _op()
    assert not op.is_linear
    with pytest.raises(AttributeError):
        _ = op.obs_matrix
    with pytest.raises(NotImplementedError):
        op.transpose(torch.zeros(1, op.num_obs))
    assert op.obs_indices_on_grid.shape == (1, N, N)
