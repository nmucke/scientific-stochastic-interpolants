"""Physics-based (nonlinear) observation operators.

:class:`VelocityMagnitudeOperator` observes the local flow SPEED of a 2-D
incompressible flow whose state is the (normalised) vorticity ``omega`` on the
periodic ``[0, L]^2`` torus (``L = 2 pi`` for the stochastic Navier--Stokes
case):

    -Delta psi = omega,   u = d psi / dy,   v = -d psi / dx,
    h(x) = L_sel( sqrt(u^2 + v^2 + eps^2) ) / velocity_scale,

where ``omega = norm_std * x + norm_mean`` undoes the dataset normalisation and
``L_sel`` is an inner (selection) :class:`LinearObservationOperator` picking the
sensor locations. Conventions follow :mod:`scisi.metrics.spectral`: array axis
``-2`` is ``x`` (wavenumber ``kx``), axis ``-1`` is ``y`` (``ky``), angular
wavenumbers ``2 pi fftfreq(N, L/N)`` and the mean (k = 0) mode of ``psi`` set to
zero. The Nyquist modes of the two derivative symbols are zeroed so the velocity
maps ``omega -> u`` / ``omega -> v`` are exactly real and antisymmetric (their
adjoint is minus themselves), which gives an exact analytic ``vjp``.

The map is nonlinear (``is_linear = False``) with analytic Jacobian

    J_h(x) v = L_sel( (u * U v + v * V v) / s ) * norm_std / velocity_scale,

with ``U``/``V`` the (linear) vorticity -> velocity maps and ``s`` the smoothed
speed. ``gram`` / ``jacobian`` use the base-class batched VJP/JVP defaults.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch

from scisi.likelihood_models.observation_operators import SensorComposedOperator

# Preprocesser base std of the stochastic NS vorticity (mean 0); see the
# checkpoint config.yaml ``preprocesser.base.std`` and
# ``paper_experiments/cases/navier_stokes/enkf_baseline.NORM_STD``.
NS_VORTICITY_NORM_STD = 3.09969


def velocity_symbols(
    N: int,
    L: float = 2 * math.pi,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fourier symbols ``(Du, Dv)`` with ``u_h = Du * omega_h``, ``v_h = Dv * omega_h``.

    ``psi_h = omega_h / |k|^2`` (k = 0 -> 0), ``u_h = i ky psi_h``,
    ``v_h = -i kx psi_h``. Nyquist rows/cols of the derivative factor are zeroed.
    Returns complex tensors of shape ``[N, N]``.
    """
    dx = L / N
    k1 = torch.fft.fftfreq(N, dx, dtype=torch.float64) * 2 * math.pi
    if N % 2 == 0:
        k1_deriv = k1.clone()
        k1_deriv[N // 2] = 0.0  # drop the Nyquist derivative (keeps maps real)
    else:
        k1_deriv = k1
    kx = k1.reshape(N, 1)
    ky = k1.reshape(1, N)
    k2 = kx**2 + ky**2
    inv_k2 = torch.where(k2 > 0, 1.0 / torch.where(k2 > 0, k2, 1.0), 0.0)
    Du = 1j * k1_deriv.reshape(1, N) * inv_k2
    Dv = -1j * k1_deriv.reshape(N, 1) * inv_k2
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    return Du.to(device=device, dtype=cdtype), Dv.to(device=device, dtype=cdtype)


class VelocityMagnitudeOperator(SensorComposedOperator):
    """Sensor speed ``|u|`` of a periodic 2-D vorticity field (nonlinear).

    Args:
        inner: Sensor-selection :class:`LinearObservationOperator` (or a dict of
            its constructor kwargs) acting on the ``(1, H, W)`` speed field.
        norm_std: ``omega_raw = norm_std * x + norm_mean`` (de-normalisation).
        norm_mean: See ``norm_std``.
        velocity_scale: The returned speed is divided by this (puts ``y`` on an
            O(1) scale; the observation noise ``R`` is applied to this output).
        eps: Smoothing of the speed ``sqrt(u^2 + v^2 + eps^2)`` (differentiable
            at stagnation points). In the output units BEFORE ``velocity_scale``.
        domain_length: Side length ``L`` of the periodic box.
        vorticity_channel: State channel holding the vorticity.
        **inner_kwargs: Merged into ``inner`` when it is a dict (e.g. ``data_size``).
    """

    def __init__(
        self,
        inner: Any,
        norm_std: float = NS_VORTICITY_NORM_STD,
        norm_mean: float = 0.0,
        velocity_scale: float = 1.0,
        eps: float = 1e-3,
        domain_length: float = 2 * math.pi,
        vorticity_channel: int = 0,
        **inner_kwargs: Any,
    ) -> None:
        super().__init__(inner, **inner_kwargs)
        inner = self.inner
        if inner.C != 1:
            raise ValueError(
                "VelocityMagnitudeOperator expects a single-channel inner sensor "
                f"operator on the (1, H, W) speed field, got C={inner.C}."
            )
        if inner.H != inner.W:
            raise ValueError("VelocityMagnitudeOperator needs a square grid.")
        self.norm_std = float(norm_std)
        self.norm_mean = float(norm_mean)
        self.velocity_scale = float(velocity_scale)
        self.eps = float(eps)
        self.domain_length = float(domain_length)
        self.vorticity_channel = int(vorticity_channel)
        self.nonlinearity = "velocity_magnitude"
        self._sym_cache: dict = {}

    # --- spectral velocity maps ------------------------------------------
    def _symbols(self, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = (ref.device, ref.dtype)
        if key not in self._sym_cache:
            self._sym_cache[key] = velocity_symbols(
                self.H, self.domain_length, device=ref.device, dtype=ref.dtype
            )
        return self._sym_cache[key]

    def _velocity(self, omega: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Linear maps ``omega [B,1,H,W] -> (u, v)`` (each ``[B,1,H,W]``)."""
        Du, Dv = self._symbols(omega)
        w_h = torch.fft.fft2(omega, dim=(-2, -1))
        u = torch.fft.ifft2(Du * w_h, dim=(-2, -1)).real
        v = torch.fft.ifft2(Dv * w_h, dim=(-2, -1)).real
        return u.to(omega.dtype), v.to(omega.dtype)

    def _omega(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1, self.H, self.W)
        w = x[:, self.vorticity_channel : self.vorticity_channel + 1]
        return w * self.norm_std + self.norm_mean

    def velocity(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw-unit velocity ``(u, v)`` of the (normalised) state ``x``."""
        return self._velocity(self._omega(x))

    def speed(self, x: torch.Tensor) -> torch.Tensor:
        """Smoothed speed field ``[B, 1, H, W]`` divided by ``velocity_scale``."""
        u, v = self.velocity(x)
        return torch.sqrt(u**2 + v**2 + self.eps**2) / self.velocity_scale

    # --- the map and its derivatives -------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(self.speed(x))

    def jvp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        u0, v0 = self.velocity(x)
        s = torch.sqrt(u0**2 + v0**2 + self.eps**2)
        dv = v.reshape(v.shape[0], -1, self.H, self.W)
        dw = dv[:, self.vorticity_channel : self.vorticity_channel + 1] * self.norm_std
        du1, dv1 = self._velocity(dw)
        ds = (u0 * du1 + v0 * dv1) / s / self.velocity_scale
        return self.inner(ds)

    def vjp(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        u0, v0 = self.velocity(x)
        s = torch.sqrt(u0**2 + v0**2 + self.eps**2)
        z = self.inner.transpose(u) / self.velocity_scale  # [B, 1, H, W]
        a = z / s
        # U, V are real antisymmetric (U^T = -U), so U^T(a u0) = -U(a u0).
        Ua, _ = self._velocity(a * u0)
        _, Vb = self._velocity(a * v0)
        g_omega = -(Ua + Vb) * self.norm_std
        x4 = x.reshape(x.shape[0], -1, self.H, self.W)
        out = torch.zeros_like(x4)
        out[:, self.vorticity_channel : self.vorticity_channel + 1] = g_omega
        return out.reshape(x.shape)


__all__ = [
    "NS_VORTICITY_NORM_STD",
    "VelocityMagnitudeOperator",
    "velocity_symbols",
]
