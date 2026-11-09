"""Physical constraints for the forecast-residual correction on uDALES.

The code base has no divergence-free projection, so this module implements a
small one, used as the correction's ``post_update(x_new, x_guided)``.

What is constrained, and why the INCREMENT
-------------------------------------------
The urban state is a HORIZONTAL SLICE ``(u, v, w, thl)`` of a 3D LES. On a slice
``du/dx + dv/dy = -dw/dz``, which is not zero and is unknown from the slice: on
the test data the RMS horizontal divergence is about as large as the individual
derivative terms (central differences: 0.077 vs 0.091 1/s, sim_170 steps
100-109). Forcing the STATE to be horizontally divergence-free would therefore
be wrong. What the correction must not do is ADD divergence or mass in
obstacles. So the increment ``delta = x'' - x'`` (in physical units) is replaced
by its orthogonal projection onto

    { delta : delta = 0 in solid cells,  D (delta_u, delta_v) = 0 },

with ``D`` a discrete horizontal divergence. Then ``D x_out = D x'`` (the
guided sample's divergence, whatever the generative model produced) and the
obstacle cells keep the guided values. ``w`` and ``thl`` are only masked.

The projection
--------------
Minimising ``||delta' - delta||^2`` under both constraints gives

    delta' = M (delta - D^T phi),    (D M D^T) phi = D M delta,

with ``M`` the 0/1 fluid mask (applied to ``u`` and ``v``). ``D M D^T`` is a
masked, PSD Poisson operator; it is solved by batched conjugate gradients (one
system per member), which needs no FFT and handles obstacles and non-periodic
edges. Its right-hand side lies in its range, so CG converges even though the
operator is singular. ``D`` is applied to every cell, so the increment also
carries no discrete flux into a solid cell.

Stencils: ``central`` (default; the data are cell-centred) or ``forward``
(staggered C-grid convention). Non-periodic axes are zero-padded: the
increment carries no net flux through the domain edge. Grid spacing is 1 m.

Axis convention (checked on the data): ``u`` varies along the W axis and ``v``
along the H axis. Next to walls normal to H the RMS of v drops to 0.09 against
0.31 for u (and the reverse for walls normal to W), so v is the H-normal
component. Both are configurable.

Checks: ``tests/test_forecast_correction_ns_urban.py`` (adjointness of ``D``,
divergence of the projected increment at CG tolerance, zero in solids,
idempotence, and a pipeline run with the correction on).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

Tensor = torch.Tensor


def _shift(a: Tensor, k: int, axis: int, periodic: bool) -> Tensor:
    """``out[i] = a[i + k]`` along ``axis``; zero outside unless periodic."""
    if periodic:
        return torch.roll(a, shifts=-k, dims=axis)
    out = torch.zeros_like(a)
    n = a.shape[axis]
    if abs(k) >= n:
        return out
    if k > 0:
        out.narrow(axis, 0, n - k).copy_(a.narrow(axis, k, n - k))
    else:
        out.narrow(axis, -k, n + k).copy_(a.narrow(axis, 0, n + k))
    return out


def derivative(
    a: Tensor, axis: int, periodic: bool, stencil: str = "central",
    spacing: float = 1.0, transpose: bool = False,
) -> Tensor:
    """Finite-difference derivative along ``axis`` (or its adjoint)."""
    if stencil == "central":
        d = (_shift(a, 1, axis, periodic) - _shift(a, -1, axis, periodic)) / (2.0 * spacing)
        return -d if transpose else d
    if stencil == "forward":
        if transpose:  # (S_+ - I)^T = S_- - I
            return (_shift(a, -1, axis, periodic) - a) / spacing
        return (_shift(a, 1, axis, periodic) - a) / spacing
    raise ValueError(f"stencil must be 'central' or 'forward', got {stencil!r}")


class IncrementProjection:
    """``post_update`` for uDALES: divergence-free, obstacle-free increments.

    Args:
        fluid_mask: ``[H, W]`` (or ``[C, H, W]``) bool, ``True`` = fluid.
        channel_std: ``[C]`` per-channel std of the model normalisation; the
            projection runs in physical units (``delta_phys = delta * std``).
        u_channel, v_channel: Channels of the horizontal velocity components.
        u_axis, v_axis: Spatial axis (``-1`` = W, ``-2`` = H) along which each
            component's derivative enters the divergence.
        periodic: ``(H, W)`` periodicity flags.
        stencil: ``"central"`` or ``"forward"``.
        project: Apply the divergence projection (else only the obstacle mask).
        mask_obstacles: Zero the increment of every channel in solid cells.
        tol, max_iter: CG relative-residual tolerance and iteration cap.
    """

    def __init__(
        self,
        fluid_mask: Tensor,
        channel_std: Sequence[float] | Tensor,
        u_channel: int = 0,
        v_channel: int = 1,
        u_axis: int = -1,
        v_axis: int = -2,
        periodic: Sequence[bool] = (False, False),
        stencil: str = "central",
        spacing: Sequence[float] = (1.0, 1.0),
        project: bool = True,
        mask_obstacles: bool = True,
        tol: float = 1e-8,
        max_iter: int = 2000,
    ) -> None:
        fm = fluid_mask.to(torch.bool)
        if fm.dim() == 3:
            fm = fm[0]
        self.fluid = fm.cpu()
        self.std = torch.as_tensor(channel_std, dtype=torch.float64).reshape(-1).cpu()
        self.u_channel, self.v_channel = int(u_channel), int(v_channel)
        self.u_axis, self.v_axis = int(u_axis), int(v_axis)
        self.periodic = (bool(periodic[0]), bool(periodic[1]))
        self.stencil = str(stencil)
        self.spacing = (float(spacing[0]), float(spacing[1]))
        self.project = bool(project)
        self.mask_obstacles = bool(mask_obstacles)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.last_info: dict = {}

    # -- operators on [N, H, W] fields ------------------------------------ #
    def _axis_args(self, axis: int) -> tuple[bool, float]:
        # axis -2 -> H (periodic[0], spacing[0]); -1 -> W.
        j = 0 if axis == -2 else 1
        return self.periodic[j], self.spacing[j]

    def divergence(self, u: Tensor, v: Tensor) -> Tensor:
        pu, hu = self._axis_args(self.u_axis)
        pv, hv = self._axis_args(self.v_axis)
        return derivative(u, self.u_axis, pu, self.stencil, hu) + derivative(
            v, self.v_axis, pv, self.stencil, hv
        )

    def divergence_t(self, phi: Tensor) -> tuple[Tensor, Tensor]:
        pu, hu = self._axis_args(self.u_axis)
        pv, hv = self._axis_args(self.v_axis)
        return (
            derivative(phi, self.u_axis, pu, self.stencil, hu, transpose=True),
            derivative(phi, self.v_axis, pv, self.stencil, hv, transpose=True),
        )

    def _interior(self, ref: Tensor) -> Tensor:
        """Fluid cells away from non-periodic domain edges (for diagnostics)."""
        keep = self.fluid.to(ref.device).clone()
        if not self.periodic[0]:
            keep[0] = False
            keep[-1] = False
        if not self.periodic[1]:
            keep[:, 0] = False
            keep[:, -1] = False
        return keep

    # -- projection -------------------------------------------------------- #
    def project_velocity(self, du: Tensor, dv: Tensor) -> tuple[Tensor, Tensor]:
        """Orthogonal projection of ``(du, dv)`` ``[N, H, W]`` (physical units)."""
        M = self.fluid.to(device=du.device, dtype=du.dtype)
        du, dv = du * M, dv * M
        b = self.divergence(du, dv)

        def A(p: Tensor) -> Tensor:
            pu, pv = self.divergence_t(p)
            return self.divergence(pu * M, pv * M)

        phi = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs = (r * r).sum(dim=(1, 2))
        # Scale for the stopping rule: ||b||, floored by the size of the increment
        # itself (over the finest spacing), so an already projected increment
        # (b ~ round-off, not in the range of A) stops at once instead of chasing
        # noise.
        h = min(self.spacing)
        inc_norm = ((du * du + dv * dv).sum(dim=(1, 2))).sqrt() / h
        b_norm = torch.maximum(rs.sqrt(), inc_norm).clamp_min(1e-300)
        iters = 0
        for iters in range(1, self.max_iter + 1):
            if bool(((rs.sqrt() / b_norm) <= self.tol).all()):
                iters -= 1
                break
            Ap = A(p)
            denom = (p * Ap).sum(dim=(1, 2))
            alpha = torch.where(denom > 0, rs / denom.clamp_min(1e-300), torch.zeros_like(rs))
            phi = phi + alpha[:, None, None] * p
            r = r - alpha[:, None, None] * Ap
            rs_new = (r * r).sum(dim=(1, 2))
            beta = torch.where(rs > 0, rs_new / rs.clamp_min(1e-300), torch.zeros_like(rs))
            p = r + beta[:, None, None] * p
            rs = rs_new
        gu, gv = self.divergence_t(phi)
        out_u, out_v = (du - gu) * M, (dv - gv) * M
        self.last_info = {
            "cg_iterations": int(iters),
            "cg_relative_residual": float((rs.sqrt() / b_norm).max()),
        }
        return out_u, out_v

    def __call__(self, x_new: Tensor, x_guided: Tensor) -> Tensor:
        """``post_update``: ``x_guided + P(x_new - x_guided)``."""
        dtype = x_new.dtype
        std = self.std.to(x_new.device)[None, :, None, None]
        delta = (x_new.to(torch.float64) - x_guided.to(torch.float64)) * std
        if self.project:
            du, dv = self.project_velocity(delta[:, self.u_channel], delta[:, self.v_channel])
            delta[:, self.u_channel] = du
            delta[:, self.v_channel] = dv
        if self.mask_obstacles:
            delta = delta * self.fluid.to(device=delta.device, dtype=delta.dtype)
        return (x_guided.to(torch.float64) + delta / std).to(dtype)

    # -- diagnostics ------------------------------------------------------- #
    def _div_rms(self, x: Tensor, mean: Optional[Tensor], keep: Tensor) -> float:
        std = self.std.to(x.device)
        u = x[:, self.u_channel].to(torch.float64) * std[self.u_channel]
        v = x[:, self.v_channel].to(torch.float64) * std[self.v_channel]
        if mean is not None:
            u = u + float(mean[self.u_channel])
            v = v + float(mean[self.v_channel])
        div = self.divergence(u, v)
        return float(div[:, keep].pow(2).mean().sqrt())

    def step_diagnostics(self, x_guided: Tensor, x_new: Tensor, x_out: Tensor) -> dict:
        """Divergence and obstacle checks before and after the correction."""
        keep = self._interior(x_guided)
        solid = ~self.fluid.to(x_guided.device)
        std = self.std.to(x_guided.device)[None, :, None, None]
        raw = (x_new.to(torch.float64) - x_guided.to(torch.float64)) * std
        out = (x_out.to(torch.float64) - x_guided.to(torch.float64)) * std
        div_raw = self.divergence(raw[:, self.u_channel], raw[:, self.v_channel])
        div_out = self.divergence(out[:, self.u_channel], out[:, self.v_channel])
        raw_n = float(raw.pow(2).sum().sqrt())
        rec = {
            # Horizontal divergence of the states (physical units, 1/s); on a
            # horizontal slice these are -dw/dz, not zero. The correction should
            # leave the guided value unchanged: div_rms_out == div_rms_guided.
            "div_rms_guided": self._div_rms(x_guided, None, keep),
            "div_rms_corrected_raw": self._div_rms(x_new, None, keep),
            "div_rms_out": self._div_rms(x_out, None, keep),
            # Divergence of the increment itself (should be ~0 after projection).
            "increment_div_rms_raw": float(div_raw[:, keep].pow(2).mean().sqrt()),
            "increment_div_rms_out": float(div_out[:, keep].pow(2).mean().sqrt()),
            # Increment inside buildings, all channels (should be 0 after).
            "solid_increment_rms_raw": float(raw[:, :, solid].pow(2).mean().sqrt())
            if bool(solid.any()) else 0.0,
            "solid_increment_rms_out": float(out[:, :, solid].pow(2).mean().sqrt())
            if bool(solid.any()) else 0.0,
            "increment_rms_raw": float(raw.pow(2).mean().sqrt()),
            "increment_rms_out": float(out.pow(2).mean().sqrt()),
            "projection_change_rel": (
                float((out - raw).pow(2).sum().sqrt()) / raw_n if raw_n > 0 else 0.0
            ),
        }
        rec.update(self.last_info)
        return rec


def from_settings(
    settings: dict, fluid_mask: Tensor, channel_std: Sequence[float] | Tensor
) -> Optional[IncrementProjection]:
    """Build the projection from the ``forecast_correction`` block (None if off)."""
    project = bool(settings.get("project_divergence", True))
    mask = bool(settings.get("mask_obstacles", True))
    if not (project or mask):
        return None
    axes = {"H": -2, "W": -1}
    return IncrementProjection(
        fluid_mask=fluid_mask,
        channel_std=channel_std,
        u_channel=int(settings.get("u_channel", 0)),
        v_channel=int(settings.get("v_channel", 1)),
        u_axis=axes[str(settings.get("u_axis", "W"))],
        v_axis=axes[str(settings.get("v_axis", "H"))],
        periodic=tuple(settings.get("periodic") or (False, False)),
        stencil=str(settings.get("divergence_stencil", "central")),
        project=project,
        mask_obstacles=mask,
        tol=float(settings.get("cg_tol", 1e-8)),
        max_iter=int(settings.get("cg_max_iter", 2000)),
    )


__all__ = ["IncrementProjection", "derivative", "from_settings"]
