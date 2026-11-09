"""Navier--Stokes pieces of the forecast-residual correction.

The generic wiring is ``common/forecast_correction.py``. Here: the ``post_update``
that keeps the domain-mean vorticity (``zero_mean_increment``) and the per-step
enstrophy-spectrum diagnostics (``diagnostics: true``).
"""

from __future__ import annotations

from typing import Sequence

import torch

#: Integer-wavenumber band edges for the NS enstrophy spectra (128^2 grid).
NS_BAND_EDGES: tuple[float, ...] = (0.5, 2, 4, 8, 16, 32, 64, 91)


def enstrophy_band_spectra(
    omega: torch.Tensor, edges: Sequence[float] = NS_BAND_EDGES
) -> torch.Tensor:
    """Per-member enstrophy ``1/2 sum |omega_hat(k)|^2`` in radial bands.

    ``omega``: ``[N, C, H, W]`` (channel 0 is used) in any units; the ratio of two
    spectra is unit-free and the mean (k=0) is excluded by the first edge.
    Returns ``[N, n_bands]``.
    """
    w = omega[:, 0].to(torch.float64)
    N, H, W = w.shape
    wh = torch.fft.fft2(w) / (H * W)
    power = 0.5 * wh.abs() ** 2
    ky = torch.fft.fftfreq(H, d=1.0 / H).to(w.device)
    kx = torch.fft.fftfreq(W, d=1.0 / W).to(w.device)
    kk = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (kk >= lo) & (kk < hi)
        out.append(power[:, sel].sum(dim=1))
    return torch.stack(out, dim=1)


def ns_step_diagnostics(
    x_guided: torch.Tensor, x_new: torch.Tensor, x_out: torch.Tensor
) -> dict:
    """Enstrophy per band before (guided x') and after (x'') the correction."""
    before = enstrophy_band_spectra(x_guided)  # [N, B]
    after = enstrophy_band_spectra(x_out)
    member_ratio = after.sum(1) / before.sum(1).clamp_min(1e-300)
    inc = (x_out - x_guided).to(torch.float64)
    raw = (x_new - x_guided).to(torch.float64)
    return {
        "band_edges": list(NS_BAND_EDGES),
        "enstrophy_before": before.mean(0).tolist(),
        "enstrophy_after": after.mean(0).tolist(),
        "enstrophy_ratio_bands": (after.mean(0) / before.mean(0).clamp_min(1e-300)).tolist(),
        "member_enstrophy_ratio_min": float(member_ratio.min()),
        "member_enstrophy_ratio_median": float(member_ratio.median()),
        "member_enstrophy_ratio_max": float(member_ratio.max()),
        # Domain mean of the increment (vorticity on the torus has a fixed mean).
        # A localised gain is NOT a combination of whole anomaly fields, so the raw
        # update can shift the mean; `zero_mean_increment` removes that shift.
        "increment_domain_mean_abs_raw": float(raw.mean(dim=(2, 3)).abs().max()),
        "increment_domain_mean_abs": float(inc.mean(dim=(2, 3)).abs().max()),
        "increment_rms_raw": float(raw.pow(2).mean().sqrt()),
        "increment_rms": float(inc.pow(2).mean().sqrt()),
    }


def zero_mean_increment(x_new: torch.Tensor, x_guided: torch.Tensor) -> torch.Tensor:
    """NS ``post_update``: remove the domain mean of the increment, per channel.

    Orthogonal projection of ``x'' - x'`` onto zero-mean fields, so the guided
    sample's domain-mean vorticity is kept exactly (periodic domain).
    """
    delta = (x_new - x_guided).to(torch.float64)
    delta = delta - delta.mean(dim=(2, 3), keepdim=True)
    return (x_guided.to(torch.float64) + delta).to(x_new.dtype)

