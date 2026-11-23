"""Spatial covariance operators for the generative (interpolant) noise.

The stochastic interpolants and their Brownian drivers draw white Gaussians
``z`` and colour them with a fixed symmetric positive semi-definite operator
``C`` through ``C^{1/2} z``. ``C`` acts on the channel and spatial indices of a
``[B, C, *spatial]`` tensor only, never on the batch or on generative time, so
it commutes with everything the driver does in time: the martingale
``M_t = C^{1/2} int_0^t gamma_r dW_r`` has ``Var(M_t - M_s) = (q(t) - q(s)) C``.

Assumption shared by every class here: additive, state-independent noise.
"""

from typing import Optional

import numpy as np
import torch
from torch import nn


class NoiseCovariance(nn.Module):
    """Symmetric positive semi-definite operator C acting on [B, C, *spatial].

    Subclasses implement ``sqrt`` (``C^{1/2} z``) and ``apply`` (``C x``);
    ``solve`` (the pseudo-inverse ``C^+ x``) is optional.
    """

    #: True only for the identity, so callers can keep today's exact code path.
    is_identity = False

    def sqrt(self, z: torch.Tensor) -> torch.Tensor:
        """Return C^{1/2} z."""
        raise NotImplementedError

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Return C x."""
        raise NotImplementedError

    def solve(self, x: torch.Tensor) -> torch.Tensor:
        """Return C^+ x (pseudo-inverse); optional."""
        raise NotImplementedError(f"{type(self).__name__} does not implement solve")


class IdentityCovariance(NoiseCovariance):
    """C = Id: independent unit-variance noise per degree of freedom.

    ``sqrt`` returns its input unchanged (the same tensor object), so every
    code path reproduces the white-noise behaviour bit for bit.
    """

    is_identity = True

    def sqrt(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def solve(self, x: torch.Tensor) -> torch.Tensor:
        return x


class FourierDiagonalCovariance(NoiseCovariance):
    """Stationary covariance on a periodic grid, diagonal in the Fourier basis.

    ``C = F^* diag(S) F`` per channel, with ``F`` the orthonormal
    (``norm="ortho"``) real FFT over all spatial axes. ``S[c, k]`` is the
    variance of the orthonormal Fourier coefficient ``k`` of channel ``c``,
    i.e. ``E|F(C^{1/2} z)_k|^2 = S_k`` for white ``z``.

    Assumption: a periodic domain (stationary, translation-invariant
    covariance) and no correlation between channels. This is the
    periodic-grid instance of ``NoiseCovariance``; non-periodic or masked
    domains need a different basis.

    Args:
        spectrum: real, non-negative array on the rfft grid, shape
            ``[C or 1, *spatial[:-1], spatial[-1] // 2 + 1]``.
        spectrum_path: ``.npz`` file holding the spectrum instead (written by
            ``research/ito_maps/experiments/estimate_noise_spectrum.py``).
        spectrum_key: array name inside ``spectrum_path``.

    The output is always real. On the self-conjugate planes of the last
    axis (index 0 and, for even length, the Nyquist index) the modes ``k`` and
    ``-k`` are the same real degree of freedom; ``S`` is averaged over that
    pair so that ``C^{1/2}`` is exactly symmetric and ``sqrt(sqrt(x)) =
    apply(x)``. A spectrum estimated from real fields is already symmetric
    there and is not changed.
    """

    def __init__(
        self,
        spectrum=None,
        spectrum_path: Optional[str] = None,
        spectrum_key: str = "spectrum",
    ) -> None:
        super().__init__()
        if (spectrum is None) == (spectrum_path is None):
            raise ValueError("Give exactly one of spectrum and spectrum_path.")
        if spectrum_path is not None:
            with np.load(spectrum_path) as archive:
                spectrum = archive[spectrum_key]
        spectrum = torch.as_tensor(np.asarray(spectrum), dtype=torch.float32)
        if spectrum.ndim < 2:
            raise ValueError("spectrum must have shape [C, *rfft_spatial_shape].")
        if not torch.isfinite(spectrum).all() or (spectrum < 0).any():
            raise ValueError("spectrum must be finite and non-negative.")
        self.register_buffer("spectrum", spectrum)

    def _symmetric_spectrum(self, last_length: int) -> torch.Tensor:
        """Average S over the pairs (k, -k) inside the self-conjugate planes."""
        spectrum = self.spectrum
        mirrored = spectrum
        for axis in range(1, spectrum.ndim - 1):  # k -> -k mod N on full axes
            mirrored = torch.roll(torch.flip(mirrored, (axis,)), 1, axis)
        planes = [0] + ([last_length // 2] if last_length % 2 == 0 else [])
        symmetric = spectrum.clone()
        symmetric[..., planes] = 0.5 * (spectrum[..., planes] + mirrored[..., planes])
        return symmetric

    def _multiply(self, x: torch.Tensor, power: float) -> torch.Tensor:
        """Return F^* diag(S^power) F x, with S^power = 0 where S = 0."""
        dims = tuple(range(2, x.ndim))
        coefficients = torch.fft.rfftn(x, dim=dims, norm="ortho")
        shape = self.spectrum.shape
        if shape[1:] != coefficients.shape[2:] or shape[0] not in (1, x.shape[1]):
            raise ValueError(
                f"spectrum shape {tuple(shape)} does not match the rfft "
                f"grid {tuple(coefficients.shape[1:])} of the state."
            )
        spectrum = self._symmetric_spectrum(x.shape[-1])
        spectrum = spectrum.to(device=x.device, dtype=coefficients.real.dtype)
        if power < 0:
            positive = spectrum > 0
            factor = torch.where(positive, spectrum, torch.ones_like(spectrum))
            factor = torch.where(positive, factor**power, torch.zeros_like(spectrum))
        else:
            factor = spectrum**power
        return torch.fft.irfftn(
            coefficients * factor, s=x.shape[2:], dim=dims, norm="ortho"
        )

    def sqrt(self, z: torch.Tensor) -> torch.Tensor:
        return self._multiply(z, 0.5)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return self._multiply(x, 1.0)

    def solve(self, x: torch.Tensor) -> torch.Tensor:
        return self._multiply(x, -1.0)


class ProjectedCovariance(NoiseCovariance):
    """``P B P`` with ``B = covariance.sqrt`` and ``P = Id - L^+ L``.

    Removes the directions of a linear invariant ``L`` from the noise, so that
    ``L C^{1/2} = 0`` (needed for invariant-closed maps). ``invariant`` is any
    object with ``project_out`` (e.g. ``scisi.models.invariants``). If ``P``
    commutes with ``B`` (true for the identity and for the spatial mean with a
    Fourier-diagonal ``B``), ``sqrt(z)`` is a square root of ``P C P``.
    """

    def __init__(self, covariance: NoiseCovariance, invariant: nn.Module) -> None:
        super().__init__()
        self.covariance = covariance
        self.invariant = invariant

    def sqrt(self, z: torch.Tensor) -> torch.Tensor:
        project = self.invariant.project_out
        return project(self.covariance.sqrt(project(z)))

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return self.sqrt(self.sqrt(x))


def same_noise_covariance(first, second) -> bool:
    """True if two interpolations colour their noise with the same C.

    A missing ``noise_covariance`` counts as the identity. Compares the class
    and every buffer, so a map can be checked against its teacher.
    """
    first, second = (
        getattr(item, "noise_covariance", None) for item in (first, second)
    )
    first = IdentityCovariance() if first is None else first
    second = IdentityCovariance() if second is None else second
    if type(first) is not type(second):
        return False
    first_state, second_state = first.state_dict(), second.state_dict()
    return first_state.keys() == second_state.keys() and all(
        torch.equal(first_state[key].cpu(), second_state[key].cpu())
        for key in first_state
    )
