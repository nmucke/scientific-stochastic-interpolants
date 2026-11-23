"""Exact linear invariants of a forecast, L y = l(h) almost surely (WP2).

An invariant-closed Itô map uses the average drift

    G = (Id - L^+ L) N + L^+ (l_t - l_s) / (t - s),   l_t = alpha_t L c + beta_t l(h),

with N the network output, so that L Phi_{s,t}(x) = L x + l_t - l_s for every
parameter value and step count. This needs a driver with L C^{1/2} = 0 (see
``scisi.sampling.noise_covariance.ProjectedCovariance``).
"""

import torch
from torch import nn


class LinearInvariant(nn.Module):
    """A linear map L on [B, C, *spatial] states with pseudo-inverse L^+.

    Assumption: L y = l(h) holds exactly for every forecast y. By default the
    invariant is conserved from the latest history frame, l(h) = L c.
    """

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Return L x, shape [B, ...]."""
        raise NotImplementedError

    def lift(self, value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Return L^+ value, shaped like ``like``."""
        raise NotImplementedError

    def project_out(self, x: torch.Tensor) -> torch.Tensor:
        """Return (Id - L^+ L) x."""
        return x - self.lift(self.value(x), like=x)

    def target(self, field_history: torch.Tensor) -> torch.Tensor:
        """Return l(h) from the history [B, C, *spatial, L]: here L of its last frame."""
        return self.value(field_history[..., -1])


class SpatialMeanInvariant(LinearInvariant):
    """Spatial mean per channel: L x = mean over all spatial axes, [B, C].

    L = 1^T / N per channel, so L^+ = 1 (a constant field) and
    (Id - L^+ L) x = x - mean(x). Any spatial rank, grid shape and channel
    count; periodicity is not assumed.
    """

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=tuple(range(2, x.ndim)))

    def lift(self, value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return value.reshape(*value.shape, *([1] * (like.ndim - 2))).expand_as(like)
