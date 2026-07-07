"""Brownian path simulation and encoding for Ito maps.

Implements the genuinely new numerics of the Ito-map paper
(arXiv:2606.11156): simulating a Brownian path W on [0, 1] together with the
martingale part M_t = int_0^t sigma_u dW_u, and compressing W into a small set
of conditioning features (Karhunen-Loeve coefficients or dyadic/Haar
martingale increments) that ride the existing ``field_cond`` input pathway.

Two path representations are provided:

* ``path`` mode simulates increments on a uniform grid (paper style). Memory
  is ~``num_grid_points`` times the state size.
* ``kl`` mode never materializes the path: it samples K i.i.d. Gaussian
  coefficient fields ``xi_n`` of the Karhunen-Loeve expansion
  ``W(u) = sum_n xi_n phi_n(u)`` and evaluates
  ``M_t - M_s = sum_n xi_n int_s^t sigma_u phi_n'(u) du`` with precomputed
  cumulative integrals.
"""

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

DEFAULT_NUM_GRID_POINTS = 64
DEFAULT_NUM_KL_TERMS = 16
DEFAULT_DENSE_GRID_SIZE = 1024


class SigmaSchedule(nn.Module):
    """Diffusion schedule sigma_t of the SDE the Ito map learns to jump."""

    #: Whether the schedule is identically zero (deterministic flow map).
    is_zero: bool = False

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate sigma at time t."""
        raise NotImplementedError


class ZeroSigmaSchedule(SigmaSchedule):
    """sigma = 0: deterministic flow map (one-step flow matching)."""

    is_zero = True

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate sigma at time t."""
        return torch.zeros_like(t)


class PaperSigmaSchedule(SigmaSchedule):
    """The paper's canonical schedule sigma_t = sqrt(scale * (1 - t))."""

    def __init__(self, scale: float = 2.0) -> None:
        """Initialize the schedule."""
        super(PaperSigmaSchedule, self).__init__()
        self.scale = scale

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate sigma at time t."""
        return torch.sqrt(self.scale * (1 - t).clamp(min=0.0))


class GammaMatchedSigmaSchedule(SigmaSchedule):
    """sigma_t = gamma_t of a stochastic interpolation.

    Matches the diffusion the repo's Follmer interpolants are trained with, so
    the diagonal drift needs no score correction.
    """

    def __init__(self, interpolation: nn.Module) -> None:
        """Initialize from a stochastic interpolation exposing ``gamma``."""
        super(GammaMatchedSigmaSchedule, self).__init__()
        self.interpolation = interpolation

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate sigma at time t."""
        return self.interpolation.gamma(t)


def _kl_frequencies(
    num_terms: int, device: torch.device | str = "cpu", dtype: torch.dtype = None
) -> torch.Tensor:
    """Frequencies omega_n = (n - 1/2) * pi of the Brownian KL basis."""
    n = torch.arange(1, num_terms + 1, device=device, dtype=dtype or torch.float32)
    return (n - 0.5) * math.pi


def _interp_last_dim(values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate ``values`` along its last dim at per-sample times.

    Args:
        values (torch.Tensor): [B, ..., G+1] cumulative values on the uniform
            grid ``linspace(0, 1, G+1)``.
        t (torch.Tensor): [B, 1] query times in [0, 1].

    Returns:
        torch.Tensor: [B, ...] interpolated values.
    """
    grid_size = values.shape[-1] - 1
    pos = (t.reshape(-1) * grid_size).clamp(0, grid_size)
    idx0 = pos.floor().long().clamp(max=grid_size - 1)
    frac = pos - idx0

    batch = values.shape[0]
    view_shape = [batch] + [1] * (values.ndim - 1)
    idx = idx0.view(view_shape).expand(*values.shape[:-1], 1)
    v0 = values.gather(-1, idx).squeeze(-1)
    v1 = values.gather(-1, idx + 1).squeeze(-1)
    frac = frac.view(view_shape[:-1])
    return v0 + frac * (v1 - v0)


class BrownianSample(ABC):
    """One batch of Brownian paths with their martingale part.

    State tensors have shape [B, C, H, W] (any trailing spatial layout works;
    only the batch-first convention is assumed).
    """

    @abstractmethod
    def w_at(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate W at per-sample times t [B, 1] -> [B, C, H, W]."""

    @abstractmethod
    def w_at_times(self, times: torch.Tensor) -> torch.Tensor:
        """Evaluate W at shared scalar times [Q] -> [B, C, H, W, Q]."""

    @abstractmethod
    def martingale_increment(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """M_t - M_s for per-sample times s, t [B, 1] -> [B, C, H, W]."""

    @abstractmethod
    def kl_coefficients(self, num_coeffs: int) -> torch.Tensor:
        """First ``num_coeffs`` KL coefficients -> [B, K, C, H, W]."""


class GridBrownianSample(BrownianSample):
    """Brownian path simulated on a uniform grid over [0, 1]."""

    def __init__(self, increments: torch.Tensor, sigma_schedule: SigmaSchedule) -> None:
        """Initialize from grid increments.

        Args:
            increments (torch.Tensor): [B, C, H, W, G] i.i.d. N(0, 1/G)
                Brownian increments on the uniform grid.
            sigma_schedule (SigmaSchedule): Diffusion schedule.
        """
        self.increments = increments
        self.grid_size = increments.shape[-1]

        device, dtype = increments.device, increments.dtype
        self.grid = torch.linspace(0, 1, self.grid_size + 1, device=device, dtype=dtype)

        zero = torch.zeros_like(increments[..., :1])
        self.w = torch.cat([zero, torch.cumsum(increments, dim=-1)], dim=-1)

        # Ito (left-endpoint) accumulation of M_t = int sigma_u dW_u.
        sigma_left = sigma_schedule(self.grid[:-1])
        self.m = torch.cat(
            [zero, torch.cumsum(sigma_left * increments, dim=-1)], dim=-1
        )

    def w_at(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate W at per-sample times t [B, 1] -> [B, C, H, W]."""
        return _interp_last_dim(self.w, t)

    def w_at_times(self, times: torch.Tensor) -> torch.Tensor:
        """Evaluate W at shared scalar times [Q] -> [B, C, H, W, Q]."""
        pos = (times.to(self.w.device) * self.grid_size).clamp(0, self.grid_size)
        idx0 = pos.floor().long().clamp(max=self.grid_size - 1)
        frac = pos - idx0
        v0 = self.w.index_select(-1, idx0)
        v1 = self.w.index_select(-1, idx0 + 1)
        return v0 + frac * (v1 - v0)

    def martingale_increment(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """M_t - M_s for per-sample times s, t [B, 1] -> [B, C, H, W]."""
        return _interp_last_dim(self.m, t) - _interp_last_dim(self.m, s)

    def kl_coefficients(self, num_coeffs: int) -> torch.Tensor:
        """First ``num_coeffs`` KL coefficients -> [B, K, C, H, W].

        Computed by Ito quadrature xi_n = sqrt(2) * sum_i cos(omega_n u_i) dW_i,
        which is exact for the discrete path.
        """
        device, dtype = self.increments.device, self.increments.dtype
        freqs = _kl_frequencies(num_coeffs, device=device, dtype=dtype)
        cos_mat = math.sqrt(2.0) * torch.cos(
            freqs[:, None] * self.grid[:-1][None, :]
        )  # [K, G]
        xi = torch.tensordot(self.increments, cos_mat, dims=([-1], [1]))
        return torch.movedim(xi, -1, 1)


class KLBrownianSample(BrownianSample):
    """Closed-form Karhunen-Loeve representation of the Brownian path.

    Never materializes the path: holds K coefficient fields and evaluates W
    and M from the (truncated) KL expansion.
    """

    def __init__(
        self,
        xi: torch.Tensor,
        cumulative_integrals: torch.Tensor,
    ) -> None:
        """Initialize from KL coefficients.

        Args:
            xi (torch.Tensor): [B, K, C, H, W] i.i.d. N(0, 1) KL coefficients.
            cumulative_integrals (torch.Tensor): [G+1, K] precomputed
                I_n(t) = int_0^t sigma_u phi_n'(u) du on a dense uniform grid.
        """
        self.xi = xi
        self.num_terms = xi.shape[1]
        self.cumulative_integrals = cumulative_integrals
        self.freqs = _kl_frequencies(self.num_terms, device=xi.device, dtype=xi.dtype)

    def _contract(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Contract per-sample coefficients [B, K] against xi -> [B, C, H, W]."""
        view_shape = list(coeffs.shape) + [1] * (self.xi.ndim - 2)
        return (self.xi * coeffs.view(view_shape)).sum(dim=1)

    def w_at(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate W at per-sample times t [B, 1] -> [B, C, H, W]."""
        phi = math.sqrt(2.0) * torch.sin(self.freqs * t) / self.freqs  # [B, K]
        return self._contract(phi)

    def w_at_times(self, times: torch.Tensor) -> torch.Tensor:
        """Evaluate W at shared scalar times [Q] -> [B, C, H, W, Q]."""
        times = times.to(self.xi.device).to(self.xi.dtype)
        phi = math.sqrt(2.0) * torch.sin(self.freqs[None, :] * times[:, None])
        phi = phi / self.freqs[None, :]  # [Q, K]
        return torch.tensordot(self.xi, phi, dims=([1], [1]))

    def _integrals_at(self, t: torch.Tensor) -> torch.Tensor:
        """Interpolate I_n at per-sample times t [B, 1] -> [B, K]."""
        grid_size = self.cumulative_integrals.shape[0] - 1
        pos = (t.reshape(-1) * grid_size).clamp(0, grid_size)
        idx0 = pos.floor().long().clamp(max=grid_size - 1)
        frac = (pos - idx0).unsqueeze(-1)
        v0 = self.cumulative_integrals[idx0]
        v1 = self.cumulative_integrals[idx0 + 1]
        return v0 + frac * (v1 - v0)

    def martingale_increment(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """M_t - M_s for per-sample times s, t [B, 1] -> [B, C, H, W]."""
        coeffs = self._integrals_at(t) - self._integrals_at(s)
        return self._contract(coeffs)

    def kl_coefficients(self, num_coeffs: int) -> torch.Tensor:
        """First ``num_coeffs`` KL coefficients -> [B, K, C, H, W]."""
        if num_coeffs > self.num_terms:
            raise ValueError(
                f"Requested {num_coeffs} KL coefficients but the sampler only "
                f"holds {self.num_terms} (increase num_kl_terms)."
            )
        return self.xi[:, :num_coeffs]


class BrownianPathSampler:
    """Simulates Brownian paths and their martingale part on [0, 1].

    Args:
        sigma_schedule (SigmaSchedule): Diffusion schedule sigma_t.
        num_grid_points (int): Grid resolution for ``path`` mode.
        mode (str): ``"path"`` (simulate a discrete path) or ``"kl"``
            (closed-form Karhunen-Loeve mode, never materializes the path).
        num_kl_terms (int): Number of KL terms held in ``kl`` mode.
        dense_grid_size (int): Dense grid used to precompute the cumulative
            integrals I_n(t) in ``kl`` mode.
    """

    def __init__(
        self,
        sigma_schedule: SigmaSchedule,
        num_grid_points: int = DEFAULT_NUM_GRID_POINTS,
        mode: str = "kl",
        num_kl_terms: int = DEFAULT_NUM_KL_TERMS,
        dense_grid_size: int = DEFAULT_DENSE_GRID_SIZE,
    ) -> None:
        """Initialize the sampler."""
        if mode not in ("path", "kl"):
            raise ValueError(f"Unknown Brownian sampler mode: {mode}")

        self.sigma_schedule = sigma_schedule
        self.num_grid_points = num_grid_points
        self.mode = mode
        self.num_kl_terms = num_kl_terms
        self.dense_grid_size = dense_grid_size

        if mode == "kl":
            self._cumulative_integrals = self._compute_cumulative_integrals()

    def _compute_cumulative_integrals(self) -> torch.Tensor:
        """Precompute I_n(t) = int_0^t sigma_u phi_n'(u) du on a dense grid.

        phi_n'(u) = sqrt(2) cos(omega_n u) with omega_n = (n - 1/2) pi.

        Returns:
            torch.Tensor: [G+1, K] cumulative integrals (float32).
        """
        u = torch.linspace(0, 1, self.dense_grid_size + 1, dtype=torch.float64)
        freqs = _kl_frequencies(self.num_kl_terms, dtype=torch.float64)
        sigma = self.sigma_schedule(u).to(torch.float64)
        integrand = sigma[:, None] * math.sqrt(2.0) * torch.cos(
            freqs[None, :] * u[:, None]
        )  # [G+1, K]
        cumulative = torch.cumulative_trapezoid(integrand, u, dim=0)
        zero = torch.zeros(1, self.num_kl_terms, dtype=torch.float64)
        return torch.cat([zero, cumulative], dim=0).to(torch.float32)

    def sample(
        self, state_shape: torch.Size, device: torch.device | str
    ) -> BrownianSample:
        """Sample a batch of Brownian paths.

        Args:
            state_shape: Shape of the state tensor [B, C, H, W].
            device: Device to sample on.

        Returns:
            BrownianSample: Path representation matching the configured mode.
        """
        if self.mode == "path":
            increments = torch.randn(
                *state_shape, self.num_grid_points, device=device
            ) * math.sqrt(1.0 / self.num_grid_points)
            return GridBrownianSample(increments, self.sigma_schedule)

        xi = torch.randn(
            state_shape[0], self.num_kl_terms, *state_shape[1:], device=device
        )
        return KLBrownianSample(xi, self._cumulative_integrals.to(device))


class BrownianEncoder(nn.Module):
    """Compresses a Brownian path into conditioning feature channels.

    Outputs are channel-stacked [B, K*C, H, W] tensors so they ride the
    existing ``field_cond`` input pathway of the drift networks.
    """

    @property
    def num_features_per_channel(self) -> int:
        """Number of feature channels produced per state channel."""
        raise NotImplementedError

    def forward(self, sample: BrownianSample) -> torch.Tensor:
        """Encode the Brownian sample into [B, K*C, H, W] features."""
        raise NotImplementedError


class KLEncoder(BrownianEncoder):
    """Karhunen-Loeve encoder: the first K sine-basis coefficients."""

    def __init__(self, num_coeffs: int = 5) -> None:
        """Initialize the encoder."""
        super(KLEncoder, self).__init__()
        self.num_coeffs = num_coeffs

    @property
    def num_features_per_channel(self) -> int:
        """Number of feature channels produced per state channel."""
        return self.num_coeffs

    def forward(self, sample: BrownianSample) -> torch.Tensor:
        """Encode the Brownian sample into [B, K*C, H, W] features."""
        xi = sample.kl_coefficients(self.num_coeffs)  # [B, K, C, H, W]
        return torch.flatten(xi, start_dim=1, end_dim=2)


class DyadicEncoder(BrownianEncoder):
    """Dyadic/Haar encoder: midpoint-displacement martingale increments.

    Features per state channel: W(1) plus the Haar midpoint displacements
    c_{j,k} = W(m) - (W(l) + W(r)) / 2 over the dyadic intervals of levels
    j = 0..depth-1, for a total of 2**depth channels.
    """

    def __init__(self, depth: int = 4) -> None:
        """Initialize the encoder."""
        super(DyadicEncoder, self).__init__()
        self.depth = depth

    @property
    def num_features_per_channel(self) -> int:
        """Number of feature channels produced per state channel."""
        return 2**self.depth

    def forward(self, sample: BrownianSample) -> torch.Tensor:
        """Encode the Brownian sample into [B, 2**depth * C, H, W] features."""
        num_points = 2**self.depth
        times = torch.arange(num_points + 1, dtype=torch.float32) / num_points
        w = sample.w_at_times(times)  # [B, C, H, W, 2**depth + 1]

        features = [w[..., -1]]
        for level in range(self.depth):
            stride = num_points >> level
            for k in range(2**level):
                left = k * stride
                right = left + stride
                mid = (left + right) // 2
                features.append(w[..., mid] - 0.5 * (w[..., left] + w[..., right]))

        stacked = torch.stack(features, dim=1)  # [B, 2**depth, C, H, W]
        return torch.flatten(stacked, start_dim=1, end_dim=2)
