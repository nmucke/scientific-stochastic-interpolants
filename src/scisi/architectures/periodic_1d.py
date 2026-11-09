"""Drift network for a 1D periodic state, such as the Lorenz 96 lattice.

The network shares the drift interface of :class:`scisi.architectures.u_net.UNet`,

    forward(x, cond, field_history, field_cond, pars_cond) -> tensor shaped like x,

with ``x`` of shape ``[B, 1, 1, d]`` and ``field_history`` of shape
``[B, 1, 1, d, L]``. The pseudo-time ``cond`` may be ``[B, 1]``, ``[1, 1]`` or
``[1]``. ``field_cond`` and ``pars_cond`` are not supported and must be ``None``.

Everything is safe under ``torch.func.jvp``: no in-place operations on activations
and no batch statistics.

Symmetry: ``PeriodicResNet1d`` is exactly equivariant under every lattice shift.
A U-Net, a transformer and an MLP were compared against it once for the Lorenz 96
case and removed afterwards; they are in the git history at commit caa08af.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from scisi.architectures.embeddings import get_cond_encoder


def circular_pad(x: torch.Tensor, padding: int) -> torch.Tensor:
    """Pad the lattice axis of ``x`` [B, d, C] periodically on both sides."""
    if padding > x.shape[1]:
        raise ValueError(
            f"Circular padding {padding} exceeds the lattice size {x.shape[1]}."
        )
    if padding == 0:
        return x
    return torch.cat([x[:, -padding:], x, x[:, :padding]], dim=1)


class CircularConv1d(nn.Module):
    """Convolution on a ring, acting on channel-last ``x`` [B, d, C].

    Written as a matmul over sliding windows: on CPU builds without oneDNN,
    ``nn.Conv1d`` is about 100 times slower than this at these sizes. The padding
    is ``(kernel_size - 1) // 2`` on both sides.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> None:
        """Initialize the circular convolution."""
        super(CircularConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.linear = nn.Linear(kernel_size * in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution."""
        x = circular_pad(x, self.padding).unfold(1, self.kernel_size, 1)
        return self.linear(x.flatten(2))


class FiLM(nn.Module):
    """Normalise each site over its features, then scale and shift by the time."""

    def __init__(self, channels: int, cond_embedding_dim: int) -> None:
        """Initialize FiLM."""
        super(FiLM, self).__init__()
        self.channels = channels
        self.to_scale_shift = nn.Linear(cond_embedding_dim, 2 * channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply to channel-last ``x`` [B, d, C] with ``cond`` [B or 1, E]."""
        scale, shift = self.to_scale_shift(cond).unsqueeze(1).chunk(2, dim=-1)
        return F.layer_norm(x, (self.channels,)) * (1.0 + scale) + shift


class GatedConvBlock(nn.Module):
    """Residual block whose branch multiplies two convolutions of the input.

    The product ``conv_a(y) * gelu(conv_b(y))`` forms the quadratic interactions
    between neighbouring sites directly, and stacking blocks raises their degree.
    Normalisation is per site, so it adds no coupling between sites.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        cond_embedding_dim: int,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize the gated convolution block."""
        super(GatedConvBlock, self).__init__()
        self.film = FiLM(channels, cond_embedding_dim)
        self.conv = CircularConv1d(channels, 2 * channels, kernel_size)
        self.out = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply the block to ``x`` [B, d, C]."""
        a, b = self.conv(self.film(x, cond)).chunk(2, dim=-1)
        return x + self.dropout(self.out(a * F.gelu(b)))


class PeriodicResNet1d(nn.Module):
    """Full-resolution residual network of gated circular convolutions.

    Exactly equivariant under every lattice shift. The receptive field spans
    ``(num_blocks + 1) * (kernel_size - 1) + 1`` sites; a lattice larger than that
    is rejected at the first forward pass.
    """

    def __init__(
        self,
        len_field_history: int = 1,
        hidden_channels: int = 96,
        num_blocks: int = 4,
        kernel_size: int = 7,
        cond_embedding_dim: int = 64,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Initialize the residual network.

        Args:
            len_field_history (int): Number of history states L.
            hidden_channels (int): Channels of the residual stream.
            num_blocks (int): Number of gated blocks.
            kernel_size (int): Odd kernel size of all convolutions.
            cond_embedding_dim (int): Dimension of the time embedding.
            dropout_rate (float): Dropout rate on the residual branches.
        """
        super(PeriodicResNet1d, self).__init__()
        self.len_field_history = len_field_history
        self.in_channels = 1 + len_field_history
        self.cond_encoder = get_cond_encoder(
            cond_dim=1, cond_embedding_dim=cond_embedding_dim
        )
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}.")
        self.receptive_field = (num_blocks + 1) * (kernel_size - 1) + 1
        self.lift = CircularConv1d(self.in_channels, hidden_channels, kernel_size)
        self.blocks = nn.ModuleList(
            GatedConvBlock(
                hidden_channels, kernel_size, cond_embedding_dim, dropout_rate
            )
            for _ in range(num_blocks)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )

    def _prepare_inputs(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        field_history: Optional[torch.Tensor],
        field_cond: Optional[torch.Tensor],
        pars_cond: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the fields [B, d, 1 + L] and the time embedding [B or 1, E]."""
        if field_history is None:
            raise ValueError("field_history is required.")
        if field_cond is not None or pars_cond is not None:
            raise ValueError("field_cond and pars_cond are not supported.")
        if field_history.shape[-1] != self.len_field_history:
            raise ValueError(
                f"Expected {self.len_field_history} history states, "
                f"got {field_history.shape[-1]}."
            )
        batch_size = x.shape[0]
        if cond.numel() not in (1, batch_size):
            raise ValueError(
                f"Cannot broadcast cond {tuple(cond.shape)} to batch {batch_size}."
            )
        history = rearrange(field_history, "b 1 1 d l -> b d l")
        fields = torch.cat(
            [
                rearrange(x, "b 1 1 d -> b d 1"),
                history.expand(batch_size, -1, -1),
            ],
            dim=-1,
        )
        return fields, self.cond_encoder(cond.reshape(-1, 1))

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass, see the module docstring for the shapes."""
        h, cond = self._prepare_inputs(x, cond, field_history, field_cond, pars_cond)
        if h.shape[1] > self.receptive_field:
            raise ValueError(
                f"Receptive field {self.receptive_field} does not cover the "
                f"lattice of {h.shape[1]} sites."
            )
        h = self.lift(h)
        for block in self.blocks:
            h = block(h, cond)
        return self.head(h).reshape(x.shape)
