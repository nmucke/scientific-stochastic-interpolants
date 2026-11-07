"""Drift networks for a 1D periodic state, such as the Lorenz 96 lattice.

All networks share the drift interface of :class:`scisi.architectures.u_net.UNet`,

    forward(x, cond, field_history, field_cond, pars_cond) -> tensor shaped like x,

with ``x`` of shape ``[B, 1, 1, d]`` and ``field_history`` of shape
``[B, 1, 1, d, L]``. The pseudo-time ``cond`` may be ``[B, 1]``, ``[1, 1]`` or
``[1]``. ``field_cond`` and ``pars_cond`` are not supported and must be ``None``.

Everything is safe under ``torch.func.jvp``: no in-place operations on activations,
no batch statistics, and attention is written out instead of calling the fused
kernels, which implement neither forward-mode AD nor double-backward on CUDA.

Symmetry: ``PeriodicResNet1d`` and ``PeriodicTransformer1d`` are exactly equivariant
under every lattice shift. ``PeriodicUNet1d`` is equivariant only under shifts that
are multiples of its total downsampling factor. ``MLP1d`` is not equivariant.
"""

from typing import List, Optional, Sequence

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
        stride: int = 1,
    ) -> None:
        """Initialize the circular convolution."""
        super(CircularConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.linear = nn.Linear(kernel_size * in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution."""
        x = circular_pad(x, self.padding).unfold(1, self.kernel_size, self.stride)
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


class PeriodicDriftNetwork(nn.Module):
    """Input handling shared by the networks in this module."""

    def __init__(self, len_field_history: int, cond_embedding_dim: int) -> None:
        """Initialize the time encoder."""
        super(PeriodicDriftNetwork, self).__init__()
        self.len_field_history = len_field_history
        self.in_channels = 1 + len_field_history
        self.cond_encoder = get_cond_encoder(
            cond_dim=1, cond_embedding_dim=cond_embedding_dim
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


class PeriodicResNet1d(PeriodicDriftNetwork):
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
        super(PeriodicResNet1d, self).__init__(len_field_history, cond_embedding_dim)
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


class PeriodicUNet1d(PeriodicDriftNetwork):
    """U-Net of gated circular convolutions.

    Every level halves the lattice with a strided convolution, so ``d`` must be
    divisible by ``factor = 2 ** (len(hidden_channels) - 1)``, which is checked at
    the first forward pass. Upsampling repeats each site and applies a circular
    convolution, because a transposed convolution would not wrap around the ring.

    Striding fixes a sampling phase, so the network is equivariant ONLY under shifts
    that are multiples of ``factor``. Any other shift changes the output by more than
    a roll (see ``tests/test_periodic_1d.py``).
    """

    def __init__(
        self,
        len_field_history: int = 1,
        hidden_channels: Sequence[int] = (64, 96, 128),
        num_bottleneck_blocks: int = 2,
        kernel_size: int = 5,
        cond_embedding_dim: int = 64,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Initialize the U-Net.

        Args:
            len_field_history (int): Number of history states L.
            hidden_channels (Sequence[int]): Channels per level, finest first.
            num_bottleneck_blocks (int): Number of gated blocks at the coarsest level.
            kernel_size (int): Odd kernel size of the gated blocks.
            cond_embedding_dim (int): Dimension of the time embedding.
            dropout_rate (float): Dropout rate on the residual branches.
        """
        super(PeriodicUNet1d, self).__init__(len_field_history, cond_embedding_dim)
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}.")
        self.factor = 2 ** (len(hidden_channels) - 1)
        # The bottleneck alone is a lower bound on the receptive field.
        self.receptive_field = num_bottleneck_blocks * (kernel_size - 1) * self.factor

        def block(channels: int) -> GatedConvBlock:
            return GatedConvBlock(
                channels, kernel_size, cond_embedding_dim, dropout_rate
            )

        fine, coarse = hidden_channels[:-1], hidden_channels[1:]
        self.lift = CircularConv1d(self.in_channels, hidden_channels[0], kernel_size)
        self.encoder_blocks = nn.ModuleList(block(c) for c in fine)
        self.down_convs = nn.ModuleList(
            CircularConv1d(c_in, c_out, 4, stride=2)
            for c_in, c_out in zip(fine, coarse)
        )
        self.bottleneck_blocks = nn.ModuleList(
            block(hidden_channels[-1]) for _ in range(num_bottleneck_blocks)
        )
        self.up_convs = nn.ModuleList(
            CircularConv1d(c_in, c_out, 3)
            for c_in, c_out in zip(coarse[::-1], fine[::-1])
        )
        self.decoder_blocks = nn.ModuleList(block(c) for c in fine[::-1])
        self.head = nn.Sequential(
            nn.Linear(hidden_channels[0], hidden_channels[0]),
            nn.GELU(),
            nn.Linear(hidden_channels[0], 1),
        )

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
        if h.shape[1] % self.factor != 0:
            raise ValueError(
                f"Lattice size {h.shape[1]} is not divisible by the total "
                f"downsampling factor {self.factor}."
            )
        if h.shape[1] > self.receptive_field:
            raise ValueError(
                f"Bottleneck receptive field {self.receptive_field} does not cover "
                f"the lattice of {h.shape[1]} sites."
            )

        h = self.lift(h)
        skips = []
        for block, down_conv in zip(self.encoder_blocks, self.down_convs):
            h = block(h, cond)
            skips.append(h)
            h = down_conv(h)
        for block in self.bottleneck_blocks:
            h = block(h, cond)
        for up_conv, block, skip in zip(
            self.up_convs, self.decoder_blocks, skips[::-1]
        ):
            h = up_conv(h.repeat_interleave(2, dim=1)) + skip
            h = block(h, cond)
        return self.head(h).reshape(x.shape)


class RingAttention(nn.Module):
    """Multi-head self-attention with a relative bias indexed by ``(i - j) mod d``."""

    def __init__(self, channels: int, num_heads: int, max_dim_state: int) -> None:
        """Initialize the attention."""
        super(RingAttention, self).__init__()
        if channels % num_heads != 0:
            raise ValueError(f"{num_heads} heads do not divide {channels} channels.")
        self.num_heads = num_heads
        self.max_dim_state = max_dim_state
        self.qkv = nn.Linear(channels, 3 * channels)
        self.out = nn.Linear(channels, channels)
        # Random, so that heads start out focused on different offsets. A short
        # Lorenz 96 fit was clearly worse from a zero initialisation.
        self.relative_bias = nn.Parameter(torch.randn(num_heads, max_dim_state))

    def _bias(self, dim_state: int, device: torch.device) -> torch.Tensor:
        """Return the bias [heads, d, d].

        Offsets are wrapped to ``[-d/2, d/2)`` so that a table entry means the same
        signed offset for every ``d``. The sign is kept: Lorenz 96 is not reflection
        symmetric.
        """
        if dim_state > self.max_dim_state:
            raise ValueError(
                f"Lattice size {dim_state} exceeds max_dim_state {self.max_dim_state}."
            )
        sites = torch.arange(dim_state, device=device)
        offset = (sites[:, None] - sites[None, :] + dim_state // 2) % dim_state
        offset = offset - dim_state // 2 + self.max_dim_state // 2
        return self.relative_bias[:, offset]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention to tokens ``x`` [B, d, C]."""
        q, k, v = rearrange(
            self.qkv(x), "b d (n h c) -> n b h d c", n=3, h=self.num_heads
        )
        logits = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5
        attention = torch.softmax(logits + self._bias(x.shape[1], x.device), dim=-1)
        return self.out(rearrange(attention @ v, "b h d c -> b d (h c)"))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with FiLM time conditioning and a gated MLP."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        multiplier: int,
        cond_embedding_dim: int,
        max_dim_state: int,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize the transformer block."""
        super(TransformerBlock, self).__init__()
        self.attention_film = FiLM(channels, cond_embedding_dim)
        self.attention = RingAttention(channels, num_heads, max_dim_state)
        self.mlp_film = FiLM(channels, cond_embedding_dim)
        self.mlp_in = nn.Linear(channels, 2 * multiplier * channels)
        self.mlp_out = nn.Linear(multiplier * channels, channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply the block to tokens ``x`` [B, d, C]."""
        x = x + self.dropout(self.attention(self.attention_film(x, cond)))
        a, b = self.mlp_in(self.mlp_film(x, cond)).chunk(2, dim=-1)
        return x + self.dropout(self.mlp_out(a * F.gelu(b)))


class PeriodicTransformer1d(PeriodicDriftNetwork):
    """Transformer with one token per lattice site.

    Exactly equivariant under every lattice shift: tokens come from a circular
    convolution and position enters only through a bias on ``(i - j) mod d``.
    """

    def __init__(
        self,
        len_field_history: int = 1,
        hidden_channels: int = 96,
        num_blocks: int = 4,
        num_heads: int = 8,
        multiplier: int = 2,
        kernel_size: int = 5,
        cond_embedding_dim: int = 64,
        max_dim_state: int = 64,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Initialize the transformer.

        Args:
            len_field_history (int): Number of history states L.
            hidden_channels (int): Token width.
            num_blocks (int): Number of transformer blocks.
            num_heads (int): Number of attention heads.
            multiplier (int): Width multiplier of the gated MLP.
            kernel_size (int): Odd kernel size of the token convolution.
            cond_embedding_dim (int): Dimension of the time embedding.
            max_dim_state (int): Largest lattice the bias table can index.
            dropout_rate (float): Dropout rate on the residual branches.
        """
        super(PeriodicTransformer1d, self).__init__(
            len_field_history, cond_embedding_dim
        )
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}.")
        self.tokenizer = CircularConv1d(self.in_channels, hidden_channels, kernel_size)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                hidden_channels,
                num_heads,
                multiplier,
                cond_embedding_dim,
                max_dim_state,
                dropout_rate,
            )
            for _ in range(num_blocks)
        )
        self.head_film = FiLM(hidden_channels, cond_embedding_dim)
        self.head = nn.Linear(hidden_channels, 1)

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
        h = self.tokenizer(h)
        for block in self.blocks:
            h = block(h, cond)
        return self.head(self.head_film(h, cond)).reshape(x.shape)


class MLP1d(PeriodicDriftNetwork):
    """Plain MLP on the flattened state, history and time embedding.

    Baseline only: it is NOT translation equivariant and must learn the lattice
    symmetry from data. Its size is tied to ``dim_state``.
    """

    def __init__(
        self,
        dim_state: int,
        len_field_history: int = 1,
        hidden_dim: int = 512,
        num_layers: int = 3,
        cond_embedding_dim: int = 64,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Initialize the MLP.

        Args:
            dim_state (int): Lattice size d.
            len_field_history (int): Number of history states L.
            hidden_dim (int): Width of the hidden layers.
            num_layers (int): Number of hidden layers.
            cond_embedding_dim (int): Dimension of the time embedding.
            dropout_rate (float): Dropout rate after the hidden layers.
        """
        super(MLP1d, self).__init__(len_field_history, cond_embedding_dim)
        layers: List[nn.Module] = []
        width = self.in_channels * dim_state + cond_embedding_dim
        for _ in range(num_layers):
            layers += [
                nn.Linear(width, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            ]
            width = hidden_dim
        self.mlp = nn.Sequential(*layers, nn.Linear(width, dim_state))

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
        h = torch.cat([h.flatten(1), cond.expand(h.shape[0], -1)], dim=1)
        return self.mlp(h).reshape(x.shape)
