import math
import pdb

import hydra
import torch
import torch.nn as nn
from einops import einsum, rearrange
from torch.nn.functional import scaled_dot_product_attention

from scisi.architectures.conv_next import MultipleConvNextBlocks
from scisi.architectures.rotary_positional_embedding import (
    RotaryPositionalEmbeddings,
    VisionRotaryPositionalEmbeddings,
)


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert image to patches.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W)
        patch_size (int): Size of each patch (P x P)

    Returns:
        torch.Tensor: Patched tensor of shape (B, L, C*P*P)
                     where L = (H//P) * (W//P) is the number of patches
    """
    B, C, H, W = x.shape

    # Calculate number of patches
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    L = num_patches_h * num_patches_w

    # Reshape to patches: (B, C, H, W) -> (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
    x = x.view(B, C, num_patches_h, patch_size, num_patches_w, patch_size)

    # Rearrange to: (B, num_patches_h, num_patches_w, C, patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)

    # Reshape to: (B, L, C*P*P)
    x = x.contiguous().view(B, L, C * patch_size * patch_size)

    return x


def unpatchify(
    x: torch.Tensor, channels: int, height: int, width: int, patch_size: int
) -> torch.Tensor:
    """
    Convert patches back to image.

    Args:
        x (torch.Tensor): Patched tensor of shape (B, L, C*P*P)
        channels (int): Number of channels C
        height (int): Original height H
        width (int): Original width W
        patch_size (int): Size of each patch (P x P)

    Returns:
        torch.Tensor: Reconstructed image of shape (B, C, H, W)
    """
    B, L, patch_dim = x.shape

    # Calculate number of patches
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size

    # Reshape to: (B, num_patches_h, num_patches_w, C, patch_size, patch_size)
    x = x.view(B, num_patches_h, num_patches_w, channels, patch_size, patch_size)

    # Rearrange to: (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5)

    # Reshape to: (B, C, H, W)
    x = x.contiguous().view(B, channels, height, width)

    return x


class PositionalEncoding(nn.Module):
    """Positional Encoding."""

    def __init__(self, dim: int, max_seq_length: int = 10000) -> None:
        """Initialize Positional Encoding."""
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x + self.pe[:, : x.size(1)]


def split_heads(x: torch.Tensor, heads: int, dim_head: int) -> torch.Tensor:
    """Split heads."""
    batch_size, seq_length, _ = x.size()
    return x.view(batch_size, seq_length, heads, dim_head).transpose(
        1, 2
    )  # (B, L, H, D)


def combine_heads(x: torch.Tensor, heads: int, dim_head: int) -> torch.Tensor:
    """Combine heads."""
    batch_size, _, seq_length, dim_head = x.size()
    return x.transpose(1, 2).contiguous().view(batch_size, seq_length, dim_head * heads)


class Attention(nn.Module):
    """
    Attention.

    Based on https://github.com/tum-pbs/autoreg-pde-diffusion/blob/main/src/turbpred/model_diffusion_blocks.py
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Initialize Attention.

        Args:
            dim (int): Dimension of the input.
            heads (int): Number of heads.
            dropout_rate (float): Dropout rate.
        """
        super(Attention, self).__init__()
        self.heads = heads
        self.dim_head = dim // heads

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        self.to_out = nn.Linear(dim, dim)

        self.dropout_rate = dropout_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            cond (torch.Tensor): Conditional tensor of shape (B, D).
            pars_cond (torch.Tensor): Pars conditional tensor of shape (B, D).
        """
        q = split_heads(self.W_q(x), self.heads, self.dim_head)  # (B, H, L, D)
        k = split_heads(self.W_k(x), self.heads, self.dim_head)  # (B, H, L, D)
        v = split_heads(self.W_v(x), self.heads, self.dim_head)  # (B, H, L, D)

        dropout_p = self.dropout_rate if self.training else 0.0
        out = scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = combine_heads(out, self.heads, self.dim_head)

        return self.to_out(out)


class AttentionWithRotaryPositionalEmbeddings(nn.Module):
    """
    Attention with Rotary Positional Embeddings.
    """

    def __init__(self, dim: int, heads: int = 4, dropout_rate: float = 0.0):
        """Initialize Attention with Rotary Positional Embeddings."""
        super(AttentionWithRotaryPositionalEmbeddings, self).__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.dropout_rate = dropout_rate

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        self.to_out = nn.Linear(dim, dim)

        self.rotary_emb_q = RotaryPositionalEmbeddings(self.dim_head)
        self.rotary_emb_k = RotaryPositionalEmbeddings(self.dim_head)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        """
        q = split_heads(self.W_q(x), self.heads, self.dim_head)  # (B, H, L, D)
        k = split_heads(self.W_k(x), self.heads, self.dim_head)  # (B, H, L, D)
        v = split_heads(self.W_v(x), self.heads, self.dim_head)  # (B, H, L, D)

        q = self.rotary_emb_q(q)
        k = self.rotary_emb_k(k)

        out = scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_rate)

        out = combine_heads(out, self.heads, self.dim_head)

        return self.to_out(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention.
    """

    def __init__(
        self,
        channels: int,
        pos_embedding_type: str = "rotary",
        embedding_dim: int | None = None,
        heads: int = 4,
        patch_size: int = 4,
        dropout_rate: float = 0.0,
    ):
        """Initialize Spatial Attention."""
        super(SpatialAttention, self).__init__()

        self.patch_size = patch_size

        if embedding_dim is None:
            self.embedding_dim = channels * patch_size * patch_size
            self.embedding = nn.Identity()
            self.to_out = nn.Identity()
        else:
            self.embedding_dim = embedding_dim
            self.embedding = nn.Linear(
                channels * patch_size * patch_size, self.embedding_dim
            )
            self.to_out = nn.Linear(
                self.embedding_dim, channels * patch_size * patch_size
            )

        self.attention_args: dict = {
            "dim": self.embedding_dim,
            "heads": heads,
            "dropout_rate": dropout_rate,
        }

        if pos_embedding_type == "rotary":
            self.attention = AttentionWithRotaryPositionalEmbeddings(
                **self.attention_args
            )
        elif pos_embedding_type == "additive":
            self.attention = nn.Sequential(
                PositionalEncoding(self.embedding_dim), Attention(**self.attention_args)
            )

        self.norm1 = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            cond (torch.Tensor): Conditional tensor of shape (B, D).
            pars_cond (torch.Tensor): Pars conditional tensor of shape (B, D).
        """
        b, c, h, w = x.size()

        x = patchify(x, self.patch_size)  # (B, C, H, W) -> (B, L, C*p*p)

        x = self.embedding(x)  # (B, L, C*p*p) -> (B, L, D)

        attn_out = self.attention(x)  # (B, L, D)

        x = self.norm1(x + attn_out)  # (B, L, D)
        x = self.to_out(x)  # (B, L, D) -> (B, L, C*p*p)
        return unpatchify(x, c, h, w, self.patch_size)  # (B, L, C*p*p) -> (B, C, H, W)


class SpatialTransformerBlock(nn.Module):
    """
    Spatial Transformer Block.
    """

    def __init__(
        self,
        channels: int,
        pos_embedding_type: str = "rotary",
        embedding_dim: int = 128,
        heads: int = 4,
        patch_size: int = 4,
        dropout_rate: float = 0.0,
        mlp_dim: int = 1024,
    ):
        """Initialize Spatial Transformer Block."""
        super(SpatialTransformerBlock, self).__init__()

        self.patch_size = patch_size

        self.embedding = nn.Linear(channels * patch_size * patch_size, embedding_dim)

        self.attention_args: dict = {
            "dim": embedding_dim,
            "heads": heads,
            "dropout_rate": dropout_rate,
        }

        if pos_embedding_type == "rotary":
            self.attention = AttentionWithRotaryPositionalEmbeddings(
                **self.attention_args
            )
        elif pos_embedding_type == "additive":
            self.attention = nn.Sequential(
                PositionalEncoding(embedding_dim), Attention(**self.attention_args)
            )

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embedding_dim),
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.to_out = nn.Linear(embedding_dim, channels * patch_size * patch_size)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            cond (torch.Tensor): Conditional tensor of shape (B, D).
            pars_cond (torch.Tensor): Pars conditional tensor of shape (B, D).
        """
        b, c, h, w = x.size()

        x = patchify(x, self.patch_size)  # (B, C, H, W) -> (B, L, C*p*p)
        x = self.embedding(x)  # (B, L, C*p*p) -> (B, L, D)

        attn_out = self.attention(x)  # (B, L, D)

        x = self.norm1(x + attn_out)  # (B, L, D)
        mlp_out = self.mlp(x)  # (B, L, D)
        x = self.norm2(x + mlp_out)  # (B, L, D)
        x = self.to_out(x)  # (B, L, D) -> (B, L, C*p*p)
        return unpatchify(x, c, h, w, self.patch_size)  # (B, L, C*p*p) -> (B, C, H, W)


class BottleneckWithAttention(nn.Module):
    """
    Bottleneck with attention.

    Based on https://github.com/tum-pbs/autoreg-pde-diffusion/blob/main/src/turbpred/model_diffusion_blocks.py
    """

    def __init__(
        self,
        channels: int,
        conv_block_args: dict,
        attention: dict,
        attention_in_layer: bool,
    ):
        """Initialize Bottleneck with attention."""
        super(BottleneckWithAttention, self).__init__()

        self.bottleneck_conv_block_1 = MultipleConvNextBlocks(
            in_channels=channels,
            out_channels=channels,
            **conv_block_args,
        )

        if attention_in_layer:
            attention["_target_"] = attention.pop("target")
            self.spatial_attention = hydra.utils.instantiate(
                attention,
                channels=channels,
            )
            attention["target"] = attention.pop("_target_")
        else:
            self.spatial_attention = lambda x, cond, pars_cond: x

        self.bottleneck_conv_block_2 = MultipleConvNextBlocks(
            in_channels=channels,
            out_channels=channels,
            **conv_block_args,
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        pars_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        x = self.bottleneck_conv_block_1(x, cond, pars_cond)
        x = self.spatial_attention(x, cond, pars_cond)
        x = self.bottleneck_conv_block_2(x, cond, pars_cond)
        return x
