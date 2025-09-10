import torch
import torch.nn as nn
from einops import einsum, rearrange

from scisinterpolant.architectures.conv_next import MultipleConvNextBlocks


class SpatialAttention(nn.Module):
    """
    Spatial Attention.

    Based on https://github.com/tum-pbs/autoreg-pde-diffusion/blob/main/src/turbpred/model_diffusion_blocks.py
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
    ):
        """
        Initialize Spatial Attention.

        Args:
            dim (int): Dimension of the input.
            heads (int): Number of heads.
            dim_head (int): Dimension of the head.
        """
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        """
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum(q, k, "b h d i, b h d j -> b h i j")
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum(attn, v, "b h i j, b h d j -> b h i d")
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class BottleneckWithAttention(nn.Module):
    """
    Bottleneck with attention.

    Based on https://github.com/tum-pbs/autoreg-pde-diffusion/blob/main/src/turbpred/model_diffusion_blocks.py
    """

    def __init__(
        self,
        channels: int,
        conv_block_args: dict,
        spatial_attention_args: dict,
    ):
        """Initialize Bottleneck with attention."""
        super().__init__()

        self.bottleneck_conv_block_1 = MultipleConvNextBlocks(
            in_channels=channels,
            out_channels=channels,
            **conv_block_args,
        )

        self.spatial_attention = SpatialAttention(channels, **spatial_attention_args)

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
        x = self.spatial_attention(x)
        x = self.bottleneck_conv_block_2(x, cond, pars_cond)
        return x
