import pdb

import torch
import torch.nn as nn
from einops import rearrange


class ConvNextBlock(nn.Module):
    """ConvNext block with conditional input."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        multiplier: int = 2,
    ) -> None:
        """
        Initialize ConvNext block with conditional input.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            cond_dim (int): Dimension of the conditional input.
            multiplier (int): Multiplier for the number of channels.
        """
        super(ConvNextBlock, self).__init__()

        self.ds_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=7, stride=1, padding=3
        )

        self.cond_mlp = nn.Linear(cond_dim, in_channels)

        self.conv_next = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(
                in_channels,
                in_channels * multiplier,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GELU(),
            nn.GroupNorm(1, in_channels * multiplier),
            nn.Conv2d(
                in_channels * multiplier,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self.res_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            cond (torch.Tensor): Conditional input tensor of shape (B, D).
        """
        res = self.res_conv(x)
        x = self.ds_conv(x)
        cond = self.cond_mlp(cond)
        x = x + rearrange(cond, "b c -> b c 1 1")
        x = self.conv_next(x)
        x = x + res
        return x


class MultipleConvNextBlocks(nn.Module):
    """Multiple ConvNext blocks with conditional input."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        multiplier: int = 2,
        num_blocks: int = 2,
    ) -> None:
        """
        Initialize multiple ConvNext blocks with conditional input.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            cond_dim (int): Dimension of the conditional input.
            multiplier (int): Multiplier for the number of channels.
            num_blocks (int): Number of ConvNext blocks.
        """
        super(MultipleConvNextBlocks, self).__init__()

        self.blocks = nn.ModuleList(
            [
                ConvNextBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    cond_dim=cond_dim,
                    multiplier=multiplier,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            cond (torch.Tensor): Conditional input tensor of shape (B, D).
        """
        for block in self.blocks:
            x = block(x, cond)
        return x
