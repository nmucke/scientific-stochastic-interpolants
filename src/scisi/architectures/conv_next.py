import pdb

import torch
import torch.nn as nn
from einops import rearrange

from scisi.architectures.architecture_utils import AddCond, AddCondNone


class ConvNextBlock(nn.Module):
    """ConvNext block with conditional input."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        multiplier: int = 2,
        pars_cond_dim: int | None = None,
        padding: nn.Module = nn.ZeroPad2d,
        dropout_rate: float = 0.0,
        layer_scale: float = 1e-6,
    ) -> None:
        """
        Initialize ConvNext block with conditional input.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            cond_dim (int): Dimension of the conditional input.
            multiplier (int): Multiplier for the number of channels.
            pars_cond_dim (int): Dimension of the pars conditional input. Can be None.
            padding (nn.Module): Padding module.
            dropout_rate (float): Dropout rate.
            layer_scale (float): Layer scale.
        """
        super(ConvNextBlock, self).__init__()

        self.ds_conv = nn.Sequential(
            padding(3), nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=1)
        )

        self.add_cond = AddCond(cond_dim, in_channels)
        if pars_cond_dim is not None:
            self.add_pars_cond = AddCond(pars_cond_dim, in_channels)
        else:
            self.add_pars_cond = AddCondNone()

        self.conv_next = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            padding(1),
            nn.Conv2d(
                in_channels,
                in_channels * multiplier,
                kernel_size=3,
                stride=1,
            ),
            nn.GELU(),
            nn.GroupNorm(1, in_channels * multiplier),
            padding(1),
            nn.Conv2d(
                in_channels * multiplier,
                out_channels,
                kernel_size=3,
                stride=1,
            ),
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.layer_scale = nn.Parameter(torch.ones(out_channels, 1, 1) * layer_scale)

        self.res_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        pars_cond: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            cond (torch.Tensor): Conditional input tensor of shape (B, D).
        """
        res = self.res_conv(x)
        x = self.ds_conv(x)
        x = self.add_cond(x, cond)
        x = self.add_pars_cond(x, pars_cond)
        x = self.conv_next(x)
        x = x * self.layer_scale
        x = self.dropout(x)
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
        pars_cond_dim: int | None = None,
        num_blocks: int = 2,
        padding: nn.Module = nn.ZeroPad2d,
        dropout_rate: float = 0.0,
        layer_scale: float = 1e-6,
    ) -> None:
        """
        Initialize multiple ConvNext blocks with conditional input.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            cond_dim (int): Dimension of the conditional input.
            multiplier (int): Multiplier for the number of channels.
            pars_cond_dim (int): Dimension of the pars conditional input. Can be None.
            num_blocks (int): Number of ConvNext blocks.
            dropout_rate (float): Dropout rate.
            padding (nn.Module): Padding module.
            dropout_rate (float): Dropout rate.
            layer_scale (float): Layer scale.
        """
        super(MultipleConvNextBlocks, self).__init__()

        self.blocks = nn.ModuleList(
            [
                ConvNextBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    cond_dim=cond_dim,
                    multiplier=multiplier,
                    pars_cond_dim=pars_cond_dim,
                    padding=padding,
                    dropout_rate=dropout_rate,
                    layer_scale=layer_scale,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, pars_cond: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            cond (torch.Tensor): Conditional input tensor of shape (B, D).
            pars_cond (torch.Tensor): Pars conditional input tensor of shape (B, D).
        """
        for block in self.blocks:
            x = block(x, cond, pars_cond)
        return x
