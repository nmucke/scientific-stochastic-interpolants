from typing import List

import torch
import torch.nn as nn

from scisinterpolant.models.conv_next import MultipleConvNextBlocks


class UNet(nn.Module):
    """UNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: List[int],
        cond_dim: int,
        multiplier: int = 2,
        num_blocks: int = 2,
    ) -> None:
        """Initialize UNet."""
        super(UNet, self).__init__()

        self.encoder_conv_blocks = nn.ModuleList(
            [
                MultipleConvNextBlocks(
                    in_channels=in_channels if i == 0 else hidden_channels[i - 1],
                    out_channels=hidden_channels[i],
                    cond_dim=cond_dim,
                    multiplier=multiplier,
                    num_blocks=num_blocks,
                )
                for i in range(len(hidden_channels))
            ]
        )

        self.down_blocks = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=hidden_channels[i],
                    out_channels=hidden_channels[i + 1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                )
                for i in range(len(hidden_channels) - 1)
            ]
        )

        self.bottleneck_conv_block = MultipleConvNextBlocks(
            in_channels=hidden_channels[-1],
            out_channels=hidden_channels[-1],
            cond_dim=cond_dim,
            multiplier=multiplier,
            num_blocks=num_blocks,
        )

        self.decoder_conv_blocks = nn.ModuleList(
            [
                MultipleConvNextBlocks(
                    in_channels=hidden_channels[i],
                    out_channels=hidden_channels[i - 1],
                    cond_dim=cond_dim,
                    multiplier=multiplier,
                    num_blocks=num_blocks,
                )
                for i in range(len(hidden_channels) - 1, 0, -1)
            ]
        )

        self.up_blocks = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=hidden_channels[i],
                    out_channels=hidden_channels[i - 1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                )
                for i in range(len(hidden_channels) - 1, 0, -1)
            ]
        )

        self.output_conv = nn.Conv2d(
            in_channels=hidden_channels[0],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for conv_block, down_block in zip(self.encoder_conv_blocks, self.down_blocks):
            x = conv_block(x, cond)
            x = down_block(x)

        x = self.bottleneck_conv_block(x, cond)

        for conv_block, up_block in zip(self.decoder_conv_blocks, self.up_blocks):
            x = conv_block(x, cond)
            x = up_block(x)

        x = self.output_conv(x)

        return x
