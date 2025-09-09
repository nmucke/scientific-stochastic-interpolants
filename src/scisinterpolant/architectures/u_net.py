import pdb
from typing import List

import torch
import torch.nn as nn

from scisinterpolant.architectures.architecture_utils import (
    get_blocks,
    get_cond_encoder,
    get_init_conv,
)
from scisinterpolant.architectures.conv_next import MultipleConvNextBlocks


class ConvDown(nn.Module):
    """Conv down block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize Conv down block."""
        super(ConvDown, self).__init__()

        self.down_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.down_conv(x)
        return x


class ConvUp(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize Conv up block."""
        super(ConvUp, self).__init__()

        self.up_conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.up_conv(x)
        return x


class UNet(nn.Module):
    """UNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: List[int],
        cond_dim: int,
        cond_embedding_dim: int,
        field_cond_channels: int | None = None,
        pars_cond_dim: int | None = None,
        pars_cond_embedding_dim: int | None = None,
        multiplier: int = 2,
        num_blocks: int = 2,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Initialize UNet.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            hidden_channels (List[int]): List of hidden channels.
            cond_dim (int): Dimension of the conditional input.
            cond_embedding_dim (int): Dimension of the conditional embedding.
            field_cond_channels (int): Number of channels in the field conditional input. Can be None.
            pars_cond_dim (int): Dimension of the pars conditional input. Can be None.
            pars_cond_embedding_dim (int): Dimension of the pars conditional embedding. Can be None.
            multiplier (int): Multiplier for the number of channels.
            num_blocks (int): Number of ConvNext blocks per layer.
            dropout_rate (float): Dropout rate.
        """
        super(UNet, self).__init__()

        self._fixed_conv_block_args = {
            "cond_dim": cond_embedding_dim,
            "multiplier": multiplier,
            "num_blocks": num_blocks,
            "pars_cond_dim": pars_cond_embedding_dim,
        }
        self._reverse_channels = hidden_channels[::-1]
        self.gelu = nn.GELU()

        self.dropout = nn.Dropout(dropout_rate)

        self.cond_encoder = get_cond_encoder(
            cond_dim=cond_dim,
            cond_embedding_dim=cond_embedding_dim,
        )

        # self.pars_cond_encoder handles None case in initialization
        self.pars_cond_encoder = get_cond_encoder(
            cond_dim=pars_cond_dim,
            cond_embedding_dim=pars_cond_embedding_dim,
        )

        self.init_conv = get_init_conv(
            in_channels=in_channels,
            out_channels=hidden_channels[0],
            field_cond_channels=field_cond_channels,
        )

        self.encoder_conv_blocks = get_blocks(
            module=MultipleConvNextBlocks,
            in_channels=hidden_channels[:-1],
            out_channels=hidden_channels[1:],
            **self._fixed_conv_block_args,
        )

        self.down_blocks = get_blocks(
            module=ConvDown,
            in_channels=hidden_channels[1:],
            out_channels=hidden_channels[1:],
        )

        self.bottleneck_conv_block = MultipleConvNextBlocks(
            in_channels=hidden_channels[-1],
            out_channels=hidden_channels[-1],
            **self._fixed_conv_block_args,  # type: ignore[arg-type]
        )

        self.up_blocks = get_blocks(
            module=ConvUp,
            in_channels=self._reverse_channels[:-1],
            out_channels=self._reverse_channels[:-1],
        )

        self.decoder_conv_blocks = get_blocks(
            module=MultipleConvNextBlocks,
            in_channels=self._reverse_channels[:-1],
            out_channels=self._reverse_channels[1:],
            **self._fixed_conv_block_args,
        )

        self.output_conv = nn.Conv2d(
            in_channels=hidden_channels[0],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor [B, C_in, H, W].
            cond (torch.Tensor): Conditional tensor [B, D].
            field_cond (torch.Tensor): Field conditional tensor [B, C_field_cond, H, W]. Can be None.
            pars_cond (torch.Tensor): pars conditional tensor [B, D_pars_cond]. Can be None.

        Returns:
            torch.Tensor: Output tensor [B, C_out, H, W].
        """
        x = self.init_conv(x, field_cond)

        cond = self.cond_encoder(cond)

        # self.pars_cond_encoder handles None case in initialization
        pars_cond = self.pars_cond_encoder(pars_cond)

        x_skip_list = []
        for conv_block, down_block in zip(self.encoder_conv_blocks, self.down_blocks):
            x = self.dropout(x)
            x = conv_block(x, cond, pars_cond)
            x_skip_list.append(x)
            x = down_block(x)

        x = self.dropout(x)
        x = self.bottleneck_conv_block(x, cond, pars_cond)
        x = self.dropout(x)

        for conv_block, up_block, x_skip in zip(
            self.decoder_conv_blocks,
            self.up_blocks,
            x_skip_list[::-1],
        ):
            x = up_block(x)
            x = self.dropout(x)
            x = x + x_skip
            x = conv_block(x, cond, pars_cond)

        x = self.output_conv(x)

        return x
