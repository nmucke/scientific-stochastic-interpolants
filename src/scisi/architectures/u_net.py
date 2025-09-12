import pdb
from typing import List

import hydra
import torch
import torch.nn as nn

from scisi.architectures.architecture_utils import get_blocks, get_init_conv
from scisi.architectures.attention import BottleneckWithAttention
from scisi.architectures.conv_next import MultipleConvNextBlocks
from scisi.architectures.embeddings import get_cond_encoder


class ConvDown(nn.Module):
    """Conv down block."""

    def __init__(
        self, in_channels: int, out_channels: int, padding: str = "torch.nn.ZeroPad2d"
    ) -> None:
        """Initialize Conv down block."""
        super(ConvDown, self).__init__()

        self.down_conv = nn.Sequential(
            hydra.utils.instantiate({"_target_": padding, "padding": 1}),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.down_conv(x)
        return x


class ConvUp(nn.Module):
    """Conv up block."""

    def __init__(
        self, in_channels: int, out_channels: int, padding: str = "torch.nn.ZeroPad2d"
    ) -> None:
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
        len_field_history: int,
        field_cond_channels: int | None = None,
        pars_cond_dim: int | None = None,
        pars_cond_embedding_dim: int | None = None,
        multiplier: int = 2,
        num_blocks: int = 2,
        dropout_rate: float = 0.0,
        padding: str = "torch.nn.ZeroPad2d",
        spatial_attention: bool = False,
        bottleneck_heads: int = 4,
        bottleneck_dim_head: int = 64,
    ) -> None:
        """
        Initialize UNet.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            hidden_channels (List[int]): List of hidden channels.
            cond_dim (int): Dimension of the conditional input.
            cond_embedding_dim (int): Dimension of the conditional embedding.
            len_field_history (int): Length of the field history.
            field_cond_channels (int): Number of channels in the field conditional input. Can be None.
            pars_cond_dim (int): Dimension of the pars conditional input. Can be None.
            pars_cond_embedding_dim (int): Dimension of the pars conditional embedding. Can be None.
            multiplier (int): Multiplier for the number of channels.
            num_blocks (int): Number of ConvNext blocks per layer.
            dropout_rate (float): Dropout rate.
            padding (str): Padding module.
            spatial_attention (bool): Whether to use spatial attention.
            bottleneck_heads (int): Number of heads for the bottleneck attention.
            bottleneck_dim_head (int): Dimension of the bottleneck attention head.
        """
        super(UNet, self).__init__()

        self._fixed_conv_block_args = {
            "cond_dim": cond_embedding_dim,
            "multiplier": multiplier,
            "num_blocks": num_blocks,
            "pars_cond_dim": pars_cond_embedding_dim,
            "padding": padding,
            "dropout_rate": dropout_rate,
        }
        self._reverse_channels = hidden_channels[::-1]
        self.len_field_history = len_field_history

        self.cond_encoder = get_cond_encoder(
            cond_dim=cond_dim,
            cond_embedding_dim=cond_embedding_dim,
        )

        # self.pars_cond_encoder handles None case in initialization
        self.pars_cond_encoder = get_cond_encoder(
            cond_dim=pars_cond_dim,
            cond_embedding_dim=pars_cond_embedding_dim,
        )

        init_conv_args = dict(
            self._fixed_conv_block_args, cond_dim=None, pars_cond_dim=None
        )
        init_conv_args.pop("num_blocks")
        self.init_conv = get_init_conv(
            in_channels=in_channels,
            out_channels=hidden_channels[0],
            len_field_history=len_field_history,
            field_cond_channels=field_cond_channels,
            **init_conv_args,
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

        if spatial_attention:
            self.bottleneck_block = BottleneckWithAttention(
                channels=hidden_channels[-1],
                conv_block_args=self._fixed_conv_block_args,
                spatial_attention_args={
                    "heads": bottleneck_heads,
                    "dim_head": bottleneck_dim_head,
                },
            )
        else:
            self.bottleneck_block = MultipleConvNextBlocks(
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
        field_history: torch.Tensor | None = None,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor [B, C_in, H, W].
            cond (torch.Tensor): Conditional tensor [B, D].
            field_history (torch.Tensor): Field history tensor [B, C, H, W, L]. Can be None.
            field_cond (torch.Tensor): Field conditional tensor [B, C_field_cond, H, W]. Can be None.
            pars_cond (torch.Tensor): pars conditional tensor [B, D_pars_cond]. Can be None.

        Returns:
            torch.Tensor: Output tensor [B, C_out, H, W].
        """
        x = self.init_conv(x, field_history, field_cond)

        cond = self.cond_encoder(cond)

        # self.pars_cond_encoder handles None case in initialization
        pars_cond = self.pars_cond_encoder(pars_cond)

        x_skip_list = []
        for conv_block, down_block in zip(self.encoder_conv_blocks, self.down_blocks):
            x = conv_block(x, cond, pars_cond)
            x_skip_list.append(x)
            x = down_block(x)

        x = self.bottleneck_block(x, cond, pars_cond)

        for conv_block, up_block, x_skip in zip(
            self.decoder_conv_blocks,
            self.up_blocks,
            x_skip_list[::-1],
        ):
            x = up_block(x)
            x = x + x_skip
            x = conv_block(x, cond, pars_cond)

        x = self.output_conv(x)

        return x
