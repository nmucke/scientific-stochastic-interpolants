import pdb
from typing import List

import torch
import torch.nn as nn
from einops import rearrange

from scisi.architectures.embeddings import FourierScalarEncoder


class InitConvWithHistory(nn.Module):
    """Init conv with field cond."""

    def __init__(
        self, in_channels: int, out_channels: int, len_field_history: int
    ) -> None:
        """Initialize init conv with field cond."""
        super(InitConvWithHistory, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels + len_field_history * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        field_history = rearrange(field_history, "b c h w t -> b (t c) h w")
        x = torch.cat([x, field_history], dim=1)
        x = self.conv(x)
        return x


class InitConvWithFieldCond(nn.Module):
    """Init conv with field cond."""

    def __init__(
        self, in_channels: int, out_channels: int, field_cond_channels: int
    ) -> None:
        """Initialize init conv with field cond."""
        super(InitConvWithFieldCond, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels + field_cond_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        field_history: torch.Tensor | None = None,
        field_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        x = torch.cat([x, field_cond], dim=1)
        x = self.conv(x)
        return x


class InitConvWithFieldCondAndHistory(nn.Module):
    """Init conv with field cond and history."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        field_cond_channels: int,
        len_field_history: int,
    ) -> None:
        """Initialize init conv with field cond and history."""
        super(InitConvWithFieldCondAndHistory, self).__init__()

        self.history_conv = InitConvWithHistory(
            in_channels,
            in_channels + len_field_history * in_channels,
            len_field_history,
        )
        self.field_cond_conv = InitConvWithFieldCond(
            out_channels, out_channels, field_cond_channels
        )

    def forward(
        self,
        x: torch.Tensor,
        field_history: torch.Tensor | None = None,
        field_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        x = self.history_conv(x, field_history)
        x = self.field_cond_conv(x, field_cond)
        return x


class InitConv(nn.Module):
    """Init conv without field cond."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize init conv without field cond."""
        super(InitConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        field_history: torch.Tensor | None = None,
        field_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        return x


def get_init_conv(
    in_channels: int,
    out_channels: int,
    field_cond_channels: int | None = None,
    len_field_history: int | None = None,
) -> nn.Module:
    """
    Get initial convolution.

    Helper function to get the initial convolution that handles the field conditional.
    This is to avoid if else statements in the forward pass.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        field_cond_channels (int): Number of field conditional channels.
    """
    if (field_cond_channels is not None) and (len_field_history is None):
        return InitConvWithFieldCond(in_channels, out_channels, field_cond_channels)
    elif (len_field_history is not None) and (field_cond_channels is None):
        return InitConvWithHistory(in_channels, out_channels, len_field_history)
    elif (field_cond_channels is not None) and (len_field_history is not None):
        return InitConvWithFieldCondAndHistory(
            in_channels, out_channels, field_cond_channels, len_field_history
        )
    else:
        return InitConv(in_channels, out_channels)


def get_blocks(  # type: ignore[no-untyped-def]
    module: nn.Module,
    in_channels: List[int],
    out_channels: List[int],
    **kwargs,
) -> nn.ModuleList:
    """
    Get blocks.

    Args:
        module (nn.Module): Module to use.
        in_channels (List[int]): List of input channels.
        out_channels (List[int]): List of output channels.
    """
    return nn.ModuleList(
        [
            module(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                **kwargs,
            )
            for i in range(len(in_channels))
        ]
    )


class AddCond(nn.Module):
    """Add cond."""

    def __init__(self, cond_dim: int, cond_embedding_dim: int) -> None:
        """Initialize add cond."""
        super(AddCond, self).__init__()

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_embedding_dim),
        )

        self.rearrange = lambda x: rearrange(x, "b c -> b c 1 1")

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        cond = self.cond_mlp(cond)
        cond = self.rearrange(cond)
        x = x + cond
        return x


class AddCondNone(nn.Module):
    """Add cond none."""

    def __init__(self) -> None:
        """Initialize add cond none."""
        super(AddCondNone, self).__init__()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x


def get_cond_encoder(  # type: ignore[no-untyped-def]
    cond_dim: int | None = None,
    cond_embedding_dim: int | None = None,
    **kwargs,
) -> nn.Module:
    """Get pars cond embedding."""
    if cond_dim == 1 and cond_embedding_dim is not None:
        return nn.Sequential(
            FourierScalarEncoder(embedding_dim=cond_embedding_dim),
            nn.GELU(),
        )
    elif (cond_dim is not None) and (cond_embedding_dim is not None):
        return nn.Sequential(
            nn.Linear(cond_dim, cond_embedding_dim),
            nn.GELU(),
            nn.Linear(cond_embedding_dim, cond_embedding_dim),
            nn.GELU(),
        )
    else:
        return nn.Identity()
