import pdb
from typing import List

import hydra
import torch
import torch.nn as nn
from einops import rearrange

from scisi.architectures.conv_next import ConvNextBlock
from scisi.architectures.embeddings import FourierScalarEncoder


class InitConvWithHistory(nn.Module):
    """Init conv with field cond."""

    def __init__(  # type: ignore[no-untyped-def]
        self,
        in_channels: int,
        out_channels: int,
        len_field_history: int,
        **kwargs,
    ) -> None:
        """Initialize init conv with field cond."""
        super(InitConvWithHistory, self).__init__()
        self.conv = ConvNextBlock(
            in_channels=in_channels + len_field_history * in_channels,
            out_channels=out_channels,
            **kwargs,
        )

    def forward(  # type: ignore[no-untyped-def]
        self,
        x: torch.Tensor,
        field_history: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass."""
        field_history = rearrange(field_history, "b c h w t -> b (t c) h w")
        x = torch.cat([x, field_history], dim=1)
        x = self.conv(x)
        return x


class InitConvWithFieldCond(nn.Module):
    """Init conv with field cond."""

    def __init__(  # type: ignore[no-untyped-def]
        self,
        in_channels: int,
        out_channels: int,
        field_cond_channels: int,
        **kwargs,
    ) -> None:
        """Initialize init conv with field cond."""
        super(InitConvWithFieldCond, self).__init__()
        self.conv = ConvNextBlock(
            in_channels=in_channels + field_cond_channels,
            out_channels=out_channels,
            **kwargs,
        )

    def forward(  # type: ignore[no-untyped-def]
        self,
        x: torch.Tensor,
        field_cond: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass."""
        x = torch.cat([x, field_cond], dim=1)
        x = self.conv(x)
        return x


class InitConvWithFieldCondAndHistory(nn.Module):
    """Init conv with field cond and history."""

    def __init__(  # type: ignore[no-untyped-def]
        self,
        in_channels: int,
        out_channels: int,
        field_cond_channels: int,
        len_field_history: int,
        **kwargs,
    ) -> None:
        """Initialize init conv with field cond and history."""
        super(InitConvWithFieldCondAndHistory, self).__init__()

        self.history_conv = InitConvWithHistory(
            in_channels,
            in_channels + len_field_history * in_channels,
            len_field_history,
            **kwargs,
        )
        self.field_cond_conv = InitConvWithFieldCond(
            in_channels + len_field_history * in_channels,
            out_channels,
            field_cond_channels,
            **kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        x = self.history_conv(x, field_history)
        x = self.field_cond_conv(x, field_cond)
        return x


class InitConv(nn.Module):
    """Init conv without field cond."""

    def __init__(  # type: ignore[no-untyped-def]
        self,
        in_channels: int,
        out_channels: int,
        **kwargs,
    ) -> None:
        """Initialize init conv without field cond."""
        super(InitConv, self).__init__()
        self.conv = ConvNextBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs,
        )

    def forward(  # type: ignore[no-untyped-def]
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        return x


def get_init_conv(  # type: ignore[no-untyped-def]
    in_channels: int,
    out_channels: int,
    field_cond_channels: int | None = None,
    len_field_history: int | None = None,
    **kwargs,
) -> nn.Module:
    """
    Get initial convolution.

    Helper function to get the initial convolution that handles the field conditional.
    This is to avoid if else statements in the forward pass.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        field_cond_channels (int): Number of field conditional channels.
        len_field_history (int): Length of the field history.
    """
    # Handle different combinations of field conditioning
    has_field_cond = field_cond_channels is not None
    has_history = len_field_history is not None

    if has_field_cond and not has_history:
        return InitConvWithFieldCond(
            in_channels=in_channels,
            out_channels=out_channels,
            field_cond_channels=field_cond_channels,  # type: ignore[arg-type]
            **kwargs,
        )
    elif has_history and not has_field_cond:
        return InitConvWithHistory(
            in_channels=in_channels,
            out_channels=out_channels,
            len_field_history=len_field_history,  # type: ignore[arg-type]
            **kwargs,
        )
    elif has_field_cond and has_history:
        return InitConvWithFieldCondAndHistory(
            in_channels=in_channels,
            out_channels=out_channels,
            field_cond_channels=field_cond_channels,  # type: ignore[arg-type]
            len_field_history=len_field_history,  # type: ignore[arg-type]
            **kwargs,
        )
    else:
        return InitConv(
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs,
        )


def get_conv_blocks(  # type: ignore[no-untyped-def]
    module: nn.Module,
    in_channels: List[int],
    out_channels: List[int],
    **kwargs,
) -> nn.ModuleList:
    """
    Get conv blocks.

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


class ParsCondIdentity(nn.Module):
    """Pars cond identity."""

    def __init__(self) -> None:
        """Initialize pars cond identity."""
        super(ParsCondIdentity, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        pars_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        return x


def get_attention_blocks(
    module_dict: dict,
    channels: List[int],
    attention_in_layers: List[bool],
) -> nn.ModuleList:
    """
    Get attention blocks.
    """

    module_dict["_target_"] = module_dict.pop("target")

    modules = nn.ModuleList()
    for i in range(len(channels)):
        if attention_in_layers[i]:
            modules.append(
                hydra.utils.instantiate(
                    module_dict,
                    channels=channels[i],
                )
            )
        else:
            modules.append(ParsCondIdentity())

    module_dict["target"] = module_dict.pop("_target_")
    return modules
