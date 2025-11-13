import pdb
from typing import Any, Optional

import torch
import torch.nn as nn
from aurora_lib.batch_adapter import BatchAdapter
from aurora_lib.model_wrapper import AuroraModelWrapper

from scisi.architectures.architecture_utils import InitConvWithFieldCond


class AuroraWrapper(nn.Module):
    """Aurora wrapper."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        len_field_history: int,
        batch_adapter: Optional[BatchAdapter] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Aurora wrapper.

        Args:
            batch_adapter: The batch adapter.
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            len_field_history: The length of the field history.
        """
        super().__init__()

        self.batch_adapter = batch_adapter

        self.model = AuroraModelWrapper(*args, **kwargs)

        self.init_convs = nn.ModuleList(
            [
                InitConvWithFieldCond(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    field_cond_channels=in_channels,
                )
                for _ in range(len_field_history)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        field_history: torch.Tensor,
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
            pars_cond (torch.Tensor): pars conditional tensor [B, 1]. Can be None.
                Note that pars_cond has to be a scalar.

        Returns:
            torch.Tensor: Output tensor [B, C_out, H, W].
        """

        field_history = torch.stack(
            [conv(x, field_history[..., i]) for i, conv in enumerate(self.init_convs)],
            dim=-1,
        )

        aurora_batch = self.batch_adapter.scisi_to_aurora(field_history)  # type: ignore[union-attr]

        pred = self.model.forward(aurora_batch, pseudo_time=cond[0])

        x_pred, _ = self.batch_adapter.aurora_to_scisi(pred)  # type: ignore[union-attr]

        zeros_pad = torch.zeros(x_pred.shape[0], x_pred.shape[1], 1, x_pred.shape[3])
        x_pred = torch.cat([x_pred, zeros_pad], dim=2)

        return x_pred
