import torch
import torch.nn as nn
from einops import rearrange
from pdetransformer.core.mixed_channels import PDETransformer

# This argument does not do anything for the model, but is required by the PDETransformer class
SAMPLE_SIZE = 1


class PDETransformerWrapper(nn.Module):
    """
    Wrapper for the PDETransformer class.

    This wrapper is used to wrap the PDETransformer class so it fits the setup in this project.

    It is based on https://github.com/tum-pbs/pde-transformer
    With corresponding paper: https://arxiv.org/pdf/2505.24717

    Currently, it only supports the mixed channels setup.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        len_field_history: int = 2,
        type: str = "PDE-B",
        patch_size: int = 4,
        periodic: bool = True,
        carrier_token_active: bool = False,
        field_cond_channels: int | None = None,
        **kwargs: dict,
    ) -> None:
        """
        Initialize the PDETransformerWrapper.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            len_field_history (int): Length of the field history.
            type (str): Type of the PDETransformer.
            patch_size (int): Patch size.
            periodic (bool): Whether to use periodic padding.
            carrier_token_active (bool): Whether to use carrier tokens.
            **kwargs: Additional arguments for the PDETransformer.
        """
        super().__init__()

        self.len_field_history = len_field_history

        in_channels += self.len_field_history * in_channels

        if field_cond_channels is not None:
            in_channels += field_cond_channels

        self.model = PDETransformer(
            sample_size=SAMPLE_SIZE,
            in_channels=in_channels,
            out_channels=out_channels,
            type=type,
            patch_size=patch_size,
            periodic=periodic,
            carrier_token_active=carrier_token_active,
            **kwargs,
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
            pars_cond (torch.Tensor): pars conditional tensor [B, 1]. Can be None.
                Note that pars_cond has to be a scalar.

        Returns:
            torch.Tensor: Output tensor [B, C_out, H, W].
        """

        if field_history is not None:
            field_history = rearrange(field_history, "b c h w l -> b (l c) h w")
            x = torch.cat([x, field_history], dim=1)

        if field_cond is not None:
            field_cond = rearrange(field_cond, "b c h w -> b c h w")
            x = torch.cat([x, field_cond], dim=1)

        cond = cond.view(-1)

        if pars_cond is not None:
            pars_cond = torch.as_tensor(pars_cond.view(-1), dtype=torch.long)

        out = self.model.forward(
            hidden_states=x,
            timestep=cond,
            class_labels=pars_cond,
        )

        return out.sample
