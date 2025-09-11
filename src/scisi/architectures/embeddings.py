import pdb

import torch
import torch.nn as nn
from einops import rearrange


class FourierScalarEncoder(nn.Module):
    """Fourier Encoder."""

    def __init__(self, embedding_dim: int) -> None:
        """Initialize Fourier Encoder."""
        super().__init__()
        self.embedding_dim = embedding_dim
        self.weights = nn.Parameter(torch.randn(1, embedding_dim // 2))

        self.sqrt_2 = torch.sqrt(torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor (B, 1).

        Returns:
            torch.Tensor: Output tensor (B, embedding_dim).
        """
        freqs = 2 * x * self.weights
        sin_embed = self.sqrt_2 * torch.sin(freqs * torch.pi)
        cos_embed = self.sqrt_2 * torch.cos(freqs * torch.pi)
        return torch.cat([sin_embed, cos_embed], dim=-1)


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
            nn.Linear(cond_embedding_dim, cond_embedding_dim),
            nn.GELU(),
            nn.Linear(cond_embedding_dim, cond_embedding_dim),
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
