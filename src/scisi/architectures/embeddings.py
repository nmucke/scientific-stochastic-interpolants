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


def _scalar_cond_encoder(cond_embedding_dim: int) -> nn.Module:
    """Fourier encoder + MLP for a single scalar conditional input."""
    return nn.Sequential(
        FourierScalarEncoder(embedding_dim=cond_embedding_dim),
        nn.Linear(cond_embedding_dim, cond_embedding_dim),
        nn.GELU(),
        nn.Linear(cond_embedding_dim, cond_embedding_dim),
        nn.GELU(),
    )


class TwoTimeCondEncoder(nn.Module):
    """Two-time (s, t) conditional encoder for Ito maps.

    Embeds each scalar time with its own Fourier encoder branch (identical in
    architecture to the single-time encoder) and sums the embeddings, so the
    output dimension matches the single-time case. Summing (rather than
    concatenating) keeps every downstream FiLM layer shape-identical to a
    single-time network, which is what makes teacher-to-student weight surgery
    exact: copy the teacher's time embedding into the t-branch and zero the
    final linear layer of the s-branch.
    """

    def __init__(self, cond_embedding_dim: int) -> None:
        """Initialize two-time cond encoder."""
        super(TwoTimeCondEncoder, self).__init__()
        self.s_encoder = _scalar_cond_encoder(cond_embedding_dim)
        self.t_encoder = _scalar_cond_encoder(cond_embedding_dim)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            cond (torch.Tensor): Conditional tensor [B, 2] holding (s, t).

        Returns:
            torch.Tensor: Output tensor (B, cond_embedding_dim).
        """
        s, t = cond[:, 0:1], cond[:, 1:2]
        return self.t_encoder(t) + self.s_encoder(s)


def get_cond_encoder(  # type: ignore[no-untyped-def]
    cond_dim: int | None = None,
    cond_embedding_dim: int | None = None,
    two_time: bool = False,
    **kwargs,
) -> nn.Module:
    """Get pars cond embedding."""
    if two_time and cond_embedding_dim is not None:
        return TwoTimeCondEncoder(cond_embedding_dim=cond_embedding_dim)
    if cond_dim == 1 and cond_embedding_dim is not None:
        return _scalar_cond_encoder(cond_embedding_dim=cond_embedding_dim)
    elif (cond_dim is not None) and (cond_embedding_dim is not None):
        return nn.Sequential(
            nn.Linear(cond_dim, cond_embedding_dim),
            nn.GELU(),
            nn.Linear(cond_embedding_dim, cond_embedding_dim),
            nn.GELU(),
        )
    else:
        return nn.Identity()
