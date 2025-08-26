import torch
import torch.nn as nn


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
