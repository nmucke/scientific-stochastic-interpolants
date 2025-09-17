import pdb

import torch
import torch.nn as nn


def get_grid_indices(height: int, width: int, skip_grid: int) -> torch.Tensor:
    """
    Get all grid indices.

    Args:
        height: Height.
        width: Width.
        skip_grid: Skip grid.

    Returns:
        Grid indices. [num_obs, 3] where each row is [C_idx, H_idx, W_idx]
    """
    return torch.tensor(
        [
            (0, i, j)
            for i in range(0, height, skip_grid)
            for j in range(0, width, skip_grid)
        ]
    )


def extract_observations(x: torch.Tensor, obs_indices: torch.Tensor) -> torch.Tensor:
    """
    Extract values from tensor x using gather.

    Args:
        x: Input tensor of shape [B, C, H, W]
        obs_indices: Indices tensor of shape [num_obs, 3] where each row is [C_idx, H_idx, W_idx]

    Returns:
        Extracted values of shape [B, num_obs]
    """
    # Convert 3D indices to linear indices
    linear_indices = (
        obs_indices[:, 0] * x.size(2) * x.size(3)
        + obs_indices[:, 1] * x.size(3)
        + obs_indices[:, 2]
    )

    linear_indices = linear_indices.to(x.device)

    # Reshape x to [B, C*H*W] for gather
    x_flat = x.view(x.size(0), -1)  # [B, C*H*W]

    # Use gather to extract values
    return x_flat.gather(1, linear_indices.unsqueeze(0).expand(x.size(0), -1))


class LinearObservationOperator(nn.Module):
    """Linear observation operator."""

    def __init__(
        self,
        obs_indices: torch.Tensor,
    ) -> None:
        """
        Initialize linear observation operator.

        Args:
            obs_indices: Observation indices. [num_obs, 3] where each row is [C_idx, H_idx, W_idx]
        """
        super(LinearObservationOperator, self).__init__()
        self.obs_indices = obs_indices

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. [B, C, H, W]

        Returns:
            Output tensor. [B, num_obs]
        """
        return extract_observations(x, self.obs_indices)
