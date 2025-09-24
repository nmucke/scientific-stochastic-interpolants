import pdb
from typing import Optional

import torch
import torch.nn as nn


def get_grid_observation_matrix(
    data_size: tuple[int, int, int], skip_grid: int
) -> torch.Tensor:
    """Get grid observation matrix."""

    C, H, W = data_size
    num_dofs = C * H * W

    # Calculate number of observed points based on grid spacing
    obs_h = H // skip_grid
    obs_w = W // skip_grid
    num_obs = C * obs_h * obs_w

    # Create observation matrix
    obs_matrix = torch.zeros(num_obs, num_dofs)

    # Fill in ones at grid points
    obs_idx = 0
    for c in range(C):
        for h in range(0, H, skip_grid):
            for w in range(0, W, skip_grid):
                flat_idx = c * (H * W) + h * W + w
                obs_matrix[obs_idx, flat_idx] = 1.0
                obs_idx += 1

    # Get indices where observations are made
    obs_indices = torch.nonzero(obs_matrix.sum(dim=0))

    return obs_matrix, num_obs, obs_indices


def get_random_observation_matrix(
    data_size: tuple[int, int, int],
    percent_obs: float,
) -> torch.Tensor:
    """Get random observation matrix."""
    num_obs = int(data_size[0] * data_size[1] * data_size[2] * percent_obs)
    num_dofs = data_size[0] * data_size[1] * data_size[2]
    perm = torch.randperm(num_dofs)
    obs_indices = perm[:num_obs]
    obs_matrix = torch.zeros(num_obs, num_dofs)
    for row in range(num_obs):
        obs_matrix[row, obs_indices[row]] = 1.0
    return obs_matrix, num_obs, obs_indices


get_observation_matrix_factory = {
    "grid": get_grid_observation_matrix,
    "random": get_random_observation_matrix,
}


class LinearObservationOperator(nn.Module):
    """Linear observation operator."""

    def __init__(
        self,
        type: str = "grid",
        data_size: tuple[int, int, int] = (1, 128, 128),
        skip_grid: Optional[int] = None,
        percent_obs: Optional[float] = None,
    ) -> None:
        """
        Initialize linear observation operator.

        Args:
            type: Type of observation operator.
            data_size: Data size.
            skip_grid: Skip grid.
            percent_obs: Percent of observations.
        """
        super(LinearObservationOperator, self).__init__()

        self.C, self.H, self.W = data_size

        self.num_dofs = self.C * self.H * self.W

        self.type = type

        (
            self.obs_matrix,
            self.num_obs,
            self.obs_indices,
        ) = get_observation_matrix_factory[type](
            data_size, skip_grid if type == "grid" else percent_obs  # type: ignore[arg-type]
        )

        self.obs_matrix = self.obs_matrix.to("cuda")

    @property
    def obs_indices_on_grid(self) -> torch.Tensor:
        """Get observation indices."""
        indices = torch.zeros(self.num_dofs)
        indices[self.obs_indices] = 1
        return indices.view(self.C, self.H, self.W)

    @property
    def obs_indices_c_h_w(self) -> torch.Tensor:
        """Get observation indices."""
        """Get x, y coordinates of observation points.

        Returns:
            Tensor of shape (num_obs, 3) containing [channel, height, width] indices of observations.
        """
        indices = self.obs_indices_on_grid
        obs_indices = torch.zeros((self.num_obs, 3), dtype=torch.long)
        idx = 0

        for c in range(self.C):
            y_coords, x_coords = torch.where(indices[c] == 1)
            num_coords = len(y_coords)
            obs_indices[idx : idx + num_coords, 0] = c
            obs_indices[idx : idx + num_coords, 1] = y_coords
            obs_indices[idx : idx + num_coords, 2] = x_coords
            idx += num_coords

        return obs_indices

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

        b = x.shape[0]

        return x.view(b, -1) @ self.obs_matrix.T
