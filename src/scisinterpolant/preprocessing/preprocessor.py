import pdb
from typing import Dict

import torch
from einops import rearrange


class Preprocesser:
    """Preprocesser for the data."""

    def __init__(
        self,
        base: Dict[str, list[float]],
        target: Dict[str, list[float]],
        field_cond: Dict[str, list[float]] | None = None,
        pars_cond: Dict[str, list[float]] | None = None,
    ):
        self.base_mean = rearrange(torch.tensor(base["mean"]), "c -> c 1 1")
        self.base_std = rearrange(torch.tensor(base["std"]), "c -> c 1 1")
        self.target_mean = rearrange(torch.tensor(target["mean"]), "c -> c 1 1")
        self.target_std = rearrange(torch.tensor(target["std"]), "c -> c 1 1")
        self.field_cond_mean = (
            rearrange(torch.tensor(field_cond["mean"]), "c -> c 1 1")
            if field_cond
            else None
        )
        self.field_cond_std = (
            rearrange(torch.tensor(field_cond["std"]), "c -> c 1 1")
            if field_cond
            else None
        )
        self.pars_cond_mean = (
            rearrange(torch.tensor(pars_cond["mean"]), "c -> c") if pars_cond else None
        )
        self.pars_cond_std = (
            rearrange(torch.tensor(pars_cond["std"]), "c -> c") if pars_cond else None
        )

        self.batched_trajectory_transform_fn = {
            (True, True): self._transform_batch_trajectory,
            (True, False): self._transform_batch,
            (False, True): self._transform_trajectory,
            (False, False): self._transform_sample,
        }

    def _transform_batch(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Transform the batch data."""

        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

        return (x - mean) / std

    def _transform_trajectory(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Transform the trajectory data."""

        mean = mean.unsqueeze(-1)
        std = std.unsqueeze(-1)

        return (x - mean) / std

    def _transform_batch_trajectory(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Transform the batch trajectory data."""
        mean = mean.unsqueeze(0).unsqueeze(-1)
        std = std.unsqueeze(0).unsqueeze(-1)
        return (x - mean) / std

    def _transform_sample(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Transform the data."""
        return (x - mean) / std

    def _transform(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        is_batch: bool = False,
        is_trajectory: bool = False,
    ) -> torch.Tensor:
        """Transform the field data."""
        return self.batched_trajectory_transform_fn[(is_batch, is_trajectory)](
            x, mean, std
        )

    def transform(
        self,
        base: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
        is_batch: bool = False,
        is_trajectory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Transform the data.

        Args:
            base: The base data. [B, C, H, W]
            target: The target data. [B, C, H, W]
            field_cond: The field condition data. [B, C, H, W]
            pars_cond: The pars condition data. [B, D]
        """

        if base is not None:
            base = self._transform(
                base, self.base_mean, self.base_std, is_batch, is_trajectory
            )
        if target is not None:
            target = self._transform(
                target, self.target_mean, self.target_std, is_batch, is_trajectory
            )
        if field_cond is not None:
            field_cond = self._transform(
                field_cond,
                self.field_cond_mean,
                self.field_cond_std,
                is_batch,
                is_trajectory,
            )
        if pars_cond is not None:
            pars_cond = self._transform(
                pars_cond,
                self.pars_cond_mean,
                self.pars_cond_std,
                is_batch,
                is_trajectory,
            )

        return {
            "base": base,
            "target": target,
            "field_cond": field_cond,
            "pars_cond": pars_cond,
        }
