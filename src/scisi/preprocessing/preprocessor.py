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

        self.batched_trajectory_fn = {
            (True, True): lambda x: x.unsqueeze(0).unsqueeze(-1),
            (True, False): lambda x: x.unsqueeze(0),
            (False, True): lambda x: x.unsqueeze(-1),
            (False, False): lambda x: x,
        }

    def _transform(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        is_batch: bool = False,
        is_trajectory: bool = False,
    ) -> torch.Tensor:
        """Transform the field data."""
        mean = self.batched_trajectory_fn[(is_batch, is_trajectory)](mean)
        std = self.batched_trajectory_fn[(is_batch, is_trajectory)](std)
        return (x - mean) / std

    def transform(
        self,
        base: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        field_history: torch.Tensor | None = None,
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
            field_history: The field history data. [B, C, H, W, L]
            field_cond: The field condition data. [B, C, H, W]
            pars_cond: The pars condition data. [B, D]
        """

        if base is not None:
            base = self._transform(
                base,
                self.base_mean.to(base.device),
                self.base_std.to(base.device),
                is_batch,
                is_trajectory,
            )
        if target is not None:
            target = self._transform(
                target,
                self.target_mean.to(target.device),
                self.target_std.to(target.device),
                is_batch,
                is_trajectory,
            )
        if field_history is not None:
            field_history = self._transform(
                field_history,
                self.base_mean.to(
                    field_history.device
                ),  # Field history is always normalized with the base mean and std
                self.base_std.to(
                    field_history.device
                ),  # Field history is always normalized with the base mean and std
                is_batch,
                True,  # Field history is always a trajectory
            )
        if field_cond is not None:
            field_cond = self._transform(
                field_cond,
                self.field_cond_mean.to(field_cond.device),
                self.field_cond_std.to(field_cond.device),
                is_batch,
                is_trajectory,
            )
        if pars_cond is not None:
            pars_cond = self._transform(
                pars_cond,
                self.pars_cond_mean.to(pars_cond.device),
                self.pars_cond_std.to(pars_cond.device),
                is_batch,
                is_trajectory,
            )

        return {
            "base": base,
            "target": target,
            "field_history": field_history,
            "field_cond": field_cond,
            "pars_cond": pars_cond,
        }

    def _inverse_transform(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        is_batch: bool = False,
        is_trajectory: bool = False,
    ) -> torch.Tensor:
        """Inverse transform the data."""
        mean = self.batched_trajectory_fn[(is_batch, is_trajectory)](mean)
        std = self.batched_trajectory_fn[(is_batch, is_trajectory)](std)
        return x * std + mean

    def inverse_transform(
        self,
        base: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        field_history: torch.Tensor | None = None,
        field_cond: torch.Tensor | None = None,
        pars_cond: torch.Tensor | None = None,
        is_batch: bool = False,
        is_trajectory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Inverse transform the data."""

        if base is not None:
            base = self._inverse_transform(
                base,
                self.base_mean.to(base.device),
                self.base_std.to(base.device),
                is_batch,
                is_trajectory,
            )
        if target is not None:
            target = self._inverse_transform(
                target,
                self.target_mean.to(target.device),
                self.target_std.to(target.device),
                is_batch,
                is_trajectory,
            )
        if field_history is not None:
            field_history = self._inverse_transform(
                field_history,
                self.base_mean.to(
                    field_history.device
                ),  # Field history is always normalized with the base mean and std
                self.base_std.to(
                    field_history.device
                ),  # Field history is always normalized with the base mean and std
                is_batch,
                True,  # Field history is always a trajectory
            )
        if field_cond is not None:
            field_cond = self._inverse_transform(
                field_cond,
                self.field_cond_mean.to(field_cond.device),
                self.field_cond_std.to(field_cond.device),
                is_batch,
                is_trajectory,
            )
        if pars_cond is not None:
            pars_cond = self._inverse_transform(
                pars_cond,
                self.pars_cond_mean.to(pars_cond.device),
                self.pars_cond_std.to(pars_cond.device),
                is_batch,
                is_trajectory,
            )
        return {
            "base": base,
            "target": target,
            "field_history": field_history,
            "field_cond": field_cond,
            "pars_cond": pars_cond,
        }
