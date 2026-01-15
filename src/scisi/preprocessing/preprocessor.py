import pdb
from typing import Dict, Optional

import numpy as np
import torch
from einops import rearrange


def expand_dims(data: torch.Tensor) -> torch.Tensor:
    """Expand the dimensions of the data."""
    if data.ndim == 1:
        return rearrange(data, "c -> c 1 1")
    elif data.ndim == 2:
        return rearrange(data, "h w -> 1 h w")
    elif data.ndim == 3:
        return rearrange(data, "c h w -> c h w")
    else:
        return data


class Preprocesser:
    """Preprocesser for the data."""

    def __init__(
        self,
        base: Dict[str, list[float] | np.ndarray],
        target: Dict[str, list[float] | np.ndarray],
        field_cond: Dict[str, list[float] | np.ndarray] | None = None,
        pars_cond: Dict[str, list[float] | np.ndarray] | None = None,
    ):
        self.base_mean = expand_dims(torch.tensor(base["mean"]))
        self.base_std = expand_dims(torch.tensor(base["std"]))
        self.target_mean = expand_dims(torch.tensor(target["mean"]))
        self.target_std = expand_dims(torch.tensor(target["std"]))

        if field_cond is not None:
            self.field_cond_mean = expand_dims(torch.tensor(field_cond["mean"]))
            self.field_cond_std = expand_dims(torch.tensor(field_cond["std"]))
        else:
            self.field_cond_mean = None
            self.field_cond_std = None

        self.pars_cond_mean = (
            rearrange(torch.tensor(pars_cond["mean"]), "c -> c") if pars_cond else None
        )
        self.pars_cond_std = (
            rearrange(torch.tensor(pars_cond["std"]), "c -> c") if pars_cond else None
        )

        # Use methods instead of lambdas for pickle compatibility with multiprocessing
        self.batched_trajectory_fn = {
            (True, True): self._unsqueeze_batch_trajectory,
            (True, False): self._unsqueeze_batch,
            (False, True): self._unsqueeze_trajectory,
            (False, False): self._no_unsqueeze,
        }

    def _unsqueeze_batch_trajectory(self, x: torch.Tensor) -> torch.Tensor:
        """Unsqueeze both batch and trajectory dimensions."""
        return x.unsqueeze(0).unsqueeze(-1)

    def _unsqueeze_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Unsqueeze batch dimension."""
        return x.unsqueeze(0)

    def _unsqueeze_trajectory(self, x: torch.Tensor) -> torch.Tensor:
        """Unsqueeze trajectory dimension."""
        return x.unsqueeze(-1)

    def _no_unsqueeze(self, x: torch.Tensor) -> torch.Tensor:
        """No unsqueeze operation."""
        return x

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
            is_batch: Whether the data is a batch.
            is_trajectory: Whether the data is a trajectory.

        Returns:
            A dictionary containing the transformed data.
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


class PixelWisePreprocesser(Preprocesser):
    """Pixel-wise preprocesser for the data."""

    def __init__(
        self,
        base: str,
        target: str,
        field_cond: Optional[str] = None,
    ) -> None:
        """Initialize the pixel-wise preprocesser."""

        _base = np.load(base)
        _target = np.load(target)
        _field_cond = np.load(field_cond) if field_cond is not None else None

        super().__init__(
            base={"mean": _base["mean"], "std": _base["std"]},
            target={"mean": _target["mean"], "std": _target["std"]},
            field_cond=(
                {"mean": _field_cond["mean"], "std": _field_cond["std"]}
                if _field_cond is not None
                else None
            ),
        )
