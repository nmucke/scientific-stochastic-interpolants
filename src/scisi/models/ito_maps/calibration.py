"""Residual-scale calibration for deterministic-to-Ito-map fine-tuning.

Stage 0 of docs/plans/deterministic_to_ito_map_finetuning.md: one gradient-free
pass over a dataloader with the frozen deterministic mean model F estimates the
per-channel scale rho of the residual R = X_1 - F(x_n, h). Normalizing the
residual by rho makes it O(1), which justifies unit-scale interpolant noise and
the default sigma schedules in residual coordinates. The residual mean is
recorded as a bias diagnostic only - it is never subtracted (the residual Ito
map learns it).
"""

import logging
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from scisi.deterministic_models.deterministic_model import DeterministicModel

logger = logging.getLogger(__name__)

# Floor for the estimated std: a (near-)deterministic channel would otherwise
# make the normalized residual blow up.
MIN_RESIDUAL_STD = 1e-8

RESIDUAL_STATS_FILENAME = "residual_stats.pt"


@dataclass
class ResidualStats:
    """Per-channel statistics of the residual X_1 - F(x_n, h).

    ``std`` is the normalization scale rho of the residual Ito map; ``mean``
    is a bias diagnostic of the deterministic model (not subtracted).
    """

    mean: torch.Tensor
    std: torch.Tensor

    def save(self, path: str) -> None:
        """Persist the statistics (typically next to the F checkpoint)."""
        torch.save({"mean": self.mean.cpu(), "std": self.std.cpu()}, path)

    @classmethod
    def load(cls, path: str) -> "ResidualStats":
        """Load statistics persisted with ``save``."""
        data = torch.load(path, map_location="cpu")
        return cls(mean=data["mean"], std=data["std"])


def estimate_residual_stats(
    det_model: DeterministicModel,
    dataloader: DataLoader,
    per_channel: bool = True,
) -> ResidualStats:
    """Estimate the residual mean and std of a deterministic model.

    One ``no_grad`` pass over the dataloader (repo train-batch layout: keys
    ``base``, ``target`` and optionally ``field_history``, ``field_cond``,
    ``pars_cond``), accumulating over (samples, H, W) per channel.

    Args:
        det_model: Trained deterministic mean model F (evaluated in eval mode
            on its own device).
        dataloader: Dataloader to calibrate on (validation data suffices).
        per_channel: If False, a single scalar statistic is computed over all
            channels jointly (returned with shape [1], which broadcasts).

    Returns:
        ResidualStats: Per-channel (or scalar) mean and std, float32 on CPU.
    """
    device = det_model.device
    was_training = det_model.training
    det_model.eval()

    count = 0
    total: torch.Tensor | None = None
    total_sq: torch.Tensor | None = None

    try:
        with torch.no_grad():
            for batch in dataloader:
                base = batch["base"].to(device)
                target = batch["target"].to(device)
                field_history = batch.get("field_history")
                field_cond = batch.get("field_cond")
                pars_cond = batch.get("pars_cond")

                pred = det_model._step(
                    base,
                    field_history=(
                        field_history.to(device) if field_history is not None else None
                    ),
                    field_cond=(
                        field_cond.to(device) if field_cond is not None else None
                    ),
                    pars_cond=(
                        pars_cond.to(device) if pars_cond is not None else None
                    ),
                )

                residual = (target - pred).to(torch.float64)
                if not per_channel:
                    residual = residual.reshape(residual.shape[0], 1, -1)

                # Reduce over everything except the channel dim.
                reduce_dims = (0,) + tuple(range(2, residual.ndim))
                batch_sum = residual.sum(dim=reduce_dims)
                batch_sum_sq = (residual**2).sum(dim=reduce_dims)

                if total is None:
                    total = batch_sum
                    total_sq = batch_sum_sq
                else:
                    total = total + batch_sum
                    total_sq = total_sq + batch_sum_sq
                count += residual.numel() // residual.shape[1]
    finally:
        det_model.train(was_training)

    if total is None:
        raise ValueError("Cannot estimate residual statistics: empty dataloader.")

    mean = total / count
    variance = (total_sq / count - mean**2).clamp(min=0.0)
    std = torch.sqrt(variance).clamp(min=MIN_RESIDUAL_STD)

    logger.info(
        f"Residual calibration over {count} values per channel: "
        f"mean={mean.tolist()}, std={std.tolist()}"
    )

    return ResidualStats(
        mean=mean.to(torch.float32).cpu(), std=std.to(torch.float32).cpu()
    )
