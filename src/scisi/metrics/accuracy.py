"""Point-accuracy metrics (spec Section 3a).

All metrics operate on torch tensors. The ensemble dimension is, by
convention, the leading axis of the ensemble tensor. Spatial metrics accept an
optional boolean ``mask`` over the per-point dimensions where ``True`` marks a
cell to *keep* (fluid) and ``False`` a cell to *exclude* (e.g. solid cells in
the urban case).
"""

from typing import Optional

import torch

from scisi.metrics.spectral import energy_spectrum_rmse  # noqa: F401  (re-export)

__all__ = ["ensemble_mean_rmse", "energy_spectrum_rmse"]


def _apply_mask(
    values: torch.Tensor, mask: Optional[torch.Tensor]
) -> torch.Tensor:
    """Flatten ``values`` to the kept entries given a boolean keep-mask.

    Args:
        values: Per-point tensor ``[...]`` of arbitrary spatial shape.
        mask: Optional boolean tensor broadcastable to ``values`` where
            ``True`` keeps a cell. ``None`` keeps everything.

    Returns:
        1-D tensor of the kept values.
    """
    if mask is None:
        return values.reshape(-1)
    mask = mask.to(torch.bool)
    if mask.shape != values.shape:
        mask = mask.expand_as(values)
    return values[mask]


def ensemble_mean_rmse(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Ensemble-mean RMSE: ``sqrt(mean_i (xbar_i - x*_i)^2)``.

    The ensemble mean ``xbar`` is formed first, then the RMSE is taken between
    ``xbar`` and the target. This is *not* the per-member RMSE averaged over
    members (which over-counts ensemble spread); it is the RMSE of the
    ensemble mean as specified in Section 3a.

    Args:
        ensemble: Ensemble tensor ``[E, *spatial]`` (leading ensemble axis).
        target: Ground truth ``x*`` of shape ``[*spatial]`` (or broadcastable).
        mask: Optional boolean keep-mask over ``*spatial`` (``True`` = keep).

    Returns:
        Scalar tensor with the ensemble-mean RMSE.
    """
    if ensemble.dim() < 1:
        raise ValueError("`ensemble` must have a leading ensemble dimension.")
    xbar = ensemble.mean(dim=0)
    sq_err = (xbar - target) ** 2
    kept = _apply_mask(sq_err, mask)
    return torch.sqrt(kept.mean())
