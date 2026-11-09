"""One-step transition pairs of Lorenz 96 for training conditional priors."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from scisi.preprocessing.preprocessor import Preprocesser


def load_preprocesser(path: str | Path) -> Preprocesser:
    """Scalar standardisation fitted on the training split at simulation time."""
    with np.load(path) as data:
        stats = {"mean": [float(data["mean"])], "std": [float(data["std"])]}
    return Preprocesser(base=stats, target=stats)


class Lorenz96PairDataset(torch.utils.data.Dataset):
    """All pairs ``(x^k, x^{k+1})`` of a contiguous range of trajectories.

    States follow the library layout ``[C, H, W] = [1, 1, d]`` and history adds a
    trailing axis of length one, so the same samplers and observation operators run
    unchanged. ``__getitem__`` takes a batch of indices, which removes per-item
    Python overhead for these tiny samples (see :func:`make_loader`).

    Args:
        path: ``data.npz`` holding ``states`` of shape ``[trajectories, steps, d]``.
        trajectories: ``(start, stop)`` range of trajectories to use.
    """

    def __init__(self, path: str | Path, trajectories: Sequence[int]) -> None:
        start, stop = trajectories
        with np.load(path) as data:
            states = torch.from_numpy(data["states"][start:stop])
        d = states.shape[-1]
        pre = load_preprocesser(path)
        base = states[:, :-1].reshape(-1, 1, 1, d)
        target = states[:, 1:].reshape(-1, 1, 1, d)
        sample = pre.transform(base=base, target=target, is_batch=True)
        self.base = sample["base"].contiguous()
        self.target = sample["target"].contiguous()

    def __len__(self) -> int:
        return self.base.shape[0]

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        base = self.base[index]
        return {
            "base": base,
            "target": self.target[index],
            "field_history": base.unsqueeze(-1),
        }


def make_loader(
    path: str | Path,
    trajectories: Sequence[int],
    batch_size: int,
    shuffle: bool,
    max_samples: int | None = None,
) -> DataLoader:
    """Build a loader that hands whole index batches to the dataset."""
    dataset = Lorenz96PairDataset(path, trajectories)
    if max_samples is not None and max_samples < len(dataset):
        keep = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(0)
        )[:max_samples]
        dataset.base, dataset.target = dataset.base[keep], dataset.target[keep]
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    return DataLoader(
        dataset,
        sampler=BatchSampler(sampler, batch_size, drop_last=False),
        batch_size=None,
    )


__all__ = ["Lorenz96PairDataset", "load_preprocesser", "make_loader"]
