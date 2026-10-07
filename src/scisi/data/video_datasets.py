"""Memory-mapped pixel video datasets using the existing forecasting batch contract."""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class VideoFrameDataset(Dataset):
    """Consecutive observed frames and their next-frame target, without eager windows.

    The JSON manifest contains ``clips`` with ``path``, ``split``, ``clip_id`` and
    ``num_frames``. Paths are relative to its directory. Arrays are uint8 THWC.
    Train mode indexes *all* valid windows, including the final target. Test mode
    returns complete clips (or their first history + horizon frames). Variable
    length clips should be evaluated individually, or with a fixed ``horizon``.
    Scaling is fixed and reversible; no validation/test statistics are fitted.
    """

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        len_field_history: int = 2,
        train_or_test: str = "train",
        window_stride: int = 1,
        max_clips: int | None = None,
        normalization: str = "minus_one_one",
        horizon: int | None = None,
        preprocesser=None,
    ):
        if preprocesser is not None:
            raise ValueError(
                "Video pixels use fixed normalization; preprocesser must be None"
            )
        if len_field_history < 1 or window_stride < 1:
            raise ValueError("History and window stride must be positive")
        if train_or_test not in ("train", "test"):
            raise ValueError("train_or_test must be train or test")
        if normalization not in ("minus_one_one", "zero_one"):
            raise ValueError("Unknown pixel normalization")
        if horizon is not None and horizon < 1:
            raise ValueError("horizon must be positive")
        if max_clips is not None and max_clips < 1:
            raise ValueError("max_clips must be positive")
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        self.len_field_history = len_field_history
        self.train_or_test = train_or_test
        self.window_stride = window_stride
        self.normalization = normalization
        self.horizon = horizon
        self.clips = [
            c
            for c in self.manifest["clips"]
            if c["split"] == split
            and c["num_frames"] >= len_field_history + (horizon or 1)
        ]
        self.clips = self.clips[:max_clips]
        if not self.clips:
            raise ValueError(f"No sufficiently long clips in split {split!r}")
        if len({c["clip_id"] for c in self.clips}) != len(self.clips):
            raise ValueError("Duplicate clip IDs in manifest")
        self.num_trajectories = len(self.clips)
        first = self._array(0)
        _, self.height, self.width, self.num_channels = first.shape
        counts = [
            (c["num_frames"] - len_field_history - 1) // window_stride + 1
            for c in self.clips
        ]
        self._cumulative = np.cumsum(counts).tolist()

    def _array(self, clip_index: int) -> np.ndarray:
        clip = self.clips[clip_index]
        array = np.load(self.manifest_path.parent / clip["path"], mmap_mode="r")
        if array.ndim != 4 or array.dtype != np.uint8:
            raise ValueError(f"Expected uint8 THWC video: {clip['path']}")
        if hasattr(self, "num_channels") and array.shape[1:] != (
            self.height,
            self.width,
            self.num_channels,
        ):
            raise ValueError(f"Inconsistent video dimensions: {clip['path']}")
        if len(array) != clip["num_frames"]:
            raise ValueError(f"Manifest frame count mismatch: {clip['path']}")
        return array

    def __len__(self) -> int:
        return (
            self._cumulative[-1] if self.train_or_test == "train" else len(self.clips)
        )

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        # Copy only the selected window; torch must not expose a read-only mmap.
        value = torch.from_numpy(np.array(array, copy=True)).permute(3, 1, 2, 0).float()
        value.div_(255.0)
        if self.normalization == "minus_one_one":
            value.mul_(2.0).sub_(1.0)
        return value

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.train_or_test == "test":
            clip = self.clips[index]
            stop = (
                None if self.horizon is None else self.len_field_history + self.horizon
            )
            return {
                "x": self._tensor(self._array(index)[:stop]),
                "clip_id": clip["clip_id"],
            }
        clip_index = bisect.bisect_right(self._cumulative, index)
        previous = 0 if clip_index == 0 else self._cumulative[clip_index - 1]
        start = (index - previous) * self.window_stride
        sample = self._tensor(
            self._array(clip_index)[start : start + self.len_field_history + 1]
        )
        return {
            "base": sample[..., -2],
            "target": sample[..., -1],
            "field_history": sample[..., :-1],
        }
