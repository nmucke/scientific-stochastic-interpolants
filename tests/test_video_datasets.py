import json
import numpy as np
import pytest
import torch
from scisi.data.video_datasets import VideoFrameDataset


def manifest(tmp_path):
    clips = []
    for i, (split, n) in enumerate(
        [("train", 6), ("train", 5), ("validation", 7), ("test", 8)]
    ):
        x = np.arange(n, dtype=np.uint8)[:, None, None, None] * 30
        x = np.broadcast_to(x, (n, 4, 4, 3)).copy()
        np.save(tmp_path / f"{i}.npy", x)
        clips.append(dict(clip_id=str(i), path=f"{i}.npy", split=split, num_frames=n))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dict(clips=clips)))
    return path


def test_last_history_target_and_final_window(tmp_path):
    data = VideoFrameDataset(manifest(tmp_path), len_field_history=2)
    assert len(data) == 7
    first = data[0]
    assert set(first) == {"base", "target", "field_history"}
    assert first["field_history"].shape == (3, 4, 4, 2)
    torch.testing.assert_close(first["base"], first["field_history"][..., -1])
    assert first["target"][0, 0, 0].item() == pytest.approx(60 / 127.5 - 1)
    assert data[3]["target"][0, 0, 0].item() == pytest.approx(150 / 127.5 - 1)
    assert data[4]["target"][0, 0, 0].item() == pytest.approx(60 / 127.5 - 1)
    torch.testing.assert_close(data[-1]["target"], data[6]["target"])
    with pytest.raises(IndexError):
        data[7]


def test_split_horizon_and_fixed_scaling(tmp_path):
    path = manifest(tmp_path)
    data = VideoFrameDataset(
        path, split="test", len_field_history=2, train_or_test="test", horizon=3
    )
    assert len(data) == 1
    assert data[0]["clip_id"] == "3"
    assert data[0]["x"].shape == (3, 4, 4, 5)
    assert data[0]["x"].min() == -1
    assert data[0]["x"].max().item() == pytest.approx(120 / 127.5 - 1)
    with pytest.raises(ValueError, match="No sufficiently long"):
        VideoFrameDataset(path, split="validation", len_field_history=2, horizon=6)


def test_strided_windows_stay_within_clip(tmp_path):
    data = VideoFrameDataset(
        manifest(tmp_path),
        len_field_history=2,
        window_stride=2,
        normalization="zero_one",
    )
    assert len(data) == 4
    assert data[1]["target"][0, 0, 0].item() == pytest.approx(120 / 255)
    assert data[2]["target"][0, 0, 0].item() == pytest.approx(60 / 255)


def test_metadata_mismatch_rejected(tmp_path):
    path = manifest(tmp_path)
    metadata = json.loads(path.read_text())
    metadata["clips"][0]["num_frames"] = 9
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="frame count mismatch"):
        VideoFrameDataset(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_stride": 0},
        {"len_field_history": 0},
        {"horizon": 0},
        {"max_clips": 0},
        {"normalization": "zscore"},
        {"preprocesser": object()},
    ],
)
def test_invalid_options(tmp_path, kwargs):
    with pytest.raises(ValueError):
        VideoFrameDataset(manifest(tmp_path), **kwargs)
