"""KTHDataset on a small synthetic npz in the scripts/download_kth.py format."""

from pathlib import Path

import numpy as np
import pytest
import torch

from scisi.data.datasets import KTH_ACTIONS, KTHDataset
from scisi.preprocessing.preprocessor import Preprocesser

SIZE = 8
L = 3


def write_npz(path, lengths, persons, actions):
    """Videos whose pixel [0, 0] is the video id and [0, 1] the frame index."""
    videos = []
    for v, n in enumerate(lengths):
        video = np.full((n, SIZE, SIZE), 128, dtype=np.uint8)
        video[:, 0, 0] = v
        video[:, 0, 1] = np.arange(n)
        video[:, -1, -1] = 255
        video[:, -1, -2] = 0
        videos.append(video)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    np.savez(
        path,
        frames=np.concatenate(videos),
        offsets=offsets,
        person=np.asarray(persons, dtype=np.int64),
        action=np.asarray(actions, dtype="<U12"),
        scenario=np.ones(len(lengths), dtype=np.int64),
        names=np.asarray(
            [f"person{p:02d}_{a}_d1" for p, a in zip(persons, actions)], dtype="<U32"
        ),
    )


# person, action, length; video 3 is shorter than a train window (L + 1 = 4).
VIDEOS = [
    (1, "walking", 10),
    (1, "boxing", 7),
    (2, "walking", 12),
    (2, "running", 3),
    (3, "handwaving", 9),
]


@pytest.fixture
def data_dir(tmp_path):
    persons, actions, lengths = zip(*VIDEOS)
    write_npz(tmp_path / "kth_64.npz", lengths, persons, actions)
    return tmp_path


def make(data_dir, **kwargs):
    kwargs.setdefault("len_field_history", L)
    return KTHDataset(paths=[str(data_dir)], files=["kth_64.npz"], **kwargs)


def ids(sample):
    """(video id, frame indices) encoded in the [0, 1]-scaled frames."""
    x = (sample * 255).round().long()
    return x[0, 0, 0], x[0, 0, 1]


def test_train_shapes_dtypes_and_range(data_dir):
    dataset = make(data_dir)
    sample = dataset[0]
    assert set(sample) == {"field_history", "base", "target"}
    assert sample["field_history"].shape == (1, SIZE, SIZE, L)
    assert sample["base"].shape == (1, SIZE, SIZE)
    assert sample["target"].shape == (1, SIZE, SIZE)
    for value in sample.values():
        assert value.dtype == torch.float32
        assert value.min() == 0.0 and value.max() == 1.0
    # The frames stay uint8 in memory.
    assert dataset.frames.dtype == torch.uint8


def test_windows_are_consecutive_and_within_videos(data_dir):
    dataset = make(data_dir)
    # Video 3 (3 frames) is skipped; each other video gives n - L windows.
    lengths = [n for _, _, n in VIDEOS]
    assert len(dataset) == sum(n - L for n in lengths if n >= L + 1)
    seen = set()
    for i in range(len(dataset)):
        sample = dataset[i]
        video, frames = ids(sample["field_history"])
        assert torch.all(video == video[0])
        video = int(video[0])
        assert video != 3
        start = int(frames[0])
        assert torch.equal(frames, torch.arange(start, start + L))
        assert torch.equal(sample["base"], sample["field_history"][..., -1])
        target_video, target_frame = ids(sample["target"][..., None])
        assert int(target_video) == video and int(target_frame) == start + L
        assert start + L < lengths[video]
        seen.add((video, start))
    assert len(seen) == len(dataset)


def test_frame_skip(data_dir):
    dataset = make(data_dir, frame_skip=2)
    # A window spans 2 * L + 1 = 7 frames.
    assert len(dataset) == sum(n - 2 * L for _, _, n in VIDEOS if n >= 2 * L + 1)
    for i in range(len(dataset)):
        sample = dataset[i]
        _, frames = ids(sample["field_history"])
        _, target = ids(sample["target"][..., None])
        start = int(frames[0])
        assert torch.equal(frames, torch.arange(start, start + 2 * L, 2))
        assert int(target) == start + 2 * L


def test_preprocesser_is_applied(data_dir):
    stats = {"mean": [0.5], "std": [0.5]}
    raw = make(data_dir)[5]
    sample = make(data_dir, preprocesser=Preprocesser(base=stats, target=stats))[5]
    for key in raw:
        assert torch.allclose(sample[key], raw[key] * 2 - 1)
    assert sample["base"].min() == -1.0 and sample["base"].max() == 1.0


@pytest.mark.parametrize("persons", ["(1, 2)", [1, 2], "(1,2)"])
def test_person_filter(data_dir, persons):
    dataset = make(data_dir, persons=persons)
    assert set(dataset.person.tolist()) == {1, 2}
    assert dataset.num_videos == 4
    videos = {int(ids(dataset[i]["base"][..., None])[0]) for i in range(len(dataset))}
    assert videos == {0, 1, 2}


def test_action_filter(data_dir):
    dataset = make(data_dir, actions=["walking"])
    assert list(dataset.names) == ["person01_walking_d1", "person02_walking_d1"]
    videos = {int(ids(dataset[i]["base"][..., None])[0]) for i in range(len(dataset))}
    assert videos == {0, 2}
    dataset = make(data_dir, persons=[2], actions=["walking", "running"])
    assert list(dataset.action) == ["walking", "running"]
    assert len(dataset) == 12 - L  # the running video is too short
    with pytest.raises(ValueError, match="Unknown KTH actions"):
        make(data_dir, actions=["swimming"])
    assert len(KTH_ACTIONS) == 6


def test_test_clips(data_dir):
    dataset = make(
        data_dir, train_or_test="test", test_clip_length=8, num_test_clips=50, seed=3
    )
    assert len(dataset) == 50
    lengths = [n for _, _, n in VIDEOS]
    for i in range(len(dataset)):
        x = dataset[i]["x"]
        assert set(dataset[i]) == {"x"}
        assert x.shape == (1, SIZE, SIZE, 8) and x.dtype == torch.float32
        video, frames = ids(x)
        assert torch.all(video == video[0])
        video = int(video[0])
        assert lengths[video] >= 8  # videos 1 and 3 are too short
        start = int(frames[0])
        assert torch.equal(frames, torch.arange(start, start + 8))
        assert start + 8 <= lengths[video]
    # Both eligible videos and several start frames are drawn.
    assert set(dataset.video_index.tolist()) == {0, 2, 4}
    assert len(set(dataset.start_index.tolist())) > 1


def test_test_clips_are_seeded(data_dir):
    kwargs = dict(train_or_test="test", test_clip_length=8, num_test_clips=20)
    a, b = make(data_dir, seed=1, **kwargs), make(data_dir, seed=1, **kwargs)
    c = make(data_dir, seed=2, **kwargs)
    assert all(torch.equal(a[i]["x"], b[i]["x"]) for i in range(len(a)))
    assert not all(torch.equal(a[i]["x"], c[i]["x"]) for i in range(len(a)))
    # Without num_test_clips, the first clip of every long enough video.
    every = make(
        data_dir, train_or_test="test", test_clip_length=8, num_test_clips=None
    )
    assert every.video_index.tolist() == [0, 2, 4]
    assert every.start_index.tolist() == [0, 0, 0]


def _compose(overrides):
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parents[1] / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="kth", overrides=overrides)


def test_hydra_instantiation(data_dir):
    from hydra.utils import instantiate

    paths = [
        f"{split}.dataset.paths=[{data_dir}]" for split in ("train_data", "val_data")
    ]
    cfg = _compose(
        paths
        + [
            f"test_data.paths=[{data_dir}]",
            "train_data.num_workers=0",
            "val_data.num_workers=0",
            "train_data.batch_size=4",
            # The synthetic videos are short.
            "len_field_history=2",
            "test_data.test_clip_length=6",
            "test_data.num_test_clips=5",
        ]
    )
    # The default person split is 1-18 / 19-20 / 21-25 (inclusive).
    assert cfg.train_data.dataset.persons == "(1, 18)"
    assert cfg.val_data.dataset.persons == "(19, 20)"
    assert cfg.test_data.persons == "(21, 25)"
    # The synthetic data only has persons 1-3.
    cfg.val_data.dataset.persons = "(3, 3)"
    cfg.test_data.persons = "(2, 3)"
    assert cfg.experiment_tracking.project == "kth"
    preprocesser = instantiate(cfg.preprocesser)
    train = instantiate(cfg.train_data, dataset={"preprocesser": preprocesser})
    val = instantiate(cfg.val_data, dataset={"preprocesser": preprocesser})
    test = instantiate(cfg.test_data)
    assert set(train.dataset.person.tolist()) == {1, 2, 3}
    assert set(val.dataset.person.tolist()) == {3}
    assert len(test) == 5 and test[0]["x"].shape == (1, SIZE, SIZE, 6)
    assert test[0]["x"].min() >= 0.0 and test[0]["x"].max() <= 1.0
    batch = next(iter(train))
    assert batch["field_history"].shape == (4, 1, SIZE, SIZE, 2)
    assert batch["base"].min() >= -1.0 and batch["base"].max() <= 1.0
    assert torch.equal(batch["base"], batch["field_history"][..., -1])
