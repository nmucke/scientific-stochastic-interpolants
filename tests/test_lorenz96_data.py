"""Lorenz 96 training pairs: pairing, layout and standardisation."""

import numpy as np
import pytest
import torch

from scisi.data.lorenz96 import Lorenz96PairDataset, load_preprocesser, make_loader

TRAJECTORIES, STEPS, D = 5, 4, 3
MEAN, STD = 2.0, 4.0


@pytest.fixture
def data_path(tmp_path):
    # Each value encodes (trajectory, step, site), so a pair can be traced back.
    states = np.zeros((TRAJECTORIES, STEPS, D), dtype=np.float32)
    for n in range(TRAJECTORIES):
        for k in range(STEPS):
            for i in range(D):
                states[n, k, i] = 100 * n + 10 * k + i
    path = tmp_path / "data.npz"
    np.savez(path, states=states, mean=np.float64(MEAN), std=np.float64(STD))
    return path


def decode(standardised):
    return standardised * STD + MEAN


def test_pairs_stay_within_trajectories_and_target_is_next_step(data_path):
    dataset = Lorenz96PairDataset(data_path, (1, 4))
    assert len(dataset) == 3 * (STEPS - 1)
    batch = dataset[torch.arange(len(dataset))]
    base, target = decode(batch["base"]), decode(batch["target"])
    site = torch.arange(D, dtype=base.dtype)
    seen = set()
    for row in range(len(dataset)):
        b, t = base[row, 0, 0], target[row, 0, 0]
        n, k = int(b[0] // 100), int(b[0] % 100 // 10)
        torch.testing.assert_close(b, 100 * n + 10 * k + site)
        torch.testing.assert_close(t, b + 10)  # the next step of the same trajectory
        assert 1 <= n < 4 and 0 <= k < STEPS - 1
        seen.add((n, k))
    assert len(seen) == len(dataset)


def test_layout_of_states_and_history(data_path):
    dataset = Lorenz96PairDataset(data_path, (0, 2))
    batch = dataset[torch.tensor([0, 3, 5])]
    assert batch["base"].shape == (3, 1, 1, D)
    assert batch["target"].shape == (3, 1, 1, D)
    assert batch["field_history"].shape == (3, 1, 1, D, 1)
    assert torch.equal(batch["field_history"][..., 0], batch["base"])


def test_standardisation_round_trip(data_path):
    dataset = Lorenz96PairDataset(data_path, (0, TRAJECTORIES))
    batch = dataset[torch.arange(len(dataset))]
    pre = load_preprocesser(data_path)
    back = pre.inverse_transform(
        base=batch["base"], target=batch["target"], is_batch=True
    )
    raw = torch.from_numpy(np.load(data_path)["states"])
    torch.testing.assert_close(back["base"].reshape(-1, D), raw[:, :-1].reshape(-1, D))
    torch.testing.assert_close(back["target"].reshape(-1, D), raw[:, 1:].reshape(-1, D))
    torch.testing.assert_close(batch["base"], (back["base"] - MEAN) / STD)


def test_loader_yields_whole_batches(data_path):
    loader = make_loader(data_path, (0, TRAJECTORIES), batch_size=4, shuffle=False)
    batches = list(loader)
    assert [b["base"].shape[0] for b in batches] == [4, 4, 4, 3]
    assert batches[0]["field_history"].shape == (4, 1, 1, D, 1)
