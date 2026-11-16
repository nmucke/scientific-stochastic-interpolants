"""The foreign initial condition reaches every NS path and only moves the IC."""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))
from cases.navier_stokes._ns_pipeline import prepare_truth_and_obs  # noqa: E402

L, T_FULL, NUM_STEPS = 5, 50, 40


class _IdentityPreprocesser:
    def transform(self, is_batch=True, is_trajectory=False, **fields):
        return dict(fields)


class _FirstPixels:
    num_obs = 3

    def __call__(self, x):
        return x.reshape(x.shape[0], -1)[:, : self.num_obs]


def _prepare(ic_index):
    # Every trajectory is constant at its own row number, so a value identifies
    # the row it came from.
    dataset = [{"x": torch.full((1, 4, 4, T_FULL), float(i))} for i in range(4)]
    prior = SimpleNamespace(
        preprocesser=_IdentityPreprocesser(), len_field_history=L, test_dataset=dataset
    )
    return prepare_truth_and_obs(
        prior=prior, test_index=1, obs_operator=_FirstPixels(), variance=1e-8,
        num_physical_steps=NUM_STEPS, obs_noise_seed=0, device="cpu", ic_index=ic_index,
    )


def test_foreign_ic_moves_history_and_base_only():
    out = _prepare(ic_index=3)
    assert torch.all(out.field_history == 3) and out.field_history.shape[-1] == L
    assert torch.all(out.init_base == 3)
    assert torch.all(out.true_trajectory == 1)
    assert out.true_trajectory.shape[-1] == NUM_STEPS
    assert out.observations.shape[-1] == NUM_STEPS
    assert out.observations.sub(1).abs().max() < 0.01


def test_default_keeps_truths_own_ic():
    out = _prepare(ic_index=None)
    assert torch.all(out.field_history == 1)
    assert torch.all(out.init_base == 1)


def test_foreign_ic_leaves_observations_unchanged():
    assert torch.equal(_prepare(3).observations, _prepare(None).observations)


def test_every_driver_call_passes_ic_index():
    tree = ast.parse((EXPERIMENT / "cases/navier_stokes/driver.py").read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "prepare_truth_and_obs"
    ]
    assert len(calls) == 3
    for call in calls:
        assert "ic_index" in {keyword.arg for keyword in call.keywords}
