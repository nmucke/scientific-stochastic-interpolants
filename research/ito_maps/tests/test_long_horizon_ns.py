"""Long autonomous forecasts: honest truth boundaries and exact CPU restarts."""

import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

PATH = (
    Path(__file__).resolve().parents[3]
    / "research/ito_maps/experiments/long_horizon_ns.py"
)
SPEC = importlib.util.spec_from_file_location("long_horizon_ns", PATH)
LONG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LONG)


class ToyModel(nn.Module):
    def __init__(self, growth=1.0, fail_above=None):
        super().__init__()
        self.network = nn.Identity()
        self.growth = growth
        self.fail_above = fail_above

    @property
    def model(self):
        return self.network

    def sample(
        self,
        field_history,
        batch_size,
        num_steps,
        base=None,
        return_field_history=False,
    ):
        history = field_history
        if history.shape[0] == 1:
            history = history.repeat(batch_size, 1, 1, 1, 1)
        value = history[..., -1] * self.growth
        for _ in range(num_steps):
            value = self.network(
                value
                + 0.01 * torch.randn_like(value)
                + 0.001 * np.random.rand()
                + 0.001 * random.random()
            )
        if self.fail_above is not None and value.max() > self.fail_above:
            value[0, 0, 0, 0] = float("inf")
        result = torch.cat([history[..., 1:], value.unsqueeze(-1)], -1)
        return (value, result) if return_field_history else value


def run(path, **kwargs):
    return LONG.run_rollout(
        kwargs.pop("model", ToyModel()),
        torch.ones(1, 1, 4, 4, 2),
        torch.ones(1, 1, 4, 4, 3),
        output_dir=path,
        trajectories=[180],
        origin=2,
        device="cpu",
        ensemble_size=3,
        horizon=kwargs.pop("horizon", 12),
        num_steps=2,
        seed=72,
        checkpoint_every=2,
        milestones=[4],
        raw_every=3,
        raw_dense=4,
        **kwargs,
    )


def test_enstrophy_spread_identity_and_no_false_skill_without_truth():
    generator = torch.Generator().manual_seed(17)
    ensemble = torch.randn(8, 1, 4, 4, generator=generator, dtype=torch.float64)
    row = LONG.state_diagnostics(ensemble, torch.zeros(1, 4, 4))
    assert row["enstrophy_member_mean"] == pytest.approx(
        row["enstrophy_of_mean"] + 7 / 16 * row["rms_spread"] ** 2, abs=1e-12
    )
    assert row["spread_skill_ratio"] == pytest.approx(
        np.sqrt(9 / 8) * row["rms_spread"] / row["rmse"]
    )
    unknown = LONG.state_diagnostics(ensemble)
    assert (
        unknown["rmse"]
        is unknown["spread_skill_ratio"]
        is unknown["truth_enstrophy"]
        is None
    )


def test_long_rollout_records_every_lead_and_only_available_truth(tmp_path):
    result = run(tmp_path, model=ToyModel(growth=2.0))
    assert result["status"] == "complete"
    assert result["completed_leads"] == len(result["rows"]) == 12
    assert len(result["excursions"]) == 3  # Continue after all amplitude alarms.
    for row in result["rows"]:
        assert row["nfe"] == 2 and row["cumulative_nfe"] == 2 * row["lead"]
        assert row["physical_time"] is None
        assert row["rms_spread"] is not None
        assert (row["rmse"] is not None) == (row["lead"] <= 3)
    leads = [row["lead"] for row in result["raw_snapshots"]]
    assert leads == [1, 2, 3, 4, 6, 9, 12]
    final = np.load(tmp_path / "raw_states/lead_000012.npy")
    assert final.shape == (1, 3, 1, 4, 4) and final.max() > 1000
    assert (tmp_path / "restart_000004.pt").exists()
    assert len((tmp_path / "diagnostics.jsonl").read_text().splitlines()) == 12


def test_cpu_resume_matches_uninterrupted_rng_and_predictions(tmp_path):
    expected = run(tmp_path / "whole")
    partial = run(tmp_path / "resume", stop_after_lead=4)
    assert partial["status"] == "paused_at_requested_lead"
    result = run(tmp_path / "resume", resume=True)
    ignored = {"seconds", "cumulative_seconds"}
    assert [
        {k: v for k, v in row.items() if k not in ignored} for row in result["rows"]
    ] == [
        {k: v for k, v in row.items() if k not in ignored} for row in expected["rows"]
    ]
    for record in result["raw_snapshots"]:
        actual = np.load(tmp_path / "resume" / record["path"])
        np.testing.assert_array_equal(
            actual, np.load(tmp_path / "whole" / record["path"])
        )
    # A longer requested horizon extends the original immutable protocol.
    old_protocol = (tmp_path / "resume/protocol.json").read_bytes()
    extended = run(tmp_path / "resume", resume=True, horizon=16)
    assert extended["completed_leads"] == 16
    assert (tmp_path / "resume/protocol.json").read_bytes() == old_protocol


def test_nonfinite_is_recorded_raw_and_stops_without_replacement(tmp_path):
    model = ToyModel(growth=2.0, fail_above=6)
    result = run(tmp_path, model=model)
    assert result["status"] == "nonfinite"
    assert result["completed_leads"] == 3
    assert result["first_failure"]["affected_case_member_indices"] == [[0, 0]]
    assert result["rows"][-1]["finite_member_fraction"] == pytest.approx(2 / 3)
    assert result["rows"][-1]["enstrophy_member_mean"] is None
    assert np.isinf(np.load(tmp_path / "raw_states/lead_000003.npy")).any()
    assert not model.model._forward_hooks
    with pytest.raises(ValueError, match="nonfinite dynamics"):
        run(tmp_path, model=ToyModel(growth=2.0, fail_above=6), resume=True)


def test_resume_checks_raw_artifacts_and_existing_outputs(tmp_path):
    run(tmp_path, stop_after_lead=4)
    with pytest.raises(FileExistsError):
        run(tmp_path)
    raw = tmp_path / "raw_states/lead_000001.npy"
    raw.write_bytes(raw.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="raw rollout state changed"):
        run(tmp_path, resume=True)
