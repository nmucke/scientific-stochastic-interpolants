"""Forecast evaluation must measure generated rollouts and fair sample scores."""

import pytest
import torch
from torch import nn

from scisi.bin.main_map_model_test import (
    aggregate_rows,
    evaluate_model,
    forecast_metrics,
)
from scisi.models.follmer_ito_map_model import FollmerItoMapModel
from scisi.models.interpolations import QuadraticStochasticInterpolation


class UnitVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, cond, **kwargs):
        return torch.ones_like(x) * self.value


class RecordingMap(FollmerItoMapModel):
    def __init__(self):
        super().__init__(
            QuadraticStochasticInterpolation(gamma_multiplier=0), UnitVelocity()
        )
        self.seen_histories = []
        self.generated = []

    def sample(self, field_history, **kwargs):
        self.seen_histories.append(field_history.clone())
        result = super().sample(field_history, **kwargs)
        self.generated.append(
            (result[0] if isinstance(result, tuple) else result).clone()
        )
        return result


class TinyDataset:
    len_field_history = 2

    def __init__(self, future=100.0):
        self.values = torch.tensor([0.0, 1.0, future, future + 100, future + 200])

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"x": self.values.expand(1, 4, 4, -1).clone()}


def test_evaluation_uses_generated_history_and_counts_only_forecast_nfe():
    model = RecordingMap()
    result = evaluate_model(
        model,
        TinyDataset(),
        None,
        num_steps=[1, 2],
        ensemble_size=3,
        origins=[2],
        forecast_horizon=2,
        device="cpu",
    )
    assert len(result["rows"]) == 4
    for row in result["rows"]:
        assert row["nfe"] == row["num_steps"]
        assert row["rmse"] == (98 if row["lead"] == 1 else 197)
    # Entry zero is the untimed warmup. The second lead must condition on 2,
    # the model forecast, rather than the observed future value 100.
    torch.testing.assert_close(
        model.seen_histories[2][..., -1], torch.full((3, 1, 4, 4), 2.0)
    )
    assert not model.model._forward_hooks


def test_future_truth_does_not_enter_forecast_inputs():
    first, second = RecordingMap(), RecordingMap()
    kwargs = dict(
        num_steps=[1], ensemble_size=2, origins=[2], forecast_horizon=2, device="cpu"
    )
    evaluate_model(first, TinyDataset(future=100), None, **kwargs)
    evaluate_model(second, TinyDataset(future=900), None, **kwargs)
    for generated_first, generated_second in zip(first.generated, second.generated):
        torch.testing.assert_close(generated_first, generated_second)
    for history_first, history_second in zip(
        first.seen_histories, second.seen_histories
    ):
        torch.testing.assert_close(history_first, history_second)


def test_energy_score_uses_fair_pairs_and_dimension_normalization():
    # E|X-y|=5/3 and .5 E_{distinct}|X-X'|=4/3 for {-1,1,3}, y=0.
    # Replication over a field should leave normalized energy score unchanged.
    for size in (2, 4):
        ensemble = torch.tensor([-1.0, 1.0, 3.0])[:, None, None, None].expand(
            3, 1, size, size
        )
        metrics = forecast_metrics(ensemble, torch.zeros(1, size, size))
        assert metrics["energy_score"] == pytest.approx(1 / 3, abs=2e-7)
        assert metrics["crps"] == pytest.approx(1 / 3, abs=2e-7)


def test_aggregation_weights_trajectories_not_unequal_origin_counts():
    keys = [
        "crps",
        "rmse",
        "energy_score",
        "spread",
        "spread_skill",
        "coverage90",
        "relative_enstrophy_error",
        "seconds",
        "nfe",
    ]
    rows = []
    for trajectory, origin, value in [(3, 2, 0.0), (3, 4, 2.0), (9, 2, 5.0)]:
        rows.append(
            {
                "trajectory": trajectory,
                "origin": origin,
                "lead": 1,
                "num_steps": 2,
                "rank_counts": [1, 2, 1],
                **{key: value for key in keys},
            }
        )
    summary = aggregate_rows(rows, n_bootstrap=100)[0]
    assert summary["num_trajectories"] == 2
    assert summary["num_forecasts"] == 3
    assert summary["crps"] == 3.0  # mean of trajectory means 1 and 5
    assert summary["rank_counts"] == [3, 6, 3]
    assert summary["crps_ci95"][0] <= 3 <= summary["crps_ci95"][1]


def test_scoring_randomness_cannot_change_subsequent_samples(monkeypatch):
    import scisi.bin.main_map_model_test as evaluation_module

    first, second = RecordingMap(), RecordingMap()
    first.gaussian_base = second.gaussian_base = True
    kwargs = dict(
        num_steps=[1], ensemble_size=2, origins=[2], forecast_horizon=2, device="cpu"
    )
    evaluate_model(first, TinyDataset(), None, **kwargs)
    original_metrics = evaluation_module.forecast_metrics

    def randomized_metrics(*args):
        torch.rand(100)
        return original_metrics(*args)

    monkeypatch.setattr(evaluation_module, "forecast_metrics", randomized_metrics)
    evaluate_model(second, TinyDataset(), None, **kwargs)
    # Warmup is deliberately unseeded; compare the seeded forecasts only.
    for a, b in zip(first.generated[1:], second.generated[1:]):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize(
    "override", [{"num_trajectories": 0}, {"num_steps": []}, {"origins": []}]
)
def test_empty_evaluation_request_is_rejected(override):
    kwargs = dict(num_steps=[1], ensemble_size=2, origins=[2], device="cpu")
    kwargs.update(override)
    with pytest.raises(ValueError):
        evaluate_model(RecordingMap(), TinyDataset(), None, **kwargs)


def test_entrypoint_preserves_source_trajectory_offset(monkeypatch, tmp_path):
    import scisi.bin.main_map_model_test as evaluation_module
    from omegaconf import OmegaConf

    model, dataset = RecordingMap(), TinyDataset()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    torch.save(model.state_dict(), checkpoint_dir / "model.pth")
    OmegaConf.save(
        OmegaConf.create({"model": {"_target_": "toy.Model"}, "preprocesser": None}),
        checkpoint_dir / "config.yaml",
    )
    cfg = OmegaConf.create(
        {
            "num_threads": 1,
            "test_data": {"_target_": "toy.Dataset", "tajectories_ids": "(180, 200)"},
            "evaluation": {
                "checkpoint_dir": str(checkpoint_dir),
                "output_dir": str(tmp_path / "result"),
                "device": "cpu",
                "ensemble_size": 2,
                "num_steps": [1],
                "origins": [2],
            },
        }
    )
    monkeypatch.setattr(
        evaluation_module,
        "instantiate",
        lambda config, **kwargs: model if config._target_ == "toy.Model" else dataset,
    )
    result = evaluation_module.run_evaluation(cfg)
    assert result["rows"][0]["trajectory"] == 180
    assert result["coordinate_system"] == "dataset original coordinates"
    assert result["tested_config"]["test_data"]["tajectories_ids"] == "(180, 200)"
    assert result["model_class"].endswith("RecordingMap")
