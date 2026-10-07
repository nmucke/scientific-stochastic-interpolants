"""CPU video evaluation checks independent of pretrained perceptual networks."""

import json

import numpy as np
import pytest
import torch
from torch import nn

from scisi.bin.main_video_model_test import (
    evaluate_video_model,
    select_cases,
    summarize_video_rows,
)
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant


class TinyClips:
    normalization = "minus_one_one"
    len_field_history = 2

    def __init__(self, future_offset=0.0):
        self.future_offset = future_offset
        self.clips = [
            {"clip_id": f"clip_{i}", "source_member": f"source_{i // 2}"}
            for i in range(6)
        ]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        frames = torch.arange(6, dtype=torch.float32) / 20 + index / 10
        frames[2:] += self.future_offset
        return {"x": frames.reshape(1, 1, 1, 6).expand(1, 4, 4, 6).clone()}


class ToySampler(nn.Module):
    """All calls are recorded, and forecasts deliberately leave the pixel range."""

    def __init__(self):
        nn.Module.__init__(self)
        self.network = nn.Identity()
        self.mask = torch.tensor(1.0)
        self.calls = []

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
        self.calls.append(
            {
                "history": field_history.clone(),
                "base": base,
                "steps": num_steps,
                "members": batch_size,
            }
        )
        history = field_history
        if history.shape[0] == 1:
            history = history.repeat(batch_size, 1, 1, 1, 1)
        generated = history[..., -1] + 2.0
        for _ in range(num_steps):
            generated = self.network(generated + 0.01 * torch.randn_like(generated))
        updated = torch.cat([history[..., 1:], generated.unsqueeze(-1)], dim=-1)
        return (generated, updated) if return_field_history else generated


class ToySI(ToySampler, FollmerStochasticInterpolant):
    """Exercise the explicit latest-frame source branch of the evaluator."""


class TinyFeatures:
    def __init__(self, stochastic=False):
        self.inputs = []
        self.stochastic = stochastic

    def __call__(self, videos):
        self.inputs.append(videos.clone())
        if self.stochastic:
            torch.randn(213)
            np.random.randn(19)
        flat = videos.flatten(1)
        return torch.stack([flat.mean(1), flat.square().mean(1)], 1).numpy()


def cheap_metrics(ensemble, truth, _lpips):
    assert ensemble.ndim == 4 and truth.ndim == 3
    return {
        "mean": float(ensemble.mean()),
        "mae": float((ensemble - truth).abs().mean()),
    }


def evaluate(model, dataset=None, **kwargs):
    return evaluate_video_model(
        model,
        TinyClips() if dataset is None else dataset,
        num_steps=(1, 3),
        ensemble_size=3,
        num_clips=3,
        forecast_horizon=4,
        seed=37,
        device="cpu",
        selected_indices=[4, 0, 2],
        feature_extractor=kwargs.pop("feature_extractor", TinyFeatures()),
        metric_fn=kwargs.pop("metric_fn", cheap_metrics),
        **kwargs,
    )


@pytest.mark.parametrize("model_type", [ToySampler, ToySI])
def test_rollout_uses_generated_history_without_clipping_and_counts_nfe(
    model_type, tmp_path
):
    model, extractor = model_type(), TinyFeatures()
    result = evaluate(
        model, feature_extractor=extractor, output_dir=tmp_path, save_qualitative=1
    )
    assert len(result["rows"]) == 2 * 3 * 4
    assert len(result["summary"]) == 2 * 4
    assert len(model.calls) == 1 + 2 * 3 * 4  # One excluded warmup call.
    for row in result["rows"]:
        assert row["nfe"] == row["num_steps"]
        assert row["cumulative_nfe"] == row["lead"] * row["num_steps"]
        assert row["cumulative_seconds"] >= row["seconds"] >= 0
    for fvd in result["fvd"]:
        assert fvd["real_clips"] == 3
        assert fvd["generated_clips"] == 9
        assert fvd["feature_dimension"] == 2
        assert len(fvd["fvd_member_values"]) == 3
        assert fvd["fvd_pooled"] >= 0
    # At each new clip/budget, the prefix resets. Subsequent histories are
    # member-specific generated frames, including values greater than one.
    for start in range(1, len(model.calls), 4):
        calls = model.calls[start : start + 4]
        for lead, call in enumerate(calls):
            assert call["members"] == 3
            if lead:
                assert call["history"].shape[0] == 3
                assert call["history"][..., -1].min() > 1
                assert (
                    call["history"][..., -1].mean()
                    > calls[lead - 1]["history"][..., -1].mean()
                )
            if isinstance(model, FollmerStochasticInterpolant):
                torch.testing.assert_close(call["base"], call["history"][..., -1])
            else:
                assert call["base"] is None
    # Real features are cached across budgets; generated features include all
    # members but never replace the observed prefix with generated values.
    assert len(extractor.inputs) == 3 + 2 * 3
    generated = [x for x in extractor.inputs if len(x) == 3]
    for item in generated:
        assert item.shape == (3, 6, 1, 4, 4)
        torch.testing.assert_close(item[0, :2], item[1, :2])
        assert item[:, 2:].min() > 1
    saved = np.load(tmp_path / "case_000_steps_1.npz")
    assert saved["truth"].shape == (6, 1, 4, 4)
    assert saved["predictions"].shape == (3, 4, 1, 4, 4)
    assert np.min(saved["predictions"]) > 1
    assert len(json.loads((tmp_path / "forecasts.json").read_text())) == 24
    assert not model.model._forward_hooks


def test_future_truth_and_stochastic_metric_networks_cannot_change_forecasts():
    plain = ToySampler()
    evaluate(plain)

    def consuming_metrics(ensemble, truth, lpips):
        torch.randn(1000)
        np.random.randn(177)
        return cheap_metrics(ensemble, truth, lpips)

    changed = ToySampler()
    evaluate(
        changed,
        TinyClips(future_offset=0.3),
        feature_extractor=TinyFeatures(stochastic=True),
        metric_fn=consuming_metrics,
    )
    # Warmup is excluded since it precedes the explicit per-clip sampling seed.
    for left, right in zip(plain.calls[1:], changed.calls[1:]):
        torch.testing.assert_close(left["history"], right["history"], rtol=0, atol=0)


def test_persistence_keeps_last_observation_for_every_member_and_lead():
    extractor = TinyFeatures()
    result = evaluate(None, feature_extractor=extractor)
    assert result["model_class"] == "persistence"
    for row in result["rows"]:
        expected = float(TinyClips()[row["case_index"]]["x"][..., 1].mean())
        assert row["mean"] == pytest.approx(expected)
        assert row["nfe"] == row["cumulative_nfe"] == 0
    for features in [x for x in extractor.inputs if len(x) == 3]:
        torch.testing.assert_close(
            features[:, 2:], features[:, 1:2].expand_as(features[:, 2:])
        )


def test_cases_are_seeded_unique_and_do_not_use_global_rng():
    dataset = TinyClips()
    first = select_cases(dataset, 3, 71)
    np.random.seed(27)
    np.random.rand(100)
    assert select_cases(dataset, 3, 71) == first
    assert len(first) == len(set(first)) == 3
    assert select_cases(dataset, 6, 71) != list(range(6))
    for count in [0, 1, 7]:
        with pytest.raises(ValueError):
            select_cases(dataset, count, 71)


def test_summary_weights_source_videos_equally_then_bootstraps_sources():
    rows = [
        {
            "num_steps": 1,
            "lead": 1,
            "source_video": source,
            "score": score,
            "undefined": None,
        }
        for source, score in [("a", 0.0), ("a", 2.0), ("b", 9.0)]
    ]
    summary = summarize_video_rows(rows, n_bootstrap=10000)[0]
    assert summary["score"] == 5.0  # mean(mean(0,2),9), not mean(0,2,9).
    assert summary["num_clips"] == 3
    assert summary["num_source_videos"] == 2
    assert summary["score_ci95"] == [1.0, 9.0]
    assert summary["undefined"] is None
    assert summary == summarize_video_rows(rows, n_bootstrap=10000)[0]


def test_invalid_case_selection_and_short_future_are_rejected():
    with pytest.raises(ValueError, match="count/uniqueness"):
        evaluate_video_model(
            None,
            TinyClips(),
            num_steps=[1],
            ensemble_size=2,
            num_clips=2,
            selected_indices=[0, 0],
            device="cpu",
            feature_extractor=TinyFeatures(),
        )
    model = ToySampler()
    with pytest.raises(ValueError, match="requested horizon"):
        evaluate_video_model(
            model,
            TinyClips(),
            num_steps=[1],
            ensemble_size=2,
            num_clips=2,
            selected_indices=[0, 1],
            forecast_horizon=10,
            device="cpu",
            feature_extractor=TinyFeatures(),
        )
    assert not model.model._forward_hooks
