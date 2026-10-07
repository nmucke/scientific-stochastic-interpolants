"""Matched-case reporting and cost aggregation, with synthetic CPU fixtures."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_path = Path(__file__).resolve().parents[1] / "research/video_forecasting/experiments/report_video_results.py"
_spec = importlib.util.spec_from_file_location("video_report", _path)
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def fixture(family="FlowMapModel", steps=(1, 8)):
    item = {key: "fixed" for key in report.MATCH_FIELDS}
    item.update(status="complete", model_class=family, dataset="kth", split="test",
                selected_indices=[3, 7], ensemble_size=2, forecast_horizon=2,
                history_length=2, num_clips=2, seed=12, frame_stride=1,
                resolution=[16, 16], feedback_clipping=False, lpips={"network": "alex"})
    item["rows"], item["summary"], item["fvd"] = [], [], []
    for step in steps:
        for lead in [1, 2]:
            metrics = dict(seconds=lead / 10, cumulative_seconds=.1 if lead == 1 else .3,
                           nfe=step, cumulative_nfe=step * lead,
                           **{key: float(lead) for key in report.QUALITY})
            item["summary"].append(dict(num_steps=step, lead=lead, **metrics))
            for index in item["selected_indices"]:
                item["rows"].append(dict(num_steps=step, lead=lead, case_index=index,
                                         source_video=f"source_{index}", clip_id=f"clip_{index}", **metrics))
        item["fvd"].append(dict(num_steps=step, real_clips=2, generated_clips=4,
                                feature_dimension=400, fvd_pooled=2,
                                fvd_member_values=[2, 4], fvd_member_mean=3,
                                fvd_member_std=np.sqrt(2)))
    return item


@pytest.mark.parametrize("field,value", [
    ("manifest_sha256", "different"), ("selected_indices", [7, 3]),
    ("ensemble_size", 3), ("forecast_horizon", 3),
    ("i3d_sha256", "different"), ("split", "validation"),
    ("lpips", {"network": "vgg"}),
])
def test_mismatched_protocol_rejected(field, value):
    first = fixture()
    second = fixture("FollmerItoMapModel")
    second[field] = value
    with pytest.raises(ValueError, match="protocol field"):
        report.validate_comparable([first, second], allow_incomplete=True)


def test_missing_duplicate_and_identity_mismatch_rejected():
    first = fixture()
    broken = copy.deepcopy(first)
    broken["rows"].pop()
    with pytest.raises(ValueError, match="forecast grid"):
        report.validate_comparable([broken], True)
    with pytest.raises(ValueError, match="Duplicate model"):
        report.validate_comparable([first, first], True)
    second = fixture("FollmerItoMapModel")
    for row in second["rows"]:
        row["clip_id"] += "bad"
    with pytest.raises(ValueError, match="Case identities"):
        report.validate_comparable([first, second], True)


def test_full_four_family_budget_check():
    items = [fixture("FlowMapModel"), fixture("FollmerItoMapModel"),
             fixture("FlowMatchingModel", (16, 64)),
             fixture("FollmerStochasticInterpolant", (16, 64))]
    report.validate_comparable(items)
    with pytest.raises(ValueError, match="Missing required"):
        report.validate_comparable(items[:-1])


def test_summary_sums_incremental_cost_not_cumulative_cost():
    rows = report.summarize_evaluation(fixture())
    assert rows[0]["rollout_seconds"] == pytest.approx(.3)
    assert rows[0]["rollout_nfe"] == 2
    assert rows[1]["rollout_nfe"] == 16
    assert rows[0]["mean_crps"] == 1.5
    assert rows[0]["last_crps"] == 2
    assert rows[0]["real_fvd_clips"] == 2
    assert rows[0]["pooled_generated_fvd_clips"] == 4


def test_report_and_fixed_case_gif(tmp_path):
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps(fixture()))
    truth = np.zeros((4, 1, 16, 16), dtype=np.float32)
    predictions = np.ones((2, 2, 1, 16, 16), dtype=np.float32)
    np.savez(tmp_path / "case_000_steps_1.npz", truth=truth, predictions=predictions, history_length=2)
    output = tmp_path / "report"
    result = report.build_report([evaluation], output, allow_incomplete=True)
    assert len(result) == 2
    assert (output / "kth/summary.csv").exists()
    assert (output / "kth/lead_metrics.png").exists()
    assert (output / "kth/fvd_vs_cost.png").exists()
    assert "2 real clips with 4 generated" in (output / "kth/README.md").read_text()
    gif = Image.open(output / "kth/FlowMapModel_steps_1.gif")
    assert gif.n_frames == 4
    manifest = json.loads((output / "kth/report_manifest.json").read_text())
    assert manifest["protocol"]["selected_indices"] == [3, 7]
